"""Tests for scripts/update_psd_prediction_files.py.

These tests cover:
- ``_addAUCColumn``: AUC computation from the ``boundaries`` column via :class:`NLCurve`.
- ``_addBoundariesColumn``: boundaries construction from ``changepoints`` + ``num_point``.
- ``main()``: error handling, backup file creation (CSV and PKL), ``is_report`` output.

All update functions take a :class:`~src.psd_prediction_file_iterator.PSDPredictionItem`,
so every test constructs one via the local ``_make_item`` helper.
"""

import os
import pickle
import tempfile

import pandas as pd  # type: ignore
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock  # type: ignore

import src.constants as cn  # type: ignore
from scripts import update_psd_prediction_files  # type: ignore
from scripts.update_psd_prediction_files import (  # noqa: E402
    PSDPredictionFileIterator,
    PSDPredictionItem,
    UpdateTask,
    _addAUCColumn,
    _addBoundariesColumn,
    _backup,
    main,
    UPDATE_TASKS,
)

# Number of time points used by default in test items.
_NUM_POINT = 100

def _make_item(
    df: pd.DataFrame,
    filepath: str | None = None,
    num_point: int = _NUM_POINT,
) -> PSDPredictionItem:
    """Build a minimal :class:`PSDPredictionItem` for unit testing.

    Args:
        df: DataFrame to attach (must contain the columns referenced by the
            update function being tested).
        filepath: Optional explicit path used as-is; otherwise derives from
            ``tempfile.mkdtemp`` so backup writes land in a known location.
        num_point: Number of time points for boundary computation.

    Returns:
        A :class:`PSDPredictionItem` with the given DataFrame and metadata.
    """
    if filepath is None:
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "test_predictions.csv")
    return PSDPredictionItem(
        filename=os.path.basename(filepath),
        filepath=filepath,
        max_fractional_reduction=0.01,
        coefficient_threshold=0.01,
        is_changepoint_removal=True,
        max_changepoint=3,
        num_point=num_point,
        repeat=None,
        manycp=False,
        df=df,
    )

# ---------------------------------------------------------------------------
# _addAUCColumn tests
# ---------------------------------------------------------------------------

class TestAddAUCColumn(unittest.TestCase):
    """Tests for :func:`_addAUCColumn`."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # -- helpers --------------------------------------------------------------

    def _item_with_boundaries(
        self, system_id: str | list[str], boundaries: list[list[int]],
    ) -> PSDPredictionItem:
        """Helper that builds an item with a ``boundaries`` column."""
        if isinstance(system_id, str):
            system_id = [system_id]
        df = pd.DataFrame(
            {cn.COL_SYSTEM_ID: system_id, cn.COL_BOUNDARIES: boundaries},
        )
        return _make_item(df, filepath=os.path.join(self._tmpdir.name, "test.csv"))

    # -- tests ----------------------------------------------------------------

    def test_empty_boundaries_raises(self) -> None:
        """An empty boundaries list raises ``ValueError`` from :class:`NLCurve`."""
        item = self._item_with_boundaries("BIOMD001", [[]])
        with self.assertRaises(ValueError) as cm:
            _addAUCColumn(item)
        self.assertIn("At least two boundaries", str(cm.exception))

    def test_single_segment_gives_zero_auc(self) -> None:
        """A single-segment curve (no changepoints) has AUC = 0."""
        item = self._item_with_boundaries(
            "BIOMD001", [[0, _NUM_POINT - 1]],
        )
        result = _addAUCColumn(item)
        self.assertEqual(result, 1)
        # NLCurve with one segment of normalized length 1.0 yields AUC=0.
        self.assertAlmostEqual(item.df[cn.COL_AUC].iloc[0], 0.0, places=5)

    def test_multiple_segments_give_positive_auc(self) -> None:
        """Multiple segments produce AUC strictly between 0 and 1."""
        item = self._item_with_boundaries(
            "BIOMD001", [[0, 50, _NUM_POINT - 1]],
        )
        result = _addAUCColumn(item)
        self.assertEqual(result, 1)
        auc_val = item.df[cn.COL_AUC].iloc[0]
        self.assertLess(auc_val, 1.0)
        self.assertGreater(auc_val, 0.0)

    def test_multiple_rows(self) -> None:
        """Every row in the DataFrame is updated."""
        item = self._item_with_boundaries(
            ["BIOMD001", "BIOMD002"],
            [[0, 50], [0, 30, _NUM_POINT - 1]],
        )
        result = _addAUCColumn(item)
        self.assertEqual(result, 2)
        self.assertIn(cn.COL_AUC, item.df.columns)

    def test_missing_boundaries_column_raises(self) -> None:
        """Missing ``boundaries`` column propagates a ``KeyError``."""
        df = pd.DataFrame({cn.COL_SYSTEM_ID: ["BIOMD001"]})
        item = _make_item(df, filepath=os.path.join(self._tmpdir.name, "test.csv"))
        with self.assertRaises(KeyError):
            _addAUCColumn(item)

# ---------------------------------------------------------------------------
# _addBoundariesColumn tests
# ---------------------------------------------------------------------------

class TestAddBoundariesColumn(unittest.TestCase):
    """Tests for :func:`_addBoundariesColumn`."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # -- helpers --------------------------------------------------------------

    def _item_with_changepoints(
        self, system_id: str | list[str], changepoints: list[list[int]],
    ) -> PSDPredictionItem:
        """Helper that builds an item with a ``changepoints`` column."""
        if isinstance(system_id, str):
            system_id = [system_id]
        df = pd.DataFrame(
            {cn.COL_SYSTEM_ID: system_id, cn.COL_CHANGEPOINTS: changepoints},
        )
        return _make_item(df, filepath=os.path.join(self._tmpdir.name, "test.csv"))

    # -- tests ----------------------------------------------------------------

    def test_single_changepoint(self) -> None:
        """One changepoint produces ``[0, cp, NUM_POINT-1]``."""
        item = self._item_with_changepoints("BIOMD001", [[50]])
        result = _addBoundariesColumn(item)
        self.assertEqual(result, 1)
        expected = [0, 50, _NUM_POINT - 1]
        actual = list(item.df[cn.COL_BOUNDARIES].iloc[0])
        self.assertEqual(actual, expected)

    def test_multiple_changepoints_preserve_order(self) -> None:
        """Changepoints are appended in their original order (NLCurve sorts later)."""
        item = self._item_with_changepoints("BIOMD001", [[75, 25]])
        result = _addBoundariesColumn(item)
        self.assertEqual(result, 1)
        # The function does *not* sort; NLCurve handles sorting internally.
        expected = [0, 75, 25, _NUM_POINT - 1]
        actual = list(item.df[cn.COL_BOUNDARIES].iloc[0])
        self.assertEqual(actual, expected)

    def test_empty_changepoints(self) -> None:
        """Empty changepoints produce ``[0, NUM_POINT-1]`` (single segment)."""
        item = self._item_with_changepoints("BIOMD001", [[]])
        result = _addBoundariesColumn(item)
        self.assertEqual(result, 1)
        expected = [0, _NUM_POINT - 1]
        actual = list(item.df[cn.COL_BOUNDARIES].iloc[0])
        self.assertEqual(actual, expected)

    def test_multiple_rows(self) -> None:
        """Each row gets its own boundaries column computed."""
        item = self._item_with_changepoints(
            ["BIOMD001", "BIOMD002"], [[50], [10, 80]],
        )
        result = _addBoundariesColumn(item)
        self.assertEqual(result, 2)
        self.assertIn(cn.COL_BOUNDARIES, item.df.columns)

# ---------------------------------------------------------------------------
# _backup tests
# ---------------------------------------------------------------------------

class TestBackup(unittest.TestCase):
    """Tests for :func:`_backup`."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # -- helpers --------------------------------------------------------------

    def _csv_item(self, df: pd.DataFrame) -> PSDPredictionItem:
        filepath = os.path.join(self._tmpdir.name, "test.csv")
        item = _make_item(df, filepath=filepath)
        df.to_csv(filepath, index=False)  # pre-create so copy2 has content
        return item

    def _pkl_item(self, df: pd.DataFrame) -> PSDPredictionItem:
        filepath = os.path.join(self._tmpdir.name, "test.pkl")
        item = _make_item(df, filepath=filepath)
        with open(filepath, "wb") as f:
            pickle.dump(df, f)  # pre-create so copy2 has content
        return item

    # -- tests ----------------------------------------------------------------

    def test_csv_backup_creates_bak_file(self) -> None:
        """CSV backup writes a ``.bak`` file next to the original."""
        df = pd.DataFrame({cn.COL_SYSTEM_ID: ["X"]})
        item = self._csv_item(df)
        backup_path = _backup(item, is_pkl=False)
        self.assertTrue(backup_path.exists())
        self.assertTrue(backup_path.name.endswith(".bak"))

    def test_pkl_backup_creates_bak_file(self) -> None:
        """PKL backup writes a ``.bak`` file and preserves pickle content."""
        df = pd.DataFrame({cn.COL_SYSTEM_ID: ["X"]})
        item = self._pkl_item(df)
        backup_path = _backup(item, is_pkl=True)
        self.assertTrue(backup_path.exists())
        # Round-trip through pickle to verify content.
        with open(backup_path, "rb") as f:
            loaded_df = pickle.load(f)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_backup_is_idempotent(self) -> None:
        """Calling ``_backup`` twice does not overwrite the existing ``.bak``."""
        df1 = pd.DataFrame({cn.COL_SYSTEM_ID: ["X"]})
        item = self._csv_item(df1)
        first_path = _backup(item, is_pkl=False)

        # Mutate the original CSV and back up again.
        df2 = pd.DataFrame({cn.COL_SYSTEM_ID: ["Y"]})
        df2.to_csv(item.csv_path, index=False)
        second_path = _backup(item, is_pkl=False)

        self.assertEqual(first_path, second_path)

# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    """Tests for :func:`main`."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # -- helpers --------------------------------------------------------------

    def _patched_iterator(self, items: list[PSDPredictionItem]) -> MagicMock:
        """Return a mock iterator yielding the given items."""
        mock_iter = MagicMock()
        mock_iter.__iter__ = lambda self: iter(items)  # type: ignore[assignment]
        return mock_iter

    @patch(
        "scripts.update_psd_prediction_files.PSDPredictionFileIterator",
    )
    def test_empty_dataframe_skipped(self, MockIterCls: MagicMock) -> None:
        """An item with an empty DataFrame is skipped (no failure)."""
        empty_df = pd.DataFrame({cn.COL_SYSTEM_ID: []})
        fake_item = _make_item(
            empty_df, filepath=os.path.join(self._tmpdir.name, "test.csv"),
        )
        MockIterCls.return_value = self._patched_iterator([fake_item])

        result = main(is_report=False)
        # Item skipped -> no failures.
        self.assertEqual(result, 0)

    @patch(
        "scripts.update_psd_prediction_files.PSDPredictionFileIterator",
    )
    def test_auc_failure_continues_to_boundaries(self, MockIterCls: MagicMock) -> None:
        """A failure in ``_addAUCColumn`` does not stop subsequent tasks."""
        df = pd.DataFrame(
            {cn.COL_SYSTEM_ID: ["BIOMD001"], cn.COL_CHANGEPOINTS: [[50]]},
        )
        item = _make_item(df, filepath=os.path.join(self._tmpdir.name, "test.csv"))
        # Pre-write CSV and PKL so both _backup() calls have files to copy.
        df.to_csv(item.csv_path, index=False)
        with open(item.pkl_path, "wb") as f:
            pickle.dump(df, f)
        MockIterCls.return_value = self._patched_iterator([item])

        with patch.object(
            update_psd_prediction_files, "_addAUCColumn",
            side_effect=RuntimeError("boom"),
        ):
            result = main(is_report=True)

        # ``fail_count`` > 0 -> return code is 1.
        self.assertEqual(result, 1)
        # But _addBoundariesColumn should have run and succeeded.
        self.assertIn(cn.COL_BOUNDARIES, item.df.columns)

    @patch(
        "scripts.update_psd_prediction_files.PSDPredictionFileIterator",
    )
    def test_is_report_prints_summary(self, MockIterCls: MagicMock) -> None:
        """``is_report=True`` emits a per-file success line and a summary."""
        df = pd.DataFrame(
            {cn.COL_SYSTEM_ID: ["BIOMD001"], cn.COL_BOUNDARIES: [[0, 50]]},
        )
        item = _make_item(df, filepath=os.path.join(self._tmpdir.name, "test.csv"))
        # Pre-write CSV and PKL so both _backup() calls have files to copy.
        df.to_csv(item.csv_path, index=False)
        with open(item.pkl_path, "wb") as f:
            pickle.dump(df, f)
        MockIterCls.return_value = self._patched_iterator([item])

        with patch("builtins.print") as mock_print:
            result = main(is_report=True)

        output = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
        self.assertIn("SUCCESS", output)
        self.assertIn("Successfully processed", output)

    def test_update_tasks_structure(self) -> None:
        """``UPDATE_TASKS`` contains exactly two tasks with expected column names."""
        self.assertEqual(len(UPDATE_TASKS), 2)
        columns = [t.column for t in UPDATE_TASKS]
        self.assertIn(cn.COL_AUC, columns)
        self.assertIn(cn.COL_BOUNDARIES, columns)

    def test_update_task_namedtuple(self) -> None:
        """Each entry is an :class:`UpdateTask` namedtuple with ``column`` and ``func``."""
        for task in UPDATE_TASKS:
            self.assertIsInstance(task, UpdateTask)
            self.assertEqual(len(task), 2)
            self.assertTrue(callable(task.func))


if __name__ == "__main__":
    unittest.main()
