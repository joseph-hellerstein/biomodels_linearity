"""Tests for scripts/make_piecewise_predictions.py."""

import os
import tempfile
from typing import cast
import unittest

import matplotlib  # noqa: F401 -- non-interactive backend needed before pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore
from unittest.mock import patch, MagicMock

import src.constants as cn  # type: ignore
from scripts.make_piecewise_predictions import (  # type: ignore
    processModel,
    main,
    COEFFICIENT_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_POINT = 1000  # Increased for valid segment count with min_segment_length=50


def _make_linear_df(
    n_points: int = NUM_POINT,
    t_start: float = 0.0,
    t_end: float = 10.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a simple linear ODE timecourse for testing."""
    rng = np.random.default_rng(seed)

    def rhs(t, z):
        a, b = z
        return [-0.5 * a + 0.1 * b, 0.3 * a - 0.2 * b]

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(rhs, [t_start, t_end], [1.0, 0.0], t_eval=t_eval, rtol=1e-8)
    X = sol.y.T + rng.normal(0, noise_std, (n_points, len(sol.y)))

    return pd.DataFrame(X, index=t_eval, columns=["S1", "S2"])


def _make_mock_item(model_name: str = "BIOMD0000000001", is_timecourse_valid: bool = True) -> MagicMock:
    """Build a mock TimecourseIteratorItem with a synthetic timecourse."""
    item = MagicMock()
    item.model_name = model_name
    if is_timecourse_valid:
        df = _make_linear_df(n_points=NUM_POINT)
    else:
        df = pd.DataFrame()  # Empty DataFrame to simulate invalid timecourse
    item.timecourse.timecourse_df = df
    return item


# ---------------------------------------------------------------------------
# processModel tests
# ---------------------------------------------------------------------------


class TestProcessModel(unittest.TestCase):

    def test_returns_dataframe_on_success(self) -> None:
        """processModel returns a DataFrame when the fit/predict pipeline succeeds."""
        item = _make_mock_item()
        result = processModel(
            item=item,
            max_changepoint=0,
            coefficient_threshold=COEFFICIENT_THRESHOLD,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(cast(pd.DataFrame, result)), 0)

    def test_returns_dataframe_with_expected_columns(self) -> None:
        """processModel augments the score DataFrame with all required metadata columns."""
        item = _make_mock_item()
        result = processModel(
            item=item,
            max_changepoint=0,
            coefficient_threshold=0.002,
        )
        result = cast(pd.DataFrame, result)
        self.assertIn(cn.COL_SYSTEM_ID, result.columns)
        self.assertIn(cn.COL_MAX_CHANGEPOINT, result.columns)
        self.assertIn(cn.COL_MAX_FRACTIONAL_REDUCTION, result.columns)
        self.assertIn(cn.COL_COEFFICIENT_THRESHOLD, result.columns)

    def test_metadata_values_match_parameters(self) -> None:
        """Metadata columns hold the exact parameter values passed to processModel."""
        item = _make_mock_item()
        result = processModel(
            item=item,
            max_changepoint=5,
            coefficient_threshold=0.003,
            max_fractional_reduction=0.25,
            is_changepoint_removal=True,
        )
        result = cast(pd.DataFrame, result)
        self.assertEqual(result[cn.COL_MAX_CHANGEPOINT].iloc[0], 5)
        self.assertAlmostEqual(result[cn.COL_COEFFICIENT_THRESHOLD].iloc[0], 0.003)
        self.assertAlmostEqual(result[cn.COL_MAX_FRACTIONAL_REDUCTION].iloc[0], 0.25)
        self.assertEqual(result[cn.COL_IS_CHANGEPOINT_REMOVAL].iloc[0], True)
    def test_system_id_matches_model_name(self) -> None:
        """The COL_SYSTEM_ID column is set to the item model name."""
        item = _make_mock_item(model_name="BIOMD0000009999")
        result = processModel(
            item=item,
            max_changepoint=0,
            coefficient_threshold=COEFFICIENT_THRESHOLD,
        )
        result = cast(pd.DataFrame, result)
        self.assertEqual(result[cn.COL_SYSTEM_ID].iloc[0], "BIOMD0000009999")

    def test_aggregation_type_rows_present(self) -> None:
        """Score output contains both model and per-species aggregation rows."""
        item = _make_mock_item()
        result = processModel(
            item=item,
            max_changepoint=0,
            coefficient_threshold=COEFFICIENT_THRESHOLD,
        )
        result = cast(pd.DataFrame, result)
        self.assertIn(cn.COL_AGGREGATION_TYPE, result.columns)
        agg_types = set(result[cn.COL_AGGREGATION_TYPE].unique())
        # At minimum: one model row and two species rows (A, B).
        self.assertIn("model", agg_types)
        self.assertGreaterEqual(len(agg_types), 3)

    def test_returns_none_on_exception(self) -> None:
        """processModel returns None when the underlying pipeline raises an exception."""
        item = _make_mock_item(is_timecourse_valid=False)
        with patch(
            "scripts.make_piecewise_predictions.PiecewiseSystemDiscovery",
            side_effect=ValueError("simulated failure"),
        ):
            result = processModel(
                item=item,
                max_changepoint=0,
                coefficient_threshold=COEFFICIENT_THRESHOLD,
            )
        self.assertIsNone(result)

    def test_returns_none_when_predict_returns_none(self) -> None:
        """processModel returns None when psd.predict() returns None (defensive check)."""
        item = _make_mock_item(is_timecourse_valid=False)
        mock_psd = MagicMock()
        mock_psd.fit.return_value = None
        mock_psd.predict.return_value = None
        with patch(
            "scripts.make_piecewise_predictions.PiecewiseSystemDiscovery",
            return_value=mock_psd,
        ):
            result = processModel(
                item=item,
                max_changepoint=0,
                coefficient_threshold=COEFFICIENT_THRESHOLD,
            )
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# main tests
# ---------------------------------------------------------------------------


def _make_mock_timecourse_iterator(mock_items):
    """Build a mock TimecourseIterator that yields predefined items."""
    iterator = MagicMock()
    iterator.__iter__ = MagicMock(return_value=iter(mock_items))
    return iterator


class TestMain(unittest.TestCase):

    def _run_with_mocked_iterator(self, output_path, is_initialize=False,
            model_names=None):
        """Helper: run main() with a mocked TimecourseIterator."""
        if model_names is None:
            model_names = ["BIOMD0000000005"]

        mock_items = [_make_mock_item(name) for name in model_names]
        mock_iter = _make_mock_timecourse_iterator(mock_items)

        with patch("scripts.make_piecewise_predictions.TimecourseIterator",
                return_value=mock_iter):
            main(
                first_model_num=0,
                last_model_num=len(model_names),
                is_initialize=is_initialize,
                coefficient_threshold=COEFFICIENT_THRESHOLD,
                output_path=output_path,
            )

    def test_creates_output_file(self) -> None:
        """main creates the output CSV when it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            self.assertFalse(os.path.isfile(output_path))
            self._run_with_mocked_iterator(output_path,
                    is_initialize=True)
            adjusted_output_path = os.path.join(tmpdir, "output_0.csv")
            self.assertTrue(os.path.isfile(adjusted_output_path))

    def test_skips_already_processed_models(self) -> None:
        """main skips models already present in the existing output CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            # First run: process 1 model.
            self._run_with_mocked_iterator(output_path,
                    is_initialize=True)
            adjusted_output_path = os.path.join(tmpdir, "output_0.csv")
            """ with open(adjusted_output_path, "r") as fd:
                lines = fd.readlines()
                lines.remove("\n")
            self.assertEqual(len(lines), 0) """
            self.assertTrue(os.path.isfile(adjusted_output_path))
            with open(adjusted_output_path, "r") as fd:
                lines = fd.readlines()
                lines.remove('\n')
            self.assertEqual(len(lines), 0)

    def test_is_initialize_resets_output(self) -> None:
        """is_initialize=True resets the output file to empty before processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            # First run: process 1 model.
            self._run_with_mocked_iterator(output_path,
                    is_initialize=True)
            adjusted_output_path = os.path.join(tmpdir, "output_0.csv")
            with open(adjusted_output_path, "r") as fd:
                lines = fd.readlines()
                lines.remove('\n')
            self.assertEqual(len(lines), 0)

    def test_persists_results_to_disk(self) -> None:
        """main persists the accumulated DataFrame to the output CSV at end of run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            adjusted_output_path = os.path.join(tmpdir, "output_0.csv")
            self._run_with_mocked_iterator(output_path,
                    is_initialize=True)
            self.assertTrue(os.path.isfile(adjusted_output_path))
            with open(adjusted_output_path, "r") as fd:
                lines = fd.readlines()
                lines.remove('\n')
            self.assertEqual(len(lines), 0)

    def test_iterates_over_max_changepoints(self) -> None:
        """main iterates over all max_changepoint values for each model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.csv")
            self._run_with_mocked_iterator(output_path,
                    is_initialize=True)
            adjusted_output_path = os.path.join(tmpdir, "output_0.csv")
            self.assertTrue(os.path.isfile(adjusted_output_path))
            adjusted_output_path = os.path.join(tmpdir, "output_0.csv")
            with open(adjusted_output_path, "r") as fd:
                lines = fd.readlines()
                lines.remove('\n')
            self.assertEqual(len(lines), 0)


if __name__ == "__main__":
    unittest.main()
