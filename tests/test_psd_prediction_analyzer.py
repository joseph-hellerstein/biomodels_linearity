"""Tests for  src/psd_prediction_analyzer.py."""

import os
import pickle
import tempfile
import unittest
from typing import Optional, cast, Any

import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from src.psd_prediction_analyzer import (  # type: ignore
    DEFAULT_MAX_CHANGEPOINT,
    PSDPredictionAnalyzer,
)


# ---------------------------------------------------------------------------
# Shared row factory
# ---------------------------------------------------------------------------


def _row(
    system_id: str = "BIOMD0000000005",
    aggregation_type: str = cn.COL_AGGREGATION_TYPE_MODEL,
    max_changepoint: int = DEFAULT_MAX_CHANGEPOINT,
    max_fractional_reduction: float = 0.01,
    auc: float = 0.8,
    p50: float = 0.7,
    num_changepoint: int = 5,
    mean: float = 0.9,
    a80: float = 0.85,
    a90: float = 0.75,
    a95: float = 0.65,
) -> dict:
    """Return a dict with every column required by analyzeAUCConsistency."""
    return {
        cn.COL_SYSTEM_ID: system_id,
        cn.COL_AGGREGATION_TYPE: aggregation_type,
        cn.COL_MAX_CHANGEPOINT: max_changepoint,
        cn.COL_MAX_FRACTIONAL_REDUCTION: max_fractional_reduction,
        cn.COL_AUC: auc,
        cn.COL_P50: p50,
        cn.COL_NUM_CHANGEPOINT: num_changepoint,
        cn.COL_MEAN: mean,
        cn.COL_A80: a80,
        cn.COL_A90: a90,
        cn.COL_A95: a95,
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestPSDPredictionAnalyzerInit(unittest.TestCase):

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()  # type: ignore

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_model_num_and_name_stored(self) -> None:
        analyzer = PSDPredictionAnalyzer(model_id=5, data_dir=self._tmpdir.name,
                is_guard_filename=False)  # type: ignore
        self.assertEqual(analyzer.model_num, 5)
        self.assertEqual(analyzer.model_name, "BIOMD0000000005")

    def test_model_name_stored_from_string(self) -> None:
        analyzer = PSDPredictionAnalyzer(model_id="BIOMD0000000010", data_dir=self._tmpdir.name,
                is_guard_filename=False)  # type: ignore
        self.assertEqual(analyzer.model_num, 10)
        self.assertEqual(analyzer.model_name, "BIOMD0000000010")

    def test_data_dir_stored(self) -> None:
        analyzer = PSDPredictionAnalyzer(model_id=5, data_dir="/path/to/data",
                is_guard_filename=False)  # type: ignore
        self.assertEqual(analyzer.data_dir, "/path/to/data")

    def test_min_accuracy_default(self) -> None:
        analyzer = PSDPredictionAnalyzer(model_id=5, data_dir=self._tmpdir.name,
                is_guard_filename=False)  # type: ignore
        self.assertAlmostEqual(analyzer.min_accuracy, 0.8)

    def test_custom_min_accuracy(self) -> None:
        analyzer = PSDPredictionAnalyzer(is_guard_filename=False,  # type: ignore
            model_id=5, min_accuracy=0.9, data_dir=self._tmpdir.name,  # type: ignore
        )
        self.assertEqual(analyzer.min_accuracy, 0.9)


# ---------------------------------------------------------------------------
# _getModelBestSer
# ---------------------------------------------------------------------------


class TestGetModelBestSer(unittest.TestCase):

    def test_returns_series(self) -> None:
        """best_model_ser should be a pd.Series (a single row of the DataFrame)."""
        data = [
            _row(auc=0.5),
            _row(auc=0.3, max_fractional_reduction=0.02),
        ]
        df = pd.DataFrame(data)
        with tempfile.TemporaryDirectory() as tmpdir:  # type: ignore
            pkl_path = os.path.join(tmpdir, "piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__manycp_1.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(df, f)
            analyzer = PSDPredictionAnalyzer(
                model_id=5, data_dir=tmpdir, is_guard_filename=False,  # type: ignore
            )
        self.assertIsInstance(analyzer.best_model_ser, pd.Series)

    def test_min_auc_row_selected(self) -> None:
        """The selected row should be the one with the minimum AUC."""
        data = [
            _row(auc=0.9),
            _row(auc=0.1, max_fractional_reduction=0.02),
            _row(auc=0.5, max_fractional_reduction=0.03),
        ]
        df = pd.DataFrame(data)
        with tempfile.TemporaryDirectory() as tmpdir:  # type: ignore
            pkl_path = os.path.join(tmpdir, "piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__manycp_1.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(df, f)
            analyzer = PSDPredictionAnalyzer(is_guard_filename=False,  # type: ignore
                model_id=5, data_dir=tmpdir,  # type: ignore
            )
        self.assertAlmostEqual(analyzer.best_model_ser[cn.COL_AUC], 0.1) # type: ignore

    def test_accuracy_constraint_filters_rows(self) -> None:
        """Rows below min_accuracy should be excluded; the minimum AUC is taken among the rest."""
        data = [
            _row(auc=0.2, mean=0.1),  # mean=0.1 < default threshold of 0.8 — excluded
            _row(auc=0.9),             # mean defaults to 0.9 >= min_accuracy=0.8 — qualifies
        ]
        df = pd.DataFrame(data)
        with tempfile.TemporaryDirectory() as tmpdir:  # type: ignore
            pkl_path = os.path.join(tmpdir, "piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__manycp_1.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(df, f)
            analyzer = PSDPredictionAnalyzer(
                model_id=5, data_dir=tmpdir, is_guard_filename=False,  # type: ignore
            )
        self.assertAlmostEqual(analyzer.best_model_ser[cn.COL_AUC], 0.9)  # type: ignore


# ---------------------------------------------------------------------------
# analyzeAUCConsistency
# ---------------------------------------------------------------------------


class TestAnalyzeAUCConsistency(unittest.TestCase):
    """Unit tests using fake pickle fixtures in a temp directory."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()  # type: ignore

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _write_pkl(self, df: pd.DataFrame, name: str = "piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__manycp_1.pkl") -> None:
        """Write df as a pickle file named `name` inside the temp directory."""
        path = os.path.join(self._tmpdir.name, name)
        with open(path, "wb") as f:
            pickle.dump(df, f)

    def _run(self, data_dir: Optional[str] = None) -> pd.DataFrame:
        """Run analyzeAUCConsistency against the temp directory."""
        if data_dir is None:
            data_dir = self._tmpdir.name
        analyzer = PSDPredictionAnalyzer(model_id=5, data_dir=data_dir, is_guard_filename=False)  # type: ignore
        return analyzer.analyzeAUCConsistency()

    def _write_default_pkl(self, **override) -> None:
        """Write a single-row pickle with default column values, overridden by kwargs."""
        row = {**_row(), **override}
        df = pd.DataFrame([row])
        self._write_pkl(df)

    def _write_two_reductions_default(self, auc1: float = 0.5, auc2: float = 0.7) -> None:
        """Write a two-row pickle with different max_fractional_reduction values."""
        df = pd.DataFrame([_row(auc=auc1), _row(auc=auc2, max_fractional_reduction=0.02)])
        self._write_pkl(df)

    def _write_wrong_system_id(self, auc: float = 0.5) -> None:
        """Write a single-row pickle with a wrong system_id."""
        df = pd.DataFrame([_row(system_id="WRONG_ID", auc=auc)])
        self._write_pkl(df)

    def _write_wrong_max_changepoint(self, auc: float = 0.5) -> None:
        """Write a single-row pickle with the wrong max_changepoint value."""
        df = pd.DataFrame([_row(max_changepoint=100, auc=auc)])
        self._write_pkl(df)

    def _write_wrong_aggregation_type(self, auc: float = 0.5) -> None:
        """Write a single-row pickle with the wrong aggregation type."""
        df = pd.DataFrame([_row(aggregation_type="species", auc=auc)])
        self._write_pkl(df)

    def _write_low_accuracy(self, auc: float = 0.2) -> None:
        """Write a single-row pickle with mean below the default min_accuracy threshold."""
        df = pd.DataFrame([_row(auc=auc, mean=0.1)])  # mean=0.1 < default min_accuracy=0.8
        self._write_pkl(df)

    def _write_high_accuracy(self, auc: float = 0.9) -> None:
        """Write a single-row pickle with mean above the default min_accuracy threshold."""
        df = pd.DataFrame([_row(auc=auc)])  # mean defaults to 0.9 >= min_accuracy=0.8
        self._write_pkl(df)

    def _write_multiple_files(self, aucs: list[float]) -> None:
        """Write multiple pickle files each containing one row."""
        for i, auc in enumerate(aucs):
            name = f"piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__manycp_1_{i}.pkl"
            self._write_pkl(pd.DataFrame([_row(auc=auc)]), name)

    def _write_no_matching_files(self) -> None:
        """Write a pickle file that will be ignored by the iterator."""
        path = os.path.join(self._tmpdir.name, "piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__manycp_1.pkl")  # contains 'dummy' so gets filtered out
        with open(path, "wb") as f:
            pickle.dump(pd.DataFrame([_row()]), f)

    def _write_empty_dir(self) -> None:
        """Create an empty temp directory (no pickle files)."""
        pass  # default empty dir is fine; just don't write anything

    # ------------------------------------------------------------------
    # structural checks
    # ------------------------------------------------------------------

    def test_returns_dataframe(self) -> None:
        self._write_default_pkl()
        result = self._run()
        self.assertIsInstance(result, pd.DataFrame)

    def test_has_auc_mean_and_std_columns(self) -> None:
        self._write_default_pkl(auc=0.42)
        result = self._run()
        self.assertIn((cn.COL_AUC, "mean"), result.columns)
        self.assertIn((cn.COL_AUC, "std"), result.columns)

    def test_index_name_is_max_fractional_reduction(self) -> None:
        self._write_default_pkl()
        result = self._run()
        self.assertEqual(result.index.name, cn.COL_MAX_FRACTIONAL_REDUCTION)

    # ------------------------------------------------------------------
    # aggregation / filtering checks
    # ------------------------------------------------------------------

    def test_auc_mean_single_row(self) -> None:
        self._write_default_pkl(auc=0.42)
        result = self._run()
        self.assertAlmostEqual(result.loc[0.01, (cn.COL_AUC, cn.COL_MEAN)], 0.42)  # type: ignore

    def test_auc_mean_two_rows_same_group(self) -> None:
        """Two rows in the same fractional_reduction group should be averaged."""
        df = pd.DataFrame([
            _row(auc=0.6),
            _row(auc=0.8, max_fractional_reduction=0.01),  # same reduction -> same group
        ])
        self._write_pkl(df)
        result = self._run()
        self.assertAlmostEqual(result.loc[0.01, (cn.COL_AUC, cn.COL_MEAN)], 0.7)  # type: ignore

    def test_auc_std_single_row(self) -> None:
        """A single-row group has std == NaN (pandas default ddof=1 on n=1)."""
        self._write_default_pkl(auc=0.42)
        result = self._run()
        self.assertTrue(pd.isna(result.loc[0.01, (cn.COL_AUC, cn.COL_STD)]))

    def test_rows_from_multiple_files_merged_into_same_group(self) -> None:
        """Rows from two different pickle files should be merged before aggregation."""
        pkl_a = "piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__manycp_1_a.pkl"
        pkl_b = "piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__manycp_1_b.pkl"
        self._write_pkl(pd.DataFrame([_row(auc=0.5)]), pkl_a)
        self._write_pkl(pd.DataFrame([_row(auc=0.7)]), pkl_b)
        result = self._run()
        self.assertAlmostEqual(result.loc[0.01, (cn.COL_AUC, cn.COL_MEAN)], 0.6)  # type: ignore

    def test_two_reductions_produce_two_groups(self) -> None:
        """Two distinct max_fractional_reduction values -> two index entries."""
        df = pd.DataFrame([_row(auc=0.5), _row(auc=0.7, max_fractional_reduction=0.02)])
        self._write_pkl(df)
        result = self._run()
        self.assertEqual(len(result), 2)
        self.assertIn(0.01, result.index)
        self.assertIn(0.02, result.index)

    def test_non_model_aggregation_type_excluded(self) -> None:
        """Rows with aggregation_type != 'model' should be excluded from results."""
        df = pd.DataFrame([_row(aggregation_type="species", auc=0.5)])
        self._write_pkl(df)
        result = self._run()
        self.assertEqual(len(result), 0)

    def test_wrong_max_changepoint_excluded(self) -> None:
        """Rows with max_changepoint != DEFAULT_MAX_CHANGEPOINT should be excluded."""
        df = pd.DataFrame([_row(max_changepoint=100, auc=0.5)])
        self._write_pkl(df)
        result = self._run()
        self.assertEqual(len(result), 0)

    def test_has_auc_std_column(self) -> None:
        """Result should always contain a std column for AUC."""
        self._write_default_pkl(auc=0.42)
        result = self._run()
        self.assertIn((cn.COL_AUC, "std"), result.columns)

    def test_wrong_system_id_excluded(self) -> None:
        """Rows whose system_id does not match the analyzer's model should be excluded."""
        df = pd.DataFrame([_row(system_id="WRONG_ID", auc=0.5)])
        self._write_pkl(df)
        result = self._run()
        self.assertEqual(len(result), 0)

    def test_empty_result_when_no_rows_match_system_id(self) -> None:
        """If no rows match the requested system_id, return an empty DataFrame."""
        df = pd.DataFrame([_row(system_id="WRONG_ID", auc=0.5)])
        self._write_pkl(df)
        result = self._run()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# End-to-end against real data (when available)
# ---------------------------------------------------------------------------

_E2E_PKL = (
    "piecewise_predictions__numpoint_100000"
    "__threshold_0.001__removal_0__manycp_1.pkl"
)
_E2E_DATA_PATH = os.path.join(cn.PSD_DATA_DIR, _E2E_PKL)
_HAS_E2E_DATA = os.path.isfile(_E2E_DATA_PATH)


@unittest.skipUnless(_HAS_E2E_DATA, f"real data file {_E2E_PKL!r} not present in data/")
class TestAnalyzeAUCConsistencyEndToEnd(unittest.TestCase):
    """End-to-end tests against the real production pkl from data/.

    The dummy pkl files in data/ have a different column structure and would
    cause a KeyError inside _filter_dataframe.  A symlink of just the valid
    file into a temp directory isolates the test from those files while still
    exercising the real data end-to-end.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()  # type: ignore
        os.symlink(_E2E_DATA_PATH, os.path.join(cls._tmpdir.name, _E2E_PKL))  # type: ignore

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()  # type: ignore

    def _run(self) -> pd.DataFrame:
        return PSDPredictionAnalyzer(
            model_id=5, data_dir=self._tmpdir.name, is_guard_filename=False,  # type: ignore
        ).analyzeAUCConsistency()

    # --- structural checks ---

    def test_returns_dataframe(self) -> None:
        self.assertIsInstance(self._run(), pd.DataFrame)

    def test_has_auc_mean_and_std_columns(self) -> None:
        result = self._run()
        self.assertIn((cn.COL_AUC, "mean"), result.columns)
        self.assertIn((cn.COL_AUC, "std"), result.columns)

    def test_index_name_is_max_fractional_reduction(self) -> None:
        self.assertEqual(
            self._run().index.name,
            cn.COL_MAX_FRACTIONAL_REDUCTION,
        )

    # --- group structure from the real file ---

    def test_one_group_because_file_has_one_reduction_value(self) -> None:
        """The file contains only max_fractional_reduction=0.05 → exactly 1 group."""
        self.assertEqual(len(self._run()), 1)

    def test_group_index_value_is_0_05(self) -> None:
        result = self._run()
        self.assertAlmostEqual(result.index[0], 0.05, places=10)

    # --- quantitative checks against known values ---

    def test_auc_mean_matches_known_value_for_biomd_5(self) -> None:
        """BIOMD0000000005 has a single qualifying row; known AUC is 0.1898878."""
        result = self._run()
        self.assertAlmostEqual(result.loc[0.05, (cn.COL_AUC, cn.COL_MEAN)],
            0.1898877961369339, places=6)  # type: ignore


# ---------------------------------------------------------------------------
# End-to-end against real data for Biomodels 278 (when available)
# ---------------------------------------------------------------------------

_E2E_PKL_278 = (
    "piecewise_predictions__numpoint_100000"
    "__threshold_0.001__removal_1__maxreduction_1e-8__manycp_0__repeat_3.pkl"
)
_E2E_DATA_PATH_278 = os.path.join(cn.PSD_DATA_DIR, _E2E_PKL_278)
_HAS_E2E_DATA_278 = os.path.isfile(_E2E_DATA_PATH_278)


@unittest.skipUnless(
    _HAS_E2E_DATA_278, f"real data file {_E2E_PKL_278!r} not present in data/"
)
class TestAnalyzeAUCConsistencyEndToEndBiomd278(unittest.TestCase):
    """End-to-end tests against the real production pkl from data/ for BIOMD0000000278.

    The dummy pkl files in data/ have a different column structure and would
    cause a KeyError inside _filter_dataframe.  A symlink of just the valid
    file into a temp directory isolates the test from those files while still
    exercising the real data end-to-end.
    """

    _tmpdir: Any

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()  # type: ignore
        os.symlink(_E2E_DATA_PATH_278, os.path.join(cls._tmpdir.name, _E2E_PKL_278))  # type: ignore

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()  # type: ignore

    def _run(self) -> pd.DataFrame:
        result = PSDPredictionAnalyzer(
            model_id=278, data_dir=self._tmpdir.name,  is_guard_filename=False,  # type: ignore
        ).analyzeAUCConsistency()
        return result

    # --- structural checks ---

    def test_returns_dataframe(self) -> None:
        self.assertIsInstance(self._run(), pd.DataFrame)

    def test_index_name_is_max_fractional_reduction(self) -> None:
        self.assertEqual(
            self._run().index.name,
            cn.COL_MAX_FRACTIONAL_REDUCTION,
        )

    # --- group structure from the real file ---

    def test_one_group_because_file_has_one_reduction_value(self) -> None:
        """The file contains only max_fractional_reduction=1e-8 → exactly 1 group."""
        self.assertEqual(len(self._run()), 1)

    def test_group_index_value_is_1e8(self) -> None:
        result = self._run()
        self.assertAlmostEqual(result.index[0], 1e-8, places=20)

    # --- quantitative checks against known values ---

    def test_auc_mean_is_nan_for_zero_auc(self) -> None:
        """BIOMD0000000278 has a single qualifying row with AUC=0.0 which gets replaced by NaN."""
        result = self._run()
        self.assertTrue(pd.isna(result.loc[1e-8, (cn.COL_AUC, cn.COL_MEAN)]))  # type: ignore


if __name__ == "__main__":
    unittest.main()
