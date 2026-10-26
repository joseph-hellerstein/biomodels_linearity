"""Tests for  src/psd_predictions_iterator.py.

These tests cover PSDPredictionsItem construction, _parse_filename parsing and error
cases, _load_dataframe CSV/pickle loading, and end-to-end iteration over synthetic files
in temporary directories (patched over cn.DATA_DIR).
"""
import os
import numpy as np  # type: ignore
import pickle
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from psd_prediction_file_iterator import PSDPredictionFileIterator, PSDPredictionItem # type: ignore
import src.util as util  # type: ignore

_HAS_PSD_DATA = os.path.isdir(cn.PSD_DATA_DIR)


def _csv_bytes(maxreduction=0.01, threshold=0.001, removal=0) -> bytes:
    """Build CSV bytes for a single model-level row of PSD metadata."""
    h = ("system_id,aggregation_type,max_fractional_reduction,"
         "coefficient_threshold,is_changepoint_removal")
    b = f"BIOMD001,model,{maxreduction},{threshold},{removal}"
    return f"{h}\n{b}".encode("utf-8")

def _make_filename(num_point=100000, manycp=0, max_fractional_reduction=0.01, coefficient_threshold=0.001,
        is_changepoint_removal=False,
            repeat=1) -> str:
    """Build a structured filename with encoded metadata."""
    dct = {
        cn.COL_NUM_POINT: num_point,
        cn.COL_MANYCP: manycp,
        cn.COL_MAX_FRACTIONAL_REDUCTION: max_fractional_reduction,
        cn.COL_COEFFICIENT_THRESHOLD: coefficient_threshold,
        cn.COL_IS_CHANGEPOINT_REMOVAL: int(is_changepoint_removal),
        cn.COL_REPEAT: repeat,
    }
    coded_str = util.dictToCodedstr(dct)
    return f"piecewise_predictions__numpoint_{coded_str}"



class TestPSDPredictionFilesIterator(unittest.TestCase):
    """Tests for PSDPredictionFilesIterator construction."""

    def test_stores_all_attributes(self) -> None:
        df = pd.DataFrame({"a": [1]})
        item = PSDPredictionItem(filepath="/path/to/t.csv",
            filename="t.csv", max_fractional_reduction=0.5, coefficient_threshold=0.2,
            is_changepoint_removal=True, max_changepoint=999, num_point=500,
            repeat=3, manycp=True, df=df)
        self.assertEqual(item.filename, "t")
        self.assertAlmostEqual(item.max_fractional_reduction, 0.5)
        self.assertAlmostEqual(item.coefficient_threshold, 0.2)
        self.assertTrue(item.is_changepoint_removal)
        self.assertEqual(item.max_changepoint, 999)
        self.assertEqual(item.num_point, 500)
        self.assertEqual(item.repeat, 3)
        self.assertTrue(item.manycp)
        self.assertIs(item.df, df)



class TestPSDPredictionFilesIteratorInit(unittest.TestCase):
    """Tests for PSDPredictionFilesIterator.__init__."""

    def test_default_is_csv(self) -> None:
        it = PSDPredictionFileIterator()
        self.assertFalse(it.is_pkl)
        self.assertEqual(it.data_dir, cn.PSD_DATA_DIR)

    def test_pickle_mode(self) -> None:
        it = PSDPredictionFileIterator(is_pkl=True)
        self.assertTrue(it.is_pkl)


class TestLoadDataframe(unittest.TestCase):
    """Tests for _load_dataframe using real CSV and pickle files."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_loads_csv_file(self) -> None:
        path = os.path.join(self._tmpdir.name, "test.csv")
        with open(path, "wb") as f:
            f.write(_csv_bytes(0.1, 0.5, removal=1))
        it = PSDPredictionFileIterator(is_pkl=False)
        df = it._load_dataframe(path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)

    def test_loads_pickle_file(self) -> None:
        src = pd.DataFrame({"a": [1], "b": [2]})
        path = os.path.join(self._tmpdir.name, "test.pkl")
        with open(path, "wb") as f:
            pickle.dump(src, f)
        it = PSDPredictionFileIterator(is_pkl=True)
        loaded = it._load_dataframe(path)
        self.assertTrue(loaded.equals(src))

    def test_load_nonexistent_csv_raises(self) -> None:
        it = PSDPredictionFileIterator(is_pkl=False)
        with self.assertRaises(FileNotFoundError):
            it._load_dataframe("/nonexistent/path.csv")


class TestIterOverSyntheticFiles(unittest.TestCase):
    """End-to-end tests using a temp directory patched over cn.DATA_DIR."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_file(self, name: str, data: bytes) -> str:
        path = os.path.join(self._tmpdir.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_iter_csv_yields_items_with_correct_metadata(self) -> None:
        filename = _make_filename(num_point=100, manycp=0, max_fractional_reduction=0.2,
                coefficient_threshold=0.1, is_changepoint_removal=True) + ".csv"
        self._write_file(filename,
            _csv_bytes(0.2, 0.1, removal=1))
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name, is_guard_filename=False))
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertAlmostEqual(item.max_fractional_reduction, 0.2, places=6)

    def test_iter_csv_skips_empty_csv_files(self) -> None:
        """Files that raise during load are skipped silently."""
        # Valid file + empty file (pd.read_csv raises on empty files).
        filename = _make_filename(num_point=100, manycp=0, max_fractional_reduction=0.1,
                coefficient_threshold=0.2, is_changepoint_removal=True) + ".csv"
        self._write_file(filename,
            _csv_bytes(0.1, 0.2))
        self._write_file("empty.csv", b"")
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name, is_guard_filename=False))
        # Only the valid file should be yielded.
        self.assertEqual(len(items), 1)

    def test_iter_csv_empty_dir(self) -> None:
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name, is_guard_filename=False))
        self.assertEqual(len(items), 0)

    def test_iter_csv_yields_dataframe_with_content(self) -> None:
        filename = _make_filename(num_point=100, manycp=0, max_fractional_reduction=0.5,
                coefficient_threshold=0.25, is_changepoint_removal=True) + ".csv"
        self._write_file(filename, _csv_bytes(0.5, 0.25))
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name, is_guard_filename=False))
        self.assertEqual(len(items), 1)
        df = items[0].df
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

    def test_iter_csv_skips_non_matching_files(self) -> None:
        """Files not starting with piecewise_predictions are skipped."""
        path = os.path.join(self._tmpdir.name, "other_file.csv")
        with open(path, "wb") as f:
            f.write(b"col1\n1")
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            items = list(PSDPredictionFileIterator(is_pkl=False,
                    data_dir=self._tmpdir.name, is_guard_filename=False))
        self.assertEqual(len(items), 0)


if __name__ == "__main__":
    unittest.main()


class TestPSDPredictionFilesIteratorQuery(unittest.TestCase):
    """Tests for PSDPredictionFilesIterator.query classmethod."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_pkl(self, name: str, df: pd.DataFrame) -> str:
        path = os.path.join(self._tmpdir.name, name + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(df, f)
        return path

    def _write_bytes(self, name: str, data: bytes) -> str:
        """Write raw bytes to a file in the temp dir (used for CSV fixtures)."""
        path = os.path.join(self._tmpdir.name, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_query_empty_dir_returns_empty_dataframe(self) -> None:
        """With no matching files, query returns an empty DataFrame."""
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(
                    data_dir=self._tmpdir.name, is_guard_filename=False)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_query_adds_filename_column_to_each_row(self) -> None:
        """Each row in the output carries back its source filename."""
        df1 = pd.DataFrame({"species": ["A", "B"], "value": [1.0, 2.0]})
        filename = _make_filename(num_point=100, manycp=0, max_fractional_reduction=0.5, coefficient_threshold=0.25, is_changepoint_removal=True)
        _ = self._write_pkl(filename, df1)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            result = PSDPredictionFileIterator.query(data_dir=self._tmpdir.name, is_model=False,
                    is_pkl=True, is_guard_filename=False)
        # item.filename strips the file extension per PSDPredictionsItem.__init__.
        self.assertTrue((result[cn.COL_FILENAME] == filename).all())

    def test_query_with_default_filter_is_identity(self) -> None:
        """Passing default (identity) filters gives the same rows as no filter."""
        df = pd.DataFrame({"species": ["A"], "value": [1.0]})
        self._write_pkl(
            "piecewise_predictions__numpoint_100__maxreduction_0.5.pkl", df)
        with patch.object(cn, "PSD_DATA_DIR", self._tmpdir.name):
            default_result = PSDPredictionFileIterator.query()
            explicit_result = PSDPredictionFileIterator.query(
                file_filter=lambda dct: True,
                dataframe_filter=lambda d, _: d)
        self.assertEqual(len(default_result), len(explicit_result))
        for col in default_result.columns:
            pd.testing.assert_series_equal(
                default_result[col], explicit_result[col])

# End to End tests 

@unittest.skipUnless(_HAS_PSD_DATA,
        "PSD data directory not found; skipping end-to-end tests.")
class TestEndToEndBioIterator(unittest.TestCase):

    def setUp(self) -> None:
        from src.timecourse_iterator import TimecourseIterator  # type: ignore

    def dataframe_filter(self, df: pd.DataFrame, dct: dict) -> pd.DataFrame:
        """Filter out rows with invalid (sentinel -1) values."""
        col = list(df.columns)[0]
        dff = pd.DataFrame({col: df[col]})
        return dff

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD data directory not found; skipping test.")
    def test_files_found_in_psd_data_dir(self) -> None:
        """Check that the PSD data directory contains files for iteration."""
        df = PSDPredictionFileIterator.query(dataframe_filter=self.dataframe_filter)
        filenames = df[cn.COL_FILENAME].unique()
        self.assertGreater(len(filenames), 0, "No files found in PSD data directory.")
        is_manycp = any(["manycp_1" in f for f in filenames])
        self.assertTrue(is_manycp, "No files with manycp=1 found in PSD data directory.")


if __name__ == "__main__":
    unittest.main()
