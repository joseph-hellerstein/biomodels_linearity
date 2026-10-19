"""Integration tests for scripts/make_biomodels_timecourse.py."""
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import src.constants as cn  # type: ignore
from scripts.make_biomodels_timecourse import main  # type: ignore

IGNORE_TESTS = False
HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)

TEST_MODEL = "BIOMD0000000005"
TEST_MODEL_NUM = 5
EXPECTED_PKL = f"{TEST_MODEL}_timecourse.pkl"


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestMain(unittest.TestCase):
    """Integration tests for main() using real BioModels data."""

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        if IGNORE_TESTS:
            return
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _runOnTestModel(
            self, model_num: int = TEST_MODEL_NUM, is_initialize: bool = True) -> None:
        """Run main() restricted to TEST_MODEL, writing to a temp directory."""
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir):
            main(
                    is_report=False,
                    is_initialize=is_initialize,
                    first_model_num=model_num,
                    last_model_num=model_num,
            )

    def _expectedPklPath(self) -> str:
        return os.path.join(self._tmpdir, EXPECTED_PKL)

    def test_pkl_file_is_written(self) -> None:
        if IGNORE_TESTS:
            return
        self._runOnTestModel()
        self.assertTrue(os.path.isfile(self._expectedPklPath()))

    def test_pkl_filename_contains_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        self._runOnTestModel()
        pkl_files = [f for f in os.listdir(self._tmpdir) if f.endswith(".pkl")]
        self.assertTrue(any(TEST_MODEL in f for f in pkl_files))

    def test_already_serialized_model_is_skipped(self) -> None:
        if IGNORE_TESTS:
            return
        # Create a valid serialized timecourse first.
        self._runOnTestModel(is_initialize=True)
        pkl_path = self._expectedPklPath()
        mtime_before = os.path.getmtime(pkl_path)
        time.sleep(0.1)

        # Run again without is_initialize — main() should skip because the pkl exists.
        self._runOnTestModel(is_initialize=False)

        diff = abs(mtime_before -  os.path.getmtime(pkl_path))
        self.assertTrue(diff < 0.001 * mtime_before,
                        f"Expected skip but file was modified (diff={diff})")

    def test_skipped_model_leaves_no_extra_files(self) -> None:
        if IGNORE_TESTS:
            return
        # Create a valid serialized timecourse first.
        self._runOnTestModel(is_initialize=True)
        pkl_files = [f for f in os.listdir(self._tmpdir) if f.endswith(".pkl")]

        # Run again with is_initialize=False — main() should skip (pkl exists),
        # leaving exactly one .pkl file behind.
        self._runOnTestModel(is_initialize=False)

        pkl_files = [f for f in os.listdir(self._tmpdir) if f.endswith(".pkl")]
        self.assertEqual(len(pkl_files), 1)

    def test_does_not_crash_when_all_models_excluded(self) -> None:
        if IGNORE_TESTS:
            return
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir):
            main(
                    is_report=False,
                    first_model_num=TEST_MODEL_NUM,
                    last_model_num=TEST_MODEL_NUM,
                    excluded_models=[TEST_MODEL],
            )
        pkl_files = [f for f in os.listdir(self._tmpdir) if f.endswith(".pkl")]
        self.assertEqual(len(pkl_files), 0)


if __name__ == "__main__":
    unittest.main()
