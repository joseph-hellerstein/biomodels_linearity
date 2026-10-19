'''Tests for scripts/check_psd_prediction_csvs.py.

The script parses the encoded metadata portion of PSD prediction CSV filenames and verifies that
every row in the file matches those values. These tests focus on three components:

* ``_parse_encoded_metadata`` -- filename parsing only (no I/O).
* ``_check_file``    -- per-file validation logic, exercised with mocked :func:`util.getPSDPredictionDF`
                        so each test controls exactly what DataFrame is returned without touching the
                        on-disk CSVs in ``cn.DATA_DIR``.
* ``main``           -- end-to-end orchestration over a temporary directory of fixture files.
'''

import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from typing import cast

import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import scripts.check_psd_prediction_csvs as m  # noqa: E402
import src.constants as cn  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _model_row(max_frac_red: float = 0.001, threshold=None, is_removal=None) -> dict:
    """Build a single model-level row dict for test DataFrames."""
    row = {cn.COL_NUM_POINT: 50,
        cn.COL_AGGREGATION_TYPE: "model", cn.COL_MAX_FRACTIONAL_REDUCTION: max_frac_red}
    if threshold is not None:
        row[cn.COL_COEFFICIENT_THRESHOLD] = threshold
    if is_removal is not None:
        row[cn.COL_IS_CHANGEPOINT_REMOVAL] = is_removal
    return row


def _model_df(*rows: dict) -> pd.DataFrame:
    """Build a model-only DataFrame. Defaults to one well-formed row."""
    if not rows:
        rows = (_model_row(),)
    return pd.DataFrame(list(rows))

# ---------------------------------------------------------------------------
# _parse_encoded_metadata
# ---------------------------------------------------------------------------

class TestParseEncodedMetadata(unittest.TestCase):
    """Tests for the filename-parsing helper that is independent of any I/O."""

    def test_scientific_notation_maxreduction(self) -> None:
        """A filename with ``maxreduction_1e-3`` parses to 0.001 (matches dictToCodedstr output)."""
        result = m._parse_encoded_metadata(
            "piecewise_predictions__numpoint_100__maxreduction_1e-3"
        )
        self.assertEqual(cast(dict, result)[cn.COL_MAX_FRACTIONAL_REDUCTION], 0.001)

    def test_decimal_maxreduction(self) -> None:
        """A filename with decimal ``maxreduction`` parses to the corresponding float."""
        result = m._parse_encoded_metadata(
            "piecewise_predictions__numpoint_50__maxreduction_0.01"
        )
        self.assertEqual(cast(dict, result)[cn.COL_MAX_FRACTIONAL_REDUCTION], 0.01)

    def test_multiple_keys_parsed(self) -> None:
        """A filename with maxreduction followed by threshold and removal parses all three."""
        result = m._parse_encoded_metadata(
            "piecewise_predictions__numpoint_50"
            "__maxreduction_0.01__threshold_0.001__removal_0"
        )
        expected_dct = {cn.COL_NUM_POINT: 50, cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                cn.COL_COEFFICIENT_THRESHOLD: 0.001, cn.COL_IS_CHANGEPOINT_REMOVAL: 0}
        self.assertEqual(result, expected_dct, f"Expected {expected_dct}, got {result}")

    def test_no_known_key_returns_none(self) -> None:
        """A filename without ``__maxreduction_`` returns None so main() can skip it."""
        self.assertIsNone(m._parse_encoded_metadata("piecewise_predictions.csv"))

    def test_aggregated_model_filename_skipped(self) -> None:
        """The aggregated model CSV stem has no encoded metadata and is skipped."""
        self.assertIsNone(
            m._parse_encoded_metadata("piecewise_predictions_model")
        )


# ---------------------------------------------------------------------------
# _check_file
# ---------------------------------------------------------------------------

class TestCheckFile(unittest.TestCase):
    """Per-file validation. getPSDPredictionDF is replaced with a controlled mock so each test can"""
    """exercise exactly the DataFrame that flows into column comparisons."""

    # The script imports ``getPSDPredictionDF`` via a from-import, so to patch it we must target
    # the importing module's global namespace rather than ``util.getPSDPredictionDF``.
    _TARGET = "scripts.check_psd_prediction_csvs.getPSDPredictionDF"

    def _run_with_df(self, df: pd.DataFrame, filename_stem: str) -> list[str]:
        fake_path = Path(f"{filename_stem}.csv")
        with patch.object(m, "getPSDPredictionDF", return_value=df) as patched:
            errors = m._check_file(fake_path)
        self.assertEqual(
            patched.call_count, 1,
            "_check_file must call getPSDPredictionDF exactly once per file.",
        )
        return errors

    def test_consistent_maxreduction_passes(self) -> None:
        """When every row's max_fractional_reduction matches the filename, no error is reported."""
        df = _model_df(_model_row(max_frac_red=0.01))
        stem = "piecewise_predictions__numpoint_50__maxreduction_0.01"
        errors = self._run_with_df(df, stem)
        self.assertEqual(errors, [])

    def test_maxreduction_mismatch_reports_error(self) -> None:
        """A column value that disagrees with the filename produces a FAIL line."""
        df = _model_df(_model_row(max_frac_red=0.99))  # filename says 0.01
        stem = "piecewise_predictions__numpoint_50__maxreduction_0.01"

        errors = self._run_with_df(df, stem)
        self.assertEqual(len(errors), 1)
        self.assertIn("inconsistent", errors[0])
        self.assertIn("maxreduction", errors[0])
        self.assertIn("0.01", errors[0])

    def test_multiple_rows_all_match_passes(self) -> None:
        """A multi-row DataFrame with every row matching the filename passes."""
        df = _model_df(
            _model_row(max_frac_red=0.01),
            _model_row(max_frac_red=0.01),
            _model_row(max_frac_red=0.01),
        )
        stem = "piecewise_predictions__numpoint_50__maxreduction_0.01"

        errors = self._run_with_df(df, stem)
        self.assertEqual(errors, [])

    def test_multiple_rows_mixed_values_fails(self) -> None:
        """If any row disagrees with the filename, _check_file reports an error."""
        df = _model_df(
            _model_row(max_frac_red=0.01),
            _model_row(max_frac_red=99.0),  # mismatch!
        )
        stem = "piecewise_predictions__numpoint_50__maxreduction_0.01"

        errors = self._run_with_df(df, stem)
        self.assertEqual(len(errors), 1)

    def test_missing_column_reported(self) -> None:
        """A CSV missing the column expected by a filename key is reported as an error."""
        df = pd.DataFrame([{cn.COL_NUM_POINT: 50, cn.COL_AGGREGATION_TYPE: "model"}])  # no max_fractional_reduction
        stem = "piecewise_predictions__numpoint_50__maxreduction_0.01"

        errors = self._run_with_df(df, stem)
        self.assertEqual(len(errors), 1)
        self.assertIn("is missing from the CSV", errors[0])
        self.assertIn(cn.COL_MAX_FRACTIONAL_REDUCTION, errors[0])

    def test_no_known_keys_skips_file(self) -> None:
        """A filename without encoded metadata is skipped (not failed) and never touches I/O."""
        fake_path = Path("piecewise_predictions.csv")
        mock_df = _model_df()  # would be returned if getPSDPredictionDF were called

        with patch.object(m, "getPSDPredictionDF", return_value=mock_df) as patched:
            errors = m._check_file(fake_path)
            self.assertEqual(
                patched.call_count, 0,
                "Mock must never be invoked when the filename has no encoded metadata.",
            )

        self.assertEqual(len(errors), 1)
        self.assertIn("No known encoded keys found", errors[0])

    def test_getPSDPredictionDF_exception_is_reported(self) -> None:
        """If getPSDPredictionDF raises, the error is captured and returned."""
        stem = "piecewise_predictions__numpoint_50__maxreduction_0.01"

        with patch.object(
            m, "getPSDPredictionDF", side_effect=IOError("boom")
        ):
            errors = m._check_file(Path(f"{stem}.csv"))

        self.assertEqual(len(errors), 1)
        self.assertIn("Failed to read CSV", errors[0])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    """End-to-end orchestration over a temporary directory of fixture files."""

    def _make_files(self, tmp_dir: str, stems: list[str]) -> None:
        for stem in stems:
            Path(tmp_dir, f"{stem}.csv").touch()

    def test_main_with_one_invalid_file_returns_nonzero(self) -> None:
        """Any file whose column disagrees with its filename causes main() to return nonzero."""
        with tempfile.TemporaryDirectory() as tmp:
            valid_stem = "piecewise_predictions__numpoint_50__maxreduction_0.01"
            invalid_stem = "piecewise_predictions__numpoint_51__maxreduction_0.01"
            self._make_files(tmp, [valid_stem, invalid_stem])

            def fake_get_psd(csv_files=None, **kw):  # noqa: ARG001
                if csv_files is None or len(csv_files) != 1:
                    raise ValueError("Expected exactly one CSV file.")
                stem = Path(str(csv_files[0])).stem
                return (
                    _model_df(_model_row(max_frac_red=0.01))
                    if "_a" in stem
                    else _model_df(_model_row(max_frac_red=99.0))  # mismatch!
                )

            with patch.object(m, "getPSDPredictionDF", side_effect=fake_get_psd), \
                patch("src.constants.DATA_DIR", tmp):
                exit_code = m.main()

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":  # pragma: no cover - invoked by pytest / unittest discover only
    unittest.main()

