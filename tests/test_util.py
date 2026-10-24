import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import util  # type: ignore
from util import getPSDPredictionDF, codedstrToDict, dictToCodedstr, parseFilename, _parseChangepoints  # type: ignore
import src.constants as cn  # noqa: E402

_HAS_PSD_DATA = os.path.exists(cn.PSD_DATA_DIR)


def _model_df(max_fractional_reduction: float = 0.001, n: int = 1) -> pd.DataFrame:
    """Return a minimal valid model-level DataFrame."""
    return pd.DataFrame(
        [
            {
                "aggregation_type": "model",
                "max_fractional_reduction": max_fractional_reduction,
                "value": v,
                cn.COL_CHANGEPOINTS: [],
                cn.COL_COUNT: 0,
            }
            for v in range(1, n + 1)
        ]
    )


def _mixed_df(max_fractional_reduction: float = 0.001) -> pd.DataFrame:
    """Return a DataFrame with both species and model rows."""
    return pd.DataFrame(
        [
            {"aggregation_type": "species", "max_fractional_reduction": max_fractional_reduction, "x": 1, cn.COL_CHANGEPOINTS: [], cn.COL_COUNT: 0},
            {"aggregation_type": "model", "max_fractional_reduction": max_fractional_reduction, "x": 2, cn.COL_CHANGEPOINTS: [], cn.COL_COUNT: 0},
        ]
    )


class TestGetPSDPredictionDFFileNotFound(unittest.TestCase):
    """getPSDPredictionDF raises FileNotFoundError when no CSV matches."""

    def test_unknown_maxreduction_raises(self) -> None:
        """A max_fractional_reduction that produces a file-selection string no filename contains raises."""
        with patch("util.os.listdir", return_value=["piecewise_predictions__numpoint_10.csv"]):
            with self.assertRaises(FileNotFoundError):
                getPSDPredictionDF(0.9999)

    def test_unknown_repeat_raises(self) -> None:
        """Passing a repeat value that no filename contains raises."""
        with patch("util.os.listdir", return_value=["piecewise_predictions__numpoint_10.csv"]):
            with self.assertRaises(FileNotFoundError):
                getPSDPredictionDF(0.001, repeat=99999)


class TestGetPSDPredictionDFSingleFile(unittest.TestCase):
    """Single-file match returns a correctly-shaped, filtered DataFrame."""

    def _run_with_fixture(
        self,
        csv_name: str,
        max_fractional_reduction: float = 0.001,
        payload=None,
    ) -> pd.DataFrame:
        if payload is None:
            payload = _model_df(max_fractional_reduction, n=3)
        with patch("util.os.listdir", return_value=[csv_name]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", return_value=payload):
            return getPSDPredictionDF(max_fractional_reduction)

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    def test_single_match_csv_file_column(self) -> None:
        """The csv_file column is populated with the matching filename."""
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        expected_name = f"piecewise_predictions__numpoint_100__{sel}.csv"
        df = self._run_with_fixture(expected_name, max_fractional_reduction=0.001)
        self.assertIn("csv_file", df.columns.tolist())
        self.assertTrue(all(expected_name in v for v in df["csv_file"].tolist()))

    FILENAME = "piecewise_predictions__numpoint_100000__threshold_0.001__removal_1__maxreduction_1e-2__manycp_0__repeat_1.pkl"
    PKL_PATH = os.path.join(cn.DATA_DIR, FILENAME)
    @unittest.skipUnless(os.path.exists(PKL_PATH), f"{FILENAME} not found")
    def test_single_match_aggregation_type_model(self) -> None:
        """All returned rows have aggregation_type == 'model' (species rows are filtered out)."""
        df = getPSDPredictionDF(max_fractional_reduction=0.01, repeat=1)
        self.assertTrue((df["aggregation_type"] == "model").all())


class TestGetPSDPredictionDFAggregation(unittest.TestCase):
    """Concatenation and filtering across multiple matching files."""

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    def test_multiple_files_concatenated(self) -> None:
        """When two files match, the result has rows from both."""
        sel = dictToCodedstr(
            {"maxreduction": 0.003}, convert_strs=["maxreduction"]
        )
        name1 = f"piecewise_predictions__numpoint_101__{sel}.csv"
        name2 = f"piecewise_predictions__numpoint_102__{sel}.csv"
        df1, df2 = _model_df(0.003, n=5), _model_df(0.003, n=7)
        with patch("util.os.listdir", return_value=[name1, name2]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", side_effect=[df1, df2]):
            result = getPSDPredictionDF(0.003)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 12)  # 5 + 7

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    def test_multiple_files_both_filenames_present(self) -> None:
        """All csv_file values are among the two matching filenames."""
        sel = dictToCodedstr(
            {"maxreduction": 0.003}, convert_strs=["maxreduction"]
        )
        name1 = f"piecewise_predictions__numpoint_101__{sel}.csv"
        name2 = f"piecewise_predictions__numpoint_102__{sel}.csv"
        with patch("util.os.listdir", return_value=[name1, name2]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", side_effect=[_model_df(0.003, n=1), _model_df(0.003, n=1)]):
            df = getPSDPredictionDF(0.003)
        self.assertEqual(set(df["csv_file"].tolist()), {str(os.path.join(cn.PSD_DATA_DIR, name)) for name in [name1, name2]})

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    def test_multiple_files_no_species_rows(self) -> None:
        """The result contains no species-level rows even when inputs do."""
        sel = dictToCodedstr(
            {"maxreduction": 0.003}, convert_strs=["maxreduction"]
        )
        name1 = f"piecewise_predictions__numpoint_101__{sel}.csv"
        payload = _mixed_df(0.003)
        with patch("util.os.listdir", return_value=[name1]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", return_value=payload):
            df = getPSDPredictionDF(0.003)
        self.assertTrue((df["aggregation_type"] == "model").all())


class TestGetPSDPredictionDFRepeatFiltering(unittest.TestCase):
    """Behavior of the repeat= parameter in the revised function."""

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    def test_no_repeat_matches_any_file_with_selection(self) -> None:
        """Without repeat, every file containing max_fractional_reduction is matched."""
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        name_no_repeat = f"piecewise_predictions__numpoint_51__{sel}.csv"
        name_with_repeat = f"piecewise_predictions__numpoint_52__{sel}__repeat_1.csv"
        with patch("util.os.listdir", return_value=[name_no_repeat, name_with_repeat]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", side_effect=[_model_df(0.001, n=1), _model_df(0.001, n=1)]):
            df = getPSDPredictionDF(0.001)
        self.assertEqual(len(df), 2)  # Both files matched (no repeat filter applied)

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    def test_repeat_filters_to_matching_file_only(self) -> None:
        """With repeat=N, only the file containing 'repeat_N' in its name is kept."""
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        rep_sel = dictToCodedstr({"repeat": 3})
        name_no_repeat = f"piecewise_predictions__numpoint_51__{sel}.csv"
        name_with_repeat = f"piecewise_predictions__numpoint_52__{sel}__{rep_sel}.csv"
        with patch("util.os.listdir", return_value=[name_no_repeat, name_with_repeat]), \
                patch("util.os.path.exists", return_value=False), \
                patch("pandas.read_csv", return_value=_model_df(0.001, n=2)):
            df = getPSDPredictionDF(0.001, repeat=3)
        self.assertEqual(len(df), 2)


class TestCodedStr2Dict(unittest.TestCase):
    """Tests for util.codedstr2Dict."""

    def test_multiple_key_value_pairs(self) -> None:
        """A string with multiple __-separated key=value pairs is parsed correctly."""
        result = codedstrToDict("key1_val1__key2_val2", is_passthru=True)
        self.assertEqual(result, {"key1": "val1", "key2": "val2"})

    def test_single_key_value(self) -> None:
        """A single key=value pair produces a one-entry dict."""
        result = codedstrToDict("foo_bar", is_passthru=True)
        self.assertEqual(result, {"foo": "bar"})

    def test_empty_string_errors(self) -> None:
        """An empty string yields an error."""
        with self.assertRaises(ValueError):
            _ = codedstrToDict("", is_passthru=True)

    def test_value_containing_underscore_preserved(self) -> None:
        """split('_', 1) keeps underscores in the value intact."""
        result = codedstrToDict("key_sub_val", is_passthru=True)
        self.assertEqual(result, {"key": "sub_val"})

    def test_part_without_underscore_skipped(self) -> None:
        """Segments without _ are silently ignored."""
        with self.assertRaises(ValueError):
            _ = codedstrToDict("key1_val1__nounderscore__key2_val2", is_passthru=True)

    def test_trailing_separator_ignored(self) -> None:
        """Trailing __ produces an empty segment that is skipped."""
        with self.assertRaises(ValueError):
            _ = codedstrToDict("key1_val1__", is_passthru=True)
        with self.assertRaises(ValueError):
            _ = codedstrToDict("key1_val1__key2")


class TestDictToCodedstr(unittest.TestCase):
    """Tests for the original dictToCodedstr (no convert_strs)."""

    def test_single_key_value(self) -> None:
        """A single key=value pair produces 'key_value'."""
        result = dictToCodedstr({"key": "value"})
        self.assertEqual(result, "key_value")

    def test_multiple_key_values(self) -> None:
        """Multiple pairs are joined by '__' in insertion order."""
        result = dictToCodedstr({"a": "1", "b": "2"})
        self.assertEqual(result, "a_1__b_2")

    def test_empty_dict_returns_empty_string(self) -> None:
        """An empty dict produces an empty string."""
        result = dictToCodedstr({})
        self.assertEqual(result, "")

    def test_numeric_values_use_repr(self) -> None:
        """Ints/floats are rendered via default repr (5, 1.23)."""
        result = dictToCodedstr({"repeat": 5, "threshold": 1.23})
        self.assertEqual(result, "repeat_5__threshold_1.23")

    def test_roundtrip_with_codedstrToDict(self) -> None:
        """Non-numeric string values survive the round-trip (keys without underscores)."""
        original = {cn.COL_COEFFICIENT_THRESHOLD: "alpha", cn.COL_IS_CHANGEPOINT_REMOVAL: "beta"}
        coded = dictToCodedstr(original)
        decoded = codedstrToDict(coded)
        self.assertEqual(decoded, original)

    def test_roundtrip_numeric_values_eval_to_int_float(self) -> None:
        """Documents that numeric strings are eval'd back to int/float (a real limitation)."""
        original = {"repeat": 3, "threshold": 0.1}
        coded = dictToCodedstr(original)
        decoded = codedstrToDict(coded)
        self.assertEqual(decoded["repeat"], 3)
        self.assertAlmostEqual(decoded[cn.COL_COEFFICIENT_THRESHOLD], 0.1, places=5)

    def test_roundtrip_succeeds_when_value_has_single_underscore(self) -> None:
        """Values with one _ survive the round-trip via split('_', 1)."""
        original = {"key": "sub_val"}
        coded = dictToCodedstr(original)
        decoded = codedstrToDict(coded, is_passthru=True)
        self.assertEqual(decoded, original)

    def test_roundtrip_fails_when_value_contains_separator(self) -> None:
        """Values containing '__' are ambiguous and break the round-trip."""
        problematic = {"key": "val__extra"}
        coded = dictToCodedstr(problematic)
        with self.assertRaises(ValueError):
            _ = codedstrToDict(coded)


class TestDictToCodedstrConvertStrs(unittest.TestCase):
    """Tests for dictToCodedstr with the convert_strs parameter (revised implementation).

    The source uses f"{value:.1e}" which produces proper scientific notation, e.g. 0.001 → "1.0e-03".
    Passing a non-numeric value raises ValueError because str cannot be formatted with '.1e'.
    """

    def test_convert_strs_formats_value(self) -> None:
        """A key in convert_strs gets its numeric value converted to scientific notation."""
        result = dictToCodedstr(
            {"max_fractional_reduction": 0.001}, convert_strs=["max_fractional_reduction"]
        )
        self.assertEqual(result, "maxreduction_1e-3")

    def test_convert_strs_with_multiple_keys(self) -> None:
        """Multiple keys in convert_strs all get scientific notation applied."""
        result = dictToCodedstr(
            {"a": 1.5, "b": 2.5}, convert_strs=["a", "b"]
        )
        self.assertEqual(result, "a_1.5e+00__b_2.5e+00")

    def test_convert_strs_non_numeric_value_raises(self) -> None:
        """Non-numeric values raise ValueError because str cannot be formatted with '.1e'."""
        with self.assertRaises(ValueError):
            dictToCodedstr(
                {"key": "value"}, convert_strs=["key"]
            )

    def test_convert_strs_ignores_missing_key(self) -> None:
        """Keys in convert_strs absent from the dict are skipped (no error)."""
        result = dictToCodedstr(
            {"a": 1}, convert_strs=["nonexistent"]
        )
        self.assertEqual(result, "a_1")

    def test_convert_strs_does_not_mutate_input(self) -> None:
        """The defensive copy (dct = dict(dct)) prevents modification of the caller's dict."""
        original = {"key": 0.5}
        original_copy = dict(original)
        result = dictToCodedstr(
            original, convert_strs=["key"]
        )
        self.assertEqual(result, "key_5e-1")
        self.assertEqual(original, original_copy)

class TestGetCSVPaths(unittest.TestCase):
    """Tests for the makeCSVPaths function."""

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    def test_makeCSVPaths_returns_list_of_paths(self) -> None:
        """makeCSVPaths returns a list of Path objects."""
        paths = util.makeCSVPaths(max_fractional_reduction=0.001, repeat=1)
        self.assertIsInstance(paths, list)
        self.assertTrue(all(isinstance(p, (os.PathLike, str)) for p in paths))

    def test_makeCSVPaths_empty_list_when_no_files(self) -> None:
        """If no files match the criteria, an empty list is returned."""
        with patch("util.os.listdir", return_value=[]):
            paths = util.makeCSVPaths(max_fractional_reduction=0.9999, repeat=99999)
            self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()


class TestMakeCSVPathsAdditional(unittest.TestCase):
    """Additional tests for makeCSVPaths function."""

    def test_makeCSVPaths_with_repeat(self) -> None:
        """makeCSVPaths includes repeat in the filename when provided."""
        # The function reads from cn.DATA_DIR directly
        with patch("os.listdir", return_value=["piecewise_predictions__numpoint_10__maxreduction_1e-3__repeat_5.csv"]),                 patch("os.path.join", side_effect=lambda d, f: d + "/" + f):
            with patch("src.constants.DATA_DIR", "/fake/path"):
                paths = util.makeCSVPaths(max_fractional_reduction=0.001, repeat=5)
                self.assertEqual(len(paths), 1)

    def test_makeCSVPaths_without_repeat(self) -> None:
        """makeCSVPaths returns paths without requiring repeat in filename."""
        with patch("os.listdir", return_value=["piecewise_predictions__numpoint_10__maxreduction_1e-3.csv"]),                 patch("os.path.join", side_effect=lambda d, f: d + "/" + f):
            with patch("src.constants.DATA_DIR", "/fake/path"):
                paths = util.makeCSVPaths(max_fractional_reduction=0.001, repeat=None)
                self.assertEqual(len(paths), 1)


class TestParseFilename(unittest.TestCase):
    """Tests for the parseFilename function."""

    def test_parseFilename_basic(self) -> None:
        """A basic filename is parsed correctly."""
        filename = "piecewise_predictions__numpoint_100.csv"
        result = parseFilename(filename)
        self.assertEqual(result, {cn.COL_NUM_POINT: 100})

    def test_parseFilename_with_maxreduction(self) -> None:
        """A filename with maxreduction is parsed correctly."""
        filename = "piecewise_predictions__numpoint_100__maxreduction_1e-3.csv"
        result = parseFilename(filename)
        self.assertEqual(result, {cn.COL_NUM_POINT: 100, cn.COL_MAX_FRACTIONAL_REDUCTION: 1e-3})

    def test_parseFilename_with_multiple_params(self) -> None:
        """A filename with multiple parameters is parsed correctly."""
        sel = dictToCodedstr({cn.COL_MAX_FRACTIONAL_REDUCTION: 0.001, cn.COL_IS_CHANGEPOINT_REMOVAL: 1}, convert_strs=["maxreduction"])
        filename = f"piecewise_predictions__numpoint_100__{sel}.csv"
        result = parseFilename(filename)
        self.assertIn(cn.COL_NUM_POINT, result)
        self.assertEqual(result[cn.COL_MAX_FRACTIONAL_REDUCTION], 0.001)
        # Note: parseFilename keeps ".csv" in the value string
        self.assertEqual(result[cn.COL_IS_CHANGEPOINT_REMOVAL], 1)

    def test_parseFilename_with_repeat(self) -> None:
        """A filename with repeat parameter is parsed correctly."""
        sel = dictToCodedstr({cn.COL_MAX_FRACTIONAL_REDUCTION: 0.001}, convert_strs=["maxreduction"])
        rep_sel = dictToCodedstr({cn.COL_REPEAT: 3})
        filename = f"piecewise_predictions__numpoint_100__{sel}__{rep_sel}.csv"
        result = parseFilename(filename)
        # Note: parseFilename keeps ".csv" in the value string
        self.assertEqual(result[cn.COL_REPEAT], 3)

    def test_parseFilename_invalid_prefix_raises(self) -> None:
        """A filename without the expected prefix raises ValueError."""
        filename = "other_predictions__numpoint_100.csv"
        with self.assertRaises(ValueError):
            parseFilename(filename)

    def test_parseFilename_empty_string_raises(self) -> None:
        """An empty string raises ValueError."""
        with self.assertRaises(ValueError):
            parseFilename("")


class TestParseChangepoints(unittest.TestCase):
    """Tests for the _parseChangepoints function."""

    def test_parseChangepoints_list_of_ints(self) -> None:
        """A list of ints is returned as-is."""
        result = _parseChangepoints([0, 5, 10])
        self.assertEqual(result, [0, 5, 10])

    def test_parseChangepoints_list_of_floats_raises(self) -> None:
        """A list of floats raises ValueError."""
        with self.assertRaises(ValueError):
            _parseChangepoints([0.0, 2.5])

    def test_parseChangepoints_string_representation(self) -> None:
        """A string representation of a list is parsed correctly."""
        result = _parseChangepoints("[0, 5, 10]")
        self.assertEqual(result, [0, 5, 10])

    def test_parseChangepoints_string_with_spaces(self) -> None:
        """A string with spaces is parsed correctly."""
        result = _parseChangepoints("[0, 5, 10]")
        self.assertEqual(result, [0, 5, 10])

    def test_parseChangepoints_np_int64_string(self) -> None:
        """A numpy.int64 string representation is parsed correctly."""
        result = _parseChangepoints("np.int64(5)")
        self.assertEqual(result, 5)

    def test_parseChangepoints_np_int64_with_value(self) -> None:
        """A numpy.int64 string with actual value is parsed correctly."""
        result = _parseChangepoints("np.int64(10)")
        self.assertEqual(result, 10)

    def test_parseChangepoints_empty_list(self) -> None:
        """An empty list is returned as-is."""
        result = _parseChangepoints([])
        self.assertEqual(result, [])

    def test_parseChangepoints_invalid_string_raises(self) -> None:
        """An invalid string raises ValueError."""
        with self.assertRaises(ValueError):
            _parseChangepoints("not_a_list")

    def test_parseChangepoints_none_raises(self) -> None:
        """None raises ValueError."""
        with self.assertRaises(ValueError):
            _parseChangepoints(None)

    def test_parseChangepoints_int_raises(self) -> None:
        """An integer raises ValueError."""
        with self.assertRaises(ValueError):
            _parseChangepoints(42)

    def test_parseChangepoints_dict_raises(self) -> None:
        """A dict raises ValueError."""
        with self.assertRaises(ValueError):
            _parseChangepoints({"key": "value"})


class TestGetPSDPredictionDFEdgeCases(unittest.TestCase):
    """Additional edge case tests for getPSDPredictionDF."""

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    def test_csv_files_provided_no_max_fractional_reduction(self) -> None:
        """When csv_files is provided, max_fractional_reduction need not be given."""
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        csv_name = f"piecewise_predictions__numpoint_100__{sel}.csv"
        payload = _model_df(0.001, n=2)
        with patch("util.os.path.exists", return_value=False),                 patch("pandas.read_csv", return_value=payload):
            df = getPSDPredictionDF(csv_files=[csv_name])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)

    def test_both_csv_files_and_max_fractional_reduction_raises(self) -> None:
        """Providing both csv_files and max_fractional_reduction raises ValueError."""
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        csv_name = f"piecewise_predictions__numpoint_100__{sel}.csv"
        with self.assertRaises(ValueError):
            getPSDPredictionDF(csv_files=[csv_name], max_fractional_reduction=0.001)

    def test_neither_csv_files_nor_max_fractional_reduction_raises(self) -> None:
        """Providing neither csv_files nor max_fractional_reduction raises ValueError."""
        with self.assertRaises(ValueError):
            getPSDPredictionDF()

    @unittest.skipIf(not _HAS_PSD_DATA, "No model CSV file found for testing.")
    def test_is_model_false_preserves_species_rows(self) -> None:
        """is_model=False allows species rows to be included."""
        payload = _mixed_df(0.001)
        sel = dictToCodedstr(
            {"maxreduction": 0.001}, convert_strs=["maxreduction"]
        )
        csv_name = f"piecewise_predictions__numpoint_100__{sel}.csv"
        with patch("util.os.path.exists", return_value=False),                 patch("pandas.read_csv", return_value=payload):
            df = getPSDPredictionDF(csv_files=[csv_name], is_model=False)
        self.assertEqual(len(df), 2)
        self.assertEqual(sum(df["aggregation_type"] == "species"), 1)
        self.assertEqual(sum(df["aggregation_type"] == "model"), 1)


class TestCodedstrToDictEdgeCases(unittest.TestCase):
    """Additional edge case tests for codedstrToDict."""

    @unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    def test_value_is_boolean_true(self) -> None:
        """A boolean 'True' value is evaluated correctly."""
        result = codedstrToDict("key_True", is_passthru=True)
        self.assertEqual(result, {"key": True})

    def test_value_is_boolean_false(self) -> None:
        """A boolean 'False' value is evaluated correctly."""
        result = codedstrToDict("key_False", is_passthru=True)
        self.assertEqual(result, {"key": False})

    def test_value_is_none(self) -> None:
        """A 'None' value is evaluated correctly."""
        result = codedstrToDict("key_None", is_passthru=True)
        self.assertEqual(result, {"key": None})

    def test_value_is_float(self) -> None:
        """A float value is evaluated correctly."""
        result = codedstrToDict("key_3.14", is_passthru=True)
        self.assertAlmostEqual(result["key"], 3.14)

    def test_empty_value_after_underscore(self) -> None:
        """An empty value after underscore is handled."""
        result = codedstrToDict("key_", is_passthru=True)
        self.assertEqual(result, {"key": ""})

    def test_multiple_empty_parts(self) -> None:
        """Multiple trailing separators are handled."""
        with self.assertRaises(ValueError):
            _ = codedstrToDict("key1_val1______", is_passthru=True)


class TestDictToCodedstrEdgeCases(unittest.TestCase):
    """Additional edge case tests for dictToCodedstr."""

    def test_convert_strs_empty_list(self) -> None:
        """An empty convert_strs list works like the original function."""
        result = dictToCodedstr({"a": 1}, convert_strs=[])
        self.assertEqual(result, "a_1")

    def test_value_with_multiple_underscores(self) -> None:
        """Values with multiple underscores are preserved (but break round-trip)."""
        result = dictToCodedstr({"key": "a_b_c"})
        self.assertEqual(result, "key_a_b_c")


if __name__ == "__main__":
    unittest.main()
