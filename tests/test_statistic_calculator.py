"""Tests for src/statistic_calculator.py: StatisticCalculator class."""

import src.constants as cn  # type: ignore
from src.statistic_calculator import StatisticCalculator, LARGE_VAL  # type: ignore

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import unittest


#########################################
# _is_percentile Tests
#########################################

class TestIsPercentile(unittest.TestCase):
    """Tests for StatisticCalculator._is_percentile()."""

    def setUp(self) -> None:
        self.calc = StatisticCalculator()

    def test_p25_is_percentile(self) -> None:
        """'p25' is recognized as a percentile."""
        self.assertTrue(self.calc._is_percentile("p25"))

    def test_p99_is_percentile(self) -> None:
        """'p99' is recognized as a percentile."""
        self.assertTrue(self.calc._is_percentile("p99"))

    def test_p0_is_percentile(self) -> None:
        """'p0' is recognized as a percentile."""
        self.assertTrue(self.calc._is_percentile("p0"))

    def test_p100_is_percentile(self) -> None:
        """'p100' is recognized as a percentile."""
        self.assertTrue(self.calc._is_percentile("p100"))

    def test_mean_is_not_percentile(self) -> None:
        """'mean' is not recognized as a percentile."""
        self.assertFalse(self.calc._is_percentile("mean"))

    def test_min_is_not_percentile(self) -> None:
        """'min' is not recognized as a percentile."""
        self.assertFalse(self.calc._is_percentile("min"))

    def test_p_is_not_percentile(self) -> None:
        """A bare 'p' with no digits is not recognized as a percentile."""
        self.assertFalse(self.calc._is_percentile("p"))

    def test_empty_string_is_not_percentile(self) -> None:
        """Empty string is not recognized as a percentile."""
        self.assertFalse(self.calc._is_percentile(""))


class TestAccuracyFractionToThreshold(unittest.TestCase):
    """Tests for StatisticCalculator.accuracyFractionToThreshold()."""

    def test_a80_returns_0_80(self) -> None:
        """'a80' maps to 0.80."""
        self.assertAlmostEqual(StatisticCalculator.accuracyFractionToThreshold("a80"), 0.80, places=5)

    def test_a90_returns_0_90(self) -> None:
        """'a90' maps to 0.90."""
        self.assertAlmostEqual(StatisticCalculator.accuracyFractionToThreshold("a90"), 0.90, places=5)

    def test_a95_returns_0_95(self) -> None:
        """'a95' maps to 0.95."""
        self.assertAlmostEqual(StatisticCalculator.accuracyFractionToThreshold("a95"), 0.95, places=5)

    def test_invalid_name_raises_value_error(self) -> None:
        """A non-accuracy-fraction column name raises ValueError."""
        with self.assertRaises(ValueError):
            StatisticCalculator.accuracyFractionToThreshold("mean")


#########################################
# StatisticCalculator Initialization Tests
#########################################

class TestStatisticCalculatorInit(unittest.TestCase):
    """Tests for StatisticCalculator.__init__."""

    def test_default_initialization(self) -> None:
        """Default initialization creates empty lists for all statistics."""
        calc = StatisticCalculator()
        for stat in cn.COLUMN_STATISTICS:
            self.assertIn(stat, calc.statistic_dct)
            self.assertEqual(calc.statistic_dct[stat], [])
        self.assertIn(cn.COL_LABEL, calc.statistic_dct)

    def test_statistic_dct_keys_match_constants(self) -> None:
        """All keys from cn.STATISTICS plus COL_LABEL are present."""
        calc = StatisticCalculator()
        expected_keys = set(cn.COLUMN_STATISTICS) | {cn.COL_LABEL}
        actual_keys = set(calc.statistic_dct.keys())
        self.assertEqual(expected_keys, actual_keys)


#########################################
# StatisticCalculator.add() - Basic Tests
#########################################

class TestAddBasic(unittest.TestCase):
    """Tests for StatisticCalculator.add() with basic valid data."""

    def setUp(self) -> None:
        self.calc = StatisticCalculator()

    def _make_arr(self, data):
        return np.array(data, dtype=float)

    def test_add_basic_mean_min_max(self) -> None:
        """Basic statistics are computed correctly for valid data."""
        arr = self._make_arr([1.0, 2.0, 3.0, 4.0, 5.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertAlmostEqual(dct[cn.COL_MEAN][0], 3.0)
        self.assertAlmostEqual(dct[cn.COL_MIN][0], 1.0)
        self.assertAlmostEqual(dct[cn.COL_MAX][0], 5.0)

    def test_add_count_is_valid_values(self) -> None:
        """count reflects number of valid (non-sentinel) values."""
        arr = self._make_arr([1.0, -1.0, 3.0, -1.0, 5.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertEqual(dct[cn.COL_COUNT][0], 3)

    def test_add_invalid_count_tracked(self) -> None:
        """invalid_count reflects number of sentinel (-1) values excluded."""
        arr = self._make_arr([1.0, -1.0, 3.0, -1.0, 5.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertEqual(dct[cn.COL_INVALID_COUNT][0], 2)

    def test_add_label_stored(self) -> None:
        """Label is stored correctly."""
        arr = self._make_arr([1.0, 2.0])
        self.calc.add(label="my_model", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertEqual(dct[cn.COL_LABEL][0], "my_model")

    def test_add_single_value(self) -> None:
        """Single value produces correct statistics."""
        arr = self._make_arr([42.0])
        self.calc.add(label="single", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertAlmostEqual(dct[cn.COL_MEAN][0], 42.0)
        self.assertAlmostEqual(dct[cn.COL_MIN][0], 42.0)
        self.assertAlmostEqual(dct[cn.COL_MAX][0], 42.0)
        self.assertEqual(dct[cn.COL_COUNT][0], 1)

    def test_add_two_values(self) -> None:
        """Two values produce correct statistics."""
        arr = self._make_arr([3.0, 7.0])
        self.calc.add(label="two", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertAlmostEqual(dct[cn.COL_MEAN][0], 5.0)
        self.assertAlmostEqual(dct[cn.COL_MIN][0], 3.0)
        self.assertAlmostEqual(dct[cn.COL_MAX][0], 7.0)
        self.assertEqual(dct[cn.COL_COUNT][0], 2)

    def test_add_multiple_calls_accumulate(self) -> None:
        """Multiple add() calls accumulate rows."""
        self.calc.add(label="a", value_arr=self._make_arr([1.0, 2.0]))
        self.calc.add(label="b", value_arr=self._make_arr([3.0, 4.0]))
        dct = self.calc.statistic_dct
        self.assertEqual(len(dct[cn.COL_LABEL]), 2)
        self.assertEqual(dct[cn.COL_LABEL][0], "a")
        self.assertEqual(dct[cn.COL_LABEL][1], "b")


#########################################
# StatisticCalculator.add() - Empty / All Invalid
#########################################

class TestAddEmptyAllInvalid(unittest.TestCase):
    """Tests for StatisticCalculator.add() with empty or all-invalid data."""

    def setUp(self) -> None:
        self.calc = StatisticCalculator()

    def _make_arr(self, data):
        return np.array(data, dtype=float)

    def test_add_empty_array_produces_row(self) -> None:
        """Empty array still produces a row with NaN values."""
        arr = self._make_arr([])
        self.calc.add(label="empty", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertEqual(len(dct[cn.COL_LABEL]), 1)
        self.assertTrue(np.isnan(dct[cn.COL_MEAN][0]))
        self.assertTrue(np.isnan(dct[cn.COL_MIN][0]))
        self.assertTrue(np.isnan(dct[cn.COL_MAX][0]))
        self.assertEqual(dct[cn.COL_COUNT][0], 0)

    def test_add_all_sentinel_produces_row(self) -> None:
        """All sentinel values still produce a row with zeroed stats."""
        arr = self._make_arr([-1.0, -1.0, -1.0])
        self.calc.add(label="all_invalid", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertEqual(len(dct[cn.COL_LABEL]), 1)
        self.assertTrue(np.isnan(dct[cn.COL_MEAN][0]))
        self.assertEqual(dct[cn.COL_COUNT][0], 0)
        self.assertEqual(dct[cn.COL_INVALID_COUNT][0], 3)

    def test_add_all_nan_produces_row(self) -> None:
        """All NaN values produce a row with zeroed stats."""
        arr = self._make_arr([float('nan'), float('nan')])
        self.calc.add(label="all_nan", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertEqual(len(dct[cn.COL_LABEL]), 1)
        self.assertTrue(np.isnan(dct[cn.COL_MEAN][0]))
        self.assertEqual(dct[cn.COL_COUNT][0], 0)

    def test_add_empty_percentiles_are_nan(self) -> None:
        """Percentile values are NaN when there is no valid data."""
        arr = self._make_arr([])
        self.calc.add(label="empty", value_arr=arr)
        dct = self.calc.statistic_dct
        for p in cn.COLUMN_STATISTICS:
            if self.calc._is_percentile(p):
                self.assertTrue(np.isnan(dct[p][0]))


#########################################
# StatisticCalculator.add() - NaN / Inf Handling
#########################################

class TestAddNaNInf(unittest.TestCase):
    """Tests for StatisticCalculator.add() with NaN and inf values."""

    def setUp(self) -> None:
        self.calc = StatisticCalculator()

    def _make_arr(self, data):
        return np.array(data, dtype=float)

    def test_add_nan_excluded_from_valid(self) -> None:
        """NaN values are excluded from valid count."""
        arr = self._make_arr([1.0, float('nan'), 3.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        # NaN is filtered out by the mask on line 72.
        self.assertEqual(dct[cn.COL_COUNT][0], 2)

    def test_add_inf_excluded_from_valid(self) -> None:
        """Positive inf values are excluded from valid count."""
        arr = self._make_arr([1.0, float('inf'), 3.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        # inf is filtered out by the mask on line 72.
        self.assertEqual(dct[cn.COL_COUNT][0], 2)

    def test_add_negative_inf_excluded_from_valid(self) -> None:
        """Negative inf values are excluded from valid count."""
        arr = self._make_arr([1.0, float('-inf'), 3.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        # -inf is filtered out by the mask on line 72.
        self.assertEqual(dct[cn.COL_COUNT][0], 2)

    def test_add_large_values_replaced_with_LARGE_VAL(self) -> None:
        """Values > LARGE_VAL are replaced with LARGE_VAL for aggregation."""
        arr = self._make_arr([1.0, 1e9, 3.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        # Mean should be computed with 1e6 replacing 1e9.
        expected_mean = (1.0 + LARGE_VAL + 3.0) / 3.0
        self.assertAlmostEqual(dct[cn.COL_MEAN][0], expected_mean, places=2)

    def test_add_large_values_counted_as_valid(self) -> None:
        """Values > LARGE_VAL are counted as valid."""
        arr = self._make_arr([1.0, 1e9, 3.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertEqual(dct[cn.COL_COUNT][0], 3)


#########################################
# StatisticCalculator.add() - Percentiles
#########################################

class TestAddPercentiles(unittest.TestCase):
    """Tests for StatisticCalculator.add() percentile computation."""

    def setUp(self) -> None:
        self.calc = StatisticCalculator()

    def _make_arr(self, data):
        return np.array(data, dtype=float)

    def test_add_percentiles_computed(self) -> None:
        """Percentile values are computed and stored."""
        arr = self._make_arr([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        # p25 should be around 3.25 (linear interpolation).
        self.assertAlmostEqual(dct["p25"][0], 3.25, places=1)
        # p95 should be around 9.55.
        self.assertAlmostEqual(dct["p95"][0], 9.55, places=1)

    def test_add_percentiles_with_sentinel(self) -> None:
        """Percentiles are computed excluding sentinel values."""
        arr = self._make_arr([1.0, -1.0, 3.0, -1.0, 5.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        # Only [1.0, 3.0, 5.0] used for percentiles.
        p25_expected = np.nanpercentile(np.array([1.0, 3.0, 5.0]), 25)
        self.assertAlmostEqual(dct["p25"][0], float(p25_expected), places=1)


#########################################
# StatisticCalculator.add() - is_non_negative Parameter
#########################################

class TestAddIsNonNegative(unittest.TestCase):
    """Tests for StatisticCalculator.add() with is_non_negative parameter."""

    def setUp(self) -> None:
        self.calc = StatisticCalculator()

    def _make_arr(self, data):
        return np.array(data, dtype=float)

    def test_is_non_negative_true_excludes_negatives(self) -> None:
        """When is_non_negative=True (default), negative values are excluded."""
        arr = self._make_arr([1.0, -5.0, 3.0])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        # Only [1.0, 3.0] are valid (non-negative).
        self.assertEqual(dct[cn.COL_COUNT][0], 2)

    def test_is_non_negative_false_includes_negatives(self) -> None:
        """When is_non_negative=False, negative values are included."""
        arr = self._make_arr([1.0, -5.0, 3.0])
        self.calc.add(label="test", value_arr=arr, is_non_negative=False)
        dct = self.calc.statistic_dct
        # All three values [1.0, -5.0, 3.0] are valid (no NaN/inf).
        self.assertEqual(dct[cn.COL_COUNT][0], 3)

    def test_is_non_negative_false_includes_negative_mean(self) -> None:
        """When is_non_negative=False, negative values affect the mean."""
        arr = self._make_arr([1.0, -5.0, 3.0])
        self.calc.add(label="test", value_arr=arr, is_non_negative=False)
        dct = self.calc.statistic_dct
        # Mean of [1.0, -5.0, 3.0] = -1/3.
        self.assertAlmostEqual(dct[cn.COL_MEAN][0], -1.0 / 3.0, places=5)

    def test_is_non_negative_false_still_excludes_nan(self) -> None:
        """Even with is_non_negative=False, NaN values are still excluded."""
        arr = self._make_arr([1.0, float('nan'), 3.0])
        self.calc.add(label="test", value_arr=arr, is_non_negative=False)
        dct = self.calc.statistic_dct
        # Only [1.0, 3.0] are valid (NaN excluded).
        self.assertEqual(dct[cn.COL_COUNT][0], 2)


#########################################
# StatisticCalculator.add() - Array Flattening
#########################################

class TestAddFlattening(unittest.TestCase):
    """Tests for StatisticCalculator.add() array flattening."""

    def setUp(self) -> None:
        self.calc = StatisticCalculator()

    def _make_arr(self, data):
        return np.array(data, dtype=float)

    def test_add_flattens_2d_array(self) -> None:
        """2D arrays are flattened before aggregation."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertAlmostEqual(dct[cn.COL_MEAN][0], 2.5)

    def test_add_flattens_3d_array(self) -> None:
        """3D arrays are flattened before aggregation."""
        arr = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.statistic_dct
        self.assertAlmostEqual(dct[cn.COL_MEAN][0], 2.5)


#########################################
# StatisticCalculator.dataframe Property Tests
#########################################

class TestDataframeProperty(unittest.TestCase):
    """Tests for StatisticCalculator.dataframe property."""

    def test_returns_dataframe(self) -> None:
        """dataframe returns a pandas DataFrame."""
        calc = StatisticCalculator()
        result = calc.dataframe
        self.assertIsInstance(result, pd.DataFrame)

    def test_dataframe_has_correct_columns(self) -> None:
        """DataFrame has columns matching cn.STATISTICS + COL_LABEL."""
        calc = StatisticCalculator()
        df = calc.dataframe
        expected_cols = (set(cn.COLUMN_STATISTICS) | {cn.COL_LABEL} 
                         | set(cn.COLUMN_ACCURACY_FRACTIONS))
        actual_cols = set(df.columns)
        self.assertEqual(expected_cols, actual_cols)

    def test_dataframe_after_add(self) -> None:
        """DataFrame contains data after add() calls."""
        calc = StatisticCalculator()
        calc.add(label="test", value_arr=np.array([1.0, 2.0, 3.0]))
        df = calc.dataframe
        self.assertEqual(len(df), 1)

    def test_dataframe_multiple_adds(self) -> None:
        """DataFrame has one row per add() call."""
        calc = StatisticCalculator()
        calc.add(label="a", value_arr=np.array([1.0]))
        calc.add(label="b", value_arr=np.array([2.0]))
        df = calc.dataframe
        self.assertEqual(len(df), 2)


#########################################
# StatisticCalculator accuracy_fraction_dct Tests
#########################################

class TestAccuracyFractionInit(unittest.TestCase):
    """Tests for StatisticCalculator.__init__ accuracy_fraction_dct."""

    def test_accuracy_fraction_dct_initialized(self) -> None:
        """accuracy_fraction_dct is initialized with entries for all COLUMN_ACCURACY_FRACTION keys."""
        calc = StatisticCalculator()
        self.assertTrue(hasattr(calc, "accuracy_fraction_dct"))
        expected_keys = set(cn.COLUMN_ACCURACY_FRACTIONS)
        actual_keys = set(calc.accuracy_fraction_dct.keys())
        self.assertEqual(expected_keys, actual_keys)

    def test_accuracy_fraction_dct_values_empty_lists(self) -> None:
        """Each entry in accuracy_fraction_dct is an empty list after init."""
        calc = StatisticCalculator()
        for key in cn.COLUMN_ACCURACY_FRACTIONS:
            self.assertEqual(calc.accuracy_fraction_dct[key], [])


class TestAccuracyFractionComputation(unittest.TestCase):
    """Tests for accuracy fraction computation in add()."""

    def setUp(self) -> None:
        self.calc = StatisticCalculator()

    def _make_arr(self, data):
        return np.array(data, dtype=float)

    def test_a80_true_when_mean_above_threshold(self) -> None:
        """a80 is True (1.0) when mean accuracy >= 80%."""
        arr = self._make_arr([0.9, 0.95, 0.85])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.accuracy_fraction_dct
        # Mean of [0.9, 0.95, 0.85] = 0.9 >= 0.8 -> True (1.0)
        self.assertEqual(dct[cn.COL_A80][0], 1.0)

    def test_a80_false_when_mean_below_threshold(self) -> None:
        """a80 is False (0.0) when mean accuracy < 80%."""
        arr = self._make_arr([0.5, 0.6, 0.7])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.accuracy_fraction_dct
        # Mean of [0.5, 0.6, 0.7] = 0.6 < 0.8 -> False (0.0)
        self.assertEqual(dct[cn.COL_A80][0], 0.0)

    def test_a90_true_when_mean_above_threshold(self) -> None:
        """a90 is True (1.0) when mean accuracy >= 90%."""
        arr = self._make_arr([0.95, 0.97])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.accuracy_fraction_dct
        # Mean of [0.95, 0.97] = 0.96 >= 0.9 -> True (1.0)
        self.assertEqual(dct[cn.COL_A90][0], 1.0)

    def test_a95_true_when_mean_above_threshold(self) -> None:
        """a95 is True (1.0) when mean accuracy >= 95%."""
        arr = self._make_arr([0.96, 0.98])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.accuracy_fraction_dct
        # Mean of [0.96, 0.98] = 0.97 >= 0.95 -> True (1.0)
        self.assertEqual(dct[cn.COL_A95][0], 1.0)

    def test_a95_false_when_mean_below_threshold(self) -> None:
        """a95 is False (0.0) when mean accuracy < 95%."""
        arr = self._make_arr([0.8, 0.9])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.accuracy_fraction_dct
        # Mean of [0.8, 0.9] = 0.85 < 0.95 -> False (0.0)
        self.assertEqual(dct[cn.COL_A95][0], 0.0)

    def test_all_accuracy_fractions_computed(self) -> None:
        """All three accuracy fraction columns are populated for a single add()."""
        arr = self._make_arr([0.85, 0.92, 0.78])
        self.calc.add(label="test", value_arr=arr)
        for col in cn.COLUMN_ACCURACY_FRACTIONS:
            self.assertIn(col, self.calc.accuracy_fraction_dct)
            # Should be a numeric float (not bool or NaN).
            self.assertIsInstance(self.calc.accuracy_fraction_dct[col][0], (float, int))

    def test_accuracy_fraction_stored_as_float(self) -> None:
        """Accuracy fraction values are Python floats, not numpy.bool_."""
        arr = self._make_arr([0.95, 0.98])
        self.calc.add(label="test", value_arr=arr)
        val = self.calc.accuracy_fraction_dct[cn.COL_A90][0]
        # Should be a numeric float (1.0), not True/False.
        self.assertIsInstance(val, (float, int))

    def test_accuracy_fraction_with_nan_excluded(self) -> None:
        """NaN values are excluded from mean calculation."""
        arr = np.array([0.9, float("nan"), 0.85])
        self.calc.add(label="test", value_arr=arr)
        dct = self.calc.accuracy_fraction_dct
        # Mean of [0.9, 0.85] (NaN excluded) = 0.875 >= 0.8 -> True (1.0)
        self.assertEqual(dct[cn.COL_A80][0], 1.0)

    def test_accuracy_fraction_nan_when_no_valid_values(self) -> None:
        """Accuracy fractions are NaN when all values are invalid."""
        arr = np.array([float("nan"), float("-inf")])
        self.calc.add(label="test", value_arr=arr, is_non_negative=False)
        for col in cn.COLUMN_ACCURACY_FRACTIONS:
            val = self.calc.accuracy_fraction_dct[col][0]
            self.assertTrue(np.isnan(val))


class TestAccuracyFractionInDataFrame(unittest.TestCase):
    """Tests that accuracy fractions appear in the output DataFrame."""

    def test_dataframe_includes_accuracy_columns(self) -> None:
        """The DataFrame has a80, a90, a95 columns after add()."""
        calc = StatisticCalculator()
        calc.add(label="test", value_arr=np.array([0.9, 0.95]))
        df = calc.dataframe
        for col in cn.COLUMN_ACCURACY_FRACTIONS:
            self.assertIn(col, df.columns)

    def test_dataframe_accuracy_columns_are_numeric(self) -> None:
        """Accuracy fraction columns are numeric (not boolean strings)."""
        calc = StatisticCalculator()
        calc.add(label="test", value_arr=np.array([0.9, 0.95]))
        df = calc.dataframe
        for col in cn.COLUMN_ACCURACY_FRACTIONS:
            # Values should be numeric (float/int), not bool or string.
            val = df[col].iloc[0]
            self.assertIsInstance(val, (float, np.floating))


if __name__ == "__main__":
    unittest.main()