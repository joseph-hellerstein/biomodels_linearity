"""Tests for NLCurve."""

import os
import numpy as np
import tempfile
import unittest

#matplotlib.use("Agg")  #  type: ignore

import matplotlib.pyplot as plt  # noqa: E402  # type: ignore
import numpy as np  # type: ignore
import pandas as pd # type: ignore
import src.constants as cn  # type: ignore
from src.util import makeCSVPaths
from src.nl_curve import NLCurve  # type: ignore


IGNORE_TESTS = False
_HAS_PSD_DATA = os.path.isdir(cn.PSD_DATA_DIR)


def _biomd_name(model_num: int) -> str:
    """Construct a system_id matching Model.getBiomodelName's format."""
    return f"BIOMD{model_num:010d}"


def _write_csv(tmpdir: str, rows: list) -> str:
    """Write a small PiecewisePredictions CSV into tmpdir and return its path."""
    path = os.path.join(tmpdir, "predictions.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestNLCurveInit(unittest.TestCase):
    """Tests for NLCurve.__init__."""

    def test_constructs_with_two_boundaries(self) -> None:
        """A two-element input produces one segment of the expected length."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 5])
        self.assertEqual(list(nl._boundaries), [0.0, 5.0])
        np.testing.assert_array_equal(nl._segment_arr, [5.0])

    def test_sorts_unsorted_boundaries(self) -> None:
        """Boundaries are sorted regardless of input order."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 3, 1])
        self.assertEqual(list(nl._boundaries), [0.0, 1.0, 3.0])

    def test_curve_ends_at_one(self) -> None:
        """The cumulative curve always reaches 1.0 at the largest segment length."""
        if IGNORE_TESTS:
            return
        for bounds in [[0, 1, 2], [0, 1, 3], [0, 2, 5]]:
            nl = NLCurve(bounds)  # type: ignore
            end_value = float(nl.curve_ser.iloc[-1])
            self.assertAlmostEqual(end_value, 1.0)

    def test_curve_values_in_unit_interval(self) -> None:
        """All curve values lie in [0, 1]."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3, 4])
        self.assertTrue((nl.curve_ser.to_numpy() >= 0).all())
        self.assertTrue((nl.curve_ser.to_numpy() <= 1.0 + 1e-9).all())

    def test_empty_boundaries_raises(self) -> None:
        """An empty list raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            NLCurve([])

    def test_single_boundary_raises(self) -> None:
        """A single-element list raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            NLCurve([5])


class TestNLCurvecopy(unittest.TestCase):
    """Tests for NLCurve.copy()."""

    def test_returns_new_instance(self) -> None:
        """copy() returns an independent NLCurve instance."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3])
        other = nl.copy()
        self.assertIsInstance(other, NLCurve)
        self.assertIsNot(nl, other)

    def test_copy_preserves_curve(self) -> None:
        """The copy has identical curve values."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 2])
        other = nl.copy()
        pd.testing.assert_series_equal(nl.curve_ser, other.curve_ser)

    def test_copy_is_independent(self) -> None:
        """Mutating the original does not affect the copy."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 2])
        other = nl.copy()
        nl._boundaries.append(5.0)
        self.assertEqual(list(other._boundaries), [0, 1, 2])


class TestNLCurveDist(unittest.TestCase):
    """Tests for NLCurve.dist()."""

    def test_self_distance_is_zero(self) -> None:
        """A curve's distance to itself is exactly 0."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 2, 3])
        self.assertAlmostEqual(nl.dist(nl), 0.0)

    def test_distance_symmetric(self) -> None:
        """dist(a, b) == dist(b, a)."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2, 5])
        b = NLCurve([0, 2, 5])
        self.assertAlmostEqual(a.dist(b), b.dist(a))

    def test_distance_positive_for_different_curves(self) -> None:
        """Different boundaries produce a strictly positive distance."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2, 3])
        b = NLCurve([0, 1, 3])
        distance = a.dist(b)
        self.assertGreater(a.dist(b), 0.0)

    def test_distance_identical_boundaries_zero(self) -> None:
        """Curves with identical boundaries have distance zero."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2])
        b = NLCurve([0, 1, 2])
        self.assertAlmostEqual(a.dist(b), 0.0)


class TestMakeMergedCurve(unittest.TestCase):
    """Tests for NLCurve.makeMergedCurve()."""

    def test_returns_series(self) -> None:
        """makeMergedCurve returns a pd.Series."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2, 4, 5])
        b = NLCurve([0, 2, 5])
        merged = a.mergeIndex(b)
        self.assertIsInstance(merged, pd.Series)

    def test_index_is_union_of_both(self) -> None:
        """The merged index contains every unique segment length from both curves."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 3, 5])   # lengths {1, 2}
        b = NLCurve([0, 2, 5])   # lengths {2, 3}
        merged = a.mergeIndex(b)
        expected_index = sorted({1.0 / 5, 2.0/ 5, 3.0/5})
        self.assertEqual(list(merged.index), expected_index)

    def test_cumsum_ends_at_one(self) -> None:
        """The returned series cumulates to 1.0."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2, 5])
        b = NLCurve([0, 2, 5])
        merged = a.mergeIndex(b)
        self.assertAlmostEqual(float(merged.iloc[-1]), 1.0)
    
    def test_cumsum_value_error_if_incompatible(self) -> None:
        """The returned series cumulates to 1.0."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 2, 7])
        b = NLCurve([0, 2, 5])
        with self.assertRaises(ValueError):
            _ = a.mergeIndex(b)

    def test_cumsum_monotonic_non_decreasing(self) -> None:
        """The returned series is monotonic non-decreasing."""
        if IGNORE_TESTS:
            return
        a = NLCurve([0, 1, 3, 5])
        b = NLCurve([0, 2, 5])
        merged = a.mergeIndex(b)
        diffs = np.diff(merged.to_numpy())
        self.assertTrue((diffs >= -1e-12).all())


class TestMakeNLCurve(unittest.TestCase):
    """Tests for NLCurve._makeNLCurve()."""

    def test_curve_is_cumulative(self) -> None:
        """The curve is monotonically non-decreasing."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3, 4, 7])
        curve = nl._makeNLCurve()
        diffs = np.diff(curve.to_numpy())
        self.assertTrue((diffs >= -1e-12).all())

    def test_curve_ends_at_one(self) -> None:
        """The final value of the curve is 1.0."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3, 4])
        value = float(nl._makeNLCurve().iloc[-1])
        self.assertAlmostEqual(value, 1.0)

    def test_curve_first_value_equals_smallest_segment_fraction(self) -> None:
        """The first curve value equals the smallest segment's share of total time."""
        if IGNORE_TESTS:
            return
        nl = NLCurve([0, 1, 3])   # segments {1, 2}; smallest=1/3
        self.assertAlmostEqual(float(nl._makeNLCurve().iloc[0]), 1.0 / 3.0)


class TestFromPSDPredictions(unittest.TestCase):
    """Tests for NLCurve.fromPSDPredictions()."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def _make_csv(self, tmpdir: str, rows: list) -> str:
        filename = f"piecewise_predictions__dummy_{np.random.randint(1, 100000)}__numpoint_50.csv"
        path = os.path.join(tmpdir, filename)
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    @unittest.skipIf(not _HAS_PSD_DATA, "No model CSV file found for testing.")
    def test_model_level_creates_curve(self) -> None:
        """Selecting model-level aggregation returns a valid NLCurve."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_csv(
                tmpdir,
                [{
                    cn.COL_CHANGEPOINTS: [2.0, 4.0],
                    cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL,
                    cn.COL_SYSTEM_ID: _biomd_name(1),
                    cn.COL_COUNT: 10,
                }],
            )
            nl = NLCurve.fromPSDPredictions(path, model_num=1)
            self.assertIsInstance(nl, NLCurve)

    @unittest.skipIf(not _HAS_PSD_DATA, "No model CSV file found for testing.")
    def test_species_level_filters_by_name(self) -> None:
        """Passing species_name selects the matching aggregation row."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_csv(
                tmpdir,
                [
                    {
                        cn.COL_CHANGEPOINTS: [2],
                        cn.COL_AGGREGATION_TYPE: "S1",
                        cn.COL_SYSTEM_ID: _biomd_name(1),
                        cn.COL_COUNT: 5,
                    },
                    {
                        cn.COL_CHANGEPOINTS: [3, 4],
                        cn.COL_AGGREGATION_TYPE: cn.COL_AGGREGATION_TYPE_MODEL,
                        cn.COL_SYSTEM_ID: _biomd_name(1),
                        cn.COL_COUNT: 5,
                    },
                ],
            )
            nl = NLCurve.fromPSDPredictions(path, model_num=1, species_name="S1")
            self.assertIn(2.0, nl._boundaries)

    @unittest.skipIf(not _HAS_PSD_DATA, "No model CSV file found for testing.")
    def test_no_matching_rows_raises(self) -> None:
        """When no rows match the filter, ValueError is raised."""
        if IGNORE_TESTS:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_csv(
                tmpdir,
                [{
                    cn.COL_CHANGEPOINTS: [2.0],
                    cn.COL_AGGREGATION_TYPE: "model",
                    cn.COL_SYSTEM_ID: _biomd_name(42),
                    cn.COL_COUNT: 5,
                }],
            )
            with self.assertRaises(ValueError):
                NLCurve.fromPSDPredictions(path, model_num=1)


class TestPlotNLCurve(unittest.TestCase):
    """Smoke test for NLCurve.plotNLCurve()."""

    def test_plot_does_not_raise(self) -> None:
        """plotNLCurve executes without raising an exception."""
        if IGNORE_TESTS:
            return
        import matplotlib.pyplot as plt  # noqa: E402  # type: ignore
        nl = NLCurve([0, 1, 2, 3, 5], name="Test Curve")
        fig = plt.figure()
        try:
            nl.plotNLCurve()
            #plt.show()
        finally:
            plt.close(fig)

class TestEnd2End(unittest.TestCase):
    """Tests for NLCurve.fromPSDPredictions()."""

    model_num = 343

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        if not os.path.isdir(cn.PSD_DATA_DIR):
            self.skipTest("No PSD data directory found.")
        model_paths = makeCSVPaths(max_fractional_reduction=0.001, repeat=3)
        if not model_paths:
            self.skipTest("No matching CSV files in PSD data directory.")
        self.nl_curve = NLCurve.fromPSDPredictions(str(model_paths[0]),
                model_num=self.model_num)

    def test_model_level_creates_curve(self) -> None:
        """Selecting model-level aggregation returns a valid NLCurve."""
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.nl_curve, NLCurve)
        self.assertGreater(len(self.nl_curve.curve_ser), 0)

    def test_plot_does_not_raise(self) -> None:
        """plotNLCurve executes without raising an exception."""
        if IGNORE_TESTS:
            return
        import matplotlib.pyplot as plt  # noqa: E402  # type: ignore
        nl = self.nl_curve
        fig = plt.figure()
        try:
            nl.plotNLCurve()
            #plt.show()
        finally:
            plt.close(fig)

    def test_multiple_plots_do_not_raise(self) -> None:
        """Multiple calls to plotNLCurve do not raise exceptions."""
        if IGNORE_TESTS:
            return
        import matplotlib.pyplot as plt  # noqa: E402  # type: ignore
        import matplotlib.axes  # noqa: E402  # type: ignore
        model_path = makeCSVPaths(max_fractional_reduction=0.001, repeat=3)[0]
        nl_curve = NLCurve.fromPSDPredictions(str(model_path),
                model_num=self.model_num)
        nl = self.nl_curve
        fig = plt.figure()
        try:
            ax = nl.plotNLCurve(data_src="test1")
            assert isinstance(ax, matplotlib.axes.Axes)
            nl_curve.plotNLCurve(ax=ax, data_src="test2")
        finally:
            plt.close(fig)

    def test_distance(self) -> None:
        """Distance between two NLCurves is symmetric."""
        if IGNORE_TESTS:
            return
        model_path = makeCSVPaths(max_fractional_reduction=0.001, repeat=3)[0]
        nl_curve = NLCurve.fromPSDPredictions(str(model_path),
                model_num=self.model_num)
        nl = self.nl_curve
        self.assertTrue(np.isclose(nl.dist(nl), nl.dist(nl)))
        distance1 = nl_curve.dist(nl)
        distance2 = nl.dist(nl_curve)
        self.assertAlmostEqual(distance1, distance2)


if __name__ == "__main__":
    unittest.main()
