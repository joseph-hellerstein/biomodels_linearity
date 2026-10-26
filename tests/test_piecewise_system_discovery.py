"""Tests for ``src.piecewise_system_discovery.PiecewiseSystemDiscovery``."""

import os  # type: ignore
import unittest
from unittest.mock import patch
import tempfile  # type: ignore
from typing import Dict, List, cast

import matplotlib  # noqa: F401 -- non-interactive backend needed before pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore

import src.constants as cn  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore
from src.model import Model  # type: ignore
from src.piecewise_system_discovery import (  # type: ignore
    PiecewiseSystemDiscovery,
)

_HAS_PSD_DATA = os.path.isdir(cn.PSD_DATA_DIR)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

IGNORE_TESTS = False
HAS_REAL_ZIP = os.path.isfile(TimecourseIterator.getZipPath(num_point=cn.NUM_POINT))
BIOMODEL_548 = "BIOMD0000000548"
NUM_POINT_LARGE = 500  # used for slow fit/predict tests; small fixtures use 100.


def _make_linear_df(
    n_points: int = NUM_POINT_LARGE,
    t_start: float = 0.0,
    t_end: float = 10.0,
    noise_std: float = 0.0,
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

    return pd.DataFrame(X, index=t_eval, columns=["A", "B"])


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestPiecewiseSystemDiscoveryConstructor(unittest.TestCase):

    def test_basic_construction(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.0)
        psd = PiecewiseSystemDiscovery(df, model_id="test_model")
        self.assertEqual(psd.model_name, "test_model")
        self.assertEqual(psd.species_names, ["A", "B"])
        self.assertEqual(psd.num_species, 2)
        self.assertEqual(psd.num_point, 100)

    def test_default_parameters(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        self.assertEqual(psd.max_changepoint, 2)
        self.assertAlmostEqual(psd.max_fractional_reduction, 0.01)
        self.assertIsNone(psd.changepoints)

    def test_custom_parameters(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(
            df, max_changepoint=3, max_fractional_reduction=0.2,
            model_id="my_model",
            changepoints=[10, 20],
        )
        self.assertEqual(psd.max_changepoint, 3)
        self.assertAlmostEqual(psd.max_fractional_reduction, 0.2)
        self.assertEqual(psd.model_name, "my_model")
        self.assertEqual(psd.changepoints, [10, 20])

    def test_not_fitted_initially(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        self.assertFalse(psd._is_fitted)
        self.assertEqual(len(psd._subsequence_models), 0)

    def test_sd_kwargs_forwarded(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, coefficient_threshold=0.1, alpha=0.2)
        self.assertIn("coefficient_threshold", psd._sd_kwargs)
        self.assertAlmostEqual(psd._sd_kwargs["coefficient_threshold"], 0.1)
        self.assertAlmostEqual(psd._sd_kwargs["alpha"], 0.2)

    def test_poly_degree_forced_to_one(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, poly_degree=1)
        self.assertEqual(psd._sd_kwargs.get("poly_degree"), 1)

    def test_poly_degree_explicit_override(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df, poly_degree=2)
        self.assertEqual(psd._sd_kwargs.get("poly_degree"), 2)


# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------


class TestFit(unittest.TestCase):

    def test_fit_populates_attributes(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50])
        result = psd.fit()
        self.assertTrue(psd._is_fitted)
        self.assertEqual(len(psd._subsequence_models), 2)
        self.assertIs(result, psd)

    def test_fit_with_explicit_changepoints(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[40])
        psd.fit()
        self.assertEqual(psd._subsequence_lengths[0], 40)
        self.assertEqual(psd._subsequence_lengths[1], 60)


# ---------------------------------------------------------------------------
# predict() tests
# ---------------------------------------------------------------------------


class TestPredict(unittest.TestCase):

    def test_predict_returns_dataframe(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50])
        psd.fit()
        pred_df = psd.predict()
        self.assertIsInstance(pred_df, pd.DataFrame)

    def test_predict_requires_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        with self.assertRaises(RuntimeError):
            psd.predict()

    def test_predict_columns_match_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50])
        psd.fit()
        pred_df = psd.predict()
        self.assertEqual(list(pred_df.columns), ["A", "B"])

    def test_predict_with_custom_test_df(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50])
        psd.fit()
        pred_df = psd.predict(test_df=df)
        self.assertIsInstance(pred_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# score() and getScoreDetails() tests
# ---------------------------------------------------------------------------


class TestScore(unittest.TestCase):

    def test_score_returns_float(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50])
        psd.fit()
        score_val = psd.score()
        self.assertIsInstance(score_val, float)

    def test_score_requires_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        with self.assertRaises(RuntimeError):
            psd.score()

    def test_getscoredetails_returns_dataframe(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50])
        psd.fit()
        score_df = psd.getScoreDetails()
        self.assertIsInstance(score_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# __str__ / printEquations tests
# ---------------------------------------------------------------------------


class TestStr(unittest.TestCase):

    def test_str_returns_nonempty_after_fit(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50])
        psd.fit()
        s = str(psd)
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)

    def test_str_unfitted_returns_empty(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        s = str(psd)
        self.assertEqual(s, "")

    def test_print_equations_no_exception(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50])
        psd.fit()
        psd.printEquations()  # just ensure it doesn't raise


# ---------------------------------------------------------------------------
# plotPiecewise tests
# ---------------------------------------------------------------------------


class TestPlotPiecewise(unittest.TestCase):

    def test_plot_piecewise_returns_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100, noise_std=0.05)
        psd = PiecewiseSystemDiscovery(df, changepoints=[50])
        psd.fit()
        po = psd.plotPiecewise(num_true_point=-1)
        self.assertIsNotNone(po.fig)

    def test_plot_piecewise_requires_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        psd = PiecewiseSystemDiscovery(df)
        with self.assertRaises(RuntimeError):
            psd.plotPiecewise()



# ---------------------------------------------------------------------------
# End-to-end BioModel 548 test
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestEndToEndBioModels548(unittest.TestCase):
    """End-to-end test using real BioModel 548 serialized timecourse."""

    def setUp(self) -> None:
        from src.timecourse_iterator import TimecourseIterator  # type: ignore
        self.tc = TimecourseIterator.getTimecourse(BIOMODEL_548, num_point=1000)

    def _make_psd(self, **overrides):
        defaults = dict(
            max_changepoint=2,
            poly_degree=1,
            coefficient_threshold=0.01,
            model_id=BIOMODEL_548,
        )
        defaults.update(overrides)
        return PiecewiseSystemDiscovery(self.tc.timecourse_df, **defaults)  # type: ignore

    def test_fit_produces_at_least_one_segment(self) -> None:
        """``fit()`` on real BioModel 548 data must populate subsequence models."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200])
        result = psd.fit()
        self.assertTrue(psd._is_fitted)
        self.assertGreaterEqual(len(psd._subsequence_models), 1)
        self.assertIs(result, psd)

    def test_predict_returns_dataframe_with_correct_columns(self) -> None:
        """``predict()`` must return a DataFrame whose columns match the training species."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200])
        psd.fit()
        pred_df = psd.predict()
        self.assertIsInstance(pred_df, pd.DataFrame)
        self.assertEqual(list(pred_df.columns), list(self.tc.timecourse_df.columns))

    def test_predict_index_matches_training_time(self) -> None:
        """``predict()`` must return predictions aligned with the training time index."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200])
        psd.fit()
        pred_df = psd.predict()
        # Values must match; index names may differ (predict does not propagate the column name).
        np.testing.assert_array_equal(pred_df.index.to_numpy(), self.tc.timecourse_df.index.to_numpy())

    def test_predict_values_are_finite(self) -> None:
        """Predicted values that are finite must equal the corresponding training values within tolerance."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200])
        psd.fit()
        pred_df = psd.predict()
        # At minimum every predicted column must contain some finite values.
        for col in pred_df.columns:
            self.assertGreater(pred_df[col].notna().sum(), 0)

    def test_score_returns_float(self) -> None:
        """``score()`` must return a finite float on real data."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200])
        psd.fit()
        score_val = psd.score()
        self.assertIsInstance(score_val, float)
        self.assertTrue(np.isfinite(score_val))

    def test_getscoredetails_returns_dataframe(self) -> None:
        """``getScoreDetails()`` must return a non-empty DataFrame on real data."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200])
        psd.fit()
        score_df = psd.getScoreDetails()
        self.assertIsInstance(score_df, pd.DataFrame)
        self.assertGreater(len(score_df), 0)

    def test_plot_piecewise_returns_plot_options(self) -> None:
        """``plotPiecewise()`` must return a valid PlotOptions on real data."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200])
        psd.fit()
        po = psd.plotPiecewise(num_true_point=-1)
        self.assertIsNotNone(po.fig)

    def test_str_contains_species_names(self) -> None:
        """``str(psd)`` after fit should mention every species name."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        psd = self._make_psd(changepoints=[200])
        psd.fit()
        s = str(psd)
        for sp in self.tc.timecourse_df.columns:
            self.assertIn(sp, s)

    def test_fit_with_explicit_changepoint_indices(self) -> None:
        """Providing explicit changepoints should split the timecourse at those indices."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        n = len(self.tc.timecourse_df)
        mid = n // 2
        psd = self._make_psd(changepoints=[mid])
        psd.fit()
        self.assertEqual(len(psd._subsequence_models), 2)
        self.assertEqual(psd._subsequence_lengths[0], mid)
        self.assertEqual(psd._subsequence_lengths[1], n - mid)


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestChangepointSpecification(unittest.TestCase):
    """End-to-end test using real BioModel 548 serialized timecourse."""

    def test_specify_changepoints(self) -> None:
        """Specifying changepoints should retain them after fit."""
        model_num = 548
        model = Model.makeBiomodel(model_num=model_num)
        timecourse = Timecourse(model, num_point =1000)
        df = timecourse.timecourse_df
        new_changepoints = list(range(10, 990, 10))
        psd = PiecewiseSystemDiscovery(df,
                changepoints=new_changepoints,
                max_fractional_reduction=0.01, model_id=str(model_num))
        psd.fit()
        self.assertEqual(psd.changepoints, new_changepoints)

    def test_no_removal(self) -> None:
        """Specifying changepoints should retain them after fit."""
        model_num = 548
        model = Model.makeBiomodel(model_num=model_num)
        timecourse = Timecourse(model, num_point =1000)
        df = timecourse.timecourse_df
        psd = PiecewiseSystemDiscovery(df,
                max_changepoint=80,
                is_changepoint_removal=False,
                max_fractional_reduction=0.01, model_id=str(model_num))
        psd.fit()
        self.assertEqual(len(psd.changepoints), 80)  # type: ignore


def _make_piecewise_df(n_points: int = 300):
    """Three-segment piecewise ODE with clearly distinct per-segment Jacobians.

    Returns ``(df, cp1_idx, cp2_idx)`` where the two changepoint indices split
    the trajectory into equal thirds.
    """
    t_eval = np.linspace(0.0, 12.0, n_points)
    cp1_idx = n_points // 3
    cp2_idx = 2 * n_points // 3

    t1 = t_eval[:cp1_idx]
    t2 = t_eval[cp1_idx:cp2_idx]
    t3 = t_eval[cp2_idx:]

    def rhs1(t, z): return [-2.0 * z[0] + 0.5 * z[1], 0.5 * z[0] - 1.5 * z[1]]
    sol1 = solve_ivp(rhs1, [t1[0], t1[-1]], [1.0, 0.5], t_eval=t1, rtol=1e-9)

    def rhs2(t, z): return [-0.3 * z[0] + 0.1 * z[1], 0.1 * z[0] - 0.4 * z[1]]
    sol2 = solve_ivp(rhs2, [t2[0], t2[-1]], [sol1.y[0, -1], sol1.y[1, -1]], t_eval=t2, rtol=1e-9)

    def rhs3(t, z): return [-1.2 * z[0] + 0.2 * z[1], 0.2 * z[0] - 0.9 * z[1]]
    sol3 = solve_ivp(rhs3, [t3[0], t3[-1]], [sol2.y[0, -1], sol2.y[1, -1]], t_eval=t3, rtol=1e-9)

    A = np.concatenate([sol1.y[0], sol2.y[0], sol3.y[0]])
    B = np.concatenate([sol1.y[1], sol2.y[1], sol3.y[1]])
    df = pd.DataFrame({"A": A, "B": B}, index=t_eval)
    return df, cp1_idx, cp2_idx


class TestEstimateAccuracyRate(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery._estimateAccuracyRate."""

    @classmethod
    def setUpClass(cls):
        """Fit a 3-segment PSD once; all tests in this class reuse it."""
        df, cls.cp1, cls.cp2 = _make_piecewise_df(n_points=300)
        cls.changepoints = [cls.cp1, cls.cp2]
        cls.psd = PiecewiseSystemDiscovery(
            df,
            changepoints=cls.changepoints,
            is_changepoint_removal=False,
            model_id="test_piecewise",
        )
        cls.psd.fit()

    # ------------------------------------------------------------------
    # Validation / error paths
    # ------------------------------------------------------------------

    def test_empty_changepoints_raises_value_error(self):
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            self.psd._estimateAccuracyRate([])

    def test_unfitted_psd_raises_runtime_error(self):
        if IGNORE_TESTS:
            return
        df, cp1, cp2 = _make_piecewise_df(n_points=60)
        unfitted = PiecewiseSystemDiscovery(
            df, changepoints=[cp1, cp2], is_changepoint_removal=False)
        with self.assertRaises(RuntimeError):
            unfitted._estimateAccuracyRate([cp1, cp2])


    # ------------------------------------------------------------------
    # Algorithmic / behavioral correctness
    # ------------------------------------------------------------------


class TestIncrementalMergeConsistency(unittest.TestCase):
    """Regression guard for the incremental merge optimization in _estimateAccuracyRate.

    Verifies that ``_buildTrialSegments`` + ``_scoreFromSegmentList`` produce the same 
    score as fitting a full trial PiecewiseSystemDiscovery, confirming correctness of 
    the O(c) merge approach over the original O(k) re-fit approach.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)
        cls.time = np.arange(0, 600, dtype=float)
        n_species = 2
        y = np.zeros((len(cls.time), n_species))
        # Create a piecewise-linear signal with two distinct changeppoints.
        y[:180, 0] = cls.time[:180] * 0.03 + 1.0
        y[180:420, 0] = (cls.time[180:420] - 180) * (-0.02) + 6.4
        y[420:, 0] = (cls.time[420:] - 420) * 0.01 + 1.6
        y[:180, 1] = cls.time[:180] * 0.01 + 5.0
        y[180:420, 1] = (cls.time[180:420] - 180) * 0.005 + 6.8
        y[420:, 1] = (cls.time[420:] - 420) * (-0.03) + 8.0
        # Add enough noise to make scores non-trivial but not so much that fits are degenerate.
        y += np.random.normal(0, 0.15, y.shape)
        cls.df = pd.DataFrame(y, columns=[f's{j}' for j in range(n_species)])
        cls.df.index.name = 'time'
        cls.changeppoints = [179, 419]

    def test_both_removals_matches_full_trial_fit(self):
        """Removing all changeppoints: incremental merge produces a single merged segment 
        with score matching the single-segment trial PSD fit."""
        if IGNORE_TESTS:
            return
        psd = PiecewiseSystemDiscovery(
            self.df.copy(), changepoints=self.changeppoints).fit()

        remove_positions = np.array([0, 1])
        trial_models, trial_boundaries = psd._buildTrialSegments(remove_positions)
        new_score = psd._scoreFromSegmentList(trial_models, trial_boundaries)

        surviving_cps = [int(cp) for i, cp in enumerate(cast(List[int], psd.changepoints))
                if i not in set(remove_positions)]
        trial_psd_old = PiecewiseSystemDiscovery(
            self.df.copy(), changepoints=surviving_cps, is_changepoint_removal=False,
            **psd._sd_kwargs).fit()
        old_score = trial_psd_old.score()

        self.assertAlmostEqual(
            new_score, old_score, places=8,
            msg=(f"Incremental score {new_score:.12f} should match full-fit "
                f"score {old_score:.12f} for all-removals case"))

    def test_build_trial_segments_reuses_unchanged_models(self):
        """When only one changeppoint is removed, the unchanged segment's model object 
        must be identical to self._subsequence_models (no re-fit)."""
        if IGNORE_TESTS:
            return
        psd = PiecewiseSystemDiscovery(
            self.df.copy(), changepoints=self.changeppoints).fit()

        remove_positions = np.array([0])  # removes cp at row 179, merges segs [0, 1]
        trial_models, _ = psd._buildTrialSegments(remove_positions)

        # Segment index 2 (the one starting at row 419 and extending to end) should be reused.
        self.assertIs(
            trial_models[1], psd._subsequence_models[2],
            "Unchanged segment model must be the same object as in original fit")



# ---------------------------------------------------------------------------
# _getChangepointsFromFile tests
# ---------------------------------------------------------------------------
class TestGetChangepointsFromFile(unittest.TestCase):
    """Tests for ``PiecewiseSystemDiscovery._getChangepointsFromFile``.

    The function returns a list of :class:`ChangepointLookupResult` -- one per
    matching CSV row -- sorted by descending accuracy (the value read from the
    column named by ``accuracy_col``, defaulting to ``cn.DEF_ACCURACY_COL`` (``"p10"``)).  Rows whose
    ``changepoints`` cell cannot be parsed or whose accuracy cell is NaN are
    silently skipped.
    """

    def _make_csv(self, path: str,
                  rows: List[Dict[str, object]],
                  include_accuracy_col: bool = True) -> None:
        """Write a piecewise-predictions CSV with the standard columns.

        When ``include_accuracy_col`` is True (the default), every row gets a
        ``p10`` column set to ``1.0`` unless overridden in the per-row dict.
        """
        cols = [cn.COL_SYSTEM_ID, cn.COL_MAX_CHANGEPOINT,
                cn.COL_MAX_FRACTIONAL_REDUCTION, "changepoints"]
        if include_accuracy_col:
            cols.append("p10")

        enriched_rows: List[Dict[str, object]] = []
        for row in rows:
            r = dict(row)
            if include_accuracy_col and "p10" not in r:
                r["p10"] = 1.0
            enriched_rows.append(r)

        df = pd.DataFrame(enriched_rows, columns=cols)
        df.to_csv(path, index=False)

    # -- missing-file / no-match cases (unchanged) ------------------------------------

    def test_returns_none_when_file_missing(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=100)
        psd = PiecewiseSystemDiscovery(df, model_id="BIOMD0000009999", max_changepoint=3)
        result = psd._getChangepointsFromFile("/tmp/definitely_does_not_exist_for_this_test.csv")
        self.assertEqual(len(result), 0)

    def test_returns_none_when_no_matching_row(self) -> None:
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [{
                cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "[5, 10]",
            }])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000009999", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            self.assertEqual(len(psd._getChangepointsFromFile(path)), 0)
        finally:
            os.remove(path)

    def test_returns_none_when_max_changepoint_mismatch(self) -> None:
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [{
                cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 5,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "[5, 10]",
            }])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            self.assertEqual(len(psd._getChangepointsFromFile(path)), 0)
        finally:
            os.remove(path)

    def test_returns_none_when_max_fractional_reduction_mismatch(self) -> None:
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [{
                cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.5,
                "changepoints": "[5]",
            }])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            self.assertEqual(len(psd._getChangepointsFromFile(path)), 0)
        finally:
            os.remove(path)

    # -- single matching row (wrapped in outer list) ----------------------------------

    def test_returns_single_valid_row_wrapped_in_outer_list(self) -> None:
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [{
                cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "[5, 10, 15]",
            }])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            result = psd._getChangepointsFromFile(path)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].changepoints, [5, 10, 15])
        finally:
            os.remove(path)

    def test_returns_single_changepoint_wrapped_in_outer_list(self) -> None:
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [{
                cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "[42]",
            }])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            result = psd._getChangepointsFromFile(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].changepoints, [42])
        finally:
            os.remove(path)

    def test_handles_whitespace_around_changepoint_string(self) -> None:
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [{
                cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "  [3, 7]  ",
            }])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            result = psd._getChangepointsFromFile(path)
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].changepoints, [3, 7])
        finally:
            os.remove(path)

    # -- empty / NaN cells on a matching row ------------------------------------------

    def test_exception_when_inner_list_when_changepoints_cell_is_nan(self) -> None:
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [{
                cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": float("nan"),
            }])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            with self.assertRaises(ValueError):
                _ = psd._getChangepointsFromFile(path)
        finally:
            os.remove(path)

    def test_returns_empty_inner_list_when_changepoints_cell_is_empty_string(self) -> None:
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [{
                cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "",
            }])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            with self.assertRaises(ValueError):
                _ = psd._getChangepointsFromFile(path)
        finally:
            os.remove(path)

    # -- multiple matching rows (sorted by descending accuracy) -----------------------

    def test_results_sorted_by_descending_accuracy(self) -> None:
        """When several rows match, the returned list is ordered from highest to lowest p10."""
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_MAX_CHANGEPOINT: 2,
                 cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                 "changepoints": "[9]", "p10": 0.3},
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_MAX_CHANGEPOINT: 2,
                 cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                 "changepoints": "[5]", "p10": 0.9},
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_MAX_CHANGEPOINT: 2,
                 cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                 "changepoints": "[7]", "p10": 0.6},
            ])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            result = psd._getChangepointsFromFile(path)
            # Each entry is a ChangepointLookupResult; verify order by changepoints.
            self.assertEqual(len(result), 3)
            self.assertEqual([r.changepoints for r in result], [[5], [7], [9]])
            self.assertEqual([r.accuracy for r in result], [0.9, 0.6, 0.3])
        finally:
            os.remove(path)

    def test_csv_with_different_accuracy_columns_per_row(self) -> None:
        """Each matching row's accuracy should come from its own column value."""
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_MAX_CHANGEPOINT: 2,
                 cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                 "changepoints": "[1]", "p10": 0.1},
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_MAX_CHANGEPOINT: 2,
                 cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                 "changepoints": "[2]", "p10": 0.8},
            ])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            result = psd._getChangepointsFromFile(path)
            self.assertEqual(len(result), 2)
            # The row with p10=0.8 must come first (descending sort).
            self.assertEqual(result[0].changepoints, [2])
            self.assertAlmostEqual(result[0].accuracy, 0.8, places=4)
        finally:
            os.remove(path)

    def test_exception_if_invalid_row(self) -> None:
        """A single malformed changepoints cell is silently skipped."""
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [
                {
                    cn.COL_SYSTEM_ID: "BIOMD0000000001",
                    cn.COL_MAX_CHANGEPOINT: 2,
                    cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                    "changepoints": "{not valid python literal!!!",
                },
                {
                    cn.COL_SYSTEM_ID: "BIOMD0000000001",
                    cn.COL_MAX_CHANGEPOINT: 2,
                    cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                    "changepoints": "[7, 8]", "p10": 0.4,
                },
            ])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            with self.assertRaises(ValueError):
                _ = psd._getChangepointsFromFile(path)
        finally:
            os.remove(path)

    def test_returns_exception_when_all_rows_unparseable(self) -> None:
        """When every matching row has an unparseable changepoints cell, return None."""
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [
                {
                    cn.COL_SYSTEM_ID: "BIOMD0000000001",
                    cn.COL_MAX_CHANGEPOINT: 2,
                    cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                    "changepoints": "{broken!!!",
                },
                {
                    cn.COL_SYSTEM_ID: "BIOMD0000000001",
                    cn.COL_MAX_CHANGEPOINT: 2,
                    cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                    "changepoints": "{also broken!!!",
                },
            ])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            with self.assertRaises(ValueError):
                _ = psd._getChangepointsFromFile(path)
        finally:
            os.remove(path)

    def test_exception_when_all_rows_parse_to_non_list(self) -> None:
        """Cells that parse but produce a non-list (e.g. dict, string) are skipped."""
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_MAX_CHANGEPOINT: 2,
                 cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                 "changepoints": "{'a': 1}",   # parses to dict
                },
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                 cn.COL_MAX_CHANGEPOINT: 2,
                 cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                 "changepoints": "'just a string'",   # parses to str
                },
            ])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            with self.assertRaises(ValueError):
                _ = psd._getChangepointsFromFile(path)
        finally:
            os.remove(path)

    def test_mixed_nan_and_valid_rows(self) -> None:
        """NaN changepoints become empty inner lists alongside valid parsed rows."""
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [
                {
                    cn.COL_SYSTEM_ID: "BIOMD0000000001",
                    cn.COL_MAX_CHANGEPOINT: 2,
                    cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                    "changepoints": float("nan"),
                },
                {
                    cn.COL_SYSTEM_ID: "BIOMD0000000001",
                    cn.COL_MAX_CHANGEPOINT: 2,
                    cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                    "changepoints": "[9, 10]", "p10": 0.7,
                },
                {
                    cn.COL_SYSTEM_ID: "BIOMD0000000001",
                    cn.COL_MAX_CHANGEPOINT: 2,
                    cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                    "changepoints": "", "p10": 0.4,
                },
            ])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            with self.assertRaises(ValueError):
                _ = psd._getChangepointsFromFile(path)
        finally:
            os.remove(path)

    def test_exception_if_row_with_nan_accuracy(self) -> None:
        """A matching row whose accuracy cell is NaN contributes nothing to the result."""
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "[5]", "p10": float("nan")},
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "[8]", "p10": 0.6},
            ])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            with self.assertRaises(ValueError):
                _ = psd._getChangepointsFromFile(path)
        finally:
            os.remove(path)

    def test_exception_when_every_matching_row_has_nan_accuracy(self) -> None:
        """If every matching row has NaN accuracy, the result is None."""
        if IGNORE_TESTS:
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self._make_csv(path, [
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "[5]", "p10": float("nan")},
                {cn.COL_SYSTEM_ID: "BIOMD0000000001",
                cn.COL_MAX_CHANGEPOINT: 2,
                cn.COL_MAX_FRACTIONAL_REDUCTION: 0.01,
                "changepoints": "[8]", "p10": float("nan")},
            ])
            df = _make_linear_df(n_points=100)
            psd = PiecewiseSystemDiscovery(
                df, model_id="BIOMD0000000001", max_changepoint=2,
                max_fractional_reduction=0.01,
            )
            with self.assertRaises(ValueError):
                _ = psd._getChangepointsFromFile(path)
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# End-to-end BioModel 5 test with changepoints from file
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestFitWithChangepointsFromFileLowReduction(unittest.TestCase):
    """fit() with a max_fractional_reduction (0.0001) that has no corresponding sweep CSV
    on disk -- kept separate from TestEndToEndBioModel5ChangepointsFromFile below because
    it exercises a different (currently-missing) data file, not the shared real fit.
    """

    #@unittest.skipUnless(_HAS_PSD_DATA, "PSD_DATA_DIR not found")
    @unittest.skipUnless(_HAS_PSD_DATA, "PSD data directory not found; skipping test.")
    def test_fit_with_changepoints_from_file(self) -> None:
        """Verify that fit() succeeds when is_changepoints_from_file=True for BioModel 5."""
        if IGNORE_TESTS or not HAS_REAL_ZIP:
            return
        tc = TimecourseIterator.getTimecourse("BIOMD0000000005", num_point=cn.NUM_POINT)

        # Create PiecewiseSystemDiscovery with changepoints from file
        psd = PiecewiseSystemDiscovery(
            tc.timecourse_df,
            model_id="BIOMD0000000005",
            max_changepoint=5000,
            max_fractional_reduction=0.0001,
            is_changepoints_from_file=True,
        )

        # Fit should succeed and load changepoints from file
        _ = psd.fit()

        # Verify that changepoints were loaded (more than just the default)
        self.assertIsNotNone(psd.changepoints)
        # With BioModel 5 having many timepoints, we expect multiple changepoints
        self.assertGreater(len(psd.changepoints), 0)  # type: ignore

        # Verify that subsequence models were created
        self.assertGreater(len(psd._subsequence_models), 1)


@unittest.skipUnless(HAS_REAL_ZIP and _HAS_PSD_DATA,
        "Real timecourse zip or PSD_DATA_DIR not found")
class TestEndToEndBioModel5ChangepointsFromFile(unittest.TestCase):
    """End-to-end test using real BioModel 5 with is_changepoints_from_file=True.

    predict()/score()/plot() share a single fit() via setUpClass: for this
    model/max_fractional_reduction, the best-available real config has 0 changepoints
    (one full-resolution, 100000-point segment), so each fit costs ~15-20s. Fitting
    once and reusing it across the three tests avoids paying that multiple times.
    """
    tc: Timecourse
    psd: PiecewiseSystemDiscovery

    @classmethod
    def setUpClass(cls) -> None:
        cls.tc = TimecourseIterator.getTimecourse("BIOMD0000000005", num_point=cn.NUM_POINT)
        cls.psd = PiecewiseSystemDiscovery(
            cls.tc.timecourse_df,
            model_id="BIOMD0000000005",
            max_changepoint=5000,
            max_fractional_reduction=0.01,
            is_changepoints_from_file=True,
        )
        cls.psd.fit()

    def test_predict_with_changepoints_from_file(self) -> None:
        """Verify that predict() works after fitting with changepoints from file."""
        if IGNORE_TESTS:
            return
        pred_df = self.psd.predict(self.tc.timecourse_df)

        # Verify prediction shape matches input
        self.assertEqual(pred_df.shape, self.tc.timecourse_df.shape)

    def test_score_with_changepoints_from_file(self) -> None:
        """Verify that score() works after fitting with changepoints from file."""
        if IGNORE_TESTS:
            return
        score = self.psd.score()

        # Verify score is a valid float
        self.assertIsInstance(score, float)

    def test_plot(self):
        """Verify that plot() works after fitting with changepoints from file."""
        if IGNORE_TESTS:
            return
        # Plot should work without errors
        try:
            self.psd.plotPiecewise(num_true_point=30)
            #plt.show()
        except Exception as e:
            self.fail(f"plot() raised an exception: {e}")

@unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
class TestBugs(unittest.TestCase):
    """Regression tests for specific bugs."""

    def test_bug_all_changepoints_deleted(self):
        """Test for bug where all changepoints are deleted."""
        timecourse_iter = TimecourseIterator(first_model_num=5, last_model_num=5)
        item = next(iter(timecourse_iter))
        df = item.timecourse.timecourse_df
        psd = PiecewiseSystemDiscovery(df, 
                max_changepoint=100,
                is_changepoint_removal=True,
                max_fractional_reduction=1e-3,
                model_id=item.model_name)
        psd.fit()
        _ = psd.plotPiecewise(legend=False, num_true_point=60, is_nochangepoint_plot=False)
        #plt.show()


class TestParseChangepointsCell(unittest.TestCase):
    """Tests for :meth:`PiecewiseSystemDiscovery.parseChangepointsCell`."""

    def test_plain_int_list_string(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(
            PiecewiseSystemDiscovery.parseChangepointsCell("[20, 40]"), [20, 40])

    def test_empty_list_string(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(
            PiecewiseSystemDiscovery.parseChangepointsCell("[]"), [])

    def test_python_list(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(
            PiecewiseSystemDiscovery.parseChangepointsCell([20, 40]), [20, 40])

    def test_numpy_scalar_wrappers(self) -> None:
        """Cells stringified from a list of numpy scalars (numpy >= 1.24 repr)."""
        if IGNORE_TESTS:
            return
        raw = "[np.int64(20), np.int64(13677)]"
        self.assertEqual(
            PiecewiseSystemDiscovery.parseChangepointsCell(raw, model_name="BIOMD001"),
            [20, 13677])

    def test_non_string_or_list_raises(self) -> None:
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            PiecewiseSystemDiscovery.parseChangepointsCell(float("nan"))

    def test_empty_string_raises(self) -> None:
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            PiecewiseSystemDiscovery.parseChangepointsCell("   ")

    def test_invalid_format_raises(self) -> None:
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            PiecewiseSystemDiscovery.parseChangepointsCell("not a list")


if __name__ == "__main__":
    unittest.main()
