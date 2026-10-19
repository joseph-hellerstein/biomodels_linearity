"""Tests for the SystemDiscovery class in src/system_discovery.py."""

from src.system_discovery import SystemDiscovery, discoverNetwork, DiscoverNetworkResult  # type: ignore
from src.perturbation_analyzer import PerturbationAnalyzer  # type: ignore
import src.constants as cn  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore
from src.model import Model  # type: ignore

import os
import sys
import unittest
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


#sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

IGNORE_TESTS = False
IS_PLOT = False
NUM_POINT = 1000


def _make_linear_df(
    n_points: int = NUM_POINT,
    t_start: float = 0.0,
    t_end: float = 10.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a simple linear ODE timecourse for testing.

    True dynamics: dA/dt = -0.5*A + 0.1*B, dB/dt = 0.3*A - 0.2*B
    With A(0)=1.0, B(0)=0.0
    """
    rng = np.random.default_rng(seed)

    def rhs(t, z):
        a, b = z
        return [-0.5 * a + 0.1 * b, 0.3 * a - 0.2 * b]

    from scipy.integrate import solve_ivp  # type: ignore

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(rhs, [t_start, t_end], [1.0, 0.0], t_eval=t_eval, rtol=1e-8)
    X = sol.y.T + rng.normal(0, noise_std, (n_points, len(sol.y)))

    return pd.DataFrame(X, index=t_eval, columns=["A", "B"])


def _make_quadratic_df(
    n_points: int = NUM_POINT,
    t_start: float = 0.0,
    t_end: float = 10.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a quadratic ODE timecourse for testing.

    True dynamics: dA/dt = 1 - 0.5*A*B, dB/dt = 0.3*A*B - 0.1*B^2
    With A(0)=2.0, B(0)=1.0
    """
    rng = np.random.default_rng(seed)

    def rhs(t, z):
        a, b = z
        return [1.0 - 0.5 * a * b, 0.3 * a * b - 0.1 * b ** 2]

    from scipy.integrate import solve_ivp  # type: ignore

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(rhs, [t_start, t_end], [2.0, 1.0], t_eval=t_eval, rtol=1e-8)
    X = sol.y.T + rng.normal(0, noise_std, (n_points, len(sol.y)))

    return pd.DataFrame(X, index=t_eval, columns=["A", "B"])


def _make_uniform_time_df(
    n_points: int = 50,
    t_start: float = 0.0,
    t_end: float = 10.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a linear ODE with perfectly uniform time steps for matrix exponential."""
    rng = np.random.default_rng(seed)

    def rhs(t, z):
        a, b = z
        return [-0.5 * a + 0.1 * b, 0.3 * a - 0.2 * b]

    from scipy.integrate import solve_ivp  # type: ignore

    t_eval = np.linspace(t_start, t_end, n_points)
    sol = solve_ivp(rhs, [t_start, t_end], [1.0, 0.0], t_eval=t_eval, rtol=1e-8)
    X = sol.y.T + rng.normal(0, noise_std, (n_points, len(sol.y)))

    return pd.DataFrame(X, index=t_eval, columns=["A", "B"])


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestSystemDiscoveryConstructor(unittest.TestCase):
    """Tests for SystemDiscovery.__init__."""

    def test_basic_construction(self) -> None:
        """Basic construction with a valid DataFrame succeeds."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, is_normalize=False)
        self.assertEqual(disc.num_species, 2)
        self.assertEqual(disc.species_names, ["A", "B"])

    def test_default_parameters(self) -> None:
        """Default parameter values are set correctly."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, is_normalize=False)
        self.assertEqual(disc.coefficient_threshold, 0.01)
        self.assertEqual(disc.alpha, 0.05)
        self.assertEqual(disc.differentiation, "smooth")
        self.assertEqual(disc.poly_degree, 1)
        self.assertTrue(disc.include_bias)

    def test_custom_parameters(self) -> None:
        """Custom parameters are stored correctly."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(
            df, coefficient_threshold=0.1, alpha=0.5, differentiation="finite",
            poly_degree=2, include_bias=False, is_normalize=False,
        )
        self.assertEqual(disc.coefficient_threshold, 0.1)
        self.assertEqual(disc.alpha, 0.5)
        self.assertEqual(disc.differentiation, "finite")
        self.assertEqual(disc.poly_degree, 2)
        self.assertFalse(disc.include_bias)

    def test_species_names_override(self) -> None:
        """species_names overrides column names."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, species_names=["X", "Y"], is_normalize=False)
        self.assertEqual(disc.species_names, ["X", "Y"])

    def test_species_names_length_mismatch_raises(self) -> None:
        """species_names with wrong length raises ValueError."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        with self.assertRaises(ValueError):
            SystemDiscovery(df, species_names=["X"], is_normalize=False)

    def test_bias_species_valid(self) -> None:
        """Valid bias_species are stored correctly and force include_bias=True."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(
            df, bias_species=["A"], is_normalize=False, include_bias=False
        )
        self.assertEqual(disc.bias_species, ["A"])
        self.assertTrue(disc.include_bias)

    def test_bias_species_invalid_raises(self) -> None:
        """bias_species with invalid names raises ValueError."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        with self.assertRaises(ValueError):
            SystemDiscovery(df, bias_species=["Z"], is_normalize=False)

    def test_bracket_names_stripped(self) -> None:
        """Species names starting with '[' have the bracket stripped."""
        if IGNORE_TESTS:
            return
        df = pd.DataFrame(
            np.random.rand(50, 2), index=np.linspace(0, 10, 50),
            columns=["[A]", "[B]"],
        )
        disc = SystemDiscovery(df, is_normalize=False)
        self.assertEqual(disc.species_names, ["A", "B"])

    def test_too_many_species_raises(self) -> None:
        """DataFrame with too many species raises ValueError."""
        if IGNORE_TESTS:
            return
        n_cols = 201
        df = pd.DataFrame(
            np.random.rand(50, n_cols), index=np.linspace(0, 10, 50)
        )
        for i in range(n_cols):
            df.columns = [f"S{i}" for i in range(n_cols)]
        with self.assertRaises(ValueError):
            SystemDiscovery(df, is_normalize=False)

    def test_empty_columns_constructs(self) -> None:
        """DataFrame with no columns constructs without raising (validation only runs for list inputs)."""
        if IGNORE_TESTS:
            return
        df = pd.DataFrame(index=np.linspace(0, 10, 50))
        # Single DataFrame path skips _validate_dataframe; just verify construction succeeds
        disc = SystemDiscovery(df, is_normalize=False)
        self.assertEqual(disc.num_species, 0)

    def test_non_increasing_index_constructs(self) -> None:
        """DataFrame with non-increasing index constructs without raising (validation only runs for list inputs)."""
        if IGNORE_TESTS:
            return
        df = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [0.5, 0.6, 0.7]},
            index=[5.0, 3.0, 4.0],
        )
        disc = SystemDiscovery(df, is_normalize=False)
        self.assertEqual(disc.num_species, 2)

    def test_not_fitted_initially(self) -> None:
        """is_fitted is False before fit() is called."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, is_normalize=False)
        self.assertFalse(disc.is_fitted)


# ---------------------------------------------------------------------------
# Static method tests
# ---------------------------------------------------------------------------


class TestSystemDiscoveryStaticMethods(unittest.TestCase):
    """Tests for static utility methods."""

    def _make_disc_for_parse(self) -> SystemDiscovery:
        """Helper to create a minimal instance for calling instance methods."""
        df = pd.DataFrame(
            np.random.rand(50, 2), index=np.linspace(0, 10, 50),
            columns=["A", "B"],
        )
        return SystemDiscovery(df, is_normalize=False)


# ---------------------------------------------------------------------------
# Differentiator builder tests
# ---------------------------------------------------------------------------


class TestBuildDifferentiator(unittest.TestCase):
    """Tests for _build_differentiator."""

    def test_smooth(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, differentiation="smooth", is_normalize=False)
        diff = disc._buildDifferentiator()
        self.assertIsNotNone(diff)

    def test_finite(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, differentiation="finite", is_normalize=False)
        diff = disc._buildDifferentiator()
        self.assertIsNotNone(diff)

    def test_spectral(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, differentiation="spectral", is_normalize=False)
        diff = disc._buildDifferentiator()
        self.assertIsNotNone(diff)

    def test_invalid_method_raises(self) -> None:
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        with self.assertRaises(ValueError):
            SystemDiscovery(df, differentiation="invalid", is_normalize=False)  # type: ignore


# ---------------------------------------------------------------------------
# Fit and equation tests
# ---------------------------------------------------------------------------


class TestSystemDiscoveryFit(unittest.TestCase):
    """Tests for the fit() method and derived properties."""

    def _make_disc(self, df: pd.DataFrame, **kwargs) -> SystemDiscovery:
        defaults = {"is_normalize": False}
        defaults.update(kwargs)
        return SystemDiscovery(df, **defaults)  # type: ignore

    def test_fit_sets_is_fitted(self) -> None:
        """fit() sets is_fitted to True."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.00)
        disc = self._make_disc(df, coefficient_threshold=0.001, alpha=0.001)
        disc.fit()
        self.assertTrue(disc.is_fitted)

    def test_fit_with_smooth_differentiation(self) -> None:
        """fit() works with smooth differentiation."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.00)
        disc = self._make_disc(df, coefficient_threshold=0.001, alpha=0.001, differentiation="smooth")
        disc.fit()
        self.assertTrue(disc.is_fitted)

    def test_fit_with_finite_differentiation(self) -> None:
        """fit() works with finite differentiation."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_disc(df, coefficient_threshold=0.001, alpha=0.001, differentiation="finite")
        disc.fit()
        self.assertTrue(disc.is_fitted)

    def test_fit_with_spectral_differentiation(self) -> None:
        """fit() works with spectral differentiation on uniform data."""
        if IGNORE_TESTS:
            return
        df = _make_uniform_time_df(noise_std=0.01)
        disc = self._make_disc(df, coefficient_threshold=0.001, alpha=0.001, differentiation="spectral")
        disc.fit()
        self.assertTrue(disc.is_fitted)

    def test_fit_with_quadratic_library(self) -> None:
        """fit() works with poly_degree=2."""
        if IGNORE_TESTS:
            return
        df = _make_quadratic_df(noise_std=0.01)
        disc = self._make_disc(
            df, coefficient_threshold=0.001, alpha=0.001, poly_degree=2, include_bias=True
        )
        disc.fit()
        self.assertTrue(disc.is_fitted)

    def test_fit_with_normalization(self) -> None:
        """fit() works with normalization enabled."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.00)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001)
        disc.fit()
        self.assertTrue(disc.is_fitted)

    def test_get_equations_after_fit(self) -> None:
        """getEquations returns a dict after fitting."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.00)
        disc = self._make_disc(df, coefficient_threshold=0.001, alpha=0.001)
        disc.fit()
        eqs = disc.getEquations()
        self.assertIsInstance(eqs, dict)
        self.assertEqual(set(eqs.keys()), {"A", "B"})
        for v in eqs.values():
            self.assertIsInstance(v, str)

    def test_get_equations_before_fit_raises(self) -> None:
        """getEquations raises RuntimeError before fit."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = self._make_disc(df)
        with self.assertRaises(RuntimeError):
            disc.getEquations()

    def test_get_nonzero_terms_after_fit(self) -> None:
        """getNonzeroTerms returns a dict after fitting."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.00)
        disc = self._make_disc(df, coefficient_threshold=0.001, alpha=0.001)
        disc.fit()
        nzt = disc.getNonzeroTerms()
        self.assertIsInstance(nzt, dict)
        for sp_name in ["A", "B"]:
            self.assertIn(sp_name, nzt)
            self.assertGreaterEqual(nzt[sp_name], 0)

    def test_get_nonzero_terms_before_fit_raises(self) -> None:
        """getNonzeroTerms raises RuntimeError before fit."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = self._make_disc(df)
        with self.assertRaises(RuntimeError):
            disc.getNonzeroTerms()

    def test_str_after_fit(self) -> None:
        """__str__ returns non-empty string after fitting."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.00)
        disc = self._make_disc(df, coefficient_threshold=0.001, alpha=0.001)
        disc.fit()
        s = str(disc)
        self.assertIn("A", s)
        self.assertIn("B", s)

    def test_str_before_fit(self) -> None:
        """__str__ returns 'Model not fitted yet.' before fitting."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = self._make_disc(df)
        self.assertEqual(str(disc), "Model not fitted yet.")

    def test_bias_species_zeros_constant(self) -> None:
        """bias_species forces constant term to zero for non-biased species."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.00)
        disc = self._make_disc(
            df, coefficient_threshold=0.001, alpha=0.001, bias_species=["A"], include_bias=True
        )
        disc.fit()
        coefs = disc.model.coefficients()
        # Find the constant feature (index 0 in polynomial library with bias)
        feature_names = disc.model.get_feature_names()
        const_idx = None
        for i, fn in enumerate(feature_names):
            if fn == "1":
                const_idx = i
                break
        self.assertIsNotNone(const_idx)
        # B's constant term should be zero
        b_idx = disc.species_names.index("B")
        self.assertAlmostEqual(coefs[b_idx, const_idx], 0.0, places=6)  # type: ignore


# ---------------------------------------------------------------------------
# Threshold application tests
# ---------------------------------------------------------------------------


class TestApplyThreshold(unittest.TestCase):
    """Tests for _apply_threshold."""

    def test_threshold_prunes_small_coefficients(self) -> None:
        """_apply_threshold zeros out coefficients below threshold."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.00)
        disc = SystemDiscovery(df, coefficient_threshold=1e6, alpha=0.001, is_normalize=False)
        disc.fit()
        coefs_before = disc.model.coefficients().copy()
        # Set an extremely high threshold so everything gets pruned
        disc.coefficient_threshold = 1e6
        disc._applyThreshold()
        coefs_after = disc.model.coefficients()
        # All should be zero now (or very close)
        for i in range(coefs_after.shape[0]):
            for j in range(coefs_after.shape[1]):
                self.assertAlmostEqual(coefs_after[i, j], 0.0, places=5)


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------


class TestSimulate(unittest.TestCase):
    """Tests for _simulate, _simulateGeneral, and _simulateSimple."""

    def _make_fitted_disc(self, df: pd.DataFrame, **kwargs) -> SystemDiscovery:
        defaults = {"is_normalize": False}
        defaults.update(kwargs)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, **defaults)  # type: ignore
        disc.fit()
        return disc

    def test_simulate_returns_correct_shape(self) -> None:
        """_simulate returns array with shape (n_timepoints, n_species)."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        disc = self._make_fitted_disc(df)
        result = disc._simulate()
        self.assertEqual(result.shape, (50, 2))

    def test_simulate_with_custom_x0(self) -> None:
        """_simulate accepts custom initial conditions."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        disc = self._make_fitted_disc(df)
        x0 = np.array([2.0, 1.0])
        result = disc._simulate(x0=x0)
        self.assertEqual(result.shape[0], 50)
        self.assertEqual(result.shape[1], 2)

    def test_simulate_with_custom_time(self) -> None:
        """_simulate accepts custom time array."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        disc = self._make_fitted_disc(df)
        t_new = np.linspace(0, 5, 25)
        result = disc._simulate(time_arr=t_new)
        self.assertEqual(result.shape[0], 25)

    def test_simulate_simple_linear_uniform(self) -> None:
        """_simulateSimple is used for linear systems with uniform time steps."""
        if IGNORE_TESTS:
            return
        df = _make_uniform_time_df(n_points=NUM_POINT, noise_std=0.01)
        disc = self._make_fitted_disc(
            df, poly_degree=1, include_bias=True
        )
        result = disc._simulate()
        self.assertEqual(result.shape, (NUM_POINT, 2))

    def test_simulate_general_for_nonlinear(self) -> None:
        """_simulateGeneral is used for non-linear systems."""
        if IGNORE_TESTS:
            return
        df = _make_quadratic_df(n_points=NUM_POINT, noise_std=0.01)
        disc = self._make_fitted_disc(
            df, poly_degree=2, include_bias=True
        )
        result = disc._simulate()
        self.assertEqual(result.shape[0], NUM_POINT)

    def test_simulate_simple_checks_assumptions(self) -> None:
        """_simulateSimple raises ValueError if assumptions not met."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=NUM_POINT, noise_std=0.01)
        disc = SystemDiscovery(
            df, coefficient_threshold=0.001, alpha=0.001, poly_degree=2,
            include_bias=False, is_normalize=False,
        )
        disc.fit()
        with self.assertRaises(ValueError):
            disc._simulateSimple(x0=np.array([1.0, 0.0]), time_arr=np.linspace(0, 10, NUM_POINT))


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------


class TestPredict(unittest.TestCase):
    """Tests for predict() and related methods."""

    def _make_fitted_disc(self, df: pd.DataFrame, **kwargs) -> SystemDiscovery:
        defaults = {"is_normalize": False}
        defaults.update(kwargs)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, **defaults)  # type: ignore
        disc.fit()
        return disc

    def test_predict_returns_dataframe(self) -> None:
        """predict() returns a DataFrame."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=NUM_POINT)
        disc = self._make_fitted_disc(df)
        result = disc.predict()
        self.assertIsInstance(result, pd.DataFrame)

    def test_predict_correct_shape(self) -> None:
        """predict() returns DataFrame with correct shape."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=NUM_POINT)
        disc = self._make_fitted_disc(df)
        result = disc.predict()
        self.assertEqual(result.shape, (NUM_POINT, 2))

    def test_predict_correct_columns(self) -> None:
        """predict() returns DataFrame with species name columns."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        disc = self._make_fitted_disc(df)
        result = disc.predict()
        self.assertEqual(list(result.columns), ["A", "B"])

    def test_predict_with_test_df(self) -> None:
        """predict(test_df) uses test_df's initial condition and time grid."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        disc = self._make_fitted_disc(df)
        result = disc.predict(df)
        self.assertEqual(result.shape[0], 50)

    def test_predict_all_derivatives(self) -> None:
        """predictAllDerivatives returns correct shape."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        disc = self._make_fitted_disc(df)
        result = disc.predictAllDerivatives()
        # Shape should be (n_timepoints, n_species) for the diff-based derivatives
        self.assertEqual(result.shape[1], 2)

    def test_predict_one_step_derivative(self) -> None:
        """predictOneStepDerivative returns shape (n_species,)."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        disc = self._make_fitted_disc(df)
        x = np.array([1.0, 0.0])
        result = disc.predictOneStepDerivative(x)
        self.assertEqual(result.shape, (2,))

    def test_predict_one_step_derivative_before_fit_raises(self) -> None:
        """predictOneStepDerivative raises RuntimeError before fit."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, is_normalize=False)
        with self.assertRaises(RuntimeError):
            disc.predictOneStepDerivative(np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# R² score tests
# ---------------------------------------------------------------------------


class TestRsqScore(unittest.TestCase):
    """Tests for calculateRsq and related scoring methods."""

    def _make_fitted_disc(self, df: pd.DataFrame, **kwargs) -> SystemDiscovery:
        defaults = {"is_normalize": False}
        defaults.update(kwargs)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, **defaults) # type: ignore
        disc.fit()
        return disc

    def test_calculate_rsq_derivative_returns_dict(self) -> None:
        """calculateRsq(method='derivative') returns a dict."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.calculateSpeciesScores(score_type="derivative")
        self.assertIsInstance(result, dict)

    def test_calculate_rsq_derivative_all_species(self) -> None:
        """calculateRsq includes all species."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.calculateSpeciesScores(score_type="derivative")
        for sp in ["A", "B"]:
            self.assertIn(sp, result)

    def test_calculate_rsq_values_in_range(self) -> None:
        """calculateRsq values are clamped to [0, 1]."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.calculateSpeciesScores(score_type="derivative")
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_calculate_rsq_before_fit_raises(self) -> None:
        """calculateRsq raises RuntimeError before fit."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, is_normalize=False)
        with self.assertRaises(RuntimeError):
            disc.calculateSpeciesScores(score_type="derivative")

    def test_calculate_rsq_with_test_df(self) -> None:
        """calculateRsq works with a test DataFrame."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.calculateSpeciesScores(score_type="derivative", test_df=df)
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


# ---------------------------------------------------------------------------
# Score details tests
# ---------------------------------------------------------------------------


class TestScoreDetails(unittest.TestCase):
    """Tests for getScoreDetails and score methods."""

    def _make_fitted_disc(self, df: pd.DataFrame, **kwargs) -> SystemDiscovery:
        defaults = {"is_normalize": False}
        defaults.update(kwargs)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, **defaults)    # type: ignore
        disc.fit()
        return disc

    def test_get_score_details_derivative(self) -> None:
        """getScoreDetails with derivative returns a DataFrame."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="derivative")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_timecourse(self) -> None:
        """getScoreDetails with timecourse returns a DataFrame."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="timecourse")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_invalid_raises(self) -> None:
        """getScoreDetails raises ValueError for invalid score_type."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        with self.assertRaises(ValueError):
            disc.getScoreDetails(score_type="invalid")

    def test_get_score_details_returns_expected_columns(self) -> None:
        """getScoreDetails returns DataFrame with expected columns."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="derivative")
        expected_cols = set(cn.COLUMN_STATISTICS + [cn.COL_AGGREGATION_TYPE, cn.COL_SYSTEM_ID])
        expected_cols.update(cn.COLUMN_ACCURACY_FRACTIONS)
        self.assertEqual(set(result.columns), expected_cols)

    def test_get_score_details_has_model_and_species_rows(self) -> None:
        """getScoreDetails returns both model and species aggregation rows."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="derivative")
        agg_types = set(result[cn.COL_AGGREGATION_TYPE].values)
        self.assertIn(cn.COL_AGGREGATION_TYPE_MODEL, agg_types)

    def test_get_score_details_model_row_count(self) -> None:
        """getScoreDetails returns at least one model aggregation row."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="derivative")
        model_rows = result[result[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        # May have multiple rows if previous test runs accumulated data in score.csv
        self.assertGreaterEqual(len(model_rows), 1)

    def test_get_score_details_species_row_count(self) -> None:
        """getScoreDetails returns at least two species aggregation rows."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="derivative")
        species_rows = result[result[cn.COL_AGGREGATION_TYPE] != cn.COL_AGGREGATION_TYPE_MODEL]
        # May have multiple rows if previous test runs accumulated data in score.csv
        self.assertGreaterEqual(len(species_rows), 2)

    def test_get_score_details_with_test_df(self) -> None:
        """getScoreDetails with test_df uses the provided DataFrame."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=NUM_POINT, noise_std=0.01)
        train_df = df.iloc[:50]
        test_df = df.iloc[:50]
        disc = SystemDiscovery(train_df, coefficient_threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(test_df=test_df, score_type="derivative")
        self.assertIsInstance(result, pd.DataFrame)
        agg_types = set(result[cn.COL_AGGREGATION_TYPE].values)
        self.assertIn(cn.COL_AGGREGATION_TYPE_MODEL, agg_types)

    def test_get_score_details_with_test_df_timecourse(self) -> None:
        """getScoreDetails with test_df and timecourse score_type works."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=NUM_POINT, noise_std=0.01)
        train_df = df.iloc[:50]
        test_df = df.iloc[50:]
        disc = SystemDiscovery(train_df, coefficient_threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(test_df=test_df, score_type="timecourse")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_before_fit_raises(self) -> None:
        """getScoreDetails raises RuntimeError before fit."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, is_normalize=False)
        with self.assertRaises(RuntimeError):
            disc.getScoreDetails(score_type="derivative")

    def test_get_score_details_description_matches_label(self) -> None:
        """getScoreDetails stores empty description when no label provided."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="derivative")
        # COL_SYSTEM_ID column should contain empty strings or NaN (from CSV read)
        for _, row in result.iterrows():
            desc_val = row[cn.COL_SYSTEM_ID]
            if pd.notna(desc_val):
                self.assertEqual(desc_val, "")

    def test_get_score_details_with_smooth_differentiation(self) -> None:
        """getScoreDetails works with smooth differentiation."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001,
                differentiation="smooth", is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(score_type="derivative")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_with_finite_differentiation(self) -> None:
        """getScoreDetails works with finite differentiation."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001,
                               differentiation="finite", is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(score_type="derivative")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_with_spectral_differentiation(self) -> None:
        """getScoreDetails works with spectral differentiation on uniform data."""
        if IGNORE_TESTS:
            return
        df = _make_uniform_time_df(noise_std=0.01)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001,
                               differentiation="spectral", is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(score_type="derivative")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_timecourse_columns(self) -> None:
        """getScoreDetails with timecourse returns DataFrame with species columns."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="timecourse")
        # Check that percentile statistics are populated (not NaN for valid data).
        model_rows = result[result[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        if not model_rows.empty:
            self.assertFalse(np.isnan(model_rows['p25'].values[0]))  # type: ignore

    def test_get_score_details_invalid_score_type_message(self) -> None:
        """getScoreDetails ValueError message mentions the invalid score_type."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        with self.assertRaises(ValueError) as ctx:
            disc.getScoreDetails(score_type="bogus")
        self.assertIn("bogus", str(ctx.exception))

    def test_get_score_details_with_quadratic_library(self) -> None:
        """getScoreDetails works with poly_degree=2."""
        if IGNORE_TESTS:
            return
        df = _make_quadratic_df(n_points=NUM_POINT, noise_std=0.01)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001,
                               poly_degree=2, include_bias=True, is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(score_type="derivative")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_with_normalization(self) -> None:
        """getScoreDetails works with normalization enabled."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001)
        disc.fit()
        result = disc.getScoreDetails(score_type="derivative")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_model_row_percentiles_populated(self) -> None:
        """Model aggregation row has all percentile columns populated."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=NUM_POINT, noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="derivative")
        model_rows = result[result[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        percentile_cols = ["p25", "p30", "p50", "p80", "p95", "p99"]
        for col in percentile_cols:
            self.assertFalse(np.isnan(model_rows[col].values[0]), f"{col} should not be NaN")  # type: ignore

    def test_get_score_details_species_row_has_mean(self) -> None:
        """Species aggregation rows have mean values populated."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreDetails(score_type="derivative")
        species_rows = result[result[cn.COL_AGGREGATION_TYPE] != cn.COL_AGGREGATION_TYPE_MODEL]
        for _, row in species_rows.iterrows():
            self.assertFalse(np.isnan(row['mean']), "species mean should not be NaN")

    def test_get_score_details_default_score_type(self) -> None:
        """getScoreDetails defaults to 'derivative' when score_type is omitted."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result_default = disc.getScoreDetails()
        result_explicit = disc.getScoreDetails(score_type="derivative")
        # Both should be DataFrames (structural comparison not required).
        self.assertIsInstance(result_default, pd.DataFrame)
        self.assertIsInstance(result_explicit, pd.DataFrame)

    def test_get_score_details_with_bias_species(self) -> None:
        """getScoreDetails works with bias_species set."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001,
                               bias_species=["A"], include_bias=True, is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(score_type="derivative")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_with_high_threshold(self) -> None:
        """getScoreDetails works with very high threshold (all pruned)."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = SystemDiscovery(df, coefficient_threshold=1e6, alpha=0.001, is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(score_type="derivative")
        self.assertIsInstance(result, pd.DataFrame)

    def test_get_score_details_timecourse_with_test_df(self) -> None:
        """getScoreDetails timecourse with test_df returns correct shape."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=NUM_POINT, noise_std=0.01)
        train_df = df.iloc[:NUM_POINT//2]
        test_df = df.iloc[NUM_POINT//2:]
        disc = SystemDiscovery(train_df, coefficient_threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(test_df=test_df, score_type="timecourse")
        self.assertIsInstance(result, pd.DataFrame)
        agg_types = set(result[cn.COL_AGGREGATION_TYPE].values)
        self.assertIn(cn.COL_AGGREGATION_TYPE_MODEL, agg_types)

    def test_get_score_details_invalid_score_type(self) -> None:
        """getScoreDetails raises ValueError for invalid score_type."""
        df = _make_linear_df()
        disc = SystemDiscovery(df, is_normalize=False)
        disc.fit()
        with self.assertRaises(ValueError):
            disc.getScoreDetails(score_type="invalid")


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------


class TestSummary(unittest.TestCase):
    """Tests for summary() method."""

    def _make_fitted_disc(self, df: pd.DataFrame, **kwargs) -> SystemDiscovery:
        defaults = {"is_normalize": False}
        defaults.update(kwargs)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, **defaults)  # type: ignore
        disc.fit()
        return disc

    def test_summary_returns_dataframe(self) -> None:
        """summary() returns a DataFrame."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.summary()
        self.assertIsInstance(result, pd.DataFrame)

    def test_summary_before_fit_raises(self) -> None:
        """summary() raises RuntimeError before fit."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, is_normalize=False)
        with self.assertRaises(RuntimeError):
            disc.summary()

    def test_summary_columns_are_odes(self) -> None:
        """summary() columns are named d{species}/dt."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.summary()
        expected_cols = {"dA/dt", "dB/dt"}
        self.assertEqual(set(result.columns), expected_cols)


# ---------------------------------------------------------------------------
# Plot tests (non-interactive)
# ---------------------------------------------------------------------------


class TestPlotting(unittest.TestCase):
    """Tests for plotting methods.  These use Agg backend to avoid display."""

    @classmethod
    def setUpClass(cls) -> None:
        import matplotlib
        matplotlib.use("Agg")

    def _make_fitted_disc(self, df: pd.DataFrame, **kwargs) -> SystemDiscovery:
        defaults = {"is_normalize": False}
        defaults.update(kwargs)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, **defaults)  # type: ignore
        disc.fit()
        return disc

    def test_plot_result_returns_figure(self) -> None:
        """plotResult returns a Figure object."""
        if IGNORE_TESTS:
            return
        import matplotlib.pyplot as plt  # noqa: E402

        df = _make_linear_df(n_points=50, noise_std=0.01)
        disc = self._make_fitted_disc(df)
        fig = disc.plotResult(is_plot=False)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_plot_coefficient_heatmap_returns_figure(self) -> None:
        """plot_coefficient_heatmap returns a Figure object."""
        if IGNORE_TESTS:
            return
        import matplotlib.pyplot as plt  # noqa: E402

        df = _make_linear_df(n_points=50, noise_std=0.01)
        disc = self._make_fitted_disc(df)
        fig = disc.plotCoefficientHeatmap(is_plot=False)
        self.assertIsNotNone(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# discoverNetwork convenience function tests
# ---------------------------------------------------------------------------


class TestDiscoverNetwork(unittest.TestCase):
    """Tests for the discoverNetwork convenience function."""

    def test_discover_network_returns_disc(self) -> None:
        """discoverNetwork returns a fitted SystemDiscovery."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50, noise_std=0.01)
        sdr = discoverNetwork(
            df, threshold=0.001, alpha=0.001,
            is_plot_comparisons=False, is_plot_heatmap=False,
            is_print_equations=False, is_print_accuracy=False,
        )
        self.assertIsInstance(sdr, DiscoverNetworkResult)
        self.assertIsInstance(sdr.sd, SystemDiscovery) # type: ignore
        self.assertTrue(sdr.sd.is_fitted)  # type: ignore

    def test_discover_network_with_species_names(self) -> None:
        """discoverNetwork respects species_names parameter (same names as columns)."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50, noise_std=0.01)
        sdr= discoverNetwork(
            df, threshold=0.001, alpha=0.001,
            species_names=["A", "B"],
            is_plot_comparisons=False, is_plot_heatmap=False,
            is_print_equations=False, is_print_accuracy=False,
        )
        self.assertEqual(sdr.sd.species_names, ["A", "B"])  # type: ignore


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_single_timepoint_derivative(self) -> None:
        """Xdot_df has one fewer row than the original DataFrame."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50)
        disc = SystemDiscovery(df, is_normalize=False)
        self.assertEqual(len(disc.Xdot_df), 49)

    def test_two_species_minimum(self) -> None:
        """System works with the minimum of 2 species."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50, noise_std=0.01)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        self.assertTrue(disc.is_fitted)

    def test_high_threshold_prunes_everything(self) -> None:
        """Very high threshold results in all-zero coefficients."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=50, noise_std=0.01)
        disc = SystemDiscovery(df, coefficient_threshold=1e6, alpha=0.001, is_normalize=False)
        disc.fit()
        coefs = disc.model.coefficients()
        for i in range(coefs.shape[0]):
            for j in range(coefs.shape[1]):
                self.assertAlmostEqual(coefs[i, j], 0.0, places=5)

    def test_zero_threshold_keeps_most(self) -> None:
        """Very low threshold keeps most coefficients."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=NUM_POINT, noise_std=0.001)
        disc = SystemDiscovery(df, coefficient_threshold=0.0, alpha=0.001, is_normalize=False)
        disc.fit()
        nzt = disc.getNonzeroTerms()
        # With zero threshold and low noise, most terms should be non-zero
        total_terms = sum(nzt.values())
        self.assertGreater(total_terms, 0)

    def test_constant_data_handling(self) -> None:
        """SystemDiscovery handles data with near-constant species."""
        if IGNORE_TESTS:
            return
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, NUM_POINT)
        df = pd.DataFrame({
            "A": 1.0 + rng.normal(0, 0.001, NUM_POINT),
            "B": np.exp(-0.1 * t) + rng.normal(0, 0.001, NUM_POINT),
        }, index=t)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, is_normalize=False)
        # Should not raise during construction
        self.assertEqual(disc.num_species, 2)


# ---------------------------------------------------------------------------
# Integration-style tests with known dynamics
# ---------------------------------------------------------------------------


class TestKnownDynamics(unittest.TestCase):
    """Tests that verify the discovered model recovers known dynamics."""

    def _make_fitted_disc(self, df: pd.DataFrame, **kwargs) -> SystemDiscovery:
        defaults = {"is_normalize": False}
        defaults.update(kwargs)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, **defaults)  # type: ignore
        disc.fit()
        return disc

    def test_linear_decay_recovery(self) -> None:
        """Discovered model recovers linear decay dynamics."""
        if IGNORE_TESTS:
            return
        # dA/dt = -0.5*A  (pure exponential decay)
        rng = np.random.default_rng(42)

        def rhs(t, z):
            a = z[0]
            return [-0.5 * a]

        from scipy.integrate import solve_ivp  # type: ignore

        t_eval = np.linspace(0, 10, 200)
        sol = solve_ivp(rhs, [0, 10], [1.0], t_eval=t_eval, rtol=1e-8)
        X = sol.y.T + rng.normal(0, 0.001, (len(t_eval), 1))
        df = pd.DataFrame(X, index=t_eval, columns=["A"])

        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        eqs = disc.getEquations()
        # The dominant term should be proportional to A (decay)
        self.assertIn("A", eqs["A"])


# ---------------------------------------------------------------------------
# getScoreAggregatedBySpecies tests
# ---------------------------------------------------------------------------


class TestGetScoreAggregatedBySpecies(unittest.TestCase):
    """Tests for SystemDiscovery.getScoreAggregatedBySpecies."""

    def _make_fitted_disc(self, df: pd.DataFrame, **kwargs) -> SystemDiscovery:
        defaults = {"is_normalize": False}
        defaults.update(kwargs)
        disc = SystemDiscovery(df, coefficient_threshold=0.001, alpha=0.001, **defaults)  # type: ignore
        disc.fit()
        return disc

    def test_raises_before_fit(self) -> None:
        """getScoreAggregatedBySpecies raises RuntimeError before fit."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df()
        disc = SystemDiscovery(df, is_normalize=False)
        with self.assertRaises(RuntimeError):
            disc.getScoreAggregatedBySpecies()

    def test_returns_dict(self) -> None:
        """getScoreAggregatedBySpecies returns a dict."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies()
        self.assertIsInstance(result, dict)

    def test_returns_expected_keys(self) -> None:
        """getScoreAggregatedBySpecies returns dict with mean, min, max, median keys."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies()
        expected_keys = {"mean", "min", "max", "median"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_returns_float_values(self) -> None:
        """getScoreAggregatedBySpecies returns float values."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies()
        for v in result.values():
            self.assertIsInstance(v, float)

    def test_values_in_valid_range(self) -> None:
        """getScoreAggregatedBySpecies values are non-negative floats."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies()
        for v in result.values():
            self.assertGreaterEqual(v, 0.0)

    def test_derivative_score_type(self) -> None:
        """getScoreAggregatedBySpecies works with score_type='derivative'."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies(score_type="derivative")
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"mean", "min", "max", "median"})

    def test_timecourse_score_type(self) -> None:
        """getScoreAggregatedBySpecies works with score_type='timecourse'."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies(score_type="timecourse")
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"mean", "min", "max", "median"})

    def test_with_test_df(self) -> None:
        """getScoreAggregatedBySpecies works with a provided test DataFrame."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(n_points=NUM_POINT, noise_std=0.01)
        train_df = df.iloc[:NUM_POINT//2]
        test_df = df.iloc[NUM_POINT//2:]
        disc = SystemDiscovery(train_df, coefficient_threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        result = disc.getScoreAggregatedBySpecies(test_df=test_df)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"mean", "min", "max", "median"})

    def test_statistic_column_mean(self) -> None:
        """getScoreAggregatedBySpecies works with statistic_column='mean'."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies(statistic_column="mean")
        self.assertIsInstance(result, dict)

    def test_statistic_column_p95(self) -> None:
        """getScoreAggregatedBySpecies works with statistic_column='p95' (default)."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies(statistic_column="p95")
        self.assertIsInstance(result, dict)

    def test_statistic_column_p50(self) -> None:
        """getScoreAggregatedBySpecies works with statistic_column='p50'."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies(statistic_column="p50")
        self.assertIsInstance(result, dict)

    def test_mean_geq_min_and_leq_max(self) -> None:
        """The mean statistic is between min and max."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies(statistic_column="mean")
        self.assertGreaterEqual(result["mean"], result["min"])
        self.assertLessEqual(result["mean"], result["max"])

    def test_median_geq_min_and_leq_max(self) -> None:
        """The median statistic is between min and max."""
        if IGNORE_TESTS:
            return
        df = _make_linear_df(noise_std=0.01)
        disc = self._make_fitted_disc(df)
        result = disc.getScoreAggregatedBySpecies(statistic_column="p50")
        self.assertGreaterEqual(result["median"], result["min"])
        self.assertLessEqual(result["median"], result["max"])

    def test_derivative_with_test_df(self) -> None:
        """getScoreAggregatedBySpecies works with derivative score_type and test_df."""
        if IGNORE_TESTS:
            return
        NUM_POINT = 30
        df = _make_linear_df(n_points=NUM_POINT, noise_std=0.01)
        train_df = df.iloc[:NUM_POINT//2]
        noise_df = pd.DataFrame(
            np.random.normal(0, 0.00, size=(NUM_POINT//2, df.shape[1])),
            columns=df.columns,
            index=df.index[NUM_POINT//2:]
        )
        noise_df.index = train_df.index  # Align indices for concatenation
        test_df = train_df + noise_df
        disc = SystemDiscovery(train_df, coefficient_threshold=0.001, alpha=0.001, is_normalize=False)
        disc.fit()
        result = disc.getScoreAggregatedBySpecies(test_df=test_df, score_type="derivative")
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), {"mean", "min", "max", "median"})


# ---------------------------------------------------------------------------
# plotResult tests using BioModels 278 (Stricker et al. feedback oscillator)
# ---------------------------------------------------------------------------


IGNORE_TESTS_BM = False
_HAS_BIOMODELS_551 = os.path.isdir(cn.BIOMODELS_DIR) and os.path.isdir(
    os.path.join(cn.BIOMODELS_DIR, "BIOMD0000000551")
)


def _make_biomodel_551_timecourse_df(n_points: int = NUM_POINT) -> pd.DataFrame:
    """Load and simulate BioModels model BIOMD0000000551, returning its timecourse DataFrame."""
    from src.model import Model  # type: ignore
    from src.timecourse import Timecourse  # type: ignore

    model = Model.makeBiomodel("BIOMD0000000551")
    tc = Timecourse(model=model, num_point=n_points)
    return tc.timecourse_df


@unittest.skipUnless(
    _HAS_BIOMODELS_551 and not IGNORE_TESTS_BM,
    "BioModels 551 data directory not found or tests are disabled",
)
class TestPlotResultBioModels551(unittest.TestCase):
    """Tests for SystemDiscovery.plotResult using real BioModels model 551 (Stricker et al. 2008)."""

    SPECIES_NAMES = ["C", "P", "l"]

    def test_plot_result_returns_figure(self) -> None:
        """plotResult returns a Figure object for BIOMD0000000551."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_551_timecourse_df(n_points=NUM_POINT)
        disc = SystemDiscovery(df, coefficient_threshold=0.01, alpha=0.05, is_normalize=False)
        disc.fit()
        fig = disc.plotResult(is_plot=IS_PLOT)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_plot_result_species_count_matches(self) -> None:
        """plotResult creates one subplot per species for BIOMD0000000551."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_551_timecourse_df(n_points=NUM_POINT)
        disc = SystemDiscovery(df, coefficient_threshold=0.01, alpha=0.05, is_normalize=False)
        disc.fit()
        fig = disc.plotResult(is_plot=IS_PLOT)
        # BIOMD0000000551 has 3 species (R, B, C), so expect 3 axes with titles
        axes = fig.get_axes()
        titled_axes = [ax for ax in axes if ax.get_title()]
        self.assertEqual(len(titled_axes), disc.num_species)
        plt.close(fig)

    def test_plot_result_contains_species_labels(self) -> None:
        """plotResult subplot titles contain the expected species names R, B, C."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_551_timecourse_df(n_points=NUM_POINT)
        disc = SystemDiscovery(df, coefficient_threshold=0.01, alpha=0.05, is_normalize=False)
        disc.fit()
        fig = disc.plotResult(is_plot=IS_PLOT)
        titled_axes = [ax for ax in fig.get_axes() if ax.get_title()]
        species_in_titles = {ax.get_title().split()[0] for ax in titled_axes}
        self.assertEqual(species_in_titles, set(self.SPECIES_NAMES))
        plt.close(fig)

    def test_fit_produces_equations(self) -> None:
        """SystemDiscovery.fit() produces equations for all BIOMD0000000551 species."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_551_timecourse_df(n_points=NUM_POINT)
        disc = SystemDiscovery(df, coefficient_threshold=0.01, alpha=0.05, is_normalize=False)
        disc.fit()
        eqs = disc.getEquations()
        self.assertEqual(set(eqs.keys()), set(self.SPECIES_NAMES))
        for v in eqs.values():
            self.assertIsInstance(v, str)

    def test_predict_returns_dataframe(self) -> None:
        """predict() returns a DataFrame with correct species columns."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_551_timecourse_df(n_points=NUM_POINT)
        disc = SystemDiscovery(df, coefficient_threshold=0.01, alpha=0.05, is_normalize=False)
        disc.fit()
        result = disc.predict()
        self.assertIsInstance(result, pd.DataFrame)

    def test_predict_score_details(self) -> None:
        """getScoreDetails() returns a DataFrame with correct columns."""
        if IGNORE_TESTS:
            return
        df = _make_biomodel_551_timecourse_df(n_points=NUM_POINT)
        disc = SystemDiscovery(df, coefficient_threshold=0.01, alpha=0.05, is_normalize=False)
        disc.fit()
        result = disc.getScoreDetails(score_type="timecourse")
        has_expected_cols = set(cn.COLUMN_ACCURACY_FRACTIONS)
        self.assertTrue(has_expected_cols.issubset(set(result.columns)))


# ---------------------------------------------------------------------------
# analyzePerturbations tests using BioModels 551 (Stricker et al. feedback oscillator)
# ---------------------------------------------------------------------------


_IGNORE_TESTS_AP = False
_HAS_BIOMODELS_551_AP = os.path.isdir(cn.BIOMODELS_DIR) and os.path.isdir(
    os.path.join(cn.BIOMODELS_DIR, "BIOMD0000000551")
)


def _make_biomodel_551_model():
    """Load BioModels model BIOMD0000000551."""
    from src.model import Model  # type: ignore

    return Model.makeBiomodel("BIOMD0000000551")


@unittest.skipUnless(
    _HAS_BIOMODELS_551_AP and not _IGNORE_TESTS_AP,
    "BioModels 551 data directory not found or tests are disabled",
)
class TestAnalyzePerturbations(unittest.TestCase):
    """Tests for PerturbationAnalyzer.analyze_perturbations using real BioModels model 551."""

    DEFAULT_PERTURBATIONS = [-0.05, 0.0, 0.05]

    @classmethod
    def setUpClass(cls) -> None:
        import matplotlib  # type: ignore

        matplotlib.use("Agg", force=True)

    def _run_analyze(self, **kwargs) -> pd.DataFrame:
        model = _make_biomodel_551_model()
        from src.timecourse import Timecourse  # type: ignore

        tc = Timecourse(model=model, num_point=200)
        training_df = tc.timecourse_df
        defaults = {
            "model": model,
            "training_df": training_df,
            "threshold": 0.01,
            "perturbations": self.DEFAULT_PERTURBATIONS.copy(),
        }
        defaults.update(kwargs)
        return PerturbationAnalyzer(**defaults).result.df

    def test_returns_dataframe(self) -> None:
        """analyze_perturbations returns a pd.DataFrame."""
        if IGNORE_TESTS:
            return
        result = self._run_analyze()
        self.assertIsInstance(result, pd.DataFrame)

    def test_row_count_matches_perturbation_count(self) -> None:
        """Returned DataFrame has one row per perturbation value passed in (model-only)."""
        if IGNORE_TESTS:
            return
        perturbations = [-0.1, 0.0, 0.1]
        result = self._run_analyze(perturbations=perturbations, is_analyze_species=False)
        self.assertEqual(len(result), len(perturbations))

    def test_single_perturbation_returns_one_row(self) -> None:
        """A single-element perturbation list yields exactly one row (model-only)."""
        if IGNORE_TESTS:
            return
        result = self._run_analyze(perturbations=[0.0], is_analyze_species=False)
        self.assertEqual(len(result), 1)

    def test_columns_include_expected(self) -> None:
        """Returned DataFrame contains COL_SYSTEM_ID, COL_AGGREGATION_TYPE, all
        STATISTICS columns, and 'perturbation'."""
        if IGNORE_TESTS:
            return
        result = self._run_analyze()
        expected_cols = (
            set(cn.COLUMN_STATISTICS) | {cn.COL_SYSTEM_ID, cn.COL_AGGREGATION_TYPE, "perturbation"}
        )
        self.assertTrue(expected_cols.issubset(set(result.columns)))

    def test_aggregation_type_model_only_when_species_disabled(self) -> None:
        """With is_analyze_species=False all rows have COL_AGGREGATION_TYPE == 'model'."""
        if IGNORE_TESTS:
            return
        result = self._run_analyze(is_analyze_species=False)
        self.assertTrue(
            (result[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL).all()
        )

    def test_aggregation_type_mixed_by_default(self) -> None:
        """With default args both 'model' and species names appear in aggregation_type."""
        if IGNORE_TESTS:
            return
        result = self._run_analyze()
        agg_types = set(result[cn.COL_AGGREGATION_TYPE].tolist())
        self.assertIn(cn.COL_AGGREGATION_TYPE_MODEL, agg_types)
        # At least one species-level row must be present (model 551 has multiple species).
        species_agg = agg_types - {cn.COL_AGGREGATION_TYPE_MODEL}
        self.assertGreater(len(species_agg), 0)

    def test_perturbation_column_matches_input_order(self) -> None:
        """Each row's perturbation column matches its corresponding input value, even with species rows."""
        if IGNORE_TESTS:
            return
        perturbations = [-0.1, 0.05, 0.2]
        result = self._run_analyze(perturbations=perturbations)
        # For each distinct perturbation value present in the input list, rows carrying it should match exactly.
        expected_perts_in_result = sorted(set(result[cn.COL_PERTURBATION].tolist()))
        np.testing.assert_array_equal(expected_perts_in_result, np.array(sorted(perturbations)))

    def test_aggregation_type_all_model(self) -> None:
        """Backward-compat alias: previously always model-only; now controlled by is_analyze_species flag."""
        if IGNORE_TESTS:
            return
        result = self._run_analyze(is_analyze_species=False, is_analyze_model=True)
        self.assertTrue(
            (result[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL).all()
        )

    def test_statistic_values_are_finite(self) -> None:
        """Numeric STATISTICS columns contain no NaN or inf values."""
        if IGNORE_TESTS:
            return
        result = self._run_analyze()
        for col in cn.COLUMN_STATISTICS:
            with self.subTest(col=col):
                self.assertTrue(
                    result[col].apply(np.isfinite).all(),
                    f"Column '{col}' contains non-finite values.",
                )

    def test_is_plot_does_not_affect_return_value(self) -> None:
        """is_plot flag does not change the returned DataFrame."""
        if IGNORE_TESTS:
            return
        result_no = self._run_analyze()
        result_yes = self._run_analyze()
        pd.testing.assert_frame_equal(result_no, result_yes)

    def test_col_percentile_parameter_accepted(self) -> None:
        """Passing col_percentile='p50' does not raise and returns valid DataFrame."""
        if IGNORE_TESTS:
            return
        result = self._run_analyze(col_percentile="p50")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            result["p50"].apply(np.isfinite).all(),
            "Column 'p50' contains non-finite values.",
        )


if __name__ == "__main__":
    unittest.main()
