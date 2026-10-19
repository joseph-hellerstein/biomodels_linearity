"""Tests for the Scaler class in src/scaler.py."""

import os
import sys
import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scaler import Scaler  # type: ignore

IGNORE_TESTS = False

# ---------------------------------------------------------------------------
# Shared fixture
#
# std([1,3,5], ddof=1) = 2.0   std([2,4,6], ddof=1) = 2.0   std([10,20,30], ddof=1) = 10.0
# ---------------------------------------------------------------------------

_DF = pd.DataFrame({
    'S1': [1.0, 3.0, 5.0],
    'S2': [2.0, 4.0, 6.0],
    'S3': [10.0, 20.0, 30.0],
})

_SIGMA = {'S1': 2.0, 'S2': 2.0, 'S3': 10.0}


# ---------------------------------------------------------------------------

class TestScalerConstructor(unittest.TestCase):
    """Constructor sets _scale_arr and _scaling_dct correctly."""

    def test_scale_arr_equals_col_stds(self) -> None:
        """_scale_arr holds per-column std (ddof=1)."""
        if IGNORE_TESTS:
            return
        s = Scaler(_DF)
        expected = np.std(_DF.values, axis=0, ddof=1)
        np.testing.assert_allclose(s._scale_arr, expected)

    def test_scaling_dct_maps_names_to_stds(self) -> None:
        """_scaling_dct maps each column name to its std."""
        if IGNORE_TESTS:
            return
        s = Scaler(_DF)
        for name, sigma in _SIGMA.items():
            self.assertAlmostEqual(s._scaling_dct[name], sigma)

    def test_null_scaler_scale_arr_is_ones(self) -> None:
        """is_null_scaler=True sets _scale_arr to all ones."""
        if IGNORE_TESTS:
            return
        s = Scaler(_DF, is_null_scaler=True)
        np.testing.assert_array_equal(s._scale_arr, np.ones(3))

    def test_null_scaler_scaling_dct_values_are_one(self) -> None:
        """is_null_scaler=True makes every _scaling_dct value 1.0."""
        if IGNORE_TESTS:
            return
        s = Scaler(_DF, is_null_scaler=True)
        for v in s._scaling_dct.values():
            self.assertEqual(v, 1.0)

    def test_constant_column_fallback_to_inverse_mean(self) -> None:
        """A constant column (std=0) uses 1/col_mean as scale."""
        if IGNORE_TESTS:
            return
        df = pd.DataFrame({'A': [5.0, 5.0, 5.0], 'B': [1.0, 3.0, 5.0]})
        s = Scaler(df)
        self.assertAlmostEqual(s._scaling_dct['A'], 1.0 / 5.0)

    def test_constant_zero_column_fallback_to_one(self) -> None:
        """A constant-zero column (std=0, mean=0) uses scale=1.0."""
        if IGNORE_TESTS:
            return
        df = pd.DataFrame({'A': [0.0, 0.0, 0.0], 'B': [1.0, 3.0, 5.0]})
        s = Scaler(df)
        self.assertAlmostEqual(s._scaling_dct['A'], 1.0)

    def test_constant_cols_identifies_zero_std_columns(self) -> None:
        """_constant_cols contains exactly the columns with zero std."""
        if IGNORE_TESTS:
            return
        df = pd.DataFrame({'A': [5.0, 5.0, 5.0], 'B': [1.0, 3.0, 5.0]})
        s = Scaler(df)
        self.assertEqual(s._constant_cols, frozenset({'A'}))

    def test_constant_cols_empty_for_variable_data(self) -> None:
        """_constant_cols is empty when all columns have nonzero std."""
        if IGNORE_TESTS:
            return
        s = Scaler(_DF)
        self.assertEqual(s._constant_cols, frozenset())

    def test_null_scaler_constant_cols_empty(self) -> None:
        """_constant_cols is empty for a null scaler."""
        if IGNORE_TESTS:
            return
        df = pd.DataFrame({'A': [5.0, 5.0, 5.0], 'B': [1.0, 3.0, 5.0]})
        s = Scaler(df, is_null_scaler=True)
        self.assertEqual(s._constant_cols, frozenset())


class TestScalerNormalize(unittest.TestCase):
    """normalize() divides each column by its std."""

    def setUp(self) -> None:
        self.s = Scaler(_DF)

    def test_normalize_divides_by_scale(self) -> None:
        """normalize(X) returns X / sigma for each column."""
        if IGNORE_TESTS:
            return
        X = np.array([[1.0, 2.0, 10.0]])
        result = self.s.normalize(X)
        expected = np.array([[1.0 / 2.0, 2.0 / 2.0, 10.0 / 10.0]])
        np.testing.assert_allclose(result, expected)

    def test_normalize_shape_preserved(self) -> None:
        """Output shape equals input shape."""
        if IGNORE_TESTS:
            return
        X = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
        self.assertEqual(self.s.normalize(X).shape, X.shape)

    def test_normalize_empty_uses_constructor_matrix(self) -> None:
        """normalize(np.array([])) normalizes self._matrix_arr."""
        if IGNORE_TESTS:
            return
        result = self.s.normalize(np.array([]))
        expected = _DF.values / np.std(_DF.values, axis=0, ddof=1)
        np.testing.assert_allclose(result, expected)

    def test_null_scaler_normalize_is_identity(self) -> None:
        """is_null_scaler=True: normalize returns the input unchanged."""
        if IGNORE_TESTS:
            return
        s = Scaler(_DF, is_null_scaler=True)
        X = np.array([[3.0, 4.0, 20.0]])
        np.testing.assert_array_equal(s.normalize(X), X)


class TestScalerDenormalize(unittest.TestCase):
    """denormalize() multiplies each column by its std."""

    def setUp(self) -> None:
        self.s = Scaler(_DF)

    def test_denormalize_constant_column_near_zero_mean(self) -> None:
        """denormalize substitutes the mean for a constant column, even when
        that mean is near zero (previously this multiplied by 1/mean instead)."""
        if IGNORE_TESTS:
            return
        df = pd.DataFrame({'A': [1e-10, 1e-10, 1e-10], 'B': [1.0, 3.0, 5.0]})
        s = Scaler(df)
        Z = np.array([[1.0, 1.0]])  # normalize() forces constant columns to 1.0
        result = s.denormalize(Z)
        self.assertAlmostEqual(result[0, 0], 1e-10)


    def test_denormalize_multiplies_by_scale(self) -> None:
        """denormalize(Z) returns Z * sigma for each column."""
        if IGNORE_TESTS:
            return
        Z = np.array([[0.5, 1.0, 1.0]])
        result = self.s.denormalize(Z)
        expected = np.array([[0.5 * 2.0, 1.0 * 2.0, 1.0 * 10.0]])
        np.testing.assert_allclose(result, expected)

    def test_roundtrip(self) -> None:
        """denormalize(normalize(X)) recovers X."""
        if IGNORE_TESTS:
            return
        X = _DF.values.copy()
        back = self.s.denormalize(self.s.normalize(X))
        np.testing.assert_allclose(back, X, atol=1e-12)

    def test_null_scaler_denormalize_is_identity(self) -> None:
        """is_null_scaler=True: denormalize returns the input unchanged."""
        if IGNORE_TESTS:
            return
        s = Scaler(_DF, is_null_scaler=True)
        Z = np.array([[0.5, 1.0, 2.0]])
        np.testing.assert_array_equal(s.denormalize(Z), Z)


class TestScalerDenormalizeCoordinate(unittest.TestCase):
    """denormalizeCoordinate applies c = c' * sigma_i / product(sigma_j^p_j)."""

    def setUp(self) -> None:
        self.s = Scaler(_DF)

    # ------------------------------------------------------------------
    # Individual term types
    # ------------------------------------------------------------------

    def test_constant_term(self) -> None:
        """factor_str='1': c = c' * sigma_i."""
        if IGNORE_TESTS:
            return
        # sigma_S2 = 2.0; c' = 3.0 -> c = 3.0 * 2.0 / 1.0 = 6.0
        result = self.s.denormalizeCoordinate('S2', '1', 3.0)
        self.assertAlmostEqual(result, 6.0)

    def test_linear_same_species(self) -> None:
        """factor_str='S2' with state_variable='S2': sigma_i / sigma_j cancels."""
        if IGNORE_TESTS:
            return
        # sigma_S2=2.0; c' * 2.0 / 2.0 = c'
        result = self.s.denormalizeCoordinate('S2', 'S2', 3.0)
        self.assertAlmostEqual(result, 3.0)

    def test_linear_other_species(self) -> None:
        """factor_str='S1', state_variable='S2': c = c' * sigma_S2 / sigma_S1."""
        if IGNORE_TESTS:
            return
        # sigma_S1=2.0, sigma_S2=2.0; c = 3.0 * 2.0 / 2.0 = 3.0
        result = self.s.denormalizeCoordinate('S2', 'S1', 3.0)
        self.assertAlmostEqual(result, 3.0)

    def test_linear_different_scales(self) -> None:
        """factor_str='S3', state_variable='S1': c = c' * sigma_S1 / sigma_S3."""
        if IGNORE_TESTS:
            return
        # sigma_S1=2.0, sigma_S3=10.0; c = 5.0 * 2.0 / 10.0 = 1.0
        result = self.s.denormalizeCoordinate('S1', 'S3', 5.0)
        self.assertAlmostEqual(result, 1.0)

    def test_quadratic_term(self) -> None:
        """factor_str='S1^2': c = c' * sigma_i / sigma_S1^2."""
        if IGNORE_TESTS:
            return
        # sigma_S1=2.0, sigma_S2=2.0; c = 3.0 * 2.0 / 4.0 = 1.5
        result = self.s.denormalizeCoordinate('S2', 'S1^2', 3.0)
        self.assertAlmostEqual(result, 1.5)

    def test_cross_term(self) -> None:
        """factor_str='S1 S2': c = c' * sigma_i / (sigma_S1 * sigma_S2)."""
        if IGNORE_TESTS:
            return
        # sigma_S1=2, sigma_S2=2, sigma_S3=10; c = 5.0 * 10.0 / (2.0 * 2.0) = 12.5
        result = self.s.denormalizeCoordinate('S3', 'S1 S2', 5.0)
        self.assertAlmostEqual(result, 12.5)

    def test_mixed_quadratic_cross_term(self) -> None:
        """factor_str='S1^2 S2': c = c' * sigma_i / (sigma_S1^2 * sigma_S2)."""
        if IGNORE_TESTS:
            return
        # sigma_S1=2, sigma_S2=2, sigma_S3=10; c = 4.0 * 10.0 / (4.0 * 2.0) = 5.0
        result = self.s.denormalizeCoordinate('S3', 'S1^2 S2', 4.0)
        self.assertAlmostEqual(result, 5.0)

    def test_null_scaler_returns_value_unchanged(self) -> None:
        """is_null_scaler=True: denormalizeCoordinate is an identity."""
        if IGNORE_TESTS:
            return
        s = Scaler(_DF, is_null_scaler=True)
        result = s.denormalizeCoordinate('S1', 'S1^2 S2', 7.5)
        self.assertAlmostEqual(result, 7.5)

    # ------------------------------------------------------------------
    # Round-trip tests
    #
    # Forward direction (normalizeCoordinate):
    #   c' = c * Π_j sigma_j^{p_j} / sigma_i
    # Inverse (denormalizeCoordinate):
    #   c  = c' * sigma_i / Π_j sigma_j^{p_j}
    # ------------------------------------------------------------------

    def _normalize_coord(self, state_var: str, factor_str: str, c: float) -> float:
        """Helper: convert original coefficient to normalized space."""
        powers = self.s._parseFeaturePowers(factor_str)
        factor_adj = 1.0
        for name, power in powers.items():
            factor_adj *= self.s._scaling_dct[name] ** power
        return c * factor_adj / self.s._scaling_dct[state_var]

    def test_roundtrip_constant_term(self) -> None:
        """denormalizeCoordinate inverts the constant-term normalization."""
        if IGNORE_TESTS:
            return
        c = 6.0
        c_norm = self._normalize_coord('S2', '1', c)
        self.assertAlmostEqual(self.s.denormalizeCoordinate('S2', '1', c_norm), c)

    def test_roundtrip_linear_term(self) -> None:
        """denormalizeCoordinate inverts a linear-term normalization."""
        if IGNORE_TESTS:
            return
        c = -3.5
        c_norm = self._normalize_coord('S1', 'S3', c)
        self.assertAlmostEqual(self.s.denormalizeCoordinate('S1', 'S3', c_norm), c)

    def test_roundtrip_quadratic_term(self) -> None:
        """denormalizeCoordinate inverts a quadratic-term normalization."""
        if IGNORE_TESTS:
            return
        c = 2.0
        c_norm = self._normalize_coord('S2', 'S1^2', c)
        self.assertAlmostEqual(self.s.denormalizeCoordinate('S2', 'S1^2', c_norm), c)

    def test_roundtrip_cross_term(self) -> None:
        """denormalizeCoordinate inverts a cross-term normalization."""
        if IGNORE_TESTS:
            return
        c = 7.0
        c_norm = self._normalize_coord('S3', 'S1 S2', c)
        self.assertAlmostEqual(self.s.denormalizeCoordinate('S3', 'S1 S2', c_norm), c)

    def test_roundtrip_mixed_term(self) -> None:
        """denormalizeCoordinate inverts a mixed quadratic-cross normalization."""
        if IGNORE_TESTS:
            return
        c = -1.25
        c_norm = self._normalize_coord('S3', 'S1^2 S2', c)
        self.assertAlmostEqual(self.s.denormalizeCoordinate('S3', 'S1^2 S2', c_norm), c)


class TestScalerNormalizeThreshold(unittest.TestCase):
    """normalizeThreshold returns the per-coefficient normalized-space threshold.

    Formula: threshold_norm = tau * Π(sigma_j^p_j) / sigma_i
    so that |c_norm| < threshold_norm  iff  |c_physical| < tau.
    """

    def setUp(self) -> None:
        self.s = Scaler(_DF)  # sigma: S1=2, S2=2, S3=10

    def test_constant_term(self) -> None:
        """factor_str='1': threshold_norm = tau / sigma_i."""
        if IGNORE_TESTS:
            return
        # tau / sigma_S2 = 1.0 / 2.0 = 0.5
        result = self.s.normalizeThreshold('S2', '1', 1.0)
        self.assertAlmostEqual(result, 0.5)

    def test_linear_same_species(self) -> None:
        """factor_str='S2', state_variable='S2': sigma cancels, threshold_norm = tau."""
        if IGNORE_TESTS:
            return
        result = self.s.normalizeThreshold('S2', 'S2', 1.0)
        self.assertAlmostEqual(result, 1.0)

    def test_linear_different_scales(self) -> None:
        """factor_str='S3', state_variable='S1': threshold_norm = tau * sigma_S3 / sigma_S1."""
        if IGNORE_TESTS:
            return
        # 1.0 * 10.0 / 2.0 = 5.0
        result = self.s.normalizeThreshold('S1', 'S3', 1.0)
        self.assertAlmostEqual(result, 5.0)

    def test_quadratic_term(self) -> None:
        """factor_str='S1^2': threshold_norm = tau * sigma_S1^2 / sigma_S2."""
        if IGNORE_TESTS:
            return
        # 1.0 * 4.0 / 2.0 = 2.0
        result = self.s.normalizeThreshold('S2', 'S1^2', 1.0)
        self.assertAlmostEqual(result, 2.0)

    def test_cross_term(self) -> None:
        """factor_str='S1 S2': threshold_norm = tau * sigma_S1 * sigma_S2 / sigma_S3."""
        if IGNORE_TESTS:
            return
        # 1.0 * 2.0 * 2.0 / 10.0 = 0.4
        result = self.s.normalizeThreshold('S3', 'S1 S2', 1.0)
        self.assertAlmostEqual(result, 0.4)

    def test_null_scaler_returns_unchanged(self) -> None:
        """is_null_scaler=True: normalizeThreshold returns physical_threshold unchanged."""
        if IGNORE_TESTS:
            return
        s = Scaler(_DF, is_null_scaler=True)
        result = s.normalizeThreshold('S1', 'S1^2 S2', 7.5)
        self.assertAlmostEqual(result, 7.5)

    def test_inverse_of_denormalize_scaling(self) -> None:
        """normalizeThreshold is the inverse scaling of denormalizeCoordinate.

        If c_physical = denormalizeCoordinate(v, f, c_norm), then
        |c_norm| == normalizeThreshold(v, f, tau) implies |c_physical| == tau.
        """
        if IGNORE_TESTS:
            return
        tau = 3.0
        c_norm = self.s.normalizeThreshold('S3', 'S1 S2', tau)
        c_physical = self.s.denormalizeCoordinate('S3', 'S1 S2', c_norm)
        self.assertAlmostEqual(c_physical, tau)


if __name__ == "__main__":
    unittest.main()
