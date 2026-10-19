# Piecewise System Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `PiecewiseSystemDiscovery` (`src/piecewise_system_discovery.py`) per `docs/piecewise_system_discovery.md`: detects Jacobian-based change points in a `Timecourse`, fits one `SystemDiscovery` per segment, and blends segment predictions with a Gaussian kernel. Also adds the prerequisite `SystemDiscovery.predictOneStepDerivative()` method the new class depends on.

**Architecture:** Two source changes. (1) `src/system_discovery.py` gets one new public method, `predictOneStepDerivative(x)`, factored from the existing `_simulate()` rhs closure — a non-integrating, single-state derivative evaluator. (2) New module `src/piecewise_system_discovery.py` defines `PiecewiseSystemDiscovery`, built in dependency order: change-point signal → change-point detection → `fit()` (wires both together, builds segment models) → `predict_derivative()` → `predict()` → `score()` → `printEquations()`/`__str__()`.

**Tech Stack:** Python, numpy, pandas, scipy (`solve_ivp`), pysindy (via `SystemDiscovery`), unittest (matches existing test style in `tests/`).

## Global Constraints

- Follow `source activate.sh` before running any test/lint command (adds `src/` to `PYTHONPATH`).
- Match existing import style in `tests/`: bare module imports (`from system_discovery import ...`, `from timecourse import Timecourse`) rather than `src.`-prefixed, since `src/` is on `PYTHONPATH` directly. Inside `src/`, use `src.`-prefixed imports (matches `src/system_discovery.py`'s own imports).
- **[CORRECTED during Task 7 review — see "Correction" note after Task 5]** Per-segment `SystemDiscovery` construction passes `**kwargs` through unmodified; `SystemDiscovery`'s own default (`is_normalize=True`) applies unless the caller overrides it. Segments are fit on raw (unscaled) data slices, never on a globally pre-scaled DataFrame. (Originally this said `is_normalize=False` was always forced over pre-scaled data — that was a confirmed units-mismatch bug, fixed before Task 8.)
- No placeholders, no speculative error handling beyond what the doc specifies.
- Run `python3 -m pytest tests/test_system_discovery.py tests/test_piecewise_system_discovery.py -v` after every task to confirm no regressions.

---

### Task 1: `SystemDiscovery.predictOneStepDerivative`

**Files:**
- Modify: `src/system_discovery.py:601` (insert new method immediately after `predict()`, before `def printEquations`)
- Test: `tests/test_system_discovery.py` (append new test class)

**Interfaces:**
- Produces: `SystemDiscovery.predictOneStepDerivative(x: np.ndarray) -> np.ndarray` — evaluates the fitted ODE's right-hand side at a single state, no integration, in physical (denormalized) units. Required by `PiecewiseSystemDiscovery.predict_derivative()` in Task 6.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_system_discovery.py` (before the final `if __name__ == "__main__":` block):

```python
class TestPredictOneStepDerivative(unittest.TestCase):
    """Tests for SystemDiscovery.predictOneStepDerivative."""

    def test_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        nd = SystemDiscovery(
            _DECAY_DF, threshold=0.01, alpha=0.01, poly_degree=1,
            include_bias=False, differentiation="finite",
        )
        with self.assertRaises(RuntimeError):
            nd.predictOneStepDerivative(np.array([5.0]))

    def test_matches_decay_rate_sign(self) -> None:
        """For dS1/dt = -0.2*S1, the derivative at a positive state is negative."""
        if IGNORE_TESTS:
            return
        nd = _get_fitted_decay()
        result = nd.predictOneStepDerivative(np.array([5.0]))
        self.assertLess(result[0], 0.0)

    def test_returns_array_of_correct_shape(self) -> None:
        if IGNORE_TESTS:
            return
        nd = _get_fitted_two_species()
        result = nd.predictOneStepDerivative(np.array([10.0, 0.0]))
        self.assertEqual(result.shape, (2,))

    def test_agrees_with_predict_first_step(self) -> None:
        """predictOneStepDerivative at x0 should be close to the finite-difference
        slope predict() produces over the first short interval."""
        if IGNORE_TESTS:
            return
        nd = _get_fitted_decay()
        x0 = nd.X[0, :]
        derivative = nd.predictOneStepDerivative(x0)
        predicted_df = nd.predict()
        dt = predicted_df.index[1] - predicted_df.index[0]
        finite_diff_slope = (predicted_df.iloc[1].to_numpy() - predicted_df.iloc[0].to_numpy()) / dt
        self.assertAlmostEqual(derivative[0], finite_diff_slope[0], delta=0.5)

    def test_normalize_and_no_normalize_agree(self) -> None:
        """Physical-units derivative should match regardless of is_normalize."""
        if IGNORE_TESTS:
            return
        nd_norm = SystemDiscovery(
            _TWO_SPECIES_DF, threshold=0.01, alpha=0.01, poly_degree=1,
            include_bias=False, differentiation="finite", is_normalize=True,
        ).fit()
        nd_raw = SystemDiscovery(
            _TWO_SPECIES_DF, threshold=0.01, alpha=0.01, poly_degree=1,
            include_bias=False, differentiation="finite", is_normalize=False,
        ).fit()
        x = np.array([10.0, 0.0])
        np.testing.assert_allclose(
            nd_norm.predictOneStepDerivative(x),
            nd_raw.predictOneStepDerivative(x),
            atol=0.5,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source activate.sh && python3 -m pytest tests/test_system_discovery.py::TestPredictOneStepDerivative -v`
Expected: FAIL with `AttributeError: 'SystemDiscovery' object has no attribute 'predictOneStepDerivative'`

- [ ] **Step 3: Implement the method**

In `src/system_discovery.py`, insert immediately after `predict()` (after line 601, the `return pd.DataFrame(...)` line, before `def printEquations`):

```python
    def predictOneStepDerivative(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the fitted ODE's right-hand side at a single state (no integration).

        Parameters
        ----------
        x : np.ndarray
            State vector in physical units, shape (n_species,), in the same
            species order as `self.species_names`.

        Returns
        -------
        np.ndarray
            Derivative dx/dt at `x`, in physical units, shape (n_species,).
        """
        self._require_fitted()
        z = self._normalizer.normalize(x)
        dz_dt = self.model.predict(z.reshape(1, -1))[0]
        return np.array(self._normalizer.denormalize(dz_dt), dtype=float)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source activate.sh && python3 -m pytest tests/test_system_discovery.py::TestPredictOneStepDerivative -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run full SystemDiscovery suite to confirm no regressions**

Run: `source activate.sh && python3 -m pytest tests/test_system_discovery.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/system_discovery.py tests/test_system_discovery.py
git commit -m "feat: add SystemDiscovery.predictOneStepDerivative for single-state RHS evaluation"
```

---

### Task 2: `PiecewiseSystemDiscovery` skeleton — constructor and guard

**Files:**
- Create: `src/piecewise_system_discovery.py`
- Test: `tests/test_piecewise_system_discovery.py`

**Interfaces:**
- Consumes: `src.timecourse.Timecourse` (`.timecourse_df: pd.DataFrame`, `.jacobian_collection_arr: np.ndarray`), `src.scaler.Scaler`, `src.system_discovery.SystemDiscovery`, `src.system_discovery.ScoreInfo`.
- Produces: `PiecewiseSystemDiscovery.__init__(timecourse, num_change_point=2, min_segment_length=100, change_point_threshold=0.1, fit_kernel_bandwidth=1.0, predict_kernel_bandwidth=1.0, **kwargs)`. Internal state: `self.timecourse`, `self.num_change_point`, `self.min_segment_length`, `self.change_point_threshold`, `self.fit_kernel_bandwidth`, `self.predict_kernel_bandwidth`, `self._kwargs`, `self._segment_models: list[SystemDiscovery]`, `self._segment_boundaries: list[tuple[float, float]]`, `self._segment_lengths: list[int]`, `self._scaler`, `self._is_fitted: bool`. `_require_fitted()` raises `RuntimeError` when `not self._is_fitted`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_piecewise_system_discovery.py`:

```python
"""Tests for PiecewiseSystemDiscovery in piecewise_system_discovery.py."""

import unittest

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore

from model import Model  # type: ignore
from timecourse import Timecourse  # type: ignore
from piecewise_system_discovery import PiecewiseSystemDiscovery  # type: ignore

IGNORE_TESTS = False

_TWO_SPECIES_ANTIMONY = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""


# ---------------------------------------------------------------------------
# Synthetic two-regime fixture: a 2-species linear decay chain whose rate
# constants change sharply at t=5, built directly with solve_ivp (no
# tellurium) so that the ground-truth Jacobian per regime is known exactly.
# Segment A (t in [0, 5)):  dS1/dt = -0.5*S1            ;  dS2/dt = 0.5*S1 - 0.3*S2
# Segment B (t in [5, 10)): dS1/dt = -0.05*S1            ;  dS2/dt = 0.05*S1 - 0.05*S2
# ---------------------------------------------------------------------------
_RATE_A = (0.5, 0.3)
_RATE_B = (0.05, 0.05)
_SPLIT_TIME = 5.0
_END_TIME = 10.0
_NUM_POINT_PER_SEGMENT = 100


def _segment_ode(rates):
    a, b = rates
    def f(_t, x):
        return [-a * x[0], a * x[0] - b * x[1]]
    return f


def _jacobian(rates) -> np.ndarray:
    a, b = rates
    return np.array([[-a, 0.0], [a, -b]])


def _makeTwoRegimeTimecourse(min_segment_length: int = 10) -> Timecourse:
    t_a = np.linspace(0.0, _SPLIT_TIME, _NUM_POINT_PER_SEGMENT, endpoint=False)
    sol_a = solve_ivp(_segment_ode(_RATE_A), [0.0, _SPLIT_TIME], [10.0, 0.0],
            t_eval=t_a, rtol=1e-10, atol=1e-12)
    x_split = sol_a.y[:, -1]
    t_b = np.linspace(_SPLIT_TIME, _END_TIME, _NUM_POINT_PER_SEGMENT)
    sol_b = solve_ivp(_segment_ode(_RATE_B), [_SPLIT_TIME, _END_TIME], x_split,
            t_eval=t_b, rtol=1e-10, atol=1e-12)
    time_arr = np.concatenate([t_a, t_b])
    data_arr = np.concatenate([sol_a.y.T, sol_b.y.T], axis=0)
    timecourse_df = pd.DataFrame(data_arr, index=time_arr, columns=["S1", "S2"])

    jac_a = np.tile(_jacobian(_RATE_A), (len(t_a), 1, 1))
    jac_b = np.tile(_jacobian(_RATE_B), (len(t_b), 1, 1))
    jacobian_collection_arr = np.concatenate([jac_a, jac_b], axis=0)

    model = Model(_TWO_SPECIES_ANTIMONY, model_name="test_model")
    return Timecourse(
        model=model,
        timecourse_df=timecourse_df,
        jacobian_collection_arr=jacobian_collection_arr,
    )


class TestPiecewiseSystemDiscoveryConstructor(unittest.TestCase):
    """Tests for the constructor and pre-fit state."""

    def test_stores_constructor_params(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(
            tc, num_change_point=1, min_segment_length=10,
            change_point_threshold=0.01, fit_kernel_bandwidth=0.5,
            predict_kernel_bandwidth=0.5,
        )
        self.assertEqual(psd.num_change_point, 1)
        self.assertEqual(psd.min_segment_length, 10)
        self.assertAlmostEqual(psd.change_point_threshold, 0.01)
        self.assertAlmostEqual(psd.fit_kernel_bandwidth, 0.5)
        self.assertAlmostEqual(psd.predict_kernel_bandwidth, 0.5)

    def test_kwargs_stored(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, threshold=0.5, poly_degree=1)
        self.assertEqual(psd._kwargs, {"threshold": 0.5, "poly_degree": 1})  # pylint: disable=protected-access

    def test_not_fitted_initially(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        self.assertFalse(psd._is_fitted)  # pylint: disable=protected-access
        self.assertEqual(psd._segment_models, [])  # pylint: disable=protected-access
        self.assertEqual(psd._segment_boundaries, [])  # pylint: disable=protected-access

    def test_require_fitted_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd._require_fitted()  # pylint: disable=protected-access


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'piecewise_system_discovery'`

- [ ] **Step 3: Implement the skeleton**

Create `src/piecewise_system_discovery.py`:

```python
"""Piecewise system discovery: detects Jacobian-based change points in a
Timecourse and fits a separate SystemDiscovery model to each segment,
blending predictions across segment boundaries with a Gaussian kernel.

See docs/piecewise_system_discovery.md for the full design.
"""

import bisect
from typing import Any, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.integrate import solve_ivp  # type: ignore

from src.scaler import Scaler  # type: ignore
from src.system_discovery import ScoreInfo, SystemDiscovery  # type: ignore
from src.timecourse import Timecourse  # type: ignore

NULL_DF = pd.DataFrame()


class PiecewiseSystemDiscovery(object):
    """Piecewise-linear ODE discovery across detected change-point segments."""

    def __init__(
        self,
        timecourse: Timecourse,
        num_change_point: int = 2,
        min_segment_length: int = 100,
        change_point_threshold: float = 0.1,
        fit_kernel_bandwidth: float = 1.0,
        predict_kernel_bandwidth: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.timecourse = timecourse
        self.num_change_point = num_change_point
        self.min_segment_length = min_segment_length
        self.change_point_threshold = change_point_threshold
        self.fit_kernel_bandwidth = fit_kernel_bandwidth
        self.predict_kernel_bandwidth = predict_kernel_bandwidth
        self._kwargs = kwargs

        self._segment_models: List[SystemDiscovery] = []
        self._segment_boundaries: List[Tuple[float, float]] = []
        self._segment_lengths: List[int] = []
        self._scaler: Scaler
        self._is_fitted: bool = False

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                    "PiecewiseSystemDiscovery must be fit() before this operation.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/piecewise_system_discovery.py tests/test_piecewise_system_discovery.py
git commit -m "feat: add PiecewiseSystemDiscovery constructor skeleton"
```

---

### Task 3: Change-point signal (`fit()` steps 1-3)

**Files:**
- Modify: `src/piecewise_system_discovery.py`
- Test: `tests/test_piecewise_system_discovery.py`

**Interfaces:**
- Produces: `PiecewiseSystemDiscovery._gaussianSmooth(times: np.ndarray, values: np.ndarray, bandwidth: float) -> np.ndarray` (static method); `PiecewiseSystemDiscovery._computeChangePointSignal(timecourse_df: pd.DataFrame, jacobian_collection_arr: np.ndarray) -> np.ndarray`, returning one smoothed signal value per interior candidate split index `1..num_point-1` (length `num_point - 1`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_piecewise_system_discovery.py` (before `if __name__ == "__main__":`):

```python
class TestGaussianSmooth(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery._gaussianSmooth."""

    def test_constant_values_unchanged(self) -> None:
        if IGNORE_TESTS:
            return
        times = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([5.0, 5.0, 5.0, 5.0])
        result = PiecewiseSystemDiscovery._gaussianSmooth(times, values, bandwidth=1.0)  # pylint: disable=protected-access
        np.testing.assert_allclose(result, values)

    def test_self_weight_dominates_for_small_bandwidth(self) -> None:
        if IGNORE_TESTS:
            return
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([0.0, 10.0, 0.0])
        result = PiecewiseSystemDiscovery._gaussianSmooth(times, values, bandwidth=0.01)  # pylint: disable=protected-access
        np.testing.assert_allclose(result, values, atol=1e-6)

    def test_smoothing_blends_neighbors_for_large_bandwidth(self) -> None:
        if IGNORE_TESTS:
            return
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([0.0, 10.0, 0.0])
        result = PiecewiseSystemDiscovery._gaussianSmooth(times, values, bandwidth=100.0)  # pylint: disable=protected-access
        self.assertAlmostEqual(result[0], result[1], delta=0.5)
        self.assertAlmostEqual(result[1], result[2], delta=0.5)


class TestComputeChangePointSignal(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery._computeChangePointSignal."""

    def test_signal_length(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, fit_kernel_bandwidth=0.05)
        signal = psd._computeChangePointSignal(  # pylint: disable=protected-access
                tc.timecourse_df, tc.jacobian_collection_arr)
        self.assertEqual(len(signal), len(tc.timecourse_df) - 1)

    def test_signal_peaks_near_regime_split(self) -> None:
        """With a small smoothing bandwidth, the largest signal should occur
        near t=5, where the Jacobian changes sharply."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, fit_kernel_bandwidth=0.05)
        signal = psd._computeChangePointSignal(  # pylint: disable=protected-access
                tc.timecourse_df, tc.jacobian_collection_arr)
        split_time_arr = tc.timecourse_df.index.to_numpy(dtype=float)[1:]
        peak_time = split_time_arr[int(np.argmax(signal))]
        self.assertAlmostEqual(peak_time, _SPLIT_TIME, delta=0.5)

    def test_signal_near_zero_within_constant_regime(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, fit_kernel_bandwidth=0.05)
        signal = psd._computeChangePointSignal(  # pylint: disable=protected-access
                tc.timecourse_df, tc.jacobian_collection_arr)
        split_time_arr = tc.timecourse_df.index.to_numpy(dtype=float)[1:]
        mid_regime_a = np.argmin(np.abs(split_time_arr - 2.0))
        self.assertLess(signal[mid_regime_a], 0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: FAIL with `AttributeError: 'PiecewiseSystemDiscovery' object has no attribute '_gaussianSmooth'` (and similarly for `_computeChangePointSignal`)

- [ ] **Step 3: Implement**

Add to `src/piecewise_system_discovery.py`, inside the `PiecewiseSystemDiscovery` class (after `_require_fitted`):

```python
    @staticmethod
    def _gaussianSmooth(times: np.ndarray, values: np.ndarray, bandwidth: float) -> np.ndarray:
        """Nadaraya-Watson Gaussian kernel smoothing of `values` over `times`."""
        delta_arr = times[:, np.newaxis] - times[np.newaxis, :]
        weight_arr = np.exp(-0.5 * (delta_arr / bandwidth) ** 2)
        return (weight_arr @ values) / weight_arr.sum(axis=1)

    def _computeChangePointSignal(self, timecourse_df: pd.DataFrame,
            jacobian_collection_arr: np.ndarray) -> np.ndarray:
        """fit() steps 2-3: normalized-Jacobian Frobenius distance, smoothed.

        Returns one value per interior candidate split index 1..num_point-1
        (signal[k] corresponds to split index k+1: the point where a new
        segment would start if a change point were placed there).
        """
        num_species = timecourse_df.shape[1]
        std_arr = timecourse_df.to_numpy(dtype=float).std(axis=0, ddof=1)
        safe_std_arr = np.where(np.isclose(std_arr, 0.0), 1.0, std_arr)
        norm_jacobian_arr = jacobian_collection_arr * (
                safe_std_arr[np.newaxis, np.newaxis, :]
                / safe_std_arr[np.newaxis, :, np.newaxis])
        diff_arr = norm_jacobian_arr[1:] - norm_jacobian_arr[:-1]
        raw_signal_arr = np.linalg.norm(
                diff_arr.reshape(diff_arr.shape[0], -1), axis=1) / (num_species ** 2)
        split_time_arr = timecourse_df.index.to_numpy(dtype=float)[1:]
        return self._gaussianSmooth(split_time_arr, raw_signal_arr, self.fit_kernel_bandwidth)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: PASS (10 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/piecewise_system_discovery.py tests/test_piecewise_system_discovery.py
git commit -m "feat: compute Jacobian-based change-point signal"
```

---

### Task 4: Change-point detection (`fit()` step 4)

**Files:**
- Modify: `src/piecewise_system_discovery.py`
- Test: `tests/test_piecewise_system_discovery.py`

**Interfaces:**
- Produces: `PiecewiseSystemDiscovery._detectChangePoints(signal_arr: np.ndarray, num_point: int) -> list[int]`, returning a sorted list of interior split indices (each in `1..num_point-1`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_piecewise_system_discovery.py`:

```python
class TestDetectChangePoints(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery._detectChangePoints."""

    def test_no_change_point_below_threshold(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, change_point_threshold=1e6)
        signal = np.array([0.1, 0.5, 0.2, 0.9, 0.3])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [])

    def test_single_clear_change_point_detected(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=2, change_point_threshold=0.05)
        # num_point=10, candidates are split indices 1..9 (signal indices 0..8).
        # A clear spike at split index 5 (signal index 4).
        signal = np.array([0.01, 0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 0.01])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [5])

    def test_rejects_candidate_violating_min_segment_length(self) -> None:
        """A spike at split index 1 would create a 1-point left segment,
        violating min_segment_length=3; it must be skipped."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=3, change_point_threshold=0.05)
        signal = np.array([1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [])

    def test_stops_at_num_change_point(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=2, change_point_threshold=0.05)
        # Two clear spikes; only the larger (split index 5) should be kept
        # since num_change_point=1.
        signal = np.array([0.01, 0.01, 0.01, 0.8, 1.0, 0.01, 0.01, 0.01, 0.01])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [5])

    def test_two_change_points_sorted_by_time(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=2,
                min_segment_length=2, change_point_threshold=0.05)
        signal = np.array([0.01, 0.01, 0.8, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01])
        result = psd._detectChangePoints(signal, num_point=10)  # pylint: disable=protected-access
        self.assertEqual(result, [3, 6])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py::TestDetectChangePoints -v`
Expected: FAIL with `AttributeError: 'PiecewiseSystemDiscovery' object has no attribute '_detectChangePoints'`

- [ ] **Step 3: Implement**

Add to `src/piecewise_system_discovery.py`, after `_computeChangePointSignal`:

```python
    def _detectChangePoints(self, signal_arr: np.ndarray, num_point: int) -> List[int]:
        """fit() step 4. signal_arr[k] is the signal for split index k+1
        (the time-grid index at which a new segment would begin).

        Returns a sorted (by time) list of accepted interior split indices.
        """
        candidate_index_arr = np.arange(1, num_point)
        order_arr = np.argsort(-signal_arr, kind="stable")
        accepted: List[int] = []
        for rank in order_arr:
            signal_value = signal_arr[rank]
            if signal_value < self.change_point_threshold:
                break
            split_idx = int(candidate_index_arr[rank])
            pos = bisect.bisect_left(accepted, split_idx)
            left_bound = accepted[pos - 1] if pos > 0 else 0
            right_bound = accepted[pos] if pos < len(accepted) else num_point
            if (split_idx - left_bound) < self.min_segment_length:
                continue
            if (right_bound - split_idx) < self.min_segment_length:
                continue
            accepted.insert(pos, split_idx)
            if len(accepted) == self.num_change_point:
                break
        return accepted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: PASS (15 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/piecewise_system_discovery.py tests/test_piecewise_system_discovery.py
git commit -m "feat: detect change points from the smoothed signal"
```

---

### Task 5: `fit()` — wire together scaling, detection, and per-segment models

**Files:**
- Modify: `src/piecewise_system_discovery.py`
- Test: `tests/test_piecewise_system_discovery.py`

**Interfaces:**
- Produces: `PiecewiseSystemDiscovery.fit() -> PiecewiseSystemDiscovery`. Populates `self._segment_models`, `self._segment_boundaries`, `self._segment_lengths`, sets `self._is_fitted = True`, returns `self`. (No `self._scaler` — see Correction note below; the global pre-scaling step was removed.)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_piecewise_system_discovery.py`:

```python
class TestFit(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.fit()."""

    def test_returns_self(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite")
        result = psd.fit()
        self.assertIs(result, psd)

    def test_sets_fitted_flag(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        self.assertTrue(psd._is_fitted)  # pylint: disable=protected-access

    def test_detects_two_segments_for_clear_regime_change(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        self.assertEqual(len(psd._segment_models), 2)  # pylint: disable=protected-access
        self.assertEqual(len(psd._segment_boundaries), 2)  # pylint: disable=protected-access

    def test_segment_boundary_near_split_time(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        boundary_time = psd._segment_boundaries[0][1]  # pylint: disable=protected-access
        self.assertAlmostEqual(boundary_time, _SPLIT_TIME, delta=0.5)

    def test_segments_default_to_is_normalize_true(self) -> None:
        """SystemDiscovery's own default applies when kwargs doesn't override it."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        for model in psd._segment_models:  # pylint: disable=protected-access
            self.assertTrue(model._is_normalize)  # pylint: disable=protected-access

    def test_segments_honor_explicit_is_normalize_override(self) -> None:
        """An explicit is_normalize in kwargs passes through unmodified."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite",
                is_normalize=False).fit()
        for model in psd._segment_models:  # pylint: disable=protected-access
            self.assertFalse(model._is_normalize)  # pylint: disable=protected-access

    def test_segment_coefficients_are_physical_units(self) -> None:
        """Regression guard for the units-mismatch bug: with the default
        is_normalize=True and raw per-segment data, the fitted cross-term
        coefficient (S1 in dS2/dt) must match the true physical-units rate
        constant, not be inflated by a global std(S1)/std(S2) ratio."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        summary_a = psd._segment_models[0].summary()  # pylint: disable=protected-access
        s1_in_ds2dt = float(summary_a.loc["S1", "dS2/dt"])
        self.assertAlmostEqual(s1_in_ds2dt, _RATE_A[0], delta=0.05)

    def test_segment_models_are_fitted(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=10, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        for model in psd._segment_models:  # pylint: disable=protected-access
            self.assertTrue(model._is_fitted)  # pylint: disable=protected-access

    def test_no_change_point_yields_single_segment(self) -> None:
        """A very high threshold rejects all candidates; entire timecourse is one segment."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                change_point_threshold=1e6, poly_degree=1, differentiation="finite").fit()
        self.assertEqual(len(psd._segment_models), 1)  # pylint: disable=protected-access
        start, end = psd._segment_boundaries[0]  # pylint: disable=protected-access
        self.assertAlmostEqual(start, tc.timecourse_df.index[0])
        self.assertAlmostEqual(end, tc.timecourse_df.index[-1])

    def test_all_segments_too_short_yields_single_segment(self) -> None:
        """min_segment_length larger than any achievable segment also collapses
        to a single segment (every candidate gets rejected)."""
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc, num_change_point=1,
                min_segment_length=1000, change_point_threshold=0.05,
                fit_kernel_bandwidth=0.05, poly_degree=1, differentiation="finite").fit()
        self.assertEqual(len(psd._segment_models), 1)  # pylint: disable=protected-access
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py::TestFit -v`
Expected: FAIL with `AttributeError: 'PiecewiseSystemDiscovery' object has no attribute 'fit'`

- [ ] **Step 3: Implement**

Add to `src/piecewise_system_discovery.py`, after `_detectChangePoints`:

```python
    def fit(self) -> "PiecewiseSystemDiscovery":
        """fit() steps 1-4: detect change points, fit per-segment models."""
        raw_df = self.timecourse.timecourse_df
        jacobian_collection_arr = self.timecourse.jacobian_collection_arr
        num_point = raw_df.shape[0]
        time_arr = raw_df.index.to_numpy(dtype=float)

        signal_arr = self._computeChangePointSignal(raw_df, jacobian_collection_arr)
        split_index_list = self._detectChangePoints(signal_arr, num_point)
        boundary_index_arr = [0] + split_index_list + [num_point]

        self._segment_models = []
        self._segment_boundaries = []
        self._segment_lengths = []
        for lo, hi in zip(boundary_index_arr[:-1], boundary_index_arr[1:]):
            segment_df = raw_df.iloc[lo:hi]
            end_time = time_arr[hi] if hi < num_point else time_arr[-1]
            self._segment_boundaries.append((float(time_arr[lo]), float(end_time)))
            self._segment_lengths.append(hi - lo)
            model = SystemDiscovery(segment_df, **self._kwargs).fit()
            self._segment_models.append(model)

        self._is_fitted = True
        return self
```

#### Correction (found during Task 7 review, fixed before Task 8)

The version of `fit()` originally specified here scaled the entire timecourse once with a global `Scaler`, sliced the *scaled* DataFrame per segment, and forced `is_normalize=False` on every segment model. This is a confirmed bug, not a stylistic choice: `is_normalize=False` makes `SystemDiscovery` treat its input as final physical units and never denormalize its output, but the input had actually been pre-scaled by `x / s_i` per species — so every cross-species coefficient came out inflated by the ratio `s_i/s_j` of the two species' *global* standard deviations. Confirmed numerically on the two-regime fixture: `std(S1)/std(S2) = 2.823` over the full combined trajectory, and the fitted cross-term coefficient (S1 in dS2/dt) came out inflated by `2.824` in both segments — self-decay terms fit exactly, since the scaling cancels on the diagonal. `predict()`'s continuous integration (Task 7) then amplified this error over the full time horizon, surfacing as `test_predict_tracks_true_trajectory_reasonably` failing with `max_abs_error=8.48` against a tolerance of `2.0`.

The fix removes the global `self._scaler` / pre-scaling step entirely and fits each segment on its own raw data slice, letting `SystemDiscovery`'s own per-segment `Scaler` (default `is_normalize=True`) normalize-then-denormalize self-consistently. `Scaler` is no longer imported or used in `src/piecewise_system_discovery.py`. This required reopening the already-committed Task 5 (`fit()`) and re-validating Tasks 6 (`predict_derivative`) and 7 (`predict()`), since both depend on segment models producing physical-unit derivatives. See `docs/piecewise_system_discovery.md`'s "Note on a prior design (corrected)" for the doc-level writeup.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: PASS (25 tests total — post-correction; see Correction note below)

- [ ] **Step 5: Commit**

```bash
git add src/piecewise_system_discovery.py tests/test_piecewise_system_discovery.py
git commit -m "feat: implement PiecewiseSystemDiscovery.fit()"
```

---

### Task 6: `predict_derivative(t, x)`

**Files:**
- Modify: `src/piecewise_system_discovery.py`
- Test: `tests/test_piecewise_system_discovery.py`

**Interfaces:**
- Consumes: `self._segment_models`, `self._segment_boundaries` (from Task 5), `SystemDiscovery.predictOneStepDerivative` (from Task 1).
- Produces: `PiecewiseSystemDiscovery.predict_derivative(t: float, x: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_piecewise_system_discovery.py`:

```python
def _fitTwoRegimePsd(**overrides) -> "PiecewiseSystemDiscovery":
    tc = _makeTwoRegimeTimecourse()
    params = dict(num_change_point=1, min_segment_length=10,
            change_point_threshold=0.05, fit_kernel_bandwidth=0.05,
            predict_kernel_bandwidth=0.2, poly_degree=1, differentiation="finite")
    params.update(overrides)
    return PiecewiseSystemDiscovery(tc, **params).fit()


class TestPredictDerivative(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.predict_derivative."""

    def test_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd.predict_derivative(0.0, np.array([10.0, 0.0]))

    def test_returns_correct_shape(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict_derivative(2.0, np.array([8.0, 1.0]))
        self.assertEqual(result.shape, (2,))

    def test_deep_in_segment_a_matches_segment_a_model(self) -> None:
        """Far from the boundary, Gaussian weighting should make the blended
        derivative closely match the nearest (dominant) segment's own
        derivative evaluator."""
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        x = np.array([8.0, 1.0])
        blended = psd.predict_derivative(0.5, x)
        segment_a_only = psd._segment_models[0].predictOneStepDerivative(x)  # pylint: disable=protected-access
        np.testing.assert_allclose(blended, segment_a_only, atol=0.3)

    def test_deep_in_segment_b_matches_segment_b_model(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        x = np.array([2.0, 3.0])
        blended = psd.predict_derivative(9.5, x)
        segment_b_only = psd._segment_models[1].predictOneStepDerivative(x)  # pylint: disable=protected-access
        np.testing.assert_allclose(blended, segment_b_only, atol=0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py::TestPredictDerivative -v`
Expected: FAIL with `AttributeError: 'PiecewiseSystemDiscovery' object has no attribute 'predict_derivative'`

- [ ] **Step 3: Implement**

Add to `src/piecewise_system_discovery.py`, after `fit()`:

```python
    def predict_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        """Blend per-segment derivative predictions at (t, x) with a Gaussian
        kernel over each segment's midpoint. See docs/piecewise_system_discovery.md.
        """
        self._require_fitted()
        midpoint_arr = np.array(
                [0.5 * (start + end) for start, end in self._segment_boundaries])
        weight_arr = np.exp(-0.5 * ((t - midpoint_arr) / self.predict_kernel_bandwidth) ** 2)
        derivative_arr = np.array([
                model.predictOneStepDerivative(x) for model in self._segment_models])
        return (weight_arr[:, np.newaxis] * derivative_arr).sum(axis=0) / weight_arr.sum()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: PASS (27 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/piecewise_system_discovery.py tests/test_piecewise_system_discovery.py
git commit -m "feat: implement PiecewiseSystemDiscovery.predict_derivative"
```

---

### Task 7: `predict(test_df)`

**Files:**
- Modify: `src/piecewise_system_discovery.py`
- Test: `tests/test_piecewise_system_discovery.py`

**Interfaces:**
- Consumes: `self.predict_derivative` (Task 6), `self._segment_models[0].species_names`.
- Produces: `PiecewiseSystemDiscovery.predict(test_df: pd.DataFrame = NULL_DF) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_piecewise_system_discovery.py`:

```python
class TestPredict(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.predict."""

    def test_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd.predict()

    def test_default_predict_returns_dataframe_matching_training_grid(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict()
        self.assertIsInstance(result, pd.DataFrame)
        np.testing.assert_allclose(
                result.index.to_numpy(dtype=float),
                psd.timecourse.timecourse_df.index.to_numpy(dtype=float))

    def test_predict_columns_are_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict()
        self.assertEqual(list(result.columns), ["S1", "S2"])

    def test_predict_starts_at_initial_condition(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict()
        x0 = psd.timecourse.timecourse_df.to_numpy(dtype=float)[0, :]
        np.testing.assert_allclose(result.iloc[0].to_numpy(), x0, atol=1e-6)

    def test_predict_tracks_true_trajectory_reasonably(self) -> None:
        """The blended piecewise prediction should stay within a modest
        absolute tolerance of the true synthetic trajectory."""
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        result = psd.predict()
        true_df = psd.timecourse.timecourse_df
        max_abs_error = (result.to_numpy() - true_df.to_numpy()).__abs__().max()
        self.assertLess(max_abs_error, 2.0)

    def test_predict_with_test_df_uses_its_initial_condition(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        test_df = pd.DataFrame(
                {"S1": [3.0, 2.0], "S2": [1.0, 1.5]}, index=[0.0, 1.0])
        result = psd.predict(test_df)
        np.testing.assert_allclose(result.iloc[0].to_numpy(), [3.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(
                result.index.to_numpy(dtype=float), test_df.index.to_numpy(dtype=float))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py::TestPredict -v`
Expected: FAIL with `AttributeError: 'PiecewiseSystemDiscovery' object has no attribute 'predict'`

- [ ] **Step 3: Implement**

Add to `src/piecewise_system_discovery.py`, after `predict_derivative`:

```python
    def predict(self, test_df: pd.DataFrame = NULL_DF) -> pd.DataFrame:
        """Integrate the blended ODE forward and return predicted concentrations."""
        self._require_fitted()
        if test_df is not NULL_DF:
            x0 = test_df.to_numpy(dtype=float)[0, :]
            time_arr = test_df.index.to_numpy(dtype=float)
        else:
            raw_df = self.timecourse.timecourse_df
            x0 = raw_df.to_numpy(dtype=float)[0, :]
            time_arr = raw_df.index.to_numpy(dtype=float)

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            return self.predict_derivative(t, x)

        sol = solve_ivp(
                rhs,
                t_span=(time_arr[0], time_arr[-1]),
                y0=x0,
                t_eval=time_arr,
                method="Radau",
                rtol=1e-6,
                atol=1e-8,
        )
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")
        species_names = self._segment_models[0].species_names
        return pd.DataFrame(sol.y.T, index=time_arr, columns=species_names)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: PASS (33 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/piecewise_system_discovery.py tests/test_piecewise_system_discovery.py
git commit -m "feat: implement PiecewiseSystemDiscovery.predict"
```

---

### Task 8: `score()`

**Files:**
- Modify: `src/piecewise_system_discovery.py`
- Test: `tests/test_piecewise_system_discovery.py`

**Interfaces:**
- Consumes: `self._segment_models`, `self._segment_lengths` (Task 5), `SystemDiscovery.score() -> ScoreInfo`.
- Produces: `PiecewiseSystemDiscovery.score() -> ScoreInfo`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_piecewise_system_discovery.py`:

```python
class TestScore(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.score."""

    def test_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd.score()

    def test_num_nonzero_term_is_sum_across_segments(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        expected = sum(m.score().num_nonzero_term for m in psd._segment_models)  # pylint: disable=protected-access
        self.assertEqual(psd.score().num_nonzero_term, expected)

    def test_values_length_is_weighted_by_segment_length(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        expected_length = sum(
                len(m.score().values) * length
                for m, length in zip(psd._segment_models, psd._segment_lengths))  # pylint: disable=protected-access
        self.assertEqual(len(psd.score().values), expected_length)

    def test_min_median_max_match_manual_computation(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        weighted_values: list = []
        for model, length in zip(psd._segment_models, psd._segment_lengths):  # pylint: disable=protected-access
            weighted_values.extend(model.score().values * length)
        info = psd.score()
        self.assertAlmostEqual(info.min, min(weighted_values))
        self.assertAlmostEqual(info.max, max(weighted_values))
        self.assertAlmostEqual(info.median, float(np.median(weighted_values)))

    def test_single_segment_score_matches_underlying_model(self) -> None:
        """With no change points, score() should reduce to the single
        segment model's own score()."""
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd(change_point_threshold=1e6)
        underlying = psd._segment_models[0].score()  # pylint: disable=protected-access
        result = psd.score()
        self.assertAlmostEqual(result.min, underlying.min)
        self.assertAlmostEqual(result.max, underlying.max)
        self.assertEqual(result.num_nonzero_term, underlying.num_nonzero_term)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py::TestScore -v`
Expected: FAIL with `AttributeError: 'PiecewiseSystemDiscovery' object has no attribute 'score'`

- [ ] **Step 3: Implement**

Add to `src/piecewise_system_discovery.py`, after `predict`:

```python
    def score(self) -> ScoreInfo:
        """Length-weighted aggregation of per-segment ScoreInfo. See
        docs/piecewise_system_discovery.md `score()` section."""
        self._require_fitted()
        weighted_values: List[float] = []
        num_nonzero_term = 0
        for model, length in zip(self._segment_models, self._segment_lengths):
            info = model.score()
            weighted_values.extend(info.values * length)
            num_nonzero_term += info.num_nonzero_term
        return ScoreInfo(
                min=float(np.min(weighted_values)),
                median=float(np.median(weighted_values)),
                max=float(np.max(weighted_values)),
                values=weighted_values,
                num_nonzero_term=num_nonzero_term,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: PASS (38 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/piecewise_system_discovery.py tests/test_piecewise_system_discovery.py
git commit -m "feat: implement PiecewiseSystemDiscovery.score"
```

---

### Task 9: `printEquations()` / `__str__()`

**Files:**
- Modify: `src/piecewise_system_discovery.py`
- Test: `tests/test_piecewise_system_discovery.py`

**Interfaces:**
- Consumes: `self._segment_models`, `self._segment_boundaries` (Task 5), `str(SystemDiscovery)` (existing).
- Produces: `PiecewiseSystemDiscovery.__str__(self) -> str`, `PiecewiseSystemDiscovery.printEquations(self) -> None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_piecewise_system_discovery.py`:

```python
class TestPrintEquations(unittest.TestCase):
    """Tests for PiecewiseSystemDiscovery.printEquations / __str__."""

    def test_print_equations_raises_before_fit(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTwoRegimeTimecourse()
        psd = PiecewiseSystemDiscovery(tc)
        with self.assertRaises(RuntimeError):
            psd.printEquations()

    def test_str_contains_one_header_per_segment(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        text = str(psd)
        self.assertEqual(text.count("Segment"), len(psd._segment_models))  # pylint: disable=protected-access

    def test_str_contains_segment_time_ranges(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        text = str(psd)
        for start, end in psd._segment_boundaries:  # pylint: disable=protected-access
            self.assertIn(f"{start:.1f}", text)
            self.assertIn(f"{end:.1f}", text)

    def test_str_contains_species_derivative_lines(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        text = str(psd)
        self.assertIn("dS1/dt", text)
        self.assertIn("dS2/dt", text)

    def test_print_equations_runs_without_error(self) -> None:
        if IGNORE_TESTS:
            return
        psd = _fitTwoRegimePsd()
        psd.printEquations()  # smoke test: just confirm no exception
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py::TestPrintEquations -v`
Expected: FAIL — `__str__` falls back to default `object.__repr__`-style output, so `"Segment" in text` assertions fail; `printEquations` raises `AttributeError`.

- [ ] **Step 3: Implement**

Add to `src/piecewise_system_discovery.py`, after `score`:

```python
    def __str__(self) -> str:
        block_list: List[str] = []
        for idx, (model, (start, end)) in enumerate(
                zip(self._segment_models, self._segment_boundaries), start=1):
            header = f"[Segment {idx}: t in [{start:.1f}, {end:.1f})]"
            equation_line_list = [f"  {line}" for line in str(model).strip().split("\n")]
            block_list.append("\n".join([header] + equation_line_list))
        return "\n\n".join(block_list)

    def printEquations(self) -> None:
        """Pretty-print the discovered ODE for each segment."""
        self._require_fitted()
        print(str(self))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source activate.sh && python3 -m pytest tests/test_piecewise_system_discovery.py -v`
Expected: PASS (43 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/piecewise_system_discovery.py tests/test_piecewise_system_discovery.py
git commit -m "feat: implement PiecewiseSystemDiscovery.printEquations and __str__"
```

---

### Task 10: Full-suite regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `source activate.sh && python3 -m pytest tests/ -v`
Expected: All tests PASS (no regressions in unrelated modules)

- [ ] **Step 2: Run pylint on the two touched/created source files**

Run: `source activate.sh && pylint src/system_discovery.py src/piecewise_system_discovery.py`
Expected: No new errors beyond pre-existing baseline (compare against `pylint src/system_discovery.py` on the pre-Task-1 version if uncertain).

- [ ] **Step 3: Commit if anything was fixed**

```bash
git add -A
git commit -m "chore: address lint findings in piecewise system discovery"
```
(Skip this step if Steps 1-2 found nothing to fix.)

---

## Known limitations carried over from the design doc (not fixed by this plan)

- `predict_derivative`'s Gaussian weights are never defensively floored — if `t` is far enough from every segment midpoint relative to `predict_kernel_bandwidth`, all weights can numerically underflow to `0.0`, producing a `0/0` NaN. The design doc doesn't call for guarding against this, so it isn't implemented; flag if `predict()` is used for substantial extrapolation.
- `score()`'s `num_nonzero_term` is a simple sum across segments — a term appearing in multiple segments' equations is counted once per segment, per the doc.
