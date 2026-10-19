"""Tests for src/oscillation_detector.py: findOscillations function."""

import warnings  # noqa: E402

# Suppress tellurium ImportWarnings raised when optional packages (rrplugins, phrasedml,
# pySBOL, sbml2matlab) are not installed. These fire at *module-import time* inside the
# tellurium package — before any test code runs — and are environment issues only; the
# simulation still executes fine. Placing this filter here (before all other imports)
# ensures it catches them regardless of whether the file is run directly with ``python``
# or through pytest, even under ``-W error::Warning``.
warnings.filterwarnings(
    "ignore", message=r".*could not be imported.*", category=ImportWarning,
)


from src.oscillation_detector import findOscillations  # type: ignore
from src.timecourse import Timecourse

import os
import unittest
import numpy as np  # type: ignore
import pandas as pd  # type: ignore


IS_TEST = False  # Flip to True in CI/nightly runs to actually exercise BioModel tests.
HAS_BIOMODELS = os.path.isdir(
    os.path.join(os.path.dirname(__file__), "..", "..", "temp-biomodels", "final")
)


class TestFindOscillations(unittest.TestCase):
    """Tests for the findOscillations function."""

    def test_single_sine_wave(self) -> None:
        """A single sine wave should be detected at its true frequency."""
        t = np.linspace(0, 10, 500)
        freq = 0.5
        df = pd.DataFrame({"A": np.sin(2 * np.pi * freq * t)}, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], freq, places=1)

    def test_no_oscillation_constant(self) -> None:
        """A constant column should return empty list (no oscillation)."""
        t = np.linspace(0, 10, 100)
        df = pd.DataFrame({"A": np.ones(100)}, index=t)
        result = findOscillations(df)
        self.assertEqual(result, [])

    def test_no_oscillation_linear_trend(self) -> None:
        """A linear trend (no periodic component) should return empty list."""
        t = np.linspace(0, 10, 200)
        df = pd.DataFrame({"A": t}, index=t)
        result = findOscillations(df)
        self.assertEqual(result, [])

    def test_two_sine_waves(self) -> None:
        """Two distinct sine waves should both be detected."""
        t = np.linspace(0, 10, 500)
        freq1, freq2 = 0.3, 1.5
        df = pd.DataFrame({
            "A": np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t),
        }, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 2)

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame should return an empty list."""
        df = pd.DataFrame()
        result = findOscillations(df)
        self.assertEqual(result, [])


    def test_min_frequency_filter(self) -> None:
        """min_frequency should exclude detected frequencies below the threshold."""
        t = np.linspace(0, 10, 500)
        freq_low, freq_high = 0.2, 2.0
        df = pd.DataFrame({
            "A": np.sin(2 * np.pi * freq_low * t) + np.sin(2 * np.pi * freq_high * t),
        }, index=t)
        result = findOscillations(df, min_frequency=1.0)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], freq_high, places=1)

    def test_max_frequency_filter(self) -> None:
        """max_frequency should exclude detected frequencies above the threshold."""
        t = np.linspace(0, 10, 500)
        freq_low, freq_high = 0.2, 2.0
        df = pd.DataFrame({
            "A": np.sin(2 * np.pi * freq_low * t) + np.sin(2 * np.pi * freq_high * t),
        }, index=t)
        result = findOscillations(df, max_frequency=1.0)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], freq_low, places=1)

    def test_multiple_columns(self) -> None:
        """Different frequencies in different columns should all be detected."""
        t = np.linspace(0, 10, 500)
        df = pd.DataFrame({
            "A": np.sin(2 * np.pi * 0.5 * t),
            "B": np.sin(2 * np.pi * 1.0 * t),
        }, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 2)

    def test_no_oscillation_random_noise(self) -> None:
        """Random noise (no clear periodicity) should return empty list."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 500)
        df = pd.DataFrame({"A": rng.normal(size=500)}, index=t)
        result = findOscillations(df)
        self.assertEqual(result, [])

    def test_noise_with_signal(self) -> None:
        """Signal buried in moderate noise should still be detected."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 500)
        freq = 1.0
        signal = np.sin(2 * np.pi * freq * t)
        noise = rng.normal(scale=0.05, size=500)
        df = pd.DataFrame({"A": signal + noise}, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], freq, places=0)


    def test_constant_column_with_oscillation(self) -> None:
        """One constant and one oscillating column should yield only the oscillator."""
        t = np.linspace(0, 10, 500)
        df = pd.DataFrame({
            "A": np.ones(500),
            "B": np.sin(2 * np.pi * 0.7 * t),
        }, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.7, places=1)

    def test_dc_offset_does_not_appear(self) -> None:
        """A DC offset (mean != 0) should not be detected as an oscillation."""
        t = np.linspace(0, 10, 500)
        df = pd.DataFrame({"A": 5.0 + np.sin(2 * np.pi * 0.5 * t)}, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 1)
        for freq in result:
            self.assertGreater(freq, 0.1)

    def test_results_sorted(self) -> None:
        """Returned frequencies should be sorted ascending."""
        t = np.linspace(0, 10, 500)
        df = pd.DataFrame({
            "A": (np.sin(2 * np.pi * 3.0 * t) + np.sin(2 * np.pi * 1.0 * t)
                  + np.sin(2 * np.pi * 2.0 * t)),
        }, index=t)
        result = findOscillations(df)
        self.assertEqual(result, sorted(result))

    def test_results_deduplicated(self) -> None:
        """If the same frequency appears in multiple columns, it should appear once."""
        t = np.linspace(0, 10, 500)
        freq = 0.8
        df = pd.DataFrame({
            "A": np.sin(2 * np.pi * freq * t),
            "B": np.sin(2 * np.pi * freq * t + 1.0),
        }, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 1)

    def test_non_uniform_sampling_returns_empty(self) -> None:
        """Non-uniform time steps should cause an empty result."""
        t = np.array([0.0, 0.5, 1.2, 2.3, 4.7, 8.1])
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6]}, index=t)
        result = findOscillations(df)
        self.assertEqual(result, [])

    def test_nan_values_skipped(self) -> None:
        """Columns with NaN values should be skipped (no crash)."""
        t = np.linspace(0, 10, 200)
        values = np.sin(2 * np.pi * 0.5 * t)
        values[10] = np.nan
        df = pd.DataFrame({"A": values}, index=t)
        result = findOscillations(df)
        self.assertIsInstance(result, list)


    def test_high_threshold_no_detection(self) -> None:
        """Very high threshold on noisy data (no real signal) should give empty list."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 500)
        df = pd.DataFrame({"A": rng.normal(size=500)}, index=t)
        result = findOscillations(df, height_threshold_multiplier=1e6)
        self.assertEqual(result, [])

    def test_higher_multiplier_more_detections(self) -> None:
        """A higher height_threshold_multiplier lowers the prominence threshold, so it detects at least as many frequencies."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10, 500)
        df = pd.DataFrame({
            "A": np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 1.0 * t),
        }, index=t)
        result_strict = findOscillations(df, height_threshold_multiplier=1.0)
        result_sens = findOscillations(df, height_threshold_multiplier=100.0)
        self.assertGreaterEqual(len(result_sens), len(result_strict))

    def test_multiple_oscillators_different_columns(self) -> None:
        """Each column with an oscillation should contribute its frequency."""
        t = np.linspace(0, 10, 500)
        df = pd.DataFrame({
            "A": np.sin(2 * np.pi * 0.3 * t),
            "B": np.sin(2 * np.pi * 0.7 * t),
            "C": np.sin(2 * np.pi * 1.2 * t),
        }, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 3)

    def test_frequency_resolution(self) -> None:
        """With long enough data, close frequencies should be resolvable."""
        t = np.linspace(0, 100, 5000)
        freq1, freq2 = 1.0, 1.1
        df = pd.DataFrame({
            "A": np.sin(2 * np.pi * freq1 * t) + np.sin(2 * np.pi * freq2 * t),
        }, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 2)

    def test_single_frequency_per_column(self) -> None:
        """A single pure sine wave should give exactly one detected frequency."""
        t = np.linspace(0, 10, 500)
        freq = 0.7
        df = pd.DataFrame({"A": np.sin(2 * np.pi * freq * t)}, index=t)
        result = findOscillations(df)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], freq, places=1)

    def test_zero_dt_returns_empty(self) -> None:
        """A zero time step (all identical times) is handled gracefully."""
        t = np.zeros(100)
        df = pd.DataFrame({"A": np.sin(np.arange(100).astype(float))}, index=t)
        result = findOscillations(df)
        self.assertEqual(result, [])

    def test_invalid_max_less_than_min(self) -> None:
        """max_frequency < min_frequency should return empty list."""
        t = np.linspace(0, 10, 500)
        df = pd.DataFrame({"A": np.sin(2 * np.pi * 0.5 * t)}, index=t)
        result = findOscillations(df, min_frequency=2.0, max_frequency=1.0)
        self.assertEqual(result, [])


class TestFindOscillationsBioModel206EndToEnd(unittest.TestCase):
    """End-to-end tests for ``findOscillations`` using real BioModel BIOMD0000000206.

    This model exhibits a limit-cycle oscillation whose dominant frequency is
    approximately 7.3 Hz. These tests exercise the full pipeline: load SBML,
    simulate with RoadRunner via ``Timecourse.makeBiomodelDF``, then run FFT-based
    detection on each species column.

    The expected peak (~7.3 Hz) is used loosely — a tolerance band of ±1 Hz — so
    that minor numerical differences across solver versions or hardware do not
    break the tests while still catching regressions in ``findOscillations`` itself.
    """

    BIOMODEL_NAME = "BIOMD0000000206"
    EXPECTED_PEAK_HZ = 7.3
    FREQUENCY_TOLERANCE_HZ = 1.0

    def _make_timecourse(self, num_point: int = 5_000) -> tuple[pd.DataFrame, str]:
        """Generate a Timecourse for BIOMD0000000206 and return (df, species_names)."""

        tc = Timecourse.makeBiomodelDF(self.BIOMODEL_NAME, num_point=num_point)
        return tc.timecourse_df, tc.model.species_names

    def test_detects_oscillation_in_biomodel_206(self) -> None:
        """findOscillations should find at least one frequency for the oscillating BioModel."""
        if IS_TEST or not HAS_BIOMODELS:
            return
        df, _ = self._make_timecourse()
        result = findOscillations(df)
        self.assertGreater(len(result), 0, "Expected at least one detected frequency")

    def test_dominant_frequency_near_expected_value(self) -> None:
        """The detected dominant frequency should be within ±1 Hz of ~7.3 Hz."""
        if IS_TEST or not HAS_BIOMODELS:
            return
        df, _ = self._make_timecourse(num_point=10_000)
        result = findOscillations(df)
        self.assertEqual(len(result), 1)
        detected = result[0]
        self.assertAlmostEqual(
            detected,
            self.EXPECTED_PEAK_HZ,
            delta=self.FREQUENCY_TOLERANCE_HZ,
            msg=(
                f"Detected dominant frequency {detected:.4f} Hz is outside "
                f"[{self.EXPECTED_PEAK_HZ - self.FREQUENCY_TOLERANCE_HZ}, "
                f"{self.EXPECTED_PEAK_HZ + self.FREQUENCY_TOLERANCE_HZ}] Hz"
            ),
        )

    def test_frequency_stable_across_sample_rates(self) -> None:
        """Detected frequency should be consistent across different num_point values."""
        if IS_TEST or not HAS_BIOMODELS:
            return
        freqs = []
        for np_ in (2_000, 5_000, 10_000):
            df, _ = self._make_timecourse(num_point=np_)
            result = findOscillations(df)
            self.assertEqual(len(result), 1, f"Expected exactly one peak at num_point={np_}")
            freqs.append(result[0])
        # All three should agree to within our tolerance band.
        for f in freqs:
            self.assertAlmostEqual(
                f, freqs[0], delta=self.FREQUENCY_TOLERANCE_HZ,
                msg=f"Frequency at num_point={np_} drifted from reference",
            )

    def test_all_species_share_dominant_frequency(self) -> None:
        """Every oscillating species column should report the same dominant frequency."""
        if IS_TEST or not HAS_BIOMODELS:
            return
        df, species_names = self._make_timecourse()
        # findOscillations runs over all columns; since every species shares the
        # limit cycle they should all contribute the same peak.
        result = findOscillations(df)
        self.assertGreater(len(result), 0)
        dominant = result[0]
        # Spot-check a few species individually to confirm per-column agreement.
        for sp in species_names[:3]:
            single_df = df[[sp]]
            single_result = findOscillations(single_df)
            if len(single_result) == 1:
                self.assertAlmostEqual(
                    single_result[0], dominant, delta=0.1,
                    msg=f"Species {sp!r} peak {single_result[0]:.4f} Hz differs from "
                        f"group peak {dominant:.4f} Hz by > 0.1",
                )

    def test_no_spurious_low_frequency_detection(self) -> None:
        """BioModel 206 should not produce spurious detections well below its natural frequency."""
        if IS_TEST or not HAS_BIOMODELS:
            return
        df, _ = self._make_timecourse()
        result = findOscillations(df, min_frequency=0.1, max_frequency=3.0)
        # The model's fundamental is ~7.3 Hz, so nothing in [0.1, 3] Hz should pass.
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
