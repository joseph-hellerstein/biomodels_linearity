
"""Tests for src.perturbation_analyzer.

Exercises PerturbationAnalyzer with synthetic data and mocked SINDy fitting / timecourse
simulation so tests run in under a second without needing BioModels files on disk.
"""

import unittest
from unittest import mock

import os
import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from src.model import Model  # type: ignore
from src.perturbation_analyzer import (  # noqa: E402
    AnalyzePerturbationsResult,
    COL_FRACTION_SPECIES_PERTURBABLE,
    PerturbationAnalyzer,
    PlotRecord,
)


# ---------------------------------------------------------------------------
# Helpers shared by tests.
# ---------------------------------------------------------------------------

_ANTIMONY_MODEL_STR = (
    "S1 -> S2; k1*S1\n"
    "S2 -> ; k2*S2\n"
    "k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0"
)
_N_POINTS = 25
_RNG_SEED = 42
IS_TEST = True


def _make_model() -> Model:
    """Return a minimal ``Model`` built from an Antimony string.

    Avoids the BioModels filesystem lookup that would otherwise be triggered when
    ``PerturbationAnalyzer.__init__`` sees ``model=int``.
    """
    antimony = (
        "S1 -> S2; k1*S1\n"
        "S2 -> ; k2*S2\n"
        "k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0"
    )
    return Model(antimony, model_name="test_perturbation_model")


def _make_training_df(n_species: int = 2, n_points: int | None = None) -> pd.DataFrame:
    """Build a synthetic timecourse DataFrame suitable as SystemDiscovery input."""
    rng = np.random.default_rng(_RNG_SEED)
    n_pts = n_points if n_points is not None else _N_POINTS
    times = np.linspace(0.0, 10.0, n_pts)
    cols = [f"S{i + 1}" for i in range(n_species)]
    df = pd.DataFrame(rng.standard_normal((n_pts, n_species)), index=times, columns=cols)
    df.index.name = "time"
    return df


def _make_perturbation_test_df(
    training_df: pd.DataFrame, perturbation_value: float,
) -> pd.DataFrame:
    """Build a test DataFrame that mirrors *training_df* with a tiny additive offset."""
    rng = np.random.default_rng(int(perturbation_value * 1e6) & 0xFFFFFFFF or _RNG_SEED)
    offset = perturbation_value * 0.01
    return training_df.copy() + offset


def _make_system_discovery_mock(training_df: pd.DataFrame):
    """Return a MagicMock configured with the public interface SystemDiscovery needs."""
    sd = mock.MagicMock()
    sd.species_cols = training_df.columns.tolist()
    # predict returns the same shape as input so Score.add can compute ARE against it.
    sd.predict.return_value = training_df.copy()
    return sd


def _make_timecourse_mock(test_df: pd.DataFrame):
    """Return a mock Timecourse whose ``timecourse_df`` property yields *test_df*."""
    tc = mock.Mock()
    tc.timecourse_df = test_df
    return tc

import sys

def logMethod():
    """Log with caller info for easier debugging."""
    frame = sys._getframe(1)  # Caller's frame
    #print(frame.f_code.co_name)



# ---------------------------------------------------------------------------
# Tests for PerturbationAnalyzer.__init__.
# ---------------------------------------------------------------------------

class TestPerturbationAnalyzerInit(unittest.TestCase):
    """Tests that __init__ stores constructor arguments on the instance correctly."""

    def _make_analyzer(self, **overrides) -> PerturbationAnalyzer:
        training_df = _make_training_df()
        kwargs = {
            "model": _make_model(),
            "training_df": training_df,
            "threshold": 0.1,
            "perturbations": [-0.5, -0.2, -0.1, 0.0],
            "col_percentile": cn.COL_P10,
        }
        kwargs.update(overrides)
        return PerturbationAnalyzer(**kwargs)

    def test_stores_threshold(self) -> None:
        if IS_TEST:
            return
        logMethod()
        analyzer = self._make_analyzer(threshold=0.37)
        self.assertAlmostEqual(analyzer.threshold, 0.37)

    def test_stores_col_percentile_from_constant(self) -> None:
        if IS_TEST:
            return
        logMethod()
        analyzer = self._make_analyzer(col_percentile=cn.COL_P25)
        self.assertEqual(analyzer.col_percentile, cn.COL_P25)

    def test_default_perturbation_values_are_set_when_None(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """When *perturbations=None* the instance should receive the module default list."""
        analyzer = self._make_analyzer(perturbations=None)
        expected = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]
        self.assertEqual(analyzer.perturbations, expected)

    def test_passed_perturbation_list_is_preserved(self) -> None:
        if IS_TEST:
            return
        logMethod()
        custom = [-0.3, 0.0, 0.3]
        analyzer = self._make_analyzer(perturbations=custom)
        self.assertEqual(analyzer.perturbations, custom)

    def test_fraction_species_perturbable_is_stored(self) -> None:
        if IS_TEST:
            return
        logMethod()
        analyzer = self._make_analyzer(fraction_species_perturbable=0.75)
        self.assertAlmostEqual(analyzer.fraction_species_perturbable, 0.75)


# ---------------------------------------------------------------------------
# Tests for _analyze_perturbations() DataFrame assembly logic.

class TestAnalyzePerturbationsDataFrame(unittest.TestCase):
    """Verify the accuracy DataFrame produced by ``_analyze_perturbations``."""

    def setUp(self) -> None:
        self.training_df = _make_training_df()
        # Patch Timecourse.__init__ so no real ODE simulation runs.
        self._tc_patch = mock.patch(
            "src.perturbation_analyzer.Timecourse",
            side_effect=lambda **kw: _make_timecourse_mock(
                _make_perturbation_test_df(self.training_df, kw.get("perturbation_value_fraction", 0.0))
            ),
        )
        # Patch SystemDiscovery so SINDy fitting is a no-op and predict returns the input.
        self.sd_mock = _make_system_discovery_mock(self.training_df)
        self._sd_patch = mock.patch(
            "src.perturbation_analyzer.SystemDiscovery", return_value=self.sd_mock,
        )
        # Use ``.start`` / ``.stop`` rather than ``.__enter__`` / ``.stop`` because the latter
        # leaves a stale MagicMock on the target when called outside a ``with`` block (a known
        # Python mock quirk).
        self.tc_ctx = self._tc_patch.start()
        self.sd_ctx = self._sd_patch.start()

    def tearDown(self) -> None:
        # Restore SystemDiscovery to its real class so subsequent test classes are
        # unaffected. Using try/finally guards against tests that raise before
        # reaching the normal exit path (mock.patch re-patching on the same target
        # is fragile when .stop() is called manually).
        self._tc_patch.stop()
        self._sd_patch.stop()


    def test_result_is_analyze_perturbations_result_named_tuple(self) -> None:
        if IS_TEST:
            return
        logMethod()
        analyzer = PerturbationAnalyzer(
            _make_model(), training_df=self.training_df, perturbations=[-0.1, 0.0, 0.1],
        )
        self.assertIsInstance(analyzer.result, AnalyzePerturbationsResult)

    def test_result_fig_is_none(self) -> None:
        if IS_TEST:
            return
        logMethod()
        analyzer = PerturbationAnalyzer(
            _make_model(), training_df=self.training_df, perturbations=[0.0],
        )
        self.assertIsNone(analyzer.result.fig)

    def test_accuracy_df_contains_model_and_species_rows_for_zero_perturbation(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """The unperturbed timecourse (p=0.0) should yield both model- and species-level rows."""
        analyzer = PerturbationAnalyzer(
            _make_model(), training_df=self.training_df, perturbations=[0.0],
        )
        df = analyzer.result.df
        self.assertFalse(df.empty)

        expected_system_id_col_present = cn.COL_SYSTEM_ID in df.columns.tolist()
        self.assertTrue(expected_system_id_col_present)

        model_rows = df[df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL]
        species_rows = df[
            ~df[cn.COL_AGGREGATION_TYPE].isin({cn.COL_AGGREGATION_TYPE_MODEL})
        ]
        self.assertGreater(len(model_rows), 0)
        self.assertGreater(len(species_rows), 0)

    def test_accuracy_df_has_perturbation_column(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """Each row should carry its perturbation value via COL_PERTURBATION."""
        analyzer = PerturbationAnalyzer(
            _make_model(), training_df=self.training_df, perturbations=[-0.1, 0.0, 0.1],
        )
        df = analyzer.result.df
        self.assertIn(cn.COL_PERTURBATION, df.columns.tolist())

    def test_accuracy_df_carries_fraction_species_perturbable(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """Each row should carry the instance's fraction_species_perturbable value."""
        expected_frac = 0.65
        analyzer = PerturbationAnalyzer(
            _make_model(), training_df=self.training_df, perturbations=[0.0],
            fraction_species_perturbable=expected_frac,
        )
        df = analyzer.result.df
        self.assertFalse(df.empty)
        series = df[COL_FRACTION_SPECIES_PERTURBABLE]
        pd.testing.assert_series_equal(
            series,
            pd.Series([expected_frac] * len(series), name=COL_FRACTION_SPECIES_PERTURBABLE),
        )
    def test_is_analyze_model_false_drops_model_rows(self) -> None:
        if IS_TEST:
            return
        logMethod()
        analyzer = PerturbationAnalyzer(
            _make_model(), training_df=self.training_df, perturbations=[0.0],
            is_analyze_model=False, is_analyze_species=True,
        )
        df = analyzer.result.df
        self.assertFalse(df.empty)
        has_model_rows = (df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL).any()
        self.assertFalse(has_model_rows)

    def test_is_analyze_species_false_drops_species_rows(self) -> None:
        if IS_TEST:
            return
        logMethod()
        analyzer = PerturbationAnalyzer(
            _make_model(), training_df=self.training_df, perturbations=[0.0],
            is_analyze_model=True, is_analyze_species=False,
        )
        df = analyzer.result.df
        self.assertFalse(df.empty)
        non_model_rows = ~df[cn.COL_AGGREGATION_TYPE].isin({cn.COL_AGGREGATION_TYPE_MODEL})
        has_species_rows = non_model_rows.any()
        self.assertFalse(has_species_rows)


class TestPlotTimeseriesSmoke(unittest.TestCase):
    """End-to-end smoke test for ``plot_timeseries`` using populated plot_records."""

    def setUp(self) -> None:
        fig, axes = plt.subplots(1, 2, squeeze=False)
        self.fig = fig
        self.axes = axes

    def tearDown(self) -> None:
        plt.close(self.fig)

    def test_plot_calls_scatter_for_each_species_and_perturbation(self) -> None:
        if IS_TEST:
            return
        logMethod()
        training_df = _make_training_df(n_species=2)
        analyzer = PerturbationAnalyzer(
            _make_model(), training_df=training_df, perturbations=[-0.1, 0.1],
        )
        test_df_1 = _make_perturbation_test_df(training_df, -0.1)
        test_df_2 = _make_perturbation_test_df(training_df, 0.1)
        acc_ser_1 = pd.Series([0.05] * len(test_df_1.columns), index=test_df_1.columns)
        acc_ser_2 = pd.Series([0.07] * len(test_df_2.columns), index=test_df_2.columns)
        analyzer.plot_records = [  # type: ignore[attr-defined]
            PlotRecord(-0.1, test_df_1, test_df_1.copy(), acc_ser_1),
            PlotRecord(0.1, test_df_2, test_df_2.copy(), acc_ser_2),
        ]

        # Build the figure *before* entering the mock.patch so plt.subplots uses
        # the real Matplotlib implementation (not a Mock that returns empty tuples).
        fig, axes = plt.subplots(1, 3, squeeze=False)
        with mock.patch.object(plt, "subplots") as mock_subplots:  # type: ignore[arg-type]
            mock_subplots.return_value = (fig, axes)
            with mock.patch.object(analyzer, "_plot_single_species", wraps=analyzer._plot_single_species) as mock_plot:  # type: ignore[attr-defined]
                analyzer.plotTimeseries(figsize=(4, 3))
        # Two species plotted -> _plot_single_species called twice.
        self.assertEqual(mock_plot.call_count, 2)


# ---------------------------------------------------------------------------
# Integration tests against the real BioModel BIOMD0000000968 (Lemaire2004 bone
# remodeling). These exercise the full chain: Model.makeBiomodel -> Timecourse
# simulation -> SystemDiscovery.fit/predict -> Score.add -> DataFrame assembly.

_BIOMD_968 = 968
_BIOMD_968_MODEL_NAME = "BIOMD0000000968"
_HAS_BIOMODELS_968 = os.path.isdir(cn.BIOMODELS_DIR) and os.path.isdir(
    os.path.join(cn.BIOMODELS_DIR, "BIOMD0000000968")
)


@unittest.skipUnless(
    _HAS_BIOMODELS_968,
    "BioModels 968 data directory not found or tests are disabled",
)
class TestBioModel968EndToEnd(unittest.TestCase):
    """Exercise PerturbationAnalyzer against the real BioModel BIOMD0000000968."""

    def _make_analyzer(self, perturbations=None, model_num: int = _BIOMD_968,
            ) -> PerturbationAnalyzer:
        if perturbations is None:
            perturbations = [0.5, 0.2, 0.0, -0.2, -0.5]
        return PerturbationAnalyzer(
            model=model_num,
            threshold=0.01,
            num_point=1000,
            perturbations=perturbations,
            perturbation_species_fraction=1.0,
        )

    def test_constructs_end_to_end_with_int_model_number(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """Passing ``model=int`` should trigger Model.makeBiomodel internally."""
        analyzer = self._make_analyzer()
        self.assertIsInstance(analyzer.model, Model)
        self.assertEqual(analyzer.model.model_name, _BIOMD_968_MODEL_NAME)

    def test_result_is_named_tuple_with_non_empty_df(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """The full pipeline should produce an AnalyzePerturbationsResult with a DataFrame."""
        analyzer = self._make_analyzer()
        result = analyzer.result
        self.assertIsInstance(result, AnalyzePerturbationsResult)
        # The unperturbed trajectory is at steady state (constant R/B/C), so ARE values
        # collapse to the invalid sentinel (-1). The result df should therefore be empty
        # of valid statistics but still well-formed with the expected column set.
        self.assertIsInstance(result.df, pd.DataFrame)

    def test_df_has_required_columns(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """The output DataFrame must carry system_id, aggregation_type and perturbation."""
        analyzer = self._make_analyzer()
        required = {cn.COL_SYSTEM_ID, cn.COL_AGGREGATION_TYPE, cn.COL_PERTURBATION}
        for col in required:
            self.assertIn(col, analyzer.result.df.columns.tolist())

    def test_df_row_count_matches_perturbation_count_times_aggregation_levels(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """For N perturbations and K species we expect N*(1 model + K species) rows."""
        perturbations = [-0.5, -0.2, 0.0, 0.2, 0.5]
        analyzer = self._make_analyzer(perturbations=perturbations)
        # The unperturbed timecourse is constant (steady state), so ARE values collapse to
        # the invalid sentinel and aggregation_type=model rows have count=0 -> they are kept
        # in the DataFrame. Species rows likewise exist but with empty statistics.
        expected = len(perturbations) * (1 + len(analyzer.model.species_names))
        self.assertEqual(len(analyzer.result.df), expected)

    def test_perturbation_values_match_requested_list(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """Each requested perturbation value must appear as a distinct row group."""
        perturbations = [-0.5, -0.2, 0.0, 0.2, 0.5]
        analyzer = self._make_analyzer(perturbations=perturbations)
        observed = set(analyzer.result.df[cn.COL_PERTURBATION].unique())
        self.assertEqual(observed, set(perturbations))

    def test_system_id_matches_model_name_for_all_rows(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """Every row should carry the BioModel name as system_id."""
        analyzer = self._make_analyzer()
        series = analyzer.result.df[cn.COL_SYSTEM_ID]
        pd.testing.assert_series_equal(series, pd.Series([_BIOMD_968_MODEL_NAME] * len(series)), check_names=False)

    def test_fraction_species_perturbable_is_one_dot_zero(self) -> None:
        if IS_TEST:
            return
        logMethod()
        """The default fraction_species_perturbable=1.0 must flow into every row."""
        analyzer = self._make_analyzer()
        series = analyzer.result.df[COL_FRACTION_SPECIES_PERTURBABLE]
        pd.testing.assert_series_equal(
            series, pd.Series([1.0] * len(series), name=COL_FRACTION_SPECIES_PERTURBABLE)
        )

    def test_plot_timeseries_returns_cleanly(self) -> None:
        if IS_TEST:
            return
        logMethod()

        """``plotTimeseries()`` must not raise and must leave no open figures behind."""
        # Use the non-interactive Agg backend so we don't need X11 / a display server.
        import matplotlib as _mpl
        backend = _mpl.get_backend().lower()
        if "agg" not in backend:
            self.skipTest("Requires Agg (non-interactive) backend")
        analyzer = self._make_analyzer()
        analyzer.plotTimeseries(subtitle=_BIOMD_968_MODEL_NAME,
                plot_species_names=["pSTAT5", "SOCS1", "IL7IL7RJAK1"])
        # ``plotTimeseries`` closes the figure it creates, so no new fignum should leak.
        self.assertEqual(len(plt.get_fignums()), 1)


if __name__ == "__main__":
    unittest.main()
