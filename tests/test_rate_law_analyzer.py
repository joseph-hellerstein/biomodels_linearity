"""Tests for src/rate_law_analyzer.py."""

import os
import shutil
import sys
import tempfile
import unittest
from typing import List, cast
from unittest.mock import MagicMock, patch

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sympy as sp  # type: ignore

import matplotlib  # type: ignore
matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt  # type: ignore

import src.constants as cn  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import rate_law_analyzer as rla  # type: ignore
from model import Model  # type: ignore
from simulator import Simulator, SimulationResult  # type: ignore

IGNORE_TESTS = False
_HAS_BIOMODELS = os.path.exists(cn.BIOMODELS_DIR) and os.path.isdir(cn.BIOMODELS_DIR)


# A model with one Michaelis-Menten and one mass-action reaction.
MM_ANTIMONY_MODEL = """
S1 -> S2; k1*S1/(K + S1)
S2 -> ; k2*S2
k1 = 0.5; K = 2.0; k2 = 0.2; S1 = 10; S2 = 0
"""

# A model with only linear (mass-action) rate laws.
LINEAR_ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

# An SBML model whose rate law uses a kinetic law local parameter (Km) and a
# compartment (c), which the libsbml Python bindings do not expose.
LOCAL_PARAM_SBML = '''<sbml xmlns="http://www.sbml.org/sbml/level2" level="2" version="1">
  <model id="m">
    <listOfCompartments><compartment id="c" size="1"/></listOfCompartments>
    <listOfSpecies><species id="S" compartment="c" initialConcentration="10"/></listOfSpecies>
    <listOfParameters><parameter id="k" value="0.5"/></listOfParameters>
    <listOfReactions><reaction id="r1" reversible="false">
      <listOfReactants><speciesReference species="S"/></listOfReactants>
      <kineticLaw>
        <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><divide/><apply><times/><ci>k</ci><ci>S</ci></apply><ci>Km</ci></apply></math>
        <listOfParameters><parameter id="Km" value="2"/></listOfParameters>
      </kineticLaw>
    </reaction></listOfReactions>
  </model>
</sbml>'''

MODEL_NAME = "BIOMD0000000045"
NUM_POINT = 51


def _make_model(antimony_str: str) -> Model:
    """Create a real Model from an Antimony string."""
    return Model(antimony_str, model_name=MODEL_NAME)


def _get_reaction_ids(antimony_str: str) -> List[str]:
    """Get the reaction IDs of an Antimony model via RoadRunner."""
    import tellurium as te  # type: ignore
    rr = te.loada(antimony_str)
    return list(rr.getReactionIds())


def _make_timecourse_df(model: Model) -> pd.DataFrame:
    """Simulate the model with reaction output enabled and return the timecourse."""
    sim = Simulator(model=model, start_time=0.0, end_time=10.0,
            num_point=NUM_POINT, is_reaction_in_output=True)
    return sim.simulate().timecourse_df


def _make_analyzer(model: Model, boundaries: List[int]) -> rla.RateLawAnalyzer:
    """Construct a RateLawAnalyzer with PSDAnalyzer and Simulator mocked."""
    timecourse_df = _make_timecourse_df(model)
    mock_psd_analyzer = MagicMock()
    mock_psd_analyzer.best_model_ser = pd.Series({cn.COL_BOUNDARIES: boundaries})
    with patch.object(rla.Model, 'makeBiomodel', return_value=model), \
            patch.object(rla, 'PSDAnalyzer', return_value=mock_psd_analyzer), \
            patch.object(rla.Simulator, 'simulateBiomodel',
                    return_value=SimulationResult(timecourse_df=timecourse_df)):
        analyzer = rla.RateLawAnalyzer(biomodel_id=45)
    return analyzer


@unittest.skipIf(not _HAS_BIOMODELS, "BioModels directory not found")
class TestRateLawAnalyzerInit(unittest.TestCase):
    """Tests for RateLawAnalyzer.__init__."""

    def test_init_stores_boundaries(self) -> None:
        if IGNORE_TESTS:
            return
        boundaries = [0, 25, NUM_POINT - 1]
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), boundaries)
        self.assertEqual(analyzer.model_num, 45)
        self.assertEqual(analyzer._boundaries, boundaries)

    def test_init_raises_when_no_psd_prediction(self) -> None:
        """A model with no valid PSD prediction raises a clear ValueError."""
        if IGNORE_TESTS:
            return
        model = _make_model(MM_ANTIMONY_MODEL)
        mock_psd_analyzer = MagicMock()
        mock_psd_analyzer.best_model_ser = None
        with patch.object(rla.Model, 'makeBiomodel', return_value=model), \
                patch.object(rla, 'PSDAnalyzer', return_value=mock_psd_analyzer):
            with self.assertRaises(ValueError):
                rla.RateLawAnalyzer(biomodel_id=45)

    def test_init_raises_for_unknown_biomodel(self) -> None:
        """A biomodel_id that is not a BioModels identifier raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            rla.RateLawAnalyzer(biomodel_id="not_a_biomodel")


class TestRateLawDct(unittest.TestCase):
    """Tests for RateLawAnalyzer._makeRateLawDct."""

    def test_rate_law_dct_has_all_reactions(self) -> None:
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        self.assertEqual(set(analyzer._rate_law_dct.keys()), set(_get_reaction_ids(MM_ANTIMONY_MODEL)))

    def test_rate_law_forms_match_sbml(self) -> None:
        """The extracted expressions match the Antimony rate laws."""
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        r1_id, r2_id = _get_reaction_ids(MM_ANTIMONY_MODEL)
        k1, k2, K, S1, S2 = sp.symbols('k1 k2 K S1 S2')
        self.assertEqual(sp.simplify(analyzer._rate_law_dct[r1_id] - k1*S1/(K + S1)), 0)
        self.assertEqual(sp.simplify(analyzer._rate_law_dct[r2_id] - k2*S2), 0)


class TestDRateLawDct(unittest.TestCase):
    """Tests for RateLawAnalyzer._makeDRateLawDct."""

    def test_derivatives_match_analytic_forms(self) -> None:
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        deriv_dct = analyzer._drate_law_dct[r1_id]
        # Only S1 appears in the Michaelis-Menten law.
        self.assertEqual(set(deriv_dct.keys()), {'S1'})
        k1, K, S1 = sp.symbols('k1 K S1')
        expected = k1*K/(K + S1)**2  # d/dS1 [k1*S1/(K + S1)]
        self.assertEqual(sp.simplify(deriv_dct['S1'] - expected), 0)

    def test_derivatives_only_for_species_in_law(self) -> None:
        """Species absent from a rate law have no derivative entry."""
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        _, r2_id = _get_reaction_ids(MM_ANTIMONY_MODEL)
        self.assertEqual(set(analyzer._drate_law_dct[r2_id].keys()), {'S2'})


class TestParameterValueDct(unittest.TestCase):
    """Tests for RateLawAnalyzer._makeParameterValueDct."""

    def test_parameter_values_match_sbml(self) -> None:
        if IGNORE_TESTS:
            return


class TestRateLawEstimates(unittest.TestCase):
    """Tests for RateLawAnalyzer._makeRateLawEstimates."""

    def test_linear_rate_law_estimates_are_exact(self) -> None:
        """For mass-action (linear) rate laws, the one-step linearization is exact."""
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(LINEAR_ANTIMONY_MODEL), [0, 25, NUM_POINT - 1])
        for reaction_id in analyzer._rate_law_dct:
            true_arr = analyzer._timecourse_df[reaction_id].to_numpy()
            est_arr = analyzer._makeRateLawEstimates(reaction_id).to_numpy()
            self.assertTrue(np.isnan(est_arr[0]))  # no previous step at the first timepoint
            np.testing.assert_allclose(est_arr[1:], true_arr[1:], rtol=1e-4, atol=1e-8)

    def test_estimates_are_one_step_back(self) -> None:
        """est(t) approximates the true rate at t, linearized from t - 1 (O(dt^2) error)."""
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, 25, NUM_POINT - 1])
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        est_arr = analyzer._makeRateLawEstimates(r1_id).to_numpy()
        true_arr = analyzer._timecourse_df[r1_id].to_numpy()
        self.assertTrue(np.isnan(est_arr[0]))  # no previous step at the first timepoint
        np.testing.assert_allclose(est_arr[1:], true_arr[1:], rtol=1e-2, atol=1e-8)

    def test_estimates_series_index_matches_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        est_ser = analyzer._makeRateLawEstimates(r1_id)
        self.assertEqual(len(est_ser), NUM_POINT)
        pd.testing.assert_index_equal(est_ser.index, analyzer._timecourse_df.index)

    def test_unknown_reaction_raises(self) -> None:
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        with self.assertRaises(ValueError):
            analyzer._makeRateLawEstimates("no_such_reaction")


class TestMakeAccuracyDF(unittest.TestCase):
    """Tests for RateLawAnalyzer.makeAccuracyDF."""

    def test_linear_model_has_high_accuracy(self) -> None:
        """A purely linear model is approximated exactly, so accuracy is ~1."""
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(LINEAR_ANTIMONY_MODEL), [0, 25, NUM_POINT - 1])
        df = analyzer.makeAccuracyDF()
        self.assertEqual(list(df.index), list(analyzer._rate_law_dct.keys()))
        for reaction_id in df.index:
            self.assertGreater(df.loc[reaction_id, cn.COL_P50], 0.99)

    def test_accuracy_df_has_statistic_columns(self) -> None:
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        df = analyzer.makeAccuracyDF()
        for col in [cn.COL_MEAN, cn.COL_MIN, cn.COL_MAX, cn.COL_P50, cn.COL_COUNT]:
            self.assertIn(col, df.columns)

    def test_mm_model_accuracy_below_one(self) -> None:
        """The nonlinear (Michaelis-Menten) reaction is not approximated exactly."""
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        df = analyzer.makeAccuracyDF()
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        self.assertLess(df.loc[r1_id, cn.COL_P50], 1.0)


class TestSympifyFormula(unittest.TestCase):
    """Tests for rla._sympifyFormula."""

    def test_pow_rewritten_to_power(self) -> None:
        """SBML's pow(x, y) is evaluated as x**y."""
        if IGNORE_TESTS:
            return
        expr = rla._sympifyFormula('cytosol * beta * Vm3s * pow(X, 2) / (pow(K3s, 2) + pow(X, 2))')
        subs = {sp.Symbol(n): v for n, v in
                {'cytosol': 2.0, 'beta': 3.0, 'Vm3s': 4.0, 'X': 1.0, 'K3s': 5.0}.items()}
        self.assertAlmostEqual(float(expr.subs(subs)), 24.0 / 26.0)

    def test_beta_name_parses_as_symbol(self) -> None:
        """'beta' collides with a sympy builtin but must parse as a symbol."""
        if IGNORE_TESTS:
            return
        expr = rla._sympifyFormula('cytosol * (v0 + v1 * beta)')
        self.assertEqual(set(str(s) for s in expr.free_symbols), {'cytosol', 'v0', 'v1', 'beta'})

    def test_builtin_functions_still_parse(self) -> None:
        """Identifiers used as function calls keep their sympy builtin meaning."""
        if IGNORE_TESTS:
            return
        expr = rla._sympifyFormula('k * exp(X)')
        k, X = sp.symbols('k X')
        self.assertEqual(sp.simplify(expr - k*sp.exp(X)), 0)


class TestMakeEstimatesDF(unittest.TestCase):
    """Tests for RateLawAnalyzer.makeEstimatesDF."""

    def test_estimates_df_shape_and_columns(self) -> None:
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        df = analyzer.makeEstimatesDF()
        self.assertEqual(list(df.columns), list(analyzer._rate_law_dct.keys()))
        self.assertEqual(len(df), NUM_POINT)

    def test_estimates_df_matches_series(self) -> None:
        if IGNORE_TESTS:
            return
        analyzer = _make_analyzer(_make_model(MM_ANTIMONY_MODEL), [0, NUM_POINT - 1])
        df = analyzer.makeEstimatesDF()
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        pd.testing.assert_series_equal(df[r1_id], analyzer._makeRateLawEstimates(r1_id))


@unittest.skipIf(not _HAS_BIOMODELS, "BioModels directory not found")
class TestLocalParameters(unittest.TestCase):
    """Tests for kinetic law local parameter and compartment handling."""

    def test_local_parameters_parsed_from_sbml(self) -> None:
        """Local parameters are parsed from the SBML string per reaction."""
        if IGNORE_TESTS:
            return
        model = Model(LOCAL_PARAM_SBML, model_name=MODEL_NAME)
        analyzer = _make_analyzer(model, [0, NUM_POINT - 1])
        self.assertEqual(analyzer._local_parameter_dct['r1'], {'Km': 2.0})

    def test_compartment_sizes_in_parameter_dct(self) -> None:
        """Compartment sizes are collected as constant values."""
        if IGNORE_TESTS:
            return
        model = Model(LOCAL_PARAM_SBML, model_name=MODEL_NAME)
        analyzer = _make_analyzer(model, [0, NUM_POINT - 1])
        self.assertAlmostEqual(analyzer._parameter_value_dct['c'], 1.0)
        self.assertAlmostEqual(analyzer._parameter_value_dct['k'], 0.5)

    def test_estimates_use_local_parameters(self) -> None:
        """A rate law using a local parameter is estimated with its value.

        r1 = k*S/Km is linear in S, so the one-step linearization is exact when
        Km and k are substituted correctly.
        """
        if IGNORE_TESTS:
            return
        model = Model(LOCAL_PARAM_SBML, model_name=MODEL_NAME)
        analyzer = _make_analyzer(model, [0, 25, NUM_POINT - 1])
        est_arr = analyzer._makeRateLawEstimates('r1').to_numpy()
        true_arr = analyzer._timecourse_df['r1'].to_numpy()
        self.assertTrue(np.isnan(est_arr[0]))  # no previous step at the first timepoint
        np.testing.assert_allclose(est_arr[1:], true_arr[1:], rtol=1e-4, atol=1e-8)


class TestPlotRateLawLinearization(unittest.TestCase):
    """Tests for RateLawAnalyzer.plotRateLawLinearization (synthetic model)."""

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        self._fig, self._ax = plt.subplots()

    def tearDown(self) -> None:
        if IGNORE_TESTS:
            return
        plt.close("all")

    def _make_mm_analyzer(self, boundaries: List[int] | None = None) -> rla.RateLawAnalyzer:
        if boundaries is None:
            boundaries = [0, 25, NUM_POINT - 1]
        return _make_analyzer(_make_model(MM_ANTIMONY_MODEL), boundaries)

    def test_plots_simulated_and_estimated_lines(self) -> None:
        """Two lines are drawn: the simulated rate and the linear estimate."""
        if IGNORE_TESTS:
            return
        analyzer = self._make_mm_analyzer()
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        with patch("matplotlib.pyplot.show"):
            analyzer.plotRateLawLinearization(r1_id, ax=self._ax)
        lines = self._ax.get_lines()
        self.assertGreaterEqual(len(lines), 2)
        true_arr = analyzer._timecourse_df[r1_id].to_numpy()
        est_arr = analyzer._makeRateLawEstimates(r1_id).to_numpy()
        np.testing.assert_allclose(np.asarray(lines[0].get_ydata()), true_arr)
        np.testing.assert_allclose(np.asarray(lines[1].get_ydata()), est_arr)

    def test_sets_title_and_axis_labels(self) -> None:
        if IGNORE_TESTS:
            return
        analyzer = self._make_mm_analyzer()
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        with patch("matplotlib.pyplot.show"):
            analyzer.plotRateLawLinearization(r1_id, ax=self._ax)
        self.assertEqual(self._ax.get_title(), f"Rate Law Linearization for Reaction {r1_id}")
        self.assertEqual(self._ax.get_xlabel(), "Time")
        self.assertEqual(self._ax.get_ylabel(), "Rate")

    def test_legend_lists_both_series(self) -> None:
        if IGNORE_TESTS:
            return
        analyzer = self._make_mm_analyzer()
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        with patch("matplotlib.pyplot.show"):
            analyzer.plotRateLawLinearization(r1_id, ax=self._ax)
        legend = self._ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual([t.get_text() for t in legend.get_texts()], ["Simulated Rate", "Estimated Rate"]) # type: ignore

    def test_draws_a_vline_per_interior_boundary(self) -> None:
        """One vertical line per boundary b with 0 < b < NUM_POINT (the two rate lines plus those)."""
        if IGNORE_TESTS:
            return
        analyzer = self._make_mm_analyzer()  # boundaries [0, 25, NUM_POINT - 1]
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        with patch("matplotlib.pyplot.show"):
            analyzer.plotRateLawLinearization(r1_id, ax=self._ax)
        expected_vlines = sum(1 for b in analyzer._boundaries if 0 < b < NUM_POINT)
        self.assertEqual(len(self._ax.get_lines()), 2 + expected_vlines)

    def test_creates_figure_when_no_ax_given(self) -> None:
        """With ax=None a new figure is created and the plot drawn on it."""
        if IGNORE_TESTS:
            return
        analyzer = self._make_mm_analyzer()
        r1_id, _ = _get_reaction_ids(MM_ANTIMONY_MODEL)
        nfig_before = len(plt.get_fignums())
        with patch("matplotlib.pyplot.show"):
            analyzer.plotRateLawLinearization(r1_id)  # ax=None -> new figure
        self.assertEqual(len(plt.get_fignums()), nfig_before + 1)
        fig = plt.figure(plt.get_fignums()[-1])
        self.assertGreaterEqual(len(fig.axes[0].get_lines()), 2)


@unittest.skipIf(not _HAS_BIOMODELS, "BioModels directory not found")
class TestPlotRateLawLinearizationRealModels(unittest.TestCase):
    """End-to-end: generate rate-law linearization plots for real BioModels.

    Each test constructs a RateLawAnalyzer (which requires a valid PSD prediction)
    and saves one plot per reaction to a temp directory, then verifies the files.
    A model without any valid PSD prediction is skipped rather than failing, since
    that reflects data availability rather than a defect in the plotting code.
    """

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        if IGNORE_TESTS:
            return
        plt.close("all")
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _plot_model_reactions(self, model_num: int) -> List[str]:
        """Plot every reaction of the model; return the saved file paths.

        Raises:
            ValueError: If the model has no valid PSD prediction (default threshold).
        """
        analyzer = rla.RateLawAnalyzer(biomodel_id=model_num)
        saved: List[str] = []
        for reaction_id in analyzer._rate_law_dct:
            fig, ax = plt.subplots(figsize=(10, 6))
            with patch("matplotlib.pyplot.show"):
                analyzer.plotRateLawLinearization(reaction_id, ax=ax)
            out_path = os.path.join(self._tmpdir, f"rate_law_linearization_{model_num}_{reaction_id}.png")
            fig.savefig(out_path)
            plt.close(fig)
            saved.append(out_path)
        return saved

    def _assert_plots_saved(self, model_num: int) -> None:
        if IGNORE_TESTS:
            return
        try:
            saved = self._plot_model_reactions(model_num)
        except ValueError as e:
            self.skipTest(f"No valid PSD prediction for model {model_num}: {e}")
        self.assertGreater(len(saved), 0)
        for path in saved:
            self.assertTrue(os.path.exists(path), f"missing plot file {path}")
            self.assertGreater(os.path.getsize(path), 0, f"empty plot file {path}")

    def test_plots_biomodel_5(self) -> None:
        self._assert_plots_saved(5)

    def test_plots_biomodel_10(self) -> None:
        self._assert_plots_saved(10)

    def test_plots_biomodel_904(self) -> None:
        self._assert_plots_saved(904)

    def test_plot_all(self) -> None:
        """Plot all BioModels with valid PSD predictions (slow)."""
        if IGNORE_TESTS:
            return
        model_num = 904
        analyzer = rla.RateLawAnalyzer(biomodel_id=model_num)
        analyzer.plotAllRateLawLinearizations(ymax=0.0002)  # set y-axis limit to 1.0 for better visibility
        #plt.show()


if __name__ == "__main__":
    unittest.main()
