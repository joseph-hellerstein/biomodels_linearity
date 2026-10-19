"""Tests for src/simulator.py."""

import os
import re
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from simulator import Simulator, SimulationResult  # type: ignore
from model import Model  # type: ignore

HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)


ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""


_ASSIGNMENT_RULE_RE = re.compile(
    r'<assignmentRule\s+[^>]*variable=["\']([^"\']+)["\'][^>]*/?>',
)


def _make_model(antimony_str: str = ANTIMONY_MODEL, model_name: str = "test_model") -> types.SimpleNamespace:
    """Create a minimal Model-like object for testing using Antimony strings."""
    import tellurium as te  # type: ignore
    rr = te.loada(antimony_str)
    species_names = rr.getFloatingSpeciesIds()
    initial_value_dct = {n: float(rr.model[f"init({n})"]) for n in species_names}
    sbml_str = rr.getSBML()
    model = types.SimpleNamespace()
    model.sbml_str = sbml_str
    model.model_name = model_name
    model.species_names = species_names
    model.initial_value_dct = initial_value_dct
    model.num_reaction = rr.getNumReactions()
    model.num_species = len(species_names)
    model.num_assignment_rule = len(rr.getAssignmentRuleIds())
    model._assignment_constrained_species = set(
        _ASSIGNMENT_RULE_RE.findall(sbml_str)
    )
    return model


class TestSimulationResult(unittest.TestCase):
    """Tests for SimulationResult namedtuple."""

    def test_is_namedtuple(self) -> None:
        result = SimulationResult(timecourse_df=pd.DataFrame(), jacobian_collection_arr=np.array([]))
        self.assertTrue(result.timecourse_df.empty)
        np.testing.assert_array_equal(result.jacobian_collection_arr, np.array([]))


class TestSimulatorInit(unittest.TestCase):
    """Tests for Simulator.__init__."""

    def test_basic_init(self) -> None:
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=100)
        self.assertEqual(sim.start_time, 0.0)
        self.assertEqual(sim.end_time, 10.0)
        self.assertEqual(sim.num_point, 100)

    def test_default_perturbation_values(self) -> None:
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=100)
        self.assertEqual(sim.perturbation_value_fraction, 0.0)
        self.assertEqual(sim.perturbation_species_fraction, 0.5)

    def test_custom_perturbation_values(self) -> None:
        model = _make_model()
        sim = Simulator(
            model=model, start_time=0.0, end_time=10.0, num_point=100,
            perturbation_value_fraction=0.5,
            perturbation_species_fraction=0.3,
        )
        self.assertEqual(sim.perturbation_value_fraction, 0.5)
        self.assertEqual(sim.perturbation_species_fraction, 0.3)

    def test_max_iterator_step_is_class_attribute(self) -> None:
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=100)
        self.assertEqual(sim.MAX_ITERATOR_STEP, 50 * int(1e6))


class TestSimulatorGetPerturbedInitialValues(unittest.TestCase):
    """Tests for Simulator._getPerturbedInitialValues."""

    def test_zero_species_fraction_returns_empty(self) -> None:
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=100,
                        perturbation_species_fraction=0.0)
        result = sim._getPerturbedInitialValues()
        self.assertEqual(result, {})

    def test_zero_value_fraction_no_perturbation(self) -> None:
        model = _make_model()
        np.random.seed(42)
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=100,
                        perturbation_value_fraction=0.0,
                        perturbation_species_fraction=1.0)
        result = sim._getPerturbedInitialValues()
        # With 0 fraction, all perturbations are 0, so values equal originals
        for species, perturbed_val in result.items():
            original = model.initial_value_dct[species]
            self.assertAlmostEqual(perturbed_val, original)

    def test_positive_fraction_increases_values(self) -> None:
        model = _make_model()
        np.random.seed(42)
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=100,
                        perturbation_value_fraction=0.5,
                        perturbation_species_fraction=1.0)
        result = sim._getPerturbedInitialValues()
        for species, perturbed_val in result.items():
            original = model.initial_value_dct[species]
            expected = original * 1.5
            self.assertAlmostEqual(perturbed_val, expected)

    def test_negative_fraction_decreases_values(self) -> None:
        model = _make_model()
        np.random.seed(42)
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=100,
                        perturbation_value_fraction=-0.5,
                        perturbation_species_fraction=1.0)
        result = sim._getPerturbedInitialValues()
        for species, perturbed_val in result.items():
            original = model.initial_value_dct[species]
            expected = original * 0.5
            self.assertAlmostEqual(perturbed_val, expected)

    def test_negative_perturbation_clamped_to_zero(self) -> None:
        """Values that would go negative are clamped to 0."""
        # Create a model with small initial values so perturbation can push them negative
        antimony_str = "S1 = 1; S2 = 0.5;\nS1 -> ; k1*S1\nS2 -> ; k2*S2\nk1=0.1;k2=0.1"
        import tellurium as te  # type: ignore
        rr = te.loada(antimony_str)
        sbml_str = rr.getSBML()
        model = _make_model(antimony_str=antimony_str, model_name="test_clamp")
        np.random.seed(42)
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=100,
                        perturbation_value_fraction=-1.5,
                        perturbation_species_fraction=1.0)
        result = sim._getPerturbedInitialValues()
        for species, perturbed_val in result.items():
            self.assertGreaterEqual(perturbed_val, 0.0)


class TestSimulatorSetInitialValues(unittest.TestCase):
    """Tests for Simulator._setInitialValues."""

    def test_sets_values_in_roadrunner(self) -> None:
        import tellurium as te  # type: ignore
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=100)

        rr = te.loadSBMLModel(model.sbml_str)
        initial_dct = {"S1": 99.0}
        sim._setInitialValues(rr, initial_dct)
        self.assertAlmostEqual(float(rr["S1"]), 99.0)


class TestSimulatorSimulate(unittest.TestCase):
    """Tests for Simulator.simulate()."""

    def test_simulate_returns_simulation_result(self) -> None:
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=51)
        result = sim.simulate(is_jacobian_collection=False)
        self.assertIsInstance(result, SimulationResult)

    def test_simulate_produces_correct_num_points(self) -> None:
        model = _make_model()
        num_point = 51
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=num_point)
        result = sim.simulate(is_jacobian_collection=False)
        self.assertEqual(result.timecourse_df.shape[0], num_point)

    def test_simulate_produces_correct_species_columns(self) -> None:
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=51)
        result = sim.simulate(is_jacobian_collection=False)
        self.assertEqual(list(result.timecourse_df.columns), model.species_names)

    def test_simulate_time_index_matches(self) -> None:
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=51)
        result = sim.simulate(is_jacobian_collection=False)
        expected_times = np.linspace(0.0, 10.0, 51)
        np.testing.assert_allclose(result.timecourse_df.index.values, expected_times)

    def test_simulate_with_jacobians(self) -> None:
        model = _make_model()
        num_point = 51
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=num_point)
        result = sim.simulate(is_jacobian_collection=True)
        self.assertEqual(result.timecourse_df.shape[0], num_point)
        # Jacobian shape: (num_timepoints, num_species, num_species)
        expected_shape = (num_point, model.num_species, model.num_species)
        self.assertEqual(result.jacobian_collection_arr.shape, expected_shape)

    def test_simulate_without_jacobians_empty_array(self) -> None:
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=51)
        result = sim.simulate(is_jacobian_collection=False)
        self.assertEqual(result.jacobian_collection_arr.size, 0)

    def test_simulate_with_start_time(self) -> None:
        """Simulation with start_time > 0 should still work."""
        model = _make_model()
        sim = Simulator(model=model, start_time=1.0, end_time=10.0, num_point=51)
        result = sim.simulate(is_jacobian_collection=False)
        self.assertEqual(result.timecourse_df.shape[0], 51)

    def test_simulate_species_conservation(self) -> None:
        """S1 starts at 10 and decreases; S2 starts at 0 and increases."""
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, end_time=10.0, num_point=51)
        result = sim.simulate(is_jacobian_collection=False)
        s1_values = result.timecourse_df["S1"].values
        s2_values = result.timecourse_df["S2"].values
        # S1 starts at 10 and ends lower
        self.assertAlmostEqual(s1_values[0], 10.0, places=5)
        self.assertLess(s1_values[-1], s1_values[0])
        # S2 starts at 0 and ends higher
        self.assertAlmostEqual(s2_values[0], 0.0, places=5)
        self.assertGreater(s2_values[-1], s2_values[0])


class TestSimulatorGetSteadyState(unittest.TestCase):
    """Tests for Simulator.getSteadyState()."""

    def test_returns_concentrations_at_steady_state(self) -> None:
        """S1 -> S2 -> (removed): both species drain to 0 at steady state."""
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, num_point=10)
        raw_ss = sim.getSteadyState()
        self.assertIsNotNone(raw_ss)
        np.testing.assert_allclose(raw_ss, [0.0, 0.0], atol=1e-6)

    def test_end_time_not_required(self) -> None:
        """getSteadyState works without end_time being set."""
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, num_point=10)
        self.assertIsNone(sim.end_time)
        self.assertIsNotNone(sim.getSteadyState())

    def test_returns_none_when_no_floating_species(self) -> None:
        """A model with no floating species has an empty concentration array."""
        model = _make_model(antimony_str="a := 1; b := 2;", model_name="no_float")
        sim = Simulator(model=model, start_time=0.0, num_point=10)
        self.assertIsNone(sim.getSteadyState())

    def test_returns_none_on_runtime_error(self) -> None:
        """If the steady-state solver raises RuntimeError, return None."""
        model = _make_model()
        sim = Simulator(model=model, start_time=0.0, num_point=10)
        with patch('simulator.te') as mock_te:
            mock_rr = MagicMock()
            mock_rr.getSteadyStateSolver.return_value.setValue.side_effect = RuntimeError("no ss")
            mock_te.loadSBMLModel.return_value = mock_rr
            result = sim.getSteadyState()
        self.assertIsNone(result)


class TestSimulatorEndToEnd(unittest.TestCase):
    """Integration tests for Simulator with a real model."""

    @unittest.skipUnless(
        os.path.isdir(os.path.join("archive", "data")),
        "Archive data directory not found"
    )
    def test_full_simulation_pipeline(self) -> None:
        """Simulate and verify the full pipeline works end-to-end."""
        # Use a simple model that doesn't require BioModels data
        model = _make_model()
        sim = Simulator(
            model=model, start_time=0.0, end_time=20.0, num_point=101,
            perturbation_value_fraction=0.1,
            perturbation_species_fraction=0.5,
        )
        result = sim.simulate(is_jacobian_collection=True)

        # Verify timecourse
        self.assertEqual(result.timecourse_df.shape[0], 101)
        self.assertEqual(list(result.timecourse_df.columns), model.species_names)

        # Verify Jacobians
        self.assertEqual(
            result.jacobian_collection_arr.shape,
            (101, model.num_species, model.num_species),
        )


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestSimulatorSimulateBiomodel(unittest.TestCase):
    """Tests for Simulator.simulateBiomodel(), using real BioModels on disk."""

    def test_happy_path_shapes(self) -> None:
        """A model present in the endtime CSV simulates with the requested num_point."""
        num_point = 15
        model = Model.makeBiomodel(model_num=45)
        result = Simulator.simulateBiomodel(model_num=45, num_point=num_point, is_jacobian_collection=True)
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.timecourse_df.shape[0], num_point)
        expected_shape = (num_point, model.num_species, model.num_species)
        self.assertEqual(result.jacobian_collection_arr.shape, expected_shape)

    def test_explicit_end_time_bypasses_csv_lookup(self) -> None:
        """An explicit end_time works even for a model absent from the endtime CSV.

        BIOMD0000000316 is not present in data/biomodels_endtime.csv, so the
        default end_time=-1.0 lookup path would fail (see
        test_missing_endtime_raises below). Passing end_time explicitly must
        avoid that lookup entirely.
        """
        num_point = 10
        result = Simulator.simulateBiomodel(model_num=316, end_time=10.0, num_point=num_point,
                is_jacobian_collection=True)
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.timecourse_df.shape[0], num_point)
        self.assertEqual(result.jacobian_collection_arr.shape[0], num_point)

    def test_missing_endtime_raises(self) -> None:
        """A model absent from the endtime CSV raises a clear ValueError.

        BIOMD0000000268 is a valid, loadable BioModel that is NOT listed in
        data/biomodels_endtime.csv. With the default end_time=-1.0,
        simulateBiomodel looks up the end time by model name and, on a miss,
        raises ValueError("End time for model ... not found.") rather than
        simulating with a bogus default.
        """
        with self.assertRaises(ValueError):
            Simulator.simulateBiomodel(model_num=268)


if __name__ == "__main__":
    unittest.main()