"""Tests for Timecourse."""
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from model import Model  # type: ignore
from src.plot_options import PlotOptions  # type: ignore
from timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

IGNORE_TESTS = False
HAS_REAL_ZIP = os.path.isfile(TimecourseIterator.getZipPath(num_point=cn.NUM_POINT))
BIOMODEL_700 = "BIOMD0000000700"

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

NUM_POINT = 11
NUM_SPECIES = 2
MODEL_NAME = "test_model"


def _makeModel() -> Model:
    return Model(ANTIMONY_MODEL, model_name=MODEL_NAME)


def _makeTimecourse() -> Timecourse:
    """Return a Timecourse pre-populated with synthetic data (no simulation)."""
    rng = np.random.default_rng(42)
    timepoint_arr = np.linspace(0.0, 10.0, NUM_POINT)
    timecourse_df = pd.DataFrame(
            rng.standard_normal((NUM_POINT, NUM_SPECIES)),
            index=timepoint_arr,
            columns=["S1", "S2"],
    )
    timecourse_df.index.name = "time"
    return Timecourse(
            model=_makeModel(),
            timecourse_df=timecourse_df,
    )


class TestTimecourseInit(unittest.TestCase):
    """Tests for Timecourse.__init__."""

    def test_stores_model(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertIsInstance(tc.model, Model)

    def test_stores_start_time(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), start_time=1.5)
        self.assertEqual(tc.start_time, 1.5)

    def test_stores_num_points(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), num_point=50)
        self.assertEqual(tc.num_point, 50)

    def test_timecourse_df_stored(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertFalse(tc._timecourse_df.empty)


class TestTimecourseSerialize(unittest.TestCase):
    """Tests for Timecourse.serialize."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_returns_string_path(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = tc.serialize()
        self.assertIsInstance(path, str)

    def test_file_exists_after_serialize(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = tc.serialize()
        self.assertTrue(os.path.isfile(path))

    def test_filename_contains_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = tc.serialize()
        self.assertIn(MODEL_NAME, os.path.basename(path))

    def test_raises_without_model_name(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=Model(ANTIMONY_MODEL))
        with self.assertRaises(ValueError):
            tc.serialize()


class TestTimecourseEq(unittest.TestCase):
    """Tests for Timecourse.__eq__."""

    def test_equal_timecourses(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        self.assertEqual(tc1, tc2)

    def test_different_model_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2.model = Model(ANTIMONY_MODEL, model_name="other")
        self.assertNotEqual(tc1, tc2)

    def test_different_start_time_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2.start_time = 99.0
        self.assertNotEqual(tc1, tc2)

    def test_different_end_time_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2.end_time = 99.0
        self.assertNotEqual(tc1, tc2)

    def test_different_num_point_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2.num_point = 999
        self.assertNotEqual(tc1, tc2)

    def test_different_timecourse_df_not_equal(self) -> None:
        if IGNORE_TESTS:
            return
        tc1 = _makeTimecourse()
        tc2 = _makeTimecourse()
        tc2._timecourse_df = tc2._timecourse_df * 2
        self.assertNotEqual(tc1, tc2)

    def test_not_equal_to_non_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        self.assertIs(tc.__eq__("not a timecourse"), NotImplemented)


class TestTimecourseRoundtrip(unittest.TestCase):
    """Roundtrip tests for Timecourse.serialize / Timecourse.deserialize."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _serializeAndDeserialize(self) -> tuple:
        """Return (original, restored) Timecourse pair via a temp directory."""
        original = _makeTimecourse()
        with patch.object(cn, "TIMECOURSE_SERIALIZATION_DIR", self._tmpdir.name):
            path = original.serialize()
        restored = Timecourse.deserialize(path)
        return original, restored

    def test_roundtrip_equal(self) -> None:
        if IGNORE_TESTS:
            return
        original, restored = self._serializeAndDeserialize()
        self.assertEqual(original, restored)

    def test_deserialize_raises_with_empty_path(self) -> None:
        if IGNORE_TESTS:
            return
        with self.assertRaises(Exception):
            Timecourse.deserialize("")


class TestTimecourseplot(unittest.TestCase):
    """Tests for Timecourse.plot."""

    def setUp(self) -> None:
        self._tc = _makeTimecourse()
        self._fig, self._ax = plt.subplots()

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        result = self._tc.plot(ax=self._ax, fig=self._fig)
        self.assertIsInstance(result, PlotOptions)

    def test_plots_one_line_per_species(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        self.assertEqual(len(self._ax.lines), NUM_SPECIES)

    def test_line_labels_match_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        labels = [line.get_label() for line in self._ax.lines]
        self.assertEqual(labels, self._tc.model.species_names)

    def test_xdata_matches_timecourse_index(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        for line in self._ax.lines:
            np.testing.assert_array_equal(line.get_xdata(),
                    self._tc.timecourse_df.index)

    def test_ydata_matches_species_values(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        for line, name in zip(self._ax.lines, self._tc.model.species_names):
            np.testing.assert_array_equal(line.get_ydata(),
                    self._tc.timecourse_df[name])

    def test_line_colors_are_distinct(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        colors = [line.get_color() for line in self._ax.lines]
        self.assertEqual(len(set(colors)), NUM_SPECIES)

    def test_default_xlabel_is_time(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        self.assertEqual(self._ax.get_xlabel(), "time")

    def test_default_ylabel_is_concentration(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig)
        self.assertEqual(self._ax.get_ylabel(), "concentration")

    def test_title_applied_when_provided(self) -> None:
        if IGNORE_TESTS:
            return
        self._tc.plot(ax=self._ax, fig=self._fig, title="My Title")
        self.assertIn("My Title", self._ax.get_title())

    def test_kwargs_forwarded_to_plot_options(self) -> None:
        if IGNORE_TESTS:
            return
        result = self._tc.plot(ax=self._ax, fig=self._fig, title="Fwd")
        self.assertEqual(result.title, "Fwd")


class TestMakeTimecourses(unittest.TestCase):
    """Tests for Timecourse.makeTimecourses."""

    model: Model

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _makeModel()

    def tearDown(self) -> None:
        plt.close("all")

    def test_returns_list(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(self.model, end_time=10.0, is_plot=False)
        self.assertIsInstance(result, list)

    def test_single_combination_length(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0],
                perturbation_species_fraction=[1.0],
                is_plot=False)
        self.assertEqual(len(result), 1)

    def test_two_value_fractions_length(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.1],
                perturbation_species_fraction=[1.0],
                is_plot=False)
        self.assertEqual(len(result), 2)

    def test_cartesian_product_length(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.1],
                perturbation_species_fraction=[0.5, 1.0],
                is_plot=False)
        self.assertEqual(len(result), 4)

    def test_all_items_are_timecourse(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.1],
                perturbation_species_fraction=[0.5, 1.0],
                is_plot=False)
        for item in result:
            self.assertIsInstance(item, Timecourse)

    def test_value_fraction_varies_slowest(self) -> None:
        if IGNORE_TESTS:
            return
        # [0.0, 0.1] x [0.5, 1.0] → indices 0,1 share value_frac=0.0; 2,3 share 0.1
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.1],
                perturbation_species_fraction=[0.5, 1.0],
                is_plot=False)
        self.assertEqual(result[0].perturbation_value_fraction,
                result[1].perturbation_value_fraction)
        self.assertNotEqual(result[0].perturbation_value_fraction,
                result[2].perturbation_value_fraction)

    def test_perturbation_value_fraction_propagated(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0, 0.2],
                perturbation_species_fraction=[1.0],
                is_plot=False)
        self.assertAlmostEqual(result[0].perturbation_value_fraction, 0.0)
        self.assertAlmostEqual(result[1].perturbation_value_fraction, 0.2)

    def test_perturbation_species_fraction_propagated(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0,
                perturbation_value_fraction=[0.0],
                perturbation_species_fraction=[0.5, 1.0],
                is_plot=False)
        self.assertAlmostEqual(result[0].perturbation_species_fraction, 0.5)
        self.assertAlmostEqual(result[1].perturbation_species_fraction, 1.0)

    def test_num_point_propagated(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, end_time=10.0, num_point=7, is_plot=False)
        self.assertEqual(result[0].num_point, 7)

    def test_start_time_propagated(self) -> None:
        if IGNORE_TESTS:
            return
        result = Timecourse.makeTimecourses(
                self.model, start_time=2.0, end_time=10.0, is_plot=False)
        self.assertAlmostEqual(result[0].start_time, 2.0)

    def test_is_plot_false_creates_no_figure(self) -> None:
        if IGNORE_TESTS:
            return
        before = set(plt.get_fignums())
        Timecourse.makeTimecourses(self.model, end_time=10.0, is_plot=False)
        self.assertEqual(before, set(plt.get_fignums()))

    def test_is_plot_true_creates_one_subplot_per_species(self) -> None:
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            Timecourse.makeTimecourses(
                    self.model, end_time=10.0, num_point=NUM_POINT,
                    perturbation_value_fraction=[0.0],
                    perturbation_species_fraction=[0.0],
                    is_plot=True)
        self.assertEqual(len(plt.gcf().axes), NUM_SPECIES)

    def test_subplot_titles_match_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            Timecourse.makeTimecourses(
                    self.model, end_time=10.0, num_point=NUM_POINT,
                    perturbation_value_fraction=[0.0],
                    perturbation_species_fraction=[0.0],
                    is_plot=True)
        titles = [ax.get_title() for ax in plt.gcf().axes]
        self.assertEqual(titles, self.model.species_names)

    def test_subplot_xlabel_is_time(self) -> None:
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            Timecourse.makeTimecourses(
                    self.model, end_time=10.0, num_point=NUM_POINT,
                    perturbation_value_fraction=[0.0],
                    perturbation_species_fraction=[0.0],
                    is_plot=True)
        for ax in plt.gcf().axes:
            self.assertEqual(ax.get_xlabel(), "time")

    def test_subplot_ylabel_is_concentration(self) -> None:
        if IGNORE_TESTS:
            return
        with patch("matplotlib.pyplot.show"):
            Timecourse.makeTimecourses(
                    self.model, end_time=10.0, num_point=NUM_POINT,
                    perturbation_value_fraction=[0.0],
                    perturbation_species_fraction=[0.0],
                    is_plot=True)
        for ax in plt.gcf().axes:
            self.assertEqual(ax.get_ylabel(), "concentration")

class TestTimecourseEqRealBiomodel(unittest.TestCase):
    """Verifies __eq__ with a real BioModel timecourse loaded from the zip archive."""

    @unittest.skipUnless(HAS_REAL_ZIP, "Real timecourse zip not found")
    def test_biomodel700_equals_itself(self) -> None:
        if IGNORE_TESTS:
            return
        from timecourse_iterator import TimecourseIterator  # type: ignore
        tc = TimecourseIterator().getTimecourse(BIOMODEL_700)
        self.assertEqual(tc, tc)



# ---------------------------------------------------------------------------
# makePerturbationTimecourse tests — pure unit tests, no BioModels required.
# Uses the synthetic antimony model at module level.
# ---------------------------------------------------------------------------


class TestTimecourseMakePerturbationTimecourse(unittest.TestCase):
    """Tests for Timecourse.makePerturbationTimecourse."""

    def test_returns_timecourse_instance(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        perturbed = tc.makePerturbationTimecourse(0.05, 0.1)
        self.assertIsInstance(perturbed, Timecourse)

    def test_preserves_model(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        perturbed = tc.makePerturbationTimecourse(0.05, 0.1)
        self.assertIs(perturbed.model, tc.model)

    def test_preserves_start_time(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), start_time=2.5)
        perturbed = tc.makePerturbationTimecourse(0.05, 0.1)
        self.assertEqual(perturbed.start_time, 2.5)

    def test_preserves_end_time(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), end_time=None)
        perturbed = tc.makePerturbationTimecourse(0.05, 0.1)
        self.assertIsNone(perturbed.end_time)

    def test_preserves_num_point(self) -> None:
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), num_point=42)
        perturbed = tc.makePerturbationTimecourse(0.05, 0.1)
        self.assertEqual(perturbed.num_point, 42)

    def test_stores_perturbation_value_fraction(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        perturbed = tc.makePerturbationTimecourse(0.25, 0.1)
        self.assertEqual(perturbed.perturbation_value_fraction, 0.25)

    def test_stores_perturbation_species_fraction(self) -> None:
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        perturbed = tc.makePerturbationTimecourse(0.05, 0.75)
        self.assertEqual(perturbed.perturbation_species_fraction, 0.75)

    def test_lazy_simulation_not_triggered(self) -> None:
        """New instance has empty _timecourse_df until .timecourse_df is accessed."""
        if IGNORE_TESTS:
            return
        tc = _makeTimecourse()
        perturbed = tc.makePerturbationTimecourse(0.05, 0.1)
        self.assertTrue(perturbed._timecourse_df.empty)

    def test_different_perturbations_produce_different_trajectories(self) -> None:
        """Perturbed and unperturbed trajectories must differ when simulated."""
        if IGNORE_TESTS:
            return
        tc = Timecourse(model=_makeModel(), num_point=50, end_time=10.0)
        perturbed = tc.makePerturbationTimecourse(0.5, 1.0)
        original_df = tc.timecourse_df
        perturbed_df = perturbed.timecourse_df
        self.assertEqual(original_df.shape, perturbed_df.shape)
        # Trajectories should not be element-wise equal.
        self.assertFalse(np.allclose(original_df.values, perturbed_df.values))


if __name__ == "__main__":
    unittest.main()
