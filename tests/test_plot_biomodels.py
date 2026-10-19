"""Tests for scripts/plot_biomodels.py."""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import matplotlib
#matplotlib.use("Agg")  # non-interactive backend for testing
import matplotlib.pyplot as plt  # type: ignore

import src.constants as cn  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from plot_biomodels import (  # type: ignore
    NUM_COL,
    NUM_ROW,
    PLOT_DIR,
    _plot_single_model,
    main,
)


IGNORE_TESTS = True
HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""


def _makeMockModel(model_name: str = "BIOMD0000000001") -> MagicMock:
    """Create a mock Model with species_names and model_name."""
    model = MagicMock()
    model.model_name = model_name
    model.species_names = ["S1", "S2"]
    return model


def _makeSimpleModel(model_name: str = "BIOMD0000000001", species: list[str] | None = None):
    """Create a mock Model using a simple object so attribute access works normally."""
    import types  # type: ignore
    model = types.SimpleNamespace()
    model.model_name = model_name
    model.species_names = species or ["S1", "S2"]
    return model


def _makeMockTimecourse(model: MagicMock) -> MagicMock:
    """Create a mock Timecourse with a timecourse_df for the given model."""
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    tc = MagicMock()
    timepoints = np.linspace(0, 10, 11)
    df = pd.DataFrame(
        {"S1": [10 * np.exp(-0.1 * t) for t in timepoints],
         "S2": [10 * (1 - np.exp(-0.1 * t)) for t in timepoints]},
        index=timepoints,
    )
    df.index.name = "time"
    tc.timecourse_df = df
    return tc


class TestPlotSingleModel(unittest.TestCase):
    """Tests for _plot_single_model."""

    def setUp(self) -> None:
        self._fig, self._ax = plt.subplots()

    def tearDown(self) -> None:
        plt.close(self._fig)

    def test_calls_timecourse_simulation(self) -> None:
        """_plot_single_model creates a Timecourse for the model."""
        if IGNORE_TESTS:
            return
        model = _makeMockModel()
        with patch("plot_biomodels.Timecourse") as mock_tc_class:
            mock_tc = _makeMockTimecourse(model)
            mock_tc_class.return_value = mock_tc
            _plot_single_model(self._ax, model, {})
            mock_tc_class.assert_called_once_with(model=model)

    def test_sets_title_to_model_number(self) -> None:
        """The subplot title is set to just the model number."""
        if IGNORE_TESTS:
            return
        model = _makeMockModel("BIOMD0000000042")
        with patch("plot_biomodels.Timecourse"):
            _plot_single_model(self._ax, model, {})
        self.assertEqual(self._ax.get_title(), "42")

    def test_displays_endtime_info_when_available(self) -> None:
        """End_time info is displayed via annotate when endtime_data has the model."""
        if IGNORE_TESTS:
            return
        model = _makeMockModel("BIOMD0000000001")
        with patch("plot_biomodels.Timecourse"):
            _plot_single_model(
                self._ax, model,
                {"BIOMD0000000001": ("MM", 62.859927567160355)}
            )
        # Check that an annotation was added (annotate creates a text object)
        texts = [t for t in self._ax.texts]
        self.assertEqual(len(texts), 1)
        self.assertIn("[MM]", texts[0].get_text())
        self.assertIn("62.86", texts[0].get_text())

    def test_no_endtime_info_when_not_in_data(self) -> None:
        """No end_time annotation when model is not in endtime_data."""
        if IGNORE_TESTS:
            return
        model = _makeMockModel("BIOMD9999999999")
        with patch("plot_biomodels.Timecourse"):
            _plot_single_model(self._ax, model, {})
        texts = [t for t in self._ax.texts]
        self.assertEqual(len(texts), 0)

    def test_endtime_info_uses_coding_scheme(self) -> None:
        """End_time info uses the correct coding scheme abbreviations."""
        if IGNORE_TESTS:
            return
        model = _makeMockModel("BIOMD0000000005")
        with patch("plot_biomodels.Timecourse"):
            _plot_single_model(
                self._ax, model,
                {"BIOMD0000000005": ("SM", 100.0)}
            )
        texts = [t for t in self._ax.texts]
        self.assertEqual(len(texts), 1)
        self.assertIn("[SM]", texts[0].get_text())

    def test_endtime_info_uses_integer_format_for_whole_numbers(self) -> None:
        """End_time is formatted as integer when it's a whole number."""
        if IGNORE_TESTS:
            return
        model = _makeMockModel("BIOMD0000000005")
        with patch("plot_biomodels.Timecourse"):
            _plot_single_model(
                self._ax, model,
                {"BIOMD0000000005": ("SM", 100.0)}
            )
        texts = [t for t in self._ax.texts]
        self.assertIn("100", texts[0].get_text())

    def test_removes_x_ticks(self) -> None:
        """X tick positions are empty after plotting."""
        if IGNORE_TESTS:
            return
        model = _makeMockModel()
        with patch("plot_biomodels.Timecourse"):
            _plot_single_model(self._ax, model, {})
        self.assertEqual(len(self._ax.get_xticks()), 0)

    def test_removes_y_ticks(self) -> None:
        """Y tick positions are empty after plotting."""
        if IGNORE_TESTS:
            return
        model = _makeMockModel()
        with patch("plot_biomodels.Timecourse"):
            _plot_single_model(self._ax, model, {})
        self.assertEqual(len(self._ax.get_yticks()), 0)

    def test_plots_one_line_per_species(self) -> None:
        """One line is plotted per species in the model."""
        if IGNORE_TESTS:
            return
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore

        model = _makeSimpleModel(species=["S1", "S2"])
        timepoints = np.linspace(0, 10, 11)
        df = pd.DataFrame(
            {"S1": [1.0] * 11, "S2": [2.0] * 11}, index=timepoints,
        )
        df.index.name = "time"

        with patch("plot_biomodels.Timecourse") as mock_tc_class:
            tc = MagicMock()
            tc.timecourse_df = df
            mock_tc_class.return_value = tc
            _plot_single_model(self._ax, model, {})
        lines = self._ax.get_lines()
        self.assertEqual(len(lines), 2)

    def test_plots_correct_number_of_species(self) -> None:
        """Number of plotted lines matches the number of species."""
        if IGNORE_TESTS:
            return
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore

        model = _makeSimpleModel(species=["A", "B", "C"])
        timepoints = np.linspace(0, 10, 11)
        df = pd.DataFrame(
            {"A": [1.0] * 11, "B": [2.0] * 11, "C": [3.0] * 11},
            index=timepoints,
        )
        df.index.name = "time"

        with patch("plot_biomodels.Timecourse") as mock_tc_class:
            tc = MagicMock()
            tc.timecourse_df = df
            mock_tc_class.return_value = tc
            _plot_single_model(self._ax, model, {})
        lines = self._ax.get_lines()
        self.assertEqual(len(lines), 3)

    @unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
    def test_plots_real_biomodels(self) -> None:
        """_plot_single_model runs end-to-end against a real, simulated
        BioModel (BIOMD0000000700) rather than a mocked Timecourse."""
        #if IGNORE_TESTS:
        #    return
        from src.model import Model  # type: ignore

        model_name = "BIOMD0000000700"
        sbml_path = os.path.join(cn.BIOMODELS_DIR, model_name, f"{model_name}_url.xml")
        with open(sbml_path) as f:
            sbml_str = f.read()
        model = Model(model_str=sbml_str, model_name=model_name)

        _plot_single_model(self._ax, model, {model_name: ("MM", 62.86)})

        lines = self._ax.get_lines()
        self.assertEqual(self._ax.get_title(), "700")
        self.assertEqual(len(self._ax.get_xticks()), 0)
        self.assertEqual(len(self._ax.get_yticks()), 0)


class TestMain(unittest.TestCase):
    """Tests for main()."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._orig_plot_dir = PLOT_DIR
        self._patcher = patch("plot_biomodels.PLOT_DIR", new=self._tmpdir)
        self._patcher.start()

    def tearDown(self) -> None:
        plt.close("all")
        self._patcher.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("plot_biomodels.BiomodelsIterator")
    def test_calls_biomoodels_iterator(self, mock_iter_class):
        """main() creates a BiomodelsIterator."""
        if IGNORE_TESTS:
            return
        xml_path = os.path.join(self._tmpdir, "test.xml")
        with open(xml_path, "w") as f:
            f.write(ANTIMONY_MODEL)
        mock_item = MagicMock()
        mock_item.sbml_paths = [xml_path]
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter([mock_item]))

        # Patch Model to avoid actual SBML parsing, and patch _flush_page to skip plotting
        with patch("plot_biomodels.Model") as mock_model_class, \
             patch("plot_biomodels._flush_page"):
            mock_model = _makeMockModel()
            mock_model_class.return_value = mock_model
            main()

        mock_iter_class.assert_called_once_with(is_report=True)

    @patch("plot_biomodels.BiomodelsIterator")
    def test_iterates_all_models(self, mock_iter_class):
        """main() iterates all models without stopping at 25."""
        if IGNORE_TESTS:
            return
        xml_path = os.path.join(self._tmpdir, "test.xml")
        with open(xml_path, "w") as f:
            f.write(ANTIMONY_MODEL)
        items = []
        for i in range(100):
            item = MagicMock()
            item.sbml_paths = [xml_path]
            items.append(item)
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))

        with patch("plot_biomodels.Model") as mock_model_class, \
             patch("plot_biomodels._flush_page"):
            mock_model_class.return_value = _makeMockModel()
            main()

    @patch("plot_biomodels.BiomodelsIterator")
    def test_skips_items_without_sbml_paths(self, mock_iter_class):
        """main() skips BiomodelsItem entries that have no SBML paths."""
        if IGNORE_TESTS:
            return
        xml_path = os.path.join(self._tmpdir, "test.xml")
        with open(xml_path, "w") as f:
            f.write(ANTIMONY_MODEL)

        item_no_sbml = MagicMock()
        item_no_sbml.sbml_paths = []

        item_with_sbml = MagicMock()
        item_with_sbml.sbml_paths = [xml_path]

        mock_iter_class.return_value.__iter__ = MagicMock(
            return_value=iter([item_no_sbml, item_with_sbml])
        )

        with patch("plot_biomodels.Model") as mock_model_class, \
             patch("plot_biomodels._flush_page"):
            mock_model_class.return_value = _makeMockModel()
            main()
            self.assertEqual(mock_model_class.call_count, 1)

    @patch("plot_biomodels.BiomodelsIterator")
    def test_handles_model_loading_errors(self, mock_iter_class):
        """main() continues past models that fail to load."""
        if IGNORE_TESTS:
            return
        xml_bad = os.path.join(self._tmpdir, "bad.xml")
        with open(xml_bad, "w") as f:
            f.write(ANTIMONY_MODEL)

        xml_good = os.path.join(self._tmpdir, "good.xml")
        with open(xml_good, "w") as f:
            f.write(ANTIMONY_MODEL)

        item_bad = MagicMock()
        item_bad.sbml_paths = [xml_bad]

        item_good = MagicMock()
        item_good.sbml_paths = [xml_good]

        def iter_items():
            yield item_bad
            yield item_good

        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(iter_items()))

        with patch("plot_biomodels.Model") as mock_model_class, \
             patch("plot_biomodels._flush_page"):
            mock_model_class.side_effect = [ValueError("bad model"), _makeMockModel()]
            main()
            # Should have called Model twice (once for bad, once for good)
            self.assertEqual(mock_model_class.call_count, 2)

    @patch("plot_biomodels.BiomodelsIterator")
    def test_loads_endtime_data(self, mock_iter_class):
        """main() loads end_time data from the CSV file."""
        if IGNORE_TESTS:
            return
        xml_path = os.path.join(self._tmpdir, "test.xml")
        with open(xml_path, "w") as f:
            f.write(ANTIMONY_MODEL)
        mock_item = MagicMock()
        mock_item.sbml_paths = [xml_path]
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter([mock_item]))

        # Patch Model and _flush_page to avoid side effects
        with patch("plot_biomodels.Model") as mock_model_class, \
             patch("plot_biomodels._flush_page"), \
             patch("plot_biomodels._load_endtime_data") as mock_load:
            mock_model = _makeMockModel()
            mock_model_class.return_value = mock_model
            mock_load.return_value = {"BIOMD0000000001": ("MM", 62.86)}
            main()
            mock_load.assert_called_once()

    @patch("plot_biomodels.BiomodelsIterator")
    def test_flushes_full_pages_incrementally(self, mock_iter_class):
        """main() flushes a page every 25 models as they accumulate."""
        if IGNORE_TESTS:
            return
        xml_path = os.path.join(self._tmpdir, "test.xml")
        with open(xml_path, "w") as f:
            f.write(ANTIMONY_MODEL)

        items = []
        for i in range(50):
            item = MagicMock()
            item.sbml_paths = [xml_path]
            items.append(item)
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))

        with patch("plot_biomodels.Model") as mock_model_class, \
             patch("plot_biomodels._flush_page") as mock_flush:
            mock_model_class.return_value = _makeMockModel()
            main()
            # 50 models / 25 per page = 2 flushes
            self.assertEqual(mock_flush.call_count, 2)

    @patch("plot_biomodels.BiomodelsIterator")
    def test_flushes_partial_final_page(self, mock_iter_class):
        """main() saves a partial final page when models don't fill exactly."""
        if IGNORE_TESTS:
            return
        xml_path = os.path.join(self._tmpdir, "test.xml")
        with open(xml_path, "w") as f:
            f.write(ANTIMONY_MODEL)

        items = []
        for i in range(30):
            item = MagicMock()
            item.sbml_paths = [xml_path]
            items.append(item)
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))

        with patch("plot_biomodels.Model") as mock_model_class, \
             patch("plot_biomodels._flush_page") as mock_flush:
            mock_model_class.return_value = _makeMockModel()
            main()
            # 30 models: first 25 flush page 0, remaining 5 flush page 1
            self.assertEqual(mock_flush.call_count, 2)

    @patch("plot_biomodels.BiomodelsIterator")
    def test_max_model_limits_iteration(self, mock_iter_class):
        """main() stops iterating when max_model is reached."""
        if IGNORE_TESTS:
            return
        xml_path = os.path.join(self._tmpdir, "test.xml")
        with open(xml_path, "w") as f:
            f.write(ANTIMONY_MODEL)

        items = []
        for i in range(10):
            item = MagicMock()
            item.sbml_paths = [xml_path]
            items.append(item)
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))

        with patch("plot_biomodels.Model") as mock_model_class, \
             patch("plot_biomodels._flush_page"):
            mock_model_class.return_value = _makeMockModel()
            main(max_model=5)
            # Only 5 models should be created (max_model=5)
            self.assertEqual(mock_model_class.call_count, 5)

    @patch("plot_biomodels.BiomodelsIterator")
    def test_no_flush_when_no_models(self, mock_iter_class):
        """main() does not call _flush_page when there are no models."""
        if IGNORE_TESTS:
            return
        items = []  # empty iterator
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))

        with patch("plot_biomodels._flush_page") as mock_flush, \
             patch("plot_biomodels._load_endtime_data", return_value={}):
            main()
            self.assertEqual(mock_flush.call_count, 0)


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_num_col_is_five(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(NUM_COL, 5)

    def test_num_row_is_five(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertEqual(NUM_ROW, 5)

    def test_plot_dir_is_string(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIsInstance(PLOT_DIR, str)


class TestExtractModelNumber(unittest.TestCase):
    """Tests for _extract_model_number."""

    def test_extracts_number_from_biomd_name(self) -> None:
        from plot_biomodels import _extract_model_number  # type: ignore
        self.assertEqual(_extract_model_number("BIOMD0000000042"), "42")

    def test_extracts_number_from_lowercase_biomd(self) -> None:
        from plot_biomodels import _extract_model_number  # type: ignore
        self.assertEqual(_extract_model_number("biomd0000000001"), "1")

    def test_returns_original_for_non_biomd_name(self) -> None:
        from plot_biomodels import _extract_model_number  # type: ignore
        self.assertEqual(_extract_model_number("MY_MODEL"), "MY_MODEL")


class TestSourceCodeMap(unittest.TestCase):
    """Tests for the source code mapping."""

    def test_max_median_cv_maps_to_mm(self) -> None:
        from plot_biomodels import SOURCE_CODE_MAP  # type: ignore
        self.assertEqual(SOURCE_CODE_MAP["max_median_cv"], "MM")

    def test_sedml_maps_to_sm(self) -> None:
        from plot_biomodels import SOURCE_CODE_MAP  # type: ignore
        self.assertEqual(SOURCE_CODE_MAP["sedml"], "SM")

    def test_steadystate_maps_to_ss(self) -> None:
        from plot_biomodels import SOURCE_CODE_MAP  # type: ignore
        self.assertEqual(SOURCE_CODE_MAP["steadystate"], "SS")


class TestGetEndtimeInfo(unittest.TestCase):
    """Tests for _get_endtime_info."""

    def test_returns_info_when_model_exists(self) -> None:
        from plot_biomodels import _get_endtime_info  # type: ignore
        data = {"BIOMD0000000001": ("MM", 62.86)}
        code, etime = _get_endtime_info("BIOMD0000000001", data)
        self.assertEqual(code, "MM")
        self.assertAlmostEqual(etime, 62.86)

    def test_returns_empty_when_model_missing(self) -> None:
        from plot_biomodels import _get_endtime_info  # type: ignore
        code, etime = _get_endtime_info("BIOMD9999999999", {})
        self.assertEqual(code, "")
        self.assertEqual(etime, 0.0)


if __name__ == "__main__":
    unittest.main()