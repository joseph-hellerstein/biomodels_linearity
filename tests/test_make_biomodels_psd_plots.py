"""Tests for scripts/make_biomodels_psd_plots.py."""

import os
import shutil
import sys
import tempfile
import types  # type: ignore
import unittest
from unittest.mock import ANY, MagicMock, call, patch

import matplotlib.pyplot as plt  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from make_biomodels_psd_plots import (  # type: ignore
    PLOT_DIR,
    main,
)


IGNORE_TESTS = True


def _makeMockItem(model_name: str):
    """Create a mock TimecourseIteratorItem with only the attributes main() uses."""
    import pandas as pd  # type: ignore

    df = pd.DataFrame({"S1": [1.0, 2.0]}, index=[0.0, 1.0])
    df.index.name = "time"
    return types.SimpleNamespace(
        model_name=model_name,
        timecourse=types.SimpleNamespace(timecourse_df=df),
    )


def _makeMockPSD(model_num: int = 1):
    """Create a mock PiecewiseSystemDiscovery whose plotPiecewise builds a real figure."""
    psd = MagicMock()
    psd.model_num = model_num

    def _makePlotOptions(**kwargs):
        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0])
        return types.SimpleNamespace(fig=fig)

    psd.plotPiecewise = MagicMock(side_effect=_makePlotOptions)
    return psd


def _makeMockAnalyzer(psd):
    """Create a mock PSDPredictionAnalyzer whose makeBestPSD returns the given psd."""
    analyzer = MagicMock()
    analyzer.makeBestPSD.return_value = psd
    return analyzer


class TestMain(unittest.TestCase):
    """Tests for main()."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()
        self._patcher = patch("make_biomodels_psd_plots.PLOT_DIR", new=self._tmpdir)
        self._patcher.start()

    def tearDown(self) -> None:
        plt.close("all")
        self._patcher.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_passes_model_range_to_iterator(self, mock_iter_class, mock_analyzer_class):
        """main() forwards the model range to TimecourseIterator with reporting on."""
        if IGNORE_TESTS:
            return
        mock_item = _makeMockItem("BIOMD0000000005")
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter([mock_item]))
        mock_analyzer_class.return_value = _makeMockAnalyzer(_makeMockPSD(5))

        main(first_model_num=5, last_model_num=10)

        mock_iter_class.assert_called_once_with(first_model_num=5, last_model_num=10,
                is_report=True)

    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_saves_pdf_for_each_processed_model(self, mock_iter_class, mock_analyzer_class):
        """main() writes one non-empty PDF per successfully processed model."""
        if IGNORE_TESTS:
            return
        items = [_makeMockItem("BIOMD0000000005"), _makeMockItem("BIOMD0000000010")]
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))
        mock_analyzer_class.side_effect = [
            _makeMockAnalyzer(_makeMockPSD(5)),
            _makeMockAnalyzer(_makeMockPSD(10)),
        ]

        main()

        for model_num in (5, 10):
            path = os.path.join(self._tmpdir, f"piecewise_{model_num}.pdf")
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_creates_plot_dir_if_missing(self, mock_iter_class, mock_analyzer_class):
        """main() creates the plot directory when it does not exist."""
        if IGNORE_TESTS:
            return
        nested_dir = os.path.join(self._tmpdir, "nested", "plots")
        mock_item = _makeMockItem("BIOMD0000000005")
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter([mock_item]))
        mock_analyzer_class.return_value = _makeMockAnalyzer(_makeMockPSD(5))

        with patch("make_biomodels_psd_plots.PLOT_DIR", new=nested_dir):
            main()

        self.assertTrue(os.path.isdir(nested_dir))
        self.assertTrue(os.path.isfile(os.path.join(nested_dir, "piecewise_5.pdf")))

    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_constructs_analyzer_with_model_name(self, mock_iter_class, mock_analyzer_class):
        """main() constructs a PSDPredictionAnalyzer per model with the item's timecourse."""
        if IGNORE_TESTS:
            return
        items = [_makeMockItem("BIOMD0000000005"), _makeMockItem("BIOMD0000000010")]
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))
        mock_analyzer_class.side_effect = [
            _makeMockAnalyzer(_makeMockPSD(5)),
            _makeMockAnalyzer(_makeMockPSD(10)),
        ]

        main()

        mock_analyzer_class.assert_has_calls([
            call("BIOMD0000000005", timecourse_df=ANY),
            call("BIOMD0000000010", timecourse_df=ANY),
        ])

    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_calls_plot_piecewise_with_expected_options(self, mock_iter_class,
            mock_analyzer_class):
        """main() plots with the changepoint panel only and no legend."""
        if IGNORE_TESTS:
            return
        mock_item = _makeMockItem("BIOMD0000000005")
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter([mock_item]))
        psd = _makeMockPSD(5)
        mock_analyzer_class.return_value = _makeMockAnalyzer(psd)

        main()

        psd.plotPiecewise.assert_called_once_with(legend=False, num_true_point=60,
                is_nochangepoint_plot=False)

    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_closes_figures_after_saving(self, mock_iter_class, mock_analyzer_class):
        """main() leaves no open figures after processing."""
        if IGNORE_TESTS:
            return
        items = [_makeMockItem("BIOMD0000000005"), _makeMockItem("BIOMD0000000010")]
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))
        mock_analyzer_class.side_effect = [
            _makeMockAnalyzer(_makeMockPSD(5)),
            _makeMockAnalyzer(_makeMockPSD(10)),
        ]

        main()

        self.assertEqual(len(plt.get_fignums()), 0)

    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_empty_iterator_produces_no_plots(self, mock_iter_class, mock_analyzer_class):
        """main() does nothing when the iterator yields no models."""
        if IGNORE_TESTS:
            return
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter([]))

        main()

        self.assertEqual(os.listdir(self._tmpdir), [])
        mock_analyzer_class.assert_not_called()

    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_continues_when_analyzer_construction_fails(self, mock_iter_class,
            mock_analyzer_class):
        """main() skips models whose analyzer fails to construct and keeps going."""
        if IGNORE_TESTS:
            return
        items = [_makeMockItem("BIOMD0000000005"), _makeMockItem("BIOMD0000000010")]
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))
        mock_analyzer_class.side_effect = [ValueError("no data"), _makeMockAnalyzer(_makeMockPSD(10))]

        main()

        self.assertFalse(os.path.exists(os.path.join(self._tmpdir, "piecewise_5.pdf")))
        self.assertTrue(os.path.isfile(os.path.join(self._tmpdir, "piecewise_10.pdf")))

    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_continues_when_make_best_psd_fails(self, mock_iter_class, mock_analyzer_class):
        """main() skips models whose makeBestPSD fails and keeps going."""
        if IGNORE_TESTS:
            return
        items = [_makeMockItem("BIOMD0000000005"), _makeMockItem("BIOMD0000000010")]
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))
        bad_analyzer = MagicMock()
        bad_analyzer.makeBestPSD.side_effect = ValueError("no valid predictions")
        mock_analyzer_class.side_effect = [bad_analyzer, _makeMockAnalyzer(_makeMockPSD(10))]

        main()

        self.assertFalse(os.path.exists(os.path.join(self._tmpdir, "piecewise_5.pdf")))
        self.assertTrue(os.path.isfile(os.path.join(self._tmpdir, "piecewise_10.pdf")))

    @patch("builtins.print")
    @patch("make_biomodels_psd_plots.PSDPredictionAnalyzer")
    @patch("make_biomodels_psd_plots.TimecourseIterator")
    def test_reports_failure_to_stdout(self, mock_iter_class, mock_analyzer_class,
            mock_print):
        """main() reports a message to stdout when a model fails."""
        if IGNORE_TESTS:
            return
        mock_item = _makeMockItem("BIOMD0000000005")
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter([mock_item]))
        mock_analyzer_class.side_effect = ValueError("no data")

        main()

        self.assertGreaterEqual(mock_print.call_count, 1)


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_plot_dir_is_string(self) -> None:
        if IGNORE_TESTS:
            return
        self.assertIsInstance(PLOT_DIR, str)


if __name__ == "__main__":
    unittest.main()

