"""Tests for scripts/plot_biomodels_comparisons.py."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from plot_biomodels_comparisons import (  # type: ignore
    NUM_COL,
    NUM_ROW,
    PLOT_DIR,
    main,
)

NUM_PER_PAGE = NUM_ROW * NUM_COL
NUM_PLOT_PER_MODEL = 3


def _makeMockItem(model_num: int, has_sbml: bool = True) -> MagicMock:
    item = MagicMock()
    item.model_name = f"BIOMD{model_num:010d}"
    item.model_num = model_num
    item.sbml_paths = ["dummy.xml"] if has_sbml else []
    return item


class TestMain(unittest.TestCase):
    """Tests for main() using mocked BiomodelsIterator/Model

    matplotlib.pyplot.subplots is spied on (called through to the real
    function) rather than mocked, so each call's real (fig, axes) can be
    inspected directly -- this lets tests verify exactly which Axes object
    each model's panels landed on, and which figure got flushed.
    """

    def setUp(self) -> None:
        self._patchers: list = []

    def tearDown(self) -> None:
        for p in self._patchers:
            p.stop()
        plt.close("all")

    def _patch(self, target, **kwargs):
        p = patch(target, **kwargs)
        mock = p.start()
        self._patchers.append(p)
        return mock

    def _run_main(self, num_models: int, **main_kwargs):
        items = [_makeMockItem(i) for i in range(num_models)]
        mock_iter_class = self._patch("plot_biomodels_comparisons.BiomodelsIterator")
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))

        mock_model_class = self._patch("plot_biomodels_comparisons.Model")
        mock_model_class.makeBiomodel.return_value = MagicMock()

        mock_flush = self._patch("plot_biomodels_comparisons._flush_page")

        created_grids: list = []
        real_subplots = plt.subplots

        def _spy_subplots(*args, **kwargs):
            result = real_subplots(*args, **kwargs)
            created_grids.append(result)
            return result

        self._patch("plot_biomodels_comparisons.plt.subplots", side_effect=_spy_subplots)

        main(**main_kwargs)
        return created_grids, mock_iter_class, mock_model_class, mock_flush

    # ------------------------------------------------------------------
    # Regression tests for the two crash-level bugs
    # ------------------------------------------------------------------

    def test_uses_makebiomodel_not_a_nonexistent_method(self) -> None:
        """main() must call Model.makeBiomodel -- a previous version called
        a nonexistent Model.makeBiomodelsModel and crashed with AttributeError."""
        _, _, mock_model_class, _ = self._run_main(3)
        self.assertEqual(mock_model_class.makeBiomodel.call_count, 3)
        mock_model_class.makeBiomodel.assert_any_call("BIOMD0000000000")

    # ------------------------------------------------------------------
    # Regression tests for the page-flush ordering bug
    # ------------------------------------------------------------------

    def test_flush_receives_the_filled_figure_not_a_fresh_one(self) -> None:
        """_flush_page must be called with the figure that was actually
        plotted on, not a newly created blank one (the original bug reassigned
        `fig` to a fresh figure before calling _flush_page)."""
        created_grids, _, _, mock_flush = self._run_main(NUM_PER_PAGE + 3)
        self.assertEqual(mock_flush.call_count, 2)
        flushed_page0_fig = mock_flush.call_args_list[0].args[0]
        flushed_page1_fig = mock_flush.call_args_list[1].args[0]
        self.assertIs(flushed_page0_fig, created_grids[0][0])
        self.assertIs(flushed_page1_fig, created_grids[1][0])

    def test_flushes_full_pages_incrementally(self) -> None:
        """Exactly two full pages produces two flushes and no trailing flush."""
        _, _, _, mock_flush = self._run_main(2 * NUM_PER_PAGE)
        self.assertEqual(mock_flush.call_count, 2)

    def test_no_flush_when_no_models(self) -> None:
        _, _, _, mock_flush = self._run_main(0)
        self.assertEqual(mock_flush.call_count, 0)

    def test_skips_items_without_sbml_paths(self) -> None:
        items = [_makeMockItem(0, has_sbml=False), _makeMockItem(1, has_sbml=True)]
        mock_iter_class = self._patch("plot_biomodels_comparisons.BiomodelsIterator")
        mock_iter_class.return_value.__iter__ = MagicMock(return_value=iter(items))
        mock_model_class = self._patch("plot_biomodels_comparisons.Model")
        mock_model_class.makeBiomodel.return_value = MagicMock()
        self._patch("plot_biomodels_comparisons._flush_page")

        main()

        self.assertEqual(mock_model_class.makeBiomodel.call_count, 1)
        mock_model_class.makeBiomodel.assert_called_once_with("BIOMD0000000001")

    # ------------------------------------------------------------------
    # Regression tests for first_model_num/last_model_num propagation
    # ------------------------------------------------------------------

    def test_custom_last_model_num_passed_through_unchanged(self) -> None:
        _, mock_iter_class, _, _ = self._run_main(
            0, first_model_num=5, last_model_num=200)
        mock_iter_class.assert_called_once_with(
            is_report=True, first_model_num=5, last_model_num=200)


class TestConstants(unittest.TestCase):
    """Tests for module-level constants."""

    def test_num_row_is_two(self) -> None:
        self.assertEqual(NUM_ROW, 2)

    def test_plot_dir_is_string(self) -> None:
        self.assertIsInstance(PLOT_DIR, str)


if __name__ == "__main__":
    unittest.main()
