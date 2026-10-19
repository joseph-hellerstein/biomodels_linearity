"""Tests for scripts.perturbation_study CLI interface."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# ---------------------------------------------------------------------------
# Stub src.* submodules so this module can be imported standalone from the repo root.
# ---------------------------------------------------------------------------
_constants_stub = types.SimpleNamespace(
    DATA_DIR="/tmp",
    COL_SYSTEM_ID="system_id",
    COL_THRESHOLD="threshold",
    COL_COEFFICIENT_THRESHOLD="coefficient_threshold",
)
_src_pkg = types.ModuleType("src")
_src_pkg.__path__ = []  # mark it as a package so src.constants resolves

_sys_mod = types.ModuleType("src.constants")
for _attr, _val in _constants_stub.__dict__.items():
    setattr(_sys_mod, _attr, _val)
sys.modules["src"] = _src_pkg
sys.modules["src.constants"] = _sys_mod
_src_pkg.constants = _sys_mod # type: ignore

class _PerturbationAnalyzerStub:
    analyze_perturbations = staticmethod(lambda **kw: None)
# Minimal stubs for other src submodules imported at module level by perturbation_study.
class _SystemDiscoveryStub:
    analyzePerturbations = staticmethod(lambda **kw: None)
_sys_disc_mod = types.ModuleType("src.system_discovery")
_sys_disc_mod.SystemDiscovery = _SystemDiscoveryStub # type: ignore
sys.modules["src.system_discovery"] = _sys_disc_mod
_src_pkg.system_discovery = _sys_disc_mod  # type: ignore

_tc_it_mod = types.ModuleType("src.timecourse_iterator")
class _TimecourseIteratorStub:
    pass  # only used inside main(); never called by tests.
_tc_it_mod.TimecourseIterator = _TimecourseIteratorStub # type: ignore 
sys.modules["src.timecourse_iterator"] = _tc_it_mod

_pa_mod = types.ModuleType("src.perturbation_analyzer")
_pa_mod.PerturbationAnalyzer = _PerturbationAnalyzerStub  # type: ignore
sys.modules["src.perturbation_analyzer"] = _pa_mod
_src_pkg.perturbation_analyzer = _pa_mod  # type: ignore
_src_pkg.timecourse_iterator = _tc_it_mod # type: ignore

from scripts import perturbation_study as ps  # type: ignore

# Remove stubs from sys.modules so other test files get the real src package.
# perturbation_study already holds direct references to the stub objects, so
# removing them here doesn't affect its behavior.
for _key in ["src", "src.constants", "src.system_discovery", "src.perturbation_analyzer", "src.timecourse_iterator"]:
    sys.modules.pop(_key, None)


# ---------------------------------------------------------------------------
# Helpers shared by integration tests.
# ---------------------------------------------------------------------------
class _FakeResult:
    """Minimal stand-in for PerturbationAnalyzer().analyze_perturbations() return value."""

    def __init__(self, df=None):
        self.df = df if df is not None else pd.DataFrame({"system_id": ["UNKNOWN"]})


def _make_item(model_name: str, n_points: int = 20) -> mock.Mock:
    """Build a fake TimecourseIterator item with synthetic timecourse data."""
    rng = np.random.default_rng(1)
    times = np.linspace(0.0, 5.0, n_points)
    df = pd.DataFrame(
        rng.standard_normal((n_points, 2)),
        index=times,
        columns=["S1", "S2"],
    )
    df.index.name = "time"
    tc = mock.Mock()
    tc.timecourse_df = df
    tc.model = mock.Mock(spec=["name"])
    item = mock.Mock()
    item.model_name = model_name
    item.timecourse = tc
    return item


def _run_main_with_empty_iter(tmpdir: str, threshold: float = 0.01,
                              is_analyze_model: bool = True,
                              is_analyze_species: bool = True) -> None:
    """Invoke main() with a patched empty TimecourseIterator and DATA_DIR."""
    tmpstr = str(tmpdir)  # TemporaryDirectory.__enter__ returns the path string.
    with mock.patch.object(ps.cn, "DATA_DIR", tmpstr), \
         mock.patch("scripts.perturbation_study.TimecourseIterator") as MockTI:
        MockTI.return_value.__iter__ = mock.Mock(return_value=iter([]))
        ps.main(
            threshold=threshold,
            is_analyze_model=is_analyze_model,
            is_analyze_species=is_analyze_species,
        )


def _run_main_with_one_item(tmpdir: str, threshold: float = 0.01,
                            is_analyze_model: bool = True,
                            is_analyze_species: bool = True) -> None:
    """Invoke main() with one fake TimecourseIterator item so the loop body runs and writes output."""
    tmpstr = str(tmpdir)  # TemporaryDirectory.__enter__ returns the path string.
    fake_result = _FakeResult(
        df=pd.DataFrame({"system_id": ["BIOMD0000000001"]}),
    )
    with mock.patch.object(ps.cn, "DATA_DIR", tmpstr), \
         mock.patch("scripts.perturbation_study.TimecourseIterator") as MockTI, \
         mock.patch("scripts.perturbation_study.PerturbationAnalyzer") as MockPA:
        item = _make_item(model_name="BIOMD0000000001", n_points=20)
        MockTI.return_value.__iter__ = mock.Mock(return_value=iter([item]))
        MockPA.return_value.result = fake_result  # type: ignore[union-attr]
        ps.main(
            threshold=threshold,
            is_analyze_model=is_analyze_model,
            is_analyze_species=is_analyze_species,
        )


# ---------------------------------------------------------------------------
# Parser tests (unchanged from prior version).
# ---------------------------------------------------------------------------
class TestPerturbationStudyParser(unittest.TestCase):
    """Tests for the argparse CLI defined in _build_parser."""

    def setUp(self) -> None:
        self.parser = ps._build_parser()

    def test_default_threshold_is_0_001(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.threshold, 0.001)

    def test_custom_threshold_float(self) -> None:
        args = self.parser.parse_args(["--threshold", "0.05"])
        self.assertAlmostEqual(args.threshold, 0.05)

    def test_custom_threshold_int_string_casts_to_float(self) -> None:
        args = self.parser.parse_args(["--threshold", "1"])
        self.assertEqual(args.threshold, 1.0)

    def test_is_analyze_model_default_true(self) -> None:
        args = self.parser.parse_args([])
        self.assertTrue(args.is_analyze_model)

    def test_no_is_analyze_model_sets_false(self) -> None:
        args = self.parser.parse_args(["--no-is-analyze-model"])
        self.assertFalse(args.is_analyze_model)

    def test_is_analyze_species_default_true(self) -> None:
        args = self.parser.parse_args([])
        self.assertTrue(args.is_analyze_species)

    def test_no_is_analyze_species_sets_false(self) -> None:
        args = self.parser.parse_args(["--no-is-analyze-species"])
        self.assertFalse(args.is_analyze_species)

    def test_all_defaults_combined(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(
            (args.threshold, args.is_analyze_model, args.is_analyze_species),
            (0.001, True, True),
        )

    def test_all_negation_flags_combined(self) -> None:
        args = self.parser.parse_args(["--no-is-analyze-model", "--no-is-analyze-species"])
        self.assertEqual(
            (args.threshold, args.is_analyze_model, args.is_analyze_species),
            (0.001, False, False),
        )

    def test_help_exits_without_error(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            self.parser.parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_negative_threshold_is_accepted(self) -> None:
        args = self.parser.parse_args(["--threshold", "-0.1"])
        self.assertAlmostEqual(args.threshold, -0.1)


# ---------------------------------------------------------------------------
# Module-level constant assertions.
# ---------------------------------------------------------------------------
class TestModuleConstants(unittest.TestCase):

    def test_default_threshold_is_positive_float(self) -> None:
        self.assertIsInstance(ps.DEFAULT_THRESHOLD, float)
        self.assertGreater(ps.DEFAULT_THRESHOLD, 0.0)

    def test_poly_degree_is_one(self) -> None:
        self.assertEqual(ps.POLY_DEGREE, 1)

    def test_species_fraction_is_one(self) -> None:
        self.assertAlmostEqual(ps.SPECIES_FRACTION, 1.0)

    def test_perturbations_sorted_ascending(self) -> None:
        for a, b in zip(ps.PERTURBATIONS, ps.PERTURBATIONS[1:]):
            self.assertLessEqual(a, b)

    def test_perturbations_include_zero(self) -> None:
        self.assertIn(0.0, ps.PERTURBATIONS)

    def test_perturbations_symmetric_around_zero(self) -> None:
        for p in ps.PERTURBATIONS:
            if p != 0.0:
                self.assertIn(-p, ps.PERTURBATIONS)

    def test_perturbations_include_expected_values(self) -> None:
        expected = [-0.50, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.50]
        for v in expected:
            self.assertIn(v, ps.PERTURBATIONS)


class TestExcludes(unittest.TestCase):

    def test_excludes_is_list_of_strings(self) -> None:
        for item in ps.EXCLUDES:
            self.assertIsInstance(item, str)

    def test_excluded_models_have_biomd_prefix(self) -> None:
        for m in ps.EXCLUDES:
            self.assertTrue(m.startswith("BIOMD"), msg=f"{m!r} missing BIOMD prefix")


# ---------------------------------------------------------------------------
# main() integration tests: path selection, iteration behavior, output format.
# Uses mocks for TimecourseIterator and SystemDiscovery; patches cn.DATA_DIR to a temp dir.
# ---------------------------------------------------------------------------
class TestMainPathSelection(unittest.TestCase):

    def test_both_flags_writes_model_species_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_main_with_one_item(tmpdir, threshold=0.01)
            expected = os.path.join(tmpdir, "perturbation_study-model_species0.01.csv")
            self.assertTrue(os.path.isfile(expected))

    def test_model_only_writes_model_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_main_with_one_item(
                tmpdir, threshold=0.05,
                is_analyze_model=True, is_analyze_species=False,
            )
            expected = os.path.join(tmpdir, "perturbation_study-model0.05.csv")
            self.assertTrue(os.path.isfile(expected))

    def test_species_only_writes_species_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_main_with_one_item(
                tmpdir, threshold=0.1,
                is_analyze_model=False, is_analyze_species=True,
            )
            expected = os.path.join(tmpdir, "perturbation_study-species0.1.csv")
            self.assertTrue(os.path.isfile(expected))

    def test_both_flags_false_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(ps.cn, "DATA_DIR", tmpdir), \
                 mock.patch("scripts.perturbation_study.TimecourseIterator") as MockTI:
                MockTI.return_value.__iter__ = mock.Mock(return_value=iter([]))
                with self.assertRaises(ValueError):
                    ps.main(threshold=0.01, is_analyze_model=False, is_analyze_species=False)


class TestMainIteration(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_skips_already_done_models(self) -> None:
        """A model already present in the CSV must not be processed again."""
        existing_df = pd.DataFrame({"system_id": ["BIOMD_SKIP"]})
        csv_path = os.path.join(
            self.tmpdir.name, "perturbation_study-model_species0.01.csv"
        )
        existing_df.to_csv(csv_path, index=False)

        with mock.patch.object(ps.cn, "DATA_DIR", self.tmpdir.name), \
             mock.patch("scripts.perturbation_study.TimecourseIterator") as MockTI:
            items = [_make_item("BIOMD_SKIP")]
            MockTI.return_value.__iter__ = mock.Mock(return_value=iter(items))
            with mock.patch(
                "scripts.perturbation_study.PerturbationAnalyzer"
            ) as ma:
                ps.main(threshold=0.01)
        self.assertEqual(ma.call_count, 0)

    def test_skips_excluded_models(self) -> None:
        """Models in ps.EXCLUDES must not be processed."""
        with mock.patch.object(ps.cn, "DATA_DIR", self.tmpdir.name), \
             mock.patch("scripts.perturbation_study.TimecourseIterator") as MockTI:
            items = [_make_item("BIOMD0000000338")]  # in EXCLUDES list
            MockTI.return_value.__iter__ = mock.Mock(return_value=iter(items))
            with mock.patch(
                "scripts.perturbation_study.PerturbationAnalyzer"
            ) as ma:
                ps.main(threshold=0.01)
        self.assertEqual(ma.call_count, 0)

    def test_continues_on_analyze_exception(self) -> None:
        """A failure on one model must not stop iteration over the next."""
        good_item = _make_item("BIOMD_OK", n_points=10)
        bad_item = _make_item("BIOMD_FAIL", n_points=30)
        good_df = pd.DataFrame({"system_id": ["BIOMD_OK"], "r2": [0.9]})

        with mock.patch.object(ps.cn, "DATA_DIR", self.tmpdir.name), \
                mock.patch("scripts.perturbation_study.TimecourseIterator") as MockTI:
            items = [bad_item, good_item]
            MockTI.return_value.__iter__ = mock.Mock(return_value=iter(items))
            with mock.patch(
                "scripts.perturbation_study.PerturbationAnalyzer"
            ) as ma:
                def side_effect_constructor(*args, **kwargs):
                    df: pd.DataFrame = kwargs.get("training_df")  # type: ignore
                    if len(df) > 15:
                        raise RuntimeError("simulated failure")
                    inst = mock.Mock()
                    inst.result = _FakeResult(df=good_df)  # type: ignore[union-attr]
                    return inst

                ma.side_effect = side_effect_constructor
                ps.main(threshold=0.01)

        self.assertEqual(ma.call_count, 2)
        written = pd.read_csv(
            os.path.join(self.tmpdir.name, "perturbation_study-model_species0.01.csv")
        )
        self.assertIn("BIOMD_OK", written["system_id"].values.tolist())

    def test_threshold_passed_to_analyze_perturbations(self) -> None:
        """The threshold argument must flow through to PerturbationAnalyzer."""
        with mock.patch.object(ps.cn, "DATA_DIR", self.tmpdir.name), \
             mock.patch("scripts.perturbation_study.TimecourseIterator") as MockTI:
            items = [_make_item("BIOMD_X")]
            MockTI.return_value.__iter__ = mock.Mock(return_value=iter(items))
            with mock.patch(
                "scripts.perturbation_study.PerturbationAnalyzer"
            ) as ma:
                ma.return_value.result = _FakeResult(df=pd.DataFrame({"system_id": ["BIOMD_X"]}))  # type: ignore[union-attr]
                ps.main(threshold=0.042)
        _, kwargs = ma.call_args
        self.assertAlmostEqual(kwargs["threshold"], 0.042)


class TestMainOutputFormat(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_through_main(self, threshold: float, system_id: str) -> pd.DataFrame:
        """Run main() with one fresh model and return the written CSV as a DataFrame."""
        item = _make_item(system_id)
        result_df = pd.DataFrame({"system_id": [system_id], "r2": [0.85]})
        with mock.patch.object(ps.cn, "DATA_DIR", self.tmpdir.name), \
             mock.patch("scripts.perturbation_study.TimecourseIterator") as MockTI:
            MockTI.return_value.__iter__ = mock.Mock(return_value=iter([item]))
            with mock.patch(
                "scripts.perturbation_study.PerturbationAnalyzer"
            ) as ma:
                ma.return_value.result = _FakeResult(df=result_df)  # type: ignore[union-attr]
                ps.main(threshold=threshold)
        path = os.path.join(self.tmpdir.name, f"perturbation_study-model_species{threshold}.csv")
        return pd.read_csv(path)

    def test_output_contains_system_id_column(self) -> None:
        df = self._write_through_main(0.01, "BIOMD_99")
        self.assertIn("system_id", df.columns.tolist())

    def test_threshold_value_matches_passed_argument(self) -> None:
        df = self._write_through_main(0.042, "BIOMD_99")
        thresholds = df[ps.cn.COL_COEFFICIENT_THRESHOLD].unique()
        self.assertEqual(len(thresholds), 1)
        self.assertAlmostEqual(float(thresholds[0]), 0.042)

    def test_threshold_in_output_not_default_constant(self) -> None:
        """The CSV threshold must come from the call argument, not DEFAULT_THRESHOLD."""
        df = self._write_through_main(0.077, "BIOMD_99")
        thresholds = df[ps.cn.COL_COEFFICIENT_THRESHOLD].unique()
        self.assertAlmostEqual(float(thresholds[0]), 0.077)
        self.assertNotAlmostEqual(float(thresholds[0]), ps.DEFAULT_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
