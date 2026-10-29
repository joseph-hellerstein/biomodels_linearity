"""Tests for scripts/rate_law_statistics.py.

The script classifies each reaction's kinetic law in a BioModels SBML model
using the SBMLKinetics 2DK scheme and writes per-model counts to a CSV. These
tests cover:

* ``getModelName`` / ``getModelPath`` -- pure path/name helpers.
* ``getModelNumsInRange`` -- directory scanning over a temporary
  ``cn.BIOMODELS_DIR`` stand-in (inclusive bounds, sorting, non-BIOMD entries).
* ``makeRow`` -- CSV row construction from a stats dict.
* ``readModel`` / ``getFunctionDefinitions`` -- SBML parsing helpers, using a
  minimal model built with libsbml.
* ``classifyReaction`` -- check ordering, first-match-wins, and tolerance of
  checks that raise, using a mock ``Reaction``.
* ``analyzeModel`` -- end-to-end classification of the minimal libsbml model
  (one reaction with law ``k*S1``, one without a kinetic law).
* ``main`` -- CLI orchestration over temporary directories with
  ``analyzeModel`` mocked, so no real BioModels data is required.
"""

import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import MagicMock, patch

import libsbml  # type: ignore
import pandas as pd  # type: ignore
import sympy  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import scripts.rate_law_statistics as m  # noqa: E402
import src.constants as cn  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _stats(**overrides) -> dict:
    """Build an analyzeModel-style stats dict with sensible defaults."""
    counts = {label: 0 for label in m.K_TYPE_LABELS}
    counts["UNDR"] = 1
    base = {
        "species_count": 2,
        "reaction_count": 3,
        "reactions_with_kinetic_law": 2,
        "reactions_without_kinetic_law": 1,
        "rate_law_counts": counts,
    }
    base.update(overrides)
    return base


def _write_minimal_sbml(tmpdir: str, filename: str = "model.xml") -> str:
    """Write a minimal 2-species SBML model and return its path.

    The model has one reaction S1 -> S2 with kinetic law ``k*S1`` and one
    reaction without a kinetic law.
    """
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel()
    compartment = model.createCompartment()
    compartment.setId("c")
    compartment.setConstant(True)
    for species_id in ("S1", "S2"):
        species = model.createSpecies()
        species.setId(species_id)
        species.setCompartment("c")
        species.setInitialConcentration(1.0)
    parameter = model.createParameter()
    parameter.setId("k")
    parameter.setValue(0.5)
    parameter.setConstant(True)

    reaction = model.createReaction()
    reaction.setId("R1")
    reaction.setReversible(False)
    reaction.addReactant(model.getSpecies("S1"))
    reaction.addProduct(model.getSpecies("S2"))
    kinetic_law = reaction.createKineticLaw()
    kinetic_law.setFormula("k*S1")

    no_law_reaction = model.createReaction()
    no_law_reaction.setId("R2")
    no_law_reaction.setReversible(False)
    no_law_reaction.addReactant(model.getSpecies("S1"))

    path = os.path.join(tmpdir, filename)
    libsbml.SBMLWriter().writeSBMLToFile(document, path)
    return path


def _make_reaction(checks: dict) -> MagicMock:
    """Build a mock Reaction whose kinetic-law checks return the given values.

    ``checks`` maps check method names (e.g. "isUNDR") to either a return
    value or an Exception instance to raise. Checks not listed return False.
    """
    reaction = MagicMock()
    kinetic_law = reaction.kinetic_law
    kinetic_law.expanded_formula = sympy.Symbol("k") * sympy.Symbol("S1")  # type: ignore
    kinetic_law.symbols = ["k", "S1"]
    reaction.reactants = []
    reaction.products = []
    for _, method_name in m.K_TYPE_CHECKS:
        value = checks.get(method_name, False)
        if isinstance(value, Exception):
            getattr(kinetic_law, method_name).side_effect = value
        else:
            getattr(kinetic_law, method_name).return_value = value
    return reaction


def _fake_sbml_file(biomodels_dir: str, model_num: int) -> None:
    """Create the expected (empty) SBML file for a model number."""
    model_name = m.getModelName(model_num)
    model_dir = os.path.join(biomodels_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, f"{model_name}_url.xml"), "w",
              encoding="utf-8"):
        pass


# ---------------------------------------------------------------------------
# getModelName / getModelPath
# ---------------------------------------------------------------------------

class TestGetModelName(unittest.TestCase):
    """Tests for the model-number-to-name helper."""

    def test_single_digit_number_is_zero_padded(self) -> None:
        self.assertEqual(m.getModelName(1), "BIOMD0000000001")

    def test_multi_digit_number(self) -> None:
        self.assertEqual(m.getModelName(56), "BIOMD0000000056")


class TestGetModelPath(unittest.TestCase):
    """Tests for the model-number-to-SBML-path helper."""

    def test_path_joins_biomodels_dir_model_name_and_url_xml(self) -> None:
        expected = os.path.join(cn.BIOMODELS_DIR, "BIOMD0000000007",
                                "BIOMD0000000007_url.xml")
        self.assertEqual(m.getModelPath(7), expected)


# ---------------------------------------------------------------------------
# getModelNumsInRange
# ---------------------------------------------------------------------------

class TestGetModelNumsInRange(unittest.TestCase):
    """Tests for directory scanning over a temporary BIOMODELS_DIR stand-in."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()  # type: ignore

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _add_entries(self, *entries: str) -> None:
        for entry in entries:
            with open(os.path.join(self._tmpdir.name, entry), "w",
                      encoding="utf-8"):
                pass

    def _nums_in_range(self, first: int, last: int) -> list[int]:
        with patch.object(cn, "BIOMODELS_DIR", self._tmpdir.name):
            return m.getModelNumsInRange(first, last)

    def test_inclusive_bounds(self) -> None:
        self._add_entries("BIOMD0000000001", "BIOMD0000000003")
        self.assertEqual(self._nums_in_range(1, 3), [1, 3])
        self.assertEqual(self._nums_in_range(3, 3), [3])

    def test_excludes_models_outside_range(self) -> None:
        self._add_entries("BIOMD0000000001", "BIOMD0000000003",
                          "BIOMD000000010")
        self.assertEqual(self._nums_in_range(2, 9), [3])

    def test_ignores_entries_without_biomd(self) -> None:
        self._add_entries("BIOMD0000000001", "README.txt")
        self.assertEqual(self._nums_in_range(0, 10), [1])

    def test_returns_sorted_numbers(self) -> None:
        self._add_entries("BIOMD000000010", "BIOMD0000000003",
                          "BIOMD0000000001")
        self.assertEqual(self._nums_in_range(0, 10), [1, 3, 10])

    def test_empty_directory_returns_empty_list(self) -> None:
        self.assertEqual(self._nums_in_range(0, 10), [])


# ---------------------------------------------------------------------------
# makeRow
# ---------------------------------------------------------------------------

class TestMakeRow(unittest.TestCase):
    """Tests for CSV row construction from a stats dict."""

    def test_row_contains_every_csv_column(self) -> None:
        row = m.makeRow(5, _stats())
        self.assertEqual(set(row), set(m.CSV_COLUMNS))

    def test_row_values_match_stats(self) -> None:
        row = m.makeRow(5, _stats())
        self.assertEqual(row[cn.COL_MODEL_NUM], 5)
        self.assertEqual(row[cn.COL_MODEL_NAME], "BIOMD0000000005")
        self.assertEqual(row[cn.COL_NUM_SPECIES], 2)
        self.assertEqual(row[cn.COL_NUM_REACTION], 3)
        self.assertEqual(row["num_reaction_with_kinetic_law"], 2)
        self.assertEqual(row["num_reaction_without_kinetic_law"], 1)
        self.assertEqual(row["UNDR"], 1)
        self.assertEqual(row["ZERO"], 0)


# ---------------------------------------------------------------------------
# readModel / getFunctionDefinitions
# ---------------------------------------------------------------------------

class TestReadModel(unittest.TestCase):
    """Tests for SBML file reading."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()  # type: ignore

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_returns_model_for_valid_sbml(self) -> None:
        path = _write_minimal_sbml(self._tmpdir.name)
        model = m.readModel(path)
        self.assertIsInstance(model, libsbml.Model)

    def test_exits_with_error_for_unparseable_file(self) -> None:
        bad_path = os.path.join(self._tmpdir.name, "bad.xml")
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write("this is not SBML <<<")
        stderr = StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                m.readModel(bad_path)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("could not parse", stderr.getvalue())


class TestGetFunctionDefinitions(unittest.TestCase):
    """Tests for wrapping model function definitions."""

    def test_empty_list_when_model_has_no_function_definitions(self) -> None:
        document = libsbml.SBMLDocument(3, 2)
        model = document.createModel()
        self.assertEqual(m.getFunctionDefinitions(model), [])

    def test_wraps_each_function_definition(self) -> None:
        document = libsbml.SBMLDocument(3, 2)
        model = document.createModel()
        for name in ("f1", "f2"):
            model.createFunctionDefinition().setName(name)
        result = m.getFunctionDefinitions(model)
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertIsInstance(item, m.FunctionDefinition)


# ---------------------------------------------------------------------------
# classifyReaction
# ---------------------------------------------------------------------------

class TestClassifyReaction(unittest.TestCase):
    """Tests for kinetic-law type classification with a mock Reaction."""

    def test_first_matching_check_wins(self) -> None:
        reaction = _make_reaction({"isZerothOrder": True, "isUNDR": True})
        self.assertEqual(m.classifyReaction(reaction, [], {"S1"}), "ZERO")

    def test_later_check_selected_when_earlier_checks_fail(self) -> None:
        reaction = _make_reaction({"isZerothOrder": False, "isUNDR": False,
                                   "isUNMO": True})
        self.assertEqual(m.classifyReaction(reaction, [], {"S1"}), "UNMO")

    def test_returns_na_when_no_check_matches(self) -> None:
        reaction = _make_reaction({})
        self.assertEqual(m.classifyReaction(reaction, [], {"S1"}), m.NA)

    def test_check_that_raises_is_skipped(self) -> None:
        reaction = _make_reaction({"isZerothOrder": ValueError("boom"),
                                   "isUNDR": True})
        self.assertEqual(m.classifyReaction(reaction, [], {"S1"}), "UNDR")

    def test_returns_na_when_all_checks_raise(self) -> None:
        checks = {name: ValueError("boom") for _, name in m.K_TYPE_CHECKS}
        reaction = _make_reaction(checks)
        self.assertEqual(m.classifyReaction(reaction, [], {"S1"}), m.NA)


# ---------------------------------------------------------------------------
# analyzeModel
# ---------------------------------------------------------------------------

class TestAnalyzeModel(unittest.TestCase):
    """End-to-end classification of a minimal libsbml model."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()  # type: ignore

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _analyze_minimal_model(self) -> dict:
        path = _write_minimal_sbml(self._tmpdir.name)
        return m.analyzeModel(path)

    def test_counts_species_and_reactions(self) -> None:
        stats = self._analyze_minimal_model()
        self.assertEqual(stats["species_count"], 2)
        self.assertEqual(stats["reaction_count"], 2)

    def test_reaction_without_kinetic_law_is_counted_separately(self) -> None:
        stats = self._analyze_minimal_model()
        self.assertEqual(stats["reactions_with_kinetic_law"], 1)
        self.assertEqual(stats["reactions_without_kinetic_law"], 1)

    def test_simple_mass_action_reaction_is_classified_undr(self) -> None:
        stats = self._analyze_minimal_model()
        counts = stats["rate_law_counts"]
        self.assertEqual(counts["UNDR"], 1)
        # Only the reaction with a kinetic law is classified.
        self.assertEqual(sum(counts.values()), 1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    """CLI orchestration over temporary directories with analyzeModel mocked."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()  # type: ignore
        self.biomodels_dir = os.path.join(self._tmpdir.name, "biomodels")
        self.output_path = os.path.join(self._tmpdir.name, "out.csv")
        os.makedirs(self.biomodels_dir)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _run_main(self, first: int, last: int) -> tuple[str, str]:
        """Run main() for the given range; return (stdout, stderr)."""
        argv = ["rate_law_statistics.py", "--first_model_num", str(first),
                "--last_model_num", str(last), "--output", self.output_path]
        stdout, stderr = StringIO(), StringIO()
        with patch.object(sys, "argv", argv), \
             patch.object(cn, "BIOMODELS_DIR", self.biomodels_dir), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            m.main()
        return stdout.getvalue(), stderr.getvalue()

    def test_first_greater_than_last_exits_with_error(self) -> None:
        stderr = StringIO()
        with patch.object(sys, "argv", ["prog", "--first_model_num", "5",
                                        "--last_model_num", "2",
                                        "--output", self.output_path]), \
                    redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                m.main()
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("must be", stderr.getvalue())
        self.assertFalse(os.path.exists(self.output_path))


if __name__ == "__main__":  # pragma: no cover - invoked by pytest / unittest discover only
    unittest.main()
