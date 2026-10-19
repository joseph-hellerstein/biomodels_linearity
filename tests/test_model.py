"""Tests for Model."""
import os
import unittest

import tellurium as te  # type: ignore

import src.constants as cn  # type: ignore
from model import Model  # type: ignore

IGNORE_TESTS = False
HAS_BIOMODELS = os.path.isdir(cn.BIOMODELS_DIR)

ANTIMONY_MODEL = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
"""

ANTIMONY_ONE_SPECIES = """
-> S1; k_in
S1 -> ; k_out*S1
k_in = 1.0; k_out = 0.1; S1 = 0
"""

ANTIMONY_ONE_REACTION = """
-> S1; k_in
k_in = 1.0; S1 = 0
"""

ANTIMONY_WITH_ASSIGNMENT_RULE = """
S1 -> S2; k1*S1
S2 -> ; k2*S2
k1 = 0.1; k2 = 0.2; S1 = 10; S2 = 0
total := S1 + S2
"""

SBML_MODEL = te.loada(ANTIMONY_MODEL).getSBML()


class TestModelInit(unittest.TestCase):
    """Tests for Model.__init__."""

    def test_constructs_from_antimony(self) -> None:
        """Constructor accepts an Antimony string without raising."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertIsInstance(model, Model)

    def test_constructs_from_sbml(self) -> None:
        """Constructor accepts an SBML string without raising."""
        if IGNORE_TESTS:
            return
        model = Model(SBML_MODEL)
        self.assertIsInstance(model, Model)

    def test_invalid_string_raises(self) -> None:
        """A string that is neither valid SBML nor Antimony raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            Model("this is not a model")

    def test_non_string_raises(self) -> None:
        """Passing a non-string raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            Model(123)  # type: ignore

    def test_model_name_stored(self) -> None:
        """model_name passed to constructor is accessible."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL, model_name="test_model")
        self.assertEqual(model.model_name, "test_model")

    def test_model_name_defaults_to_empty(self) -> None:
        """model_name defaults to empty string when not provided."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(model.model_name, "")


class TestModelSbmlStr(unittest.TestCase):
    """Tests for Model.sbml_str."""

    def test_sbml_str_is_string(self) -> None:
        """sbml_str is a str."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertIsInstance(model.sbml_str, str)

    def test_sbml_str_is_valid_xml(self) -> None:
        """sbml_str contains an XML declaration or sbml root tag."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        content = model.sbml_str.strip()
        self.assertTrue("<?xml" in content or "<sbml" in content)

    def test_antimony_converted_to_sbml(self) -> None:
        """sbml_str from an Antimony input is loadable as SBML."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        rr = te.loadSBMLModel(model.sbml_str)
        self.assertIsNotNone(rr)

    def test_sbml_input_preserved(self) -> None:
        """sbml_str from an SBML input equals the original string."""
        if IGNORE_TESTS:
            return
        model = Model(SBML_MODEL)
        self.assertEqual(model.sbml_str, SBML_MODEL)


class TestModelSpeciesNames(unittest.TestCase):
    """Tests for Model.species_names."""

    def test_returns_list(self) -> None:
        """species_names returns a list."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertIsInstance(model.species_names, list)

    def test_elements_are_strings(self) -> None:
        """Each element of species_names is a str."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        for name in model.species_names:
            self.assertIsInstance(name, str)

    def test_correct_species_for_antimony_model(self) -> None:
        """species_names returns ['S1', 'S2'] for the two-species Antimony model."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(model.species_names, ["S1", "S2"])

    def test_correct_species_for_one_species_model(self) -> None:
        """species_names returns ['S1'] for the one-species model."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_ONE_SPECIES)
        self.assertEqual(model.species_names, ["S1"])

    def test_no_brackets_in_names(self) -> None:
        """species_names strips any enclosing brackets from RoadRunner IDs."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        for name in model.species_names:
            self.assertFalse(name.startswith("["))
            self.assertFalse(name.endswith("]"))

    def test_cached_on_second_access(self) -> None:
        """species_names returns the same list object on repeated access."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        first = model.species_names
        second = model.species_names
        self.assertIs(first, second)

    def test_same_species_from_sbml_and_antimony(self) -> None:
        """SBML and Antimony inputs for the same model give the same species_names."""
        if IGNORE_TESTS:
            return
        from_antimony = Model(ANTIMONY_MODEL)
        from_sbml = Model(SBML_MODEL)
        self.assertEqual(from_antimony.species_names, from_sbml.species_names)


class TestModelNumSpecies(unittest.TestCase):
    """Tests for Model.num_species."""

    def test_returns_int(self) -> None:
        """num_species returns an int."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertIsInstance(model.num_species, int)

    def test_correct_count_two_species(self) -> None:
        """num_species returns 2 for the two-species model."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(model.num_species, 2)

    def test_correct_count_one_species(self) -> None:
        """num_species returns 1 for the one-species model."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_ONE_SPECIES)
        self.assertEqual(model.num_species, 1)

    def test_matches_len_species_names(self) -> None:
        """num_species equals len(species_names)."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(model.num_species, len(model.species_names))


class TestModelMakeBiomodel(unittest.TestCase):
    """Tests for Model.makeBiomodel."""

    def test_invalid_prefix_raises(self) -> None:
        """model_name not starting with 'BIOMD' raises ValueError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(ValueError):
            Model.makeBiomodel("NOT_A_MODEL")

    def test_missing_directory_raises(self) -> None:
        """A model_name with no matching directory raises FileNotFoundError."""
        if IGNORE_TESTS:
            return
        with self.assertRaises(FileNotFoundError):
            Model.makeBiomodel("BIOMD9999999999")

    @unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
    def test_returns_model(self) -> None:
        """Returns a Model instance for a known BioModel."""
        if IGNORE_TESTS:
            return
        model = Model.makeBiomodel("BIOMD0000000001")
        self.assertIsInstance(model, Model)

    @unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
    def test_model_name_stored(self) -> None:
        """model_name on the returned object matches the argument."""
        if IGNORE_TESTS:
            return
        model = Model.makeBiomodel("BIOMD0000000001")
        self.assertEqual(model.model_name, "BIOMD0000000001")

    @unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
    def test_sbml_str_is_valid(self) -> None:
        """sbml_str of the returned object is loadable as SBML."""
        if IGNORE_TESTS:
            return
        model = Model.makeBiomodel("BIOMD0000000001")
        rr = te.loadSBMLModel(model.sbml_str)
        self.assertIsNotNone(rr)

    @unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
    def test_species_names_nonempty(self) -> None:
        """The loaded model has at least one floating species."""
        if IGNORE_TESTS:
            return
        model = Model.makeBiomodel("BIOMD0000000001")
        self.assertGreater(len(model.species_names), 0)


class TestModelEq(unittest.TestCase):
    """Tests for Model.__eq__."""

    def test_equal_models(self) -> None:
        """Two Model instances built from the same string are equal."""
        if IGNORE_TESTS:
            return
        model1 = Model(ANTIMONY_MODEL)
        model2 = Model(ANTIMONY_MODEL)
        self.assertEqual(model1, model2)

    def test_different_model_str_not_equal(self) -> None:
        """Models built from different strings are not equal."""
        if IGNORE_TESTS:
            return
        model1 = Model(ANTIMONY_MODEL)
        model2 = Model(ANTIMONY_ONE_SPECIES)
        self.assertNotEqual(model1, model2)

    def test_different_model_name_not_equal(self) -> None:
        """Same sbml_str but different model_name are not equal."""
        if IGNORE_TESTS:
            return
        model1 = Model(SBML_MODEL, model_name="a")
        model2 = Model(SBML_MODEL, model_name="b")
        self.assertNotEqual(model1, model2)

    def test_not_equal_to_non_model(self) -> None:
        """Comparing a Model to a non-Model returns NotImplemented."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertIs(model.__eq__("not a model"), NotImplemented)


@unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
class TestModelBiomodel(unittest.TestCase):
    """Integration tests using a real BioModels SBML file."""

    def setUp(self) -> None:
        if IGNORE_TESTS:
            return
        model_name = "BIOMD0000000001"
        sbml_path = os.path.join(cn.BIOMODELS_DIR, model_name,
                f"{model_name}_url.xml")
        with open(sbml_path) as fh:
            sbml_str = fh.read()
        self.model = Model(sbml_str, model_name=model_name)

    def test_model_name_stored(self) -> None:
        """model_name is stored correctly for a real BioModel."""
        if IGNORE_TESTS:
            return
        self.assertEqual(self.model.model_name, "BIOMD0000000001")

    def test_species_names_nonempty(self) -> None:
        """A real BioModel has at least one floating species."""
        if IGNORE_TESTS:
            return
        self.assertGreater(len(self.model.species_names), 0)

    def test_num_species_positive(self) -> None:
        """num_species is positive for a real BioModel."""
        if IGNORE_TESTS:
            return
        self.assertGreater(self.model.num_species, 0)


class TestModelInitialValueDct(unittest.TestCase):
    """Tests for Model.initial_value_dct."""

    def test_returns_dict(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertIsInstance(model.initial_value_dct, dict)

    def test_keys_match_species_names(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(list(model.initial_value_dct.keys()), model.species_names)

    def test_values_are_floats(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        for v in model.initial_value_dct.values():
            self.assertIsInstance(v, float)

    def test_nonzero_initial_value(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(model.initial_value_dct["S1"], 10.0)

    def test_zero_initial_value(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(model.initial_value_dct["S2"], 0.0)

    def test_one_species_model(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_ONE_SPECIES)
        self.assertEqual(list(model.initial_value_dct.keys()), ["S1"])
        self.assertEqual(model.initial_value_dct["S1"], 0.0)

    def test_length_matches_num_species(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(len(model.initial_value_dct), model.num_species)


class TestModelNumReaction(unittest.TestCase):
    """Tests for Model.num_reaction."""

    def test_returns_int(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertIsInstance(model.num_reaction, int)

    def test_correct_count_two_reactions(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(model.num_reaction, 2)

    def test_correct_count_one_reaction(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_ONE_REACTION)
        self.assertEqual(model.num_reaction, 1)


class TestModelNumAssignmentRule(unittest.TestCase):
    """Tests for Model.num_assignment_rule."""

    def test_returns_int(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertIsInstance(model.num_assignment_rule, int)

    def test_zero_for_model_without_rules(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL)
        self.assertEqual(model.num_assignment_rule, 0)

    def test_one_for_model_with_assignment_rule(self) -> None:
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_WITH_ASSIGNMENT_RULE)
        self.assertEqual(model.num_assignment_rule, 1)



# ---------------------------------------------------------------------------
# checkModifableSpecies tests — synthetic SBML + real BioModels case.
# ---------------------------------------------------------------------------

SBML_WITH_CONSTRAINED_SPECIES = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
  <model id="test_constrained" name="test_constrained">
    <listOfCompartments>
      <compartment id="vol" size="1" boundaryCondition="false" constant="true"/>
    </listOfCompartments>
    <listOfParameters>
      <parameter id="p0" value="5.0" constant="true"/>
    </listOfParameters>
    <listOfSpecies>
      <species id="S1" compartment="vol" initialConcentration="10" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>
    </listOfSpecies>
    <listOfRules>
      <assignmentRule variable="S1">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><ci>p0</ci></math>
      </assignmentRule>
    </listOfRules>
  </model>
</sbml>"""


class TestModelCheckModifableSpecies(unittest.TestCase):
    """Tests for Model.checkModifableSpecies (called explicitly, not during __init__)."""

    def test_synthetic_constrained_species_constructs(self) -> None:
        """Construction succeeds; checkModifableSpecies() reports the non-modifiable species."""
        if IGNORE_TESTS:
            return
        model = Model(SBML_WITH_CONSTRAINED_SPECIES, model_name="synthetic_constrained")
        self.assertEqual(len(model.species_names), 1)
        with self.assertRaises(ValueError) as ctx:
            model.checkModifableSpecies()
        msg = str(ctx.exception)
        self.assertIn("non-modifiable species", msg)
        self.assertIn("S1", msg)

    def test_synthetic_constrained_species_getModifableSubset(self) -> None:
        """getModifableSpecies returns only the modifiable subset when one species is constrained."""
        if IGNORE_TESTS:
            return
        model = Model(SBML_WITH_CONSTRAINED_SPECIES, model_name="synthetic_constrained")
        self.assertEqual(model.getModifableSpecies(), [])

    def test_normal_model_passes_check(self) -> None:
        """A normal antimony model has no non-modifiable species and loads successfully."""
        if IGNORE_TESTS:
            return
        model = Model(ANTIMONY_MODEL, model_name="normal")
        self.assertEqual(model.num_species, 2)
        self.assertIn("S1", model.species_names)

    @unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
    def test_biomd_538_constructs(self) -> None:
        """BIOMD0000000538 has a non-modifiable species; construction succeeds but checkModifableSpecies() fails."""
        if IGNORE_TESTS:
            return
        model = Model.makeBiomodel("BIOMD0000000538")
        self.assertIn("ONLrom", model.species_names)
        with self.assertRaises(ValueError) as ctx:
            model.checkModifableSpecies()
        msg = str(ctx.exception)
        self.assertIn("non-modifiable species", msg)

    @unittest.skipUnless(HAS_BIOMODELS, "BioModels data directory not found")
    def test_biomd_538_getModifableSubset(self) -> None:
        """getModifableSpecies reports a strict subset of species_names for BIOMD 538."""
        if IGNORE_TESTS:
            return
        model = Model.makeBiomodel("BIOMD0000000538")
        modifiable = set(model.getModifableSpecies())
        self.assertTrue(modifiable.issubset(set(model.species_names)))
        self.assertNotIn("ONLrom", modifiable)


if __name__ == "__main__":
    unittest.main()
