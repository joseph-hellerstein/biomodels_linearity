"""Represents the static (specification-derived) properties of a model."""

import src.constants as cn  # type: ignore

import os
import re
import tellurium as te  # type: ignore
from typing import Dict, List, Optional, Set, Union, Tuple, cast


_ASSIGNMENT_RULE_RE = re.compile(
    r'<assignmentRule\s+[^>]*variable=["\']([^"\']+)["\'][^>]*/?>',
)


class Model(object):
    """Static properties of a model derived from its specification, not its execution.

    Accepts an SBML or Antimony string. Antimony is converted to SBML on
    construction. RoadRunner is used transiently and not stored.
    """

    def __init__(self, model_str: str, model_name: str = "") -> None:
        """
        Parameters
        ----------
        model_str : str
            SBML XML string or Antimony model string.
        model_name : str
            Optional identifier for the model (e.g. 'BIOMD0000000001').
        """
        if not isinstance(model_str, str):
            raise ValueError("model_str must be a string.")
        self.model_name = model_name
        self.sbml_str = self._toSBML(model_str)
        if self.sbml_str == "":
            raise ValueError(f"Could not process model {model_name}.")
        self._species_names: List[str] = []
        #
        rr = te.loadSBMLModel(self.sbml_str)
        self.species_names = rr.getFloatingSpeciesIds() + rr.getBoundarySpeciesIds()
        self.initial_value_dct = {n: float(rr.model[f"init({n})"])
                for n in self.species_names}
        self.num_reaction = rr.getNumReactions()
        self.num_species = len(self.species_names)
        self.num_assignment_rule = len(rr.getAssignmentRuleIds())
        # Species constrained by assignment rules cannot have their initial values set independently.
        self._assignment_constrained_species: Set[str] = set(
            _ASSIGNMENT_RULE_RE.findall(self.sbml_str)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Model):
            return NotImplemented
        return self.sbml_str == other.sbml_str and self.model_name == other.model_name

    def getModifableSpecies(self) -> List[str]:
        """
        Finds the species whose initial values can be modified.
        """
        rr = te.loadSBMLModel(self.sbml_str)
        modifable_species_names = []
        for species_name in self.species_names:
            try:
                rr.model[f"init({species_name})"] = rr.model[f"init({species_name})"]
                modifable_species_names.append(species_name)
            except Exception as e:
                pass
        #
        return modifable_species_names

    def checkModifableSpecies(self) -> None:
        """Check that the model has modifiable species."""
        modifable_species_names = self.getModifableSpecies()
        if (len(self.species_names) == 0) or len(modifable_species_names) != len(self.species_names):
            raise ValueError(f"model {self.model_name} has non-modifiable species: "
                    f"{set(self.species_names) - set(modifable_species_names)}")

    @staticmethod
    def getBiomodelNumberFromName(model_name: str) -> int:
        """Return the BioModels number if the model name is a BioModels identifier."""
        if model_name.startswith("BIOMD"):
            return int(model_name[5:])
        else:
            raise ValueError(f"Model name '{model_name}' is not a BioModels identifier.")

    @classmethod
    def getModelNumberAndName(cls, model_id: Union[str, int]) -> Tuple[int, str]:
        """Return the model name and number. Special handling for BioModels identifiers.
        Args:
            model_id (Union[str, int]): The identifier of the model, either the number or its name.
        Returns:
            Tuple[int, str]: A tuple containing the BioModels number and name.
        Raises:
            ValueError: If the model_id is not a valid BioModels identifier.
        """
        model_num = -1
        model_name = ""
        try:
            if isinstance(model_id, int):
                model_num = model_id
                model_name = cls.getBiomodelName(model_id)
            elif isinstance(model_id, str):
                model_name = model_id
                model_num = cls.getBiomodelNumberFromName(model_id)
        except:
            # Pass through for unknown Biomodel name
            pass
        if model_num == -1:
            model_name = cast(str,model_id)
        return model_num, model_name

    def getBiomodelNumber(self) -> int:
        """Return the BioModels number if the model name is a BioModels identifier."""
        return self.getBiomodelNumberFromName(self.model_name)

    @staticmethod
    def getBiomodelNum(model_name: str) -> int:
        """Return the BioModels number if the model name is a BioModels identifier."""
        return Model.getBiomodelNumberFromName(model_name)

    @staticmethod 
    def getBiomodelName(model_num: int) -> str:
        return f"BIOMD{model_num:010d}"

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def makeBiomodel(cls, model_name: str="", model_num: int = -1) -> "Model":
        """
        Create a Model from a BioModels SBML file in cn.BIOMODELS_DIR.

        Parameters
        ----------
        model_name : str
            BioModel identifier (e.g. 'BIOMD0000000001'). Must start with 'BIOMD'.
        model_num : int, optional
            The index of the model to load (default is -1, which loads the first one).
            If the model_num is present, model_name is ignored and constructed as 'BIOMD{model_num:010d}'.

        Returns
        -------
        Model
        """
        if model_num > 0:
            model_name = f"BIOMD{model_num:010d}"
        if not model_name.startswith("BIOMD"):
            raise ValueError(f"model_name must start with 'BIOMD', got '{model_name}'.")
        model_dir = os.path.join(cn.BIOMODELS_DIR, model_name)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        xml_files = [
            f for f in os.listdir(model_dir)
            if f.endswith(".xml") and f != "manifest.xml"
        ]
        if not xml_files:
            raise FileNotFoundError(f"No SBML file found in {model_dir}")
        path = os.path.join(model_dir, sorted(xml_files)[0])
        with open(path) as fh:
            sbml_str = fh.read()
        return cls(sbml_str, model_name=model_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _toSBML(self, model_str: str) -> str:
        """Return an SBML string, converting from Antimony if necessary."""
        stripped = model_str.strip()
        try:
            if "<?xml" in stripped or "<sbml" in stripped:
                te.loadSBMLModel(model_str)  # validate
                return model_str
            try:
                rr = te.loada(model_str)
                return rr.getSBML()
            except Exception as e:
                raise ValueError(f"Could not load model: {e}")
        except Exception as e:
            print(f"{self.model_name} failed to load: {e}")
            return ""
