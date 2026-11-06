'''Analyzes the accuracy of linear approximations to reaction rate laws.'''

import src.constants as cn
from model import Model  # type: ignore
from psd_analyzer import PSDAnalyzer  # type: ignore
from simulator import Simulator  # type: ignore
from statistic_calculator import StatisticCalculator  # type: ignore

import re
import sys
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import sympy as sp  # type: ignore
import libsbml  # type: ignore
from SBMLKinetics.common.function_definition import FunctionDefinition  # type: ignore
from SBMLKinetics.common.reaction import Reaction  # type: ignore
from typing import Union, cast, Dict, List

# Minimum |true| value for which accuracy is defined; matches Score.MIN_SIGNIFICANT_VALUE.
MIN_SIGNIFICANT_VALUE = 1e-5

# SBML's pow(x, y) is not parseable by sympy (it collides with the Python
# builtin); rewrite it as x**y before sympifying.
_POW_RE = re.compile(r'pow\(([^,]+),\s*([^)]+)\)')
# Identifiers not used as function calls; mapped to Symbols so names that
# collide with sympy builtins (e.g. 'beta') parse as symbols, not functions.
_IDENT_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(?!\s*\()')
# Reaction blocks and parameter tags, for parsing kinetic law local parameters
# directly from the SBML string (see _makeLocalParameterDct).
_REACTION_BLOCK_RE = re.compile(r'<reaction\b[^>]*>.*?</reaction>', re.DOTALL)
_PARAM_TAG_RE = re.compile(r'<parameter\b[^>]*/?>')
_ATTR_ID_RE = re.compile(r'\bid="([^"]*)"')
_ATTR_VALUE_RE = re.compile(r'\bvalue="([^"]*)"')


def _sympifyFormula(formula: str) -> sp.Expr:
    '''Converts an SBML kinetic law formula string to a sympy expression.

    Rewrites pow(x, y) as x**y and maps bare identifiers to Symbols so that
    names colliding with sympy builtins (e.g. 'beta') parse correctly.

    Args:
        formula (str): The kinetic law formula, with function definitions expanded.

    Returns:
        sp.Expr: The sympy expression for the formula.
    '''
    rewritten = _POW_RE.sub(r'(\1)**(\2)', formula)
    local_dict = {name: sp.Symbol(name) for name in _IDENT_RE.findall(rewritten)}
    return sp.sympify(rewritten, locals=local_dict)


class RateLawAnalyzer(object):
    '''Analyzes the accuracy of linear approximations to reaction rate laws.

    For each reaction, the SBML rate law is extracted as a sympy expression
    (via SBMLKinetics) and linearized, via its first-order Taylor expansion,
    around the species concentrations at the start of each segment defined by
    the best PSD prediction's boundaries. The piecewise linear estimates are
    compared against the rates observed in a simulation with reaction output.
    '''

    def __init__(self, biomodel_id: Union[str, int],
            min_accuracy: float = cn.DEF_MIN_ACCURACY,
            accuracy_stat: str = cn.DEF_ACCURACY_STAT) -> None:
        """
        Args:
            biomodel_id (Union[str, int]): The BioModel number or name.
            min_accuracy (float, optional): Minimum accuracy for a PSD prediction to be eligible. Defaults to cn.DEF_MIN_ACCURACY.
            accuracy_stat (str, optional): Accuracy statistic used to select the best PSD prediction. Defaults to cn.DEF_ACCURACY_STAT.
        """
        self.model_num, self.model_name = Model.getModelNumberAndName(biomodel_id)
        if self.model_num < 0:
            raise ValueError(f"biomodel_id '{biomodel_id}' is not a valid BioModels identifier.")
        self.min_accuracy = min_accuracy
        self.accuracy_stat = accuracy_stat
        self.model = Model.makeBiomodel(model_num=self.model_num)
        # Calculated
        self._sbml_model = self._loadSBMLModel()
        self._rate_law_dct = self._makeRateLawDct()
        self._drate_law_dct = self._makeDRateLawDct()
        self._parameter_value_dct = self._makeParameterValueDct()
        self._local_parameter_dct = self._makeLocalParameterDct()
        self._timecourse_df = self._simulate()
        analyzer = PSDAnalyzer(model_id=self.model_num,
                min_accuracy=self.min_accuracy, accuracy_stat=self.accuracy_stat)
        best_model_ser = analyzer.best_model_ser
        if best_model_ser is None:
            raise ValueError(f"No valid PSD predictions found for model {self.model_name} "
                    f"with min_accuracy {self.min_accuracy}.")
        self._best_model_ser = cast(pd.Series, best_model_ser)
        self._boundaries: List[int] = [int(b) for b in self._best_model_ser[cn.COL_BOUNDARIES]]

    def makeAccuracyDF(self) -> pd.DataFrame:
        '''Compares the piecewise linear rate estimates against the simulated rates.

        Accuracy follows the project convention (see Score):
            Accuracy = 1 - clip(|estimate - true| / |true|, 0, 1),
        with sentinel -1 where the true rate is not significant (|true| <
        MIN_SIGNIFICANT_VALUE) or non-finite.

        Returns:
            DataFrame: One row per reaction (indexed by reaction id) with the
                standard StatisticCalculator columns.
        '''
        calculator = StatisticCalculator()
        for reaction_id in self._rate_law_dct:
            true_arr = self._timecourse_df[reaction_id].to_numpy(dtype=float)
            est_arr = self._makeRateLawEstimates(reaction_id).to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ape = np.abs(est_arr - true_arr) / np.abs(true_arr)
            accuracy = 1.0 - np.clip(ape, 0.0, 1.0)
            invalid = ~np.isfinite(accuracy) | (np.abs(true_arr) < MIN_SIGNIFICANT_VALUE)
            accuracy = np.where(invalid, -1.0, accuracy)
            calculator.add(reaction_id, accuracy)
        return calculator.dataframe.set_index(cn.COL_LABEL)

    def makeEstimatesDF(self) -> pd.DataFrame:
        '''Creates a DataFrame of the piecewise linear rate estimates for all reactions.

        Returns:
            DataFrame: Indexed by time with one column per reaction holding the
                estimated rate.
        '''
        return pd.concat(
            [self._makeRateLawEstimates(reaction_id) for reaction_id in self._rate_law_dct],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _loadSBMLModel(self) -> libsbml.Model:
        '''Parses the model's SBML string and returns the libsbml Model.

        Returns:
            libsbml.Model: The parsed model.

        Raises:
            ValueError: If the SBML string cannot be parsed.
        '''
        document = libsbml.SBMLReader().readSBMLFromString(self.model.sbml_str)
        sbml_model = document.getModel() if document is not None else None
        if sbml_model is None:
            raise ValueError(f"Could not parse SBML for model {self.model_name}.")
        return sbml_model

    def _makeRateLawDct(self) -> Dict[str, sp.Expr]:
        '''Creates a dictionary of rate laws from the SBML model.

        Uses SBMLKinetics to expand function definitions into each kinetic
        law formula, which is then converted to a sympy expression.

        Returns:
            dict:
                key: reaction_id
                value: sympy expression of the rate law
        '''
        function_definitions = [
            FunctionDefinition(self._sbml_model.getFunctionDefinition(n))
            for n in range(self._sbml_model.getNumFunctionDefinitions())
        ]
        dct: Dict[str, sp.Expr] = {}
        for n in range(self._sbml_model.getNumReactions()):
            libsbml_reaction = self._sbml_model.getReaction(n)
            reaction_id = libsbml_reaction.getId()
            if libsbml_reaction.getKineticLaw() is None:
                print(f"Warning: reaction {reaction_id} has no kinetic law; skipping.", file=sys.stderr)
                continue
            try:
                reaction = Reaction(libsbml_reaction, function_definitions=function_definitions)
                reaction.kinetic_law.mkSymbolExpression(function_definitions)
                dct[reaction_id] = _sympifyFormula(reaction.kinetic_law.expanded_formula)
            except Exception as e:
                print(f"Warning: could not parse rate law for reaction {reaction_id}: {e}", file=sys.stderr)
        return dct

    def _makeDRateLawDct(self) -> Dict[str, Dict[str, sp.Expr]]:
        '''Creates a dictionary of derivatives of rate laws with respect to species.

        Returns:
            dict:
                key: reaction_id
                value: dict of derivatives with respect to each species in the rate law
        '''
        species_set = set(self.model.species_names)
        dct: Dict[str, Dict[str, sp.Expr]] = {}
        for reaction_id, expr in self._rate_law_dct.items():
            # Differentiate only with respect to species; parameters are constants.
            dct[reaction_id] = {
                symbol.name: sp.simplify(expr.diff(symbol))
                for symbol in expr.free_symbols
                if symbol.name in species_set
            }
        return dct

    def _makeParameterValueDct(self) -> Dict[str, float]:
        '''Collects the values of all global and local parameters in the model.

        Returns:
            dict: key: parameter id, value: parameter value
        '''
        dct: Dict[str, float] = {}
        for n in range(self._sbml_model.getNumParameters()):
            param = self._sbml_model.getParameter(n)
            name = param.getId() if param.getId() else param.getName()
            if name:
                dct[name] = float(param.getValue())
        for n in range(self._sbml_model.getNumCompartments()):
            compartment = self._sbml_model.getCompartment(n)
            name = compartment.getId()
            if name:
                dct[name] = float(compartment.getSize())
        for n in range(self._sbml_model.getNumReactions()):
            kinetic_law = self._sbml_model.getReaction(n).getKineticLaw()
            if kinetic_law is None:
                continue
            for m in range(kinetic_law.getNumLocalParameters()):
                local_param = kinetic_law.getLocalParameter(m)
                name = local_param.getId() if local_param.getId() else local_param.getName()
                if name:
                    dct[name] = float(local_param.getValue())
        return dct

    def _makeLocalParameterDct(self) -> Dict[str, Dict[str, float]]:
        '''Parses kinetic law local parameters directly from the SBML string.

        The libsbml Python bindings in this environment do not expose kinetic
        law local parameters (getNumLocalParameters returns 0), so they are
        parsed from the XML. In SBML, a <parameter> element inside a reaction
        block is always a kinetic law local parameter.

        Returns:
            dict: key: reaction_id, value: dict of local parameter id -> value
        '''
        dct: Dict[str, Dict[str, float]] = {}
        for block in _REACTION_BLOCK_RE.finditer(self.model.sbml_str):
            tag_match = re.match(r'<reaction\b([^>]*)>', block.group(0))
            id_match = _ATTR_ID_RE.search(tag_match.group(1)) if tag_match else None
            if id_match is None:
                continue
            local_dct: Dict[str, float] = {}
            for param_tag in _PARAM_TAG_RE.finditer(block.group(0)):
                id_m = _ATTR_ID_RE.search(param_tag.group(0))
                value_m = _ATTR_VALUE_RE.search(param_tag.group(0))
                if id_m is None or value_m is None:
                    continue
                try:
                    local_dct[id_m.group(1)] = float(value_m.group(1))
                except ValueError:
                    continue  # Non-numeric value (e.g. a formula); skip it.
            if local_dct:
                dct[id_match.group(1)] = local_dct
        return dct

    def _simulate(self) -> pd.DataFrame:
        '''Simulates the model to obtain species concentrations and reaction rates.

        Returns:
            DataFrame: Wide-format timecourse indexed by time with one column
                per floating species and, because is_reaction_in_output=True,
                one column per reaction holding the observed rate.
        '''
        return Simulator.simulateBiomodel(self.model_num,
                is_reaction_in_output=True).timecourse_df

    def _makeSegmentStarts(self) -> List[int]:
        '''Returns the row indices at which each linearization segment starts.

        Follows PiecewiseSystemDiscovery's convention: the boundaries list is
        [0] + changepoints + [num_point - 1], and segments are half-open
        intervals between consecutive changepoints, with the final segment
        extending to the end of the timecourse.

        Returns:
            list[int]: Segment start row indices in increasing order, always including 0.
        '''
        num_points = len(self._timecourse_df)
        bounds = [b for b in self._boundaries if 0 <= b < num_points]
        # Drop the terminal marker (the final row index), if present.
        if bounds and bounds[-1] == num_points - 1:
            bounds = bounds[:-1]
        starts = [0] + [b for b in bounds if b > 0]
        return list(dict.fromkeys(starts))

    def _makeRateLawEstimates(self, reaction_id: str) -> pd.Series:
        '''Creates a series of estimated reaction rates based on the piecewise linear approximation.

        Within each segment defined by _makeSegmentStarts, the rate law is
        linearized around the species concentrations at the segment's starting
        point: est(t) = f(x0) + sum_i (df/dx_i)(x0) * (x_i(t) - x0_i).

        Args:
            reaction_id (str): The identifier of the reaction.

        Returns:
            pd.Series: Estimated rates indexed by time, named after the reaction.

        Raises:
            ValueError: If the reaction has no rate law in the model.
        '''
        if reaction_id not in self._rate_law_dct:
            raise ValueError(f"Reaction {reaction_id} has no rate law in model {self.model_name}.")
        expr = self._rate_law_dct[reaction_id]
        deriv_dct = self._drate_law_dct[reaction_id]
        num_points = len(self._timecourse_df)
        # Species value series over the whole timecourse; species absent from
        # the output (e.g. boundary species) are held at their initial value.
        species_arr_dct: Dict[str, np.ndarray] = {}
        for species_name in deriv_dct:
            if species_name in self._timecourse_df.columns:
                species_arr_dct[species_name] = self._timecourse_df[species_name].to_numpy(dtype=float)
            else:
                species_arr_dct[species_name] = np.full(num_points, self.model.initial_value_dct[species_name])
        param_subs = {sp.Symbol(name): value for name, value in self._parameter_value_dct.items()}
        param_subs.update(
            {sp.Symbol(name): value for name, value in self._local_parameter_dct.get(reaction_id, {}).items()})
        starts = self._makeSegmentStarts()
        estimates = np.empty(num_points)
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else num_points
            subs0 = {sp.Symbol(name): arr[start] for name, arr in species_arr_dct.items()}
            subs0.update(param_subs)
            f0 = float(expr.subs(subs0))
            estimates[start:end] = f0 + sum(
                float(deriv.subs(subs0)) * (species_arr_dct[name][start:end] - species_arr_dct[name][start])
                for name, deriv in deriv_dct.items()
            )
        return pd.Series(estimates, index=self._timecourse_df.index, name=reaction_id)