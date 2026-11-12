'''Analyzes the accuracy of linear approximations to reaction rate laws. '''

import src.constants as cn
from src.model import Model  # type: ignore
from src.psd_analyzer import PSDAnalyzer  # type: ignore
from src.simulator import Simulator  # type: ignore
from src.statistic_calculator import StatisticCalculator  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

import re
import sys
import matplotlib.pyplot as plt  # type: ignore
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
    around the species concentrations at the previous timepoint:
    est(t) = f(x_{t-1}) + J(x_{t-1}) * (x_t - x_{t-1}). The estimates are
    compared against the rates observed in a simulation with reaction output
    (est(t) vs. the true rate at t).
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
        # Cache of compiled (numpy) rate-law callables, keyed by reaction id.
        self._compiled_cache: Dict[str, object] = {}
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
        '''Compares the one-step rate estimates against the simulated rates.

        est(t) is a linearization anchored at time t - 1 evaluated at t, so it
        aligns directly with the true rate at t. The first timepoint has no
        previous step and becomes a sentinel.

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
        '''Creates a DataFrame of the one-step rate estimates for all reactions.

        Returns:
            DataFrame: Indexed by time with one column per reaction holding the
                estimated rate; est(t) estimates the rate at t from a linearization
                anchored at t - 1, and the first row is NaN.
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
        df = Simulator.simulateBiomodel(self.model_num,
                is_reaction_in_output=True).timecourse_df
        return df

    def _compiled_functions_for(self, reaction_id: str):
        '''Compile the rate law and its derivatives to numpy functions for *reaction_id*.

        Parameters are substituted first (they are constant over time), so the
        compiled callables take only the species that appear in the law. Returns
        ``(f_func, deriv_funcs)`` or ``None`` if any free symbol remains after the
        parameter substitution (in which case the caller falls back to symbolic
        evaluation). Results are cached per reaction for the lifetime of the analyzer.

        Args:
            reaction_id (str): The identifier of the reaction.

        Returns:
            Tuple[Callable, Dict[str, Callable]] | None: Compiled numpy callables, or None.
        '''
        if reaction_id in self._compiled_cache:
            return self._compiled_cache[reaction_id]
        expr = self._rate_law_dct[reaction_id]
        deriv_dct = self._drate_law_dct[reaction_id]
        species_names = list(deriv_dct)
        sym_list = [sp.Symbol(n) for n in species_names]
        param_subs = {sp.Symbol(name): value for name, value in self._parameter_value_dct.items()}
        param_subs.update({sp.Symbol(name): value
                for name, value in self._local_parameter_dct.get(reaction_id, {}).items()})
        expr_sp = expr.subs(param_subs)
        deriv_sp = {name: d.subs(param_subs) for name, d in deriv_dct.items()}
        sym_set = set(sym_list)
        if not (expr_sp.free_symbols <= sym_set and all(d.free_symbols <= sym_set for d in deriv_sp.values())):
            self._compiled_cache[reaction_id] = None
            return None
        f_func = sp.lambdify(sym_list, expr_sp, modules='numpy')
        deriv_funcs = {name: sp.lambdify(sym_list, d, modules='numpy') for name, d in deriv_sp.items()}
        self._compiled_cache[reaction_id] = (f_func, deriv_funcs)
        return f_func, deriv_funcs

    def _makeRateLawEstimates(self, reaction_id: str) -> pd.Series:
        '''Creates a series of estimated reaction rates based on one-step linearization.

        At each timepoint t, the rate law is linearized around the species
        concentrations at the previous timepoint and evaluated at t:
            est(t) = f(x_{t-1}) + sum_i (df/dx_i)(x_{t-1}) * (x_i(t) - x_i(t-1)).
        The first timepoint has no previous step and is set to NaN.

        The expression and its derivatives are compiled to numpy callables once per
        reaction (parameters substituted first, since they are constant over time),
        so the whole timecourse is evaluated in a single vectorized pass. If any free
        symbol remains after the parameter substitution, it falls back to symbolic
        evaluation.

        Args:
            reaction_id (str): The identifier of the reaction.

        Returns:
            pd.Series: Estimated rates indexed by time, named after the reaction;
                est(t) estimates the rate at t and the first entry is NaN.

        Raises:
            ValueError: If the reaction has no rate law in the model.
        '''
        if reaction_id not in self._rate_law_dct:
            raise ValueError(f"Reaction {reaction_id} has no rate law in model {self.model_name}.")
        deriv_dct = self._drate_law_dct[reaction_id]
        num_points = len(self._timecourse_df)
        # Species value series over the whole timecourse; species absent from
        # the output (e.g. boundary species) are held at their initial value.
        species_arr_dct: Dict[str, np.ndarray] = {}
        for species_name in deriv_dct:
            if species_name in self._timecourse_df.columns:
                species_arr_dct[species_name] = self._timecourse_df[species_name].to_numpy(dtype=float)
            else:
                init_val = self.model.initial_value_dct[species_name]
                species_arr_dct[species_name] = np.full(num_points, init_val)

        compiled = self._compiled_functions_for(reaction_id)
        if compiled is not None:
            f_func, deriv_funcs = compiled  # type: ignore
            estimates = self._estimate_compiled(species_arr_dct, f_func, deriv_funcs)
        else:
            estimates = self._estimate_symbolic(reaction_id, species_arr_dct)
        ser = pd.Series(estimates, index=self._timecourse_df.index, name=reaction_id)
        return ser

    def _estimate_compiled(self, species_arr_dct: Dict[str, np.ndarray],
            f_func, deriv_funcs) -> np.ndarray:
        '''One-step estimates using the compiled numpy callables.

        Args:
            species_arr_dct (dict): Species name -> full timecourse array. Its value
                order matches the symbol order used to compile f_func / deriv_funcs.
            f_func (Callable): Compiled rate law; takes species values in order.
            deriv_funcs (dict): Species name -> compiled derivative callable.

        Returns:
            np.ndarray: Estimated rates over the whole timecourse; est(t) estimates
                the rate at t (linearization anchored at t - 1) and the first entry
                is NaN.
        '''
        return self._backward_one_step_estimates(species_arr_dct, f_func, deriv_funcs)

    def _estimate_symbolic(self, reaction_id: str,
            species_arr_dct: Dict[str, np.ndarray]) -> np.ndarray:
        '''One-step estimates via symbolic (lambdified) callables.

        Fallback path used when the rate law cannot be compiled to numpy callables
        by _compiled_functions_for. The parameter-substituted expression and its
        derivatives are lambdified over the species symbols; if any free symbol
        remains that is not a species, evaluation is impossible and an error is raised.

        Args:
            reaction_id (str): The identifier of the reaction.
            species_arr_dct (dict): Species name -> full timecourse array.

        Returns:
            np.ndarray: Estimated rates over the whole timecourse; est(t) estimates
                the rate at t (linearization anchored at t - 1) and the first entry
                is NaN.

        Raises:
            RuntimeError: If a free symbol remains after parameter substitution that
                is not one of the species in the rate law.
        '''
        param_subs = {sp.Symbol(name): value for name, value in self._parameter_value_dct.items()}
        param_subs.update(
            {sp.Symbol(name): value for name, value in self._local_parameter_dct.get(reaction_id, {}).items()})
        expr_sp = self._rate_law_dct[reaction_id].subs(param_subs)
        deriv_dct = self._drate_law_dct[reaction_id]
        unresolved = {s.name for s in expr_sp.free_symbols} - set(species_arr_dct)
        if unresolved:
            raise RuntimeError(f"Cannot evaluate rate law for reaction {reaction_id}: "
                    f"unresolved symbols {sorted(unresolved)} after parameter substitution.")
        sym_list = [sp.Symbol(name) for name in deriv_dct]
        f_func = sp.lambdify(sym_list, expr_sp, modules='numpy')
        deriv_funcs = {name: sp.lambdify(sym_list, d.subs(param_subs), modules='numpy')
                for name, d in deriv_dct.items()}
        return self._backward_one_step_estimates(species_arr_dct, f_func, deriv_funcs)

    def _backward_one_step_estimates(self, species_arr_dct: Dict[str, np.ndarray],
            f_func, deriv_funcs) -> np.ndarray:
        '''Vectorized one-step linearization estimates.

        est(t) = f(x_{t-1}) + sum_i (df/dx_i)(x_{t-1}) * (x_i(t) - x_i(t-1))
        for t = 1 .. n-1; the first timepoint has no previous step and is NaN.

        Args:
            species_arr_dct (dict): Species name -> full timecourse array. Its value
                order must match the symbol order used to compile f_func / deriv_funcs.
            f_func (Callable): Rate law callable; takes species arrays in order and
                returns an array of values.
            deriv_funcs (dict): Species name -> derivative callable with the same
                calling convention as f_func.

        Returns:
            np.ndarray: Estimated rates over the whole timecourse; est(t) estimates
                the rate at t (linearization anchored at t - 1) and the first entry
                is NaN.
        '''
        num_points = len(self._timecourse_df)
        estimates = np.full(num_points, np.nan)
        if num_points < 2:
            return estimates
        x_prev = [species_arr_dct[name][:-1] for name in species_arr_dct]
        est_arr = np.asarray(f_func(*x_prev), dtype=float)
        for species_name, deriv_wrt_species in deriv_funcs.items():
            dx_arr = species_arr_dct[species_name][1:] - species_arr_dct[species_name][:-1]
            est_arr = est_arr + np.asarray(deriv_wrt_species(*x_prev), dtype=float) * dx_arr
        estimates[1:] = est_arr
        return estimates

    def plotRateLawLinearization(self, reaction_id: str, ax=None, 
            is_reaction_rate : bool = True, is_plot: bool = True) -> None:
        '''Plots the piecewise linear rate estimates against the simulated rates.

        Args:
            reaction_id (str): The identifier of the reaction.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
            is_reaction_rate (bool, optional): Whether to plot the reaction rate or relative error in estimate relative error in estimate
        '''
        import matplotlib.pyplot as plt  # type: ignore

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        true_rates = self._timecourse_df[reaction_id]
        est_rates = self._makeRateLawEstimates(reaction_id)
        err_rate = (est_rates - true_rates).abs() / true_rates.abs()

        if is_reaction_rate:
            ax.plot(true_rates.index, cast(np.ndarray, true_rates.values), label='Simulated Rate', color='blue')
            ax.plot(est_rates.index, cast(np.ndarray, est_rates.values), label='Estimated Rate', color='orange', linestyle='--')
        else:
            ax.plot(err_rate.index, cast(np.ndarray, err_rate.values), label='Relative Error', color='red', linestyle='--')

        for boundary in self._boundaries:
            if 0 < boundary < len(true_rates):
                ax.axvline(x=true_rates.index[boundary], color='gray', linestyle=':', alpha=0.5)

        ax.set_title(f'Rate Law Linearization for Reaction {reaction_id}')
        ax.set_xlabel('Time')
        if is_reaction_rate:
            ax.set_ylabel('Rate')
        else:
            ax.set_ylabel('Relative Error')
            ax.set_ylim(0, 1)
        ax.legend()
        if is_plot:
            plt.show()

    def deprecatedplotAllRateLawLinearizations(self, num_col: int = 2, is_reaction_rate: bool = True) -> None:
        '''
        Plots the piecewise linear rate estimates against the simulated rates for all reactions.

        Args:
            ncol (int, optional): Number of columns in the subplot grid. Defaults to 2 
            is_reaction_rate (bool, optional):
                Whether to plot the reaction rate or relative error in estimate relative error in estimate
        '''
        import matplotlib.pyplot as plt  # type: ignore

        num_reactions = len(self._rate_law_dct)
        num_row = (num_reactions + num_col - 1) // num_col

        fig, axes = plt.subplots(num_row, num_col, figsize=(12, num_row * 4))
        axes = axes.flatten()

        i = 0
        for i, reaction_id in enumerate(self._rate_law_dct):
            ax = axes[i]
            self.plotRateLawLinearization(reaction_id, ax=ax, is_plot=False, is_reaction_rate=is_reaction_rate)
            if i//num_col < num_row - 1:
                ax.set_xlabel('')
                ax.set_xticklabels([])

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.legend(list(self._rate_law_dct.keys()))

        plt.tight_layout()
        plt.show()

    def plotAllRateLawLinearizations(self, ymax: float | None = None) -> None:
        '''
        Plots the piecewise linear rate estimates against the simulated rates for all reactions.

        Args:
            ncol (int, optional): Number of columns in the subplot grid. Defaults to 2 
            is_reaction_rate (bool, optional):
                Whether to plot the reaction rate or relative error in estimate relative error in estimate
        ''' 

        # Construct the true and estimated rate DataFrames, aligning them by dropping species columns from the true rates.
        true_rate_df = self._timecourse_df.copy()
        est_rate_df = self.makeEstimatesDF()
        species_cols = [col for col in true_rate_df.columns if col not in est_rate_df.columns]
        for col in species_cols:
            true_rate_df.drop(columns=col, inplace=True)
        #
        rel_error_df = (est_rate_df - true_rate_df).abs()
        dt_ser = true_rate_df.index.to_series().diff()
        dt_ser = dt_ser.fillna(0)
        dt = dt_ser.median()
        for col in rel_error_df.columns:
            rel_error_df[col] = rel_error_df[col] / true_rate_df[col].abs()
            rel_error_df[col] = rel_error_df[col] / dt
        rel_error_df = rel_error_df.fillna(0)  # Replace NaN with 0 for plotting

        # Plot the relative errors as a stacked area. fill_between draws one
        # polygon per reaction instead of one bar patch per time point, which
        # is orders of magnitude faster for long timecourses.
        _, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(rel_error_df))
        for col in rel_error_df.columns:
            top = bottom + rel_error_df[col].to_numpy()
            ax.fill_between(rel_error_df.index, bottom, top, label=col)
            bottom = top
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.set_xlabel('time')
        ax.set_ylabel('time normalized, relative error')
        #ax.set_ylim(0, rel_error_df.max().max())
        if ymax is not None:
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(0, 1)

        # Plot the segments
        for boundary in self._boundaries:
            if 0 < boundary < len(true_rate_df):
                ax.axvline(x=true_rate_df.index[boundary], color='gray', linestyle=':', alpha=0.5)