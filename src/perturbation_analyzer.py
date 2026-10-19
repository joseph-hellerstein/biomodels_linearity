"""Performs perturbation analysis on BioModels using SINDy-based system discovery.

This module analyzes how well a discovered ODE system generalizes to perturbed
initial conditions, quantifying robustness of the model across different starting values.
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Union, Tuple, Optional  # type: ignore

from collections import namedtuple
import src.constants as cn  # type: ignore
from src.model import Model  # type: ignore
from src.score import Score  # type: ignore
from src.system_discovery import NULL_DF, SystemDiscovery  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

PlotRecord = namedtuple("PlotRecord", ["perturbation", "test_df", "pred_df", "accuracy_ser"])


# Named tuple used by perturbation analysis result rows.
class AnalyzePerturbationsResult:
    """Named tuple for perturbation analysis results.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame containing accuracy metrics for each perturbation and aggregation type.
    fig : matplotlib.figure.Figure | None
        Figure containing the time series plots for each perturbation. None if no plots were generated.
    """
    def __init__(self, df: pd.DataFrame, fig):
        self.df = df
        self.fig = fig

# Scatter-plot density step for perturbation analysis (only used by analyze_perturbations)
DEFAULT_FRAC_KEEP: float = 0.2

# Column name for fraction of species perturbed in results.
COL_FRACTION_SPECIES_PERTURBABLE: str = "fraction_species_perturbable"
class PerturbationAnalyzer:
    """Performs perturbation analysis on a BioModel.

    Fits SINDy models to unperturbed timecourse data and evaluates accuracy against
    perturbed versions of the original system. It quantifies how well the discovered model
    generalizes across different initial starting values.

    Example usage::

        analyzer = PerturbationAnalyzer(
            model=1004, perturbations=[-50, -10, 0, 10, 50], frac_scatter_skip=0.05
        )
        print(analyzer.result.df)
    """

    def __init__(
        self,
        model: Union[Model, int],
        training_df=NULL_DF,
        threshold: float = 0.001,
        num_point: int = cn.NUM_POINT,
        perturbations: Optional[list[float]] = None,
        perturbation_species_fraction: float = 1.0,
        fraction_species_perturbable: float = 1.0,
        col_percentile: str = cn.COL_P10,
        poly_degree: int = 1,
        is_analyze_model: bool = True,
        is_analyze_species: bool = True,
    ) -> None:
        """Configure and immediately run perturbation analysis.

        All arguments mirror :meth:`analyze_perturbations`.  The
        :class:`AnalyzePerturbationsResult` is stored on the instance as
        :attr:`self.result`.
        """
        if isinstance(model, int):
            model = Model.makeBiomodel(model_num=model)
        self.model = model
        if training_df is NULL_DF:
            training_df = TimecourseIterator().getTimecourse(model.model_name, num_point=num_point).timecourse_df
        self.num_point = len(training_df)
        self.end_time = training_df.index.to_list()[-1]
        self.training_df = training_df
        self.species_names = training_df.columns.tolist()
        self.start_time = training_df.index[0]
        self.num_point = len(training_df)
        self.threshold = threshold
        self.perturbations = (
            perturbations
            if perturbations is not None
            else [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]
        )
        self.perturbation_species_fraction = perturbation_species_fraction
        self.fraction_species_perturbable = fraction_species_perturbable
        self.col_percentile = col_percentile
        self.is_analyze_model = is_analyze_model
        self.is_analyze_species = is_analyze_species
        self.poly_degree = poly_degree

        # Default perturbations to the same list used by analyze_perturbations.
        self.perturbations: list[float] = (
            perturbations if perturbations is not None else [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]
        )
        # Do the analysis
        self.plot_records, self.result = self._analyze_perturbations()

    def _analyze_perturbations(self) -> Tuple[list[PlotRecord], AnalyzePerturbationsResult]:
        """Fit on training_df; evaluate timecourse accuracy on perturbed timecourses.

        For each value in *perturbations* a fresh Timecourse is simulated from
        *model* with that perturbation_value_fraction.  The SINDy model
        (fitted on the unperturbed *training_df*) is evaluated against each
        perturbed timecourse using absolute relative error (ARE) accuracy.

        When *is_analyze_model* and/or *is_analyze_species* are True, the returned DataFrame
        includes rows at that level of aggregation:
            - COL_AGGREGATION_TYPE == 'model'  -- one model-level row per
              perturbation value with aggregated statistics across species.
            - Species-level rows (aggregation type = species name) -- one row
              per species per perturbation with per-species ARE values.

        By default both levels are included. Set is_analyze_model=False or
        is_analyze_species=False to drop the corresponding level from the output.


        Parameters
        ----------
        model : Model
            Used to simulate ground-truth timecourses for each perturbation.
        training_df : pd.DataFrame
            Unperturbed timecourse used to fit the SINDy model.
        threshold : float
            STLSQ sparsity threshold.
        perturbations : list[float]
            Signed fractional perturbation values (e.g. [-0.05, 0.0, 0.05]).
        col_percentile : str
            The column name in the accuracy dataframe
        perturbation_species_fraction : float
            Fraction of species whose initial values are perturbed if their initial value > 0.
        figsize : tuple, optional
            Figure size in inches.  Auto-sized when None.
        poly_degree : int
            Degree of the polynomial library.
        frac_scatter_skip : float
            Scatter-plot density: num_skip = max(1, int(n_points * frac_scatter_skip)).
        is_plot : bool
            Show a trajectory comparison figure when True.
        is_analyze_model : bool
            Include model-level rows (aggregation_type='model') in the returned DataFrame.
        is_analyze_species : bool
            Include per-species rows (aggregation_type=species name) in the returned DataFrame.
        plot_species_names : list[str] | None
            List of species to plot. If None, all species are plotted.

        Returns
        -------
        AnalyzePerturbationsResult
            A named tuple containing the accuracy metrics and the plot figure.
                pd.DataFrame: accuracy metrics — one row per perturbation and aggregation type
                fig
        """
        # Initializations
        self.plot_records: list[PlotRecord] = []
        # Create the SystemDiscovery
        self.system_discovery = SystemDiscovery(self.training_df,
            coefficient_threshold=self.threshold, poly_degree=self.poly_degree)
        self.system_discovery.fit()
        # Create the perturbation timecourses
        score = Score()
        for p in self.perturbations:
            tc = Timecourse(
                model=self.model,
                start_time=self.start_time,
                end_time=self.end_time,
                num_point=self.num_point,
                perturbation_value_fraction=p,
                perturbation_species_fraction=self.perturbation_species_fraction,
            )
            test_df = tc.timecourse_df
            self.system_discovery._checkColumns(  # pylint: disable=protected-access
                test_df.columns.tolist(),
            )
            pred_df = None
            try:
                pred_df = self.system_discovery.predict(test_df)
                score.add(test_df, pred_df, system_id=str(p))
            except (ValueError, RuntimeError):
                print(f"Warning: perturbation {p} failed; skipping.")
            if pred_df is not None:
                new_df = score.score_df[score.score_df[cn.COL_SYSTEM_ID] == str(p)]
                new_ser = new_df[self.col_percentile]
                new_ser.index = new_df[cn.COL_AGGREGATION_TYPE]
                self.plot_records.append(PlotRecord(p, test_df, pred_df, new_ser))

        # Build per-row records so each row carries its own perturbation value.
        meta_cols = [cn.COL_PERTURBATION, COL_FRACTION_SPECIES_PERTURBABLE, cn.COL_SYSTEM_ID]
        rows: list[dict] = []
        if not score.score_df.empty:
            for p in self.perturbations:
                if self.is_analyze_model:
                    model_row_df = (
                        score.score_df[
                            (score.score_df[cn.COL_SYSTEM_ID] == str(p)) &
                            (score.score_df[cn.COL_AGGREGATION_TYPE]
                             == cn.COL_AGGREGATION_TYPE_MODEL)
                        ]
                    )
                    if not model_row_df.empty:
                        for _, r in model_row_df.iterrows():
                            rec = {**r.to_dict(), **{
                                cn.COL_PERTURBATION: p,
                                COL_FRACTION_SPECIES_PERTURBABLE: self.fraction_species_perturbable,
                                cn.COL_SYSTEM_ID: self.model.model_name,
                            }}
                            rows.append(rec)
                if self.is_analyze_species:
                    species_row_df = (
                        score.score_df[
                            (score.score_df[cn.COL_SYSTEM_ID] == str(p)) &
                            (score.score_df[cn.COL_AGGREGATION_TYPE]
                            != cn.COL_AGGREGATION_TYPE_MODEL)
                        ]
                    )
                    if not species_row_df.empty:
                        for _, r in species_row_df.iterrows():
                            rec = {**r.to_dict(), **{
                                cn.COL_PERTURBATION: p,
                                COL_FRACTION_SPECIES_PERTURBABLE: self.fraction_species_perturbable,
                                cn.COL_SYSTEM_ID: self.model.model_name,
                            }}
                            rows.append(rec)
        accuracy_df = pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=list(dict.fromkeys(meta_cols + list(score.score_df.columns))))
        #
        return (self.plot_records, AnalyzePerturbationsResult(df=accuracy_df, fig=None))

    def _plot_single_species(self, ax, sp_name, num_skip, is_label: bool = True):
        """Plot test + predicted trajectories for *sp_name* across all perturbations."""
        sp_idx = self.model.species_names.index(sp_name)
        system_discovery = self.system_discovery
        sp_col = system_discovery.species_cols[sp_idx]
        for p_idx, (p, test_df, pred_df, accuracy_ser) in enumerate(self.plot_records):
            finite = np.isfinite(accuracy_ser.loc[sp_name])
            acc = f"{accuracy_ser.loc[sp_name]:.3f}" if finite else "0"
            color = f"C{p_idx}"
            label = f"{p:.2f}: {acc}" if is_label else None
            ax.scatter(
                test_df.index[::num_skip],
                test_df[sp_col][::num_skip],
                s=10, color=color, alpha=0.6,
                label=label,
            )
            if pred_df is not None:
                ax.plot(
                    pred_df.index, pred_df[sp_name],
                    linestyle="--", color=color, lw=1.5,
                )

    def _hide_excess_axes(self, axes, n_species, nrows, ncols):
        """Hide subplot slots that have no species to plot."""
        for idx in range(n_species, nrows * ncols):
            ax_row, ax_col = divmod(idx, ncols)
            axes[ax_row][ax_col].set_visible(False)

    def plotTimeseries(self,
            figsize: Optional[tuple[float, float]] = None,
            subtitle: str = "",
            frac_scatter_skip: float = DEFAULT_FRAC_KEEP,
            plot_species_names: Optional[list[str]] = None,
            ) -> None:
        """Plot the time series for each perturbation.

        Parameters
        ----------
        figsize : tuple[float, float] | None
            Figure size in inches.  Auto-sized when None.
        subtitle : str
            Subtitle for the plot.
        frac_scatter_skip : float
            Fraction of points to skip in the scatter plot.
        plot_species_names : list[str] | None
            List of species names to plot. If None, all species are plotted.
        """
        if not self.plot_records:
            print("No plot records available. Run analyze_perturbations first.")
            return

        if figsize is None:
            figsize = (10, 6)

        if plot_species_names is None:
            plot_species_names = self.species_names
        num_point = len(self.training_df)

        n = len(plot_species_names) # type: ignore
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        if figsize is None:
            figsize = (5 * ncols, 3.5 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        fig.suptitle(subtitle, fontsize=14)
        num_skip = max(1, int(num_point * frac_scatter_skip))
        for pos_idx, sp_name in enumerate(plot_species_names):
            #is_label = True if pos_idx == 0 else False
            is_label = True
            ax_row, ax_col = divmod(pos_idx, ncols)
            ax = axes[ax_row][ax_col]
            self._plot_single_species(ax, sp_name, num_skip, is_label)
            ax.set_title(sp_name)
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        self._hide_excess_axes(axes, n, nrows, ncols)
        self.result.fig = fig
        fig.tight_layout()