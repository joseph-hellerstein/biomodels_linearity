'''Creates the plots used in the paper.'''

from src.biomodels_iterator import BiomodelsIterator
import src.constants as cn
from src.model import Model
from src.score import Score
from src.simulator import Simulator
from src.system_discovery import discoverNetwork
from src.perturbation_analyzer import PerturbationAnalyzer
from src.timecourse import Timecourse
from src.piecewise_system_discovery import PiecewiseSystemDiscovery

IS_PLOT = False
IS_ALL = False

if not IS_PLOT:
    import matplotlib # type: ignore
    matplotlib.use("PDF")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt # type: ignore
import numpy as np  # type: ignore
import os
import pandas as pd # type: ignore
from typing import Optional

NUM_POINT = cn.NUM_POINT

################ Data #####################
threshold = 0.001
filename = os.path.join(cn.DATA_DIR, f"perturbation_study-species{threshold}.csv")
df = pd.read_csv(filename)
DF_P_DCT = {p: df[p] for p in ['min', 'p10', 'p50']}
#

############### Helper Functions####################

def doPlot(model_num: int, poly_degree=1, threshold=0.001, species_names: Optional[list[str]] = None,
        num_point:int=NUM_POINT):
    start_time = 0
    model = Model.makeBiomodel(model_num=model_num)
    timecourse = Timecourse(model, start_time=start_time, num_point=num_point)
    sdr = discoverNetwork(timecourse.timecourse_df, poly_degree=poly_degree, threshold=threshold,
            plot_species_names=species_names, is_plot=IS_PLOT, subtitle=f"BioModel {model_num}",
            is_plot_heatmap=False, is_print_equations=False, is_plot_comparisons=True, is_print_accuracy=False)
    return sdr

def plotPerturbation(metric_name: str, threshold: float = 0.001, is_analyze_species: bool = True,
        perturbations: list[float] = [-0.50, -0.10, 0.0, 0.10, 0.50],
        is_plot: bool = True) -> None:
    """Plots CDF for a specified metric.

    Args:
        metric_name (str): The name of the metric to plot.
        threshold (float): The threshold for the perturbation study.
        is_analyze_species (bool): If True, plots for species scores; if False, plots for species scores.
    """
    # Get the data
    if is_analyze_species:
        path = os.path.join(cn.DATA_DIR, f"perturbation_study-species{threshold}.csv")
        aggregation_type = "species"
    else:
        path = os.path.join(cn.DATA_DIR, f"perturbation_study-model{threshold}.csv")
        aggregation_type = "model"
    full_df = pd.read_csv(path)
    # Construct the plot
    pivot_df = full_df.pivot(columns='perturbation', values=metric_name,
            index=['system_id', 'aggregation_type'])
    for perturbation in perturbations:
        if perturbation not in pivot_df.columns:
            raise ValueError(f"Perturbation {perturbation} not found in data.")
        value_arr = pivot_df[perturbation].dropna().to_numpy()
        if len(value_arr) == 0:
            raise ValueError(f"No data found for perturbation {perturbation}.")
        _ = Score.plotCDFArray(value_arr, is_plot=True)
    plt.legend([str(v) for v in perturbations])
    plt.title(f"{metric_name} {aggregation_type} CDF")
    if is_plot:
        plt.show()

################################################
# Linear Fits
################################################
############### Time course ####################
if IS_ALL:
    sdr = doPlot(968, species_names=["SOCS1", "IL7IL7RJAK1"])
    sdr.fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_968.pdf"), bbox_inches="tight", dpi=300) # type: ignore
    ##
    sdr = doPlot(1004, species_names= ["IL6ext", "STAT3mRNA"])
    sdr.fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_1004.pdf"), bbox_inches="tight", dpi=300) # type: ignore
#
############### CDFs ####################
if IS_ALL:
    filename = os.path.join(cn.DATA_DIR, "linear_predictor_scores-0.001.csv")
    score = Score.deserialize(filename)
    fig = score.plotCDF(["min", "p10", "p50", "max"], xlabel= "model accuracy", is_plot_species=False,
            is_plot_model=True, title=f"BioModel Models").fig
    if IS_PLOT:
        plt.show()
    fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_model_cdf.pdf"), bbox_inches="tight", dpi=300) # type: ignore
    #
    fig = score.plotCDF(["min", "p10", "p50", "max"], xlabel= "model accuracy", is_plot_species=True,
            is_plot_model=False, title=f"BioModel Species").fig
    fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_species_cdf.pdf"), bbox_inches="tight", dpi=300) # type: ignore
    if IS_PLOT:
        plt.show()

################################################
# Perturbations
################################################
############### Time course ####################
if IS_ALL:
    analyzer = PerturbationAnalyzer(1004, perturbations=[-50, -10, 0, 10, 50])
    analyzer.plotTimeseries(subtitle="BioModel 1004", plot_species_names=["IL6ext","IL6int"])
    analyzer.result.fig.savefig(os.path.join(cn.PAPER_DIR, "perturbation_fit_1004.pdf"), bbox_inches="tight", dpi=300) # type: ignore
    #
    analyzer = PerturbationAnalyzer(968, perturbations=[-50, -10, 0, 10, 50])
    analyzer.plotTimeseries(frac_scatter_skip=0.05,
            subtitle=f"BioModel 968", plot_species_names=["SOCS1", "IL7IL7RJAK1"])
    analyzer.result.fig.savefig(os.path.join(cn.PAPER_DIR, "perturbation_fit_968.pdf"), bbox_inches="tight", dpi=300) # type: ignore
    #
    for percentile in ["p10", "p50"]:
        for aggregation in ["model", "species"]:
            if aggregation == "species":
                is_analyze_species = True
            else:
                is_analyze_species = False
            plotPerturbation(percentile, is_analyze_species=is_analyze_species, is_plot=IS_PLOT)
            fig = plt.gcf()
            filename = f"perturbation_{aggregation}_{percentile}_cdf.pdf"
            fig.savefig(os.path.join(cn.PAPER_DIR, filename), bbox_inches="tight", dpi=300) # type: ignore
            plt.close()


################################################
# Perturbations
################################################
model_num = 1045
model = Model.makeBiomodel(model_num=model_num)
timecourse = Timecourse(model, num_point=cn.NUM_POINT)
df = timecourse.timecourse_df
for max_changepoint in [0, 5, 10, 20]:
    for max_fractional_reduction in [0.001, 0.01]:
        psd = PiecewiseSystemDiscovery(df, max_changepoint=max_changepoint,
                max_fractional_reduction=max_fractional_reduction,
                model_name=str(model_num))
        psd.fit()
        fig = psd.plotPiecewise(num_true_point=60, suptitle="",
                is_nochangepoint_plot=False).fig
        filename = f"piecewise_prediction_1045_{max_changepoint}_{max_fractional_reduction}_{max_changepoint}.pdf"
        path = os.path.join(cn.PAPER_DIR, filename)
        fig.savefig(path, bbox_inches="tight", dpi=300)  # type: ignore