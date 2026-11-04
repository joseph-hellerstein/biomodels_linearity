'''Creates the plots used in the paper.'''

from src.biomodels_iterator import BiomodelsIterator
import src.constants as cn
from src.model import Model
from src.perturbation_analyzer import PerturbationAnalyzer
from src.piecewise_system_discovery import PiecewiseSystemDiscovery
from src.psd_analyzer import PSDAnalyzer
from src.score import Score
from src.simulator import Simulator
from src.system_discovery import discoverNetwork
from src.timecourse import Timecourse
from src.cumulative_segment_curve import CumulativeSegmentCurve

IS_PLOT = True
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
""" if IS_ALL:
    sdr = doPlot(968, species_names=["SOCS1", "IL7IL7RJAK1"])
    sdr.fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_968.pdf"), bbox_inches="tight", dpi=300) # type: ignore
    ##
    sdr = doPlot(1004, species_names= ["IL6ext", "STAT3mRNA"])
    sdr.fig.savefig(os.path.join(cn.PAPER_DIR, "linear_fit_1004.pdf"), bbox_inches="tight", dpi=300) # type: ignore
# """
""" ############### CDFs ####################
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
"""
""" 
################################################
# Perturbations
################################################
if IS_ALL:
    model_num = 1045
    model = Model.makeBiomodel(model_num=model_num)
    timecourse = Timecourse(model, num_point=cn.NUM_POINT)
    df = timecourse.timecourse_df
    for max_changepoint in [0, 5, 10, 20]:
        for max_fractional_reduction in [0.001, 0.01]:
            psd = PiecewiseSystemDiscovery(df, max_changepoint=max_changepoint,
                    max_fractional_reduction=max_fractional_reduction,
                    model_id=str(model_num))
            psd.fit()
            fig = psd.plotPiecewise(num_true_point=60, suptitle="",
                    is_nochangepoint_plot=False).fig
            filename = f"piecewise_prediction_1045_{max_changepoint}_{max_fractional_reduction}_{max_changepoint}.pdf"
            path = os.path.join(cn.PAPER_DIR, filename)
            fig.savefig(path, bbox_inches="tight", dpi=300)  # type: ignore
"""
#######################
# Examples of piecewise predictions
#######################
model_nums = [799, 799, 799, 799]    # test only
model_nums = [745, 904, 821, 697]
if IS_ALL:
    t_fig, t_axes = plt.subplots(1, 4, figsize=(10, 4), constrained_layout=True)
    extended_plot_options_list: list = []
    for idx, model_num in enumerate(model_nums):
        ax = t_axes[idx]
        extended_plot_options = PSDAnalyzer.plotBestLinearFits(model_id=model_num, ax=ax,
                min_accuracy=0.8, accuracy_statistic="p50", num_true_point=15)
        extended_plot_options_list.append(extended_plot_options)
        text = "1 segment" if extended_plot_options.num_changepoint == 0 else f"{extended_plot_options.num_changepoint + 1} segments"
        title = f"{model_num}: {text}"
        ax.set_title(title)
        if idx > 0:
            ax.set_ylabel("")
            ax.set_xlabel("")
        ax.legend([])
    filename = "example_timecourse.pdf"
    path = os.path.join(cn.PAPER_DIR, filename)
    t_fig.savefig(path, bbox_inches="tight", dpi=300)  # type: ignore
    plt.close()
    # Do the segment curve
    c_fig, c_axes = plt.subplots(1, 4, figsize=(10, 4), constrained_layout=True)
    for idx, model_num in enumerate(model_nums):
        extended_plot_options = extended_plot_options_list[idx]
        ax = c_axes[idx]
        boundaries = extended_plot_options.psd.boundaries
        csc = CumulativeSegmentCurve(boundaries)
        csc.plot(ax=ax)
        title = f"AUC={extended_plot_options.auc:.2f}"
        ax.set_title(title)
        ax.set_xlabel("Normalized Segment Length")
        ax.set_ylabel("Cumulative Area")
        ax.legend([])
        if idx > 0:
            ax.set_ylabel("")
            ax.set_xlabel("")
    filename = "example_segment_curve_examples.pdf"
    path = os.path.join(cn.PAPER_DIR, filename)
    c_fig.savefig(path, bbox_inches="tight", dpi=300)  # type: ignore
    plt.close()

########################
# Distribution of longest segment for models with different min_accuarcy
########################
if True:
    _, ax = PSDAnalyzer.makeLongestSegmentCDFPlot(min_accuracy=0.8, accuracy_stat="p50")
    _, _ = PSDAnalyzer.makeLongestSegmentCDFPlot(min_accuracy=0.9, accuracy_stat="p50", ax=ax)
    ax.legend(["min_accuracy=0.8", "min_accuracy=0.9"])
    filename = "psd_longest_segment_cdf.pdf"
    path = os.path.join(cn.PAPER_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=300)  # type: ignore
    plt.close()

#######################
# Accuracy CDF and number of changepoints
######################"
if IS_ALL:
    path = os.path.join(cn.PSD_DATA_DIR,
            "piecewise_predictions__numpoint_100000__threshold_0.001__removal_0__manycp_1__repeat_1.csv")
    df = pd.read_csv(path)
    MAX_CHANGEPOINTS = [0, 5, 50, 500, 5000]
    for idx, max_changepoint in enumerate(MAX_CHANGEPOINTS):
        mask = df[cn.COL_MAX_CHANGEPOINT] == max_changepoint
        mask &= df[cn.COL_AGGREGATION_TYPE] == cn.COL_AGGREGATION_TYPE_MODEL
        values = np.sort(df[mask][cn.COL_P50].to_numpy())
        _ = Score.plotCDFArray(values, is_plot=True)
    legend = [f"{max_changepoint + 1} segments" for max_changepoint in MAX_CHANGEPOINTS]
    legend[0] = "1 segment"
    plt.legend(legend)
    plt.ylabel("fraction of models")
    filename = "piecewise_prediction_accuracy_cdf.pdf"
    path = os.path.join(cn.PAPER_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=300)  # type: ignore
    plt.close()



######################
# AUC Distribution
######################
if IS_ALL:
    PSDAnalyzer.makeAUCCDFPlots()
    filename = "piecewise_auc_cdf.pdf"
    path = os.path.join(cn.PAPER_DIR, filename)
    plt.savefig(path, bbox_inches="tight", dpi=300)  # type: ignore
    plt.close()