"""Makes plots of biomodels fits to 'best' changepoints."""

from src.timecourse_iterator import TimecourseIterator  # type: ignore
from src.psd_prediction_analyzer import PSDPredictionAnalyzer  # type: ignore

import argparse
import matplotlib  # type: ignore
matplotlib.use("Agg")  # Non-interactive backend for batch runs
import matplotlib.pyplot as plt  # type: ignore
import os


PLOT_DIR = "plots"


def main(first_model_num: int = 0, last_model_num: int = -1) -> None:
    '''
    Construct the piecewise plots with best changepoints.

    Args:
        first_model_num: The first model number to process (inclusive).
        last_model_num: The last model number to process (inclusive). If -1, process all models.
    '''
    os.makedirs(PLOT_DIR, exist_ok=True)
    t_iter = TimecourseIterator(first_model_num=first_model_num, last_model_num=last_model_num,
            is_report=True)
    for item in t_iter:
        try:
            analyzer = PSDPredictionAnalyzer(item.model_name,
                    timecourse_df=item.timecourse.timecourse_df)
            psd = analyzer.makeBestPSD()
        except Exception as e:
            print(f"Could not process {item.model_name}: {e}")
            continue
        po = psd.plotPiecewise(legend=False, num_true_point=60, is_nochangepoint_plot=False)
        path = os.path.join(PLOT_DIR, f"piecewise_{psd.model_num}.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.close(po.fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create piecewise prediction plots for BioModels.")
    parser.add_argument("--first_model_num", type=int, default=0)
    parser.add_argument("--last_model_num", type=int, default=-1)
    args = parser.parse_args()
    main(
        first_model_num=args.first_model_num,
        last_model_num=args.last_model_num,
    )
