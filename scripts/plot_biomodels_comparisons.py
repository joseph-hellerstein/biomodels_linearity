"""
Plots comparisons of different estimates of characteristics times for
BioModels models.
 """
from src.biomodels_iterator import BiomodelsIterator
from src.model import Model # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import os


NUM_COL = 5
NUM_ROW = 2
PLOT_DIR = "plots/biomodels_comparisons"


def _flush_page(fig, page_idx: int) -> None:
    try:
        fig.tight_layout()
        filename = os.path.join(PLOT_DIR, f"plot_comparisons_{page_idx}.png")
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved {filename}")
        plt.close(fig)
    except Exception as e:
        print(f"Error occurred while saving page {page_idx}: {e}")


def main(first_model_num: int = 409, last_model_num: int = -1) -> None:
    '''
    Does comparison plots for models. A comparison plot has 3 panels.

    Args:
        first_model_num (int): The number of the first model to plot.
        last_model_num (int): The number of the last model to plot. If -1, plot all models.
    '''
    os.makedirs(PLOT_DIR, exist_ok=True)
    page_idx = 38
    page_models: list[Model] = []
    model_idx = 0  # index of the model on the current page
    num_per_page = NUM_ROW * NUM_COL
    num_plot_per_model = 3

    fig, axes = plt.subplots(nrows=NUM_ROW*num_plot_per_model,
            ncols=NUM_COL, figsize=(12, 8))
    # BiomodelsIterator's "no limit" convention is int(1e9), not -1.
    effective_last = int(1e9) if last_model_num == -1 else last_model_num
    for item in BiomodelsIterator(
            is_report=True, first_model_num=first_model_num, last_model_num=effective_last):
        if not item.sbml_paths:
            continue
        # Calculate the row and column indices for the current model's plots
        irow = (model_idx // NUM_COL) * num_plot_per_model
        icol = model_idx % NUM_COL
        # Process the model
        ax_sb = axes[irow, icol]
        ax_ss = axes[irow+1, icol]
        ax_mc = axes[irow+2, icol]
        try:
            model = Model.makeBiomodel(item.model_name)
        except Exception as e:
            print(f"Error occurred while creating model {item.model_name}: {e}")
            continue
        ax_sb.set_title(f"{item.model_num}", fontsize=10, fontweight='bold')
        page_models.append(model)
        model_idx += 1
        # When the current page is full, save it and start a new one
        if model_idx >= num_per_page:
            model_idx = 0
            _flush_page(fig, page_idx)
            fig, axes = plt.subplots(nrows=NUM_ROW*num_plot_per_model,
                    ncols=NUM_COL, figsize=(12, 8))
            page_idx += 1

    # Flush any remaining models on a partial final page
    if model_idx > 0:
        _flush_page(fig, page_idx)


if __name__ == "__main__":
    main()