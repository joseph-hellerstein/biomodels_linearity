"""Plot the models in BioModels."""
from __future__ import annotations

import csv

from src.biomodels_iterator import BiomodelsIterator, getBiomodelsEndtimes  # type: ignore
from src.model import Model  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import os


NUM_COL = 5
NUM_ROW = 5
PLOT_DIR = "plots"

# Coding scheme for end_time_source
SOURCE_CODE_MAP = {
    "max_median_cv": "MM",
    "sedml": "SM",
    "steadystate": "SS",
}

ENDTIME_CSV = os.path.join("data", "biomodels_endtime.csv")

def _load_endtime_data() -> dict[str, tuple[str, float]]:
    """Load end_time data from CSV. Returns {model_name: (source_code, end_time)}.

    Note: getBiomodelsEndtimes returns (end_time_float, source_string) tuples,
    so we swap them to match the declared return type (source_code, end_time).
    """
    raw_dct = getBiomodelsEndtimes(endtimes_csv_path=ENDTIME_CSV,
            is_include_endtime_source=True)
    # Swap from (end_time_float, source_string) -> (source_string, end_time_float)
    result_dct: dict[str, tuple[str, float]] = {}
    for k, v in raw_dct.items():
        if isinstance(v, tuple) and len(v) == 2:
            # v is (end_time_float, source_string), swap to (source_string, end_time_float)
            result_dct[k] = (v[1], float(v[0]))
    return result_dct


def _extract_model_number(model_name: str) -> str:
    '''Extract the numeric part from a model name like "BIOMD0000000001" → "1".'''
    if "BIOMD" in model_name.upper():
        prefix = "BIOMD" if "BIOMD" in model_name else "biomd"
        idx = model_name.upper().index("BIOMD")
        numeric_part = model_name[idx + len(prefix):]
        return str(int(numeric_part))  # strip leading zeros
    return model_name


def _get_endtime_info(model_name: str, endtime_data: dict[str, tuple[str, float]]) -> tuple[str, float]:
    '''Get (source_code, end_time) for a model. Returns ("", 0.0) if not found.'''
    info = endtime_data.get(model_name)
    if info is None:
        return "", 0.0
    return info


def _plot_single_model(ax, model: Model, endtime_data: dict[str, tuple[str, float]]) -> None:
    '''Plot all species for one model on the given axes.

    Title shows just the model number (e.g., "1" from "BIOMD0000000001").
    End_time info is displayed below the title using a coding scheme and value.
    No legend, no x-tick labels, no y-tick labels.
    '''
    timecourse = TimecourseIterator().getTimecourse(model.model_name)

    # Extract just the model number for the title
    model_num = _extract_model_number(model.model_name)

    # Get end_time info (note: getBiomodelsEndtimes returns (end_time_float, source_string))
    source_code, end_time_val = _get_endtime_info(model.model_name, endtime_data)

    # Build title: model number only
    ax.set_title(f"{model_num}", fontsize=10)

    # Display end_time info below the title using annotate to avoid overflow
    if source_code and end_time_val > 0:
        # Format end_time: use integer if whole number, else 2 decimal places
        if end_time_val == int(end_time_val):
            et_str = str(int(end_time_val))
        else:
            et_str = f"{end_time_val:.2f}"
        info_text = f"[{source_code}] t={et_str}"
        ax.annotate(
            info_text,
            xy=(0.5, -0.18),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=7,
            color="darkred",
        )

    ax.set_xticks([])
    ax.set_yticks([])

    # Plot one line per species from the timecourse dataframe, normalized by std dev.
    try:
        for species in model.species_names:
            if species in timecourse.timecourse_df.columns:
                values = timecourse.timecourse_df[species]
                std = float(values.std())
                mean = float(values.mean())
                if std == 0:
                    ax.plot(timecourse.timecourse_df.index, [0.0] * len(values))
                else:
                    ax.plot(timecourse.timecourse_df.index, (values - mean) / std)
    except Exception as e:
        print(f"Error plotting {model.model_name}: {e}")


def _flush_page(
    page_models: list[Model],
    endtime_data: dict[str, tuple[str, float]],
    page_idx: int,
) -> None:
    '''Save a partially or fully filled 5x5 grid figure.'''
    if not page_models:
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=NUM_ROW, ncols=NUM_COL, figsize=(12, 8)
    )

    num_per_page = NUM_ROW * NUM_COL
    for idx, model in enumerate(page_models):
        irow = idx // NUM_COL
        icol = idx % NUM_COL
        ax = axes[irow, icol]
        _plot_single_model(ax, model, endtime_data)

    # Hide any unused subplots
    for idx in range(len(page_models), num_per_page):
        irow = idx // NUM_COL
        icol = idx % NUM_COL
        axes[irow, icol].set_visible(False)

    fig.tight_layout()

    filename = os.path.join(PLOT_DIR, f"plot_biomodels_{page_idx}.png")
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close(fig)


def main(max_model: int = -1) -> None:
    '''
    Plot all models from BioModels. Plots are constructed incrementally as
    models are accumulated — a new figure is started every 25 models and
    flushed immediately when full. Any remaining models on a partial page
    are saved at the end.

    Args:
        max_model (int): Number of models to plot. If -1, plot all models.
    '''
    endtime_data = _load_endtime_data()
    if endtime_data:
        print(f"Loaded end_time data for {len(endtime_data)} models")

    page_models: list[Model] = []
    page_idx = 0
    num_per_page = NUM_ROW * NUM_COL

    for num_model, item in enumerate(BiomodelsIterator(is_report=True)):
        if not item.sbml_paths:
            continue
        if max_model > 0 and num_model >= max_model:
            break
        try:
            with open(item.sbml_paths[0], "r") as f:
                sbml_str = f.read()
            model = Model(model_str=sbml_str, model_name=item.model_name)
        except Exception as e:
            print(f"Error loading {item.model_name}: {e}")
            continue

        page_models.append(model)

        # When the current page is full, save it and start a new one
        if len(page_models) == num_per_page:
            _flush_page(page_models, endtime_data, page_idx)
            page_idx += 1
            page_models = []

    # Flush any remaining models on a partial final page
    if page_models:
        _flush_page(page_models, endtime_data, page_idx)


if __name__ == "__main__":
    main()
