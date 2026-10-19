'''Creates piecewise predictions for elements of BioModels.'''

"""
Usage:
    python scripts/make_piecewise_predictions.py --first_model_num 0 --last_model_num 100 --min_segment_length 50
"""

import src.constants as cn  # type: ignore
from src.score import Score  # type: ignore
from src.piecewise_system_discovery import PiecewiseSystemDiscovery  # type: ignore
from src.timecourse_iterator import TimecourseIterator, TimecourseIteratorItem  # type: ignore

import argparse
import os
import pandas as pd  # type: ignore
from typing import List, Optional

EXCLUDED_MODELS: List[str] = [
    "BIOMD0000000339",
]
IS_CHANGEPONT_REMOVAL = False # Whether to remove change points that do not significantly improve the model.
MANY_MAX_CHANGEPOINTS = [0, 1, 2, 3, 4, 5, 10, 50, 500, 5000]
ONE_MAX_CHANGEPOINT = [5000]
MAX_FRACTIONAL_REDUCTION = 0.01  # Maximum fractional reduction in the sum of squared errors required to accept a new change point.
COEFFICIENT_THRESHOLD = 0.001  # Threshold for coefficient magnitude to consider a species as linear.
NUM_POINT = 100000

#################################################################
# Functions
#################################################################
def makeFilePath(num_point: int, coefficient_threshold: float,
        is_changepoint_removal: bool, max_fractional_reduction: float,
        is_many_maxchangepoints: bool) -> str:
    filename = "piecewise_predictions"
    filename += f"__numpoint_{num_point}"
    filename += f"__threshold_{coefficient_threshold}"
    filename += f"__removal_{int(is_changepoint_removal)}"
    filename += f"__maxreduction_{max_fractional_reduction}"
    filename += f"__manycp_{int(is_many_maxchangepoints)}"
    filename += ".csv"
    return filename


#################################################################
# Preliminaries
#################################################################
if os.path.isfile(os.path.join(cn.DATA_DIR, "badmodels.txt")):
    with open(os.path.join(cn.DATA_DIR, "badmodels.txt"), "r") as f:
        for line in f:
            model_name = line.strip()
            if model_name and not model_name.startswith("#"):
                EXCLUDED_MODELS.append(model_name.upper())


#################################################################
# Processing for a single model and parameters
#################################################################
def processModel(
        item: TimecourseIteratorItem,
        max_changepoint: int,
        coefficient_threshold: float,
        max_fractional_reduction: float = MAX_FRACTIONAL_REDUCTION,
        is_changepoint_removal: bool = IS_CHANGEPONT_REMOVAL,
) -> Optional[pd.DataFrame]:
    """
    Process a single item

    Args:
        item (TimecourseIteratorItem): Information on current model and its timecourse.
        max_changepoint (int): Maximum number of change points to consider.
        max_fractional_reduction (float): Maximum fractional reduction in the sum of squared errors required to accept a new change point.
        coefficient_threshold (float): Threshold for the coefficient of determination (R-squared) to consider a model valid.

    Returns:
        Optional[pd.DataFrame]: _description_
    """
    model_name = item.model_name
    df = item.timecourse.timecourse_df
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"Skipping {model_name} (no timecourse data)")
        return None
    try:
        psd = PiecewiseSystemDiscovery(df,
                max_changepoint=max_changepoint,
                model_name=model_name,
                coefficient_threshold=coefficient_threshold,
                is_changepoint_removal=is_changepoint_removal,
                max_fractional_reduction=max_fractional_reduction,
        )
        psd.fit()
        pred_df = psd.predict()
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        return None
    if pred_df is None:
        print(f"Skipping {item.model_name} (no new data)")
        return None
    # Create the score
    score = Score()
    score.add(df, pred_df, system_id=model_name)
    accuracy_df = score.score_df
    # Augment the dataframe with additional columns for the model
    accuracy_df[cn.COL_SYSTEM_ID] = model_name
    accuracy_df[cn.COL_CHANGEPOINTS] = str(psd.changepoints)
    accuracy_df[cn.COL_MAX_CHANGEPOINT] = max_changepoint
    accuracy_df[cn.COL_MAX_FRACTIONAL_REDUCTION] = max_fractional_reduction
    accuracy_df[cn.COL_COEFFICIENT_THRESHOLD] = coefficient_threshold
    accuracy_df[cn.COL_NUM_CHANGEPOINT] = psd.num_changepoint  # Number of change points detected in the piecewise model. 
    accuracy_df[cn.COL_IS_CHANGEPOINT_REMOVAL] = is_changepoint_removal
    accuracy_df[cn.COL_NUM_SPECIES] = len(psd.species_names)
    #
    return accuracy_df


#################################################################
# Iterate across models and make piecewise predictions
#################################################################
def main(
        process_idx: int = 0,
        first_model_num: int = 0,
        last_model_num: int = int(1e9),
        is_initialize: bool = False, # Ignore existing serialized Timecourse when initializing (for testing).
        coefficient_threshold: float = COEFFICIENT_THRESHOLD,
        max_fractional_reduction: float = 0.01,
        is_many_maxchangepoints: bool = False,  # 0 means "accept any ASS reduction" — aggressive batch mode across thousands of models.
        output_path: str = cn.PIECEWISE_PREDICTIONS_PATH,
        is_changepoint_removal: bool = IS_CHANGEPONT_REMOVAL,
) -> None:
    '''
    Main function to make piecewise predictions. Iterate across models in the timecourse zip file, and for each model, fit a piecewise linear model
    and make predictions.

    Parameters
    ----------
    process_idx : int
        Index of the current process (for parallel processing). 
    first_model_num : int
        First model number to include (inclusive).
    last_model_num : int
        Last model number to include (inclusive).
    is_initialize : bool
        Whether to initialize the output file.
    is_changepoint_removal : bool
        Whether to remove change points that do not significantly improve the model.
    coefficient_threshold : float
        Threshold for coefficient magnitude to consider a species as linear.
    is_many_maxchangepoint: bool
        Use more maxchangepoints than 50000
    max_fractional_reduction : float
        Maximum fractional reduction in the sum of squared errors required to accept a new change point.
    output_path : str
        Path to the output file. Is modified by the process index
    '''
    # Initializations
    if is_many_maxchangepoints:
        max_changepoints = MANY_MAX_CHANGEPOINTS
    else:
        max_changepoints = ONE_MAX_CHANGEPOINT
    output_path = output_path.replace(".csv", f"_{process_idx}.csv")
    if os.path.isfile(output_path) and (not is_initialize):
        current_df = pd.read_csv(output_path)
        existing_model_names = set(current_df[cn.COL_SYSTEM_ID].unique())
    else:
        existing_model_names = set()
        current_df = pd.DataFrame()
    # Process the max_changepoint values in order, so that the output file is sorted by max_changepoint.
    for item in TimecourseIterator(
            is_curated=False,  # Curation is applied post-processing
            num_point=NUM_POINT,
            first_model_num=first_model_num,
            last_model_num=last_model_num):
        # See if this is a model to skip
        if item.model_name.upper() in EXCLUDED_MODELS:
            print(f"Skipping {item.model_name} (excluded)")
            continue
        # Process the model for each max_changepoint valuea
        for max_changepoint in max_changepoints:
            if item.model_name in existing_model_names:
                model_df = current_df[current_df[cn.COL_SYSTEM_ID] == item.model_name]
                if max_changepoint in model_df[cn.COL_MAX_CHANGEPOINT].values:
                    print(f"Skipping {item.model_name}/{max_changepoint} (already processed)")
                    continue
            pred_df = None
            msg = f"Processing {item.model_name} (max_changepoint={max_changepoint})"
            print(msg)
            pred_df = processModel(item,
                    max_changepoint=max_changepoint,
                    max_fractional_reduction=max_fractional_reduction,
                    coefficient_threshold=coefficient_threshold,
                    is_changepoint_removal=is_changepoint_removal,
                    )
            if pred_df is None:
                print(f"Skipping {item.model_name} {max_changepoint}--no prediction.")
                continue
            if pred_df is not None:
                current_df = pd.concat([current_df, pred_df], ignore_index=True)
                current_df.to_csv(output_path, index=False)

    # Persist results to disk.
    current_df.to_csv(output_path, index=False)
    # Construct the desired final file name
    print(makeFilePath(num_point=NUM_POINT,
        coefficient_threshold = coefficient_threshold,
        max_fractional_reduction=max_fractional_reduction,
        is_changepoint_removal=is_changepoint_removal,
        is_many_maxchangepoints=is_many_maxchangepoints))


#################################################################
# Collect parameters and run the script.
#################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Piecewise predictions for BioModels.")
    parser.add_argument("--process_idx", type=int, default=0)
    parser.add_argument("--first_model_num", type=int, default=0)
    parser.add_argument("--last_model_num", type=int, default=int(1e9))
    parser.add_argument("--initialize", action="store_true",
                        help="Reset output file to empty (reprocess all models).")
    parser.add_argument("--changepoint_removal", action="store_true",
                        help="Remove change points that do not significantly improve the model.")
    parser.add_argument("--many_maxchangepoints", # type: ignore
                        action="store_true",
                        help="Use one max changepoint (5000)"),
    parser.add_argument("--max_reduction", type=float, default=0.01,  # type: ignore
                        help="Maximum amount of reduction in the accuracy."),
    parser.add_argument("--coefficient_threshold", type=float, default=COEFFICIENT_THRESHOLD,
                        help="Threshold for coefficient magnitude to consider a species as linear.")
    args = parser.parse_args()
    main(
            process_idx=args.process_idx,
            first_model_num=args.first_model_num,
            last_model_num=args.last_model_num,
            is_initialize=args.initialize,
            is_many_maxchangepoints=args.many_maxchangepoints,
            coefficient_threshold=args.coefficient_threshold,
            max_fractional_reduction=args.max_reduction,
            is_changepoint_removal=args.changepoint_removal,
        )