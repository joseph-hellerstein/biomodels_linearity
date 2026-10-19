'''Script that serializes Timecourse (timecourse_df + Jacobians) for all BioModels.'''

"""
Usage:
    python scripts/make_biomodels_timecourse.py --first_model_num 0 --last_model_num 100 

"""

import src.constants as cn  # type: ignore
from src.biomodels_iterator import BiomodelsIterator  # type: ignore
from src.model import Model  # type: ignore
from src.timecourse import Timecourse  # type: ignore

import argparse
import os
from typing import List

NUM_POINT = 100000
EXCLUDED_MODELS: List[str] = [
    "BIOMD0000000055",
    "BIOMD0000000148",
    "BIOMD0000000205",  # Long processing
    "BIOMD0000000235",
    "BIOMD0000000255",
    "BIOMD0000000268",
    "BIOMD0000000469",
    "BIOMD0000000470",
    "BIOMD0000000471",
    "BIOMD0000000472",
    "BIOMD0000000473",
    "BIOMD0000000490",
    "BIOMD0000000566", # seg fault
    "BIOMD0000000567", # seg fault
    "BIOMD0000000574",  # Long processing
    "BIOMD0000000606",
    "BIOMD0000000625",
    "BIOMD0000000693",
    "BIOMD0000000794",
    "BIOMD0000000806",
    "BIOMD0000000810",
    "BIOMD0000001020",
    "BIOMD0000001077",  # bad endtime
    "BIOMD0000001078",  # bad endtime
    "BIOMD0000001079",  # bad endtime
    "BIOMD0000001080",  # bad endtime
]
if os.path.isfile(os.path.join(cn.DATA_DIR, "badmodels.txt")):
    with open(os.path.join(cn.DATA_DIR, "badmodels.txt"), "r") as f:
        for line in f:
            model_name = line.strip()
            if model_name and not model_name.startswith("#"):
                EXCLUDED_MODELS.append(model_name)

def main(
        is_report: bool = True,
        first_model_num: int = 0,
        last_model_num: int = int(1e9),
        excluded_models: List[str] = EXCLUDED_MODELS,
        is_initialize: bool = False, # Ignore existing serialized Timecourse when initializing (for testing).
        num_point: int = NUM_POINT,
) -> None:
    '''Serialize timecourses for all BioModels.

    Parameters
    ----------
    is_report : bool
        Whether to print progress.
    first_model_num : int
        First model number to include (inclusive).
    last_model_num : int
        Last model number to include (inclusive).
    excluded_models : List[str]
        Model names to skip.
    is_initialize: bool
        If True, ignore existing serialized Timecourse when initializing (for testing).
    '''
    os.makedirs(cn.TIMECOURSE_SERIALIZATION_DIR, exist_ok=True)
    for item in BiomodelsIterator(
            excluded_models=excluded_models,
            is_report=is_report,
            first_model_num=first_model_num,
            last_model_num=last_model_num):
        model_name = item.model_name
        if not item.sbml_paths:
            print(f"Skipping {model_name} (no SBML files)")
            continue
        pkl_path = os.path.join(cn.TIMECOURSE_SERIALIZATION_DIR,
                f"{model_name}_timecourse.pkl")
        try:
            model = Model.makeBiomodel(model_name)
        except ValueError as e:
            print(f"Skipping {model_name} (error loading model: {e})")
            continue
        if os.path.isfile(pkl_path) and (not is_initialize):
            timecourse = Timecourse.deserialize(path=pkl_path)
            if not timecourse.timecourse_df.empty:
                print(f"Skipping {model_name} (already serialized)")
                continue
        try:
            timecourse = Timecourse(model=model, end_time=item.end_time,
                    num_point=num_point)
            _ = timecourse.timecourse_df  # Force calculations
            path = timecourse.serialize()
            serialized_timecourse = Timecourse.deserialize(path=path)
            if not serialized_timecourse == timecourse:
                raise ValueError(f"Deserialized timecourse does not match original for {model_name}.")  
        except Exception as e:
            print(f"Error processing {model_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Serialize Timecourses for BioModels.")
    parser.add_argument("--first_model_num", type=int, default=0)
    parser.add_argument("--last_model_num", type=int, default=int(1e9))
    parser.add_argument("--num_point", type=int, default=NUM_POINT)
    parser.add_argument("--initialize", action="store_true", help="Ignore existing serialized Timecourse when initializing (for testing).")
    args = parser.parse_args()
    main(first_model_num=args.first_model_num, last_model_num=args.last_model_num,
            num_point=args.num_point, is_initialize=args.initialize)
