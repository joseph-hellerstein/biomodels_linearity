'''Curates models that are not suitable for piecewise linear analysis.'''

"""
Find models that are not suitable for piecewise linear analysis.
1. The model contains events, which is not predictable.
2. The model has extreme differences in numerical values, which can cause numerical instability in the piecewise linear analysis.
3. The model has conserved moieties.
"""

import src.constants as cn  # type: ignore
from src.biomodels_iterator import BiomodelsIterator  # type: ignore
from src.model import Model  # type: ignore
from src.timecourse import Timecourse  # type: ignore
from src.simulator import Simulator  # type: ignore

import argparse
import pandas as pd  # type: ignore
import os
from typing import List, Optional

EXCLUDED_MODELS: List[str] = [
    "BIOMD0000000055",
    "BIOMD0000000148",
    "BIOMD0000000205",  # Long processing
    "BIOMD0000000235",
    "BIOMD0000000255",
    "BIOMD0000000268",
    "BIOMD0000000378",
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
# Augment excluded files
if os.path.isfile(os.path.join(cn.DATA_DIR, "badmodels.txt")):
    with open(os.path.join(cn.DATA_DIR, "badmodels.txt"), "r") as f:
        for line in f:
            model_name = line.strip()
            if model_name and not model_name.startswith("#"):
                EXCLUDED_MODELS.append(model_name)

def main(
        first_model_num: int = 0,
        last_model_num: int = int(1e9),
        excluded_models: List[str] = EXCLUDED_MODELS,
        output_path: str = cn.CURATION_PATH,
) -> None:
    result_dct: dict = {cn.COL_SYSTEM_ID: [], cn.COL_REASON: []}
    for item in BiomodelsIterator(
            excluded_models=excluded_models,
            first_model_num=first_model_num,
            last_model_num=last_model_num):
        model_name = item.model_name
        if not item.sbml_paths:
            print(f"Skipping {model_name} (no SBML files)")
            continue
        if item.endtime_source != "sedml":
            print(f"Skipping {model_name} (no SED-ML endtime)")
            continue
        # Create the roadrunner instance
        model = Model.makeBiomodel(item.model_name)
        simulator = Simulator(model, end_time=item.end_time, num_point=cn.NUM_POINT)
        rr, _ = simulator.makeRoadRunner()
        # Check reasons for exclusion
        reason: Optional[str] = None
        if rr.getNumEvents() > 0:
            reason = "Events"
        elif rr.getNumConservedMoieties() > 0:
            reason = "Conserved moieties"
        elif rr.getNumFloatingSpecies() > 0:
            df = simulator.simulate().timecourse_df
            if df.min().min() < 0:
                reason = "Negative species values"
            else:
                max_ser = df.max()
                max_ser = max_ser[max_ser > 0]
                min_val = max_ser.min() if not max_ser.empty else 0
                max_val = max_ser.max() if not max_ser.empty else 0
                if max_val / min_val > 1e6:
                    reason = "Extreme differences in species values"
        if reason is not None:
            result_dct[cn.COL_SYSTEM_ID].append(model_name)
            result_dct[cn.COL_REASON].append(reason)
        # Save the results
        result_df = pd.DataFrame(result_dct)
        result_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()