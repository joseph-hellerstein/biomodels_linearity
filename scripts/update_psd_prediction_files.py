#!/usr/bin/env python3
'''Update columns to every ``data/piecewise_predictions*.csv`` file. Create bestmodel files.

For each row, AUC is computed from that row's own changepoints via :class:`NLCurve`:

* If the changepoints list is empty (``"[]"``), AUC is set to **1.0** -- the
  value of a single-segment curve over the full timecourse.
* Otherwise boundaries are built from the parsed changepoints as
  ``[0, *sorted(changepoints), NUM_POINT - 1]``, an :class:`NLCurve` is
  constructed, and :meth:`~src.nl_curve.NLCurve.calculateAUC` returns the value.

Each CSV is backed up to ``<path>.bak`` before being overwritten in place so a
re-run can be reverted easily if something looks wrong.

Usage::

    python scripts/update_psd_prediction_files.py [--dry-run] [--verbose]
'''
import os
import numpy as np
import ast



import src.constants as cn  # noqa: E402
import src.util as util  # type: ignore
from src.cumulative_segment_curve import CumulativeSegmentCurve # type: ignore
from src.piecewise_system_discovery import PiecewiseSystemDiscovery  # type: ignore
from src.psd_prediction_file_iterator import PSDPredictionFileIterator, PSDPredictionItem  # type: ignore
from src.psd_analyzer import PSDAnalyzer  # type: ignore

import argparse
import sys
from pathlib import Path
import pandas as pd  # type: ignore
import shutil
import pickle
from collections import namedtuple


UpdateTask = namedtuple("UpdateTask", ["column", "func"])

def _backup(item: PSDPredictionItem, is_pkl: bool = False) -> Path:
    """Back up the original CSV and PKL files before overwriting them.

    Args:
        item (PSDPredictionItem): The item containing the file paths to back up.
        is_pkl (bool, optional): Whether to back up the PKL file as well. Defaults to False.
    Returns:
        Path: The path to the backup file.
    """
    path = Path(item.pkl_path) if is_pkl else Path(item.csv_path)
    filepath = Path(path)
    backup_path = Path(filepath.with_suffix(filepath.suffix + ".bak"))
    if os.path.exists(filepath) and not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
    #
    if is_pkl:
        with open(filepath, "wb") as f:
            pickle.dump(item.df, f)
    else:
        item.df.to_csv(filepath, index=False)
    return backup_path


#############################
# Column editing functions
#############################

############################
def _addAUCColumn(item: PSDPredictionItem) -> int:
    """Append an ``auc`` column to *df* in place

    Args:
        item (PSDPredictionItem): The item containing the DataFrame to update
    Returns:
        int: Number of rows updated

    Note that this function may fail if the changepoints list is empty or invalid, in which case the caller should handle the exception.
    """
    item.df[cn.COL_AUC] = item.df.apply(lambda row: CumulativeSegmentCurve(row[cn.COL_BOUNDARIES], name=row[cn.COL_SYSTEM_ID]).auc, axis=1)
    return len(item.df)

############################
def _addBoundariesColumn(item: PSDPredictionItem) -> int:
    """Append a ``boundaries`` column to *df* in place

    Args:
        item (PSDPredictionItem): The item containing the DataFrame to update
    Returns:
        int: Number of rows updated

    Note that this function may fail if the changepoints list is empty or invalid, in which case the caller should handle the exception.
    """
    df = item.df
    try:
        # Convert if needed
        df[cn.COL_CHANGEPOINTS] = df[cn.COL_CHANGEPOINTS].apply(ast.literal_eval)
    except Exception as e:
        pass
    boundaries_list = []
    for _, row in df.iterrows():
        changepoints = PiecewiseSystemDiscovery.parseChangepointsCell(row[cn.COL_CHANGEPOINTS], model_name=row[cn.COL_SYSTEM_ID])
        boundaries = [0] + sorted(changepoints) + [item.num_point - 1]  # type: ignore
        if not isinstance(boundaries[0], (int, float)):
            raise ValueError(f"Expected a list of integers or floats for changepoints.")
        boundaries_list.append(boundaries)
    df[cn.COL_BOUNDARIES] = boundaries_list
    return len(df)

# Order matters in the list of tasks
UPDATE_TASKS = [
    UpdateTask(column=cn.COL_BOUNDARIES, func=_addBoundariesColumn),
    UpdateTask(column=cn.COL_AUC, func=_addAUCColumn),
]

############################
# Main script
############################

def main(is_report: bool = False, is_update_bestmodels: bool = True) -> int:
    """Update the piecewise prediction files and optionally build the bestmodel CSVs.

    Args:
        is_report (bool, optional): Output progress. Defaults to False.
        is_update_bestmodels (bool, optional): Construct the bestmodel files. Defaults to True.

    Returns:
        int: 0 if all updates and bestmodel generations succeeded, 1 otherwise.
    """

    ##
    def message(msg: str) -> None:
        if is_report:
            print(msg)

    psd_iterator = PSDPredictionFileIterator(is_pkl=False, is_report=True)

    fail_count = 0
    success_count = 0

    for item in psd_iterator:
        # Save the file
        if item.df.empty:
            message("  SKIP: empty DataFrame after reading.")
            continue
        else:
            success_count += 1
        # Process the updates
        n_rows = 0
        for task in UPDATE_TASKS:
            try:
                n_rows = task.func(item)
                message(f"  SUCCESS: {task.column} computation succeeded for {item.filepath} ({n_rows} rows).")
            except Exception as exc:
                message(f"  FAIL: {task.column} computation failed for {item.filepath}: {exc}; keeping backup.")
                fail_count += 1
                continue
        # Back up the original file before mutating it.
        backup_path = _backup(item, is_pkl=False)
        _ = _backup(item, is_pkl=True)
        if is_report:
            message(f"    backed up -> {backup_path.name}")

    if is_report:
        print(f"Successfully processed {success_count}. {fail_count} FAIL.") 

    # Creating best model CSVs for each minaccuracy/accuracystat combination
    if is_update_bestmodels:
        print("Creating psd_analyzer_bestmodel CSVs for each minaccuracy/accuracystat combination.")
        min_accuracy_list = [0.8, 0.9]
        accuracy_stat_list = [cn.COL_MEAN, cn.COL_P50]  # must be columns of the prediction files
        for min_accuracy in min_accuracy_list:
            for accuracy_stat in accuracy_stat_list:
                if is_report:
                    print(f"Processing min_accuracy={min_accuracy}, accuracy_stat={accuracy_stat}...")
                try:
                    _ = PSDAnalyzer.makeBestPSDModelDF(min_accuracy=min_accuracy, accuracy_stat=accuracy_stat,
                            is_force_update=True)
                except Exception as exc:
                    message(f"  FAIL: PSDAnalyzer.makeBestPSDModelDF failed for min_accuracy={min_accuracy}, accuracy_stat={accuracy_stat}: {exc}")
                    fail_count += 1
                    continue

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update columns to every piecewise_predictions CSV.",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-file and progress output.")
    parser.add_argument("--skip-bestmodels", action="store_true",
                        help="Skip creating the psd_analyzer_bestmodel CSVs.")
    args = parser.parse_args()
    sys.exit(main(is_report=args.verbose, is_update_bestmodels=not args.skip_bestmodels))
