"""Perturbation study: how do perturbed initial conditions affect SystemDiscovery R²?

Runs only on models with a explicitly specified end time.
For each qualifying model, SystemDiscovery.analyzePerturbations is called with
perturbation_value_fractions of -50%, -20%, -10%, -5%, 0%, +5%, +10%, +20%, +50%.
R² is computed using the derivative method.

Output CSV: data/perturbation_study.csv
Columns: model_name, threshold, and for each perturbation level
"""

import argparse
import os
import sys

import pandas as pd  # type: ignore

import src.constants as cn
from src.perturbation_analyzer import PerturbationAnalyzer
from src.timecourse_iterator import TimecourseIterator

DEFAULT_THRESHOLD = 0.001
POLY_DEGREE = 1
SPECIES_FRACTION = 1.0
PERTURBATIONS: list[float] = [-0.50, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.50]

MIN_R2 = 0.8
COL_DEG1_MODEL_MAX = "deg1_max"

EXCLUDES = [
            "BIOMD0000000338", # Got many errors
            "BIOMD0000000339", # Got many errors
            "BIOMD0000000378", # Got many errors
            "BIOMD0000000531", # Got many errors
            "BIOMD0000000532", # Got many errors
            "BIOMD0000000555", # Got many errors
            "BIOMD0000000559", # Got many errors
            "BIOMD0000000561", # Got many errors
            "BIOMD0000000570", # Got many errors
            "BIOMD0000000572", # Got many errors
            "BIOMD0000000627", # Got many errors
            "BIOMD0000000673", # Got many errors
            "BIOMD0000000711", # Got many errors
            "BIOMD0000000718", # Got many errors
            "BIOMD0000000721", # Got many errors
            "BIOMD0000000734", # Got many errors
            "BIOMD0000000763", # Got many errors
            "BIOMD0000000787", # Got many errors
            "BIOMD0000000809", # Got many errors
            "BIOMD0000000810", # Got many errors
            "BIOMD0000000834", # Got many errors
            "BIOMD0000000856", # Got many errors
            "BIOMD0000000864", # Got many errors
            "BIOMD0000000876", # Got many errors
            "BIOMD0000000879", # Got many errors
            "BIOMD0000000923", # Got many errors
            "BIOMD0000000943", # Got many errors
            "BIOMD0000000961", # Got many errors
            "BIOMD0000000972", # Got many errors
            "BIOMD0000000989", # Got many errors
            "BIOMD0000000990", # Got many errors
            "BIOMD0000001019", # Got many errors
            "BIOMD0000001020", # Got many errors
            "BIOMD0000001027", # Got many errors
            ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Perturbation study: how do perturbed initial conditions affect SystemDiscovery R²?",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Feature selection threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--is-analyze-model",
        dest="is_analyze_model",
        action="store_true",
        default=True,
        help="Analyze model-level results (default: True)",
    )
    parser.add_argument(
        "--no-is-analyze-model",
        dest="is_analyze_model",
        action="store_false",
        help="Disable model-level analysis",
    )
    parser.add_argument(
        "--is-analyze-species",
        dest="is_analyze_species",
        action="store_true",
        default=True,
        help="Analyze species-level results (default: True)",
    )
    parser.add_argument(
        "--no-is-analyze-species",
        dest="is_analyze_species",
        action="store_false",
        help="Disable species-level analysis",
    )
    return parser


def main(is_initialize: bool = False, is_analyze_model: bool = True,
         is_analyze_species: bool = True, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    if is_analyze_model and is_analyze_species:
        print("Analyzing both model-level and species-level results...")
        output_path = os.path.join(cn.DATA_DIR, f"perturbation_study-model_species{threshold}.csv")
    elif is_analyze_model and not is_analyze_species:
        print("Analyzing model-level results only...")
        output_path = os.path.join(cn.DATA_DIR, f"perturbation_study-model{threshold}.csv")
    elif not is_analyze_model and is_analyze_species:
        print("Analyzing species-level results only...")
        output_path = os.path.join(cn.DATA_DIR, f"perturbation_study-species{threshold}.csv")
    else:
        raise ValueError("At least one of is_analyze_model or is_analyze_species must be True.")
    if not is_initialize and os.path.isfile(output_path):
        print(f"Loading existing results from {output_path}...")
        initial_df = pd.read_csv(output_path)
    else:
        initial_df = pd.DataFrame()

    already_done: set[str] = set()
    if len(initial_df) > 0:
        already_done = set(initial_df[cn.COL_SYSTEM_ID].values)

    for item in TimecourseIterator():
        if item.model_name in already_done:
            print(f"Skipping {item.model_name} (already processed)", flush=True)
            continue
        if item.model_name in EXCLUDES:
            print(f"Skipping {item.model_name} (excluded)", flush=True)
            continue
        print(f"Processing {item.model_name}...", flush=True)
        try:
            analyze_df = PerturbationAnalyzer(
                model=item.timecourse.model,
                training_df=item.timecourse.timecourse_df,
                threshold=threshold,
                perturbations=PERTURBATIONS,
                perturbation_species_fraction=SPECIES_FRACTION,
                poly_degree=POLY_DEGREE,
                is_analyze_model=is_analyze_model,
                is_analyze_species=is_analyze_species,
            ).result.df
        except Exception as exc:
            print(f"  [error] {item.model_name}: {exc}", file=sys.stderr)
            continue

        analyze_df[cn.COL_COEFFICIENT_THRESHOLD] = threshold
        current_df = pd.read_csv(output_path) if os.path.isfile(output_path) else pd.DataFrame()
        # Ensure analyze_df is a DataFrame before concatenating/writing.
        if isinstance(analyze_df, pd.Series):
            analyze_df = pd.DataFrame([analyze_df.to_dict()]).df
        full_df = pd.concat([current_df, analyze_df], ignore_index=True) if len(current_df) > 0 else analyze_df
        full_df.to_csv(output_path, index=False)

    if os.path.isfile(output_path):
        full_df = pd.read_csv(output_path)
    else:
        full_df = initial_df
    print(f"\nDone. {len(full_df)} rows in {output_path}")
    return full_df


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        is_analyze_model=args.is_analyze_model,
        is_analyze_species=args.is_analyze_species,
        threshold=args.threshold,
    )
