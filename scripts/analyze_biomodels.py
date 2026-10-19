"""
Partition Jacobians for all BioModels SBML files and write results to a CSV.

Walks a directory of BioModel subdirectories, simulates each SBML model,
partitions the simulation timeline into clusters, computes Jacobian CVs per
partition, and writes the results to a NaN-padded CSV.

Usage:
    source activate.sh
    python scripts/analyze_biomodels.py [--directory DIR] [--output FILE]
                                        [--n_cluster N] [--sequential]

Arguments:
    --directory   Directory containing BioModel subdirectories
                  (default: cn.BIOMODELS_DIR)
    --output      Path for the output CSV file. If omitted, the filename is
                  derived from --n_cluster and --sequential.
    --n_cluster   Number of partitions for k-means clustering (default: 5).
    --sequential  Use contiguous (sequential) partitioning instead of k-means.
"""
import src.constants as cn  # noqa: E402
from src.biomodels_cluster import BiomodelsCluster  # type: ignore

import argparse
import os


EXCLUDED_MODELS = [
    "BIOMD0000000014",
    "BIOMD0000000019",
    "BIOMD0000000088",
    "BIOMD0000000205",
    "BIOMD0000000235",
    "BIOMD0000000255",
    "BIOMD0000000293",
    "BIOMD0000000469",  # 2500 reactions
    "BIOMD0000000470",  # 2500 reactions
    "BIOMD0000000471",  # 2100 reactions
    "BIOMD0000000472",  # >2000 reactions
    "BIOMD0000000473",  # >2000reactions
    "BIOMD0000000574",
    "BIOMD0000000678",
]

import src.constants as cn  # noqa: E402

N_CLUSTER = 5
OUTPUT_FILENAME = f"{N_CLUSTER}_linearity_analysis_data.csv"
OUTPUT_PTH = os.path.join(cn.DATA_DIR, OUTPUT_FILENAME)
NO_PATH = "no_path"


def main() -> None:
    """
    Parse CLI arguments and run LinearAnalyzer.partitionBiomodelsJacobians.

    Resolves the output path (deriving a filename from n_cluster and
    partitioning mode when --output is not provided), prints a summary of
    the run configuration, delegates to partitionBiomodelsJacobians, and
    reports the number of successfully processed models.
    """
    parser = argparse.ArgumentParser(
        description="Compute Jacobian CV for all BioModel SBML files."
    )
    parser.add_argument(
        "--directory",
        default=cn.BIOMODELS_DIR,
        help="Directory containing BioModel subdirectories (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=NO_PATH,
        help="Path for the output CSV file (default: %(default)s)",
    )
    parser.add_argument(
        "--n_cluster",
        default=N_CLUSTER,
        type=int,
        help="Number of clusters for k-means clustering (default: %(default)s)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        default=False,
        help="Use sequential (contiguous) partitioning instead of k-means (default: False)",
    )
    args = parser.parse_args()

    if args.output == NO_PATH:
        if args.sequential:
            output_filename = f"{args.n_cluster}s_linearity_analysis_data_sequential.csv"
        else:
            output_filename = f"{args.n_cluster}_linearity_analysis_data.csv"
        output_path = os.path.join(cn.DATA_DIR, output_filename)
    else:
        output_path = args.output

    print("\n**********************************************************")
    print(f"***Processing models in: {args.directory}")
    print(f"***Output path: {output_path}")
    print(f"***Number of clusters: {args.n_cluster}")
    print(f"***Using sequential partitioning: {args.sequential}")
    print("**********************************************************\n")

    result_dct = BiomodelsCluster.clusterAnalysis(
        directory=args.directory,
        output_data_file=output_path,
        excluded_models=EXCLUDED_MODELS,
        n_cluster=args.n_cluster,
        is_sequential_partition=args.sequential,
        is_report=True,
    )

    n_success = len(result_dct)
    print(f"\nDone. Successfully processed {n_success} model(s).")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
