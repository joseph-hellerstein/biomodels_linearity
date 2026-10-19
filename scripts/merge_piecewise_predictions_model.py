"""Merge piecewise prediction CSVs and keep only model-aggregated rows.

Finds all ``data/piecewise_predictions__numpoint*.csv`` files, concatenates
them (single header), adds a source-file column, and filters to rows where
``aggregation_type == 'model'``. Writes the result as a single CSV.
"""

import argparse
from pathlib import Path
import src.constants as cn

import pandas as pd  # type: ignore


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "piecewise_predictions_model.csv"
PATTERN = "piecewise_predictions__numpoint*.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge piecewise prediction CSVs, keeping only model rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    csv_files = sorted(DATA_DIR.glob(PATTERN))
    if not csv_files:
        raise SystemExit(f"No files matching '{PATTERN}' found in {DATA_DIR}")

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        df[cn.COL_CSV_FILE] = path.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    total_before_filter = len(merged)

    filtered = merged[merged["aggregation_type"] == "model"].reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)

    print(f"Input files: {len(csv_files)}")
    print(f"Merged rows (all aggregation_types): {total_before_filter}")
    print(f"Rows after filtering aggregation_type='model': {len(filtered)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
