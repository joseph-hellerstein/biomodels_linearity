'''Assess PiecewiseSystemDiscovery for consistency of metrics.'''

import argparse
import os
import sys

import pandas as pd  # type: ignore

import src.constants as cn
from src.piecewise_system_discovery import PiecewiseSystemDiscovery
from src.timecourse_iterator import TimecourseIterator

OUTPUT_PATH = os.path.join(cn.DATA_DIR, "analyze_piecewise_system_discovery.csv")

def main(max_fractional_reduction: int, output_path: str=OUTPUT_PATH):

    if not is_initialize and os.path.isfile(output_path):
        print(f"Loading existing results from {output_path}...")
        initial_df = pd.read_csv(output_path)
    else:
        initial_df = pd.DataFrame()

    already_done: set[str] = set()
    if len(initial_df) > 0:
        already_done = set(initial_df[COL_MODEL_NAME].values)

    rows: list[dict] = []
    for item in TimecourseIterator():
        if item.model_name in already_done:
            print(f"Skipping {item.model_name} (already processed)", flush=True)
            continue
        print(f"Processing {item.model_name}...", flush=True)
        try:
            psd = PiecewiseSystemDiscovery(
                item.timecourse.timecourse_df,
                max_changepoint=num_change_point,
                num_trail=num_trail,
            ).fit()
            info = psd.getScoreSummary()
        except Exception as exc:
            print(f"  [error] {item.model_name}: {exc}", file=sys.stderr)
            continue
        rows.append({
            COL_MODEL_NAME: item.model_name,
            COL_SCORE_MIN: info.min,
            COL_SCORE_MEDIAN: info.median,
            COL_SCORE_MAX: info.max,
            COL_NUM_NONZERO_TERM: info.num_nonzero_term,
        })
        full_df = pd.concat([initial_df, pd.DataFrame(rows)], ignore_index=True)
        full_df.to_csv(output_path, index=False)

    full_df = (
        pd.concat([initial_df, pd.DataFrame(rows)], ignore_index=True)
        if rows
        else initial_df
    )
    print(f"\nDone. {len(full_df)} rows written to {output_path}")
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_change_point", type=int, default=2,
        help="Number of change points for PiecewiseSystemDiscovery (default: 2)",
    )
    parser.add_argument(
        "--num_trail", type=int, default=1,
        help="Number of random change points to try (default: 1)",
    )
    parser.add_argument(
        "--initialize", action="store_true",
        help="Ignore existing results and start fresh",
    )
    args = parser.parse_args()
    main(num_change_point=args.num_change_point, is_initialize=args.initialize, num_trail=args.num_trail)
