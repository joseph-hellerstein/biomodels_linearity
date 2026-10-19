'''Benchmark _makeChangepointsEfficient vs _makeChangepointsDivideandconquor on real BioModels.'''


import src.constants as cn  # type: ignore
from src.piecewise_system_discovery import PiecewiseSystemDiscovery  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

import argparse
import os
import sys
import time


def run_model(model_name, df, max_changepoint, min_segment_length, max_fractional_reduction):
    """Run both methods on one model's DataFrame and return result dicts."""
    base_kwargs = dict(
        max_changepoint=max_changepoint, min_segment_length=min_segment_length,
        max_fractional_reduction=max_fractional_reduction,
    )
    results = []

    for method_name in ("Efficient", "Divideandconquor"):
        rec = dict(
            method=f"PiecewiseSystemDiscovery.{method_name}", model_name=model_name,
            num_changepoints=0, score_p10=float("nan"), detect_time_sec=0.0, fit_time_sec=0.0,
            failed=False, error="",
        )
        try:
            psd = PiecewiseSystemDiscovery(df, **base_kwargs)
            t0 = time.perf_counter()
            cps = getattr(psd, f"_makeChangepoints{method_name}")()
            rec["detect_time_sec"] = time.perf_counter() - t0

            # Fit piecewise model with detected changepoints and measure score.
            t1 = time.perf_counter()
            _models, _bounds, _lens = psd._fitSegments(cps)
            psd._subsequence_models = _models
            psd._subsequence_boundaries = _bounds
            psd._subsequence_lengths = _lens
            saved_f = psd._is_fitted
            psd._is_fitted = True
            rec["score_p10"] = float(psd.score(test_df=df))
            psd._is_fitted = saved_f
            rec["fit_time_sec"] = time.perf_counter() - t1
            rec["num_changepoints"] = len(cps)
        except Exception as e:  # pragma: no cover -- defensive for real BioModels
            rec["failed"], rec["error"] = True, f"{type(e).__name__}: {e}"
        results.append(rec)
    return results


def _agg_mean(records, col):
    vals = [r[col] for r in records if isinstance(r.get(col), (int, float)) and not (isinstance(r[col], float) and r[col] != r[col])]  # noqa -- NaN filter
    return sum(vals) / len(vals) if vals else float("nan")


def _agg_median(records, col):
    import numpy as np
    vals = sorted(r[col] for r in records if isinstance(r.get(col), (int, float)) and not (isinstance(r[col], float) and r[col] != r[col]))  # noqa -- NaN filter
    return float(np.median(vals)) if vals else float("nan")

def _print_summary(all_rows):
    eff = [r for r in all_rows if "Efficient" in r["method"] and not r["failed"]]
    dnc = [r for r in all_rows if "Divideandconquor" in r["method"] and not r["failed"]]

    print()
    print("=" * 90)
    print(f"{'Metric':<32} {'Efficient':>14} {'Divideandconquor':>16}")
    print("-" * 90)
    eff_failed = sum(1 for r in all_rows if "Efficient" in r["method"] and r["failed"])
    dnc_failed = sum(1 for r in all_rows if "Divideandconquor" in r["method"] and r["failed"])
    total_det_e = sum(r["detect_time_sec"] for r in eff)
    total_det_d = sum(r["detect_time_sec"] for r in dnc)
    for label, e_val, d_val in [
        ("Models completed",           len(eff),                len(dnc)),
        ("Models failed",              eff_failed,              dnc_failed),
        ("Mean num changepoints",      _agg_mean(eff, "num_changepoints"),
                                      _agg_mean(dnc, "num_changepoints")),
        ("Median num changepoints",    _agg_median(eff, "num_changepoints"),
                                      _agg_median(dnc, "num_changepoints")),
        ("Mean score (p10)",           _agg_mean(eff, "score_p10"),
                                      _agg_mean(dnc, "score_p10")),
        ("Median score (p10)",         _agg_median(eff, "score_p10"),
                                      _agg_median(dnc, "score_p10")),
        ("Mean detect time (s)",       _agg_mean(eff, "detect_time_sec"),
                                      _agg_mean(dnc, "detect_time_sec")),
        ("Total detect time (s)",      total_det_e,             total_det_d),
        ("Mean fit time (s)",          _agg_mean(eff, "fit_time_sec"),
                                      _agg_mean(dnc, "fit_time_sec")),
    ]:
        print(f"{label:<32} {e_val:>14.4f} {d_val:>16.4f}")
    print("=" * 90)

    # Per-model table (first 15).
    eff_map = {r["model_name"]: r for r in eff}
    dnc_map = {r["model_name"]: r for r in dnc}
    print()
    print(f"{'Model':<28} {'Eff CPs':>9} {'Eff p10':>9} {'DnC CPs':>9} {'DnC p10':>9}")
    print("-" * 70)
    for mn in sorted(eff_map.keys())[:15]:
        e, d = eff_map.get(mn), dnc_map.get(mn)
        ep = f"{e['score_p10']:.3f}" if e else "  --"
        dp = f"{d['score_p10']:.3f}" if d else "  --"
        ec = str(e["num_changepoints"]) if e else "--"
        dc = str(d["num_changepoints"]) if d else "--"
        print(f"{mn:<28} {ec:>9} {ep:>9} {dc:>9} {dp:>9}")


def main(num_model, max_cp, min_seg_len, max_frac_red, output_path, initialize=False,
        first_model_num=None, last_model_num=None):
    # Use first_model_num / last_model_num to slice BioModel ranges (BIOMD0000000001 is num=1).
    if first_model_num is not None:
        start = first_model_num
    else:
        start = 1
    if last_model_num is None:
        end = start + num_model - 1
    else:
        print("Using explicit last_model_num, instead of calculating from num_model")
        end = last_model_num
    iterator = TimecourseIterator(first_model_num=start, last_model_num=end)
    all_rows = []
    if initialize:
        if output_path and os.path.exists(output_path):
            os.remove(output_path)

    for item in iterator:
        model_name = item.model_name
        try:
            rows = run_model(model_name, item.timecourse.timecourse_df, max_cp, min_seg_len, max_frac_red)
            all_rows.extend(rows)
            print(f"{model_name}: Efficient p10={rows[0]['score_p10']:.3f}, DnC p10={rows[1]['score_p10']:.3f}", file=sys.stderr)
        except Exception as e:  # pragma: no cover
            print(f"ERROR processing {model_name}: {e}", file=sys.stderr)
            continue

    path = output_path or os.path.join(cn.DATA_DIR, "benchmark_changepoints.csv")
    write_header = not os.path.exists(path)
    header = ["method", "model_name", "num_changepoints", "score_p10",
              "detect_time_sec", "fit_time_sec", "failed", "error"]
    with open(path, "a" if not write_header else "w") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        for r in all_rows:
            vals = []
            for h in header:
                v = r.get(h, "")
                if isinstance(v, str):
                    vals.append(f'"{v}"')  # escape strings with commas / quotes
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"Wrote {len(all_rows)} rows to {path}", file=sys.stderr)
    _print_summary(all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark changepoint detection methods on BioModels.")
    parser.add_argument("--num_models", type=int, default=100)
    parser.add_argument("--first_model_num", type=int, default=100)
    parser.add_argument("--last_model_num", type=int, default=100)
    parser.add_argument("--initialize", action="store_true",
                        help="Reset output file to empty (reprocess all models).")
    parser.add_argument("--max_changepoint", type=int, default=10)
    parser.add_argument("--min_segment_length", type=int, default=50)
    parser.add_argument("--max_fractional_reduction", type=float, default=0.01)
    parser.add_argument("--output_path", type=str, default=None, help="Optional CSV output path.")
    args = parser.parse_args()
    main(args.num_models, args.max_changepoint, args.min_segment_length,
            args.max_fractional_reduction, args.output_path, args.initialize,
            first_model_num=args.first_model_num, last_model_num=args.last_model_num)

