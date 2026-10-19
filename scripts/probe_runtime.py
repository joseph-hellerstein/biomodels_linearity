'''Probe how long _fitSegments and score() take on real BioModels data.'''

import time
from src.piecewise_system_discovery import PiecewiseSystemDiscovery  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

it = TimecourseIterator(num_model=1)
item = next(iter(it))
df = item.timecourse.timecourse_df
print(f"Model: {item.model_name}")
print(f"Species: {list(df.columns)}")
print(f"Timepoints: {len(df)}")


def probe(method, max_cp):
    t0 = time.perf_counter()
    psd = PiecewiseSystemDiscovery(
        df, max_changepoint=max_cp, min_segment_length=100, max_fractional_reduction=-1.0)
    cps = getattr(psd, f"_makeChangepoints{method}")()
    dt = time.perf_counter() - t0
    print(f"  _makeChangepoints{method}(max_cp={max_cp}): {dt:.2f}s  -> {len(cps)} changepoints")

    # Fit segments and assign to instance.
    t1 = time.perf_counter()
    psd.changepoints = cps
    _models, _bounds, _lens = psd._fitSegments(cps)
    psd._subsequence_models = _models
    psd._subsequence_boundaries = _bounds
    psd._subsequence_lengths = _lens
    fs = time.perf_counter() - t1
    print(f"  _fitSegments({len(cps)} -> {len(_models)} segments): {fs:.2f}s")

    # Mark fitted and score.
    saved_f = psd._is_fitted
    t2 = time.perf_counter()
    psd._is_fitted = True
    score = float(psd.score(test_df=df))
    st = time.perf_counter() - t2
    print(f"  psd.score(): {st:.2f}s -> p10={score:.3f}")
    psd._is_fitted = saved_f

    return cps


# Probe Efficient.
cps_e = probe("Efficient", max_cp=5)
print()
# Probe Divideandconquor.
cps_d = probe("Divideandconquor", max_cp=5)

print(f"\nCPs: Efficient={cps_e}, DnC={cps_d}")

