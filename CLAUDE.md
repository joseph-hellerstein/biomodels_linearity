# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Note: `AGENTS.md` in this repo covers environment setup, test/lint commands, and a first-pass architecture
summary, and is also loaded automatically. This file focuses on the deeper cross-file architecture that
doesn't fit there. If the two ever disagree on a fact, trust what you verify in the code over either file.

## Project purpose

Assess how well an SBML model's kinetics are approximated by linear (or piecewise-linear) ODE models, based
on (a) consistency of the Jacobian and (b) accuracy reproducing a non-linear simulation — and identify which
reactions ("linearity bottlenecks") limit the viability of a linear approximation.

## Environment

Activate with `source activate.sh` before running anything. It sources `../BaseStack/bin/setup_run.sh`, puts
`src/` and `scripts/` on `PYTHONPATH`, and activates the local venv at `bl/` (not `mla/`, despite what
`AGENTS.md` currently says — `bl/` is the real one; the venv directory name appears to have changed since
that doc was written).

```bash
source activate.sh && python3 -m pytest tests/ -v
source activate.sh && python3 -m pytest tests/test_system_discovery.py -v
source activate.sh && python3 -m pytest tests/test_system_discovery.py::TestGetScoreAggregatedBySpecies::test_returns_expected_keys -v
```

There's no `pyproject.toml`/`setup.py`/`pytest.ini` — the project is pure `PYTHONPATH`-based, not pip-installed.

## Architecture: how the pieces compose

The codebase is a layered pipeline. Reading any one file in isolation tends to mislead; the relationships
between layers are where the real behavior lives.

**Layer 1 — raw simulation.** `Model` (`src/model.py`) holds static SBML properties (species, initial
values, reaction count); RoadRunner is used transiently to simulate and never stored. `Timecourse`
(`src/timecourse.py`) wraps a `Model`, lazily simulates it into `timecourse_df` + a Jacobian collection, and
pickles itself. `TimecourseIterator` (`src/timecourse_iterator.py`) iterates pre-simulated `Timecourse`
pickles out of a zip (`data/serialize/timecourse/timecourse_<N>.zip`, where `N` is the point count: 1000,
10000, or 100000). **Always prefer `TimecourseIterator.getTimecourse(model_id, num_point=...)` for a single
known model** — it opens one zip entry directly. Iterating the whole `TimecourseIterator()` from scratch to
find one model walks every entry before it in the zip in sorted order, unpickling each one.
`BiomodelsIterator` (`src/biomodels_iterator.py`) is the lower-level iterator over raw SBML directories under
`BIOMODELS_DIR`, used mainly by the pipeline scripts that build the serialized timecourses in the first
place, not by day-to-day analysis code.

**Layer 2 — ODE discovery.** `SystemDiscovery` (`src/system_discovery.py`) wraps PySINDy to fit a sparse
polynomial ODE system to a timecourse DataFrame. `PiecewiseSystemDiscovery`
(`src/piecewise_system_discovery.py`) is the central, most complex class: it detects changepoints in a
timecourse and fits one `SystemDiscovery` (forced to `poly_degree=1`) per segment. Changepoints can come from
two genuinely different places, and conflating them is an easy mistake:
- **Live detection** (`is_changepoints_from_file=False`, the default): `_makeChangepointsWithElimination()`
  generates `max_changepoint` evenly-spaced candidates, then `eliminateChangepoints()` prunes them by
  estimating an accuracy-rate-per-Frobenius-distance from randomized trial removals
  (`_estimateAccuracyRate` → `_buildTrialSegments`, which merges-and-refits only the segments a trial
  removal touches, reusing already-fitted models for everything else).
- **Precomputed lookup** (`is_changepoints_from_file=True`): `fit()` reads `util.getPSDPredictionDF()`,
  which filters sweep-output CSVs in `data/psd_data/` by `max_fractional_reduction`, and picks the
  minimum-AUC row meeting `min_accuracy`. This is a *different* data source from
  `data/piecewise_predictions_model.csv` (read by `_getChangepointsFromFile()`), which exists but is not
  currently called from `fit()` at all. Check which path a given run is actually taking before assuming
  what `max_changepoint`/`max_fractional_reduction` will do — the live-elimination cost scales with
  `max_changepoint` (each candidate gets a real `SystemDiscovery.fit()` call), so it is realistic to see
  this take tens of seconds for a `max_changepoint` in the thousands on a 100k-point timecourse.
`CumulativeSegmentCurve` (`src/cumulative_segment_curve.py`) builds an AUC-style curve from a piecewise
model's segment boundaries, used to rank candidate segmentations.

**Layer 3 — scoring.** `Score` (`src/score.py`), backed by `StatisticCalculator`
(`src/statistic_calculator.py`), is the single scoring primitive everything else builds on. `Score.add(true_df,
pred_df)` produces one row per species plus one aggregation row (`aggregation_type == "model"`), and **every
row carries all of `cn.COLUMN_STATISTICS`** (`mean`, `min`, `max`, `p05`...`p99`, etc.) as columns — not just
whichever one you asked for. This is why `accuracy_col` (which percentile column, e.g. `"p10"`) and
`accuracy_stat` (which statistic to read off the `"model"` row, e.g. `"mean"`/`"p50"`) are threaded as plain
constructor parameters through `SystemDiscovery`, `PiecewiseSystemDiscovery`, and `Score` — the right pattern
for reading a score is always "score once, then read the column you want off the model row," never a bespoke
per-statistic aggregation function. `PiecewiseSystemDiscovery.score()` follows this: it reads
`self.score_df`, computed once in `fit()` by scoring the whole stitched piecewise prediction against
`training_df`, not by re-deriving anything per segment.

**Layer 4 — downstream analysis.** `PSDAnalyzer` (`src/psd_analyzer.py`) picks the best precomputed PSD
prediction for a model (by min AUC subject to an accuracy floor) and builds a fitted `PiecewiseSystemDiscovery`
from it. `PSDPredictionFileIterator` (`src/psd_prediction_file_iterator.py`) iterates the
`piecewise_predictions__*.csv`/`.pkl` sweep-output files in `data/psd_data/`, whose filenames encode
metadata as `__key_value` pairs (e.g. `piecewise_predictions__numpoint_100000__maxreduction_1e-3.csv`) —
`check_psd_prediction_csvs.py` verifies a file's contents match its own filename-encoded metadata.
`PerturbationAnalyzer` (`src/perturbation_analyzer.py`) and `OscillationDetector`/`RateLawAnalyzer` are
narrower, model-diagnostic tools downstream of the same `Timecourse`/`Score` primitives.

## Data pipeline (`scripts/`)

Roughly in dependency order:
1. `make_biomodels_endtime.py` → simulation end times → `data/biomodels_endtime.csv`.
2. `curate_models.py` → excludes models unsuitable for piecewise-linear analysis (events, extreme numeric
   ranges, conserved moieties) → `data/curation.csv`. `TimecourseIterator(is_curated=True)` filters by this.
3. `make_biomodels_timecourse.py` → simulates all models → serialized `Timecourse` pickles in the zip.
4. `make_piecewise_predictions.py` → sweeps `PiecewiseSystemDiscovery` over models/parameters → per-run
   `data/psd_data/piecewise_predictions__*.csv`/`.pkl` files.
5. `merge_piecewise_predictions_model.py` / `update_psd_prediction_files.py` → consolidate those sweep
   outputs and (re)compute derived columns like AUC → `data/piecewise_predictions_model.csv`.
6. `check_psd_prediction_csvs.py` → integrity check: file contents vs. filename-encoded metadata.
7. `make_biomodels_psd_plots.py`, `plot_biomodels*.py`, `make_paper_plots.py` → visualization from the above.

## BioModels data

SBML sources live one directory above this repo, at `../temp-biomodels/final/<BIOMD...>/<ID>_url.xml`
(`manifest.xml` in the same directory is not a model file). `data/curation.csv` is the current mechanism for
excluding broken/unsuitable models from batch runs (not `data/badmodels.txt`, which no longer exists).

## Tuning `SystemDiscovery`

From the project README, still applicable:
- Needs sufficient data — often in the thousands of points.
- Lower `coefficient_threshold` to get sparser coefficients.
- Set `includebias=True`.
- Don't include boundary species.
