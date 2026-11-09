# Data Provenance

Reference for every file in `data/`: **(a)** how it is used, **(b)** how it is created,
**(c)** its contents. Paths are relative to the repository root.

> Notes on conventions:
> - Filenames containing `0.1`, `0.01`, or `0.001` were produced with a SINDy
>   coefficient threshold of that value (see `data/README.md`).
> - Several files were created by scripts that no longer exist in the tree
>   (marked *writer removed* below); their provenance is reconstructed from git
>   history, `AGENTS.md`, and the readers of the files.
> - No code in this repository creates `.zip` archives; all zips below were made
>   manually (e.g. `zip -r`) from the files they contain.

## Summary table

| File / directory | Created by | Used by |
| --- | --- | --- |
| `biomodels_endtime.csv` | *writer removed* (`make_biomodels_endtime.py`) | `src/biomodels_iterator.py`, `src/timecourse_iterator.py`, `scripts/plot_biomodels.py` |
| `curation.csv` | `scripts/curate_models.py` | `src/timecourse_iterator.py` (`CurationChecker`) |
| `curation_removed_models.csv` | manual | none (record-keeping) |
| `find_oscillating_models.csv` | *writer removed* (`scripts/find_oscillating_models.py`) | `notebooks/oscillations.ipynb`, `notebooks/piecewise_predictions.ipynb` |
| `benchmark_changepoints.csv` | `scripts/benchmark_changepoint_methods.py` | none (analysis artifact) |
| `linear_predictor_scores-0.001.csv` | *writer removed* (`scripts/calculate_linear_prediction_scores.py`) | `scripts/make_paper_plots.py`, notebooks |
| `perturbation_study-model0.001.csv` | `scripts/perturbation_study.py` | `scripts/make_paper_plots.py`, tests |
| `perturbation_study-species0.001.csv` | `scripts/perturbation_study.py` | `scripts/make_paper_plots.py`, tests |
| `piecewise_predictions_model.zip` | CSV by `scripts/merge_piecewise_predictions_model.py`; zip manual | `cn.PIECEWISE_PREDICTIONS_MODEL_PATH`, tests |
| `piecewise_predictions_pkl.zip` | manual archive of `psd_data/` pkls | none (historical) |
| `psd_analyzer_bestmodel.zip` | manual archive of bestmodel CSVs | `src/psd_analyzer.py::PSDAnalyzer.makeBestPSDModelDF` (default `zip_path`) |
| `psd_data/` | `scripts/make_piecewise_predictions.py` (+ `run/make_piecewise_predictions/make`); columns updated by `scripts/update_psd_prediction_files.py` | `src/psd_prediction_file_iterator.py`, `src/psd_analyzer.py`, `scripts/make_paper_plots.py` |
| `psd_prediction_analyzer_bestmodel__*.csv` | `src/psd_analyzer.py::PSDAnalyzer.makeBestPSDModelDF` (via `scripts/update_psd_prediction_files.py`, notebooks) | `notebooks/psd_prediction_analysis.ipynb`, `PSDAnalyzer` CDF plots |
| `rate_law_statistics.csv` | `scripts/rate_law_statistics.py` | none (analysis artifact); tests cover the writer |
| `serialize/timecourse/` | `scripts/make_biomodels_timecourse.py` (+ `scripts/run_parallel_timecourse.sh`); zips manual | `src/timecourse_iterator.py` (all downstream analyses) |
| `badmodels.txt` *(optional, currently absent)* | manual skip list | `scripts/make_biomodels_timecourse.py`, `scripts/curate_models.py`, `scripts/make_piecewise_predictions.py` |

---

## `biomodels_endtime.csv`

- **(a) Used by**
  - `src/biomodels_iterator.py::getBiomodelsEndtimes()` (default path
    `cn.CALCULATED_ENTIMES_PATH`): supplies each model's end time and its source.
  - `BiomodelsIterator` items expose `end_time` / `endtime_source` from it.
  - `TimecourseIterator.__init__`: with the default `is_sedml_endtime=True`, only
    models whose source is `sedml` are considered valid.
  - `scripts/plot_biomodels.py` reads it directly (`ENDTIME_CSV`).
- **(b) Created by** `make_biomodels_endtime.py` (*writer removed*; named in
  `AGENTS.md`, never committed to git). End times come from the model's SED-ML
  when available (`sedml`), otherwise are computed heuristically (e.g.
  `max_median_cv`, `steadystate` — see the `ENDTIME_SOURCE_*` constants in
  `src/constants.py`).
- **(c) Contents** One row per BioModel:
  `model_name, end_time_source, endtime` (e.g.
  `BIOMD0000000001,max_median_cv,62.859…`).

## `curation.csv`

- **(a) Used by** `TimecourseIterator(is_curated=True)` via
  `src/timecourse_iterator.py::CurationChecker`, which excludes listed models from
  iteration. Written path is `cn.CURATION_PATH`.
- **(b) Created by** `scripts/curate_models.py`: simulates each BioModel and
  records models unsuitable for piecewise-linear analysis. The file is rewritten
  on every model processed (incremental).
- **(c) Contents** `system_id, reason` where reason is one of: `Events`,
  `Conserved moieties`, `Negative species values`,
  `Extreme differences in species values` (max/min species ratio > 1e6).

## `curation_removed_models.csv`

- **(a) Used by** nothing in the codebase — it is a record of manual curation
  decisions.
- **(b) Created by** hand (the file begins with a comment line: *"The following
  models are removed from curation because for the reasons indicated."*).
- **(c) Contents** `system_id, reason` with free-text reasons (e.g. *"Has a
  bursting behavior that is not differentiable."*).

## `find_oscillating_models.csv`

- **(a) Used by**
  - `notebooks/oscillations.ipynb`: splits species into oscillating vs.
    non-oscillating (`frequencies == "[]"` means non-oscillating).
  - `notebooks/piecewise_predictions.ipynb`: selects non-oscillating species
    timecourses for piecewise prediction.
- **(b) Created by** `scripts/find_oscillating_models.py` (*writer removed*; only
  a compiled `.pyc` remains in `scripts/__pycache__/`). Detection is based on the
  FFT-based `src/oscillation_detector.py::findOscillations`.
- **(c) Contents** One row per (model, species): `system_id, species_name,
  frequencies, endtime`, where `frequencies` is a stringified list of detected
  oscillation frequencies in Hz (e.g. `[0.02997]` or `[]`).

## `benchmark_changepoints.csv`

- **(a) Used by** nothing in the codebase — an analysis artifact comparing
  changepoint-detection methods.
- **(b) Created by** `scripts/benchmark_changepoint_methods.py` (default output
  path is this file). Incremental: appends results per model; `--initialize`
  resets the file.
- **(c) Contents** One row per (model, method): `method` (`PiecewiseSystemDiscovery.Efficient`
  or `.Divideandconquor`), `model_name`, `num_changepoints`, `score_p10`,
  `detect_time_sec`, `fit_time_sec`, `failed`, `error`.

## `linear_predictor_scores-0.001.csv`

- **(a) Used by**
  - `scripts/make_paper_plots.py`: reads the `-0.001.csv` file for paper figures
    (accuracy CDFs).
  - `notebooks/linearity_analysis.ipynb` and `notebooks/perturbation_analysis.ipynb`.
- **(b) Created by** `scripts/calculate_linear_prediction_scores.py` (*writer
  removed*; never committed). It was launched in parallel by the (also removed)
  root-level `calculate_linear_prediction_scores.sh`, which sharded models across
  processes and wrote per-process files that were merged into the final CSV. Per
  `AGENTS.md`, it ran `LinearPredictor` on each model and wrote score CSVs to
  `data/`. The `-0.001` suffix is the SINDy coefficient threshold used.
- **(c) Contents** Accuracy statistics (Absolute Relative Error based, see
  `src/score.py`) of linear predictions: one row per model and per species —
  `system_id, aggregation_type` (`model` or a species name), then the standard
  statistic columns `mean, min, max, count, invalid_count, p05 … p99`.

## `perturbation_study-model0.001.csv` and `perturbation_study-species0.001.csv`

- **(a) Used by**
  - `scripts/make_paper_plots.py`: reads both files for paper figures.
  - `tests/test_perturbation_study.py` exercises the producing code path.
- **(b) Created by** `scripts/perturbation_study.py`. For each model (with an
  explicitly specified end time, minus a hardcoded exclusion list) it runs
  `PerturbationAnalyzer` with initial-condition perturbations of −50%, −20%,
  −10%, −5%, 0%, +5%, +10%, +20%, +50% and records R² (derivative method) at
  threshold `--threshold` (default 0.001). Output filename is
  `perturbation_study-{model|species|model_species}{threshold}.csv` depending on
  the `--is-analyze-model` / `--is-analyze-species` flags. Incremental: models
  already present in the file are skipped on re-run.
- **(c) Contents** One row per (model or species, perturbation level): the
  standard statistic columns (`mean … p99`, `count`, `invalid_count`) of R², plus
  `perturbation` (fraction, e.g. `-0.5`), `fraction_species_perturbable`, and
  `coefficient_threshold`.

## `piecewise_predictions_model.zip`

- **(a) Used by** referenced as `cn.PIECEWISE_PREDICTIONS_MODEL_PATH`
  (`data/piecewise_predictions_model.csv`) and by
  `tests/test_check_psd_prediction_csvs.py`. The zip is a portable copy of that
  CSV.
- **(b) Created by** the inner CSV is written by
  `scripts/merge_piecewise_predictions_model.py`, which concatenates all
  `data/piecewise_predictions__numpoint*.csv` files (adding a `source_file`
  column) and keeps only rows with `aggregation_type == 'model'`. The zip itself
  was created manually.
- **(c) Contents** A single `piecewise_predictions_model.csv`: model-level rows
  from every piecewise-prediction run — the standard prediction columns (see
  `psd_data/` below) plus `source_file`.

## `piecewise_predictions_pkl.zip`

- **(a) Used by** nothing in the codebase — a historical archive (see
  `data/README.md`: "archives are older files that may be needed for historical
  purposes").
- **(b) Created by** manual archiving of pickle prediction files (22
  `piecewise_predictions__*.pkl` entries), i.e. a snapshot of what is now in
  `psd_data/`.
- **(c) Contents** Pickled DataFrames with the same schema as the `psd_data/`
  pickles (see below).

## `psd_analyzer_bestmodel.zip`

- **(a) Used by** `src/psd_analyzer.py::PSDAnalyzer.makeBestPSDModelDF`, which by
  default looks for the desired bestmodel CSV (by filename) inside this archive
  (`BESTMODEL_ZIP_PATH`) and loads it if present. Otherwise it generates the CSV,
  writes it to `data/`, and stores it back in this archive (adding the entry if the
  archive exists without it, creating the archive if missing; `is_force_update`
  replaces an existing entry). The individual
  `psd_prediction_analyzer_bestmodel__*.csv` files in `data/` remain the live
  artifacts; this zip is a portable snapshot.
- **(b) Created/maintained by** `makeBestPSDModelDF`, which adds a bestmodel CSV to
  this archive whenever it generates one (creating the archive on first use). It was
  originally seeded by manual archiving of six bestmodel CSVs (min_accuracy 0.8/0.9
  × accuracy_stat max/min/p50). Regenerate the CSVs via
  `scripts/update_psd_prediction_files.py` (which calls with `is_force_update=True`)
  to refresh the snapshot in place.
- **(c) Contents** The six `psd_prediction_analyzer_bestmodel__*.csv` files
  (schema described below).

## `psd_data/` (directory)

- **(a) Used by**
  - `src/psd_prediction_file_iterator.py`: default data directory
    (`cn.PSD_DATA_DIR`); iterates/queries the CSV and PKL prediction files.
  - `src/psd_analyzer.py::PSDAnalyzer`: queries these files per model to select
    the best piecewise prediction.
  - `scripts/make_paper_plots.py`: reads specific files directly (e.g. the
    `manycp_1__repeat_1` CSV for accuracy CDFs).
  - `scripts/check_psd_prediction_csvs.py`: integrity-checks that each file's
    rows match the metadata encoded in its filename.
- **(b) Created by** `scripts/make_piecewise_predictions.py`, normally launched in
  parallel via the `run/make_piecewise_predictions/make` wrapper (which shards
  model ranges across processes; see `local_make_files.sh` for example
  invocations). Each run is configured by parameters that are encoded into the
  filename (`numpoint`, `threshold`, `removal`, optional `maxreduction`,
  `manycp`, and a `repeat` index for repeated runs). The `boundaries` and `auc`
  columns are added/refreshed in place by `scripts/update_psd_prediction_files.py`
  (which also backs up each file to `.bak` before overwriting).
- **(c) Contents** 92 files = 46 CSV/PKL pairs named
  `piecewise_predictions__numpoint_100000__threshold_<t>__removal_<0|1>[__maxreduction_<r>]__manycp_<0|1>__repeat_<n>.{csv,pkl}`.
  Each file holds one row per (model, `max_changepoint`) prediction configuration:
  `system_id`, `aggregation_type` (`model` or species name), accuracy statistics
  (`mean … p99`, `count`, `invalid_count`), accuracy fractions (`a80, a90, a95`),
  `changepoints` (list of detected change-point indices), `max_changepoint`,
  `max_fractional_reduction`, `coefficient_threshold`, `num_changepoint`,
  `is_changepoint_removal`, `num_species`, `boundaries` (segment boundary
  indices), and `auc` (area under the cumulative segment curve for that row's
  changepoints).

## `psd_prediction_analyzer_bestmodel__minaccuracy_<m>__accuracystat_<s>.csv`

Seven files: `minaccuracy` ∈ {0.8, 0.9} × `accuracystat` ∈
{max, min, p50, mean}.

- **(a) Used by**
  - `notebooks/psd_prediction_analysis.ipynb`: generates and consumes them via
    `PSDAnalyzer.makeBestPSDModelDF` (the method caches to exactly these paths).
  - `PSDAnalyzer.makeAUCCDFPlots` / `makeLongestSegmentCDFPlot`: call
    `makeBestPSDModelDF`, loading these files when they already exist.
- **(b) Created by** `src/psd_analyzer.py::PSDAnalyzer.makeBestPSDModelDF`
  (default path pattern `OUTPUT_PATH_PAT`). For every model in the timecourse
  archive it selects, among that model's PSD prediction rows meeting
  `accuracy_stat >= min_accuracy`, the row with minimum AUC. The file is cached:
  an existing file is loaded instead of regenerated unless `is_force_update=True`.
  `scripts/update_psd_prediction_files.py` regenerates the mean/p50 combinations
  with `is_force_update=True` after updating the prediction files (skippable via
  `--skip-bestmodels`). Models with no qualifying row are skipped and reported in
  a warning.
- **(c) Contents** One row per BioModel: `index` (row counter), then the full
  prediction-row columns from `psd_data/` (`system_id`, accuracy statistics,
  `changepoints`, `boundaries`, `auc`, …) plus `filename` (the source prediction
  file the row came from).

## `rate_law_statistics.csv`

- **(a) Used by** nothing in the codebase — an analysis artifact summarizing
  reaction kinetics across BioModels. `tests/test_rate_law_statistics.py` covers
  the producing code.
- **(b) Created by** `scripts/rate_law_statistics.py` (default output is this
  file). Classifies every reaction's kinetic law in each BioModel SBML into the
  twelve mutually exclusive SBMLKinetics 2DK types (ZERO, UNDR1, UNDR2, UNMO,
  BIDR1, BIDR2, BIMO, MM, MMCAT, HILL, FR, NA) following the SBMLKinetics
  reference implementation (Xu et al. 2023). Incremental: writes after each model
  and skips already-processed models on re-run.
- **(c) Contents** One row per model: `model_num`, `system_id`, `num_species`,
  `num_reaction`, `num_reaction_with_kinetic_law`,
  `num_reaction_without_kinetic_law`, a count column per kinetics type, and a
  comma-separated reaction-name list per type (`ZERO_reaction_names`, …).

## `serialize/timecourse/` (directory)

- **(a) Used by** `src/timecourse_iterator.py::TimecourseIterator`, the entry
  point for every analysis that needs a model's simulated timecourse (piecewise
  predictions, perturbation study, PSD analysis, plotting). The default archive is
  `cn.TIMECOURSE_ZIP_PATH` = `timecourse_100000.zip`;
  `TimecourseIterator.getZipPath(num_point)` selects the 1k/10k/100k archive by
  point count. If a model is missing from the zip, the iterator falls back to
  simulating it from SBML.
- **(b) Created by** `scripts/make_biomodels_timecourse.py` (optionally via the
  parallel wrapper `scripts/run_parallel_timecourse.sh <N>`). The script writes
  one pickle per model (`<model_name>_timecourse.pkl`) into this directory,
  verifying each by round-trip deserialization; the three zips are manual
  archives of those pickles at 1,000 / 10,000 / 100,000 points. A hardcoded
  `EXCLUDED_MODELS` list (plus optional `data/badmodels.txt`) is skipped.
- **(c) Contents** `timecourse_1000.zip` (~35 MB), `timecourse_10000.zip`
  (~304 MB), `timecourse_100000.zip` (~2.7 GB). Each entry is a pickled
  `Timecourse`: the SBML model, start/end times, point count, and the simulated
  `timecourse_df` (species concentrations vs. time). Per `data/README.md`, the
  1,000-point archive contains models with explicitly specified, non-default end
  times and no Jacobians.

## `badmodels.txt` (optional — currently absent)

- **(a) Used by** read as an optional skip list at import time by
  `scripts/make_biomodels_timecourse.py`, `scripts/curate_models.py`, and
  `scripts/make_piecewise_predictions.py` (one model name per line, `#` comments
  allowed). Also documented in `AGENTS.md`.
- **(b) Created by** hand, when a model is found to be broken (segfaults, bad
  endtimes, etc.). The file does not currently exist in `data/`; the scripts
  simply skip reading it.
- **(c) Contents** Plain-text list of BioModel names to exclude from batch runs.

## `README.md`

- **(a) Used by** human readers of the data directory.
- **(b) Created by** hand.
- **(c) Contents** Conventions for the data files (threshold suffixes, timecourse
  archive description, linear-predictor score files, archives). Note: it refers to
  `data/serialized/timecourse/timecourse.zip` and an `@archives` directory; the
  actual layout is `data/serialize/timecourse/timecourse_<n>.zip` and there is no
  `archives/` directory — treat those lines as slightly out of date.
