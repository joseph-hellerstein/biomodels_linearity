# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Most important

1. Don’t assume. Don’t hide confusion. Surface tradeoffs.
2. Minimum code that solves the problem. Nothing speculative.
3. Touch only what you must. Clean up only your own mess.
4. Define success criteria. Loop until verified.

## Environment Setup

The project uses a local venv at `mla/`. Always activate it before running anything:

```bash
source activate.sh
```

## Access to BioModels

* You can iterate through BioModels using biomodels_iterator.BiomodelsIterator
* You can iterate through previously run simulation so many BioModels using timecourse_itertor.TimecourseIterator.
* You can access a the timecourse for a specific model using timecourse_iterator.getTimecourse.

`activate.sh` also adds `src/` to `PYTHONPATH`, so imports like `from src.trajectory import Trajectory` work in tests without a package install.

## Commands

```bash
# Run all tests
source activate.sh && python3 -m pytest tests/ -v

# Run a single test file
source activate.sh && python3 -m pytest tests/test_trajectory.py -v

# Run a single test by name
source activate.sh && python3 -m pytest tests/test_trajectory.py::TestTrajectory::test_makeFromSimulation -v

# Lint
source activate.sh && pylint src/

# Run with coverage
source activate.sh && python3 -m pytest tests/ --cov=src
```

## Architecture

All library code lives in `src/`, all tests in `tests/`, batch pipeline scripts in `scripts/`.

**Core class hierarchy:**

- **[src/model.py](src/model.py)** — `Model`: static SBML properties (species names, initial values, reaction count). Accepts SBML XML or Antimony strings; Antimony is converted to SBML on construction. RoadRunner is used transiently and not stored. Factory: `Model.makeBiomodel(model_name)`.

- **[src/timecourse.py](src/timecourse.py)** — `Timecourse`: newer sibling of `Trajectory`. Simulates and stores `timecourse_df` + `jacobian_collection_arr` lazily (no forcing inputs). Supports perturbation of initial species values. Serializes/deserializes via pickle to `data/serialize/timecourse/`. The simulation runs only once; accessing either property triggers it.

- **[src/score.py](src/score.py)** — `Score`: scores predictions against true timecourses using Absolute Relative Error (ARE = |predicted − true| / |true|). Persists results to CSV via `DataframeSerializer`. One `ScoreInfo` row per model (aggregation_type="model") plus one per species.

- **[src/biomodels_iterator.py](src/biomodels_iterator.py)** — `BiomodelsIterator`: iterates over BioModel directories in `cn.BIOMODELS_DIR`, yielding `BiomodelsItem`s. Supports skip-lists and model number ranges. `getBiomodelsEndtimes()` loads the pre-computed end-time CSV.

- **[src/timecourse_iterator.py](src/timecourse_iterator.py)** — `TimecourseIterator`: iterates over serialized `Timecourse` pickles inside `data/serialize/timecourse/timecourse.zip`.

- **[src/system_discovery.py](src/system_discovery.py)** — Uses PySINDy to discover sparse ODE systems from timecourse DataFrames. Assumes polynomial (up to quadratic) rate laws.

**[src/constants.py](src/constants.py)** — Project-wide paths and defaults. Notable: `BIOMODELS_DIR` points one level up to `../temp-biomodels/final/`; `TIMECOURSE_ZIP_PATH` points to the serialized timecourse archive.

**Data pipeline** (`scripts/`): `make_biomodels_endtime.py` → computes end times → `data/biomodels_endtime.csv`. `make_biomodels_timecourse.py` → simulates all models → serialized pickles in zip. `calculate_linear_prediction_scores.py` → runs `LinearPredictor` on each and writes score CSVs to `data/`.

## BioModels Data

SBML models are stored in `/Users/jlheller/home/Technical/repos/temp-biomodels/final/`. Each subdirectory (e.g. `BIOMD0000000001/`) contains `<ID>_url.xml` (the SBML file) and `manifest.xml` (skip this). Models listed in `data/badmodels.txt` are known-broken and excluded from batch runs.

## Coding Style

Delegate all coding style to ``python-coder.md`` $HOME/.clinerules and $HOME/.claude/agents.

## Tests

Delegate all coding style to ``test-builder.md`` in $HOME/.clinerules and $HOME/.claude/agents.

## Folder Notes

* @data contains data produced in analyses
* Many directories have a ``README.md`` that describe how the contents of the directory
