# Analyze the accuracy of a linear approximations for model kinetics

## Objectives

* Assess how well an SBML is approximated by one or more linear models based on
  * Consistency of Jacobian
  * Accuracy of reproducing a non-linear simulation
* Identify "linearity bottleneck" reactions, those that must limit the viability of a linear approximation

## Repository Structure

## Scripts

The `scripts/` directory contains executable tools for running analyses:

| Script | Description |
|--------|-------------|
| `analyze_biomodels.py` | Analyzes BioModels for linearity properties |
| `calculate_linear_prediction_scores.py` | Computes linear prediction scores for models |
| `evaluate_monomial_models.py` | Evaluates monomial-based kinetic models |
| `make_biomodels_endtime.py` | Determines simulation end times for BioModels |
| `make_biomodels_timecourse.py` | Generates timecourse data from BioModels |
| `perturbation_study.py` | Runs perturbation studies to test model stability |
| `pwla.py` | Piecewise Linear Approximation utilities |
| `run_parallel_timecourse.sh` | Shell script for parallelized timecourse simulations |

## Documentation

The `docs/` directory contains technical documentation and design documents:

| Document | Description |
|----------|-------------|
| `calculating_endtime_using_cv.md` | Methods for determining simulation end times using coefficient of variation |
| `find_simulation_end_time.md` | Algorithms for detecting simulation end points |
| `implementing_piecewise_segmentations.md` | Guide to implementing piecewise linear segmentation |
| `linear_predictions.md` | Theory and implementation of linear prediction methods |
| `model_based_design.md` | Model-based design principles and approaches |
| `network_discovery.md` | Techniques for discovering network structures |
| `perturbation_study.md` | Documentation for perturbation study methodology |
| `piecewise_system_discovery.md` | Piecewise linear approximation using cluster-based Jacobian fitting |
| `research_agenda.md` | Research roadmap and future directions |
| `score.md` | Scoring methodology documentation |
| `specification.md` | Project specifications and requirements |
| `technical_notes.md` | General technical notes and implementation details |

Subdirectories:
- `bugs/` — Documented bugs and issues
- `references/` — Reference materials and papers
- `superpowers/` — Advanced features and extensions

## Source Code

The `src/` directory contains the core Python library:

| Module | Description |
|--------|-------------|
| `biomodels_iterator.py` | Iterates over BioModels for analysis |
| `constants.py` | Project-wide constants and configuration |
| `crn_builder.py` | Builds Chemical Reaction Network models |
| `dataframe_serializer.py` | Serialization utilities for DataFrames |
| `linear_predictor.py` | **Core module** — Predicts model timecourses using linear ODE approximations with Jacobian-based methods |
| `model.py` | Base model class for SBML models |
| `multiple_linear_predictor.py` | Handles multiple simultaneous linear predictions |
| `plot_options.py` | Visualization and plotting utilities |
| `scaler.py` | Data scaling utilities |
| `score.py` | Scoring and evaluation metrics |
| `system_discovery.py` | Discovers system parameters and structures |
| `timecourse_iterator.py` | Iterates over timecourse data |
| `timecourse.py` | Timecourse data handling and processing |
| `trajectory.py` | Trajectory class for model simulations |
| `utils.py` | General utility functions |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis on a BioModel
python scripts/analyze_biomodels.py

# Calculate prediction scores
python scripts/calculate_linear_prediction_scores.py
```

## Tuning SystemDiscovery

* Need sufficient data, maybe in the 1000s.
* Reduce threshold (sensitivity) to get more sparse coefficients
* Set includebias = True
* Don't include boundary species

## Analyses


## Versions