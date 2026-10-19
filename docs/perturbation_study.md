# Perturbation Study

This study explores how perturbing initial species values affects the quality of linear fits for models and for individual species.

In `data/evaluate_monomial_models-0.01`, about 15% of the models have a `deg1_min` of at least 0.9 on their training data. This study compares those unperturbed results with perturbations applied to initial values. In each run, the training timecourse is simulated unperturbed. Separately, new timecourses are constructed where all species' initial values are changed by a signed fractional amount: ±5%, ±10%, ±20%, ±50% (plus an unperturbed 0% reference), with `perturbation_species_fraction=1.0`. For each perturbation level a separate ``Timecourse`` is simulated, and then the SINDy model trained on the unperturbed training data is used to predict each perturbed timecourse. Use the class method ``SystemDiscovery.analyzePerturbations`` (formerly ``perturbationAnalysis``) for this analysis. The per-model results are written as CSV files in `data/`.

## Parameters of ``analyzePerturbations``

| Parameter | Type | Default | Description |
|---|---|---|---|
| ``model`` | Model \| int | — | Simulates ground-truth timecourses for each perturbation. Accepts a model object or an integer BioModel number. |
| ``training_df`` | pd.DataFrame | NULL_DF | Unperturbed timecourse used to fit the SINDy model. |
| ``threshold`` | float | 0.001 | STLSQ sparsity threshold. |
| ``poly_degree`` | int | 1 | Degree of the polynomial library (default linear). |
| ``perturbations`` | list[float] | [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5] | Signed fractional perturbation values applied to initial species concentrations. |
| ``col_percentile`` | str | "p10" | Column of the accuracy DataFrame used for per-perturbation summaries in plots. |
| ``perturbation_species_fraction`` | float | 1.0 | Fraction of non-zero species whose initial values are perturbed when simulating each perturbed timecourse. |
| ``frac_scatter_skip`` | float | 0.2 | Scatter-plot density: `num_skip = max(1, int(n_points * frac_scatter_skip))`. |
| ``figsize`` | tuple[float, float] \| None | None | Figure size in inches for the trajectory comparison plot; auto-sized when None. |
| ``plot_species_names`` | list[str] \| None | None | Species to include in the per-perturbation trajectory figure. Defaults to all species. |
| ``subtitle`` | str | "Perturbation Analysis" | Title rendered on the output figure. |
| ``is_analyze_model`` | bool | True | Include model-level rows (`aggregation_type='model'`) in the returned DataFrame. |
| ``is_analyze_species`` | bool | True | Include per-species rows (one row per species name) in the returned DataFrame. |
| ``is_plot`` | bool | True | Whether to show a trajectory comparison figure alongside the accuracy metrics. |

## What is calculated

For each perturbed timecourse, the fitted SINDy model predicts concentrations at every timepoint. Accuracy is computed pointwise as:

```
accuracy = max(0, 1 - abs(prediction - actual) / abs(actual))
```

Values are clipped to `[0, 1]`. Timepoints where `actual` is zero or non-finite receive a sentinel accuracy of `-1` and are excluded from aggregation. For each species × timepoint pair the Accuracy score aggregates across time via `StatisticCalculator`, producing percentile columns (`mean`, `min`, `max`, `count`, `invalid_count`, `p05`, `p10`, `p20`, `p25`, `p30`, `p50`, `p80`, `p90`, `p95`, `p99`).

## Output structure

``analyzePerturbations`` returns an ``AnalyzePerturbationsResult`` named tuple containing:

- ``.df`` — a DataFrame with one row per perturbation value at each requested aggregation level.
- ``.fig`` — the trajectory comparison figure (or `None` if `is_plot=False`).

### Rows in the DataFrame

| Column | Source | Notes |
|---|---|---|
| ``perturbation`` | set from input list | The signed fractional perturbation value for this row; preserved exactly per-row. |
| ``fraction_species_perturbable`` | `perturbation_species_fraction` argument | Same for every row in the result. |
| ``system_id`` | `model.model_name` | Model name, repeated on each row. |
| ``aggregation_type`` | computed | `'model'` for aggregated rows; species name string for per-species rows. |
| ``mean``, ``min``, ``max``, ``count``, ``invalid_count``, ``p05``–``p99`` | `StatisticCalculator` on the Accuracy score | Aggregated across valid (non-sentinel) timepoints within that aggregation level. |

### Model-level vs species-level rows

- With **both** flags True (the default), the result contains both:
  - One model-level row per perturbation (`aggregation_type='model'`); statistics are aggregated across all non-zero species at each timepoint.
  - One per-species row per perturbation (`aggregation_type=<species name>`).
- Set `is_analyze_model=False` or `is_analyze_species=False` to drop the corresponding level from the output entirely.

### CSV outputs in ``scripts/perturbation_study.py``

The script writes three variant CSVs, chosen by which aggregation levels are requested:

| Mode | Path pattern |
|---|---|
| model + species (default) | `data/perturbation_study-model_species{THRESHOLD}.csv` |
| model only | `data/perturbation_study-model{THRESHOLD}.csv` |
| species only | `data/perturbation_study-species{THRESHOLD}.csv` |

Each row in a CSV file is one perturbation × aggregation-level combination (i.e. not one row per model). The script skips models listed in its `EXCLUDES` set and resumes from an existing CSV on subsequent runs by checking `system_id`.
