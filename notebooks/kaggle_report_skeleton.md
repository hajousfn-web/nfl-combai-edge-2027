# NFL CombAI Edge: Uncovering Non-Intuitive Tracking Metrics

> **Report status:** Structural draft. No competition data has been loaded and no
> empirical finding is claimed in this document.
>
> Replace every bracketed placeholder with verified dataset details or measured
> results before publishing the Kaggle report. Keep raw data local to its
> authorized environment; do not download full datasets just to populate this
> skeleton.

## Contents

1. [Executive Summary & Sovereign Edge](#1-executive-summary--sovereign-edge)
2. [Data Ingestion & Rookie Cohort](#2-data-ingestion--rookie-cohort)
3. [Micro-Kinematic Feature Engineering](#3-micro-kinematic-feature-engineering)
4. [Statistical Integration & Baseline](#4-statistical-integration--baseline)
5. [Visualizations & Strategic Conclusion](#5-visualizations--strategic-conclusion)
6. [Reproducibility and limitations](#6-reproducibility-and-limitations)

## 1. Executive Summary & Sovereign Edge

### Research question

Do high-frequency (10 Hz) Combine movement characteristics—especially
deceleration and sharp-cut speed retention—show useful associations with
rookie-season on-field performance for players drafted in 2023–2025?

### Why look beyond conventional summaries?

Route clusters and aggregate testing results can hide short changes in speed
and direction. This project studies frame-to-frame movement as a complementary,
interpretable view. It does not assume that a new metric is useful until the
cohort, data quality, and empirical results have been checked.

### Sovereign engineering approach

- Keep the processing, feature engineering, statistical integration, and
  visualization modules separate.
- Do not fetch competition datasets from this notebook. Configure paths only
  to data already available in an authorized local or Kaggle input.
- Inspect schemas and small samples before deciding how to execute transforms.
- Keep generated reports, plots, and model artifacts in `outputs/`; do not
  commit raw, intermediate, or restricted data.
- Treat cohort definitions and conclusions as human-reviewed decisions.

**Executive finding:** [Complete after analysis; do not infer from mock tests.]

## 2. Data Ingestion & Rookie Cohort

### Expected tables

| Table | Intended use | Required relationship / fields to verify |
|---|---|---|
| `players.csv` | Player identity and rookie cohort | `nflId` or documented player key; `draft_year` |
| `combine_results.csv` | Combine testing results | Player key; testing columns and units |
| `combine_tracking.csv` | High-frequency Combine observations | Player key, `game_id`/event context as applicable, `frame_id`, `x`, `y` |
| `games.csv` | Game-season context | Unique `game_id`; `season` |
| `player_play.csv` | On-field player/play outcomes | Player key, `game_id`, YAC/EPA/stop fields as available |

These are expected names, not a claim that files or fields have been inspected.
Confirm the actual competition schema, units, missing-value codes, ID types,
and table grain before joining.

### Cohort definition

The planned rookie cohort is:

```text
draft_year ∈ {2023, 2024, 2025}
rookie-season performance where season == draft_year
```

If `player_play.csv` does not carry `season`, attach it from `games.csv` using
`game_id`. Validate that `games.csv` has one season per `game_id` and that all
join keys have compatible representations. Do not infer draft class or playing
season from row order or player IDs.

### Schema preflight checklist

- [ ] Record row counts and unique-player/game counts without loading all rows.
- [ ] Verify that `draft_year`, `season`, `game_id`, and player identifiers have
  the expected types and values.
- [ ] Check uniqueness/grain for each table before merging.
- [ ] Confirm whether performance fields are per play, per game, or already
  player-season aggregates.
- [ ] Document how missing, inactive, and non-applicable values are encoded.
- [ ] Confirm that YAC, EPA, and defensive-stop fields exist and define their
  meanings before analysis.

**Verified cohort size:** [Players after validated cohort filter]

**Season coverage:** [Seasons and games represented]

**Excluded records and reasons:** [Document counts and rules]

## 3. Micro-Kinematic Feature Engineering

The module `src/feature_engineering.py` expects an in-memory pandas DataFrame
with `game_id`, `play_id`, `player_id`, `frame_id`, `x`, and `y`; it sorts by
track and frame before calculating movement. When available, a consistent
`draft_year` is retained in player summaries.

### Measures

- **Frame-to-frame speed:** displacement divided by elapsed frame time at 10 Hz.
- **Deceleration:** positive speed loss between successive movement segments,
  expressed in coordinate-units per second squared.
- **Direction change:** angle between consecutive non-zero displacement
  vectors.
- **Sharp cut:** a direction change strictly greater than 45 degrees.
- **Cut efficiency:** post-cut speed divided by pre-cut speed for a sharp cut.
  This is a speed-retention proxy; without mass and force measurements, it is
  not physical momentum.

Coordinates are assumed by the implementation to be yards. Confirm that this
matches the source data before interpreting speeds or decelerations.

### Engineering decisions to document

- [ ] Confirm that `frame_id` is the correct 10 Hz time index.
- [ ] Check duplicate frames and missing/interrupted observations.
- [ ] Verify track boundaries (`game_id`, `play_id`, `player_id`) and coordinate
  direction conventions.
- [ ] Determine whether gaps in frames need a maximum permitted duration.
- [ ] Inspect the number of valid sharp cuts and zero-speed edge cases.
- [ ] State whether values are summarized per test, player, or another unit.

**Feature summary:** [Insert validated counts, distributions, and units.]

**Quality checks:** [Insert checks and outcomes.]

## 4. Statistical Integration & Baseline

The module `src/model_integration.py` takes:

```python
integrate_combine_with_season_performance(
    player_summary_df,
    regular_season_df,
    games_df=None,
)
```

The Combine summary must contain `draft_year`, a player identifier, and a cut
efficiency column. Performance rows must include `season`, or the games table
must supply `game_id` and `season`. The function filters to draft years
2023–2025 and keeps observations whose `season == draft_year`, then aggregates
repeat rows per player before its left join.

Candidate outcomes are YAC, EPA, and defensive stops **only when the actual
input schema contains verified numeric columns for them**. Inspect whether
each quantity should be summed, averaged, or treated another way at the
player-season level; the module's generic repeated-row aggregation must not be
mistaken for a domain-validated outcome definition.

### Analysis to report

| Outcome | Eligible players | Missingness | Pearson correlation | Ridge coefficient | Validation design |
|---|---:|---:|---:|---:|---|
| YAC | [ ] | [ ] | [ ] | [ ] | [ ] |
| EPA | [ ] | [ ] | [ ] | [ ] | [ ] |
| Defensive stops | [ ] | [ ] | [ ] | [ ] | [ ] |

The integration function's one-feature standardized Ridge baseline is
descriptive and reports in-sample R². For a predictive baseline, the module
also exposes `run_model_pipeline(features_df)`, which expects one already
aggregated row per player-season, with numeric movement features, player ID,
`draft_year`, `season`, and both EPA and YAC labels. It rejects repeated
player-season rows rather than guessing how play outcomes should aggregate.
For each target, the function uses median imputation and scaling inside a Ridge
pipeline; inner K-fold CV tunes regularization and outer-fold predictions
provide RMSE, MAE, and R². It saves aggregate-only metrics to
`outputs/model_pipeline_metrics.csv`, not player records or model artifacts.

Populate the validation design and metrics only after running against verified
labels. Report eligible sample counts and missingness; a small rookie cohort
can produce unstable estimates. Neither the descriptive Ridge baseline nor
cross-validated prediction establishes causality.

**Observed relationships:** [Populate only after running on verified data.]

**Alternative explanations and confounders:** [Document.]

**Human review / decision:** [Record what the evidence does and does not support.]

## 5. Visualizations & Strategic Conclusion

`src/visualization.py` uses Matplotlib's non-interactive Agg backend and a
color-blind-friendly palette. `generate_project_visualizations()` can plot
raw/precomputed 10 Hz velocity and sharp-cut speed retention, and can compare
paired out-of-sample actual/predicted EPA and YAC. Prediction comparison files
must supply actual and predicted values. Per-player vertical bounds are shown
only when lower and upper prediction-bound columns are both present; a
paired-bootstrap 95% confidence interval for mean prediction error is also
reported. No individual prediction intervals are fabricated.

The current modeling entry point saves aggregate cross-validation metrics, not
player-level predictions. Therefore, prediction comparison plots require
separately retained out-of-fold predictions from a reviewed evaluation run.
The legacy scatter utility remains available for efficiency/performance CSVs
and the aggregate correlation-vs-Ridge-coefficient plot.

Example after producing an authorized, prepared player-level file:

```bash
python -c "from src.visualization import generate_project_visualizations; generate_project_visualizations('outputs/tracking_features.csv', 'outputs/model_predictions.csv')"
```

**Figures to include:**

1. [Cut efficiency vs. YAC, with cohort/sample size and missingness stated.]
2. [Cut efficiency vs. EPA, if verified and analyzed.]
3. [Cut efficiency vs. defensive stops, if verified and analyzed.]
4. [Optional: correlation/Ridge-coefficient overview, clearly labelled as
   descriptive.]

**Scout/coach interpretation:** [Human-authored, tied to observed evidence.]

**Strategic recommendation:** [Human decision; include uncertainty and scope.]

Do not present an association as a causal mechanism or scouting rule without
appropriate validation and domain review.

## 6. Reproducibility and limitations

Run mock-data validation from the project root:

```bash
python src/pipeline.py --self-test
python src/feature_engineering.py --self-test
python src/model_integration.py --self-test
python src/visualization.py --self-test
```

The tests validate code paths on small fixtures, not competition data, cohort
coverage, outcome definitions, statistical significance, or generalization.
Feature engineering currently takes a pandas DataFrame in memory; do not pass a
full large raw tracking dataset until a bounded chunking/aggregation approach
has been designed and validated. The CSV tracking pipeline streams input but
requires rows sorted by game, play, player, and frame.

### Final publication checklist

- [ ] All tables, field names, units, joins, and cohort counts verified.
- [ ] Rookie-season filtering (`season == draft_year`) independently checked.
- [ ] Outcome aggregation choices documented and reviewed.
- [ ] Results generated from authorized data, with no mock values in the report.
- [ ] Plots include readable labels, sample sizes, and appropriate caveats.
- [ ] Conclusions are human-reviewed and distinguish association from
  prediction/causation.
- [ ] Notebook executes in the intended Kaggle environment with no network
  downloads or local-machine path assumptions.
- [ ] No raw or restricted data is embedded in notebook outputs or committed.
