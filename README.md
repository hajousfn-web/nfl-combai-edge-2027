# NFL CombAI Edge: Uncovering Non-Intuitive Tracking Metrics

An analytics project for the NFL Big Data Bowl 2027 competition. We aim to find
tracking patterns that are informative but easy to miss with conventional
box-score and aggregate metrics.

## Sovereign workflow and core strategy

Keep raw and derived data local; do not download the full 2.31 GB competition
datasets to this workspace. Use streaming processing, bounded samples, and mock
fixtures for local validation. Keep feature engineering, statistical
integration, and visualization in separate modules. Generated artifacts stay
under `outputs/` and are excluded from Git. No script downloads data or sends
project data to a remote service.

The research objective is to assess whether 10 Hz Combine movement,
deceleration, and cut-efficiency metrics have non-obvious associations with
rookie-season YAC, EPA, and defensive stops for the 2023–2025 rookie classes.
Keep the analysis descriptive and transparent: human reviewers choose cohort
definitions and interpret results; correlation and the lightweight Ridge
baseline are not evidence of causality or out-of-sample predictive value.

## Project structure

- `data/` — local input samples and datasets; data files are excluded from Git.
- `src/pipeline.py` — bounded end-to-end orchestration plus a legacy streaming
  CSV preparation utility.
- `src/feature_engineering.py` — smoothed 10 Hz velocity/acceleration and
  sharp-cut speed-retention features; preserves validated `draft_year` when
  present.
- `src/model_integration.py` — player-level combine/season integration,
  descriptive associations, and nested-cross-validated Ridge baselines.
- `src/visualization.py` — bounded-memory, headless PNG plots from local
  processed-metric or aggregate-insight CSVs.
- `notebooks/` — [Kaggle report skeleton](notebooks/kaggle_report_skeleton.md)
  and exported notebooks.
- `outputs/` — local charts, reports, and model artifacts (excluded from Git).

## Quick start

Run the complete workflow on a bounded tracking CSV with `game_id`, `play_id`,
`player_id`, `frame_id`, `x`, `y`, and `draft_year` columns:

```bash
python src/pipeline.py data/combine_tracking.csv \
  --performance-csv data/player_season_outcomes.csv
```

The performance file must already contain one player-season row per player,
with `nflId` (or another supported player identifier), `season`, `epa`, and
`yac`/`yards_after_catch`. The pipeline keeps only rookie-season rows where
`season == draft_year`, runs feature engineering and nested Ridge CV, and
prints/saves aggregate evaluation metrics. It deliberately rejects repeated
play-level outcome rows instead of guessing how EPA or YAC should be
aggregated. If no performance file is supplied, the feature summary is returned
in memory and model evaluation is skipped.

Tracking CSVs are loaded with a strict default ceiling of 250,000 rows; larger
inputs are rejected, not truncated, because truncation can split player
tracks. Use a filtered, complete-track sample, or pass a DataFrame directly to
`run_full_pipeline()` from Python. Raw tracking and player-level summaries are
not written to disk by the orchestrator. Model metrics are atomically saved
under `outputs/`.

The earlier low-memory displacement-only preparation utility remains
available when both input and output paths are provided:

```bash
python src/pipeline.py data/tracking_sample.csv outputs/tracking_metrics.csv
```

This preparation mode streams rows and keeps only the prior observation for
the active track. Input must be ordered by `game_id`, `play_id`, `player_id`,
then `frame_id`. It adds `dx`, `dy`, `displacement`, and `speed_10hz`; speed
accounts for frame gaps. The first observation for each track has empty
derived values. CSV outputs are atomically written only directly inside
`outputs/`.

Run each module's mock-data checks without loading competition data:

```bash
python src/pipeline.py --self-test
python src/feature_engineering.py --self-test
python src/model_integration.py --self-test
python src/visualization.py --self-test
```

Place only data you are permitted to use in `data/`; large or competition-
restricted datasets should remain local and must not be committed.

## Combine/season integration

Call `integrate_combine_with_season_performance(player_summary_df,
regular_season_df, games_df=None)` from `src/model_integration.py`. The combine
summary must include `draft_year` and a player ID (`nflId` preferred); the
function keeps 2023–2025 draft cohorts and matches performance from each
player's rookie season (`season == draft_year`). Performance data must include
`season`, or pass a `games_df` containing `game_id` and `season` to attach it to
`player_play`-style rows. Player-play `game_id` values are joined to the unique
game-season table before cohort filtering.

The function preserves cohort players in a left join, imputes missing
cut-efficiency values with the median, and compares the feature with recognized
numeric outcomes including YAC, EPA, and defensive stops. Repeated rookie-season
game rows are aggregated before the player-level join. It prints a correlation
and standardized Ridge-baseline report and writes aggregate-only insights to
`outputs/model_integration_insights.csv`; it does not export player-level
records. Keep cohort and field definitions human-reviewed; the statistical
summary is descriptive and does not establish causality or generalizable
prediction.

## Predictive baseline

`run_model_pipeline(features_df)` evaluates EPA and YAC using a scaled Ridge
regression pipeline with median imputation. The input must already be a
player-season table with exactly one row per player and rookie season, a player
identifier, `draft_year`, `season`, numeric movement features, and numeric EPA
and YAC columns (for example, `epa` and `yards_after_catch`). It retains only
the 2023–2025 draft cohorts where `season == draft_year`; repeated player-play
rows are rejected rather than aggregated with an assumed outcome definition.

Regularized-model tuning occurs inside nested shuffled K-fold validation, and
reported RMSE, MAE, and R² are computed from outer-fold predictions. Identifier,
cohort, coordinate, and recognized outcome columns are not used as predictors;
missing feature values are imputed within each training fold. At least six
valid player-season labels per outcome are required. The final full-data alpha
search is reported for reproducibility, but fitted estimators and player-level
predictions are not persisted. Aggregate results are atomically written to
`outputs/model_pipeline_metrics.csv`. Run
`python src/model_integration.py --self-test` to validate both integration and
the mock-data nested-CV path without reading competition data.
Cross-validation metrics are not evidence of causality and should be
interpreted cautiously when the eligible rookie sample is small.

Generated output files and model artifacts are intentionally kept local and
excluded from Git. The `outputs/` directory is preserved by a placeholder file;
publish selected final assets separately only when they are safe to share.

`generate_project_visualizations()` in `src/visualization.py` creates a
publication-styled velocity/cut dashboard from raw tracking rows or
feature-engineered observations, plus actual-vs-predicted EPA/YAC figures when
a paired evaluation table is supplied. For example, from Python:

```bash
python -c "from src.visualization import generate_project_visualizations; generate_project_visualizations('outputs/tracking_features.csv', 'outputs/model_predictions.csv')"
```

Tracking input must contain raw tracking fields or precomputed `speed`,
`direction_change_degrees`, and `proprietary_cut_efficiency` fields. The
dashboard shows a representative player's smoothed 10 Hz speed profile and
sharp-cut speed-retention scores. The model comparison table must contain
paired, out-of-sample actual and predicted EPA and/or YAC columns, e.g.
`actual_epa`/`predicted_epa` and `actual_yac`/`predicted_yac`. If per-row
prediction bounds are available, provide both lower and upper columns (e.g.
`epa_lower`/`epa_upper`); plotted vertical error bars use those supplied
intervals. The chart also reports a paired-bootstrap 95% confidence interval
for mean prediction error. It does not fabricate per-player intervals when
none are supplied. The current model pipeline exports aggregate CV scores
only, not per-player predictions, so those must come from a retained
out-of-fold evaluation workflow before comparison plots can be made.

Generated PNGs are saved directly under `outputs/`. CSV inputs are restricted
to that directory and capped at 250,000 rows; raw files are never modified.
The legacy cut-efficiency/performance scatter command remains available:

```bash
python src/visualization.py outputs/player_metrics.csv --y-column epa
```

It also continues to accept aggregate insights CSVs for descriptive
correlation-versus-Ridge-coefficient plots. Rendering uses Matplotlib's
non-interactive Agg backend and a color-blind-friendly palette.

## Roadmap

1. Inspect the competition data, document its schema, and validate the
   streaming preparation pipeline.
2. Explore contextual movement, spacing, acceleration, and interaction
   features at the play and player levels.
3. Test whether candidate metrics add insight beyond established baselines.
4. Refine the strongest findings with targeted visualizations and reproducible
   analysis notebooks.
5. Deliver a concise report explaining the methods, evidence, and limitations.
