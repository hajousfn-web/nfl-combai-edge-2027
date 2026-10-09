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
- `src/pipeline.py` — streaming CSV preparation and 10 Hz displacement-speed
  derivation.
- `src/feature_engineering.py` — per-player deceleration, speed, and
  sharp-cut-efficiency summaries; preserves validated `draft_year` when present.
- `src/model_integration.py` — player-level combine/season left join,
  correlation and Ridge-baseline report.
- `src/visualization.py` — bounded-memory, headless PNG plots from local
  processed-metric or aggregate-insight CSVs.
- `notebooks/` — exported Kaggle notebooks and exploratory analysis.
- `outputs/` — local charts, reports, and model artifacts (excluded from Git).

## Quick start

Run the pipeline with a tracking CSV that has `game_id`, `play_id`, `player_id`,
`frame_id`, `x`, and `y` columns:

```bash
python src/pipeline.py data/tracking_sample.csv outputs/tracking_metrics.csv
```

The script streams rows rather than loading the full dataset into memory and
keeps only the prior observation for the active track. Input must be ordered
by `game_id`, `play_id`, `player_id`, then `frame_id`. It adds `dx`, `dy`,
`displacement`, and `speed_10hz`; speed accounts for frame gaps. The first
observation for each track has empty derived values. CSV outputs are atomically
written only directly inside `outputs/`.

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

Generated output files and model artifacts are intentionally kept local and
excluded from Git. The `outputs/` directory is preserved by a placeholder file;
publish selected final assets separately only when they are safe to share.

To create a plot from player-level metrics, place the processed CSV in
`outputs/` and run:

```bash
python src/visualization.py outputs/player_metrics.csv
```

The default plot is saved to `outputs/cut_efficiency_vs_performance.png`.
Player-level CSVs should include cut efficiency and a recognized performance
column such as `yards_after_catch` or `defensive_stops`. The integration
insights CSV is also supported and plots the Pearson correlation against its
standardized Ridge coefficient for each outcome. Rendering uses Matplotlib's
non-interactive Agg backend and samples at most 10,000 valid rows.

## Roadmap

1. Inspect the competition data, document its schema, and validate the
   streaming preparation pipeline.
2. Explore contextual movement, spacing, acceleration, and interaction
   features at the play and player levels.
3. Test whether candidate metrics add insight beyond established baselines.
4. Refine the strongest findings with targeted visualizations and reproducible
   analysis notebooks.
5. Deliver a concise report explaining the methods, evidence, and limitations.
