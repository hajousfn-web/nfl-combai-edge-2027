# NFL CombAI Edge: Uncovering Non-Intuitive Tracking Metrics

An analytics project for the NFL Big Data Bowl 2027 competition. We aim to find
tracking patterns that are informative but easy to miss with conventional
box-score and aggregate metrics.

## Core strategy

Build a reproducible, data-efficient workflow around player-level 10 Hz tracking
data. Start with careful data validation and play context, derive interpretable
movement and interaction metrics, and evaluate whether those metrics reveal
meaningful behaviors beyond standard summaries. Favor transparent analysis and
clear visual evidence over complexity for its own sake.

## Project structure

- `data/` — local input samples and datasets; data files are excluded from Git.
- `src/pipeline.py` — streaming CSV preparation and 10 Hz displacement-speed
  derivation.
- `src/feature_engineering.py` — per-player deceleration, speed, and
  sharp-cut-efficiency summaries.
- `src/model_integration.py` — player-level combine/season left join,
  correlation and Ridge-baseline report.
- `notebooks/` — exported Kaggle notebooks and exploratory analysis.
- `outputs/` — local charts, reports, and model artifacts (excluded from Git).

## Quick start

Run the pipeline with a tracking CSV that has `game_id`, `play_id`, `player_id`,
`frame_id`, `x`, and `y` columns:

```bash
python src/pipeline.py data/tracking.csv outputs/tracking_metrics.csv
```

The script streams rows rather than loading the full dataset into memory. It
adds `dx`, `dy`, `displacement`, and `speed_10hz` columns. Speed is estimated
from consecutive observations for each game/play/player using the frame
difference and the 10 Hz sampling rate. The first observation for each track
has empty derived values. Input rows should be ordered chronologically within
each game/play/player track.

Place only data you are permitted to use in `data/`; large or competition-
restricted datasets should remain local and must not be committed.

## Combine/season integration

Call `integrate_combine_with_season_performance(player_summary_df,
regular_season_df)` from `src/model_integration.py` with a player identifier
(`nflId` preferred, or a player ID fallback) and the cut-efficiency feature.
The function preserves combine-summary players in a left join, imputes missing
cut-efficiency values with the median, and compares the feature with numeric
season outcomes such as yards after catch and defensive stops. It prints a
correlation and standardized Ridge-baseline report and writes aggregate-only
insights to `outputs/model_integration_insights.csv`; it does not export
player-level records.

Generated output files and model artifacts are intentionally kept local and
excluded from Git. The `outputs/` directory is preserved by a placeholder file;
publish selected final assets separately only when they are safe to share.

## Roadmap

1. Inspect the competition data, document its schema, and validate the
   streaming preparation pipeline.
2. Explore contextual movement, spacing, acceleration, and interaction
   features at the play and player levels.
3. Test whether candidate metrics add insight beyond established baselines.
4. Refine the strongest findings with targeted visualizations and reproducible
   analysis notebooks.
5. Deliver a concise report explaining the methods, evidence, and limitations.
