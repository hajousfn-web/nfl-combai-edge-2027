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
- `notebooks/` — exported Kaggle notebooks and exploratory analysis.
- `outputs/` — charts and final report assets.

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

## Roadmap

1. Inspect the competition data, document its schema, and validate the
   streaming preparation pipeline.
2. Explore contextual movement, spacing, acceleration, and interaction
   features at the play and player levels.
3. Test whether candidate metrics add insight beyond established baselines.
4. Refine the strongest findings with targeted visualizations and reproducible
   analysis notebooks.
5. Deliver a concise report explaining the methods, evidence, and limitations.
