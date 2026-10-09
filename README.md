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

The core innovation is a dual-layer design: offline, bounded 10 Hz Combine
kinematics convert spatial movement and Cut Efficiency (CE) into auditable
features for rookie-season EPA/YAC analysis, while a separate industrial-style
edge contract evaluates a deployment-owned risk classification through an
offline-first FSM telemetry engine. Statistical validation never sets edge
risk thresholds or controls equipment. Human reviewers choose cohort
definitions and interpret results; correlation and Ridge scores are not
evidence of causality or safety certification.

## Project structure

- `data/` — local input samples and datasets; data files are excluded from Git.
- `src/pipeline.py` — bounded end-to-end orchestration plus a legacy streaming
  CSV preparation utility.
- `src/feature_engineering.py` — smoothed 10 Hz velocity/acceleration and
  sharp-cut speed-retention features; preserves validated `draft_year` when
  present.
- `src/model_integration.py` — player-level combine/season integration,
  descriptive associations, and nested-cross-validated Ridge baselines.
- `src/statistical_audit.py` — strict linkage checks, partial Spearman
  permutation inference, event-anchored prelaunch motion summaries, and
  forward draft-class Ridge validation.
- `src/edge_safety_fsm.py` — bounded in-memory risk-state telemetry with
  immediate escalation, stable recovery, sensor-fault/fail-safe latches, and
  manually authorized software reset acknowledgement.
- `src/zone_loader.py`, `src/safety_kernel.py`, and `src/safety_engine.py` —
  strict loading of caller-owned distance thresholds, proximity hysteresis,
  LiDAR/kinematics sample evaluation, and offline FSM orchestration.
- `src/execution_ring.py` — fixed-capacity preallocated numeric FIFO for
  bounded sample transfer; it does not remove Python allocations or scheduling.
- `src/visualization.py` — bounded-memory, headless PNG plots for kinematics,
  prediction comparisons, validation-window partial Spearman estimates,
  forward-class Base-vs-Base+CE deltas, and an explicitly illustrative FSM
  transition sequence.
- `notebooks/` — [Kaggle report skeleton](notebooks/kaggle_report_skeleton.md),
  the [submission notebook](notebooks/visishield_edge_submission.ipynb), and
  exported notebooks.
- `outputs/` — local charts, reports, and model artifacts (excluded from Git).

## Quick start

Run the complete workflow on a bounded tracking CSV with `game_id`, `play_id`,
`player_id` (or `nfl_id`), `frame_id`, `x`, `y`, and `draft_year` columns.
When present, `entity_type` must be populated and only `PLAYER` rows are used:

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
the active player track; non-PLAYER entities are excluded when `entity_type`
is present. Input must be ordered by `game_id`, `play_id`, player ID,
then `frame_id`. It adds `dx`, `dy`, `displacement`, and `speed_10hz`; speed
accounts for frame gaps. The first observation for each track has empty
derived values. CSV outputs are atomically written only directly inside
`outputs/`.

Run each module's mock-data checks without loading competition data:

```bash
python src/pipeline.py --self-test
python src/feature_engineering.py --self-test
python src/model_integration.py --self-test
python src/statistical_audit.py
python src/edge_safety_fsm.py
python src/visualization.py --self-test
```

Place only data you are permitted to use in `data/`; large or competition-
restricted datasets should remain local and must not be committed.

## Offline edge-risk and FSM telemetry contract

The architecture has two distinct analytical layers and one safety-state
contract:

1. **Offline tracking analytics:** bounded ingestion feeds 10 Hz spatial
   coordinates into `feature_engineering.py`, which derives velocity,
   acceleration, direction changes, and Cut Efficiency (CE). The pipeline
   rejects oversized inputs rather than truncating tracks.
2. **Statistical performance analysis:** the player-season modeling workflow
   uses nested Ridge cross-validation to assess EPA/YAC associations and
   generalization. This is an offline research result, not a real-time safety
   decision or a safety certification.
3. **Edge risk-state telemetry:** a separately validated, deployment-owned
   policy may map current kinematics, CE, data quality, and operational context
   to `SAFE`, `WARNING`, `DANGER`, or `EMERGENCY_STOP`. The
   `EdgeSafetyFSM` consumes that classification and records ordered telemetry;
   it does not invent risk thresholds, calculate a hazard score, or actuate
   equipment.

The FSM escalates immediately, requires a configurable number of consecutive
lower-risk samples before recovery, and latches `EMERGENCY_STOP`. An invalid or
stale observation must be reported by the caller as invalid (or with a fault
reason), which requests the latched stop. Clearing the lockout requires both
the configured consecutive `SAFE` observations and a manual reset with a
non-empty operator identity plus an explicit `safe_to_reset=True` confirmation.
The reset is recorded in a bounded in-memory event history; it is not a
physical lockout/tagout action. Timestamps must be timezone-aware and strictly
increasing.

No hazard thresholds are supplied by this repository: calibrate and validate
them for the actual deployment and have the responsible safety engineer approve
the policy. This Python reference contract is telemetry/state logic only; it
is not a certified real-time controller, safety PLC, or substitute for
independent interlocks. The current NFL dataset workflow is offline and does
not connect the competition analysis to live equipment.

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

## Linkage, inference, and forward validation

The in-memory pipeline rejects missing or blank `game_id`, `play_id`, player
ID, or `frame_id` linkage values and repeated composite frame keys. It accepts
`player_id` or `nfl_id`/`nflId`; if `entity_type` is supplied, each label must
be present and only `PLAYER` rows enter kinematic calculations. Season
integration fails on missing player/season linkage keys rather than silently
dropping them. The predictive path requires one row per player-season.

When outcomes are supplied, `run_full_pipeline()` additionally runs
`statistical_audit.py`:

- Partial Spearman analysis rank-transforms CE, EPA/YAC, and available
  controls (average deceleration and max speed), residualizes against ranked
  controls, then runs a two-sided residual permutation test. Rows with
  non-finite values are excluded listwise; p-values use a finite-permutation
  `+1` correction. The test assumes exchangeable independent rows. Repeated or
  clustered observations need a cluster-aware inference design before
  inferential claims.
- Forward validation trains only on earlier draft classes, tunes Ridge alpha
  with inner K-fold CV on those training rows, and evaluates the next class.
  Insufficient training data is explicitly reported as
  `insufficient_training_data`. Aggregate outputs are written to
  `outputs/partial_spearman_permutation_audit.csv` and
  `outputs/forward_draft_class_validation.csv`.
- `audit_prelaunch_motion()` requires a per-track `launch_frame_id` event
  marker and reports whether measurable speed existed before it. Without
  validated event timing, we cannot conclude whether a test began from motion
  or rest. The channel is descriptive and does not establish intent or cause.

These analyses are exploratory on a small cohort. Report sample counts,
uncertainty, design assumptions, and limitations. They are not a substitute
for independent evaluation or the safety-approved edge risk policy consumed
by the FSM.

## Offline analysis and edge telemetry are separate

Offline evaluation consists of nested Ridge CV, forward draft-class tests,
and partial Spearman/permutation summaries on verified player-season labels.
The generated audit figures in `outputs/` use actual audit tables when
available; otherwise they explicitly display that empirical data is
unavailable. The FSM sequence figure is always a contract illustration, never
a measured risk result. These visualizations do not infer the risk state.

At deployment, a separately approved policy may consume fresh 10 Hz speed,
acceleration, direction-change/CE, and data-validity signals to emit
`SAFE`, `WARNING`, `DANGER`, or `EMERGENCY_STOP`. The reference FSM also
represents `SENSOR_FAULT` and `FAIL_SAFE_LOCKED`, each latched until configured
safe observations and an explicitly recorded software reset acknowledgement.
The offline `SafetyKernel` consumes caller-supplied distance readings and
kinematics; no physical LiDAR/device bindings are included. `zone_loader.py`
validates explicit enter/clear thresholds, including hysteresis; values such as
1.1 m appear only in mock tests/notebook examples and are not approved operating
limits.

The execution ring preallocates bounded numeric storage and refuses overflow.
The Python path still creates objects, runs under interpreter scheduling, and
may be interrupted by the runtime/OS. `safety_engine.py --self-test` checks
behavior; timing observations from `benchmark_processing()` are local empirical
measurements, not a deterministic <10 ms guarantee. No code disables or
bypasses Python's garbage collector. Threshold selection, sensor qualification,
stale-data detection, persistence, alarms, independent interlocks, physical
lockout/tagout, and certification remain deployment responsibilities. This
repository is an offline reference simulator, not a safety controller.

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

`generate_audit_visualizations()` writes `partial_spearman_validation_windows.png`,
`forward_draft_class_performance_deltas.png`, and
`edge_fsm_transition_sequence.png` directly under `outputs/`. The audit figures
use pipeline audit CSVs when present and otherwise show an explicit
data-unavailable message; the FSM sequence is always marked illustrative.
Positive RMSE/MAE improvement and R² improvement are oriented to favor
Base+CE. None of these offline charts establishes causality or substitutes
for deployment review.

## Roadmap

1. Inspect the competition data, document its schema, and validate the
   streaming preparation pipeline.
2. Explore contextual movement, spacing, acceleration, and interaction
   features at the play and player levels.
3. Test whether candidate metrics add insight beyond established baselines.
4. Refine the strongest findings with targeted visualizations and reproducible
   analysis notebooks.
5. Deliver a concise report explaining the methods, evidence, and limitations.
