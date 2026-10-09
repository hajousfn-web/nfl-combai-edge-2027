# NFL CombAI Edge: Uncovering Non-Intuitive Tracking Metrics

> **Report status:** Structural draft. No competition data has been loaded and no
> empirical finding is claimed in this document.
>
> Replace every bracketed placeholder with verified dataset details or measured
> results before publishing the Kaggle report. Keep raw data local to its
> authorized environment; do not download full datasets just to populate this
> skeleton.
>
> **Ownership:** Original VisiShield-Edge(TM) project code and report materials
> are proprietary to Soufiane Hajou under the repository root LICENSE. Check
> competition and Kaggle terms before publishing or redistributing any
> submission material; those terms and third-party/data rights still apply.

## Contents

1. [Executive Summary & Sovereign Edge](#1-executive-summary--sovereign-edge)
2. [Data Ingestion & Rookie Cohort](#2-data-ingestion--rookie-cohort)
3. [Micro-Kinematic Feature Engineering](#3-micro-kinematic-feature-engineering)
4. [Statistical Integration & Baseline](#4-statistical-integration--baseline)
5. [Edge Risk Evaluation & FSM Telemetry](#5-edge-risk-evaluation--fsm-telemetry)
6. [Visualizations & Strategic Conclusion](#6-visualizations--strategic-conclusion)
7. [Reproducibility and limitations](#7-reproducibility-and-limitations)

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

- Bridge raw 10 Hz spatial kinematics and Cut Efficiency (CE) to two clearly
  separated outcomes: offline statistical evidence for EPA/YAC and a
  deployment-owned edge risk policy that feeds an industrial-style FSM
  telemetry contract.
- Keep the processing, feature engineering, statistical integration, and
  visualization modules separate.
- Do not fetch competition datasets from this notebook. Configure paths only
  to data already available in an authorized local or Kaggle input.
- Inspect schemas and small samples before deciding how to execute transforms.
- Keep generated reports, plots, and model artifacts in `outputs/`; do not
  commit raw, intermediate, or restricted data.
- Treat cohort definitions and conclusions as human-reviewed decisions.
- Keep offline tracking/model analysis separate from any deployment-owned
  real-time risk policy and its FSM telemetry.

The analytical layer reports nested Ridge cross-validation, forward
draft-class validation, and partial Spearman/permutation audit results. The
edge layer does not reuse those fitted scores as a risk controller: a
separately reviewed deployment policy classifies current observations, and the
FSM manages state transitions, faults, telemetry, and manual reset. Neither
the policy thresholds nor actuator behavior are defined here.

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

These 10 Hz measurements and CE are analytical inputs only. If a deployment
uses them for edge-risk evaluation, a separately reviewed policy must classify
the current sample before the finite-state telemetry contract receives it.
This report does not define operational hazard limits.

### Engineering decisions to document

- [ ] Confirm that `frame_id` is the correct 10 Hz time index.
- [ ] Check duplicate frames and missing/interrupted observations.
- [ ] Filter `entity_type == "PLAYER"` before calculating movement; document
  whether the source includes non-player/ball entities.
- [ ] Verify track boundaries (`game_id`, `play_id`, `player_id`) and coordinate
  direction conventions.
- [ ] Determine whether gaps in frames need a maximum permitted duration.
- [ ] Inspect the number of valid sharp cuts and zero-speed edge cases.
- [ ] State whether values are summarized per test, player, or another unit.

**Feature summary:** [Insert validated counts, distributions, and units.]

**Quality checks:** [Insert checks and outcomes.]

### Prelaunch acceleration channel

`statistical_audit.audit_prelaunch_motion()` requires a validated per-track
`launch_frame_id`. It reports prelaunch frame count, mean speed and
acceleration, and whether speed exceeded a documented tolerance before the
event marker. The helper cannot determine the event from coordinates alone.
Use an event-aligned complete track and predeclare the tolerance; records
without enough prelaunch samples are marked insufficient, not classified as
starting from rest. No pre-motion finding is asserted until this audit is run
on verified competition rows with a valid event marker.

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

### Partial Spearman and permutation audit

`src/statistical_audit.py` rank-transforms Cut Efficiency, target, and available
controls (average deceleration and max speed), residualizes each ranked signal
against the controls, and estimates partial Spearman rho from the residual
correlation. A two-sided residual permutation test reports a `+1`-corrected
p-value. Rows with missing/non-finite analysis values are excluded listwise.
This simple permutation design assumes independent, exchangeable rows; player,
game, or team clustering and repeated-measure dependence must be addressed
before inferential claims. Report permutation count, control set, eligible
sample size, and limitations.

### Forward validation across draft classes

`forward_validate_ridge_by_draft_class()` tunes alpha with inner K-fold CV on
earlier classes only, then evaluates the next available class (for example,
train 2023 and test 2024; train 2023–2024 and test 2025). The test class is not
used for imputation, scaling, or tuning. Folds below the configured minimum
training count are explicitly marked `insufficient_training_data`; do not
interpret a missing score as a successful validation. Aggregate outputs are
written to `outputs/partial_spearman_permutation_audit.csv` and
`outputs/forward_draft_class_validation.csv`.

**Observed relationships:** [Populate only after running on verified data.]

**Alternative explanations and confounders:** [Document.]

**Human review / decision:** [Record what the evidence does and does not support.]

## 5. Edge Risk Evaluation & FSM Telemetry

This is a separate edge-facing state/telemetry contract, not an extension of
the Ridge model. At each observation, a deployment-owned and safety-reviewed
policy may evaluate the current 10 Hz speed, acceleration/deceleration,
direction change, CE, sample freshness, and verified spatial proximity. Its
ordinary classifications are `SAFE`, `WARNING`, `DANGER`, and
`EMERGENCY_STOP`; the telemetry FSM additionally records `SENSOR_FAULT` and
`FAIL_SAFE_LOCKED`. This repository does not infer risk from the statistical
EPA/YAC model.

`src/zone_loader.py` validates explicit distance enter/clear pairs from a
caller-supplied JSON configuration. Hysteresis requires ordered thresholds to
avoid boundary chatter; no field threshold is supplied as an approved default.
The 1.1 m value shown in isolated mock fixtures is illustrative only.
`src/safety_kernel.py` combines one supplied LiDAR distance with kinematics and
emits a classification; `src/safety_engine.py` connects that reference kernel
to `src/execution_ring.py`, a fixed-capacity numeric FIFO. No hardware bindings,
physical LiDAR drivers, or certified industrial kernel are present.

`src/edge_safety_fsm.py` records bounded, ordered telemetry:

- Escalation is immediate; downgrade requires a configured count of
  consecutive lower-risk observations to avoid state flapping.
- Invalid input requests latched `SENSOR_FAULT`; an explicit internal
  fail-safe event may latch `FAIL_SAFE_LOCKED`.
- `EMERGENCY_STOP`, `SENSOR_FAULT`, and `FAIL_SAFE_LOCKED` remain latched even
  after later safe observations. They cannot clear automatically.
- Manual reset requires the configured consecutive safe observations, a
  non-empty operator identity, and explicit `safe_to_reset=True`; each accepted
  reset is recorded with the operator and timestamp.
- Timestamps must be timezone-aware and strictly increasing. Event history is
  bounded in memory; persistence and operational alarm delivery belong to the
  deployment integration and must be reviewed separately.

The reset records software acknowledgement only; it is not a physical
lockout/tagout procedure. The modules do not actuate hardware, implement a
safety PLC, provide a certified stale-data timer, or certify the policy. The
ring preallocates bounded numeric storage, but Python still allocates objects
and remains subject to interpreter/OS scheduling and garbage collection.
`benchmark_processing()` reports observed local timing distributions only;
there is no hard-real-time or sub-10 ms guarantee, and this code does not
disable or bypass the garbage collector. Treat these modules as offline
reference simulation until responsible safety engineers validate thresholds,
sensor-failure behavior, end-to-end timing, and independent interlocks for a
specific target deployment.

**Approved risk policy and thresholds:** [Document owner, version, validation
evidence, and approved operating context. Leave unset until reviewed.]

**FSM configuration and reset authority:** [Document recovery sample counts,
operator identity source, safe-to-reset evidence, and audit/persistence plan.]

## 6. Visualizations & Strategic Conclusion

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

`generate_audit_visualizations()` adds three high-resolution PNGs directly to
`outputs/`: partial Spearman estimates across pooled/class windows,
out-of-class Base-vs-Base+CE metric deltas, and an FSM transition-contract
illustration. The first two are drawn only from actual audit results; absent
files produce an explicit no-data graphic rather than fabricated values. The
FSM graphic is explicitly illustrative, not a measured risk sequence. Positive
RMSE/MAE reductions and R² gains favor Base+CE, but no generalization claim
should be made from a small or incomplete set of draft classes.

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
5. `outputs/partial_spearman_validation_windows.png` — evidence by window or
   explicit data-availability status.
6. `outputs/forward_draft_class_performance_deltas.png` — paired forward-test
   deltas; interpret only valid folds.
7. `outputs/edge_fsm_transition_sequence.png` — illustrative contract only;
   not an empirical NFL risk result.

**Scout/coach interpretation:** [Human-authored, tied to observed evidence.]

**Strategic recommendation:** [Human decision; include uncertainty and scope.]

Do not present an association as a causal mechanism or scouting rule without
appropriate validation and domain review.

## 7. Reproducibility and limitations

Run mock-data validation from the project root:

```bash
python src/pipeline.py --self-test
python src/feature_engineering.py --self-test
python src/model_integration.py --self-test
python src/edge_safety_fsm.py
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
- [ ] Join keys are non-null and the composite game/play/player/frame key is
  unique; non-player entities are filtered before kinematics.
- [ ] Outcome aggregation choices documented and reviewed.
- [ ] Partial Spearman controls, permutation assumptions, and forward test
  classes/sample sizes reported.
- [ ] Pre-motion claims are based on a validated launch marker, declared
  tolerance, and sufficient prelaunch frames; otherwise left unclaimed.
- [ ] Results generated from authorized data, with no mock values in the report.
- [ ] Plots include readable labels, sample sizes, and appropriate caveats.
- [ ] Conclusions are human-reviewed and distinguish association from
  prediction/causation.
- [ ] Edge-policy thresholds and invalid/stale data handling are approved by
  the deployment safety owner; the reference FSM is not represented as a
  certified controller.
- [ ] Notebook executes in the intended Kaggle environment with no network
  downloads or local-machine path assumptions.
- [ ] No raw or restricted data is embedded in notebook outputs or committed.
