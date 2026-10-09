"""Strict linkage, partial-rank inference, and draft-class validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable
import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PLAYER_ID_CANDIDATES = ("player_id", "nfl_id", "nflid", "playerid")
PLAYER_ID_NORMALIZED_NAMES = {"playerid", "nflid"}
PERFORMANCE_TARGETS = {
    "epa": ("epa", "expected_points_added"),
    "yards_after_catch": ("yards_after_catch", "yac"),
}
DEFAULT_CONTROLS = ("average_deceleration", "max_speed")
DEFAULT_PERMUTATIONS = 999
MAX_PERMUTATIONS = 9_999
MIN_FORWARD_TRAINING_ROWS = 6
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
RANDOM_STATE = 2027


def _normalize_name(value: object) -> str:
    return "".join(char for char in str(value).casefold() if char.isalnum())


def _find_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    columns = {_normalize_name(column): column for column in frame.columns}
    return next(
        (columns[_normalize_name(name)] for name in candidates
         if _normalize_name(name) in columns),
        None,
    )


def validate_linkage_keys(
    frame: pd.DataFrame,
    *,
    table_name: str,
    key_columns: Iterable[str],
    unique: bool = True,
) -> None:
    """Fail on absent, null/blank, or (when requested) duplicate linkage keys."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{table_name} must be a pandas DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError(f"{table_name} must not contain duplicate column names.")
    keys = tuple(key_columns)
    if not keys:
        raise ValueError("At least one linkage key is required.")
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{table_name} is missing linkage key columns: {', '.join(missing)}."
        )
    null_counts: dict[str, int] = {}
    for column in keys:
        values = frame[column]
        missing_values = values.isna()
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            missing_values |= values.astype("string").str.strip().eq("").fillna(False)
        if missing_values.any():
            null_counts[column] = int(missing_values.sum())
    if null_counts:
        details = ", ".join(
            f"{column}={count:,}" for column, count in null_counts.items()
        )
        raise ValueError(f"{table_name} has missing/blank linkage keys: {details}.")
    if unique:
        duplicate_count = int(frame.duplicated(list(keys), keep=False).sum())
        if duplicate_count:
            raise ValueError(
                f"{table_name} has {duplicate_count:,} rows with duplicate "
                f"linkage key(s): {', '.join(keys)}."
            )


def validate_player_identifier_consistency(
    frame: pd.DataFrame,
    *,
    preferred_column: str,
    table_name: str,
) -> None:
    """Reject conflicting player-ID aliases when a table includes more than one."""
    if preferred_column not in frame.columns:
        raise ValueError(
            f"{table_name} is missing preferred player ID {preferred_column!r}."
        )
    preferred = frame[preferred_column].astype("string").str.strip()
    for column in frame.columns:
        if (
            column == preferred_column
            or _normalize_name(column) not in PLAYER_ID_NORMALIZED_NAMES
        ):
            continue
        candidate = frame[column].astype("string").str.strip()
        if not preferred.eq(candidate).all():
            raise ValueError(
                f"{table_name} has conflicting player identifier columns "
                f"{preferred_column!r} and {column!r}."
            )


def prepare_tracking_players(tracking: pd.DataFrame) -> pd.DataFrame:
    """Filter PLAYER entities, normalize player IDs, and audit frame linkage.

    Existing ``player_id`` is accepted for legacy inputs; otherwise ``nfl_id``
    (or ``nflId``) is normalized to it. If both columns exist, they must agree.
    When ``entity_type`` exists, missing labels fail and only PLAYER rows remain.
    """
    if not isinstance(tracking, pd.DataFrame):
        raise TypeError("tracking must be a pandas DataFrame.")
    if not tracking.columns.is_unique:
        raise ValueError("Tracking data must not contain duplicate columns.")
    prepared = tracking.copy()
    entity_column = _find_column(prepared, ("entity_type",))
    if entity_column is not None:
        entity_values = prepared[entity_column].astype("string").str.strip().str.upper()
        if entity_values.isna().any() or entity_values.eq("").any():
            raise ValueError("entity_type must be present for every tracking row.")
        prepared = prepared.loc[entity_values.eq("PLAYER")].copy()
        if prepared.empty:
            raise ValueError("Tracking data contains no entity_type == 'PLAYER' rows.")

    player_column = _find_column(prepared, PLAYER_ID_CANDIDATES)
    if player_column is None:
        raise ValueError("Tracking data requires player_id or nfl_id.")
    if player_column != "player_id":
        prepared["player_id"] = prepared[player_column]
    validate_player_identifier_consistency(
        prepared,
        preferred_column="player_id",
        table_name="Tracking data",
    )

    linkage = ("game_id", "play_id", "player_id", "frame_id")
    validate_linkage_keys(
        prepared,
        table_name="Tracking data",
        key_columns=linkage,
        unique=True,
    )
    return prepared


def partial_spearman_permutation_test(
    frame: pd.DataFrame,
    *,
    predictor: str,
    outcome: str,
    controls: Iterable[str] = (),
    permutations: int = DEFAULT_PERMUTATIONS,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Estimate partial Spearman rho with a two-sided residual permutation test.

    Variables and controls are rank-transformed, then each ranked variable is
    residualized against the ranked controls and an intercept. The permutation
    test shuffles predictor residuals under an exchangeability assumption.
    Rows with non-finite predictor/outcome/control values are listwise excluded.
    The returned p-value uses the finite-sample +1 correction.
    """
    control_columns = list(dict.fromkeys(controls))
    required = [predictor, outcome, *control_columns]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Statistical audit is missing columns: {', '.join(missing)}.")
    if predictor == outcome or predictor in control_columns or outcome in control_columns:
        raise ValueError("Predictor, outcome, and control columns must be distinct.")
    if (
        isinstance(permutations, bool)
        or not isinstance(permutations, int)
        or permutations < 1
        or permutations > MAX_PERMUTATIONS
    ):
        raise ValueError(
            f"permutations must be an integer from 1 to {MAX_PERMUTATIONS:,}."
        )

    numeric_frame = frame.loc[:, required].copy()
    for column in required:
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
    values = numeric_frame.to_numpy(dtype=float)
    complete = np.isfinite(values).all(axis=1)
    numeric_frame = numeric_frame.loc[complete]
    sample_count = len(numeric_frame)
    minimum = len(control_columns) + 4
    base: dict[str, object] = {
        "predictor": predictor,
        "outcome": outcome,
        "controls": "|".join(control_columns),
        "sample_count": sample_count,
        "partial_spearman_rho": np.nan,
        "permutation_p_value": np.nan,
        "permutations": permutations,
        "status": "insufficient_data",
    }
    if sample_count < minimum:
        return base

    ranked = numeric_frame.rank(method="average").to_numpy(dtype=float)
    control_values = ranked[:, 2:]
    design = np.column_stack((np.ones(sample_count), control_values))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        base["status"] = "rank_deficient_controls"
        return base

    predictor_rank = ranked[:, 0]
    outcome_rank = ranked[:, 1]
    predictor_residual = predictor_rank - design @ np.linalg.lstsq(
        design, predictor_rank, rcond=None
    )[0]
    outcome_residual = outcome_rank - design @ np.linalg.lstsq(
        design, outcome_rank, rcond=None
    )[0]
    predictor_norm = float(np.linalg.norm(predictor_residual))
    outcome_norm = float(np.linalg.norm(outcome_residual))
    if predictor_norm == 0.0 or outcome_norm == 0.0:
        base["status"] = "constant_after_controls"
        return base

    observed = float(
        np.dot(predictor_residual, outcome_residual)
        / (predictor_norm * outcome_norm)
    )
    rng = np.random.default_rng(random_state)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(predictor_residual)
        statistic = float(
            np.dot(permuted, outcome_residual)
            / (np.linalg.norm(permuted) * outcome_norm)
        )
        exceedances += abs(statistic) >= abs(observed)

    base.update(
        partial_spearman_rho=observed,
        permutation_p_value=(exceedances + 1) / (permutations + 1),
        status="ok",
    )
    return base


def forward_validate_ridge_by_draft_class(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    target_column: str,
    baseline_feature_columns: Iterable[str] = (),
    player_id_column: str | None = None,
    draft_year_column: str = "draft_year",
    season_column: str = "season",
    min_training_rows: int = MIN_FORWARD_TRAINING_ROWS,
) -> pd.DataFrame:
    """Train on earlier draft classes and evaluate each later class in turn.

    Inner shuffled K-fold CV tunes Ridge alpha using training classes only.
    Test rows are the next available draft class, with rookie season required
    to equal draft year when a season column is present. No test-fold labels
    participate in imputation, scaling, or hyperparameter selection.
    """
    enhanced_features = list(dict.fromkeys(feature_columns))
    baseline_features = list(dict.fromkeys(baseline_feature_columns))
    feature_sets: list[tuple[str, list[str]]] = []
    if baseline_features:
        feature_sets.append(("base", baseline_features))
    if enhanced_features:
        feature_sets.append(("base_plus_metric", enhanced_features))
    features = list(dict.fromkeys([*baseline_features, *enhanced_features]))
    if not features:
        raise ValueError("At least one numeric feature column is required.")
    if (
        isinstance(min_training_rows, bool)
        or not isinstance(min_training_rows, int)
        or min_training_rows < 3
    ):
        raise ValueError("min_training_rows must be an integer of at least 3.")
    player_key = player_id_column or _find_column(frame, PLAYER_ID_CANDIDATES)
    if player_key is None or player_key not in frame.columns:
        raise ValueError("Forward validation requires a player identifier column.")
    required = [player_key, draft_year_column, target_column, *features]
    if season_column in frame.columns:
        required.append(season_column)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Forward validation is missing columns: {', '.join(missing)}.")
    if target_column in features:
        raise ValueError("The target column cannot also be a predictor.")
    if not feature_sets:
        feature_sets.append(("base_plus_metric", enhanced_features))

    data = frame.loc[:, list(dict.fromkeys(required))].copy()
    validate_linkage_keys(
        data,
        table_name="Forward-validation player-season data",
        key_columns=(player_key, draft_year_column),
        unique=True,
    )
    try:
        data[draft_year_column] = pd.to_numeric(
            data[draft_year_column], errors="raise"
        )
        data[target_column] = pd.to_numeric(data[target_column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Draft years and forward-validation targets must be numeric.") from error
    data[features] = data[features].replace([np.inf, -np.inf], np.nan)
    for feature in features:
        try:
            data[feature] = pd.to_numeric(data[feature], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Forward-validation feature {feature!r} must be numeric.") from error
    invalid_year = data[draft_year_column].notna() & (
        ~np.isfinite(data[draft_year_column].to_numpy(dtype=float))
        | data[draft_year_column].mod(1).ne(0)
    )
    if invalid_year.any():
        raise ValueError("draft_year must contain finite whole-number years.")
    data = data.loc[
        data[draft_year_column].notna()
        & data[target_column].notna()
        & data[draft_year_column].between(2023, 2025)
        & data[draft_year_column].mod(1).eq(0)
    ].copy()
    if season_column in data.columns:
        try:
            data[season_column] = pd.to_numeric(data[season_column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("season must contain numeric years.") from error
        if data[season_column].isna().any():
            raise ValueError("season must not be missing for forward validation.")
        data = data.loc[data[season_column].eq(data[draft_year_column])].copy()
    data = data.loc[np.isfinite(data[target_column].to_numpy(dtype=float))].copy()
    years = sorted(data[draft_year_column].astype(int).unique())
    output: list[dict[str, object]] = []
    for test_year in years[1:]:
        train = data.loc[data[draft_year_column].lt(test_year)]
        test = data.loc[data[draft_year_column].eq(test_year)]
        for model_name, model_features in feature_sets:
            row: dict[str, object] = {
                "target_metric": target_column,
                "test_draft_year": int(test_year),
                "training_draft_years": "|".join(
                    str(int(year)) for year in sorted(train[draft_year_column].unique())
                ),
                "model_variant": model_name,
                "feature_count": len(model_features),
                "training_player_count": len(train),
                "test_player_count": len(test),
                "inner_cv_folds": 0,
                "selected_alpha": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "status": "insufficient_training_data",
            }
            if len(train) < min_training_rows or test.empty:
                output.append(row)
                continue
            observed_features = [
                column for column in model_features if train[column].notna().any()
            ]
            if not observed_features:
                row["status"] = "no_observed_training_features"
                output.append(row)
                continue
            folds = min(5, len(train))
            estimator = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge()),
                ]
            )
            search = GridSearchCV(
                estimator=estimator,
                param_grid={"regressor__alpha": RIDGE_ALPHAS},
                scoring="neg_root_mean_squared_error",
                cv=KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE),
                n_jobs=1,
                error_score="raise",
            )
            search.fit(train[observed_features], train[target_column])
            prediction = search.predict(test[observed_features])
            actual = test[target_column].to_numpy(dtype=float)
            row.update(
                inner_cv_folds=folds,
                selected_alpha=float(search.best_params_["regressor__alpha"]),
                rmse=float(np.sqrt(mean_squared_error(actual, prediction))),
                mae=float(mean_absolute_error(actual, prediction)),
                r2=(
                    float(r2_score(actual, prediction))
                    if len(actual) >= 2
                    else np.nan
                ),
                status="ok",
            )
            output.append(row)
    return pd.DataFrame(output)


def audit_prelaunch_motion(
    tracking: pd.DataFrame,
    *,
    launch_frame_column: str = "launch_frame_id",
    speed_tolerance: float = 1e-6,
    minimum_prelaunch_frames: int = 2,
) -> pd.DataFrame:
    """Summarize whether measurable movement precedes a supplied launch marker.

    A launch event marker is required; this helper cannot infer the event from
    position data. ``speed_tolerance`` is in the coordinate-distance/second
    units established by the source coordinates.
    """
    if not isinstance(tracking, pd.DataFrame):
        raise TypeError("tracking must be a pandas DataFrame.")
    if launch_frame_column not in tracking.columns:
        raise ValueError(
            f"Prelaunch audit requires event marker {launch_frame_column!r}; "
            "movement before a launch cannot be inferred without it."
        )
    if (
        isinstance(speed_tolerance, bool)
        or not isinstance(speed_tolerance, (int, float))
        or not np.isfinite(speed_tolerance)
        or speed_tolerance < 0
    ):
        raise ValueError("speed_tolerance must be a finite non-negative number.")
    if (
        isinstance(minimum_prelaunch_frames, bool)
        or not isinstance(minimum_prelaunch_frames, int)
        or minimum_prelaunch_frames < 1
    ):
        raise ValueError("minimum_prelaunch_frames must be a positive integer.")

    if __package__:
        from .feature_engineering import calculate_velocity_and_acceleration
    else:
        from feature_engineering import calculate_velocity_and_acceleration

    prepared = prepare_tracking_players(tracking)
    validate_linkage_keys(
        prepared,
        table_name="Prelaunch tracking data",
        key_columns=("game_id", "play_id", "player_id", "frame_id"),
        unique=True,
    )
    markers = pd.to_numeric(prepared[launch_frame_column], errors="coerce")
    if markers.isna().any() or not np.isfinite(markers.to_numpy(dtype=float)).all():
        raise ValueError(f"{launch_frame_column} must contain finite frame indices.")
    if not markers.mod(1).eq(0).all():
        raise ValueError(f"{launch_frame_column} must contain whole-number frame indices.")
    prepared[launch_frame_column] = markers
    marker_counts = prepared.groupby(
        ["game_id", "play_id", "player_id"], observed=True
    )[launch_frame_column].nunique()
    if marker_counts.gt(1).any():
        raise ValueError(f"{launch_frame_column} must be constant within each track.")

    features = calculate_velocity_and_acceleration(prepared)
    track_keys = ["game_id", "play_id", "player_id"]
    marker_values = prepared.loc[:, [*track_keys, launch_frame_column]].drop_duplicates()
    features = features.merge(
        marker_values,
        how="left",
        on=track_keys,
        validate="many_to_one",
        sort=False,
    )
    rows: list[dict[str, object]] = []
    for key, track in features.groupby(track_keys, sort=False, observed=True):
        launch_frame = float(track[launch_frame_column].iloc[0])
        prelaunch = track.loc[track["frame_id"].lt(launch_frame)]
        speeds = prelaunch["speed"].dropna()
        enough_frames = len(prelaunch) >= minimum_prelaunch_frames
        rows.append(
            {
                **dict(zip(track_keys, key if isinstance(key, tuple) else (key,))),
                "launch_frame_id": launch_frame,
                "prelaunch_frame_count": len(prelaunch),
                "usable_speed_count": int(speeds.notna().sum()),
                "mean_prelaunch_speed": (
                    float(speeds.mean()) if not speeds.empty else np.nan
                ),
                "mean_prelaunch_acceleration": (
                    float(prelaunch["acceleration"].mean())
                    if prelaunch["acceleration"].notna().any()
                    else np.nan
                ),
                "pre_motion_detected": (
                    bool(speeds.gt(speed_tolerance).any()) if enough_frames else pd.NA
                ),
                "status": "ok" if enough_frames else "insufficient_prelaunch_frames",
            }
        )
    return pd.DataFrame(rows)


def _save_csv_atomic(frame: pd.DataFrame, filename: str) -> Path:
    """Atomically save aggregate audit results within project outputs/."""
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = Path(__file__).resolve().parent.parent.resolve()
    if output_dir.resolve() != resolved_root / "outputs":
        raise ValueError("The outputs directory must resolve inside the project.")
    path = output_dir.resolve() / filename
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=".statistical-audit-",
            dir=path.parent,
            delete=False,
        ) as output_file:
            temporary = Path(output_file.name)
            frame.to_csv(output_file, index=False, na_rep="")
        os.replace(temporary, path)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise OSError(f"Could not atomically write {filename}: {error}") from error
    return path


@dataclass(frozen=True)
class StatisticalAuditResult:
    """Aggregate inference and forward-validation outputs."""

    partial_correlations: pd.DataFrame
    forward_validation: pd.DataFrame


def run_statistical_audit(
    player_season: pd.DataFrame,
    *,
    permutations: int = DEFAULT_PERMUTATIONS,
    min_training_rows: int = MIN_FORWARD_TRAINING_ROWS,
    save_outputs: bool = True,
) -> StatisticalAuditResult:
    """Run partial CE/outcome inference and draft-class forward validation."""
    feature = _find_column(
        player_season,
        ("proprietary_cut_efficiency", "average_proprietary_cut_efficiency"),
    )
    if feature is None:
        raise ValueError("Player-season data requires a Cut Efficiency column.")
    player_column = _find_column(player_season, PLAYER_ID_CANDIDATES)
    if player_column is None:
        raise ValueError("Player-season data requires a player identifier.")
    validate_linkage_keys(
        player_season,
        table_name="Statistical-audit player-season data",
        key_columns=(player_column, "draft_year"),
        unique=True,
    )

    controls = [column for column in DEFAULT_CONTROLS if column in player_season]
    partial_rows: list[dict[str, object]] = []
    year_values = pd.to_numeric(player_season["draft_year"], errors="coerce")
    windows: list[tuple[str, pd.DataFrame]] = [
        ("pooled_2023_2025", player_season.loc[year_values.between(2023, 2025)])
    ]
    windows.extend(
        (
            f"draft_{year}",
            player_season.loc[year_values.eq(year)],
        )
        for year in (2023, 2024, 2025)
    )
    for metric, aliases in PERFORMANCE_TARGETS.items():
        target = _find_column(player_season, aliases)
        if target is None:
            continue
        for window_name, window_frame in windows:
            partial_rows.append(
                {
                    **partial_spearman_permutation_test(
                        window_frame,
                        predictor=feature,
                        outcome=target,
                        controls=controls,
                        permutations=permutations,
                    ),
                    "target_metric": metric,
                    "validation_window": window_name,
                }
            )

    baseline_features = [
        column
        for column in (
            "average_deceleration",
            "max_speed",
            "total_sharp_cuts",
        )
        if column in player_season
    ]
    model_features = list(dict.fromkeys([*baseline_features, feature]))
    if not baseline_features:
        baseline_features = [
            column for column in model_features if column != feature
        ]
    validation_rows: list[pd.DataFrame] = []
    for metric, aliases in PERFORMANCE_TARGETS.items():
        target = _find_column(player_season, aliases)
        if target is not None:
            validation_rows.append(
                forward_validate_ridge_by_draft_class(
                    player_season,
                    feature_columns=model_features,
                    baseline_feature_columns=baseline_features,
                    target_column=target,
                    player_id_column=player_column,
                    min_training_rows=min_training_rows,
                ).assign(target_metric=metric)
            )
    partial_frame = pd.DataFrame(partial_rows)
    validation_frame = (
        pd.concat(validation_rows, ignore_index=True)
        if validation_rows
        else pd.DataFrame()
    )
    if save_outputs:
        if not partial_frame.empty:
            _save_csv_atomic(partial_frame, "partial_spearman_permutation_audit.csv")
        if not validation_frame.empty:
            _save_csv_atomic(validation_frame, "forward_draft_class_validation.csv")
    return StatisticalAuditResult(partial_frame, validation_frame)


class _StatisticalAuditTest(unittest.TestCase):
    def test_linkage_rejects_missing_and_duplicate_keys(self) -> None:
        valid = pd.DataFrame(
            {"game_id": [1, 1], "play_id": [2, 2], "nfl_id": [10, 10], "frame_id": [1, 2]}
        )
        validate_linkage_keys(
            valid,
            table_name="test",
            key_columns=("game_id", "play_id", "nfl_id", "frame_id"),
        )
        with self.assertRaisesRegex(ValueError, "missing/blank"):
            validate_linkage_keys(
                valid.assign(nfl_id=[10, np.nan]),
                table_name="test",
                key_columns=("nfl_id",),
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_linkage_keys(
                valid.assign(frame_id=[1, 1]),
                table_name="test",
                key_columns=("game_id", "play_id", "nfl_id", "frame_id"),
            )

    def test_tracking_entity_filter_and_identifier_fallback(self) -> None:
        tracking = pd.DataFrame(
            {
                "game_id": [1, 1],
                "play_id": [2, 2],
                "nfl_id": [10, -1],
                "frame_id": [1, 1],
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "entity_type": ["PLAYER", "BALL"],
            }
        )
        prepared = prepare_tracking_players(tracking)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared.iloc[0]["player_id"], 10)

    def test_rejects_conflicting_player_id_aliases(self) -> None:
        tracking = pd.DataFrame(
            {
                "game_id": [1],
                "play_id": [2],
                "player_id": [10],
                "nfl_id": [11],
                "frame_id": [1],
                "x": [0.0],
                "y": [0.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "conflicting player identifier"):
            prepare_tracking_players(tracking)

    def test_partial_spearman_and_permutation_are_reproducible(self) -> None:
        values = np.linspace(0.0, 1.0, 30)
        frame = pd.DataFrame(
            {
                "ce": values + np.sin(values * 11) * 0.02,
                "outcome": values * 3 + np.cos(values * 13) * 0.04,
                "control": values * 2 + np.sin(values * 5) * 0.01,
            }
        )
        result = partial_spearman_permutation_test(
            frame,
            predictor="ce",
            outcome="outcome",
            controls=("control",),
            permutations=199,
            random_state=11,
        )
        repeat = partial_spearman_permutation_test(
            frame,
            predictor="ce",
            outcome="outcome",
            controls=("control",),
            permutations=199,
            random_state=11,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["partial_spearman_rho"], repeat["partial_spearman_rho"])
        self.assertEqual(result["permutation_p_value"], repeat["permutation_p_value"])

    def test_forward_validation_holds_out_later_draft_classes(self) -> None:
        rows = []
        for year in (2023, 2024, 2025):
            for player in range(6):
                feature = player + (year - 2023) * 0.2
                rows.append(
                    {
                        "player_id": f"{year}-{player}",
                        "draft_year": year,
                        "season": year,
                        "feature": feature,
                        "outcome": feature * 2 + player * 0.03,
                    }
                )
        result = forward_validate_ridge_by_draft_class(
            pd.DataFrame(rows),
            feature_columns=("feature",),
            target_column="outcome",
        )
        self.assertEqual(result["test_draft_year"].tolist(), [2024, 2025])
        self.assertTrue(result["training_draft_years"].iloc[0] == "2023")
        self.assertTrue(result["training_draft_years"].iloc[1] == "2023|2024")
        self.assertTrue(result["status"].eq("ok").all())

    def test_prelaunch_motion_requires_explicit_event_marker(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires event marker"):
            audit_prelaunch_motion(pd.DataFrame())

    def test_prelaunch_motion_uses_event_marker_and_reports_observed_motion(self) -> None:
        tracking = pd.DataFrame(
            {
                "game_id": [1] * 5,
                "play_id": [2] * 5,
                "player_id": [3] * 5,
                "frame_id": [1, 2, 3, 4, 5],
                "x": [0.0, 0.2, 0.5, 0.9, 1.4],
                "y": [0.0] * 5,
                "launch_frame_id": [5] * 5,
            }
        )
        result = audit_prelaunch_motion(tracking)
        self.assertTrue(result.loc[0, "pre_motion_detected"])
        self.assertEqual(result.loc[0, "prelaunch_frame_count"], 4)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_StatisticalAuditTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
