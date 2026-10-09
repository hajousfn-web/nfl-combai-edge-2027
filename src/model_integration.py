"""Join combine tracking features to regular-season metrics and assess signal."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from uuid import uuid4
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from . import statistical_audit
else:
    import statistical_audit


PLAYER_KEY_CANDIDATES = (
    "nflid",
    "playerid",
    "playeridentifier",
    "player",
)
CUT_EFFICIENCY_CANDIDATES = (
    "proprietary_cut_efficiency",
    "average_proprietary_cut_efficiency",
)
RIDGE_ALPHA = 1.0
OUTPUT_FILENAME = "model_integration_insights.csv"
MODEL_PIPELINE_OUTPUT_FILENAME = "model_pipeline_metrics.csv"
ROOKIE_DRAFT_YEARS = (2023, 2024, 2025)
RIDGE_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
MIN_CV_SAMPLES = 6
RANDOM_STATE = 2027


def _normalized_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    by_normalized_name = {_normalized_name(column): column for column in frame.columns}
    for candidate in candidates:
        column = by_normalized_name.get(_normalized_name(candidate))
        if column is not None:
            return column
    return None


def _is_target_metric(column: Any) -> bool:
    normalized = _normalized_name(column)
    return (
        normalized == "yac"
        or "yardsaftercatch" in normalized
        or "defensivestop" in normalized
        or normalized == "stops"
        or normalized == "epa"
        or "expectedpointsadded" in normalized
    )


def _is_context_column(column: Any) -> bool:
    normalized = _normalized_name(column)
    return (
        normalized
        in {
            "season",
            "draftyear",
            "rookieyear",
            "week",
            "gameid",
            "playid",
            "frame",
            "frameid",
            "x",
            "y",
            "player",
            "playeridentifier",
        }
        or normalized.endswith("playerid")
        or normalized.endswith("nflid")
    )


def _target_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Find the supported outcome columns by normalized schema name."""
    targets: dict[str, str] = {}
    for column in frame.columns:
        normalized = _normalized_name(column)
        if normalized == "epa" or "expectedpointsadded" in normalized:
            targets.setdefault("epa", column)
        elif (
            normalized in {"yac", "yardsaftercatch"}
            or "yardsaftercatch" in normalized
        ):
            targets.setdefault("yards_after_catch", column)
    return targets


def _model_pipeline() -> Pipeline:
    """Create an impute-scale-Ridge pipeline fitted independently per fold."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge()),
        ]
    )


def _tune_ridge(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    folds: int,
) -> GridSearchCV:
    """Tune Ridge regularization with shuffled, reproducible K-fold CV."""
    cross_validator = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        estimator=_model_pipeline(),
        param_grid={"regressor__alpha": RIDGE_ALPHA_GRID},
        scoring="neg_root_mean_squared_error",
        cv=cross_validator,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(features, target)
    return search


def _save_model_metrics(metrics: pd.DataFrame) -> Path:
    """Atomically save aggregate evaluation metrics inside outputs/ only."""
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = project_root.resolve()
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir != resolved_root / "outputs":
        raise ValueError("The outputs directory must not resolve outside the project.")

    output_path = resolved_output_dir / MODEL_PIPELINE_OUTPUT_FILENAME
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=".model-metrics-",
            dir=resolved_output_dir,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            metrics.to_csv(output_file, index=False, na_rep="")
        os.replace(temp_path, output_path)
    except OSError as error:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise OSError(f"Could not securely write model metrics: {error}") from error
    return output_path


def _safe_player_key(values: pd.Series) -> pd.Series:
    """Normalize mixed numeric/string IDs so equivalent identifiers can join."""
    key = values.astype("string").str.strip()
    return key.mask(key.eq(""))


def _aggregate_season_rows(
    season: pd.DataFrame,
    key_column: str,
    outcome_columns: list[str],
) -> pd.DataFrame:
    """Reduce repeated game rows to one season-level row per player."""
    duplicate_mask = season[key_column].duplicated(keep=False)
    if not duplicate_mask.any():
        return season

    aggregations: dict[str, str] = {}
    for column in season.columns:
        if column == key_column:
            continue
        if column in outcome_columns and pd.api.types.is_numeric_dtype(season[column]):
            aggregations[column] = "sum"
        elif pd.api.types.is_numeric_dtype(season[column]):
            aggregations[column] = "mean"
        else:
            aggregations[column] = "first"
    return season.groupby(key_column, as_index=False, sort=False).agg(aggregations)


def _save_insights(insights: pd.DataFrame) -> Path:
    """Atomically write aggregate-only insights inside the project outputs dir."""
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_root = project_root.resolve()
    resolved_output_dir = output_dir.resolve()
    if resolved_output_dir != resolved_root / "outputs":
        raise ValueError("The outputs directory must not resolve outside the project.")

    output_path = resolved_output_dir / OUTPUT_FILENAME
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=".model-integration-",
            dir=resolved_output_dir,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            insights.to_csv(output_file, index=False, na_rep="")
        os.replace(temp_path, output_path)
    except OSError as error:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise OSError(f"Could not securely write integration insights: {error}") from error
    return output_path


def integrate_combine_with_season_performance(
    player_summary_df: pd.DataFrame,
    regular_season_df: pd.DataFrame,
    games_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-join player tracking summaries to season metrics and assess signal.

    The input summary must contain ``draft_year`` and is restricted to the
    2023–2025 rookie cohorts. Player IDs are matched using ``nflId`` when
    available, otherwise a ``playerId``, ``player_identifier``, or ``player``
    column. Performance rows must contain ``season``; for player_play-style
    rows without it, provide ``games_df`` with ``game_id`` and ``season`` so
    game seasons can be attached before filtering. Only each player's
    draft-year (rookie) season is retained. Repeated game rows are collapsed
    per player after this filter, preventing many-to-many expansion.

    Missing cut-efficiency values are imputed with the observed median. Missing
    outcomes are excluded from that metric's pairwise analysis. The function
    computes Pearson correlations and a standardized univariate Ridge baseline
    (alpha=1; in-sample R²) for numeric season outcomes, including YAC, EPA,
    and defensive stops. Aggregate-only insights are written atomically to
    ``outputs/model_integration_insights.csv``; player-level records are not
    written to disk.

    Args:
        player_summary_df: One-row-per-player combine/tracking summary.
        regular_season_df: Player-level or repeated game-level performance.
        games_df: Optional games table used to attach ``season`` to performance
            rows by ``game_id`` when that column is absent.

    Returns:
        The merged player-level DataFrame, retaining combine-summary rows with
        unmatched season metrics represented as missing values.

    Raises:
        TypeError: If either argument is not a pandas DataFrame.
        ValueError: If identifiers, cohort/season fields, or cut-efficiency
            features are missing/ambiguous, or combine summaries are not unique.
        OSError: If the insights CSV cannot be written.
    """
    if not isinstance(player_summary_df, pd.DataFrame):
        raise TypeError("player_summary_df must be a pandas DataFrame.")
    if not isinstance(regular_season_df, pd.DataFrame):
        raise TypeError("regular_season_df must be a pandas DataFrame.")
    if games_df is not None and not isinstance(games_df, pd.DataFrame):
        raise TypeError("games_df must be a pandas DataFrame when provided.")
    if not player_summary_df.columns.is_unique or not regular_season_df.columns.is_unique:
        raise ValueError("Input DataFrames must not contain duplicate column names.")
    if games_df is not None and not games_df.columns.is_unique:
        raise ValueError("games_df must not contain duplicate column names.")

    summary_key = _find_column(player_summary_df, PLAYER_KEY_CANDIDATES)
    season_key = _find_column(regular_season_df, PLAYER_KEY_CANDIDATES)
    if summary_key is None or season_key is None:
        raise ValueError(
            "Both inputs require a player identifier (nflId preferred, then "
            "playerId, player_identifier, or player)."
        )
    statistical_audit.validate_player_identifier_consistency(
        player_summary_df,
        preferred_column=summary_key,
        table_name="Combine player summaries",
    )
    statistical_audit.validate_player_identifier_consistency(
        regular_season_df,
        preferred_column=season_key,
        table_name="Regular-season performance data",
    )
    draft_year_column = _find_column(player_summary_df, ("draft_year",))
    if draft_year_column is None:
        raise ValueError(
            "player_summary_df must contain draft_year to select the 2023–2025 rookie cohorts."
        )

    feature_column = _find_column(player_summary_df, CUT_EFFICIENCY_CANDIDATES)
    if feature_column is None:
        raise ValueError(
            "player_summary_df must contain proprietary_cut_efficiency or "
            "average_proprietary_cut_efficiency."
        )
    available_features = [
        _find_column(player_summary_df, (candidate,))
        for candidate in CUT_EFFICIENCY_CANDIDATES
    ]
    available_features = list(dict.fromkeys(column for column in available_features if column))
    if len(available_features) > 1:
        raise ValueError(
            "player_summary_df contains multiple cut-efficiency feature columns."
        )

    # Restrict analysis to numeric performance values, never IDs, season labels,
    # or other context fields.
    numeric_season_columns = [
        column
        for column in regular_season_df.select_dtypes(include="number").columns
        if column != season_key and not _is_context_column(column)
    ]
    known_outcomes = [
        column for column in numeric_season_columns if _is_target_metric(column)
    ]
    outcome_columns = known_outcomes or numeric_season_columns
    if not outcome_columns:
        raise ValueError(
            "regular_season_df must contain numeric performance outcomes such "
            "as yards after catch or defensive stops."
        )

    summary_columns = list(dict.fromkeys([*player_summary_df.columns, draft_year_column]))
    summary = player_summary_df.loc[:, summary_columns].copy()
    statistical_audit.validate_linkage_keys(
        summary,
        table_name="Combine player summaries",
        key_columns=(summary_key,),
        unique=True,
    )
    summary[draft_year_column] = pd.to_numeric(
        summary[draft_year_column], errors="coerce"
    )
    summary = summary.loc[
        summary[draft_year_column].isin(ROOKIE_DRAFT_YEARS)
    ].copy()
    if feature_column != "proprietary_cut_efficiency":
        summary.rename(
            columns={feature_column: "proprietary_cut_efficiency"},
            inplace=True,
        )
    summary["proprietary_cut_efficiency"] = pd.to_numeric(
        summary["proprietary_cut_efficiency"], errors="coerce"
    )
    summary["_integration_player_key"] = _safe_player_key(summary[summary_key])
    summary = summary.loc[summary["_integration_player_key"].notna()].copy()
    if summary["_integration_player_key"].duplicated().any():
        raise ValueError(
            "player_summary_df must have one row per player identifier to avoid "
            "a many-to-many merge."
        )

    season_column = _find_column(regular_season_df, ("season",))
    game_id_column = _find_column(regular_season_df, ("game_id",))
    season_columns = list(dict.fromkeys([season_key, *outcome_columns]))
    if season_column is not None:
        season_columns.append(season_column)
    elif games_df is not None:
        games_game_id = _find_column(games_df, ("game_id",))
        games_season = _find_column(games_df, ("season",))
        if game_id_column is None or games_game_id is None or games_season is None:
            raise ValueError(
                "To attach seasons, regular_season_df and games_df must both "
                "contain game_id, and games_df must contain season."
            )
        season_columns.append(game_id_column)
    else:
        raise ValueError(
            "regular_season_df must contain season, or pass games_df with "
            "game_id and season columns."
        )

    # Keep only join keys, cohort fields, and outcomes before joining so
    # unrelated tracking/game columns cannot multiply memory use.
    season = regular_season_df.loc[:, list(dict.fromkeys(season_columns))].copy()
    season["_integration_player_key"] = _safe_player_key(season[season_key])
    season = season.loc[season["_integration_player_key"].notna()].copy()

    statistical_audit.validate_linkage_keys(
        season,
        table_name="Regular-season performance data",
        key_columns=tuple(
            column
            for column in (season_key, season_column, game_id_column if season_column is None else None)
            if column is not None
        ),
        unique=False,
    )
    if season_column is None:
        if games_df is None or game_id_column is None:
            raise ValueError(
                "games_df and game_id are required to attach game-season values."
            )
        games_game_id = _find_column(games_df, ("game_id",))
        games_season = _find_column(games_df, ("season",))
        if games_game_id is None or games_season is None:
            raise ValueError("games_df must contain game_id and season columns.")
        games = games_df.loc[:, [games_game_id, games_season]].copy()
        statistical_audit.validate_linkage_keys(
            games,
            table_name="Games data",
            key_columns=(games_game_id, games_season),
            unique=False,
        )
        statistical_audit.validate_linkage_keys(
            games,
            table_name="Games data",
            key_columns=(games_game_id,),
            unique=True,
        )
        games["_integration_game_key"] = _safe_player_key(games[games_game_id])
        games[games_season] = pd.to_numeric(games[games_season], errors="coerce")
        games = games.loc[
            games["_integration_game_key"].notna()
            & games[games_season].notna(),
            ["_integration_game_key", games_season],
        ]
        if games["_integration_game_key"].duplicated().any():
            raise ValueError("games_df must contain one season row per game_id.")
        season["_integration_game_key"] = _safe_player_key(season[game_id_column])
        season = season.merge(
            games,
            how="inner",
            on="_integration_game_key",
            validate="many_to_one",
            sort=False,
        )
        season.drop(columns="_integration_game_key", inplace=True)
        season_column = games_season

    season[season_column] = pd.to_numeric(season[season_column], errors="coerce")
    cohort = summary.loc[
        :, ["_integration_player_key", draft_year_column]
    ].rename(columns={draft_year_column: "_integration_draft_year"})
    season = season.merge(
        cohort,
        how="inner",
        on="_integration_player_key",
        validate="many_to_one",
        sort=False,
    )
    season = season.loc[
        season[season_column].eq(season["_integration_draft_year"])
    ].drop(columns="_integration_draft_year")
    season = season.loc[
        :, ["_integration_player_key", *outcome_columns]
    ]
    season = _aggregate_season_rows(
        season, "_integration_player_key", outcome_columns
    )

    merged = summary.merge(
        season,
        on="_integration_player_key",
        how="left",
        suffixes=("_combine", "_season"),
        sort=False,
        validate="one_to_one",
    )
    merged.drop(columns="_integration_player_key", inplace=True)
    if "proprietary_cut_efficiency" not in merged.columns:
        combine_feature = "proprietary_cut_efficiency_combine"
        if combine_feature not in merged.columns:
            raise ValueError("Could not identify the merged combine cut-efficiency feature.")
        merged.rename(
            columns={combine_feature: "proprietary_cut_efficiency"},
            inplace=True,
        )

    # Find the merged season outcome name, accounting for overlapping columns.
    merged_outcome_names: dict[str, str] = {}
    for outcome in outcome_columns:
        merged_name = (
            f"{outcome}_season"
            if outcome in summary.columns and outcome != feature_column
            else outcome
        )
        if merged_name not in merged.columns:
            # Feature-column normalization can affect collision suffixes.
            candidates = [
                column
                for column in merged.columns
                if _normalized_name(column).startswith(_normalized_name(outcome))
                and _normalized_name(column).endswith("season")
            ]
            if len(candidates) == 1:
                merged_name = candidates[0]
            else:
                raise ValueError(
                    f"Could not identify merged season metric column for {outcome!r}."
                )
        merged_outcome_names[outcome] = merged_name

    feature = merged["proprietary_cut_efficiency"]
    observed_feature = feature.replace([np.inf, -np.inf], np.nan)
    feature_median = observed_feature.median()
    imputed_count = int(observed_feature.isna().sum())
    if pd.notna(feature_median):
        merged["proprietary_cut_efficiency"] = observed_feature.fillna(feature_median)
    else:
        merged["proprietary_cut_efficiency"] = observed_feature

    # A one-feature correlation matrix is explicit and small regardless of the
    # number of input players or season columns.
    insight_rows: list[dict[str, Any]] = []
    for original_outcome, merged_outcome in merged_outcome_names.items():
        target = pd.to_numeric(merged[merged_outcome], errors="coerce")
        x = merged["proprietary_cut_efficiency"].replace(
            [np.inf, -np.inf], np.nan
        )
        y = target.replace([np.inf, -np.inf], np.nan)
        paired = pd.DataFrame({"feature": x, "outcome": y}).dropna()
        n_observations = len(paired)
        correlation = float("nan")
        ridge_coefficient = float("nan")
        ridge_r2 = float("nan")

        if n_observations >= 2:
            correlation_matrix = paired.corr(method="pearson", min_periods=2)
            correlation = float(
                correlation_matrix.loc["feature", "outcome"]
            )
            feature_std = float(paired["feature"].std(ddof=0))
            outcome_std = float(paired["outcome"].std(ddof=0))
            if feature_std > 0 and outcome_std > 0:
                standardized_x = (
                    paired["feature"].to_numpy(dtype=float)
                    - float(paired["feature"].mean())
                ) / feature_std
                standardized_y = (
                    paired["outcome"].to_numpy(dtype=float)
                    - float(paired["outcome"].mean())
                ) / outcome_std
                ridge_coefficient = float(
                    np.dot(standardized_x, standardized_y)
                    / (np.dot(standardized_x, standardized_x) + RIDGE_ALPHA)
                )
                predictions = standardized_x * ridge_coefficient
                total_variance = float(np.square(standardized_y).sum())
                if total_variance > 0:
                    ridge_r2 = float(
                        1.0
                        - np.square(standardized_y - predictions).sum()
                        / total_variance
                    )

        insight_rows.append(
            {
                "outcome_metric": str(original_outcome),
                "paired_player_count": n_observations,
                "imputed_cut_efficiency_count": imputed_count,
                "pearson_correlation": correlation,
                "ridge_standardized_coefficient": ridge_coefficient,
                "ridge_in_sample_r2": ridge_r2,
            }
        )

    insights = pd.DataFrame(
        insight_rows,
        columns=(
            "outcome_metric",
            "paired_player_count",
            "imputed_cut_efficiency_count",
            "pearson_correlation",
            "ridge_standardized_coefficient",
            "ridge_in_sample_r2",
        ),
    )
    output_path = _save_insights(insights)

    print("Combine tracking vs. regular-season performance")
    print(
        f"Cohort: draft_year in {ROOKIE_DRAFT_YEARS}; rookie season equals draft_year"
    )
    print(f"Merge: left join on {summary_key} = {season_key}")
    print(f"Players retained: {len(merged):,}")
    print(f"Cut-efficiency values imputed: {imputed_count:,}")
    print(f"Ridge baseline: standardized single feature, alpha={RIDGE_ALPHA:g}")
    if insights.empty:
        print("No numeric performance metrics were available for analysis.")
    else:
        print(insights.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Aggregate insights saved to: {output_path}")
    return merged


def run_model_pipeline(features_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-validate Ridge baselines for rookie-season EPA and YAC.

    ``features_df`` must contain one aggregated player-season row per player,
    with player ID, ``draft_year``, ``season``, numeric movement features, and
    EPA/YAC outcome columns. Outcomes must already be aggregated to this same
    unit; this function deliberately does not guess how play-level labels
    should be combined. Only 2023–2025 rookie-season rows are modeled.

    Nested cross-validation tunes Ridge regularization inside each training
    fold and reports out-of-fold RMSE, MAE, and R². No player records or model
    artifacts are persisted; only the aggregate metrics are written to
    ``outputs/model_pipeline_metrics.csv``.
    """
    if not isinstance(features_df, pd.DataFrame) or features_df.empty:
        raise ValueError("features_df must be a non-empty pandas DataFrame.")

    id_column = _find_column(features_df, PLAYER_KEY_CANDIDATES)
    draft_year_column = _find_column(features_df, ("draft_year",))
    season_column = _find_column(features_df, ("season",))
    if id_column is None:
        raise ValueError("features_df must include a player identifier.")
    if draft_year_column is None or season_column is None:
        raise ValueError("features_df must include draft_year and season columns.")

    target_columns = _target_columns(features_df)
    if set(target_columns) != {"epa", "yards_after_catch"}:
        raise ValueError(
            "features_df must include numeric EPA and YAC outcome columns "
            "(for example, epa and yards_after_catch)."
        )

    selected = features_df.copy(deep=False)
    draft_years = pd.to_numeric(selected[draft_year_column], errors="coerce")
    seasons = pd.to_numeric(selected[season_column], errors="coerce")
    player_keys = _safe_player_key(selected[id_column])
    rookie_mask = (
        draft_years.isin(ROOKIE_DRAFT_YEARS)
        & seasons.eq(draft_years)
        & player_keys.notna()
    )
    selected = selected.loc[rookie_mask].copy()
    selected["_model_player_key"] = player_keys.loc[rookie_mask]
    if selected.empty:
        raise ValueError(
            "No identified player rows match draft_year 2023–2025 and "
            "season == draft_year."
        )

    duplicate_rows = selected["_model_player_key"].duplicated(keep=False)
    if duplicate_rows.any():
        raise ValueError(
            "features_df must have one aggregated row per player in the rookie "
            "season; aggregate play-level rows using domain-reviewed rules first."
        )

    excluded_columns = {
        id_column,
        draft_year_column,
        season_column,
        *target_columns.values(),
    }
    feature_columns = [
        column
        for column in selected.select_dtypes(include=[np.number]).columns
        if column not in excluded_columns
        and not _is_context_column(column)
        and not _is_target_metric(column)
    ]
    if not feature_columns:
        raise ValueError("No numeric movement features are available for modeling.")

    results: list[dict[str, Any]] = []
    for target_name in ("epa", "yards_after_catch"):
        target_column = target_columns[target_name]
        target_values = pd.to_numeric(
            selected[target_column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        target_mask = target_values.notna()
        target_rows = selected.loc[target_mask]
        target_values = target_values.loc[target_mask].astype(float)
        if len(target_values) < MIN_CV_SAMPLES:
            raise ValueError(
                f"{target_name} has {len(target_values)} valid rookie samples; "
                f"at least {MIN_CV_SAMPLES} are required for nested CV."
            )

        model_features = target_rows[feature_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        model_features = model_features.replace([np.inf, -np.inf], np.nan)
        model_features = model_features.dropna(axis=1, how="all")
        if model_features.empty or model_features.shape[1] == 0:
            raise ValueError(
                f"No non-empty numeric movement features are available for {target_name}."
            )

        row_count = len(target_values)
        outer_folds = min(5, row_count // 2)
        outer_cv = KFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        actuals = target_values.to_numpy()
        predictions = np.full(row_count, np.nan, dtype=float)
        selected_alphas: list[float] = []
        for train_indices, test_indices in outer_cv.split(model_features):
            inner_folds = min(5, len(train_indices) // 2)
            search = _tune_ridge(
                model_features.iloc[train_indices],
                target_values.iloc[train_indices],
                folds=inner_folds,
            )
            predictions[test_indices] = search.predict(
                model_features.iloc[test_indices]
            )
            selected_alphas.append(float(search.best_params_["regressor__alpha"]))

        full_data_inner_folds = min(5, row_count // 2)
        final_search = _tune_ridge(
            model_features,
            target_values,
            folds=full_data_inner_folds,
        )
        has_target_variation = np.ptp(actuals) > 0
        results.append(
            {
                "target_metric": target_name,
                "player_season_count": row_count,
                "feature_count": model_features.shape[1],
                "outer_cv_folds": outer_folds,
                "inner_cv_folds": full_data_inner_folds,
                "selected_alpha_median": float(np.median(selected_alphas)),
                "full_data_best_alpha": float(
                    final_search.best_params_["regressor__alpha"]
                ),
                "cv_rmse": float(np.sqrt(mean_squared_error(actuals, predictions))),
                "cv_mae": float(mean_absolute_error(actuals, predictions)),
                "cv_r2": (
                    float(r2_score(actuals, predictions))
                    if has_target_variation
                    else np.nan
                ),
            }
        )

    metrics = pd.DataFrame(results)
    output_path = _save_model_metrics(metrics)
    print("Rookie-season movement features: nested Ridge cross-validation")
    print(
        "Cohort: draft_year in "
        f"{ROOKIE_DRAFT_YEARS}; rookie season equals draft_year"
    )
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Aggregate cross-validation metrics saved to: {output_path}")
    return metrics


class _ModelIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output_path = (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / f".model-integration-test-{uuid4().hex}.csv"
        )

    def tearDown(self) -> None:
        self.output_path.unlink(missing_ok=True)

    def test_left_join_imputation_and_predictive_insights(self) -> None:
        summary = pd.DataFrame(
            {
                "player_id": [101, 102, 103, 104],
                "draft_year": [2023, 2024, 2025, 2024],
                "average_proprietary_cut_efficiency": [0.5, np.nan, 1.5, 0.75],
                "max_speed": [18.0, 20.0, 22.0, 19.0],
            }
        )
        season = pd.DataFrame(
            {
                "nflId": [101, 101, 101, 102, 103],
                "yards_after_catch": [4.0, 6.0, 100.0, 12.0, 20.0],
                "defensive_stops": [1.0, 2.0, 100.0, 4.0, 5.0],
                "proprietary_cut_efficiency": [0.8, 0.9, 0.1, 0.7, 0.6],
                "season": [2023, 2023, 2024, 2024, 2025],
            }
        )

        with patch(f"{__name__}.OUTPUT_FILENAME", self.output_path.name):
            merged = integrate_combine_with_season_performance(summary, season)

        self.assertEqual(len(merged), 4)
        self.assertEqual(merged["proprietary_cut_efficiency"].isna().sum(), 0)
        self.assertAlmostEqual(
            merged.loc[merged["player_id"].eq(101), "proprietary_cut_efficiency"].item(),
            0.5,
        )
        self.assertEqual(merged.loc[merged["player_id"].eq(101), "yards_after_catch"].item(), 10.0)
        self.assertEqual(merged.loc[merged["player_id"].eq(103), "defensive_stops"].item(), 5.0)
        self.assertTrue(
            pd.isna(merged.loc[merged["player_id"].eq(104), "yards_after_catch"].item())
        )
        self.assertTrue(self.output_path.is_file())
        saved = pd.read_csv(self.output_path)
        self.assertEqual(set(saved["outcome_metric"]), {"yards_after_catch", "defensive_stops"})
        self.assertTrue(saved["pearson_correlation"].notna().all())

    def test_joins_games_to_player_play_before_rookie_season_filter(self) -> None:
        summary = pd.DataFrame(
            {
                "nflId": [201],
                "draft_year": [2024],
                "proprietary_cut_efficiency": [0.8],
            }
        )
        player_play = pd.DataFrame(
            {
                "nflId": [201, 201],
                "game_id": [1, 2],
                "yards_after_catch": [7.0, 70.0],
                "epa": [1.0, 10.0],
            }
        )
        games = pd.DataFrame({"game_id": [1, 2], "season": [2024, 2025]})

        with patch(f"{__name__}.OUTPUT_FILENAME", self.output_path.name):
            merged = integrate_combine_with_season_performance(
                summary, player_play, games_df=games
            )

        self.assertEqual(merged.loc[0, "yards_after_catch"], 7.0)
        self.assertEqual(merged.loc[0, "epa"], 1.0)

    def test_epa_is_a_supported_target_metric(self) -> None:
        self.assertTrue(_is_target_metric("epa"))
        self.assertTrue(_is_target_metric("expected_points_added"))

    def test_run_model_pipeline_nested_cv_with_mock_player_seasons(self) -> None:
        player_count = 12
        movement = np.linspace(0.2, 1.3, player_count)
        mock_features = pd.DataFrame(
            {
                "nflId": np.arange(1001, 1001 + player_count),
                "draft_year": [2023, 2024, 2025] * 4,
                "season": [2023, 2024, 2025] * 4,
                "average_proprietary_cut_efficiency": movement,
                "average_deceleration": np.linspace(0.5, 4.0, player_count),
                "max_speed": np.linspace(15.0, 22.0, player_count),
                "epa": 0.4 * movement + np.linspace(-0.2, 0.2, player_count),
                "yards_after_catch": 8.0 * movement + np.linspace(-1.0, 1.0, player_count),
                "defensive_stops": np.arange(player_count, dtype=float),
            }
        )
        mock_features.loc[3, "max_speed"] = np.nan

        with patch(
            f"{__name__}.MODEL_PIPELINE_OUTPUT_FILENAME",
            self.output_path.name,
        ):
            metrics = run_model_pipeline(mock_features)

        self.assertEqual(set(metrics["target_metric"]), {"epa", "yards_after_catch"})
        self.assertTrue(metrics["cv_rmse"].notna().all())
        self.assertTrue(metrics["cv_mae"].notna().all())
        self.assertTrue(metrics["cv_r2"].notna().all())
        self.assertTrue((metrics["player_season_count"] == player_count).all())
        self.assertTrue((metrics["feature_count"] == 3).all())
        self.assertTrue(self.output_path.is_file())
        saved = pd.read_csv(self.output_path)
        self.assertEqual(len(saved), 2)

    def test_run_model_pipeline_rejects_unaggregated_player_rows(self) -> None:
        repeated_rows = pd.DataFrame(
            {
                "player_id": [1, 1],
                "draft_year": [2024, 2024],
                "season": [2024, 2024],
                "speed": [10.0, 11.0],
                "epa": [0.2, 0.3],
                "yac": [4.0, 5.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "one aggregated row per player"):
            run_model_pipeline(repeated_rows)

    def test_requires_player_identifiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "player identifier"):
            with patch(f"{__name__}.OUTPUT_FILENAME", self.output_path.name):
                integrate_combine_with_season_performance(
                    pd.DataFrame({"proprietary_cut_efficiency": [1.0]}),
                    pd.DataFrame({"yards_after_catch": [3.0]}),
                )


def main() -> int:
    """Run mock-data validation without reading competition datasets."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate player-level combine/season integration and the "
            "nested-CV Ridge baseline."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run mock-data tests without loading competition data.",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test to run the local validation suite")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_ModelIntegrationTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
