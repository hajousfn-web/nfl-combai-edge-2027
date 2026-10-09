"""Join combine tracking features to regular-season metrics and assess signal."""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
    )


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
) -> pd.DataFrame:
    """Left-join player tracking summaries to season metrics and assess signal.

    Player IDs are matched using ``nflId`` when available, otherwise a
    ``playerId``, ``player_identifier``, or ``player`` column. The two inputs
    may use different names for the same key. Repeated season rows are
    collapsed to player-season level (sum recognized outcome metrics, mean
    other numeric fields, and first non-numeric values) before the left merge,
    preventing an accidental many-to-many memory expansion.

    Missing cut-efficiency values are imputed with the observed median. Missing
    outcomes are excluded from that metric's pairwise analysis. The function
    computes Pearson correlations and a standardized univariate Ridge baseline
    (alpha=1; in-sample R²) for numeric season outcomes. Known YAC/defensive-stop
    outcomes are preferred; if none are present, all numeric season metrics are
    analyzed. Aggregate-only insights are written atomically to
    ``outputs/model_integration_insights.csv``; player-level records are not
    written to disk.

    Args:
        player_summary_df: One-row-per-player combine/tracking summary.
        regular_season_df: Player-level or repeated game-level season metrics.

    Returns:
        The merged player-level DataFrame, retaining combine-summary rows with
        unmatched season metrics represented as missing values.

    Raises:
        TypeError: If either argument is not a pandas DataFrame.
        ValueError: If IDs or cut-efficiency features are missing/ambiguous, or
            combine summaries are not unique by player.
        OSError: If the insights CSV cannot be written.
    """
    if not isinstance(player_summary_df, pd.DataFrame):
        raise TypeError("player_summary_df must be a pandas DataFrame.")
    if not isinstance(regular_season_df, pd.DataFrame):
        raise TypeError("regular_season_df must be a pandas DataFrame.")
    if not player_summary_df.columns.is_unique or not regular_season_df.columns.is_unique:
        raise ValueError("Input DataFrames must not contain duplicate column names.")

    summary_key = _find_column(player_summary_df, PLAYER_KEY_CANDIDATES)
    season_key = _find_column(regular_season_df, PLAYER_KEY_CANDIDATES)
    if summary_key is None or season_key is None:
        raise ValueError(
            "Both inputs require a player identifier (nflId preferred, then "
            "playerId, player_identifier, or player)."
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

    # Only known outcome columns or numeric season metrics enter the analysis.
    numeric_season_columns = [
        column
        for column in regular_season_df.select_dtypes(include="number").columns
        if column != season_key
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

    summary = player_summary_df.copy()
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

    # Work with one-season-row-per-player and avoid constructing game-level
    # many-to-many joins in memory.
    season = regular_season_df.copy()
    season["_integration_player_key"] = _safe_player_key(season[season_key])
    season = season.loc[season["_integration_player_key"].notna()].copy()
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


class _ModelIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.output_path = Path(__file__).resolve().parent.parent / "outputs" / OUTPUT_FILENAME
        self.output_path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.output_path.unlink(missing_ok=True)

    def test_left_join_imputation_and_predictive_insights(self) -> None:
        summary = pd.DataFrame(
            {
                "player_id": [101, 102, 103, 104],
                "average_proprietary_cut_efficiency": [0.5, np.nan, 1.5, 0.75],
                "max_speed": [18.0, 20.0, 22.0, 19.0],
            }
        )
        season = pd.DataFrame(
            {
                "nflId": [101, 101, 102, 103],
                "yards_after_catch": [4.0, 6.0, 12.0, 20.0],
                "defensive_stops": [1.0, 2.0, 4.0, 5.0],
                "proprietary_cut_efficiency": [0.8, 0.9, 0.7, 0.6],
            }
        )

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

    def test_requires_player_identifiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "player identifier"):
            integrate_combine_with_season_performance(
                pd.DataFrame({"proprietary_cut_efficiency": [1.0]}),
                pd.DataFrame({"yards_after_catch": [3.0]}),
            )


if __name__ == "__main__":
    unittest.main()
