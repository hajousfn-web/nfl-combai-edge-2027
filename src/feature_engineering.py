"""Kinematic features for NFL 10 Hz player-tracking data."""

from __future__ import annotations

import argparse
import unittest

import numpy as np
import pandas as pd


TRACK_KEYS = ("game_id", "play_id", "player_id")
REQUIRED_COLUMNS = (*TRACK_KEYS, "frame_id", "x", "y")
SUMMARY_COLUMNS = (
    "player_id",
    "average_deceleration",
    "max_speed",
    "average_proprietary_cut_efficiency",
    "total_sharp_cuts",
)
OPTIONAL_SUMMARY_COLUMNS = ("draft_year",)
SAMPLE_RATE_HZ = 10.0
SMOOTHING_WINDOW = 3
SHARP_CUT_THRESHOLD_DEGREES = 45.0


def _prepare_tracking(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort only the columns needed for kinematic features."""
    if not isinstance(tracking_df, pd.DataFrame):
        raise TypeError("tracking_df must be a pandas DataFrame.")

    missing = sorted(set(REQUIRED_COLUMNS) - set(tracking_df.columns))
    if missing:
        raise ValueError(f"tracking_df is missing required columns: {', '.join(missing)}")
    if not tracking_df.columns.is_unique:
        raise ValueError("tracking_df must not contain duplicate column names.")

    # Keep only the inputs needed for these features to limit working memory.
    included_columns = [
        *REQUIRED_COLUMNS,
        *[column for column in OPTIONAL_SUMMARY_COLUMNS if column in tracking_df.columns],
    ]
    tracks = tracking_df.loc[:, included_columns].copy().reset_index(drop=True)
    if tracks.loc[:, TRACK_KEYS].isna().any().any():
        raise ValueError("game_id, play_id, and player_id must not be missing.")

    for column in ("frame_id", "x", "y"):
        try:
            tracks[column] = pd.to_numeric(tracks[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain numeric values.") from error
        if not np.isfinite(tracks[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite values.")

    if "draft_year" in tracks.columns:
        try:
            tracks["draft_year"] = pd.to_numeric(tracks["draft_year"], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("draft_year must contain numeric years.") from error
        if not np.isfinite(tracks["draft_year"].to_numpy(dtype=float)).all():
            raise ValueError("draft_year must contain only finite years.")
        if not tracks["draft_year"].mod(1).eq(0).all():
            raise ValueError("draft_year must contain whole-number years.")
        year_counts = tracks.groupby("player_id", observed=True)["draft_year"].nunique()
        if year_counts.gt(1).any():
            raise ValueError("draft_year must be consistent for each player_id.")

    if not tracks["frame_id"].mod(1).eq(0).all():
        raise ValueError("frame_id must contain whole-number frame indices.")

    tracks.sort_values([*TRACK_KEYS, "frame_id"], inplace=True, kind="mergesort")
    track_groups = tracks.groupby(list(TRACK_KEYS), sort=False, observed=True)
    frame_delta = track_groups["frame_id"].diff()
    if frame_delta.dropna().le(0).any():
        raise ValueError("frame_id must be unique and increasing within each track.")
    tracks["_elapsed_seconds"] = frame_delta / SAMPLE_RATE_HZ
    return tracks


def calculate_velocity_and_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Add smoothed 10 Hz velocity, speed, and acceleration columns.

    A causal trailing three-observation rolling mean smooths frame-to-frame
    velocities independently within each game/play/player track; it does not
    use future observations. At the edges, available observations are used.
    Acceleration components are finite
    differences of smoothed velocity divided by the actual frame elapsed time;
    the first observation in each track has undefined velocity/acceleration.

    Coordinates are assumed to be measured in yards. The returned DataFrame
    retains all input columns and adds ``vx``, ``vy``, ``speed``, ``ax``, ``ay``,
    and ``acceleration`` (vector magnitude). Input rows are sorted in the
    returned copy; the caller's DataFrame is not modified.

    Args:
        df: Tracking data with game_id, play_id, player_id, frame_id, x, y.

    Returns:
        A sorted copy of tracking data with per-observation kinematic columns.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If required columns or values are invalid.
    """
    tracks = _prepare_tracking(df)
    groups = tracks.groupby(list(TRACK_KEYS), sort=False, observed=True)

    frame_dx = groups["x"].diff()
    frame_dy = groups["y"].diff()
    frame_seconds = tracks["_elapsed_seconds"]
    tracks["_vx_raw"] = frame_dx.div(frame_seconds)
    tracks["_vy_raw"] = frame_dy.div(frame_seconds)

    track_keys = [tracks[column] for column in TRACK_KEYS]
    velocity_groups = tracks["_vx_raw"].groupby(track_keys, sort=False, observed=True)
    tracks["vx"] = velocity_groups.transform(
        lambda values: values.rolling(
            window=SMOOTHING_WINDOW, min_periods=1
        ).mean()
    )
    velocity_groups = tracks["_vy_raw"].groupby(track_keys, sort=False, observed=True)
    tracks["vy"] = velocity_groups.transform(
        lambda values: values.rolling(
            window=SMOOTHING_WINDOW, min_periods=1
        ).mean()
    )
    tracks["speed"] = np.hypot(tracks["vx"], tracks["vy"])

    smoothed_groups = tracks.groupby(list(TRACK_KEYS), sort=False, observed=True)
    tracks["ax"] = smoothed_groups["vx"].diff().div(frame_seconds)
    tracks["ay"] = smoothed_groups["vy"].diff().div(frame_seconds)
    tracks["acceleration"] = np.hypot(tracks["ax"], tracks["ay"])
    derived_columns = ["vx", "vy", "speed", "ax", "ay", "acceleration"]
    if np.isinf(tracks[derived_columns].to_numpy(dtype=float)).any():
        raise ValueError("Kinematic calculations produced infinite values.")
    return tracks.drop(columns=["_elapsed_seconds", "_vx_raw", "_vy_raw"])


def compute_cut_efficiency(
    df: pd.DataFrame,
    angle_threshold: float = SHARP_CUT_THRESHOLD_DEGREES,
) -> pd.DataFrame:
    """Add direction-change angles and cut-efficiency speed-retention metrics.

    A sharp cut is a change in direction strictly greater than
    ``angle_threshold`` degrees between consecutive non-zero smoothed velocity
    vectors. For such a cut, ``proprietary_cut_efficiency`` is exit speed
    divided by entry speed. It is a speed-retention proxy, not physical
    momentum, because player mass is not measured. Non-cut rows and cuts with
    zero/undefined entry speed receive NaN efficiency; division by zero is
    never performed.

    Args:
        df: Tracking DataFrame. If ``vx``, ``vy``, and ``speed`` are absent,
            they are calculated first.
        angle_threshold: Sharp-cut threshold in degrees, in [0, 180).

    Returns:
        A sorted copy with ``direction_change_degrees``, ``is_sharp_cut``,
        ``entry_speed``, and ``proprietary_cut_efficiency`` columns.

    Raises:
        TypeError: If df is not a pandas DataFrame.
        ValueError: If threshold or tracking data is invalid.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(angle_threshold, (int, float)) or not np.isfinite(
        angle_threshold
    ):
        raise ValueError("angle_threshold must be a finite number.")
    if not 0.0 <= angle_threshold < 180.0:
        raise ValueError("angle_threshold must be in [0, 180).")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    velocity_columns = {"vx", "vy", "speed"}
    if velocity_columns.issubset(df.columns):
        _prepare_tracking(df)
        tracks = df.copy().reset_index(drop=True)
        for column in ("frame_id", "x", "y"):
            tracks[column] = pd.to_numeric(tracks[column], errors="raise")
        tracks.sort_values(
            [*TRACK_KEYS, "frame_id"], inplace=True, kind="mergesort"
        )
        tracks.reset_index(drop=True, inplace=True)
        for column in ("vx", "vy", "speed"):
            try:
                tracks[column] = pd.to_numeric(tracks[column], errors="raise")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{column} must contain numeric values.") from error
        velocity_values = tracks[["vx", "vy", "speed"]].to_numpy(dtype=float)
        if np.isinf(velocity_values).any() or tracks["speed"].lt(0).any():
            raise ValueError(
                "vx, vy, and speed cannot be infinite; speed cannot be negative."
            )
    else:
        tracks = calculate_velocity_and_acceleration(df)

    groups = tracks.groupby(list(TRACK_KEYS), sort=False, observed=True)
    frame_seconds = groups["frame_id"].diff() / SAMPLE_RATE_HZ
    direction_dx = groups["x"].diff().div(frame_seconds)
    direction_dy = groups["y"].diff().div(frame_seconds)
    group_keys = [tracks[column] for column in TRACK_KEYS]
    prior_dx = direction_dx.groupby(group_keys, sort=False, observed=True).shift()
    prior_dy = direction_dy.groupby(group_keys, sort=False, observed=True).shift()
    entry_speed = groups["speed"].shift()
    cross_product = prior_dx * direction_dy - prior_dy * direction_dx
    dot_product = prior_dx * direction_dx + prior_dy * direction_dy
    previous_distance = (prior_dx.pow(2) + prior_dy.pow(2)).pow(0.5)
    current_distance = (direction_dx.pow(2) + direction_dy.pow(2)).pow(0.5)
    has_direction = previous_distance.gt(0) & current_distance.gt(0)
    direction_angles = pd.Series(
        np.degrees(np.arctan2(cross_product.abs(), dot_product)),
        index=tracks.index,
    )
    tracks["direction_change_degrees"] = direction_angles.where(has_direction)
    tracks["is_sharp_cut"] = tracks["direction_change_degrees"].gt(angle_threshold)
    tracks["entry_speed"] = entry_speed
    tracks["proprietary_cut_efficiency"] = (
        tracks["speed"].div(entry_speed)
    ).where(tracks["is_sharp_cut"] & entry_speed.gt(0))
    return tracks


def run_feature_pipeline(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Run kinematic and cut-efficiency feature extraction with concise logs.

    Args:
        tracking_df: Raw tracking DataFrame. This function operates in memory;
            use bounded samples for local validation.

    Returns:
        Sorted per-observation tracking data with velocity, acceleration,
        sharp-cut, and cut-efficiency features.
    """
    print("Feature pipeline: validating and sorting tracking observations...")
    features = calculate_velocity_and_acceleration(tracking_df)
    print(
        f"Feature pipeline: computed kinematics for {len(features):,} observations "
        f"across {features['player_id'].nunique():,} players."
    )
    features = compute_cut_efficiency(features)
    sharp_cuts = int(features["is_sharp_cut"].sum())
    valid_efficiencies = int(features["proprietary_cut_efficiency"].notna().sum())
    print(
        f"Feature pipeline: detected {sharp_cuts:,} sharp cuts; "
        f"{valid_efficiencies:,} have a defined speed-retention score."
    )
    return features


def calculate_deceleration_efficiency(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize speed, positive deceleration, and sharp-cut efficiency/player.

    Retained for compatibility with existing integration consumers. Coordinates
    are assumed to be in yards; deceleration is the positive decrease in
    smoothed speed divided by elapsed frame time (yards/s²).
    """
    features = run_feature_pipeline(tracking_df)
    track_groups = features.groupby(list(TRACK_KEYS), sort=False, observed=True)
    previous_speed = track_groups["speed"].shift()
    elapsed_seconds = track_groups["frame_id"].diff() / SAMPLE_RATE_HZ
    features["deceleration"] = (
        (previous_speed - features["speed"]).div(elapsed_seconds).clip(lower=0)
    )
    aggregations: dict[str, tuple[str, str]] = {
        "average_deceleration": ("deceleration", "mean"),
        "max_speed": ("speed", "max"),
        "average_proprietary_cut_efficiency": (
            "proprietary_cut_efficiency",
            "mean",
        ),
        "total_sharp_cuts": ("is_sharp_cut", "sum"),
    }
    if "draft_year" in features.columns:
        aggregations["draft_year"] = ("draft_year", "first")
    summary = features.groupby("player_id", sort=True, observed=True).agg(**aggregations)
    summary["total_sharp_cuts"] = summary["total_sharp_cuts"].astype("int64")
    columns: list[str] = list(SUMMARY_COLUMNS)
    if "draft_year" in features.columns:
        columns.insert(1, "draft_year")
    return summary.reset_index()[columns]


class _FeatureEngineeringTest(unittest.TestCase):
    def test_calculates_smoothed_velocity_and_acceleration(self) -> None:
        tracking = pd.DataFrame(
            [
                (1, 1, 1, 3, 3.0, 0.0),
                (1, 1, 1, 1, 0.0, 0.0),
                (1, 1, 1, 2, 1.0, 0.0),
                (1, 1, 1, 4, 6.0, 0.0),
            ],
            columns=REQUIRED_COLUMNS,
            index=[0, 0, 1, 1],
        )

        result = calculate_velocity_and_acceleration(tracking)

        self.assertEqual(result["frame_id"].tolist(), [1, 2, 3, 4])
        np.testing.assert_allclose(
            result["vx"].to_numpy(dtype=float),
            [np.nan, 10.0, 15.0, 20.0],
        )
        self.assertTrue(np.isnan(result["ax"].iloc[0]))
        self.assertAlmostEqual(float(result["speed"].iloc[2]), 15.0)

    def test_cut_efficiency_detects_turn_and_guards_zero_entry_speed(self) -> None:
        tracking = pd.DataFrame(
            [
                (1, 1, 1, 1, 0.0, 0.0, 10.0, 0.0, 10.0),
                (1, 1, 1, 2, 1.0, 0.0, 10.0, 0.0, 10.0),
                (1, 1, 1, 3, 1.0, 1.0, 0.0, 10.0, 10.0),
                (1, 1, 2, 1, 0.0, 0.0, 0.0, 0.0, 0.0),
                (1, 1, 2, 2, 0.0, 0.0, 5.0, 0.0, 5.0),
            ],
            columns=(*REQUIRED_COLUMNS, "vx", "vy", "speed"),
        )

        result = compute_cut_efficiency(tracking)

        player_one = result.loc[result["player_id"].eq(1)]
        player_two = result.loc[result["player_id"].eq(2)]
        self.assertEqual(int(player_one["is_sharp_cut"].sum()), 1)
        self.assertAlmostEqual(
            float(player_one["proprietary_cut_efficiency"].dropna().iloc[0]),
            1.0,
        )
        self.assertFalse(player_two["is_sharp_cut"].any())
        self.assertTrue(player_two["proprietary_cut_efficiency"].isna().all())

    def test_run_pipeline_returns_per_observation_features(self) -> None:
        tracking = pd.DataFrame(
            [
                (1, 1, 1, 1, 0.0, 0.0),
                (1, 1, 1, 2, 1.0, 0.0),
                (1, 1, 1, 3, 1.0, 1.0),
            ],
            columns=REQUIRED_COLUMNS,
        )

        result = run_feature_pipeline(tracking)

        self.assertEqual(len(result), len(tracking))
        self.assertTrue(
            {"vx", "vy", "speed", "ax", "ay", "acceleration",
             "direction_change_degrees", "is_sharp_cut",
             "proprietary_cut_efficiency"}.issubset(result.columns)
        )

    def test_sorts_tracks_and_calculates_sharp_cut_efficiency(self) -> None:
        tracking = pd.DataFrame(
            [
                # Deliberately out of frame order: a 90-degree cut at frame 3.
                (1, 7, 10, 4, 1, 3),
                (1, 7, 10, 2, 1, 0),
                (1, 7, 10, 3, 1, 1),
                (1, 7, 10, 1, 0, 0),
                # Straight movement, with a speed loss on the final segment.
                (1, 7, 20, 1, 0, 0),
                (1, 7, 20, 2, 2, 0),
                (1, 7, 20, 3, 3, 0),
            ],
            columns=(*TRACK_KEYS, "frame_id", "x", "y"),
        )

        result = calculate_deceleration_efficiency(tracking).set_index("player_id")

        sharp_cuts = result["total_sharp_cuts"].to_numpy(dtype=np.int64)
        cut_efficiency = result[
            "average_proprietary_cut_efficiency"
        ].to_numpy(dtype=np.float64)
        max_speed = result["max_speed"].to_numpy(dtype=np.float64)
        deceleration = result["average_deceleration"].to_numpy(dtype=np.float64)

        self.assertEqual(sharp_cuts[0], 1)
        self.assertTrue(np.isfinite(cut_efficiency[0]))
        self.assertAlmostEqual(max_speed[0], float(np.hypot(10.0 / 3.0, 10.0)))
        self.assertEqual(sharp_cuts[1], 0)
        self.assertTrue(
            np.isnan(cut_efficiency[1])
        )
        self.assertAlmostEqual(deceleration[1], 50.0)

    def test_empty_input_returns_summary_columns(self) -> None:
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
        result = calculate_deceleration_efficiency(empty)

        self.assertEqual(list(result.columns), list(SUMMARY_COLUMNS))
        self.assertTrue(result.empty)

    def test_preserves_draft_year_for_cohort_integration(self) -> None:
        tracking = pd.DataFrame(
            [
                (1, 7, 10, 1, 0, 0, 2024),
                (1, 7, 10, 2, 1, 0, 2024),
            ],
            columns=(*REQUIRED_COLUMNS, "draft_year"),
        )

        result = calculate_deceleration_efficiency(tracking)

        self.assertIn("draft_year", result.columns)
        self.assertEqual(
            result["draft_year"].to_numpy(dtype=np.int64)[0],
            2024,
        )


def main() -> int:
    """Run the mock-data feature checks without loading any tracking dataset."""
    parser = argparse.ArgumentParser(
        description="Validate the tracking feature calculations locally."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run mock-data tests without loading competition data.",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.error("use --self-test to run the local validation suite")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_FeatureEngineeringTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
