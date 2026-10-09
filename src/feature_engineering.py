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
SHARP_CUT_THRESHOLD_DEGREES = 45.0


def calculate_deceleration_efficiency(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize deceleration, peak speed, and sharp-cut efficiency by player.

    Coordinates are assumed to be in yards and sampled at 10 Hz. Speed is
    calculated from consecutive positions within each game/play/player track;
    deceleration is the positive loss of speed between segments, in yards/s².
    A sharp cut is a change in movement direction greater than 45 degrees.
    Its efficiency is the ratio of post-cut to pre-cut speed: 1.0 means all
    speed was retained, values below 1.0 indicate speed loss, and values above
    1.0 indicate the player accelerated. Mass is not present in tracking data,
    so speed retention is used as a momentum-retention proxy.

    Args:
        tracking_df: DataFrame with game_id, play_id, player_id, frame_id, x,
            and y columns.

    Returns:
        One row per player_id, with average_deceleration, max_speed,
        average_proprietary_cut_efficiency, and total_sharp_cuts.

    Raises:
        TypeError: If tracking_df is not a pandas DataFrame.
        ValueError: If required values are missing, invalid, or ambiguous.
    """
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
    tracks = tracking_df.loc[:, included_columns].copy()
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
    groups = tracks.groupby(list(TRACK_KEYS), sort=False, observed=True)
    frame_delta = groups["frame_id"].diff()
    if frame_delta.dropna().le(0).any():
        raise ValueError("frame_id must be unique and increasing within each track.")

    # Each row after the first in a track describes the movement segment ending
    # at that frame. Frame gaps are respected when deriving elapsed time.
    tracks["_elapsed_seconds"] = frame_delta / SAMPLE_RATE_HZ
    tracks["_dx"] = groups["x"].diff()
    tracks["_dy"] = groups["y"].diff()
    tracks["_speed"] = np.hypot(tracks["_dx"], tracks["_dy"]) / tracks[
        "_elapsed_seconds"
    ]

    speed_groups = tracks.groupby(list(TRACK_KEYS), sort=False, observed=True)[
        "_speed"
    ]
    previous_speed = speed_groups.shift()
    elapsed_between_speeds = frame_delta / SAMPLE_RATE_HZ
    speed_loss_rate = (previous_speed - tracks["_speed"]) / elapsed_between_speeds
    tracks["_deceleration"] = speed_loss_rate.clip(lower=0)

    previous_dx = groups["_dx"].shift()
    previous_dy = groups["_dy"].shift()
    previous_distance = (previous_dx.pow(2) + previous_dy.pow(2)).pow(0.5)
    current_distance = (tracks["_dx"].pow(2) + tracks["_dy"].pow(2)).pow(0.5)
    valid_turn = previous_distance.gt(0) & current_distance.gt(0)
    cosine = (
        (previous_dx * tracks["_dx"] + previous_dy * tracks["_dy"])
        .div(previous_distance * current_distance)
        .where(valid_turn)
    )
    tracks["_direction_change"] = np.degrees(
        np.arccos(cosine.clip(lower=-1, upper=1))
    )
    tracks["_is_sharp_cut"] = tracks["_direction_change"].gt(
        SHARP_CUT_THRESHOLD_DEGREES
    )
    tracks["proprietary_cut_efficiency"] = (
        tracks["_speed"] / previous_speed
    ).where(tracks["_is_sharp_cut"] & previous_speed.gt(0))

    if tracks.empty:
        summary_columns: list[str] = list(SUMMARY_COLUMNS)
        if "draft_year" in tracks.columns:
            summary_columns.insert(1, "draft_year")
        return pd.DataFrame(columns=summary_columns)

    aggregations: dict[str, tuple[str, str]] = {
        "average_deceleration": ("_deceleration", "mean"),
        "max_speed": ("_speed", "max"),
        "average_proprietary_cut_efficiency": (
            "proprietary_cut_efficiency",
            "mean",
        ),
        "total_sharp_cuts": ("_is_sharp_cut", "sum"),
    }
    if "draft_year" in tracks.columns:
        aggregations["draft_year"] = ("draft_year", "first")

    summary = tracks.groupby("player_id", sort=True, observed=True).agg(
        **aggregations,
    )
    summary["total_sharp_cuts"] = summary["total_sharp_cuts"].astype("int64")
    summary_columns: list[str] = list(SUMMARY_COLUMNS)
    if "draft_year" in tracks.columns:
        summary_columns.insert(1, "draft_year")
    return summary.reset_index()[summary_columns]


class _FeatureEngineeringTest(unittest.TestCase):
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
        self.assertAlmostEqual(cut_efficiency[0], 1.0)
        self.assertAlmostEqual(max_speed[0], 20.0)
        self.assertEqual(sharp_cuts[1], 0)
        self.assertTrue(
            np.isnan(cut_efficiency[1])
        )
        self.assertAlmostEqual(deceleration[1], 100.0)

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
