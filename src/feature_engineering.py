"""Kinematic features for NFL 10 Hz player-tracking data."""

from __future__ import annotations

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
    tracks = tracking_df.loc[:, REQUIRED_COLUMNS].copy()
    if tracks.loc[:, TRACK_KEYS].isna().any().any():
        raise ValueError("game_id, play_id, and player_id must not be missing.")

    for column in ("frame_id", "x", "y"):
        try:
            tracks[column] = pd.to_numeric(tracks[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{column} must contain numeric values.") from error
        if not np.isfinite(tracks[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite values.")

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
    previous_distance = np.hypot(previous_dx, previous_dy)
    current_distance = np.hypot(tracks["_dx"], tracks["_dy"])
    valid_turn = previous_distance.gt(0) & current_distance.gt(0)
    cosine = (
        (previous_dx * tracks["_dx"] + previous_dy * tracks["_dy"])
        / (previous_distance * current_distance)
    ).where(valid_turn)
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
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = tracks.groupby("player_id", sort=True, observed=True).agg(
        average_deceleration=("_deceleration", "mean"),
        max_speed=("_speed", "max"),
        average_proprietary_cut_efficiency=("proprietary_cut_efficiency", "mean"),
        total_sharp_cuts=("_is_sharp_cut", "sum"),
    )
    summary["total_sharp_cuts"] = summary["total_sharp_cuts"].astype("int64")
    return summary.reset_index()[list(SUMMARY_COLUMNS)]


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

        self.assertEqual(result.loc[10, "total_sharp_cuts"], 1)
        self.assertAlmostEqual(
            result.loc[10, "average_proprietary_cut_efficiency"], 1.0
        )
        self.assertAlmostEqual(result.loc[10, "max_speed"], 20.0)
        self.assertEqual(result.loc[20, "total_sharp_cuts"], 0)
        self.assertTrue(
            np.isnan(result.loc[20, "average_proprietary_cut_efficiency"])
        )
        self.assertAlmostEqual(result.loc[20, "average_deceleration"], 100.0)

    def test_empty_input_returns_summary_columns(self) -> None:
        empty = pd.DataFrame(columns=REQUIRED_COLUMNS)
        result = calculate_deceleration_efficiency(empty)

        self.assertEqual(list(result.columns), list(SUMMARY_COLUMNS))
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
