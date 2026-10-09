"""Stream tracking CSV data and derive per-observation movement metrics."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import TextIO


REQUIRED_COLUMNS = ("game_id", "play_id", "player_id", "frame_id", "x", "y")
OUTPUT_COLUMNS = ("dx", "dy", "displacement", "speed_10hz")


def process_tracking(input_file: TextIO, output_file: TextIO) -> int:
    """Write tracking rows with movement metrics; return the number of rows."""
    reader = csv.DictReader(input_file)
    if reader.fieldnames is None:
        raise ValueError("Input CSV is empty or has no header.")

    fieldnames = [name.strip() for name in reader.fieldnames]
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("Input CSV contains duplicate column names after trimming.")
    missing = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    conflicting = sorted(set(OUTPUT_COLUMNS) & set(fieldnames))
    if conflicting:
        raise ValueError(
            f"Input CSV already contains derived columns: {', '.join(conflicting)}"
        )

    reader.fieldnames = fieldnames
    writer = csv.DictWriter(output_file, fieldnames=fieldnames + list(OUTPUT_COLUMNS))
    writer.writeheader()

    previous: dict[tuple[str, str, str], tuple[int, float, float]] = {}
    row_count = 0
    row_count = 0
    for row_count, row in enumerate(reader, start=1):
        try:
            track = (row["game_id"], row["play_id"], row["player_id"])
            frame = int(row["frame_id"])
            x = float(row["x"])
            y = float(row["y"])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("coordinates must be finite numbers")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid tracking values on CSV data row {row_count}: {error}") from error

        prior = previous.get(track)
        if prior is None:
            dx = dy = displacement = speed = ""
        else:
            previous_frame, previous_x, previous_y = prior
            frame_delta = frame - previous_frame
            if frame_delta <= 0:
                raise ValueError(
                    f"Frames must increase within each track; invalid frame on CSV data row {row_count}."
                )
            dx = x - previous_x
            dy = y - previous_y
            displacement = math.hypot(dx, dy)
            speed = displacement * 10.0 / frame_delta

        row.update(
            dx=dx,
            dy=dy,
            displacement=displacement,
            speed_10hz=speed,
        )
        writer.writerow(row)
        previous[track] = (frame, x, y)

    return row_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream a tracking CSV and add 10 Hz movement metrics."
    )
    parser.add_argument("input_csv", type=Path, help="Tracking-data CSV input")
    parser.add_argument("output_csv", type=Path, help="Processed CSV output")
    args = parser.parse_args()

    try:
        if args.input_csv.resolve() == args.output_csv.resolve():
            raise ValueError("Input and output paths must be different.")
        with args.input_csv.open("r", newline="", encoding="utf-8-sig") as input_file:
            with args.output_csv.open("w", newline="", encoding="utf-8") as output_file:
                row_count = process_tracking(input_file, output_file)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Processed {row_count} rows into {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
