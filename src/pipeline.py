"""Stream tracking CSV data and derive per-observation movement metrics."""

from __future__ import annotations

import argparse
import csv
from io import StringIO
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import TextIO
import unittest


REQUIRED_COLUMNS = ("game_id", "play_id", "player_id", "frame_id", "x", "y")
OUTPUT_COLUMNS = ("dx", "dy", "displacement", "speed_10hz")


def _outputs_dir() -> Path:
    """Return the resolved project outputs directory."""
    return (Path(__file__).resolve().parent.parent / "outputs").resolve()


def _resolve_output_path(path: Path) -> Path:
    """Require CSV output to be written directly inside the project outputs/."""
    outputs_dir = _outputs_dir()
    candidate = path if path.is_absolute() else Path(__file__).resolve().parent.parent / path
    resolved = candidate.resolve()
    if resolved.parent != outputs_dir:
        raise ValueError(f"Output CSV must be directly inside outputs/: {path}")
    if resolved.suffix.casefold() != ".csv":
        raise ValueError("Output file must have a .csv extension.")
    return resolved


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

    current_track: tuple[str, str, str] | None = None
    previous: tuple[int, float, float] | None = None
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

        if current_track is not None and track < current_track:
            raise ValueError(
                "Input rows must be sorted by game_id, play_id, and player_id "
                f"before processing; track order decreased on CSV data row {row_count}."
            )
        if track != current_track:
            current_track = track
            previous = None

        if previous is None:
            dx = dy = displacement = speed = ""
        else:
            previous_frame, previous_x, previous_y = previous
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
        previous = (frame, x, y)

    return row_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream a tracking CSV and add 10 Hz movement metrics."
    )
    parser.add_argument("input_csv", type=Path, nargs="?", help="Tracking-data CSV input")
    parser.add_argument("output_csv", type=Path, nargs="?", help="Processed CSV output in outputs/")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run mock-data tests without reading or writing project data.",
    )
    args = parser.parse_args()

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(_PipelineTest)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.input_csv is None or args.output_csv is None:
        parser.error("input_csv and output_csv are required unless --self-test is used")

    try:
        output_path = _resolve_output_path(args.output_csv)
        if args.input_csv.resolve() == output_path:
            raise ValueError("Input and output paths must be different.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with args.input_csv.open("r", newline="", encoding="utf-8-sig") as input_file:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    suffix=".tmp",
                    prefix=".tracking-pipeline-",
                    dir=output_path.parent,
                    delete=False,
                ) as output_file:
                    temporary_path = Path(output_file.name)
                    row_count = process_tracking(input_file, output_file)
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Processed {row_count} rows into {output_path}")
    return 0


class _PipelineTest(unittest.TestCase):
    def test_processes_tracking_rows_and_derives_speed(self) -> None:
        input_file = StringIO(
            "game_id,play_id,player_id,frame_id,x,y\n"
            "1,2,3,1,0,0\n"
            "1,2,3,2,3,4\n"
        )
        output_file = StringIO()
        self.assertEqual(process_tracking(input_file, output_file), 2)
        self.assertIn("speed_10hz", output_file.getvalue().splitlines()[0])
        self.assertTrue(output_file.getvalue().splitlines()[2].endswith("50.0"))

    def test_rejects_decreasing_track_order(self) -> None:
        input_file = StringIO(
            "game_id,play_id,player_id,frame_id,x,y\n"
            "2,1,1,1,0,0\n"
            "1,1,1,1,0,0\n"
        )
        with self.assertRaisesRegex(ValueError, "sorted"):
            process_tracking(input_file, StringIO())

    def test_rejects_output_outside_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "outputs"):
            _resolve_output_path(Path("data/processed.csv"))


if __name__ == "__main__":
    raise SystemExit(main())
