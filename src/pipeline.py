"""Stream tracking CSV data and derive per-observation movement metrics."""

# Author: Soufiane Hajou
# VisiShield-Edge(TM) | Copyright (c) 2026 Soufiane Hajou. All Rights Reserved.

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from io import StringIO
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import IO, TextIO, TypeAlias
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

if __package__:
    from . import feature_engineering, model_integration, statistical_audit
else:
    import feature_engineering
    import model_integration
    import statistical_audit


REQUIRED_COLUMNS = ("game_id", "play_id", "player_id", "frame_id", "x", "y")
OUTPUT_COLUMNS = ("dx", "dy", "displacement", "speed_10hz")
DEFAULT_TRACKING_CSV = Path(__file__).resolve().parent.parent / "data" / "combine_tracking.csv"
DEFAULT_PERFORMANCE_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "player_season_outcomes.csv"
)
DEFAULT_MAX_TRACKING_ROWS = 250_000
PERFORMANCE_ID_CANDIDATES = ("nflid", "playerid", "playeridentifier", "player")
EPA_CANDIDATES = ("epa", "expected_points_added")
YAC_CANDIDATES = ("yac", "yards_after_catch")
LOGGER = logging.getLogger("nfl_combai_edge.pipeline")

FrameSource: TypeAlias = pd.DataFrame | Path | str


@dataclass(frozen=True)
class PipelineResult:
    """In-memory outputs of feature extraction and optional model evaluation."""

    player_summary: pd.DataFrame
    model_metrics: pd.DataFrame | None
    partial_correlations: pd.DataFrame | None = None
    forward_validation: pd.DataFrame | None = None
    prelaunch_audit: pd.DataFrame | None = None


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


def _normalized_column_name(name: object) -> str:
    """Normalize schema names for case- and punctuation-insensitive matching."""
    return "".join(character for character in str(name).casefold() if character.isalnum())


def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    by_normalized_name = {
        _normalized_column_name(column): column for column in frame.columns
    }
    for candidate in candidates:
        match = by_normalized_name.get(_normalized_column_name(candidate))
        if match is not None:
            return match
    return None


def _prepare_tracking_entities(tracking: pd.DataFrame) -> pd.DataFrame:
    """Keep player entities and audit unique per-frame tracking linkage."""
    return statistical_audit.prepare_tracking_players(tracking)


def _load_frame(source: FrameSource, *, name: str, max_rows: int) -> pd.DataFrame:
    """Load a DataFrame or bounded CSV, rejecting ambiguous/oversized inputs."""
    if isinstance(source, pd.DataFrame):
        frame = source
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{name} CSV does not exist or is not a file: {path}")
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as input_file:
                header = next(csv.reader(input_file), None)
            if header is None:
                raise ValueError(f"{name} CSV is empty: {path}")
            normalized_header = [column.strip() for column in header]
            if len(normalized_header) != len(set(normalized_header)):
                raise ValueError(f"{name} CSV contains duplicate column names.")
            frame = pd.read_csv(path, nrows=max_rows + 1, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError, UnicodeError) as error:
            raise ValueError(f"Could not read {name} CSV {path}: {error}") from error

    if not frame.columns.is_unique:
        raise ValueError(f"{name} data must not contain duplicate column names.")
    if len(frame) > max_rows:
        raise ValueError(
            f"{name} data exceeds the configured {max_rows:,}-row memory bound; "
            "provide a filtered, complete-track sample."
        )
    if frame.empty:
        raise ValueError(f"{name} data contains no rows.")
    return frame


def _build_model_input(
    player_summary: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """Join player summaries to validated, pre-aggregated rookie outcomes."""
    if "draft_year" not in player_summary.columns:
        raise ValueError(
            "Model integration requires draft_year in the tracking input; "
            "include player cohort metadata before running the full model stage."
        )

    player_column = _find_column(performance, PERFORMANCE_ID_CANDIDATES)
    season_column = _find_column(performance, ("season",))
    epa_column = _find_column(performance, EPA_CANDIDATES)
    yac_column = _find_column(performance, YAC_CANDIDATES)
    if player_column is None or season_column is None:
        raise ValueError("Performance data must include a player ID and season.")
    if epa_column is None or yac_column is None:
        raise ValueError("Performance data must include numeric EPA and YAC columns.")
    statistical_audit.validate_player_identifier_consistency(
        performance,
        preferred_column=player_column,
        table_name="Performance data",
    )
    statistical_audit.validate_linkage_keys(
        performance,
        table_name="Performance data",
        key_columns=(player_column, season_column),
        unique=False,
    )

    summary = player_summary.copy()
    summary["_pipeline_join_key"] = summary["player_id"].astype("string").str.strip()
    if summary["_pipeline_join_key"].isna().any() or summary[
        "_pipeline_join_key"
    ].eq("").any():
        raise ValueError("Tracking summaries contain missing player identifiers.")
    if summary["_pipeline_join_key"].duplicated().any():
        raise ValueError("Tracking summaries must contain one row per player.")
    summary["draft_year"] = pd.to_numeric(summary["draft_year"], errors="coerce")
    if summary["draft_year"].isna().any():
        raise ValueError("draft_year must be present and numeric for every player.")
    summary["season"] = summary["draft_year"]

    outcome = performance.loc[
        :, [player_column, season_column, epa_column, yac_column]
    ].copy()
    outcome["_pipeline_join_key"] = (
        outcome[player_column].astype("string").str.strip()
    )
    outcome["_pipeline_season"] = pd.to_numeric(
        outcome[season_column],
        errors="coerce",
    )
    draft_years = summary.loc[
        :, ["_pipeline_join_key", "draft_year"]
    ].drop_duplicates()
    outcome = outcome.merge(
        draft_years,
        how="inner",
        on="_pipeline_join_key",
        validate="many_to_one",
        sort=False,
    )
    outcome = outcome.loc[
        outcome["_pipeline_season"].eq(outcome["draft_year"])
        & outcome["_pipeline_season"].isin(
            model_integration.ROOKIE_DRAFT_YEARS
        )
    ].copy()
    if outcome["_pipeline_join_key"].duplicated().any():
        raise ValueError(
            "Performance data must contain one pre-aggregated rookie-season "
            "row per player; aggregate play-level outcomes using reviewed rules."
        )

    outcome = outcome.loc[
        :, ["_pipeline_join_key", epa_column, yac_column]
    ].rename(
        columns={
            epa_column: "epa",
            yac_column: "yards_after_catch",
        }
    )
    model_input = summary.merge(
        outcome,
        how="left",
        on="_pipeline_join_key",
        validate="one_to_one",
        sort=False,
    )
    return model_input.drop(columns="_pipeline_join_key")


def run_full_pipeline(
    tracking_data: FrameSource | None = None,
    performance_data: FrameSource | None = None,
    *,
    max_tracking_rows: int = DEFAULT_MAX_TRACKING_ROWS,
) -> PipelineResult:
    """Run bounded tracking feature extraction and optional EPA/YAC modeling.

    Tracking input must be a DataFrame or CSV with the feature-engineering
    schema. CSV reads are capped to ``max_tracking_rows`` and oversized files
    are rejected rather than silently truncated, since truncation can split
    player tracks. Optional performance data must already be aggregated to one
    player/rookie-season row with player ID, season, EPA, and YAC. No raw or
    player-level files are written by this orchestrator.
    """
    if (
        isinstance(max_tracking_rows, bool)
        or not isinstance(max_tracking_rows, int)
        or max_tracking_rows < 1
    ):
        raise ValueError("max_tracking_rows must be a positive integer.")

    tracking_source: FrameSource = (
        DEFAULT_TRACKING_CSV if tracking_data is None else tracking_data
    )
    LOGGER.info("Loading tracking observations (limit: %s rows).", f"{max_tracking_rows:,}")
    try:
        tracking = _load_frame(
            tracking_source,
            name="Tracking",
            max_rows=max_tracking_rows,
        )
        tracking = _prepare_tracking_entities(tracking)
        missing_columns = sorted(
            set(feature_engineering.REQUIRED_COLUMNS) - set(tracking.columns)
        )
        if missing_columns:
            raise ValueError(
                "Tracking data is missing required columns: "
                + ", ".join(missing_columns)
            )

        prelaunch_audit: pd.DataFrame | None = None
        if "launch_frame_id" in tracking.columns:
            LOGGER.info("Auditing event-anchored prelaunch motion.")
            prelaunch_audit = statistical_audit.audit_prelaunch_motion(tracking)

        LOGGER.info("Extracting 10 Hz kinematics and cut-efficiency summaries.")
        player_summary = feature_engineering.calculate_deceleration_efficiency(
            tracking
        )
        if player_summary.empty:
            raise ValueError("Feature engineering returned no player summaries.")

        if performance_data is None and DEFAULT_PERFORMANCE_CSV.is_file():
            performance_source: FrameSource | None = DEFAULT_PERFORMANCE_CSV
        else:
            performance_source = performance_data

        model_metrics: pd.DataFrame | None = None
        partial_correlations: pd.DataFrame | None = None
        forward_validation: pd.DataFrame | None = None
        if performance_source is None:
            LOGGER.warning(
                "No performance outcomes supplied; returning feature summaries "
                "without model evaluation."
            )
        else:
            LOGGER.info("Loading bounded, player-season performance outcomes.")
            performance = _load_frame(
                performance_source,
                name="Performance",
                max_rows=max_tracking_rows,
            )
            model_input = _build_model_input(player_summary, performance)
            LOGGER.info("Running nested-CV Ridge baselines for EPA and YAC.")
            model_metrics = model_integration.run_model_pipeline(model_input)
            LOGGER.info(
                "Running partial Spearman permutation audits and forward "
                "draft-class validation."
            )
            audit_result = statistical_audit.run_statistical_audit(model_input)
            partial_correlations = audit_result.partial_correlations
            forward_validation = audit_result.forward_validation

        LOGGER.info(
            "Pipeline completed: %s player summaries; model metrics %s; "
            "statistical audit %s.",
            f"{len(player_summary):,}",
            "available" if model_metrics is not None else "not requested",
            "available" if partial_correlations is not None else "not requested",
        )
        return PipelineResult(
            player_summary=player_summary,
            model_metrics=model_metrics,
            partial_correlations=partial_correlations,
            forward_validation=forward_validation,
            prelaunch_audit=prelaunch_audit,
        )
    except (OSError, TypeError, ValueError) as error:
        LOGGER.error("Full pipeline failed: %s", error)
        raise


def process_tracking(input_file: TextIO, output_file: IO[str]) -> int:
    """Write PLAYER rows with movement metrics; return the number written."""
    reader = csv.DictReader(input_file)
    if reader.fieldnames is None:
        raise ValueError("Input CSV is empty or has no header.")

    fieldnames = [name.strip() for name in reader.fieldnames]
    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("Input CSV contains duplicate column names after trimming.")
    normalized_fields = {
        _normalized_column_name(name): name for name in fieldnames
    }
    player_field = next(
        (
            normalized_fields[_normalized_column_name(candidate)]
            for candidate in ("player_id", "nfl_id", "player_id")
            if _normalized_column_name(candidate) in normalized_fields
        ),
        None,
    )
    missing: list[str] = sorted(
        (set(REQUIRED_COLUMNS) - {"player_id"}) - set(fieldnames)
    )
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    if player_field is None:
        raise ValueError("Input CSV requires a player_id or nfl_id column.")
    entity_field = normalized_fields.get("entitytype")
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
    written_count = 0
    for row_count, row in enumerate(reader, start=1):
        if entity_field is not None:
            entity_type = (row.get(entity_field) or "").strip().upper()
            if not entity_type:
                raise ValueError(
                    f"Input row {row_count} has a missing entity_type linkage label."
                )
            if entity_type != "PLAYER":
                continue
        try:
            raw_game_id = row["game_id"]
            raw_play_id = row["play_id"]
            raw_player_id = row[player_field]
            if not all(
                isinstance(value, str)
                for value in (raw_game_id, raw_play_id, raw_player_id)
            ):
                raise ValueError("game_id, play_id, and player identifier must be strings")
            track: tuple[str, str, str] = (
                raw_game_id.strip(),
                raw_play_id.strip(),
                raw_player_id.strip(),
            )
            frame = int(row["frame_id"])
            x = float(row["x"])
            y = float(row["y"])
            if not all(track):
                raise ValueError("game_id, play_id, and player identifier must be non-empty")
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
        written_count += 1
        previous = (frame, x, y)

    return written_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded tracking-to-model workflow, or prepare a tracking "
            "CSV with the legacy streaming mode."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        help="Tracking CSV (default: data/combine_tracking.csv)",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        nargs="?",
        help="Optional streaming-preparation output in outputs/",
    )
    parser.add_argument(
        "--performance-csv",
        type=Path,
        help="Optional pre-aggregated player-season EPA/YAC CSV.",
    )
    parser.add_argument(
        "--max-tracking-rows",
        type=int,
        default=DEFAULT_MAX_TRACKING_ROWS,
        help=f"Maximum in-memory tracking rows (default: {DEFAULT_MAX_TRACKING_ROWS:,}).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run mock-data tests without reading or writing project data.",
    )
    args = parser.parse_args()

    if args.self_test:
        suite = unittest.TestSuite(
            [
                unittest.defaultTestLoader.loadTestsFromTestCase(_PipelineTest),
                unittest.defaultTestLoader.loadTestsFromTestCase(_FullPipelineTest),
            ]
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    try:
        if args.output_csv is not None:
            if args.input_csv is None:
                parser.error("input_csv is required when output_csv is specified")
            if args.performance_csv is not None:
                parser.error(
                    "--performance-csv is only supported in full-pipeline mode"
                )
            output_path = _resolve_output_path(args.output_csv)
            if args.input_csv.resolve() == output_path:
                raise ValueError("Input and output paths must be different.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with args.input_csv.open(
                    "r",
                    newline="",
                    encoding="utf-8-sig",
                ) as input_file:
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
                        row_count = process_tracking(input_file, output_file.file)
                os.replace(temporary_path, output_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            print(f"Processed {row_count} rows into {output_path}")
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(levelname)s: %(message)s",
            )
            run_full_pipeline(
                tracking_data=args.input_csv,
                performance_data=args.performance_csv,
                max_tracking_rows=args.max_tracking_rows,
            )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

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

    def test_streaming_mode_filters_non_player_entities(self) -> None:
        input_file = StringIO(
            "game_id,play_id,nfl_id,frame_id,x,y,entity_type\n"
            "1,2,3,1,0,0,PLAYER\n"
            "1,2,,1,0,0,BALL\n"
            "1,2,3,2,1,0,PLAYER\n"
        )
        output_file = StringIO()
        self.assertEqual(process_tracking(input_file, output_file), 2)
        self.assertEqual(len(output_file.getvalue().splitlines()), 3)
        self.assertTrue(output_file.getvalue().splitlines()[2].endswith("10.0"))

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


class _FullPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics_path = (
            Path(__file__).resolve().parent.parent
            / "outputs"
            / f".full-pipeline-test-{os.urandom(8).hex()}.csv"
        )

    def tearDown(self) -> None:
        self.metrics_path.unlink(missing_ok=True)

    @staticmethod
    def _mock_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        player_count = 12
        years = [2023, 2024, 2025] * 4
        tracking_rows: list[tuple[int, int, int, int, float, float, int]] = []
        points = ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (2.0, 2.0))
        for player_number, draft_year in enumerate(years, start=1):
            for frame, (x, y) in enumerate(points, start=1):
                tracking_rows.append(
                    (1, 1, player_number, frame, x, y, draft_year)
                )
        tracking = pd.DataFrame(
            tracking_rows,
            columns=(
                "game_id",
                "play_id",
                "player_id",
                "frame_id",
                "x",
                "y",
                "draft_year",
            ),
        )
        movement = np.linspace(0.2, 1.3, player_count)
        performance = pd.DataFrame(
            {
                "nflId": np.arange(1, player_count + 1),
                "season": years,
                "epa": 0.4 * movement + np.linspace(-0.2, 0.2, player_count),
                "yac": 8.0 * movement + np.linspace(-1.0, 1.0, player_count),
            }
        )
        return tracking, performance

    def test_full_pipeline_runs_features_and_nested_cv(self) -> None:
        tracking, performance = self._mock_inputs()
        with patch.object(
            model_integration,
            "MODEL_PIPELINE_OUTPUT_FILENAME",
            self.metrics_path.name,
        ), patch.object(statistical_audit, "_save_csv_atomic", return_value=Path("mock")):
            result = run_full_pipeline(tracking, performance)

        self.assertEqual(len(result.player_summary), 12)
        self.assertIsNotNone(result.model_metrics)
        assert result.model_metrics is not None
        self.assertEqual(
            set(result.model_metrics["target_metric"]),
            {"epa", "yards_after_catch"},
        )
        self.assertIsNotNone(result.partial_correlations)
        self.assertIsNotNone(result.forward_validation)
        assert result.forward_validation is not None
        self.assertIn(2025, result.forward_validation["test_draft_year"].tolist())
        self.assertTrue(self.metrics_path.is_file())

    def test_feature_only_run_does_not_require_outcomes(self) -> None:
        tracking, _ = self._mock_inputs()
        result = run_full_pipeline(tracking)

        self.assertEqual(len(result.player_summary), 12)
        self.assertIsNone(result.model_metrics)

    def test_rejects_play_level_duplicate_outcomes(self) -> None:
        tracking, performance = self._mock_inputs()
        duplicated = pd.concat(
            [performance, performance.iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "one pre-aggregated"):
            run_full_pipeline(tracking, duplicated)

    def test_enforces_tracking_row_limit(self) -> None:
        tracking, _ = self._mock_inputs()
        with self.assertRaisesRegex(ValueError, "memory bound"):
            run_full_pipeline(tracking, max_tracking_rows=10)

    def test_filters_non_player_entities_and_normalizes_nfl_id(self) -> None:
        tracking, _ = self._mock_inputs()
        tracking.rename(columns={"player_id": "nfl_id"}, inplace=True)
        ball = tracking.iloc[[0]].copy()
        ball["nfl_id"] = -1
        tracking["entity_type"] = "PLAYER"
        ball["entity_type"] = "BALL"
        combined = pd.concat([tracking, ball], ignore_index=True)
        prepared = _prepare_tracking_entities(combined)
        self.assertEqual(len(prepared), len(tracking))
        self.assertIn("player_id", prepared.columns)
        self.assertTrue(prepared["entity_type"].eq("PLAYER").all())

    def test_rejects_duplicate_tracking_linkage_key(self) -> None:
        tracking, _ = self._mock_inputs()
        duplicated = pd.concat([tracking, tracking.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate linkage"):
            run_full_pipeline(duplicated)


if __name__ == "__main__":
    raise SystemExit(main())
