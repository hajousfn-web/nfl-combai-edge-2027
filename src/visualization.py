"""Create compact, non-interactive plots from local output CSV files."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAX_PLOT_POINTS = 10_000
CHUNK_SIZE = 50_000
PNG_DPI = 120
EFFICIENCY_COLUMNS = (
    "proprietary_cut_efficiency",
    "average_proprietary_cut_efficiency",
)
PERFORMANCE_COLUMNS = (
    "yards_after_catch",
    "yards_after_catch_season",
    "yac",
    "defensive_stops",
    "stops",
    "performance",
    "performance_metric",
)
INSIGHT_COLUMNS = (
    "outcome_metric",
    "pearson_correlation",
    "ridge_standardized_coefficient",
)

PlotMode = Literal["player", "insights"]


def _outputs_dir() -> Path:
    """Return the resolved project outputs directory."""
    return (Path(__file__).resolve().parent.parent / "outputs").resolve()


def _resolve_output_file(path: str | Path, *, default_name: str) -> Path:
    """Resolve a file argument and reject paths outside outputs/."""
    outputs_dir = _outputs_dir()
    candidate = Path(path) if str(path) else Path(default_name)
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0].casefold() == "outputs":
            candidate = Path(__file__).resolve().parent.parent / candidate
        else:
            candidate = outputs_dir / candidate

    resolved = candidate.resolve()
    if resolved.parent != outputs_dir:
        raise ValueError(f"File must be directly inside the outputs directory: {path}")
    return resolved


def _normalized_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def _match_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {_normalized_name(column): column for column in columns}
    for candidate in candidates:
        match = lookup.get(_normalized_name(candidate))
        if match is not None:
            return match
    return None


def _choose_mode(
    columns: list[str],
    y_column: str | None,
) -> tuple[PlotMode, str, str, str | None]:
    if all(
        _match_column(columns, (candidate,)) is not None
        for candidate in INSIGHT_COLUMNS
    ):
        x = _match_column(columns, ("pearson_correlation",))
        y = _match_column(columns, ("ridge_standardized_coefficient",))
        label = _match_column(columns, ("outcome_metric",))
        if x is None or y is None or label is None:
            raise ValueError("Aggregate insights CSV is missing required columns.")
        return "insights", x, y, label

    x = _match_column(columns, EFFICIENCY_COLUMNS)
    if x is None:
        raise ValueError(
            "CSV must contain proprietary_cut_efficiency (or its player-average "
            "summary) and numeric performance columns."
        )
    y = _match_column(columns, (y_column,)) if y_column else None
    if y_column and y is None:
        raise ValueError(f"Requested performance column was not found: {y_column}")
    if not y_column:
        y = _match_column(columns, PERFORMANCE_COLUMNS)
    if y is None:
        raise ValueError(
            "Could not find a recognized performance column; pass y_column explicitly."
        )
    return "player", x, y, None


def _load_plot_sample(
    csv_path: Path,
    *,
    x_column: str,
    y_column: str,
    label_column: str | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, list[str] | None, int]:
    """Read a bounded deterministic reservoir sample without loading all rows."""
    use_columns = [x_column, y_column]
    if label_column is not None:
        use_columns.append(label_column)

    rng = np.random.default_rng(2027)
    x_values: list[float] = []
    y_values: list[float] = []
    labels: list[str] | None = [] if label_column is not None else None
    valid_seen = 0

    try:
        chunks = pd.read_csv(
            csv_path,
            usecols=use_columns,
            chunksize=CHUNK_SIZE,
            low_memory=True,
        )
        for chunk in chunks:
            x_numeric = pd.to_numeric(chunk[x_column], errors="coerce")
            y_numeric = pd.to_numeric(chunk[y_column], errors="coerce")
            valid = np.isfinite(x_numeric.to_numpy(dtype=float)) & np.isfinite(
                y_numeric.to_numpy(dtype=float)
            )
            valid_indices = np.flatnonzero(valid)
            label_values = chunk[label_column] if label_column is not None else None

            for index in valid_indices:
                valid_seen += 1
                x_value = float(x_numeric.iloc[index])
                y_value = float(y_numeric.iloc[index])
                label_value = (
                    str(label_values.iloc[index]) if label_values is not None else None
                )

                if len(x_values) < max_points:
                    x_values.append(x_value)
                    y_values.append(y_value)
                    if labels is not None and label_value is not None:
                        labels.append(label_value)
                else:
                    replacement = int(rng.integers(valid_seen))
                    if replacement < max_points:
                        x_values[replacement] = x_value
                        y_values[replacement] = y_value
                        if labels is not None and label_value is not None:
                            labels[replacement] = label_value
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"Could not read plot data from {csv_path}: {error}") from error

    if not x_values:
        raise ValueError("CSV has no rows with finite numeric plot values.")
    return (
        np.asarray(x_values),
        np.asarray(y_values),
        labels,
        valid_seen,
    )


def generate_scatter_plot(
    input_path: str | Path = "model_integration_insights.csv",
    output_path: str | Path = "cut_efficiency_vs_performance.png",
    *,
    y_column: str | None = None,
    max_points: int = MAX_PLOT_POINTS,
) -> Path:
    """Plot player-level cut efficiency against performance or insight scores.

    Both CSV paths must resolve directly inside the project ``outputs/``
    directory. Player-level files should contain a cut-efficiency column and a
    performance metric such as ``yards_after_catch`` or ``defensive_stops``.
    The aggregate insights CSV produced by ``model_integration.py`` is also
    supported; in that mode, each outcome is plotted by Pearson correlation
    against its standardized Ridge coefficient.

    Data is read in chunks and reservoir-sampled to cap memory at
    ``max_points`` observations. The source file is never modified.

    Args:
        input_path: CSV filename or path inside ``outputs/``.
        output_path: PNG filename or path inside ``outputs/``.
        y_column: Optional performance-column override for player-level data.
        max_points: Maximum number of valid rows retained for rendering.

    Returns:
        The resolved path of the saved PNG image.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
        ValueError: If a path, column, or numeric sample is invalid.
        OSError: If the PNG cannot be written.
    """
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 1:
        raise ValueError("max_points must be a positive integer.")

    csv_path = _resolve_output_file(input_path, default_name="model_integration_insights.csv")
    png_path = _resolve_output_file(output_path, default_name="cut_efficiency_vs_performance.png")
    if csv_path == png_path:
        raise ValueError("Input CSV and output PNG paths must be different.")
    if csv_path.suffix.casefold() != ".csv":
        raise ValueError("Input file must have a .csv extension.")
    if png_path.suffix.casefold() != ".png":
        raise ValueError("Output image must have a .png extension.")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")

    try:
        columns = list(pd.read_csv(csv_path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"Could not read CSV header from {csv_path}: {error}") from error

    mode, x_column, y_plot_column, label_column = _choose_mode(columns, y_column)
    x_values, y_values, labels, valid_count = _load_plot_sample(
        csv_path,
        x_column=x_column,
        y_column=y_plot_column,
        label_column=label_column,
        max_points=max_points,
    )

    figure, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    temporary_path: Path | None = None
    try:
        axis.scatter(
            x_values,
            y_values,
            s=20,
            alpha=0.72,
            color="#176b87",
            edgecolors="none",
        )
        if mode == "insights":
            axis.set_xlabel("Pearson correlation with cut efficiency")
            axis.set_ylabel("Standardized Ridge coefficient")
            axis.set_title("Cut-efficiency signal across outcomes")
            if labels is not None:
                for index, label in enumerate(labels[:20]):
                    axis.annotate(
                        label,
                        (x_values[index], y_values[index]),
                        xytext=(4, 3),
                        textcoords="offset points",
                        fontsize=7,
                    )
        else:
            axis.set_xlabel(x_column.replace("_", " ").title())
            axis.set_ylabel(y_plot_column.replace("_", " ").title())
            axis.set_title(f"Cut efficiency vs. {y_plot_column.replace('_', ' ')}")
        axis.grid(True, linewidth=0.5, alpha=0.25)
        with tempfile.NamedTemporaryFile(
            dir=png_path.parent,
            prefix=".visualization-",
            suffix=".png",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        figure.savefig(
            temporary_path,
            format="png",
            dpi=PNG_DPI,
            bbox_inches="tight",
            metadata={"Software": "NFL CombAI Edge"},
        )
        os.replace(temporary_path, png_path)
    except OSError as error:
        raise OSError(f"Could not save plot to {png_path}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        plt.close(figure)

    print(f"Plot mode: {mode}")
    print(f"Valid observations: {valid_count:,}; plotted: {len(x_values):,}")
    print(f"PNG saved to: {png_path}")
    return png_path


class _VisualizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.outputs_dir = _outputs_dir()
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.input_path = self.outputs_dir / ".visualization-test.csv"
        self.output_path = self.outputs_dir / ".visualization-test.png"
        self.input_path.unlink(missing_ok=True)
        self.output_path.unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.input_path.unlink(missing_ok=True)
        self.output_path.unlink(missing_ok=True)

    def test_player_level_scatter_is_saved_as_png(self) -> None:
        pd.DataFrame(
            {
                "proprietary_cut_efficiency": [0.5, 0.8, 1.0, 1.2],
                "yards_after_catch": [2, 5, 8, 11],
            }
        ).to_csv(self.input_path, index=False)

        saved = generate_scatter_plot(
            self.input_path,
            self.output_path,
            max_points=3,
        )

        self.assertEqual(saved, self.output_path)
        self.assertTrue(saved.is_file())
        self.assertEqual(saved.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertLess(saved.stat().st_size, 250_000)

    def test_aggregate_insights_scatter_is_supported(self) -> None:
        pd.DataFrame(
            {
                "outcome_metric": ["yards_after_catch", "defensive_stops"],
                "pearson_correlation": [0.42, -0.1],
                "ridge_standardized_coefficient": [0.31, -0.08],
            }
        ).to_csv(self.input_path, index=False)

        saved = generate_scatter_plot(self.input_path, self.output_path)

        self.assertTrue(saved.is_file())
        self.assertEqual(saved.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_rejects_paths_outside_outputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "outputs directory"):
            generate_scatter_plot("..\\data\\tracking.csv", self.output_path)


def main() -> int:
    """Run validation tests or render a chart from a local outputs CSV."""
    parser = argparse.ArgumentParser(
        description="Render a compact scatter plot from a CSV in outputs/."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="model_integration_insights.csv",
        help="Input CSV path inside outputs/ (default: model integration insights)",
    )
    parser.add_argument(
        "--output",
        default="cut_efficiency_vs_performance.png",
        help="PNG filename or path inside outputs/",
    )
    parser.add_argument(
        "--y-column",
        help="Performance metric column for player-level CSV data",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=MAX_PLOT_POINTS,
        help=f"Maximum rows to plot (default: {MAX_PLOT_POINTS})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run standalone mock-data tests and remove their temporary artifacts",
    )
    args = parser.parse_args()

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(_VisualizationTest)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    try:
        generate_scatter_plot(
            args.input_path,
            args.output,
            y_column=args.y_column,
            max_points=args.max_points,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
