"""Create publication-ready, non-interactive NFL tracking visualizations."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


MAX_PLOT_POINTS = 10_000
CHUNK_SIZE = 50_000
MAX_INPUT_ROWS = 250_000
PNG_DPI = 240
PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "dark": "#263238",
    "muted": "#66757F",
    "grid": "#D9E1E5",
}
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
PREDICTION_COLUMN_CANDIDATES = {
    "epa": ("predicted_epa", "epa_prediction", "epa_pred"),
    "yards_after_catch": (
        "predicted_yac",
        "predicted_yards_after_catch",
        "yards_after_catch_prediction",
        "yac_prediction",
        "yac_pred",
    ),
}
ACTUAL_COLUMN_CANDIDATES = {
    "epa": ("actual_epa", "epa_actual", "epa"),
    "yards_after_catch": (
        "actual_yards_after_catch",
        "yards_after_catch_actual",
        "actual_yac",
        "yards_after_catch",
        "yac",
    ),
}

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


def _apply_publication_style(axis: Axes) -> None:
    """Apply restrained spines, labels, and grid styling to one axes."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(PALETTE["muted"])
    axis.spines["bottom"].set_color(PALETTE["muted"])
    axis.tick_params(colors=PALETTE["dark"], labelsize=9)
    axis.set_xlabel(axis.get_xlabel(), fontsize=10)
    axis.set_ylabel(axis.get_ylabel(), fontsize=10)
    axis.set_title(axis.get_title(), fontsize=12, fontweight="bold")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.7, alpha=0.8)


def _load_input_frame(
    source: pd.DataFrame | str | Path,
    *,
    name: str,
    max_rows: int = MAX_INPUT_ROWS,
) -> pd.DataFrame:
    """Load a DataFrame or an outputs-only CSV under a strict row bound."""
    if isinstance(source, pd.DataFrame):
        frame = source
    else:
        path = _resolve_output_file(source, default_name="")
        if path.suffix.casefold() != ".csv":
            raise ValueError(f"{name} input must be a CSV file.")
        if not path.is_file():
            raise FileNotFoundError(f"{name} CSV does not exist: {path}")
        try:
            frame = pd.read_csv(path, nrows=max_rows + 1, low_memory=True)
        except (OSError, pd.errors.ParserError, UnicodeError, ValueError) as error:
            raise ValueError(f"Could not read {name} CSV {path}: {error}") from error

    if not frame.columns.is_unique:
        raise ValueError(f"{name} data must not contain duplicate columns.")
    if len(frame) > max_rows:
        raise ValueError(
            f"{name} data exceeds the {max_rows:,}-row visualization limit."
        )
    if frame.empty:
        raise ValueError(f"{name} data contains no rows.")
    return frame.copy(deep=False)


def _save_figure_atomic(figure: Figure, output_path: Path) -> Path:
    """Save one compact PNG atomically and close the figure on every path."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
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
        os.replace(temporary_path, output_path)
    except OSError as error:
        raise OSError(f"Could not save plot to {output_path}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        plt.close(figure)
    return output_path


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

    figure, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    try:
        axis.scatter(
            x_values,
            y_values,
            s=25,
            alpha=0.78,
            color=PALETTE["blue"],
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
                        fontsize=8,
                        color=PALETTE["dark"],
                    )
        else:
            axis.set_xlabel(x_column.replace("_", " ").title())
            axis.set_ylabel(y_plot_column.replace("_", " ").title())
            axis.set_title(f"Cut efficiency vs. {y_plot_column.replace('_', ' ')}")
        axis.text(
            0.99,
            0.02,
            f"n = {valid_count:,}" + (
                f" (sampled {len(x_values):,})"
                if valid_count > len(x_values)
                else ""
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=PALETTE["muted"],
        )
        _apply_publication_style(axis)
        _save_figure_atomic(figure, png_path)
    finally:
        if plt.fignum_exists(figure.number):
            plt.close(figure)

    print(f"Plot mode: {mode}")
    print(f"Valid observations: {valid_count:,}; plotted: {len(x_values):,}")
    print(f"PNG saved to: {png_path}")
    return png_path


def _bootstrap_mean_interval(
    residuals: np.ndarray,
    *,
    seed: int = 2027,
    replicates: int = 1_000,
) -> tuple[float, float]:
    """Estimate a percentile 95% CI with at most 5,000 paired residuals."""
    if residuals.size < 2:
        raise ValueError("At least two paired observations are required for a CI.")
    rng = np.random.default_rng(seed)
    bootstrap_source = residuals
    if residuals.size > 5_000:
        bootstrap_source = rng.choice(residuals, size=5_000, replace=False)
    sample_indices = rng.integers(
        0,
        len(bootstrap_source),
        size=(replicates, len(bootstrap_source)),
    )
    bootstrap_means = bootstrap_source[sample_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return float(lower), float(upper)


def _target_columns_for_predictions(
    frame: pd.DataFrame,
    target: str,
) -> tuple[str, str, str | None, str | None] | None:
    """Resolve actual/predicted and optional per-row interval columns."""
    columns = [str(column) for column in frame.columns]
    actual_column = _match_column(columns, ACTUAL_COLUMN_CANDIDATES[target])
    predicted_column = _match_column(columns, PREDICTION_COLUMN_CANDIDATES[target])
    if actual_column is None or predicted_column is None:
        return None

    aliases = ("yac", "yards_after_catch") if target == "yards_after_catch" else ("epa",)
    lower_candidates = tuple(
        candidate
        for alias in aliases
        for candidate in (
            f"{alias}_prediction_lower",
            f"{alias}_ci_lower",
            f"{alias}_lower",
        )
    )
    upper_candidates = tuple(
        candidate
        for alias in aliases
        for candidate in (
            f"{alias}_prediction_upper",
            f"{alias}_ci_upper",
            f"{alias}_upper",
        )
    )
    lower_column = _match_column(columns, lower_candidates)
    upper_column = _match_column(columns, upper_candidates)
    if (lower_column is None) != (upper_column is None):
        raise ValueError(
            f"{target} prediction intervals require both lower and upper bounds."
        )
    return actual_column, predicted_column, lower_column, upper_column


def _plot_prediction_comparison(
    frame: pd.DataFrame,
    *,
    target: str,
    output_path: Path,
    max_points: int,
) -> Path:
    """Plot actual versus out-of-sample predictions with honest uncertainty."""
    resolved = _target_columns_for_predictions(frame, target)
    if resolved is None:
        raise ValueError(
            f"Prediction data must contain actual and predicted {target} columns."
        )
    actual_column, predicted_column, lower_column, upper_column = resolved
    required_columns = [actual_column, predicted_column]
    if lower_column is not None and upper_column is not None:
        required_columns.extend([lower_column, upper_column])
    plot_data = frame.loc[:, required_columns].apply(pd.to_numeric, errors="coerce")
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    plot_data.dropna(subset=[actual_column, predicted_column], inplace=True)
    if len(plot_data) < 2:
        raise ValueError(
            f"At least two paired finite actual/predicted {target} rows are required."
        )

    residuals = (
        plot_data[predicted_column].to_numpy(dtype=float)
        - plot_data[actual_column].to_numpy(dtype=float)
    )
    if lower_column is not None and upper_column is not None:
        all_lower = plot_data[lower_column].to_numpy(dtype=float)
        all_upper = plot_data[upper_column].to_numpy(dtype=float)
        supplied_bounds = np.isfinite(all_lower) & np.isfinite(all_upper)
        if np.any(supplied_bounds & (all_lower > all_upper)):
            raise ValueError(
                f"{target} prediction lower bounds cannot exceed upper bounds."
            )
        all_predictions = plot_data[predicted_column].to_numpy(dtype=float)
        if np.any(
            supplied_bounds
            & ((all_lower > all_predictions) | (all_predictions > all_upper))
        ):
            raise ValueError(
                f"{target} prediction intervals must enclose each prediction."
            )
    ci_lower, ci_upper = _bootstrap_mean_interval(residuals)
    full_count = len(plot_data)
    if full_count > max_points:
        plot_data = plot_data.sample(
            n=max_points,
            random_state=2027,
        ).sort_index()
    actual = plot_data[actual_column].to_numpy(dtype=float)
    predicted = plot_data[predicted_column].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    try:
        if lower_column is not None and upper_column is not None:
            lower = plot_data[lower_column].to_numpy(dtype=float)
            upper = plot_data[upper_column].to_numpy(dtype=float)
            supplied_bounds = np.isfinite(lower) & np.isfinite(upper)
            valid_intervals = (
                supplied_bounds
            )
            if np.any(valid_intervals):
                axis.errorbar(
                    actual[valid_intervals],
                    predicted[valid_intervals],
                    yerr=np.vstack(
                        (
                            predicted[valid_intervals] - lower[valid_intervals],
                            upper[valid_intervals] - predicted[valid_intervals],
                        )
                    ),
                    fmt="none",
                    ecolor=PALETTE["muted"],
                    elinewidth=0.7,
                    capsize=2,
                    alpha=0.5,
                    zorder=1,
                )

        axis.scatter(
            actual,
            predicted,
            s=27,
            color=PALETTE["blue"],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.35,
            zorder=2,
        )
        value_min = float(min(actual.min(), predicted.min()))
        value_max = float(max(actual.max(), predicted.max()))
        padding = (value_max - value_min) * 0.04 or max(abs(value_min), 1.0) * 0.04
        axis.plot(
            [value_min - padding, value_max + padding],
            [value_min - padding, value_max + padding],
            linestyle="--",
            linewidth=1.1,
            color=PALETTE["orange"],
            label="Perfect prediction",
        )
        axis.set_xlim(value_min - padding, value_max + padding)
        axis.set_ylim(value_min - padding, value_max + padding)
        target_label = "Yards after catch" if target == "yards_after_catch" else "EPA"
        axis.set_xlabel(f"Actual {target_label}")
        axis.set_ylabel(f"Predicted {target_label}")
        axis.set_title(f"{target_label}: predicted vs. actual")
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        mae = float(np.mean(np.abs(residuals)))
        interval_label = (
            "Per-player bounds: provided prediction intervals"
            if lower_column is not None
            else "Per-player prediction intervals not supplied"
        )
        axis.text(
            0.03,
            0.97,
            (
                f"n = {full_count:,}  |  RMSE = {rmse:.3g}  |  MAE = {mae:.3g}\n"
                f"Mean error 95% paired-bootstrap CI: [{ci_lower:.3g}, {ci_upper:.3g}]\n"
                f"{interval_label}"
            ),
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color=PALETTE["dark"],
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": PALETTE["grid"],
                "alpha": 0.95,
            },
        )
        axis.legend(frameon=False, loc="lower right", fontsize=8)
        _apply_publication_style(axis)
        return _save_figure_atomic(figure, output_path)
    finally:
        if plt.fignum_exists(figure.number):
            plt.close(figure)


def _plot_kinematics(
    source: pd.DataFrame | str | Path,
    *,
    output_path: Path,
    max_points: int,
) -> Path:
    """Render one representative 10 Hz velocity profile and sharp-cut scores."""
    frame = _load_input_frame(source, name="Kinematics")
    frame = frame.reset_index(drop=True)
    if {"vx", "vy", "speed"}.issubset(frame.columns):
        features = frame.copy(deep=False)
    elif {"game_id", "play_id", "player_id", "frame_id", "x", "y"}.issubset(
        frame.columns
    ):
        from_feature_engineering = _feature_engineering_module()
        features = from_feature_engineering.run_feature_pipeline(frame)
    else:
        raise ValueError(
            "Kinematics input needs vx/vy/speed features or raw tracking columns."
        )

    required = {"player_id", "frame_id", "speed"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(
            "Kinematics data is missing columns: " + ", ".join(missing)
        )
    frame_id = pd.to_numeric(features["frame_id"], errors="coerce")
    speed = pd.to_numeric(features["speed"], errors="coerce")
    finite = np.isfinite(frame_id.to_numpy(dtype=float)) & np.isfinite(
        speed.to_numpy(dtype=float)
    )
    features = features.loc[finite].copy()
    features["frame_id"] = frame_id.loc[finite]
    features["speed"] = speed.loc[finite]
    if features.empty:
        raise ValueError("Kinematics data has no finite frame/speed observations.")

    track_columns = [
        column
        for column in ("game_id", "play_id", "player_id")
        if column in features.columns
    ]
    chosen_track = (
        features.sort_values([*track_columns, "frame_id"], kind="mergesort")
        .groupby(track_columns, sort=False, dropna=False, observed=True)
        .head(1)
        .iloc[0]
    )
    chosen_mask = pd.Series(True, index=features.index)
    for column in track_columns:
        chosen_mask &= features[column].eq(chosen_track[column])
    profile = features.loc[chosen_mask].sort_values("frame_id", kind="mergesort")
    if len(profile) > max_points:
        positions = np.linspace(0, len(profile) - 1, max_points, dtype=int)
        profile = profile.iloc[positions]

    times = (profile["frame_id"].to_numpy(dtype=float) - float(profile["frame_id"].iloc[0])) / 10.0
    figure, (speed_axis, cut_axis) = plt.subplots(
        2,
        1,
        figsize=(8.0, 6.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    try:
        speed_axis.plot(
            times,
            profile["speed"].to_numpy(dtype=float),
            color=PALETTE["blue"],
            linewidth=1.8,
            label="Smoothed speed",
        )
        if "is_sharp_cut" in profile.columns:
            sharp = profile["is_sharp_cut"].fillna(False).astype(bool)
            speed_axis.scatter(
                times[sharp.to_numpy()],
                profile.loc[sharp, "speed"],
                s=42,
                color=PALETTE["orange"],
                edgecolors="white",
                linewidths=0.6,
                label="Sharp cut (>45°)",
                zorder=3,
            )
        speed_axis.set_title(
            f"10 Hz velocity profile · player {chosen_track['player_id']}"
        )
        speed_axis.set_xlabel("Elapsed time (s)")
        speed_axis.set_ylabel("Speed (yd/s)")
        speed_axis.legend(frameon=False, ncol=2, fontsize=8)
        _apply_publication_style(speed_axis)

        if {
            "direction_change_degrees",
            "proprietary_cut_efficiency",
        }.issubset(profile.columns):
            cut_data = profile.loc[
                profile.get(
                    "is_sharp_cut",
                    pd.Series(False, index=profile.index),
                ).fillna(False).astype(bool)
            ]
            angles = pd.to_numeric(
                cut_data["direction_change_degrees"],
                errors="coerce",
            )
            efficiencies = pd.to_numeric(
                cut_data["proprietary_cut_efficiency"],
                errors="coerce",
            )
            valid = np.isfinite(angles.to_numpy(dtype=float)) & np.isfinite(
                efficiencies.to_numpy(dtype=float)
            )
            cut_axis.scatter(
                angles.loc[valid],
                efficiencies.loc[valid],
                s=42,
                color=PALETTE["green"],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.5,
            )
            cut_axis.axvline(
                45.0,
                linestyle="--",
                linewidth=1.0,
                color=PALETTE["orange"],
                label="45° threshold",
            )
            if not np.any(valid):
                cut_axis.text(
                    0.5,
                    0.5,
                    "No sharp cuts with a defined efficiency in this track",
                    transform=cut_axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=PALETTE["muted"],
                )
        else:
            cut_axis.text(
                0.5,
                0.5,
                "Cut angle/efficiency columns are not available",
                transform=cut_axis.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=PALETTE["muted"],
            )
        cut_axis.set_title("Sharp-cut speed retention")
        cut_axis.set_xlabel("Direction change (degrees)")
        cut_axis.set_ylabel("Exit speed / entry speed")
        if {
            "direction_change_degrees",
            "proprietary_cut_efficiency",
        }.issubset(profile.columns):
            cut_axis.legend(frameon=False, fontsize=8)
        _apply_publication_style(cut_axis)
        return _save_figure_atomic(figure, output_path)
    finally:
        if plt.fignum_exists(figure.number):
            plt.close(figure)


def _feature_engineering_module():
    """Import feature engineering for both script and package execution."""
    if __package__:
        from . import feature_engineering
    else:
        import feature_engineering
    return feature_engineering


def generate_project_visualizations(
    tracking_data: pd.DataFrame | str | Path | None = None,
    predictions_data: pd.DataFrame | str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    max_points: int = MAX_PLOT_POINTS,
) -> dict[str, Path]:
    """Generate available project plots and return their output paths.

    ``tracking_data`` may be raw tracking or feature-engineered observations.
    ``predictions_data`` must contain paired, genuinely out-of-sample actual
    and predicted EPA/YAC values. Optional ``epa_lower``/``epa_upper`` (or
    corresponding YAC interval fields) add per-observation prediction bounds;
    the plotted mean-error 95% interval is bootstrapped from paired residuals.
    No interval is invented when per-player prediction bounds are absent.

    CSV inputs are restricted to ``outputs/`` and capped at 250,000 rows.
    DataFrames are accepted for in-memory use. Figures are saved atomically
    only directly under the project ``outputs/`` directory.
    """
    if (
        isinstance(max_points, bool)
        or not isinstance(max_points, int)
        or max_points < 2
    ):
        raise ValueError("max_points must be an integer of at least 2.")

    outputs_dir = _outputs_dir()
    if output_dir is not None:
        requested_output = Path(output_dir)
        if not requested_output.is_absolute():
            requested_output = Path(__file__).resolve().parent.parent / requested_output
        resolved_output = requested_output.resolve()
        if resolved_output != outputs_dir:
            raise ValueError("All generated images must be saved directly in outputs/.")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if tracking_data is None:
        default_tracking = outputs_dir / "tracking_features.csv"
        if default_tracking.is_file():
            tracking_data = default_tracking
    if predictions_data is None:
        default_predictions = outputs_dir / "model_predictions.csv"
        if default_predictions.is_file():
            predictions_data = default_predictions
    if tracking_data is None and predictions_data is None:
        raise ValueError(
            "Provide tracking_data and/or predictions_data; no default visualization "
            "inputs were found in outputs/."
        )

    outputs: dict[str, Path] = {}
    if tracking_data is not None:
        outputs["kinematics"] = _plot_kinematics(
            tracking_data,
            output_path=outputs_dir / "kinematics_and_cut_efficiency.png",
            max_points=max_points,
        )
    if predictions_data is not None:
        predictions = _load_input_frame(predictions_data, name="Prediction")
        for target, output_name in (
            ("epa", "epa_actual_vs_predicted.png"),
            ("yards_after_catch", "yac_actual_vs_predicted.png"),
        ):
            if _target_columns_for_predictions(predictions, target) is not None:
                outputs[target] = _plot_prediction_comparison(
                    predictions,
                    target=target,
                    output_path=outputs_dir / output_name,
                    max_points=max_points,
                )
        if not any(key in outputs for key in ("epa", "yards_after_catch")):
            raise ValueError(
                "Prediction data must contain actual and predicted EPA and/or YAC."
            )

    for name, path in outputs.items():
        print(f"{name} visualization saved to: {path}")
    return outputs


def _load_optional_audit_frame(
    source: pd.DataFrame | str | Path | None,
    *,
    default_name: str,
    label: str,
) -> pd.DataFrame | None:
    if source is None:
        candidate = _outputs_dir() / default_name
        if not candidate.is_file():
            return None
        source = candidate
    return _load_input_frame(source, name=label)


def _empty_state_message(axis: Axes, message: str) -> None:
    axis.text(
        0.5,
        0.5,
        message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color=PALETTE["muted"],
        fontsize=10,
        wrap=True,
    )
    axis.set_axis_off()


def _plot_partial_spearman_windows(
    frame: pd.DataFrame | None,
    *,
    output_path: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    try:
        if frame is None or frame.empty:
            _empty_state_message(
                axis,
                "No partial-Spearman audit data available.\n"
                "Run the pipeline with verified player-season outcomes.",
            )
            figure.suptitle(
                "Partial Spearman by validation window",
                fontsize=15,
                fontweight="bold",
                color=PALETTE["dark"],
            )
            return _save_figure_atomic(figure, output_path)

        required = {
            "validation_window",
            "target_metric",
            "partial_spearman_rho",
            "permutation_p_value",
            "sample_count",
            "status",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                "Partial-Spearman audit is missing columns: " + ", ".join(missing)
            )
        usable = frame.loc[
            frame["status"].eq("ok")
            & pd.to_numeric(frame["partial_spearman_rho"], errors="coerce").notna()
        ].copy()
        windows = frame["validation_window"].astype(str).drop_duplicates().tolist()
        targets = frame["target_metric"].astype(str).drop_duplicates().tolist()
        if usable.empty:
            statuses = ", ".join(
                f"{row.validation_window}: {row.status}"
                for row in frame[["validation_window", "status"]]
                .drop_duplicates()
                .itertuples(index=False)
            )
            _empty_state_message(
                axis,
                "No window has enough valid observations for an estimate.\n"
                + statuses,
            )
        else:
            x_positions = np.arange(len(windows), dtype=float)
            offsets = np.linspace(-0.18, 0.18, max(1, len(targets)))
            for target_index, target in enumerate(targets):
                selected = usable.loc[usable["target_metric"].astype(str).eq(target)]
                window_indices = {
                    window: index for index, window in enumerate(windows)
                }
                xs = [
                    window_indices[str(window)]
                    + offsets[target_index]
                    for window in selected["validation_window"]
                ]
                ys = pd.to_numeric(
                    selected["partial_spearman_rho"], errors="coerce"
                ).to_numpy(dtype=float)
                p_values = pd.to_numeric(
                    selected["permutation_p_value"], errors="coerce"
                ).to_numpy(dtype=float)
                sizes = pd.to_numeric(
                    selected["sample_count"], errors="coerce"
                ).fillna(0).to_numpy(dtype=float)
                color = (
                    PALETTE["blue"]
                    if target_index % 2 == 0
                    else PALETTE["orange"]
                )
                axis.scatter(
                    xs,
                    ys,
                    s=58,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    label=target.replace("_", " ").upper(),
                    zorder=3,
                )
                for x_value, y_value, p_value, count in zip(
                    xs, ys, p_values, sizes
                ):
                    axis.annotate(
                        f"n={int(count)}, p={p_value:.2g}",
                        (x_value, y_value),
                        xytext=(0, 9),
                        textcoords="offset points",
                        ha="center",
                        fontsize=8,
                        color=PALETTE["muted"],
                    )
            axis.axhline(0, color=PALETTE["dark"], linewidth=0.9, alpha=0.65)
            axis.set_xticks(x_positions, windows, rotation=15, ha="right")
            axis.set_ylim(-1.05, 1.05)
            axis.set_ylabel("Partial Spearman ρ")
            axis.set_xlabel("Audit window (overall and draft-class strata)")
            axis.legend(frameon=False, ncols=min(2, len(targets)))
            _apply_publication_style(axis)

        figure.suptitle(
            "Partial Spearman by validation window",
            fontsize=15,
            fontweight="bold",
            color=PALETTE["dark"],
        )
        figure.text(
            0.5,
            -0.025,
            "Permutation inference is exploratory; per-class estimates may be unavailable "
            "when sample size or controls are insufficient.",
            ha="center",
            fontsize=8,
            color=PALETTE["muted"],
        )
        return _save_figure_atomic(figure, output_path)
    finally:
        if plt.fignum_exists(figure.number):
            plt.close(figure)


def _plot_forward_validation_deltas(
    frame: pd.DataFrame | None,
    *,
    output_path: Path,
) -> Path:
    figure, axes = plt.subplots(
        1, 3, figsize=(13.0, 5.4), constrained_layout=True
    )
    try:
        metrics = (
            ("rmse", "RMSE Δ (Base − Base+CE)", True),
            ("mae", "MAE Δ (Base − Base+CE)", True),
            ("r2", "R² Δ (Base+CE − Base)", False),
        )
        if frame is None or frame.empty:
            for axis, (_, title, _) in zip(axes, metrics):
                axis.set_title(title)
                _empty_state_message(
                    axis,
                    "No forward-validation results.\n"
                    "Run the pipeline with verified labels.",
                )
            figure.suptitle(
                "Forward draft-class validation: Base vs. Base + CE",
                fontsize=15,
                fontweight="bold",
                color=PALETTE["dark"],
            )
            return _save_figure_atomic(figure, output_path)

        required = {
            "target_metric",
            "test_draft_year",
            "model_variant",
            "status",
            "rmse",
            "mae",
            "r2",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                "Forward-validation data is missing columns: " + ", ".join(missing)
            )
        clean = frame.loc[frame["status"].eq("ok")].copy()
        paired = clean.pivot_table(
            index=["target_metric", "test_draft_year"],
            columns="model_variant",
            values=["rmse", "mae", "r2"],
            aggfunc="first",
        )
        has_variants = (
            (not paired.empty)
            and ("base" in paired.columns.get_level_values(1))
            and ("base_plus_metric" in paired.columns.get_level_values(1))
        )
        if not has_variants:
            for axis, (_, title, _) in zip(axes, metrics):
                axis.set_title(title)
                _empty_state_message(
                    axis,
                    "No paired Base and Base+CE folds.\n"
                    "Check fold sample counts and model statuses.",
                )
        else:
            categories = [
                f"{metric.upper()} · {int(year)}"
                for metric, year in paired.index
            ]
            x_values = np.arange(len(categories))
            for axis, (metric, title, lower_is_better) in zip(axes, metrics):
                base_values = paired[(metric, "base")].to_numpy(dtype=float)
                enhanced_values = paired[(metric, "base_plus_metric")].to_numpy(
                    dtype=float
                )
                delta = (
                    base_values - enhanced_values
                    if lower_is_better
                    else enhanced_values - base_values
                )
                finite = np.isfinite(delta)
                if not finite.any():
                    axis.set_title(title)
                    _empty_state_message(
                        axis,
                        "No finite paired metric scores in valid folds.",
                    )
                    continue
                metric_positions = x_values[finite]
                metric_categories = np.asarray(categories)[finite]
                delta = delta[finite]
                colors = [
                    PALETTE["green"] if value > 0 else PALETTE["orange"]
                    for value in delta
                ]
                axis.bar(metric_positions, delta, color=colors, width=0.68)
                axis.axhline(0, color=PALETTE["dark"], linewidth=0.9)
                axis.set_xticks(
                    metric_positions, metric_categories, rotation=45, ha="right"
                )
                axis.set_title(title)
                axis.set_ylabel("Positive favors Base+CE")
                _apply_publication_style(axis)
                if len(delta):
                    padding = max(float(np.nanmax(np.abs(delta))) * 0.08, 0.01)
                    for position, value in zip(metric_positions, delta):
                        axis.annotate(
                            f"{value:+.3g}",
                            (position, value),
                            xytext=(0, 4 if value >= 0 else -12),
                            textcoords="offset points",
                            ha="center",
                            fontsize=8,
                        )
                    axis.set_ylim(
                        min(0.0, float(np.nanmin(delta)) - padding),
                        max(0.0, float(np.nanmax(delta)) + padding),
                    )
        figure.suptitle(
            "Forward draft-class validation: Base vs. Base + CE",
            fontsize=15,
            fontweight="bold",
            color=PALETTE["dark"],
        )
        figure.text(
            0.5,
            -0.025,
            "Positive deltas favor Base+CE; scores are out-of-class only when both "
            "paired fold variants are valid.",
            ha="center",
            fontsize=8,
            color=PALETTE["muted"],
        )
        return _save_figure_atomic(figure, output_path)
    finally:
        if plt.fignum_exists(figure.number):
            plt.close(figure)


def _plot_fsm_sequence(*, output_path: Path) -> Path:
    """Render all reference states and manual-reset paths; illustrative only."""
    nodes = {
        "SAFE": (0.0, 1.0, PALETTE["green"]),
        "WARNING": (2.2, 1.0, PALETTE["yellow"]),
        "DANGER": (4.4, 1.0, PALETTE["orange"]),
        "EMERGENCY_STOP": (6.6, 1.0, PALETTE["purple"]),
        "SENSOR_FAULT": (2.2, -0.6, PALETTE["sky"]),
        "FAIL_SAFE_LOCKED": (4.4, -0.6, PALETTE["dark"]),
    }
    transitions = [
        ("SAFE", "WARNING", "escalate"),
        ("WARNING", "DANGER", "escalate"),
        ("DANGER", "EMERGENCY_STOP", "latch"),
        ("SAFE", "SENSOR_FAULT", "invalid sample"),
        ("SENSOR_FAULT", "FAIL_SAFE_LOCKED", "system fault"),
        ("EMERGENCY_STOP", "SAFE", "manual reset"),
        ("SENSOR_FAULT", "SAFE", "manual reset"),
        ("FAIL_SAFE_LOCKED", "SAFE", "manual reset"),
    ]
    figure, axis = plt.subplots(figsize=(11.2, 5.2), constrained_layout=True)
    try:
        for source, target, label in transitions:
            source_x, source_y, _ = nodes[source]
            target_x, target_y, _ = nodes[target]
            is_reset = label == "manual reset"
            arc = 0.25 if (source, target) in {
                ("FAIL_SAFE_LOCKED", "SAFE"),
                ("EMERGENCY_STOP", "SAFE"),
            } else 0.0
            axis.add_patch(
                FancyArrowPatch(
                    (source_x + 0.46, source_y),
                    (target_x - 0.48, target_y),
                    connectionstyle=f"arc3,rad={arc}",
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.5,
                    linestyle="--" if is_reset else "-",
                    color=PALETTE["blue"] if is_reset else PALETTE["muted"],
                    zorder=1,
                )
            )
            midpoint_x = (source_x + target_x) / 2
            midpoint_y = (source_y + target_y) / 2 + (0.12 if is_reset else 0.16)
            axis.text(
                midpoint_x,
                midpoint_y,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=PALETTE["blue"] if is_reset else PALETTE["muted"],
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                },
                zorder=3,
            )
        for name, (x_value, y_value, color) in nodes.items():
            axis.add_patch(
                FancyBboxPatch(
                    (x_value - 0.48, y_value - 0.22),
                    0.96,
                    0.44,
                    boxstyle="round,pad=0.06",
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.2,
                    zorder=2,
                )
            )
            axis.text(
                x_value,
                y_value,
                name,
                ha="center",
                va="center",
                fontsize=7 if len(name) > 12 else 8,
                fontweight="bold",
                color="white" if name in {"DANGER", "EMERGENCY_STOP", "FAIL_SAFE_LOCKED"} else PALETTE["dark"],
                zorder=3,
            )
        axis.set_xlim(-0.9, 7.55)
        axis.set_ylim(-1.15, 1.7)
        axis.set_axis_off()
        axis.set_title(
            "Edge FSM state transitions — illustrative contract",
            fontsize=14,
            fontweight="bold",
            color=PALETTE["dark"],
        )
        figure.text(
            0.5,
            0.005,
            "Illustration only: no measured risk scores or approved thresholds. "
            "Latched states clear only by explicit software manual reset.",
            ha="center",
            fontsize=8,
            color=PALETTE["muted"],
        )
        return _save_figure_atomic(figure, output_path)
    finally:
        if plt.fignum_exists(figure.number):
            plt.close(figure)


def generate_audit_visualizations(
    partial_spearman_data: pd.DataFrame | str | Path | None = None,
    forward_validation_data: pd.DataFrame | str | Path | None = None,
) -> dict[str, Path]:
    """Generate two evidence-driven audit charts plus an FSM contract figure.

    Audit charts show results only when supplied or present in ``outputs/``;
    otherwise they contain an explicit no-data message. The FSM sequence is
    always labeled illustrative and contains no risk thresholds.
    """
    outputs_dir = _outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    partial = _load_optional_audit_frame(
        partial_spearman_data,
        default_name="partial_spearman_permutation_audit.csv",
        label="Partial-Spearman audit",
    )
    forward = _load_optional_audit_frame(
        forward_validation_data,
        default_name="forward_draft_class_validation.csv",
        label="Forward-validation audit",
    )
    results = {
        "partial_spearman": _plot_partial_spearman_windows(
            partial,
            output_path=outputs_dir / "partial_spearman_validation_windows.png",
        ),
        "forward_validation_delta": _plot_forward_validation_deltas(
            forward,
            output_path=outputs_dir / "forward_draft_class_performance_deltas.png",
        ),
        "fsm_sequence": _plot_fsm_sequence(
            output_path=outputs_dir / "edge_fsm_transition_sequence.png",
        ),
    }
    for name, path in results.items():
        print(f"{name} visualization saved to: {path}")
    return results


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
        for generated_name in (
            "kinematics_and_cut_efficiency.png",
            "epa_actual_vs_predicted.png",
            "yac_actual_vs_predicted.png",
            "partial_spearman_validation_windows.png",
            "forward_draft_class_performance_deltas.png",
            "edge_fsm_transition_sequence.png",
        ):
            (self.outputs_dir / generated_name).unlink(missing_ok=True)

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

    def test_generates_kinematics_and_prediction_comparison_figures(self) -> None:
        from_feature_engineering = _feature_engineering_module()
        tracking = pd.DataFrame(
            {
                "game_id": [1] * 5,
                "play_id": [2] * 5,
                "player_id": [10] * 5,
                "frame_id": [1, 2, 3, 4, 5],
                "x": [0.0, 1.0, 2.0, 2.0, 2.0],
                "y": [0.0, 0.0, 0.0, 1.0, 2.0],
            }
        )
        features = from_feature_engineering.run_feature_pipeline(tracking)
        predictions = pd.DataFrame(
            {
                "actual_epa": [-0.2, 0.1, 0.3, 0.5],
                "predicted_epa": [-0.1, 0.0, 0.25, 0.45],
                "epa_lower": [-0.3, -0.1, 0.1, 0.3],
                "epa_upper": [0.1, 0.1, 0.4, 0.6],
                "actual_yac": [1.0, 4.0, 7.0, 10.0],
                "predicted_yac": [1.5, 3.5, 7.5, 9.5],
            }
        )
        with patch(
            f"{__name__}._outputs_dir",
            return_value=self.outputs_dir.resolve(),
        ):
            generated = generate_project_visualizations(
                features,
                predictions,
                max_points=10,
            )

        self.assertEqual(set(generated), {"kinematics", "epa", "yards_after_catch"})
        for path in generated.values():
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            path.unlink()

    def test_generates_audit_and_illustrative_fsm_figures(self) -> None:
        partial = pd.DataFrame(
            {
                "validation_window": ["pooled", "2024", "2025", "pooled"],
                "target_metric": ["epa", "epa", "epa", "yac"],
                "partial_spearman_rho": [0.2, 0.1, -0.1, 0.3],
                "permutation_p_value": [0.12, 0.42, 0.58, 0.08],
                "sample_count": [30, 12, 14, 30],
                "status": ["ok", "ok", "ok", "ok"],
            }
        )
        forward = pd.DataFrame(
            {
                "target_metric": ["epa", "epa", "yac", "yac"],
                "test_draft_year": [2024, 2024, 2025, 2025],
                "model_variant": [
                    "base",
                    "base_plus_metric",
                    "base",
                    "base_plus_metric",
                ],
                "status": ["ok"] * 4,
                "rmse": [0.8, 0.7, 4.0, 3.8],
                "mae": [0.5, 0.4, 2.5, 2.3],
                "r2": [0.1, 0.2, 0.0, 0.05],
            }
        )
        with patch(
            f"{__name__}._outputs_dir",
            return_value=self.outputs_dir.resolve(),
        ):
            generated = generate_audit_visualizations(partial, forward)

        self.assertEqual(
            set(generated),
            {"partial_spearman", "forward_validation_delta", "fsm_sequence"},
        )
        for path in generated.values():
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 1_000)
            path.unlink()

    def test_audit_figures_explain_missing_data(self) -> None:
        with patch(
            f"{__name__}._outputs_dir",
            return_value=self.outputs_dir.resolve(),
        ):
            generated = generate_audit_visualizations()

        for path in generated.values():
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            path.unlink()

    def test_prediction_requires_paired_actual_and_predicted_targets(self) -> None:
        incomplete = pd.DataFrame(
            {
                "actual_epa": [0.1, 0.2],
                "actual_yac": [2.0, 3.0],
            }
        )
        with patch(
            f"{__name__}._outputs_dir",
            return_value=self.outputs_dir.resolve(),
        ):
            with self.assertRaisesRegex(ValueError, "actual and predicted EPA and/or YAC"):
                generate_project_visualizations(predictions_data=incomplete)

    def test_prediction_interval_bounds_must_be_paired(self) -> None:
        frame = pd.DataFrame(
            {
                "actual_epa": [0.1, 0.2],
                "predicted_epa": [0.0, 0.3],
                "epa_lower": [-0.1, 0.1],
            }
        )
        with self.assertRaisesRegex(ValueError, "both lower and upper"):
            _target_columns_for_predictions(frame, "epa")

    def test_prediction_interval_must_enclose_prediction(self) -> None:
        frame = pd.DataFrame(
            {
                "actual_epa": [0.1, 0.2],
                "predicted_epa": [0.0, 0.3],
                "epa_lower": [-0.2, 0.4],
                "epa_upper": [0.1, 0.5],
            }
        )
        with self.assertRaisesRegex(ValueError, "must enclose each prediction"):
            _plot_prediction_comparison(
                frame,
                target="epa",
                output_path=self.output_path,
                max_points=10,
            )


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
        "--tracking-data",
        type=Path,
        help="Raw tracking or feature CSV in outputs/ for the velocity/cut dashboard.",
    )
    parser.add_argument(
        "--predictions-data",
        type=Path,
        help=(
            "CSV in outputs/ containing actual/predicted EPA and/or YAC "
            "for comparison plots."
        ),
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
        if args.tracking_data is not None or args.predictions_data is not None:
            if args.y_column is not None:
                parser.error("--y-column applies only to the legacy scatter plot mode")
            generate_project_visualizations(
                tracking_data=args.tracking_data,
                predictions_data=args.predictions_data,
                max_points=args.max_points,
            )
        else:
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
