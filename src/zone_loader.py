"""Strict loader for externally approved proximity thresholds.

This module validates configuration shape only. It does not approve distances
for a real venue, sensor, sport, or safety function.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ZoneThresholds:
    """Monotonic enter/clear distances for one caller-named zone."""

    warning_enter_m: float
    warning_clear_m: float
    danger_enter_m: float
    danger_clear_m: float
    emergency_enter_m: float
    emergency_clear_m: float

    def __post_init__(self) -> None:
        values = (
            self.emergency_enter_m,
            self.emergency_clear_m,
            self.danger_enter_m,
            self.danger_clear_m,
            self.warning_enter_m,
            self.warning_clear_m,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise ValueError("Zone distances must be finite, non-negative numbers.")
        if not (
            self.emergency_enter_m
            < self.emergency_clear_m
            <= self.danger_enter_m
            < self.danger_clear_m
            <= self.warning_enter_m
            < self.warning_clear_m
        ):
            raise ValueError(
                "Zone enter/clear distances must be strictly ordered to provide "
                "non-overlapping hysteresis bands."
            )


def load_zone_thresholds(path: str | Path, *, zone_id: str) -> ZoneThresholds:
    """Load a named zone from an explicit JSON config file.

    Expected shape:
    ``{"schema_version": 1, "zones": {"zone-id": {threshold fields...}}}``.
    Paths are supplied by the caller; no absolute path is embedded here.
    """
    if not isinstance(zone_id, str) or not zone_id.strip():
        raise ValueError("zone_id must be a non-empty string.")
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            payload: Any = json.load(config_file)
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Could not load zone configuration {config_path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Zone configuration must use schema_version 1.")
    zones = payload.get("zones")
    if not isinstance(zones, dict) or zone_id not in zones:
        raise ValueError(f"Zone configuration does not define zone {zone_id!r}.")
    values = zones[zone_id]
    required = {
        "warning_enter_m",
        "warning_clear_m",
        "danger_enter_m",
        "danger_clear_m",
        "emergency_enter_m",
        "emergency_clear_m",
    }
    if not isinstance(values, dict) or set(values) != required:
        raise ValueError(
            f"Zone {zone_id!r} must contain exactly: {', '.join(sorted(required))}."
        )
    try:
        return ZoneThresholds(**values)
    except TypeError as error:
        raise ValueError(f"Zone {zone_id!r} threshold fields are invalid: {error}") from error


def _self_test() -> int:
    import tempfile
    import unittest

    class ZoneLoaderTest(unittest.TestCase):
        def test_loads_exact_zone_and_validates_order(self) -> None:
            values = {
                "warning_enter_m": 1.1,
                "warning_clear_m": 1.2,
                "danger_enter_m": 0.8,
                "danger_clear_m": 0.9,
                "emergency_enter_m": 0.4,
                "emergency_clear_m": 0.5,
            }
            with tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "zones.json"
                config.write_text(
                    json.dumps({"schema_version": 1, "zones": {"mock": values}}),
                    encoding="utf-8",
                )
                thresholds = load_zone_thresholds(config, zone_id="mock")
            self.assertEqual(thresholds.warning_enter_m, 1.1)

        def test_rejects_missing_and_unordered_thresholds(self) -> None:
            with self.assertRaises(ValueError):
                ZoneThresholds(1.1, 1.2, 0.8, 0.9, 0.4, 0.3)

        def test_rejects_unknown_zone(self) -> None:
            with tempfile.TemporaryDirectory() as directory:
                config = Path(directory) / "zones.json"
                config.write_text(
                    json.dumps({"schema_version": 1, "zones": {}}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "does not define"):
                    load_zone_thresholds(config, zone_id="mock")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ZoneLoaderTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
