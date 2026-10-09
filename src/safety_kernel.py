"""Offline proximity/kinematics policy evaluation and FSM integration.

Thresholds must come from an explicit, reviewed local configuration. This
module is a reference simulator, not a certified safety kernel or controller.
"""

# Author: Soufiane Hajou
# VisiShield-Edge(TM) | Copyright (c) 2026 Soufiane Hajou. All Rights Reserved.

from __future__ import annotations

from datetime import datetime, timezone
import math
from numbers import Real

try:
    from .edge_safety_fsm import EdgeSafetyFSM, SafetyState, TelemetryEvent
    from .zone_loader import ZoneThresholds
except ImportError:
    from edge_safety_fsm import EdgeSafetyFSM, SafetyState, TelemetryEvent
    from zone_loader import ZoneThresholds


class SafetyKernel:
    """Evaluate configured LiDAR proximity with hysteresis and FSM debouncing."""

    def __init__(
        self,
        *,
        thresholds: ZoneThresholds,
        fsm: EdgeSafetyFSM | None = None,
        zone_id: str = "configured-zone",
    ) -> None:
        if not isinstance(thresholds, ZoneThresholds):
            raise TypeError("thresholds must be a validated ZoneThresholds instance.")
        if not isinstance(zone_id, str) or not zone_id.strip():
            raise ValueError("zone_id must be a non-empty string.")
        self.thresholds = thresholds
        self.fsm = fsm or EdgeSafetyFSM()
        self.zone_id = zone_id.strip()
        self._proximity_state = SafetyState.SAFE

    @property
    def proximity_state(self) -> SafetyState:
        """Return the last geometric classification before FSM debouncing."""
        return self._proximity_state

    def evaluate(
        self,
        *,
        timestamp: datetime,
        distance_m: float | int,
        speed_mps: float | int,
        acceleration_mps2: float | int,
        cut_efficiency: float | int | None = None,
        sensor_valid: bool = True,
        fault_reason: str | None = None,
    ) -> TelemetryEvent:
        """Classify one LiDAR distance and attach player kinematics telemetry.

        Invalid or non-finite sensor readings latch ``SENSOR_FAULT``. No policy
        maps Cut Efficiency or EPA/YAC predictions to a hazard classification.
        """
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime.")
        if not isinstance(sensor_valid, bool):
            raise TypeError("sensor_valid must be a bool.")
        invalid_reason = fault_reason
        if not sensor_valid and invalid_reason is None:
            invalid_reason = "sensor_marked_invalid"

        measured = {
            "distance_m": distance_m,
            "speed_mps": speed_mps,
            "acceleration_mps2": acceleration_mps2,
        }
        if cut_efficiency is not None:
            measured["cut_efficiency"] = cut_efficiency
        for name, value in measured.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                invalid_reason = invalid_reason or f"invalid_{name}"
        if isinstance(distance_m, Real) and distance_m < 0:
            invalid_reason = invalid_reason or "negative_lidar_distance"

        metrics: dict[str, int | float | None] = {
            "distance_m": (
                float(distance_m) if isinstance(distance_m, Real) and math.isfinite(distance_m) else None
            ),
            "speed_mps": (
                float(speed_mps) if isinstance(speed_mps, Real) and math.isfinite(speed_mps) else None
            ),
            "acceleration_mps2": (
                float(acceleration_mps2)
                if isinstance(acceleration_mps2, Real)
                and math.isfinite(acceleration_mps2)
                else None
            ),
            "cut_efficiency": (
                float(cut_efficiency)
                if cut_efficiency is not None
                and isinstance(cut_efficiency, Real)
                and math.isfinite(cut_efficiency)
                else None
            ),
        }
        if invalid_reason is not None:
            return self.fsm.observe(
                timestamp=timestamp,
                risk_state=SafetyState.SENSOR_FAULT,
                metrics=metrics,
                fault_reason=invalid_reason,
            )

        self._proximity_state = self._classify_distance(float(distance_m))
        return self.fsm.observe(
            timestamp=timestamp,
            risk_state=self._proximity_state,
            metrics=metrics,
        )

    def latch_fail_safe(
        self,
        *,
        timestamp: datetime,
        reason: str,
    ) -> TelemetryEvent:
        """Latch software fail-safe telemetry after an explicit system fault."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")
        return self.fsm.observe(
            timestamp=timestamp,
            risk_state=SafetyState.FAIL_SAFE_LOCKED,
            fault_reason=reason.strip(),
        )

    def manual_reset(
        self,
        *,
        timestamp: datetime,
        operator_id: str,
        safe_to_reset: bool,
    ) -> TelemetryEvent:
        """Forward explicit software reset acknowledgement to the FSM."""
        return self.fsm.manual_reset(
            timestamp=timestamp,
            operator_id=operator_id,
            safe_to_reset=safe_to_reset,
        )

    def _classify_distance(self, distance_m: float) -> SafetyState:
        thresholds = self.thresholds
        current = self._proximity_state
        if current is SafetyState.EMERGENCY_STOP:
            if distance_m < thresholds.emergency_clear_m:
                return current
        if distance_m <= thresholds.emergency_enter_m:
            return SafetyState.EMERGENCY_STOP

        if distance_m <= thresholds.danger_enter_m:
            return SafetyState.DANGER
        if current is SafetyState.DANGER and distance_m < thresholds.danger_clear_m:
            return current

        if distance_m <= thresholds.warning_enter_m:
            return SafetyState.WARNING
        if current is SafetyState.WARNING and distance_m < thresholds.warning_clear_m:
            return current
        return SafetyState.SAFE


def timestamp_from_epoch_ns(timestamp_ns: int) -> datetime:
    """Convert a UTC Unix timestamp in nanoseconds to timezone-aware datetime."""
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
        raise TypeError("timestamp_ns must be an integer.")
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).replace(
        microsecond=nanoseconds // 1_000
    )


def _self_test() -> int:
    import unittest
    from datetime import timedelta

    class SafetyKernelTest(unittest.TestCase):
        def setUp(self) -> None:
            self.thresholds = ZoneThresholds(
                warning_enter_m=1.1,
                warning_clear_m=1.2,
                danger_enter_m=0.8,
                danger_clear_m=0.9,
                emergency_enter_m=0.4,
                emergency_clear_m=0.5,
            )
            self.origin = datetime(2027, 1, 1, tzinfo=timezone.utc)
            self.kernel = SafetyKernel(
                thresholds=self.thresholds,
                fsm=EdgeSafetyFSM(
                    recovery_samples_required=2,
                    reset_safe_samples_required=2,
                ),
            )

        def evaluate(self, second: int, distance: float) -> TelemetryEvent:
            return self.kernel.evaluate(
                timestamp=self.origin + timedelta(seconds=second),
                distance_m=distance,
                speed_mps=2.0,
                acceleration_mps2=0.1,
                cut_efficiency=0.8,
            )

        def test_proximity_escalation_hysteresis_and_manual_reset(self) -> None:
            self.assertEqual(self.evaluate(0, 1.1).state, SafetyState.WARNING)
            self.assertEqual(self.evaluate(1, 0.75).state, SafetyState.DANGER)
            self.assertEqual(self.evaluate(2, 0.4).state, SafetyState.EMERGENCY_STOP)
            self.assertEqual(self.evaluate(3, 0.45).state, SafetyState.EMERGENCY_STOP)
            self.assertEqual(self.evaluate(4, 1.3).state, SafetyState.EMERGENCY_STOP)
            self.assertEqual(self.evaluate(5, 1.3).state, SafetyState.EMERGENCY_STOP)
            reset = self.kernel.manual_reset(
                timestamp=self.origin + timedelta(seconds=6),
                operator_id="mock-operator",
                safe_to_reset=True,
            )
            self.assertEqual(reset.state, SafetyState.SAFE)

        def test_bad_sensor_data_latches_fault(self) -> None:
            event = self.kernel.evaluate(
                timestamp=self.origin,
                distance_m=math.nan,
                speed_mps=1.0,
                acceleration_mps2=0.0,
            )
            self.assertEqual(event.state, SafetyState.SENSOR_FAULT)
            self.assertTrue(event.lockout_active)

        def test_fail_safe_is_explicit_and_latched(self) -> None:
            event = self.kernel.latch_fail_safe(
                timestamp=self.origin,
                reason="mock kernel fault",
            )
            self.assertEqual(event.state, SafetyState.FAIL_SAFE_LOCKED)
            self.assertTrue(event.lockout_active)
            still_locked = self.evaluate(1, 1.5)
            self.assertEqual(still_locked.state, SafetyState.FAIL_SAFE_LOCKED)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SafetyKernelTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
