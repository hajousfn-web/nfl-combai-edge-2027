"""Offline sensor-ring orchestration and empirical latency measurements."""

# Author: Soufiane Hajou
# VisiShield-Edge(TM) | Copyright (c) 2026 Soufiane Hajou. All Rights Reserved.

from __future__ import annotations

from datetime import datetime, timezone
import math
import statistics
import time
import unittest

try:
    from .edge_safety_fsm import EdgeSafetyFSM, SafetyState, TelemetryEvent
    from .execution_ring import DeterministicExecutionRing
    from .safety_kernel import SafetyKernel, timestamp_from_epoch_ns
    from .zone_loader import ZoneThresholds
except ImportError:
    from edge_safety_fsm import EdgeSafetyFSM, SafetyState, TelemetryEvent
    from execution_ring import DeterministicExecutionRing
    from safety_kernel import SafetyKernel, timestamp_from_epoch_ns
    from zone_loader import ZoneThresholds


class SafetyEngine:
    """Connect a bounded sample ring to the offline SafetyKernel.

    This class has no hardware bindings. Callers adapt sensor data into the
    explicit scalar sample interface, then inspect each telemetry result.
    """

    def __init__(
        self,
        *,
        thresholds: ZoneThresholds,
        ring_capacity: int = 256,
        recovery_samples_required: int = 3,
        reset_safe_samples_required: int = 3,
        zone_id: str = "configured-zone",
    ) -> None:
        self.ring = DeterministicExecutionRing(ring_capacity)
        self.kernel = SafetyKernel(
            thresholds=thresholds,
            fsm=EdgeSafetyFSM(
                recovery_samples_required=recovery_samples_required,
                reset_safe_samples_required=reset_safe_samples_required,
            ),
            zone_id=zone_id,
        )
        self._last_latency_ns: int | None = None

    @property
    def last_latency_ns(self) -> int | None:
        """Return the latest measured local processing duration."""
        return self._last_latency_ns

    @property
    def state(self) -> SafetyState:
        return self.kernel.fsm.state

    def submit_reading(
        self,
        *,
        timestamp_ns: int,
        distance_m: float,
        speed_mps: float,
        acceleration_mps2: float,
        cut_efficiency: float | None = None,
        sensor_valid: bool = True,
    ) -> None:
        """Queue one sensor/kinematic observation without overwriting old data."""
        self.ring.push(
            timestamp_ns=timestamp_ns,
            distance_m=distance_m,
            speed_mps=speed_mps,
            acceleration_mps2=acceleration_mps2,
            cut_efficiency=cut_efficiency,
            sensor_valid=sensor_valid,
        )

    def process_next(self) -> TelemetryEvent:
        """Process one queued observation and record its observed wall time."""
        start_ns = time.perf_counter_ns()
        sample = self.ring.pop()
        event = self.kernel.evaluate(
            timestamp=timestamp_from_epoch_ns(sample.timestamp_ns),
            distance_m=sample.distance_m,
            speed_mps=sample.speed_mps,
            acceleration_mps2=sample.acceleration_mps2,
            cut_efficiency=(
                sample.cut_efficiency
                if math.isfinite(sample.cut_efficiency)
                else None
            ),
            sensor_valid=sample.sensor_valid,
        )
        self._last_latency_ns = time.perf_counter_ns() - start_ns
        return event

    def manual_reset(
        self,
        *,
        timestamp_ns: int,
        operator_id: str,
        safe_to_reset: bool,
    ) -> TelemetryEvent:
        """Apply software reset acknowledgement using a monotonic UTC time."""
        return self.kernel.manual_reset(
            timestamp=timestamp_from_epoch_ns(timestamp_ns),
            operator_id=operator_id,
            safe_to_reset=safe_to_reset,
        )


def benchmark_processing(
    *,
    iterations: int = 2_000,
    thresholds: ZoneThresholds,
) -> dict[str, float | int]:
    """Measure local reference-loop latency; no deadline is asserted."""
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 10:
        raise ValueError("iterations must be an integer of at least 10.")
    engine = SafetyEngine(
        thresholds=thresholds,
        ring_capacity=1,
        recovery_samples_required=1,
    )
    durations: list[int] = []
    origin_ns = 1_798_000_000_000_000_000
    for index in range(iterations):
        engine.submit_reading(
            timestamp_ns=origin_ns + index * 100_000_000,
            distance_m=2.0,
            speed_mps=2.0,
            acceleration_mps2=0.1,
            cut_efficiency=0.9,
        )
        engine.process_next()
        assert engine.last_latency_ns is not None
        durations.append(engine.last_latency_ns)
    ordered = sorted(durations)
    return {
        "iterations": iterations,
        "p50_us": statistics.median(ordered) / 1_000,
        "p95_us": ordered[int((iterations - 1) * 0.95)] / 1_000,
        "p99_us": ordered[int((iterations - 1) * 0.99)] / 1_000,
        "max_us": ordered[-1] / 1_000,
    }


class _SafetyEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = ZoneThresholds(
            warning_enter_m=1.1,
            warning_clear_m=1.2,
            danger_enter_m=0.8,
            danger_clear_m=0.9,
            emergency_enter_m=0.4,
            emergency_clear_m=0.5,
        )
        self.engine = SafetyEngine(
            thresholds=self.thresholds,
            ring_capacity=2,
            recovery_samples_required=1,
            reset_safe_samples_required=1,
            zone_id="mock-zone",
        )
        self.origin_ns = int(datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9)

    def submit(self, offset_ns: int, distance: float, valid: bool = True) -> TelemetryEvent:
        self.engine.submit_reading(
            timestamp_ns=self.origin_ns + offset_ns,
            distance_m=distance,
            speed_mps=2.0,
            acceleration_mps2=0.0,
            cut_efficiency=0.9,
            sensor_valid=valid,
        )
        return self.engine.process_next()

    def test_ring_sensor_to_fsm_flow_and_latched_manual_reset(self) -> None:
        self.assertEqual(self.submit(0, 0.3).state, SafetyState.EMERGENCY_STOP)
        self.assertEqual(self.submit(100_000_000, 1.5).state, SafetyState.EMERGENCY_STOP)
        reset = self.engine.manual_reset(
            timestamp_ns=self.origin_ns + 200_000_000,
            operator_id="mock-operator",
            safe_to_reset=True,
        )
        self.assertEqual(reset.state, SafetyState.SAFE)
        self.assertGreater(self.engine.last_latency_ns or 0, 0)

    def test_invalid_sensor_emits_fault_not_default_safe_state(self) -> None:
        event = self.submit(0, 1.5, valid=False)
        self.assertEqual(event.state, SafetyState.SENSOR_FAULT)

    def test_benchmark_reports_measured_statistics(self) -> None:
        result = benchmark_processing(iterations=10, thresholds=self.thresholds)
        self.assertEqual(result["iterations"], 10)
        self.assertGreaterEqual(result["max_us"], result["p50_us"])


def _self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_SafetyEngineTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
