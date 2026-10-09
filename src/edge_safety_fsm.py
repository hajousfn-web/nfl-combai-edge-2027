"""Offline-first risk-state and lockout telemetry contract for edge analysis.

This module consumes a risk classification produced by a separately validated
policy. It records state transitions only; it does not control actuators or
define operational hazard thresholds.
"""

# Author: Soufiane Hajou
# VisiShield-Edge(TM) | Copyright (c) 2026 Soufiane Hajou. All Rights Reserved.

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
import math
from numbers import Real
from typing import Mapping


class SafetyState(IntEnum):
    """Ordered risk classification used by the telemetry state machine."""

    SAFE = 0
    WARNING = 1
    DANGER = 2
    EMERGENCY_STOP = 3
    SENSOR_FAULT = 4
    FAIL_SAFE_LOCKED = 5


_LOCKED_STATES = frozenset(
    {
        SafetyState.EMERGENCY_STOP,
        SafetyState.SENSOR_FAULT,
        SafetyState.FAIL_SAFE_LOCKED,
    }
)


@dataclass(frozen=True)
class TelemetryEvent:
    """Immutable state snapshot for one observation or manual reset."""

    sequence: int
    timestamp: datetime
    state: SafetyState
    requested_state: SafetyState
    lockout_active: bool
    transitioned: bool
    reason: str
    operator_id: str | None
    metrics: tuple[tuple[str, float | None], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a serialization-friendly representation of this event."""
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.name,
            "requested_state": self.requested_state.name,
            "lockout_active": self.lockout_active,
            "transitioned": self.transitioned,
            "reason": self.reason,
            "operator_id": self.operator_id,
            "metrics": dict(self.metrics),
        }


class EdgeSafetyFSM:
    """Track classified risk states with fail-safe stop and manual reset.

    Risk thresholds are deliberately external: a validated, deployment-owned
    policy maps 10 Hz kinematics, cut efficiency, and data-quality checks to a
    ``SafetyState``. Escalations are immediate; recovery requires a configurable
    number of consecutive lower-risk observations. EMERGENCY_STOP,
    SENSOR_FAULT, and FAIL_SAFE_LOCKED remain latched until explicitly reset.
    """

    def __init__(
        self,
        *,
        recovery_samples_required: int = 3,
        reset_safe_samples_required: int = 3,
        history_limit: int = 1_000,
    ) -> None:
        for name, value in (
            ("recovery_samples_required", recovery_samples_required),
            ("reset_safe_samples_required", reset_safe_samples_required),
            ("history_limit", history_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        self.recovery_samples_required = recovery_samples_required
        self.reset_safe_samples_required = reset_safe_samples_required
        self._state = SafetyState.SAFE
        self._last_timestamp: datetime | None = None
        self._recovery_candidate: SafetyState | None = None
        self._recovery_count = 0
        self._reset_safe_count = 0
        self._sequence = 0
        self._events: deque[TelemetryEvent] = deque(maxlen=history_limit)

    @property
    def state(self) -> SafetyState:
        """Return the current latched state."""
        return self._state

    @property
    def lockout_active(self) -> bool:
        """Return whether any manual-reset-required state is latched."""
        return self._state in _LOCKED_STATES

    @property
    def events(self) -> tuple[TelemetryEvent, ...]:
        """Return the bounded in-memory event history."""
        return tuple(self._events)

    def observe(
        self,
        *,
        timestamp: datetime,
        risk_state: SafetyState | str,
        metrics: Mapping[str, int | float | None] | None = None,
        sample_valid: bool = True,
        fault_reason: str | None = None,
    ) -> TelemetryEvent:
        """Record one ordered observation and apply the transition contract.

        An explicitly invalid sample or supplied fault reason immediately
        requests SENSOR_FAULT. Invalid arguments and non-monotonic timestamps
        raise rather than silently creating misleading telemetry.
        """
        normalized_timestamp = self._normalize_timestamp(timestamp)
        self._validate_timestamp_order(normalized_timestamp)
        requested_state = self._normalize_state(risk_state)
        normalized_metrics = self._normalize_metrics(metrics)
        if not isinstance(sample_valid, bool):
            raise TypeError("sample_valid must be a bool.")
        if fault_reason is not None and (
            not isinstance(fault_reason, str) or not fault_reason.strip()
        ):
            raise ValueError("fault_reason must be a non-empty string when supplied.")

        reason = "risk_policy_classification"
        if not sample_valid:
            requested_state = SafetyState.SENSOR_FAULT
            reason = fault_reason.strip() if fault_reason else "invalid_telemetry_sample"
        elif fault_reason is not None and requested_state is not SafetyState.FAIL_SAFE_LOCKED:
            requested_state = SafetyState.SENSOR_FAULT
            reason = fault_reason.strip()
        elif fault_reason is not None:
            reason = fault_reason.strip()

        previous_state = self._state
        if requested_state is SafetyState.FAIL_SAFE_LOCKED:
            self._state = SafetyState.FAIL_SAFE_LOCKED
            self._reset_safe_count = 0
            self._recovery_candidate = None
            self._recovery_count = 0
            reason = reason if reason != "risk_policy_classification" else "fail_safe_latched"
        if self.lockout_active:
            if (
                requested_state is SafetyState.SAFE
                and sample_valid
                and fault_reason is None
            ):
                self._reset_safe_count += 1
            else:
                self._reset_safe_count = 0
            if (
                requested_state is SafetyState.SENSOR_FAULT
                and self._state is not SafetyState.FAIL_SAFE_LOCKED
            ):
                self._state = SafetyState.SENSOR_FAULT
            reason = (
                reason
                if self._state is SafetyState.FAIL_SAFE_LOCKED
                else f"{self._state.name.casefold()}_latched"
            )
        elif requested_state in _LOCKED_STATES:
            self._state = requested_state
            self._reset_safe_count = 0
            self._recovery_candidate = None
            self._recovery_count = 0
            reason = reason if reason != "risk_policy_classification" else f"{requested_state.name.casefold()}_latched"
        elif requested_state >= self._state:
            self._state = requested_state
            self._recovery_candidate = None
            self._recovery_count = 0
            self._reset_safe_count = 0
            if requested_state is SafetyState.EMERGENCY_STOP:
                reason = reason if reason != "risk_policy_classification" else "risk_policy_emergency_stop"
        elif requested_state is self._state:
            self._recovery_candidate = None
            self._recovery_count = 0
        else:
            if requested_state is self._recovery_candidate:
                self._recovery_count += 1
            else:
                self._recovery_candidate = requested_state
                self._recovery_count = 1
            if self._recovery_count >= self.recovery_samples_required:
                self._state = requested_state
                self._recovery_candidate = None
                self._recovery_count = 0
                reason = "stable_lower_risk_recovery"
            else:
                reason = "awaiting_stable_lower_risk_samples"

        self._last_timestamp = normalized_timestamp
        return self._record(
            timestamp=normalized_timestamp,
            requested_state=requested_state,
            previous_state=previous_state,
            reason=reason,
            metrics=normalized_metrics,
        )

    def manual_reset(
        self,
        *,
        timestamp: datetime,
        operator_id: str,
        safe_to_reset: bool,
    ) -> TelemetryEvent:
        """Clear a latched stop/fault after explicit operator confirmation.

        ``safe_to_reset`` must be asserted by a separately verified inspection
        or interlock. This software event is not a physical lockout/tagout action.
        """
        normalized_timestamp = self._normalize_timestamp(timestamp)
        self._validate_timestamp_order(normalized_timestamp)
        if not isinstance(operator_id, str) or not operator_id.strip():
            raise ValueError("operator_id must identify the resetting operator.")
        if not isinstance(safe_to_reset, bool):
            raise TypeError("safe_to_reset must be a bool.")
        if not self.lockout_active:
            raise RuntimeError("Manual reset is only valid while lockout is active.")
        if not safe_to_reset:
            raise RuntimeError("Manual reset denied: safe_to_reset was not confirmed.")
        if self._reset_safe_count < self.reset_safe_samples_required:
            raise RuntimeError(
                "Manual reset denied: insufficient consecutive SAFE observations "
                f"({self._reset_safe_count}/{self.reset_safe_samples_required})."
            )

        previous_state = self._state
        self._state = SafetyState.SAFE
        self._reset_safe_count = 0
        self._recovery_candidate = None
        self._recovery_count = 0
        self._last_timestamp = normalized_timestamp
        return self._record(
            timestamp=normalized_timestamp,
            requested_state=SafetyState.SAFE,
            previous_state=previous_state,
            reason="manual_lockout_reset",
            metrics=(),
            operator_id=operator_id.strip(),
        )

    def _record(
        self,
        *,
        timestamp: datetime,
        requested_state: SafetyState,
        previous_state: SafetyState,
        reason: str,
        metrics: tuple[tuple[str, float | None], ...],
        operator_id: str | None = None,
    ) -> TelemetryEvent:
        self._sequence += 1
        event = TelemetryEvent(
            sequence=self._sequence,
            timestamp=timestamp,
            state=self._state,
            requested_state=requested_state,
            lockout_active=self.lockout_active,
            transitioned=self._state is not previous_state,
            reason=reason,
            operator_id=operator_id,
            metrics=metrics,
        )
        self._events.append(event)
        return event

    @staticmethod
    def _normalize_state(value: SafetyState | str) -> SafetyState:
        if isinstance(value, SafetyState):
            return value
        if isinstance(value, str):
            try:
                return SafetyState[value.strip().upper()]
            except KeyError as error:
                raise ValueError(f"Unknown safety state: {value!r}") from error
        raise TypeError("risk_state must be a SafetyState or its name.")

    @staticmethod
    def _normalize_timestamp(value: datetime) -> datetime:
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError("timestamp must be a timezone-aware datetime.")
        return value.astimezone(timezone.utc)

    def _validate_timestamp_order(self, timestamp: datetime) -> None:
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("Telemetry timestamps must be strictly increasing.")

    @staticmethod
    def _normalize_metrics(
        metrics: Mapping[str, int | float | None] | None,
    ) -> tuple[tuple[str, float | None], ...]:
        if metrics is None:
            return ()
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping of names to numeric values.")

        normalized: list[tuple[str, float | None]] = []
        for name, value in metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Metric names must be non-empty strings.")
            if value is None:
                normalized.append((name.strip(), None))
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Metric {name!r} must be a real number or None.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Metric {name!r} must be finite.")
            normalized.append((name.strip(), numeric_value))
        return tuple(sorted(normalized))


def _self_test() -> int:
    """Run lightweight transition and lockout checks without external data."""
    import unittest

    class EdgeSafetyFSMTest(unittest.TestCase):
        def test_escalation_recovery_and_manual_lockout_reset(self) -> None:
            from datetime import timedelta

            origin = datetime(2027, 1, 1, tzinfo=timezone.utc)
            fsm = EdgeSafetyFSM(recovery_samples_required=2, reset_safe_samples_required=2)
            warning = fsm.observe(
                timestamp=origin,
                risk_state=SafetyState.WARNING,
                metrics={"speed": 4.2, "proprietary_cut_efficiency": 0.8},
            )
            self.assertEqual(warning.state, SafetyState.WARNING)
            self.assertEqual(
                fsm.observe(
                    timestamp=origin + timedelta(milliseconds=100),
                    risk_state=SafetyState.DANGER,
                ).state,
                SafetyState.DANGER,
            )
            fsm.observe(
                timestamp=origin + timedelta(milliseconds=200),
                risk_state=SafetyState.SAFE,
            )
            self.assertEqual(fsm.state, SafetyState.DANGER)
            fsm.observe(
                timestamp=origin + timedelta(milliseconds=300),
                risk_state=SafetyState.SAFE,
            )
            self.assertEqual(fsm.state, SafetyState.SAFE)

            stopped = fsm.observe(
                timestamp=origin + timedelta(milliseconds=400),
                risk_state=SafetyState.EMERGENCY_STOP,
            )
            self.assertTrue(stopped.lockout_active)
            held = fsm.observe(
                timestamp=origin + timedelta(milliseconds=500),
                risk_state=SafetyState.SAFE,
            )
            self.assertEqual(held.state, SafetyState.EMERGENCY_STOP)
            fsm.observe(
                timestamp=origin + timedelta(milliseconds=600),
                risk_state=SafetyState.SAFE,
            )
            reset = fsm.manual_reset(
                timestamp=origin + timedelta(milliseconds=700),
                operator_id="operator-1",
                safe_to_reset=True,
            )
            self.assertEqual(reset.state, SafetyState.SAFE)
            self.assertFalse(reset.lockout_active)
            self.assertEqual(reset.operator_id, "operator-1")

        def test_invalid_sample_latches_stop_and_requires_reset(self) -> None:
            from datetime import timedelta

            origin = datetime(2027, 1, 1, tzinfo=timezone.utc)
            fsm = EdgeSafetyFSM(reset_safe_samples_required=1)
            event = fsm.observe(
                timestamp=origin,
                risk_state=SafetyState.SAFE,
                sample_valid=False,
                fault_reason="tracking heartbeat lost",
            )
            self.assertEqual(event.state, SafetyState.SENSOR_FAULT)
            self.assertTrue(event.lockout_active)
            fsm.observe(
                timestamp=origin + timedelta(milliseconds=100),
                risk_state=SafetyState.SAFE,
            )
            with self.assertRaises(RuntimeError):
                fsm.manual_reset(
                    timestamp=origin + timedelta(milliseconds=200),
                    operator_id="operator-1",
                    safe_to_reset=False,
                )
            self.assertTrue(fsm.lockout_active)

        def test_rejects_non_monotonic_timestamps(self) -> None:
            from datetime import timedelta

            origin = datetime(2027, 1, 1, tzinfo=timezone.utc)
            fsm = EdgeSafetyFSM()
            fsm.observe(timestamp=origin, risk_state=SafetyState.SAFE)
            with self.assertRaises(ValueError):
                fsm.observe(
                    timestamp=origin - timedelta(milliseconds=100),
                    risk_state=SafetyState.DANGER,
                )
            self.assertEqual(fsm.state, SafetyState.SAFE)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(EdgeSafetyFSMTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
