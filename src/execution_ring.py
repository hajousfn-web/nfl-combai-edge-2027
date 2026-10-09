"""Bounded preallocated sample ring for offline edge-loop experiments.

Python object allocation, interpreter scheduling, and garbage collection remain
present. This is not a hard-real-time or zero-allocation execution guarantee.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from numbers import Real
from typing import TypeAlias
import unittest

NumericInput: TypeAlias = float | int


@dataclass(frozen=True)
class SensorFrame:
    """One unpacked LiDAR/kinematics sample."""

    timestamp_ns: int
    distance_m: float
    speed_mps: float
    acceleration_mps2: float
    cut_efficiency: float
    sensor_valid: bool


class DeterministicExecutionRing:
    """Fixed-capacity FIFO with numeric storage allocated once at construction."""

    def __init__(self, capacity: int = 256) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer.")
        self.capacity = capacity
        self._timestamps = array("q", [0]) * capacity
        self._distances = array("d", [0.0]) * capacity
        self._speeds = array("d", [0.0]) * capacity
        self._accelerations = array("d", [0.0]) * capacity
        self._cut_efficiencies = array("d", [0.0]) * capacity
        self._valid = bytearray(capacity)
        self._head = 0
        self._tail = 0
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def push(
        self,
        *,
        timestamp_ns: int,
        distance_m: NumericInput,
        speed_mps: NumericInput,
        acceleration_mps2: NumericInput,
        cut_efficiency: NumericInput | None = None,
        sensor_valid: bool = True,
    ) -> None:
        """Copy a validated sample into the next preallocated slot."""
        if self._count == self.capacity:
            raise BufferError("Execution ring is full; sample was not overwritten.")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise TypeError("timestamp_ns must be an integer.")
        if not isinstance(sensor_valid, bool):
            raise TypeError("sensor_valid must be a bool.")
        values = (distance_m, speed_mps, acceleration_mps2)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("Distance, speed, and acceleration must be finite numbers.")
        if cut_efficiency is not None and (
            isinstance(cut_efficiency, bool)
            or not isinstance(cut_efficiency, Real)
            or not math.isfinite(cut_efficiency)
        ):
            raise ValueError("cut_efficiency must be finite or None.")

        slot = self._tail
        self._timestamps[slot] = timestamp_ns
        self._distances[slot] = float(distance_m)
        self._speeds[slot] = float(speed_mps)
        self._accelerations[slot] = float(acceleration_mps2)
        self._cut_efficiencies[slot] = (
            math.nan if cut_efficiency is None else float(cut_efficiency)
        )
        self._valid[slot] = sensor_valid
        self._tail = (slot + 1) % self.capacity
        self._count += 1

    def pop(self) -> SensorFrame:
        """Remove the oldest sample; empty reads are surfaced explicitly."""
        if self._count == 0:
            raise IndexError("Execution ring is empty.")
        slot = self._head
        sample = SensorFrame(
            timestamp_ns=self._timestamps[slot],
            distance_m=self._distances[slot],
            speed_mps=self._speeds[slot],
            acceleration_mps2=self._accelerations[slot],
            cut_efficiency=self._cut_efficiencies[slot],
            sensor_valid=bool(self._valid[slot]),
        )
        self._head = (slot + 1) % self.capacity
        self._count -= 1
        return sample


class _ExecutionRingTest(unittest.TestCase):
    def test_fixed_ring_is_fifo_and_does_not_overwrite(self) -> None:
        ring = DeterministicExecutionRing(capacity=2)
        for timestamp in (10, 20):
            ring.push(
                timestamp_ns=timestamp,
                distance_m=1.1,
                speed_mps=2.0,
                acceleration_mps2=0.2,
            )
        with self.assertRaises(BufferError):
            ring.push(
                timestamp_ns=30,
                distance_m=1.0,
                speed_mps=2.0,
                acceleration_mps2=0.2,
            )
        self.assertEqual(ring.pop().timestamp_ns, 10)
        self.assertEqual(ring.pop().timestamp_ns, 20)
        with self.assertRaises(IndexError):
            ring.pop()

    def test_rejects_invalid_numeric_inputs(self) -> None:
        ring = DeterministicExecutionRing()
        with self.assertRaises(ValueError):
            ring.push(
                timestamp_ns=1,
                distance_m=math.nan,
                speed_mps=0.0,
                acceleration_mps2=0.0,
            )


def _self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_ExecutionRingTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
