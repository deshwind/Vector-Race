"""Reference trajectory model and portable CSV export."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _array(values: Iterable[float], name: str) -> FloatArray:
    result = np.asarray(tuple(values), dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


@dataclass(frozen=True)
class RacingTrajectory:
    s_m: FloatArray
    x_m: FloatArray
    y_m: FloatArray
    heading_rad: FloatArray
    curvature_1pm: FloatArray
    speed_mps: FloatArray
    lateral_offset_m: FloatArray
    width_left_m: FloatArray
    width_right_m: FloatArray
    metadata: dict[str, Any] = field(default_factory=dict)
    local_grip_mu: FloatArray | None = None

    def __post_init__(self) -> None:
        names = (
            "s_m",
            "x_m",
            "y_m",
            "heading_rad",
            "curvature_1pm",
            "speed_mps",
            "lateral_offset_m",
            "width_left_m",
            "width_right_m",
        )
        arrays = {name: _array(getattr(self, name), name) for name in names}
        local_grip = (
            None
            if self.local_grip_mu is None
            else _array(self.local_grip_mu, "local_grip_mu")
        )
        lengths = {len(value) for value in arrays.values()}
        if local_grip is not None:
            lengths.add(len(local_grip))
        if len(lengths) != 1:
            raise ValueError("all trajectory arrays must have the same length")
        if len(arrays["s_m"]) < 8:
            raise ValueError("trajectory needs at least eight points")
        if np.any(np.diff(arrays["s_m"]) <= 0):
            raise ValueError("s_m must be strictly increasing")
        if np.any(arrays["speed_mps"] <= 0):
            raise ValueError("trajectory speeds must be positive")
        if np.any(arrays["width_left_m"] <= 0) or np.any(
            arrays["width_right_m"] <= 0
        ):
            raise ValueError("trajectory widths must be positive")
        if local_grip is not None and np.any(local_grip <= 0):
            raise ValueError("trajectory local grip values must be positive")
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if local_grip is not None:
            local_grip.setflags(write=False)
        object.__setattr__(self, "local_grip_mu", local_grip)

    @property
    def point_count(self) -> int:
        return len(self.x_m)

    @property
    def segment_lengths_m(self) -> FloatArray:
        return np.hypot(
            np.roll(self.x_m, -1) - self.x_m,
            np.roll(self.y_m, -1) - self.y_m,
        )

    @property
    def length_m(self) -> float:
        return float(np.sum(self.segment_lengths_m))

    @property
    def lap_time_s(self) -> float:
        next_speed = np.roll(self.speed_mps, -1)
        mean_speed = np.maximum((self.speed_mps + next_speed) / 2.0, 1.0e-6)
        return float(np.sum(self.segment_lengths_m / mean_speed))

    @property
    def normal_x(self) -> FloatArray:
        return -np.sin(self.heading_rad)

    @property
    def normal_y(self) -> FloatArray:
        return np.cos(self.heading_rad)

    def nearest_index(self, x_m: float, y_m: float) -> int:
        distances_squared = (self.x_m - x_m) ** 2 + (self.y_m - y_m) ** 2
        return int(np.argmin(distances_squared))

    def to_csv(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "s_m",
            "x_m",
            "y_m",
            "heading_rad",
            "curvature_1pm",
            "speed_mps",
            "lateral_offset_m",
            "width_left_m",
            "width_right_m",
        ]
        if self.local_grip_mu is not None:
            fields.append("grip_mu")
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(self.point_count):
                row = {
                    name: getattr(self, name)[index]
                    for name in fields
                    if name != "grip_mu"
                }
                if self.local_grip_mu is not None:
                    row["grip_mu"] = self.local_grip_mu[index]
                writer.writerow(row)
        return destination

    @classmethod
    def from_csv(cls, path: str | Path) -> "RacingTrajectory":
        source = Path(path)
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        required = {
            "s_m",
            "x_m",
            "y_m",
            "heading_rad",
            "curvature_1pm",
            "speed_mps",
            "lateral_offset_m",
            "width_left_m",
            "width_right_m",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"trajectory CSV is missing columns: {sorted(required)}")
        if not rows:
            raise ValueError("trajectory CSV is empty")
        values = {
            name: np.asarray([float(row[name]) for row in rows]) for name in required
        }
        local_grip = (
            np.asarray([float(row["grip_mu"]) for row in rows])
            if "grip_mu" in (reader.fieldnames or ())
            else None
        )
        return cls(
            **values,
            local_grip_mu=local_grip,
            metadata={"source": str(source)},
        )
