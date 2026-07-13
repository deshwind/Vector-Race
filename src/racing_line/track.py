"""Track data model and CSV interchange."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _float_array(values: Iterable[float], name: str) -> FloatArray:
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


@dataclass(frozen=True)
class Track:
    """Closed track centreline and per-point widths.

    Positive lateral offsets are to the left of the direction of travel.
    The first point must not be duplicated at the end; duplicate closure points
    are removed automatically for convenience.
    """

    x_m: FloatArray
    y_m: FloatArray
    width_left_m: FloatArray
    width_right_m: FloatArray
    local_grip_mu: FloatArray | None = None
    name: str = "track"

    def __post_init__(self) -> None:
        x = _float_array(self.x_m, "x_m")
        y = _float_array(self.y_m, "y_m")
        left = _float_array(self.width_left_m, "width_left_m")
        right = _float_array(self.width_right_m, "width_right_m")
        grip = (
            None
            if self.local_grip_mu is None
            else _float_array(self.local_grip_mu, "local_grip_mu")
        )

        lengths = {len(x), len(y), len(left), len(right)}
        if grip is not None:
            lengths.add(len(grip))
        if len(lengths) != 1:
            raise ValueError("all track arrays must have the same length")
        if len(x) >= 2 and np.hypot(x[-1] - x[0], y[-1] - y[0]) < 1.0e-9:
            x, y, left, right = x[:-1], y[:-1], left[:-1], right[:-1]
            if grip is not None:
                grip = grip[:-1]
        if len(x) < 8:
            raise ValueError("a closed track needs at least eight distinct points")
        if np.any(left <= 0) or np.any(right <= 0):
            raise ValueError("track widths must be positive")
        if grip is not None and np.any(grip <= 0):
            raise ValueError("local grip values must be positive")
        segment_lengths = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
        if np.any(segment_lengths < 1.0e-6):
            raise ValueError("track contains duplicate consecutive points")

        for name, value in (
            ("x_m", x),
            ("y_m", y),
            ("width_left_m", left),
            ("width_right_m", right),
            ("local_grip_mu", grip),
        ):
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, name, value)

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

    @classmethod
    def from_csv(cls, path: str | Path, name: str | None = None) -> "Track":
        """Load ``x_m,y_m,width_left_m,width_right_m[,grip_mu]``."""

        csv_path = Path(path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"x_m", "y_m", "width_left_m", "width_right_m"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    "track CSV needs columns: x_m, y_m, width_left_m, width_right_m"
                )
            rows = list(reader)
        if not rows:
            raise ValueError("track CSV is empty")
        has_grip = "grip_mu" in (reader.fieldnames or ())
        return cls(
            x_m=np.asarray([float(row["x_m"]) for row in rows]),
            y_m=np.asarray([float(row["y_m"]) for row in rows]),
            width_left_m=np.asarray([float(row["width_left_m"]) for row in rows]),
            width_right_m=np.asarray([float(row["width_right_m"]) for row in rows]),
            local_grip_mu=(
                np.asarray([float(row["grip_mu"]) for row in rows])
                if has_grip
                else None
            ),
            name=name or csv_path.stem,
        )

    def to_csv(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["x_m", "y_m", "width_left_m", "width_right_m"]
        if self.local_grip_mu is not None:
            fieldnames.append("grip_mu")
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(self.point_count):
                row = {
                    "x_m": self.x_m[index],
                    "y_m": self.y_m[index],
                    "width_left_m": self.width_left_m[index],
                    "width_right_m": self.width_right_m[index],
                }
                if self.local_grip_mu is not None:
                    row["grip_mu"] = self.local_grip_mu[index]
                writer.writerow(row)
        return destination


def make_demo_track(point_count: int = 180) -> Track:
    """Create a deterministic, non-symmetric synthetic test circuit."""

    if point_count < 32:
        raise ValueError("demo track needs at least 32 points")
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    radius = (
        105.0
        + 18.0 * np.sin(2.0 * theta + 0.25)
        - 9.0 * np.cos(3.0 * theta)
        + 5.0 * np.sin(5.0 * theta)
    )
    x = radius * np.cos(theta) + 10.0 * np.cos(2.0 * theta)
    y = 0.70 * radius * np.sin(theta) + 5.0 * np.sin(3.0 * theta)
    left = 6.0 + 0.45 * np.sin(theta - 0.4)
    right = 6.0 + 0.45 * np.cos(theta + 0.7)
    return Track(x, y, left, right, name="synthetic_club_circuit")


def make_silverstone_track(point_count: int | None = None) -> Track:
    """Load the bundled clockwise Silverstone Grand Prix circuit dataset.

    The source columns are ``x, y, right width, left width``. They are mapped
    into :class:`Track`'s left/right convention here so the vendored CSV can
    remain numerically identical to the pinned upstream research dataset.
    When ``point_count`` is supplied, periodic arc-length resampling produces
    a smaller or larger portable track while retaining interpolated widths.
    """

    source = resources.files("racing_line").joinpath("data", "Silverstone.csv")
    with resources.as_file(source) as path:
        values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] < 8:
        raise RuntimeError("bundled Silverstone circuit data is malformed")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("bundled Silverstone circuit data is not finite")

    track = Track(
        x_m=values[:, 0],
        y_m=values[:, 1],
        width_left_m=values[:, 3],
        width_right_m=values[:, 2],
        name="Silverstone Circuit GP",
    )
    if point_count is None or point_count == track.point_count:
        return track
    if isinstance(point_count, bool) or int(point_count) != point_count:
        raise ValueError("Silverstone point_count must be an integer")
    if point_count < 32:
        raise ValueError("Silverstone track needs at least 32 points")

    from .geometry import resample_closed_path

    x_m, y_m, width_left_m, width_right_m = resample_closed_path(
        track.x_m,
        track.y_m,
        int(point_count),
        track.width_left_m,
        track.width_right_m,
    )
    return Track(
        x_m=x_m,
        y_m=y_m,
        width_left_m=width_left_m,
        width_right_m=width_right_m,
        name=track.name,
    )
