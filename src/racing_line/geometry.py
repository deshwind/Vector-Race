"""Geometry primitives for deterministic closed-track calculations.

The functions in this module deliberately operate on plain NumPy arrays.  A
closed path is represented without repeating its first point at the end; all
neighbour operations are periodic.  Tangents point in increasing sample order
and normals point to the left of travel, so positive lateral offsets are left
offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PathGeometry:
    """Differential geometry sampled around a closed path.

    ``s`` is the cumulative polygonal arc length at each point and
    ``segment_lengths[i]`` is the distance from point ``i`` to point
    ``(i + 1) % n``.  Curvature is signed: it is positive for a left turn.
    """

    x: FloatArray
    y: FloatArray
    s: FloatArray
    segment_lengths: FloatArray
    lap_length: float
    tangent_x: FloatArray
    tangent_y: FloatArray
    normal_x: FloatArray
    normal_y: FloatArray
    curvature: FloatArray

    @property
    def points(self) -> FloatArray:
        """Return points as an ``(n, 2)`` array."""

        return np.column_stack((self.x, self.y))

    @property
    def tangents(self) -> FloatArray:
        """Return unit tangents as an ``(n, 2)`` array."""

        return np.column_stack((self.tangent_x, self.tangent_y))

    @property
    def normals(self) -> FloatArray:
        """Return left-pointing unit normals as an ``(n, 2)`` array."""

        return np.column_stack((self.normal_x, self.normal_y))


@dataclass(frozen=True)
class TrackBoundaries:
    """Left and right physical track boundaries sampled at matching points."""

    left_x: FloatArray
    left_y: FloatArray
    right_x: FloatArray
    right_y: FloatArray

    @property
    def left(self) -> FloatArray:
        return np.column_stack((self.left_x, self.left_y))

    @property
    def right(self) -> FloatArray:
        return np.column_stack((self.right_x, self.right_y))


def _as_1d_finite(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _coerce_xy(x: ArrayLike, y: ArrayLike) -> FloatArray:
    x_array = _as_1d_finite(x, "x")
    y_array = _as_1d_finite(y, "y")
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same length")
    if x_array.size < 3:
        raise ValueError("a closed path needs at least three points")
    return np.column_stack((x_array, y_array))


def _length_tolerance(points: FloatArray) -> float:
    scale = max(1.0, float(np.ptp(points, axis=0).max(initial=0.0)))
    return 64.0 * np.finfo(float).eps * scale


def _prepare_closed_points(
    x: ArrayLike,
    y: ArrayLike,
    scalar_fields: Sequence[ArrayLike] = (),
) -> tuple[FloatArray, list[FloatArray]]:
    """Remove a repeated endpoint and zero-length consecutive segments."""

    points = _coerce_xy(x, y)
    fields = [_as_1d_finite(field, f"scalar_fields[{i}]") for i, field in enumerate(scalar_fields)]
    for i, field in enumerate(fields):
        if field.size != points.shape[0]:
            raise ValueError(
                f"scalar_fields[{i}] has length {field.size}; expected {points.shape[0]}"
            )

    tolerance = _length_tolerance(points)
    if np.linalg.norm(points[-1] - points[0]) <= tolerance:
        points = points[:-1]
        fields = [field[:-1] for field in fields]

    if points.shape[0] < 3:
        raise ValueError("a closed path needs at least three distinct points")

    previous = np.roll(points, 1, axis=0)
    keep = np.linalg.norm(points - previous, axis=1) > tolerance
    # Always retain the first point.  The closing duplicate, if present, was
    # already handled above.
    keep[0] = True
    points = points[keep]
    fields = [field[keep] for field in fields]

    if points.shape[0] < 3:
        raise ValueError("a closed path needs at least three distinct points")
    closing_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if np.any(closing_lengths <= tolerance):
        raise ValueError("path contains indistinguishable consecutive points")
    return points, fields


def _periodic_splines(
    points: FloatArray,
) -> tuple[FloatArray, CubicSpline, CubicSpline]:
    segment_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    parameter = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    closed = np.vstack((points, points[0]))
    spline_x = CubicSpline(parameter, closed[:, 0], bc_type="periodic")
    spline_y = CubicSpline(parameter, closed[:, 1], bc_type="periodic")
    return parameter, spline_x, spline_y


def resample_closed_path(
    x: ArrayLike,
    y: ArrayLike,
    n_points: int,
    *scalar_fields: ArrayLike,
) -> tuple[FloatArray, ...]:
    """Resample a closed path at near-uniform arc-length intervals.

    A periodic cubic spline is fitted using chord length as its initial
    parameter.  The spline is densely integrated and inverted to obtain an
    approximately arc-length-uniform grid.  Any positional ``scalar_fields``
    (for example left/right widths) are interpolated periodically at the same
    locations and appended to the returned ``(x, y, ...)`` tuple.

    A repeated final copy of the first input point is accepted and removed.
    Consecutive duplicate input points are also removed deterministically.
    Scalar fields must match the original input length.
    """

    if isinstance(n_points, bool) or int(n_points) != n_points or n_points < 3:
        raise ValueError("n_points must be an integer of at least 3")
    n_points = int(n_points)
    points, fields = _prepare_closed_points(x, y, scalar_fields)
    parameter, spline_x, spline_y = _periodic_splines(points)

    # Dense chord integration makes the returned samples much closer to equal
    # true arc length than sampling the original chord-length parameter alone.
    dense_intervals = max(2048, 16 * n_points, 64 * points.shape[0])
    dense_parameter = np.linspace(0.0, parameter[-1], dense_intervals + 1)
    dense_points = np.column_stack(
        (spline_x(dense_parameter), spline_y(dense_parameter))
    )
    dense_steps = np.linalg.norm(np.diff(dense_points, axis=0), axis=1)
    dense_s = np.concatenate(([0.0], np.cumsum(dense_steps)))
    lap_length = float(dense_s[-1])
    if not np.isfinite(lap_length) or lap_length <= _length_tolerance(points):
        raise ValueError("path has negligible total length")

    target_s = np.arange(n_points, dtype=float) * (lap_length / n_points)
    target_parameter = np.interp(target_s, dense_s, dense_parameter)
    outputs: list[FloatArray] = [
        np.asarray(spline_x(target_parameter), dtype=float),
        np.asarray(spline_y(target_parameter), dtype=float),
    ]

    field_parameter = parameter
    for field in fields:
        closed_field = np.concatenate((field, field[:1]))
        field_spline = CubicSpline(field_parameter, closed_field, bc_type="periodic")
        outputs.append(np.asarray(field_spline(target_parameter), dtype=float))
    return tuple(outputs)


def resample_closed_track(points: ArrayLike, n_points: int) -> PathGeometry:
    """Array-oriented convenience wrapper returning resampled geometry."""

    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    x_new, y_new = resample_closed_path(point_array[:, 0], point_array[:, 1], n_points)
    return path_geometry(x_new, y_new)


def path_geometry(x: ArrayLike, y: ArrayLike) -> PathGeometry:
    """Compute arc length, tangent, left normal, and signed curvature.

    Derivatives come from a periodic cubic spline through the supplied points.
    The input need not be perfectly uniformly spaced.  For best-conditioned
    curvature estimates, call :func:`resample_closed_path` first.
    """

    points, _ = _prepare_closed_points(x, y)
    parameter, spline_x, spline_y = _periodic_splines(points)
    sample_parameter = parameter[:-1]
    dx = np.asarray(spline_x(sample_parameter, 1), dtype=float)
    dy = np.asarray(spline_y(sample_parameter, 1), dtype=float)
    ddx = np.asarray(spline_x(sample_parameter, 2), dtype=float)
    ddy = np.asarray(spline_y(sample_parameter, 2), dtype=float)

    derivative_norm = np.hypot(dx, dy)
    derivative_tolerance = 256.0 * np.finfo(float).eps
    bad = derivative_norm <= derivative_tolerance
    if np.any(bad):
        # Periodic centred chords provide a stable fallback for pathological
        # spline stationary points.
        chord = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        chord_norm = np.linalg.norm(chord, axis=1)
        if np.any(chord_norm[bad] <= _length_tolerance(points)):
            raise ValueError("cannot determine a tangent at one or more path points")
        dx[bad] = chord[bad, 0]
        dy[bad] = chord[bad, 1]
        derivative_norm[bad] = chord_norm[bad]

    tangent_x = dx / derivative_norm
    tangent_y = dy / derivative_norm
    normal_x = -tangent_y
    normal_y = tangent_x
    curvature = (dx * ddy - dy * ddx) / np.maximum(
        derivative_norm**3, np.finfo(float).tiny
    )

    segment_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    lap_length = float(np.sum(segment_lengths))
    s = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
    return PathGeometry(
        x=np.asarray(points[:, 0], dtype=float),
        y=np.asarray(points[:, 1], dtype=float),
        s=np.asarray(s, dtype=float),
        segment_lengths=np.asarray(segment_lengths, dtype=float),
        lap_length=lap_length,
        tangent_x=tangent_x,
        tangent_y=tangent_y,
        normal_x=normal_x,
        normal_y=normal_y,
        curvature=np.asarray(curvature, dtype=float),
    )


def arc_length(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray, float]:
    """Return ``(s, segment_lengths, lap_length)`` for a closed path."""

    geometry = path_geometry(x, y)
    return geometry.s, geometry.segment_lengths, geometry.lap_length


def unit_tangents(x: ArrayLike, y: ArrayLike) -> FloatArray:
    """Return closed-path unit tangents as an ``(n, 2)`` array."""

    return path_geometry(x, y).tangents


def unit_normals(x: ArrayLike, y: ArrayLike) -> FloatArray:
    """Return left-pointing unit normals as an ``(n, 2)`` array."""

    return path_geometry(x, y).normals


def signed_curvature(x: ArrayLike, y: ArrayLike) -> FloatArray:
    """Return periodic signed curvature, positive for left turns."""

    return path_geometry(x, y).curvature


def _broadcast_width(values: ArrayLike, n_points: int, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(n_points, float(array), dtype=float)
    elif array.ndim == 1 and array.size == n_points:
        array = np.asarray(array, dtype=float)
    else:
        raise ValueError(f"{name} must be a scalar or an array of length {n_points}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} cannot contain negative widths")
    return array


def lateral_offset_bounds(
    width_left: ArrayLike,
    width_right: ArrayLike,
    n_points: int,
    *,
    vehicle_width: float = 0.0,
    safety_margin: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Return feasible lateral-offset lower and upper bounds.

    Positive offset is left.  The vehicle centre must remain half a vehicle
    width plus ``safety_margin`` inside both physical boundaries.
    """

    if isinstance(n_points, bool) or int(n_points) != n_points or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    n_points = int(n_points)
    left = _broadcast_width(width_left, n_points, "width_left")
    right = _broadcast_width(width_right, n_points, "width_right")
    if not np.isfinite(vehicle_width) or vehicle_width < 0.0:
        raise ValueError("vehicle_width must be finite and non-negative")
    if not np.isfinite(safety_margin) or safety_margin < 0.0:
        raise ValueError("safety_margin must be finite and non-negative")
    clearance = 0.5 * float(vehicle_width) + float(safety_margin)
    lower = -right + clearance
    upper = left - clearance
    infeasible = np.flatnonzero(lower > upper)
    if infeasible.size:
        first = int(infeasible[0])
        raise ValueError(
            "vehicle and safety margins do not fit inside the track at "
            f"sample {first}: required clearance {clearance:.6g} m per side, "
            f"available widths are left={left[first]:.6g} m and "
            f"right={right[first]:.6g} m"
        )
    return lower, upper


def offset_path(
    x: ArrayLike,
    y: ArrayLike,
    offsets: ArrayLike,
    *,
    normals: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Apply signed lateral offsets to a closed path."""

    points = _coerce_xy(x, y)
    if np.linalg.norm(points[-1] - points[0]) <= _length_tolerance(points):
        points = points[:-1]
    offset_array = _as_1d_finite(offsets, "offsets")
    if offset_array.size != points.shape[0]:
        raise ValueError(f"offsets must have length {points.shape[0]}")

    if normals is None:
        normal_array = path_geometry(points[:, 0], points[:, 1]).normals
    else:
        normal_array = np.asarray(normals, dtype=float)
        if normal_array.shape != points.shape or not np.all(np.isfinite(normal_array)):
            raise ValueError(f"normals must be finite and have shape {points.shape}")
        norm = np.linalg.norm(normal_array, axis=1)
        if np.any(norm <= np.finfo(float).eps):
            raise ValueError("normals must be non-zero")
        normal_array = normal_array / norm[:, None]

    offset_points = points + offset_array[:, None] * normal_array
    return offset_points[:, 0], offset_points[:, 1]


def track_boundaries(
    x: ArrayLike,
    y: ArrayLike,
    width_left: ArrayLike,
    width_right: ArrayLike,
    *,
    normals: ArrayLike | None = None,
) -> TrackBoundaries:
    """Construct physical left and right boundaries from centreline widths."""

    points = _coerce_xy(x, y)
    if np.linalg.norm(points[-1] - points[0]) <= _length_tolerance(points):
        points = points[:-1]
    n_points = points.shape[0]
    left = _broadcast_width(width_left, n_points, "width_left")
    right = _broadcast_width(width_right, n_points, "width_right")
    if normals is None:
        normal_array = path_geometry(points[:, 0], points[:, 1]).normals
    else:
        normal_array = np.asarray(normals, dtype=float)
        if normal_array.shape != points.shape or not np.all(np.isfinite(normal_array)):
            raise ValueError(f"normals must be finite and have shape {points.shape}")
        norm = np.linalg.norm(normal_array, axis=1)
        if np.any(norm <= np.finfo(float).eps):
            raise ValueError("normals must be non-zero")
        normal_array = normal_array / norm[:, None]

    left_points = points + left[:, None] * normal_array
    right_points = points - right[:, None] * normal_array
    return TrackBoundaries(
        left_x=left_points[:, 0],
        left_y=left_points[:, 1],
        right_x=right_points[:, 0],
        right_y=right_points[:, 1],
    )


__all__ = [
    "PathGeometry",
    "TrackBoundaries",
    "arc_length",
    "lateral_offset_bounds",
    "offset_path",
    "path_geometry",
    "resample_closed_path",
    "resample_closed_track",
    "signed_curvature",
    "track_boundaries",
    "unit_normals",
    "unit_tangents",
]
