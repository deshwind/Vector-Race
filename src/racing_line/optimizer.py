"""Bounded minimum-curvature racing-line optimisation.

The decision variable is one signed lateral offset per centreline point.
Positive offsets follow the left-pointing normals from
``racing_line.geometry``.  The objective uses the exact curvature of the
offset samples under periodic, non-uniform finite differences, plus optional
offset smoothness and regularisation terms.  Bounds account for vehicle width
and a safety margin on both sides of the car.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, minimize

from .geometry import lateral_offset_bounds, path_geometry


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MinimumCurvatureConfig:
    """Configuration for the bounded lateral-offset optimiser.

    ``smoothness_weight`` multiplies the summed squared second derivative of
    lateral offset.  ``offset_regularization`` multiplies summed squared offset;
    a small positive value resolves otherwise equivalent solutions on long
    straights.  Both must be non-negative.
    """

    vehicle_width_m: float = 2.0
    safety_margin_m: float = 0.25
    smoothness_weight: float = 0.05
    offset_regularization: float = 1.0e-8
    max_iterations: int = 300
    function_tolerance: float = 1.0e-11
    gradient_tolerance: float = 1.0e-7
    max_line_search_steps: int = 40


@dataclass(frozen=True)
class OptimizationDiagnostics:
    """Deterministic solver and solution-quality diagnostics."""

    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    initial_objective: float
    final_objective: float
    initial_curvature_rms: float
    final_curvature_rms: float
    projected_gradient_inf_norm: float
    active_lower_bounds: int
    active_upper_bounds: int
    centreline_spacing_cv: float
    objective_history: tuple[float, ...]


@dataclass(frozen=True)
class MinimumCurvatureResult:
    """Optimised racing line and solver status."""

    x: FloatArray
    y: FloatArray
    offset: FloatArray
    curvature: FloatArray
    lower_bound: FloatArray
    upper_bound: FloatArray
    success: bool
    message: str
    diagnostics: OptimizationDiagnostics

    @property
    def points(self) -> FloatArray:
        return np.column_stack((self.x, self.y))

    @property
    def offsets(self) -> FloatArray:
        """Alias for callers that prefer a plural array name."""

        return self.offset


def _validate_config(config: MinimumCurvatureConfig) -> None:
    finite_nonnegative = {
        "vehicle_width_m": config.vehicle_width_m,
        "safety_margin_m": config.safety_margin_m,
        "smoothness_weight": config.smoothness_weight,
        "offset_regularization": config.offset_regularization,
    }
    for name, value in finite_nonnegative.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if isinstance(config.max_iterations, bool) or config.max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if int(config.max_iterations) != config.max_iterations:
        raise ValueError("max_iterations must be an integer")
    if (
        not np.isfinite(config.function_tolerance)
        or config.function_tolerance <= 0.0
    ):
        raise ValueError("function_tolerance must be finite and positive")
    if (
        not np.isfinite(config.gradient_tolerance)
        or config.gradient_tolerance <= 0.0
    ):
        raise ValueError("gradient_tolerance must be finite and positive")
    if (
        isinstance(config.max_line_search_steps, bool)
        or int(config.max_line_search_steps) != config.max_line_search_steps
        or config.max_line_search_steps < 1
    ):
        raise ValueError("max_line_search_steps must be a positive integer")


def _aligned_field(
    values: ArrayLike,
    original_size: int,
    result_size: int,
    repeated_endpoint: bool,
    name: str,
) -> ArrayLike:
    """Trim a field paired with an accepted repeated closing endpoint."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return float(array)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a one-dimensional array")
    if array.size == result_size:
        return array
    if repeated_endpoint and array.size == original_size and result_size == original_size - 1:
        return array[:-1]
    raise ValueError(f"{name} must be a scalar or an array of length {result_size}")


class _MinimumCurvatureObjective:
    """Exact objective and analytic gradient for fixed centreline normals."""

    def __init__(
        self,
        centreline: FloatArray,
        normals: FloatArray,
        segment_lengths: FloatArray,
        smoothness_weight: float,
        regularization_weight: float,
    ) -> None:
        self.centreline = centreline
        self.normals = normals
        self.n = centreline.shape[0]
        self.smoothness_weight = float(smoothness_weight)
        self.regularization_weight = float(regularization_weight)

        h_previous = np.roll(segment_lengths, 1)
        h_next = segment_lengths
        total = h_previous + h_next
        self.first_previous = -h_next / (h_previous * total)
        self.first_current = (h_next - h_previous) / (h_previous * h_next)
        self.first_next = h_previous / (h_next * total)
        self.second_previous = 2.0 / (h_previous * total)
        self.second_current = -2.0 / (h_previous * h_next)
        self.second_next = 2.0 / (h_next * total)

    def _derivatives(self, values: FloatArray) -> tuple[FloatArray, FloatArray]:
        previous = np.roll(values, 1, axis=0)
        following = np.roll(values, -1, axis=0)
        if values.ndim == 2:
            first = (
                self.first_previous[:, None] * previous
                + self.first_current[:, None] * values
                + self.first_next[:, None] * following
            )
            second = (
                self.second_previous[:, None] * previous
                + self.second_current[:, None] * values
                + self.second_next[:, None] * following
            )
        else:
            first = (
                self.first_previous * previous
                + self.first_current * values
                + self.first_next * following
            )
            second = (
                self.second_previous * previous
                + self.second_current * values
                + self.second_next * following
            )
        return first, second

    @staticmethod
    def _transpose_three_point(
        values: FloatArray,
        previous_coeff: FloatArray,
        current_coeff: FloatArray,
        next_coeff: FloatArray,
    ) -> FloatArray:
        """Apply the transpose of a periodic three-point stencil."""

        if values.ndim == 2:
            previous_term = previous_coeff[:, None] * values
            current_term = current_coeff[:, None] * values
            next_term = next_coeff[:, None] * values
        else:
            previous_term = previous_coeff * values
            current_term = current_coeff * values
            next_term = next_coeff * values
        return (
            np.roll(previous_term, -1, axis=0)
            + current_term
            + np.roll(next_term, 1, axis=0)
        )

    def points(self, offsets: FloatArray) -> FloatArray:
        return self.centreline + offsets[:, None] * self.normals

    def curvature(self, offsets: FloatArray) -> FloatArray:
        first, second = self._derivatives(self.points(offsets))
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        speed_squared = np.einsum("ij,ij->i", first, first)
        denominator = np.maximum(speed_squared, 1.0e-24) ** 1.5
        return cross / denominator

    def value_and_gradient(self, offsets: FloatArray) -> tuple[float, FloatArray]:
        points = self.points(offsets)
        first, second = self._derivatives(points)
        speed_squared = np.einsum("ij,ij->i", first, first)
        safe_speed_squared = np.maximum(speed_squared, 1.0e-24)
        inverse_speed_cubed = safe_speed_squared ** -1.5
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        curvature = cross * inverse_speed_cubed

        # A sum (rather than a mean) keeps per-variable gradients well scaled
        # as sample count grows.  All three objective terms use the same scale,
        # so discretisation does not alter their relative weighting.
        value = float(np.sum(curvature * curvature))
        curvature_factor = 2.0 * curvature
        dcurvature_dfirst = np.column_stack((second[:, 1], -second[:, 0]))
        dcurvature_dfirst *= inverse_speed_cubed[:, None]
        dcurvature_dfirst -= (
            3.0
            * cross[:, None]
            * first
            * (safe_speed_squared ** -2.5)[:, None]
        )
        dcurvature_dsecond = np.column_stack((-first[:, 1], first[:, 0]))
        dcurvature_dsecond *= inverse_speed_cubed[:, None]
        gradient_first = curvature_factor[:, None] * dcurvature_dfirst
        gradient_second = curvature_factor[:, None] * dcurvature_dsecond

        point_gradient = self._transpose_three_point(
            gradient_first,
            self.first_previous,
            self.first_current,
            self.first_next,
        )
        point_gradient += self._transpose_three_point(
            gradient_second,
            self.second_previous,
            self.second_current,
            self.second_next,
        )
        gradient = np.einsum("ij,ij->i", point_gradient, self.normals)

        if self.smoothness_weight:
            _, offset_second = self._derivatives(offsets)
            value += self.smoothness_weight * float(np.sum(offset_second**2))
            gradient += (
                (2.0 * self.smoothness_weight)
                * self._transpose_three_point(
                    offset_second,
                    self.second_previous,
                    self.second_current,
                    self.second_next,
                )
            )
        if self.regularization_weight:
            value += self.regularization_weight * float(np.sum(offsets**2))
            gradient += (2.0 * self.regularization_weight) * offsets
        return value, np.asarray(gradient, dtype=float)


def optimize_racing_line(
    centreline: ArrayLike,
    width_left: ArrayLike,
    width_right: ArrayLike,
    *,
    config: MinimumCurvatureConfig | None = None,
    initial_offsets: ArrayLike | None = None,
) -> MinimumCurvatureResult:
    """Optimise a closed centreline supplied as an ``(n, 2)`` array.

    The input should normally be arc-length-resampled first.  Non-uniform
    spacing is supported, and its coefficient of variation is reported in the
    diagnostics.  A repeated final endpoint is accepted.  Other consecutive
    duplicate points are rejected because their associated width semantics are
    ambiguous.
    """

    config = config or MinimumCurvatureConfig()
    _validate_config(config)
    raw_points = np.asarray(centreline, dtype=float)
    if raw_points.ndim != 2 or raw_points.shape[1] != 2:
        raise ValueError("centreline must have shape (n, 2)")
    if raw_points.shape[0] < 3 or not np.all(np.isfinite(raw_points)):
        raise ValueError("centreline needs at least three finite points")

    scale = max(1.0, float(np.max(np.ptp(raw_points, axis=0))))
    repeated_endpoint = bool(
        np.linalg.norm(raw_points[-1] - raw_points[0])
        <= 64.0 * np.finfo(float).eps * scale
    )
    geometry = path_geometry(raw_points[:, 0], raw_points[:, 1])
    expected_size = raw_points.shape[0] - int(repeated_endpoint)
    if geometry.x.size != expected_size:
        raise ValueError(
            "centreline contains consecutive duplicate points; clean or "
            "resample it before optimisation"
        )
    n_points = geometry.x.size
    left = _aligned_field(
        width_left,
        raw_points.shape[0],
        n_points,
        repeated_endpoint,
        "width_left",
    )
    right = _aligned_field(
        width_right,
        raw_points.shape[0],
        n_points,
        repeated_endpoint,
        "width_right",
    )
    lower, upper = lateral_offset_bounds(
        left,
        right,
        n_points,
        vehicle_width=config.vehicle_width_m,
        safety_margin=config.safety_margin_m,
    )

    centre = geometry.points
    objective = _MinimumCurvatureObjective(
        centre,
        geometry.normals,
        geometry.segment_lengths,
        config.smoothness_weight,
        config.offset_regularization,
    )
    if initial_offsets is None:
        initial = np.clip(np.zeros(n_points, dtype=float), lower, upper)
    else:
        aligned_initial = _aligned_field(
            initial_offsets,
            raw_points.shape[0],
            n_points,
            repeated_endpoint,
            "initial_offsets",
        )
        initial = np.asarray(aligned_initial, dtype=float)
        if initial.ndim == 0:
            initial = np.full(n_points, float(initial), dtype=float)
        if not np.all(np.isfinite(initial)):
            raise ValueError("initial_offsets must contain only finite values")
        initial = np.clip(initial, lower, upper)

    initial_objective, _ = objective.value_and_gradient(initial)
    initial_curvature = objective.curvature(initial)
    history: list[float] = [float(initial_objective)]

    def callback(offsets: FloatArray) -> None:
        value, _ = objective.value_and_gradient(np.asarray(offsets, dtype=float))
        history.append(float(value))

    solution = minimize(
        objective.value_and_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=Bounds(lower, upper, keep_feasible=True),
        callback=callback,
        options={
            "maxiter": int(config.max_iterations),
            "maxfun": max(1000, 25 * int(config.max_iterations)),
            "ftol": float(config.function_tolerance),
            "gtol": float(config.gradient_tolerance),
            "maxls": int(config.max_line_search_steps),
        },
    )

    offsets = np.clip(np.asarray(solution.x, dtype=float), lower, upper)
    final_objective, final_gradient = objective.value_and_gradient(offsets)
    if not np.isclose(history[-1], final_objective, rtol=0.0, atol=1.0e-15):
        history.append(float(final_objective))
    racing_points = objective.points(offsets)
    final_curvature = objective.curvature(offsets)

    bound_tolerance = max(
        1.0e-9,
        1.0e-8 * max(1.0, float(np.max(np.maximum(np.abs(lower), np.abs(upper))))),
    )
    at_lower = offsets <= lower + bound_tolerance
    at_upper = offsets >= upper - bound_tolerance
    projected_gradient = final_gradient.copy()
    projected_gradient[at_lower & (final_gradient > 0.0)] = 0.0
    projected_gradient[at_upper & (final_gradient < 0.0)] = 0.0
    spacing_mean = float(np.mean(geometry.segment_lengths))
    spacing_cv = float(np.std(geometry.segment_lengths) / spacing_mean)

    diagnostics = OptimizationDiagnostics(
        iterations=int(getattr(solution, "nit", 0)),
        function_evaluations=int(getattr(solution, "nfev", 0)),
        gradient_evaluations=int(getattr(solution, "njev", 0)),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        initial_curvature_rms=float(np.sqrt(np.mean(initial_curvature**2))),
        final_curvature_rms=float(np.sqrt(np.mean(final_curvature**2))),
        projected_gradient_inf_norm=float(np.max(np.abs(projected_gradient))),
        active_lower_bounds=int(np.count_nonzero(at_lower)),
        active_upper_bounds=int(np.count_nonzero(at_upper)),
        centreline_spacing_cv=spacing_cv,
        objective_history=tuple(history),
    )
    return MinimumCurvatureResult(
        x=np.asarray(racing_points[:, 0], dtype=float),
        y=np.asarray(racing_points[:, 1], dtype=float),
        offset=offsets,
        curvature=np.asarray(final_curvature, dtype=float),
        lower_bound=lower,
        upper_bound=upper,
        success=bool(solution.success),
        message=str(solution.message),
        diagnostics=diagnostics,
    )


def optimize_minimum_curvature(
    center_x: ArrayLike,
    center_y: ArrayLike,
    width_left: ArrayLike,
    width_right: ArrayLike,
    vehicle_width_m: float = 2.0,
    safety_margin_m: float = 0.25,
    smoothness_weight: float = 0.05,
    offset_regularization: float = 1.0e-8,
    max_iterations: int = 300,
    *,
    initial_offsets: ArrayLike | None = None,
    function_tolerance: float = 1.0e-11,
    gradient_tolerance: float = 1.0e-7,
) -> MinimumCurvatureResult:
    """Integration-friendly x/y wrapper for minimum-curvature optimisation."""

    x = np.asarray(center_x, dtype=float)
    y = np.asarray(center_y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("center_x and center_y must be one-dimensional and equal length")
    config = MinimumCurvatureConfig(
        vehicle_width_m=vehicle_width_m,
        safety_margin_m=safety_margin_m,
        smoothness_weight=smoothness_weight,
        offset_regularization=offset_regularization,
        max_iterations=max_iterations,
        function_tolerance=function_tolerance,
        gradient_tolerance=gradient_tolerance,
    )
    return optimize_racing_line(
        np.column_stack((x, y)),
        width_left,
        width_right,
        config=config,
        initial_offsets=initial_offsets,
    )


__all__ = [
    "MinimumCurvatureConfig",
    "MinimumCurvatureResult",
    "OptimizationDiagnostics",
    "optimize_minimum_curvature",
    "optimize_racing_line",
]
