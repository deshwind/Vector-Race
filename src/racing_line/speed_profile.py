"""Closed-loop speed-envelope and lap-time calculations.

The model first limits speed by lateral friction and path curvature.  It then
performs repeated cyclic forward-acceleration and backward-braking sweeps until
the closed-loop constraints converge.  No sample is privileged as a start
line, aside from negligible floating-point ordering effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import path_geometry


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpeedProfileConfig:
    """Vehicle and numerical settings for a closed-track speed profile."""

    friction_coefficient: float | ArrayLike = 1.5
    gravity: float = 9.80665
    safety_factor: float = 0.95
    max_speed: float = 100.0
    max_acceleration: float = 8.0
    max_braking: float = 12.0
    curvature_epsilon: float = 1.0e-10
    max_iterations: int = 1000
    tolerance: float = 1.0e-7
    use_friction_circle: bool = False


@dataclass(frozen=True)
class SpeedProfileDiagnostics:
    """Deterministic convergence and active-constraint information."""

    converged: bool
    iterations: int
    final_max_speed_change: float
    max_constraint_violation: float
    curvature_limited_points: int
    acceleration_limited_segments: int
    braking_limited_segments: int


@dataclass(frozen=True)
class SpeedProfileResult:
    """A speed profile sampled at the input path points."""

    speed: FloatArray
    speed_envelope: FloatArray
    s: FloatArray
    segment_lengths: FloatArray
    segment_times: FloatArray
    longitudinal_acceleration: FloatArray
    lateral_acceleration: FloatArray
    lap_time_s: float
    diagnostics: SpeedProfileDiagnostics

    @property
    def speeds(self) -> FloatArray:
        """Alias for the nodal speed array."""

        return self.speed

    @property
    def lap_time(self) -> float:
        """Alias for lap time in seconds."""

        return self.lap_time_s


def _finite_1d(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_per_sample(
    values: ArrayLike,
    name: str,
    sample_count: int,
) -> FloatArray:
    """Return a positive scalar-or-vector input aligned to path samples."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        scalar = float(array)
        if not np.isfinite(scalar) or scalar <= 0.0:
            raise ValueError(f"{name} must contain finite positive values")
        return np.full(sample_count, scalar, dtype=float)
    if array.ndim != 1 or array.size != sample_count:
        raise ValueError(f"{name} must be a scalar or match the path length")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def _validate_config(config: SpeedProfileConfig) -> None:
    friction = np.asarray(config.friction_coefficient, dtype=float)
    if friction.ndim > 1 or friction.size == 0:
        raise ValueError(
            "friction_coefficient must be a positive scalar or one-dimensional array"
        )
    if not np.all(np.isfinite(friction)) or np.any(friction <= 0.0):
        raise ValueError("friction_coefficient must contain finite positive values")
    positive = {
        "gravity": config.gravity,
        "safety_factor": config.safety_factor,
        "max_speed": config.max_speed,
        "curvature_epsilon": config.curvature_epsilon,
        "tolerance": config.tolerance,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.safety_factor > 1.0:
        raise ValueError("safety_factor must be no greater than 1.0")
    for name, value in {
        "max_acceleration": config.max_acceleration,
        "max_braking": config.max_braking,
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        isinstance(config.max_iterations, bool)
        or int(config.max_iterations) != config.max_iterations
        or config.max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(config.use_friction_circle, (bool, np.bool_)):
        raise ValueError("use_friction_circle must be boolean")


def curvature_speed_envelope(
    curvature: ArrayLike,
    friction_coefficient: float | ArrayLike,
    max_speed: float,
    *,
    gravity: float = 9.80665,
    safety_factor: float = 1.0,
    curvature_epsilon: float = 1.0e-10,
    speed_limits: ArrayLike | None = None,
) -> FloatArray:
    """Return the friction/curvature speed ceiling at each path point.

    The lateral-grip relationship is ``v <= sqrt(mu*g/abs(kappa))``.  A
    scalar friction coefficient is applied everywhere; a one-dimensional
    array supplies one coefficient per path point.  The safety factor scales
    available lateral acceleration, not speed directly.  Optional per-point
    ``speed_limits`` are combined by taking the minimum.
    """

    kappa = _finite_1d(curvature, "curvature")
    if kappa.size < 3:
        raise ValueError("curvature must contain at least three samples")
    for name, value in {
        "max_speed": max_speed,
        "gravity": gravity,
        "safety_factor": safety_factor,
        "curvature_epsilon": curvature_epsilon,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if safety_factor > 1.0:
        raise ValueError("safety_factor must be no greater than 1.0")

    friction = _positive_per_sample(
        friction_coefficient,
        "friction_coefficient",
        kappa.size,
    )
    lateral_limit = friction * gravity * safety_factor
    absolute_curvature = np.abs(kappa)
    envelope = np.full(kappa.size, float(max_speed), dtype=float)
    curved = absolute_curvature > curvature_epsilon
    envelope[curved] = np.minimum(
        envelope[curved],
        np.sqrt(lateral_limit[curved] / absolute_curvature[curved]),
    )
    if speed_limits is not None:
        limits = np.asarray(speed_limits, dtype=float)
        if limits.ndim == 0:
            limits = np.full(kappa.size, float(limits), dtype=float)
        if limits.shape != kappa.shape:
            raise ValueError("speed_limits must be a scalar or match curvature length")
        if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
            raise ValueError("speed_limits must contain finite positive values")
        envelope = np.minimum(envelope, limits)
    return envelope


def _available_longitudinal_acceleration(
    base_limit: float,
    speed: float,
    curvature: float,
    lateral_limit: float,
    use_friction_circle: bool,
) -> float:
    if not use_friction_circle:
        return base_limit
    lateral_fraction = min(1.0, abs(curvature) * speed * speed / lateral_limit)
    grip_available = lateral_limit * np.sqrt(max(0.0, 1.0 - lateral_fraction**2))
    return min(base_limit, float(grip_available))


def cyclic_speed_passes(
    speed_envelope: ArrayLike,
    segment_lengths: ArrayLike,
    max_acceleration: float,
    max_braking: float,
    *,
    curvature: ArrayLike | None = None,
    lateral_acceleration_limit: float | ArrayLike | None = None,
    use_friction_circle: bool = False,
    max_iterations: int = 1000,
    tolerance: float = 1.0e-7,
) -> tuple[FloatArray, SpeedProfileDiagnostics]:
    """Apply cyclic forward acceleration and backward braking constraints.

    ``segment_lengths[i]`` joins node ``i`` to node ``(i + 1) % n``.
    Speeds are monotonically reduced from ``speed_envelope`` and the sweeps are
    repeated because a closed track has no fixed initial speed.  When the
    friction circle is enabled, ``lateral_acceleration_limit`` may be a scalar
    or one positive value per path point.
    """

    envelope = _finite_1d(speed_envelope, "speed_envelope")
    lengths = _finite_1d(segment_lengths, "segment_lengths")
    if envelope.size < 3 or lengths.shape != envelope.shape:
        raise ValueError("speed_envelope and segment_lengths must match and have length >= 3")
    if np.any(envelope <= 0.0):
        raise ValueError("speed_envelope must contain positive values")
    if np.any(lengths <= 0.0):
        raise ValueError("segment_lengths must contain positive values")
    for name, value in {
        "max_acceleration": max_acceleration,
        "max_braking": max_braking,
    }.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        isinstance(max_iterations, bool)
        or int(max_iterations) != max_iterations
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    if curvature is None:
        kappa = np.zeros_like(envelope)
    else:
        kappa = _finite_1d(curvature, "curvature")
        if kappa.shape != envelope.shape:
            raise ValueError("curvature must match speed_envelope length")
    if use_friction_circle:
        if lateral_acceleration_limit is None:
            raise ValueError(
                "lateral_acceleration_limit is required when use_friction_circle is true"
            )
        lateral_limit = _positive_per_sample(
            lateral_acceleration_limit,
            "lateral_acceleration_limit",
            envelope.size,
        )
    else:
        lateral_limit = np.ones_like(envelope)

    n_points = envelope.size
    speed = envelope.copy()
    converged = False
    final_change = float("inf")
    iterations = 0
    for iteration in range(1, int(max_iterations) + 1):
        previous_speed = speed.copy()

        for i in range(n_points):
            following = (i + 1) % n_points
            acceleration = _available_longitudinal_acceleration(
                float(max_acceleration),
                float(speed[i]),
                float(kappa[i]),
                float(lateral_limit[i]),
                use_friction_circle,
            )
            reachable = np.sqrt(max(0.0, speed[i] ** 2 + 2.0 * acceleration * lengths[i]))
            if reachable < speed[following]:
                speed[following] = reachable

        for i in range(n_points - 1, -1, -1):
            following = (i + 1) % n_points
            braking = _available_longitudinal_acceleration(
                float(max_braking),
                float(speed[i]),
                float(kappa[i]),
                float(lateral_limit[i]),
                use_friction_circle,
            )
            admissible = np.sqrt(max(0.0, speed[following] ** 2 + 2.0 * braking * lengths[i]))
            if admissible < speed[i]:
                speed[i] = admissible

        final_change = float(np.max(previous_speed - speed))
        iterations = iteration
        if final_change <= tolerance:
            converged = True
            break

    following_speed = np.roll(speed, -1)
    forward_reachable = np.empty(n_points, dtype=float)
    backward_admissible = np.empty(n_points, dtype=float)
    for i in range(n_points):
        acceleration = _available_longitudinal_acceleration(
            float(max_acceleration),
            float(speed[i]),
            float(kappa[i]),
            float(lateral_limit[i]),
            use_friction_circle,
        )
        braking = _available_longitudinal_acceleration(
            float(max_braking),
            float(speed[i]),
            float(kappa[i]),
            float(lateral_limit[i]),
            use_friction_circle,
        )
        forward_reachable[i] = np.sqrt(
            max(0.0, speed[i] ** 2 + 2.0 * acceleration * lengths[i])
        )
        backward_admissible[i] = np.sqrt(
            max(0.0, following_speed[i] ** 2 + 2.0 * braking * lengths[i])
        )

    constraint_violation = max(
        0.0,
        float(np.max(following_speed - forward_reachable)),
        float(np.max(speed - backward_admissible)),
        float(np.max(speed - envelope)),
    )
    active_tolerance = max(10.0 * tolerance, 1.0e-6)
    acceleration_active = np.abs(following_speed - forward_reachable) <= active_tolerance
    braking_active = np.abs(speed - backward_admissible) <= active_tolerance
    curvature_limited = np.abs(speed - envelope) <= active_tolerance
    diagnostics = SpeedProfileDiagnostics(
        converged=converged,
        iterations=iterations,
        final_max_speed_change=final_change,
        max_constraint_violation=constraint_violation,
        curvature_limited_points=int(np.count_nonzero(curvature_limited)),
        acceleration_limited_segments=int(np.count_nonzero(acceleration_active)),
        braking_limited_segments=int(np.count_nonzero(braking_active)),
    )
    return speed, diagnostics


def lap_time_from_speed(
    speed: ArrayLike,
    segment_lengths: ArrayLike,
) -> tuple[float, FloatArray]:
    """Return closed-lap time and per-segment times.

    Segment time is ``2*ds/(v_i + v_{i+1})``, the exact expression for
    constant longitudinal acceleration over a segment.
    """

    velocity = _finite_1d(speed, "speed")
    lengths = _finite_1d(segment_lengths, "segment_lengths")
    if velocity.shape != lengths.shape or velocity.size < 3:
        raise ValueError("speed and segment_lengths must match and have length >= 3")
    if np.any(velocity < 0.0):
        raise ValueError("speed cannot contain negative values")
    if np.any(lengths <= 0.0):
        raise ValueError("segment_lengths must contain positive values")
    denominator = velocity + np.roll(velocity, -1)
    segment_times = np.divide(
        2.0 * lengths,
        denominator,
        out=np.full_like(lengths, np.inf),
        where=denominator > np.finfo(float).tiny,
    )
    return float(np.sum(segment_times)), segment_times


def compute_speed_profile(
    curvature: ArrayLike,
    segment_lengths: ArrayLike,
    *,
    config: SpeedProfileConfig | None = None,
    speed_limits: ArrayLike | None = None,
) -> SpeedProfileResult:
    """Compute a closed speed profile from curvature and segment lengths."""

    config = config or SpeedProfileConfig()
    _validate_config(config)
    kappa = _finite_1d(curvature, "curvature")
    lengths = _finite_1d(segment_lengths, "segment_lengths")
    if kappa.shape != lengths.shape or kappa.size < 3:
        raise ValueError("curvature and segment_lengths must match and have length >= 3")
    if np.any(lengths <= 0.0):
        raise ValueError("segment_lengths must contain positive values")

    friction = _positive_per_sample(
        config.friction_coefficient,
        "friction_coefficient",
        kappa.size,
    )
    envelope = curvature_speed_envelope(
        kappa,
        friction,
        config.max_speed,
        gravity=config.gravity,
        safety_factor=config.safety_factor,
        curvature_epsilon=config.curvature_epsilon,
        speed_limits=speed_limits,
    )
    lateral_limit = friction * config.gravity * config.safety_factor
    speed, diagnostics = cyclic_speed_passes(
        envelope,
        lengths,
        config.max_acceleration,
        config.max_braking,
        curvature=kappa,
        lateral_acceleration_limit=lateral_limit,
        use_friction_circle=config.use_friction_circle,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
    )
    lap_time, segment_times = lap_time_from_speed(speed, lengths)
    following_speed = np.roll(speed, -1)
    longitudinal_acceleration = (following_speed**2 - speed**2) / (2.0 * lengths)
    lateral_acceleration = speed**2 * kappa
    s = np.concatenate(([0.0], np.cumsum(lengths[:-1])))
    return SpeedProfileResult(
        speed=speed,
        speed_envelope=envelope,
        s=s,
        segment_lengths=lengths,
        segment_times=segment_times,
        longitudinal_acceleration=longitudinal_acceleration,
        lateral_acceleration=lateral_acceleration,
        lap_time_s=lap_time,
        diagnostics=diagnostics,
    )


def generate_speed_profile(
    x: ArrayLike,
    y: ArrayLike,
    curvature: ArrayLike | None = None,
    mu: float | ArrayLike = 1.5,
    max_speed: float = 100.0,
    max_accel: float = 8.0,
    max_brake: float = 12.0,
    gravity: float = 9.80665,
    safety_factor: float = 0.95,
    max_iterations: int = 1000,
    tolerance: float = 1.0e-7,
    *,
    speed_limits: ArrayLike | None = None,
    use_friction_circle: bool = False,
) -> SpeedProfileResult:
    """Integration-friendly x/y wrapper returning speeds and lap time.

    If ``curvature`` is omitted, it is calculated from a periodic spline.
    ``mu`` may be a scalar or one friction coefficient per path point.  A
    repeated closing endpoint is accepted and removed; curvature, friction,
    or speed-limit arrays that include that endpoint are trimmed
    correspondingly.
    """

    raw_x = _finite_1d(x, "x")
    raw_y = _finite_1d(y, "y")
    if raw_x.shape != raw_y.shape or raw_x.size < 3:
        raise ValueError("x and y must match and contain at least three points")
    geometry = path_geometry(raw_x, raw_y)
    scale = max(
        1.0,
        float(np.max(np.ptp(np.column_stack((raw_x, raw_y)), axis=0))),
    )
    repeated_endpoint = bool(
        np.hypot(raw_x[-1] - raw_x[0], raw_y[-1] - raw_y[0])
        <= 64.0 * np.finfo(float).eps * scale
    )
    expected_size = raw_x.size - int(repeated_endpoint)
    if geometry.x.size != expected_size:
        raise ValueError(
            "path contains consecutive duplicate points; clean or resample it first"
        )

    if curvature is None:
        kappa = geometry.curvature
    else:
        kappa = _finite_1d(curvature, "curvature")
        if repeated_endpoint and kappa.size == raw_x.size:
            kappa = kappa[:-1]
        if kappa.size != geometry.x.size:
            raise ValueError(f"curvature must have length {geometry.x.size}")

    aligned_limits = speed_limits
    if speed_limits is not None:
        limits = np.asarray(speed_limits, dtype=float)
        if repeated_endpoint and limits.ndim == 1 and limits.size == raw_x.size:
            limits = limits[:-1]
        aligned_limits = limits

    friction = np.asarray(mu, dtype=float)
    if repeated_endpoint and friction.ndim == 1 and friction.size == raw_x.size:
        friction = friction[:-1]
    aligned_friction: float | ArrayLike
    if friction.ndim == 0:
        aligned_friction = float(friction)
    else:
        aligned_friction = friction

    config = SpeedProfileConfig(
        friction_coefficient=aligned_friction,
        gravity=gravity,
        safety_factor=safety_factor,
        max_speed=max_speed,
        max_acceleration=max_accel,
        max_braking=max_brake,
        max_iterations=max_iterations,
        tolerance=tolerance,
        use_friction_circle=use_friction_circle,
    )
    return compute_speed_profile(
        kappa,
        geometry.segment_lengths,
        config=config,
        speed_limits=aligned_limits,
    )


__all__ = [
    "SpeedProfileConfig",
    "SpeedProfileDiagnostics",
    "SpeedProfileResult",
    "compute_speed_profile",
    "curvature_speed_envelope",
    "cyclic_speed_passes",
    "generate_speed_profile",
    "lap_time_from_speed",
]
