from __future__ import annotations

import numpy as np
import pytest

from racing_line.geometry import path_geometry
from racing_line.speed_profile import (
    curvature_speed_envelope,
    cyclic_speed_passes,
    generate_speed_profile,
)


def test_curvature_envelope_uses_local_friction_coefficients() -> None:
    curvature = np.array([0.02, -0.02, 0.02, 0.0])
    friction = np.array([0.5, 1.0, 2.0, 0.25])

    envelope = curvature_speed_envelope(
        curvature,
        friction,
        max_speed=100.0,
        gravity=10.0,
        safety_factor=0.8,
    )

    expected = np.array(
        [
            np.sqrt(0.5 * 10.0 * 0.8 / 0.02),
            np.sqrt(1.0 * 10.0 * 0.8 / 0.02),
            np.sqrt(2.0 * 10.0 * 0.8 / 0.02),
            100.0,
        ]
    )
    np.testing.assert_allclose(envelope, expected)

    scalar = curvature_speed_envelope(curvature, 1.25, 100.0)
    constant_array = curvature_speed_envelope(
        curvature,
        np.full(curvature.size, 1.25),
        100.0,
    )
    np.testing.assert_array_equal(scalar, constant_array)


def test_friction_circle_uses_local_longitudinal_limits() -> None:
    speed_envelope = np.array([5.0, 15.0, 15.0, 15.0] * 2)
    segment_lengths = np.full(speed_envelope.size, 5.0)
    curvature = np.full(speed_envelope.size, 0.005)
    lateral_limits = np.array([2.0, 20.0, 20.0, 2.0] * 2)
    max_acceleration = 100.0
    max_braking = 100.0

    local_speed, diagnostics = cyclic_speed_passes(
        speed_envelope,
        segment_lengths,
        max_acceleration,
        max_braking,
        curvature=curvature,
        lateral_acceleration_limit=lateral_limits,
        use_friction_circle=True,
        tolerance=1.0e-10,
    )
    uniform_speed, _ = cyclic_speed_passes(
        speed_envelope,
        segment_lengths,
        max_acceleration,
        max_braking,
        curvature=curvature,
        lateral_acceleration_limit=20.0,
        use_friction_circle=True,
        tolerance=1.0e-10,
    )

    assert diagnostics.converged
    assert diagnostics.max_constraint_violation <= 1.0e-10
    assert local_speed[1] < uniform_speed[1]
    assert local_speed[3] < uniform_speed[3]

    lateral_acceleration = np.abs(curvature) * local_speed**2
    longitudinal_available = np.sqrt(
        np.maximum(0.0, lateral_limits**2 - lateral_acceleration**2)
    )
    acceleration_available = np.minimum(max_acceleration, longitudinal_available)
    braking_available = np.minimum(max_braking, longitudinal_available)
    speed_squared_change = np.roll(local_speed, -1) ** 2 - local_speed**2
    assert np.all(
        speed_squared_change
        <= 2.0 * acceleration_available * segment_lengths + 1.0e-9
    )
    assert np.all(
        -speed_squared_change
        <= 2.0 * braking_available * segment_lengths + 1.0e-9
    )


def test_generate_profile_trims_repeated_endpoint_from_local_grip() -> None:
    point_count = 24
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    x_m = 30.0 * np.cos(theta)
    y_m = 20.0 * np.sin(theta)
    geometry = path_geometry(x_m, y_m)
    friction = 1.1 + 0.2 * np.cos(theta)
    kwargs = {
        "max_speed": 50.0,
        "max_accel": 10.0,
        "max_brake": 15.0,
        "gravity": 9.81,
        "safety_factor": 0.9,
        "use_friction_circle": True,
        "max_iterations": 300,
        "tolerance": 1.0e-10,
    }

    open_path = generate_speed_profile(
        x_m,
        y_m,
        curvature=geometry.curvature,
        mu=friction,
        **kwargs,
    )
    repeated_path = generate_speed_profile(
        np.append(x_m, x_m[0]),
        np.append(y_m, y_m[0]),
        curvature=np.append(geometry.curvature, geometry.curvature[0]),
        mu=np.append(friction, friction[0]),
        **kwargs,
    )

    np.testing.assert_array_equal(repeated_path.speed_envelope, open_path.speed_envelope)
    np.testing.assert_array_equal(repeated_path.speed, open_path.speed)
    np.testing.assert_array_equal(repeated_path.segment_lengths, open_path.segment_lengths)
    assert repeated_path.lap_time_s == open_path.lap_time_s


def test_generate_profile_preserves_scalar_grip_compatibility() -> None:
    point_count = 20
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    x_m = 25.0 * np.cos(theta)
    y_m = 16.0 * np.sin(theta)
    curvature = path_geometry(x_m, y_m).curvature

    scalar = generate_speed_profile(
        x_m,
        y_m,
        curvature=curvature,
        mu=1.2,
        use_friction_circle=True,
    )
    constant_array = generate_speed_profile(
        x_m,
        y_m,
        curvature=curvature,
        mu=np.full(point_count, 1.2),
        use_friction_circle=True,
    )

    np.testing.assert_array_equal(constant_array.speed_envelope, scalar.speed_envelope)
    np.testing.assert_array_equal(constant_array.speed, scalar.speed)
    assert constant_array.lap_time_s == scalar.lap_time_s


@pytest.mark.parametrize(
    "friction",
    [np.ones(7), np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])],
)
def test_generate_profile_rejects_invalid_local_grip(friction: np.ndarray) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    with pytest.raises(ValueError, match="friction_coefficient"):
        generate_speed_profile(
            np.cos(theta),
            np.sin(theta),
            mu=friction,
        )
