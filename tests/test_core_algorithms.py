from __future__ import annotations

import numpy as np
import pytest

from racing_line.geometry import path_geometry, resample_closed_path
from racing_line.optimizer import optimize_minimum_curvature
from racing_line.speed_profile import generate_speed_profile


def test_periodic_circle_resampling_and_geometry() -> None:
    radius_m = 20.0
    coarse_theta = np.linspace(0.0, 2.0 * np.pi, 17)
    x_m = radius_m * np.cos(coarse_theta)
    y_m = radius_m * np.sin(coarse_theta)
    width_m = 3.0 + 0.2 * np.cos(coarse_theta)

    x_resampled, y_resampled, width_resampled = resample_closed_path(
        x_m, y_m, 64, width_m
    )
    geometry = path_geometry(x_resampled, y_resampled)

    assert geometry.x.shape == (64,)
    np.testing.assert_allclose(
        np.hypot(x_resampled, y_resampled), radius_m, atol=2.0e-3
    )
    assert np.std(geometry.segment_lengths) / np.mean(
        geometry.segment_lengths
    ) < 1.0e-4
    assert geometry.lap_length == pytest.approx(
        2.0 * np.pi * radius_m, rel=1.0e-3
    )
    np.testing.assert_allclose(
        np.linalg.norm(geometry.tangents, axis=1), 1.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(geometry.normals, axis=1), 1.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", geometry.tangents, geometry.normals),
        0.0,
        atol=1.0e-12,
    )
    assert np.all(geometry.curvature > 0.0)
    assert np.mean(geometry.curvature) == pytest.approx(1.0 / radius_m, rel=2.0e-3)
    np.testing.assert_allclose(
        width_resampled, 3.0 + 0.2 * x_resampled / radius_m, atol=1.0e-12
    )


def test_optimizer_respects_asymmetric_clearance_bounds_and_improves_line() -> None:
    point_count = 40
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    radius_m = 30.0
    center_x = radius_m * np.cos(theta)
    center_y = radius_m * np.sin(theta)
    width_left = 2.8 + 0.25 * np.sin(theta)
    width_right = 4.5 + 0.35 * np.cos(2.0 * theta)
    vehicle_width_m = 2.0
    safety_margin_m = 0.25

    result = optimize_minimum_curvature(
        center_x,
        center_y,
        width_left,
        width_right,
        vehicle_width_m=vehicle_width_m,
        safety_margin_m=safety_margin_m,
        smoothness_weight=0.01,
        offset_regularization=1.0e-8,
        max_iterations=40,
    )

    clearance_m = vehicle_width_m / 2.0 + safety_margin_m
    np.testing.assert_allclose(result.lower_bound, -width_right + clearance_m)
    np.testing.assert_allclose(result.upper_bound, width_left - clearance_m)
    assert not np.allclose(result.lower_bound, -result.upper_bound)
    assert np.all(result.offset >= result.lower_bound - 1.0e-10)
    assert np.all(result.offset <= result.upper_bound + 1.0e-10)
    assert result.success, result.message
    assert (
        result.diagnostics.final_objective
        < result.diagnostics.initial_objective
    )
    assert (
        result.diagnostics.final_curvature_rms
        < result.diagnostics.initial_curvature_rms
    )


def test_generated_speed_profile_obeys_all_cyclic_limits() -> None:
    point_count = 64
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    x_m = 60.0 * np.cos(theta)
    y_m = 25.0 * np.sin(theta)
    geometry = path_geometry(x_m, y_m)
    friction_coefficient = 1.3
    gravity_mps2 = 9.80665
    safety_factor = 0.9
    max_accel_mps2 = 3.0
    max_brake_mps2 = 5.0

    profile = generate_speed_profile(
        x_m,
        y_m,
        curvature=geometry.curvature,
        mu=friction_coefficient,
        max_speed=60.0,
        max_accel=max_accel_mps2,
        max_brake=max_brake_mps2,
        gravity=gravity_mps2,
        safety_factor=safety_factor,
        max_iterations=200,
        tolerance=1.0e-9,
    )

    assert profile.diagnostics.converged
    assert profile.diagnostics.acceleration_limited_segments > 0
    assert profile.diagnostics.braking_limited_segments > 0
    assert np.all(np.isfinite(profile.speed))
    assert np.all(profile.speed > 0.0)
    assert np.all(profile.speed <= profile.speed_envelope + 1.0e-10)

    lateral_limit = friction_coefficient * gravity_mps2 * safety_factor
    assert np.max(np.abs(profile.lateral_acceleration)) <= lateral_limit + 1.0e-9

    following_speed = np.roll(profile.speed, -1)
    speed_squared_change = following_speed**2 - profile.speed**2
    assert np.all(
        speed_squared_change
        <= 2.0 * max_accel_mps2 * profile.segment_lengths + 1.0e-9
    )
    assert np.all(
        -speed_squared_change
        <= 2.0 * max_brake_mps2 * profile.segment_lengths + 1.0e-9
    )
    assert np.isfinite(profile.lap_time_s)
    assert profile.lap_time_s > 0.0
    assert profile.lap_time_s == pytest.approx(float(np.sum(profile.segment_times)))
