from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pytest

from racing_line.config import AppConfig, VehicleConfig
from racing_line.environment import ResidualRacingCore
from racing_line.trajectory import RacingTrajectory
from racing_line.vehicle import (
    SingleTrackModel,
    VehicleControl,
    VehicleState,
)


def _assert_finite_state(state: VehicleState) -> None:
    assert np.all(np.isfinite(np.asarray(astuple(state), dtype=float)))


def test_single_track_low_speed_step_is_finite_and_rate_limited() -> None:
    config = VehicleConfig()
    model = SingleTrackModel(config)
    dt_s = 0.1

    result = model.step(
        VehicleState(x_m=0.0, y_m=0.0, yaw_rad=0.0, vx_mps=0.0),
        VehicleControl(steering_rad=10.0, acceleration_mps2=100.0),
        dt_s,
    )

    _assert_finite_state(result)
    assert result.steering_rad == pytest.approx(config.max_steer_rate_rad_s * dt_s)
    assert result.vx_mps == pytest.approx(config.max_accel_mps2 * dt_s)
    assert result.vy_mps == 0.0
    assert result.x_m > 0.0


def test_single_track_high_speed_step_is_finite_and_bounded() -> None:
    config = VehicleConfig()
    model = SingleTrackModel(config)
    dt_s = 0.01
    state = VehicleState(
        x_m=4.0,
        y_m=-2.0,
        yaw_rad=0.2,
        vx_mps=40.0,
        vy_mps=0.5,
        yaw_rate_rad_s=0.1,
    )

    result = model.step(
        state,
        VehicleControl(steering_rad=0.2, acceleration_mps2=2.0),
        dt_s,
        grip_mu=1.5,
    )

    _assert_finite_state(result)
    assert 0.0 <= result.vx_mps <= config.max_speed_mps
    assert result.steering_rad == pytest.approx(config.max_steer_rate_rad_s * dt_s)
    assert result.x_m != pytest.approx(state.x_m)


@pytest.mark.parametrize(
    ("dt_s", "grip_mu", "message"),
    [(0.0, None, "dt_s must be positive"), (0.1, 0.0, "grip_mu must be positive")],
)
def test_single_track_rejects_invalid_step_inputs(
    dt_s: float, grip_mu: float | None, message: str
) -> None:
    model = SingleTrackModel(VehicleConfig())

    with pytest.raises(ValueError, match=message):
        model.step(
            VehicleState(0.0, 0.0, 0.0, 5.0),
            VehicleControl(0.0, 0.0),
            dt_s,
            grip_mu,
        )


def _circular_trajectory(point_count: int = 64) -> RacingTrajectory:
    radius_m = 40.0
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    return RacingTrajectory(
        s_m=radius_m * theta,
        x_m=radius_m * np.cos(theta),
        y_m=radius_m * np.sin(theta),
        heading_rad=theta + np.pi / 2.0,
        curvature_1pm=np.full(point_count, 1.0 / radius_m),
        speed_mps=np.full(point_count, 15.0),
        lateral_offset_m=np.zeros(point_count),
        width_left_m=np.full(point_count, 6.0),
        width_right_m=np.full(point_count, 6.0),
    )


def test_residual_racing_core_reset_and_one_step_are_dependency_free() -> None:
    core = ResidualRacingCore(_circular_trajectory(), AppConfig())

    observation, reset_info = core.reset(seed=123)
    result = core.step(np.zeros(2, dtype=np.float32))

    assert observation.shape == (core.observation_size,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert 0.0 < reset_info["grip_mu"]
    assert result.observation.shape == (core.observation_size,)
    assert result.observation.dtype == np.float32
    assert np.all(np.isfinite(result.observation))
    assert np.isfinite(result.reward)
    assert isinstance(result.terminated, bool)
    assert isinstance(result.truncated, bool)
    assert result.info["inside_track"]
    assert result.info["requested_action"].lateral_offset_m == 0.0
