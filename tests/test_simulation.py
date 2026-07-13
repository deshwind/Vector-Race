from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest

from racing_line.config import AppConfig
from racing_line.safety import ResidualAction
from racing_line.simulation import LapSimulator
from racing_line.trajectory import RacingTrajectory
from racing_line.vehicle import VehicleControl, VehicleState


def _circular_trajectory(point_count: int = 16) -> RacingTrajectory:
    radius_m = 10.0
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    return RacingTrajectory(
        s_m=radius_m * theta,
        x_m=radius_m * np.cos(theta),
        y_m=radius_m * np.sin(theta),
        heading_rad=theta + np.pi / 2.0,
        curvature_1pm=np.full(point_count, 1.0 / radius_m),
        speed_mps=np.full(point_count, 8.0),
        lateral_offset_m=np.zeros(point_count),
        width_left_m=np.full(point_count, 2.5),
        width_right_m=np.full(point_count, 2.5),
    )


def _state_at_offset(offset_m: float, **overrides: float) -> VehicleState:
    values = {
        "x_m": 10.0 - offset_m,
        "y_m": 0.0,
        "yaw_rad": np.pi / 2.0,
        "vx_mps": 8.0,
    }
    values.update(overrides)
    return VehicleState(**values)


class _StaticController:
    def reset(self, _index: int = 0) -> None:
        return None

    def nearest_index(self, _x_m: float, _y_m: float) -> int:
        return 0

    def control(
        self,
        _state: VehicleState,
        _target_speed_mps: float,
        _residual_offset_m: float,
    ) -> tuple[VehicleControl, SimpleNamespace]:
        return VehicleControl(0.0, 0.0), SimpleNamespace(heading_error_rad=0.0)


class _IndexSequenceController(_StaticController):
    def __init__(self, indices: list[int]):
        self.indices = iter(indices)
        self.reset_calls: list[int] = []

    def reset(self, index: int = 0) -> None:
        self.reset_calls.append(index)

    def nearest_index(self, _x_m: float, _y_m: float) -> int:
        return next(self.indices)


class _SequenceModel:
    def __init__(self, states: list[VehicleState]):
        self.states = iter(states)

    def step(
        self,
        _state: VehicleState,
        _control: VehicleControl,
        _dt_s: float,
        _grip_mu: float,
    ) -> VehicleState:
        return next(self.states)


class _ContinuousModel:
    def step(
        self,
        state: VehicleState,
        _control: VehicleControl,
        _dt_s: float,
        _grip_mu: float,
    ) -> VehicleState:
        return replace(state, x_m=state.x_m + 0.001)


def _short_config(steps: int, dt_s: float = 0.1) -> AppConfig:
    config = AppConfig()
    return replace(
        config,
        simulation=replace(
            config.simulation,
            time_step_s=dt_s,
            max_time_s=steps * dt_s,
            start_speed_mps=8.0,
        ),
    )


def test_reentry_resets_off_track_termination_streak() -> None:
    simulator = LapSimulator(_circular_trajectory(), _short_config(steps=9))
    simulator.controller = _StaticController()
    # The evaluated offsets are 0, 2, 2, 2, 0, 2, 2, 2, 0. Six samples
    # are outside, but neither excursion lasts for the terminating 0.5 seconds.
    simulator.model = _SequenceModel(
        [_state_at_offset(value) for value in (2, 2, 2, 0, 2, 2, 2, 0, 0)]
    )

    result = simulator.run()

    assert result.summary.termination_reason == "time_limit"
    assert result.summary.steps == 9
    assert result.summary.off_track_events == 6


def test_five_consecutive_off_track_samples_terminate_run() -> None:
    simulator = LapSimulator(_circular_trajectory(), _short_config(steps=9))
    simulator.controller = _StaticController()
    simulator.model = _SequenceModel(
        [_state_at_offset(value) for value in (2, 2, 2, 2, 2, 0, 0, 0, 0)]
    )

    result = simulator.run()

    assert result.summary.termination_reason == "off_track"
    assert result.summary.steps == 6
    assert result.summary.off_track_events == 5


def test_policy_observation_exposes_replay_state_and_previous_filter() -> None:
    simulator = LapSimulator(_circular_trajectory(), _short_config(steps=2))
    simulator.controller = _StaticController()
    simulator.model = _SequenceModel(
        [
            _state_at_offset(0.0, yaw_rate_rad_s=0.25, steering_rad=0.05),
            _state_at_offset(0.0),
        ]
    )
    observations: list[Mapping[str, float]] = []

    def policy(observation: Mapping[str, float]) -> ResidualAction:
        observations.append(dict(observation))
        return ResidualAction(lateral_offset_m=1.0, speed_scale_delta=0.10)

    result = simulator.run(policy=policy)

    assert len(observations) == 2
    first, second = observations
    assert first["yaw_rate_rad_s"] == 0.0
    assert first["steering_rad"] == 0.0
    assert first["reference_half_width_m"] == pytest.approx(2.5)
    assert first["previous_filtered_offset_m"] == 0.0
    assert first["previous_filtered_speed_delta"] == 0.0
    assert second["yaw_rate_rad_s"] == pytest.approx(0.25)
    assert second["steering_rad"] == pytest.approx(0.05)
    assert second["previous_filtered_offset_m"] == pytest.approx(
        result.telemetry[0]["filtered_offset_m"]
    )
    assert second["previous_filtered_speed_delta"] == pytest.approx(
        result.telemetry[0]["filtered_speed_delta"]
    )


def test_trajectory_grip_is_default_and_explicit_profile_takes_precedence() -> None:
    base = _circular_trajectory()
    trajectory = replace(
        base,
        local_grip_mu=np.linspace(1.15, 1.45, base.point_count),
    )
    config = _short_config(steps=1)

    observed_default: list[float] = []
    LapSimulator(trajectory, config).run(
        policy=lambda observation: (
            observed_default.append(observation["grip_mu"]) or ResidualAction()
        )
    )

    observed_override: list[float] = []
    LapSimulator(trajectory, config).run(
        policy=lambda observation: (
            observed_override.append(observation["grip_mu"]) or ResidualAction()
        ),
        grip_profile=lambda _progress: 0.85,
    )

    assert observed_default == [pytest.approx(1.15)]
    assert observed_override == [pytest.approx(0.85)]


def test_two_laps_are_continuous_and_report_individual_times() -> None:
    point_count = 8
    trajectory = _circular_trajectory(point_count)
    trajectory = replace(
        trajectory,
        width_left_m=np.full(point_count, 100.0),
        width_right_m=np.full(point_count, 100.0),
    )
    simulator = LapSimulator(
        trajectory,
        _short_config(steps=20),
    )
    controller = _IndexSequenceController(
        [index for _ in range(3) for index in range(point_count) for _ in range(2)]
    )
    simulator.controller = controller
    simulator.model = _ContinuousModel()

    result = simulator.run(lap_count=2)

    assert result.summary.completed
    assert result.summary.termination_reason == "lap_complete"
    assert result.summary.requested_laps == 2
    assert result.summary.completed_laps == 2
    assert result.summary.lap_times_s == pytest.approx((1.3, 1.6))
    assert result.summary.lap_time_s == pytest.approx(2.9)
    assert result.summary.simulated_time_s == pytest.approx(2.9)
    assert controller.reset_calls == [0]
    assert result.telemetry[-1]["requested_laps"] == 2
    assert result.telemetry[-1]["completed_laps"] == 2
    assert result.telemetry[-1]["progress_laps"] == pytest.approx(1.75)
    assert float(result.telemetry[-1]["x_m"]) > float(
        result.telemetry[0]["x_m"]
    )


def test_time_limit_scales_with_requested_laps() -> None:
    simulator = LapSimulator(_circular_trajectory(), _short_config(steps=3))
    simulator.controller = _StaticController()
    simulator.model = _ContinuousModel()

    result = simulator.run(lap_count=3)

    assert result.summary.termination_reason == "time_limit"
    expected_steps = (
        int(
            np.ceil(
                simulator.config.simulation.max_time_s
                / simulator.config.simulation.time_step_s
            )
        )
        * 3
    )
    assert result.summary.steps == expected_steps
    assert result.summary.requested_laps == 3
    assert result.summary.completed_laps == 0
    assert result.summary.lap_times_s == ()


@pytest.mark.parametrize("lap_count", [0, 101])
def test_lap_count_must_be_within_supported_range(lap_count: int) -> None:
    simulator = LapSimulator(_circular_trajectory(), _short_config(steps=1))

    with pytest.raises(ValueError, match="between 1 and 100"):
        simulator.run(lap_count=lap_count)


@pytest.mark.parametrize("lap_count", [True, 1.5, "2"])
def test_lap_count_must_be_an_integer(lap_count: object) -> None:
    simulator = LapSimulator(_circular_trajectory(), _short_config(steps=1))

    with pytest.raises(TypeError, match="must be an integer"):
        simulator.run(lap_count=lap_count)  # type: ignore[arg-type]
