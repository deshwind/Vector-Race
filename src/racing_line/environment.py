"""Residual racing environment with an optional Gymnasium facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import AppConfig
from .controller import TrajectoryController
from .safety import ResidualAction, SafetyContext, SafetySupervisor
from .trajectory import RacingTrajectory
from .vehicle import SingleTrackModel, VehicleState


@dataclass(frozen=True)
class CoreStep:
    observation: NDArray[np.float32]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class ResidualRacingCore:
    """Dependency-free episode engine used by PPO and smoke tests.

    Actions are normalized to ``[-1, 1]`` and represent lateral reference-line
    offset and speed-reference scaling. Raw steering/throttle are intentionally
    left to the baseline controller and safety layer.
    """

    observation_size = 10
    action_size = 2

    def __init__(
        self,
        trajectory: RacingTrajectory,
        config: AppConfig,
        grip_range: tuple[float, float] | None = None,
    ):
        self.trajectory = trajectory
        self.config = config
        self.grip_range = grip_range or (
            0.65 * config.vehicle.base_grip_mu,
            1.05 * config.vehicle.base_grip_mu,
        )
        if self.grip_range[0] <= 0 or self.grip_range[1] < self.grip_range[0]:
            raise ValueError("grip_range must be positive and increasing")
        self.model = SingleTrackModel(config.vehicle)
        self.controller = TrajectoryController(
            trajectory, config.vehicle, config.simulation
        )
        self.safety = SafetySupervisor(config.safety)
        self.rng = np.random.default_rng(config.simulation.seed)
        self.state = self._initial_state()
        self.grip_mu = config.vehicle.base_grip_mu
        self.previous_index = 0
        self.unwrapped_index = 0
        self.step_count = 0
        self.off_track_steps = 0
        self.previous_filtered = ResidualAction()

    @property
    def max_steps(self) -> int:
        return int(
            np.ceil(
                self.config.simulation.max_time_s
                / self.config.simulation.time_step_s
            )
        )

    def _initial_state(self) -> VehicleState:
        return VehicleState(
            x_m=float(self.trajectory.x_m[0]),
            y_m=float(self.trajectory.y_m[0]),
            yaw_rad=float(self.trajectory.heading_rad[0]),
            vx_mps=min(
                self.config.simulation.start_speed_mps,
                float(self.trajectory.speed_mps[0]),
            ),
        )

    def reset(self, seed: int | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.grip_mu = float(self.rng.uniform(*self.grip_range))
        self.state = self._initial_state()
        self.previous_index = 0
        self.unwrapped_index = 0
        self.step_count = 0
        self.off_track_steps = 0
        self.previous_filtered = ResidualAction()
        self.controller.reset(0)
        self.safety.reset()
        observation, details = self._observation(0)
        return observation, {"grip_mu": self.grip_mu, **details}

    def _nearest(self) -> int:
        return self.controller.nearest_index(self.state.x_m, self.state.y_m)

    def _lateral_error(self, index: int) -> float:
        dx = self.state.x_m - self.trajectory.x_m[index]
        dy = self.state.y_m - self.trajectory.y_m[index]
        return float(
            dx * self.trajectory.normal_x[index]
            + dy * self.trajectory.normal_y[index]
        )

    def _observation(
        self, index: int
    ) -> tuple[NDArray[np.float32], dict[str, float]]:
        lateral_error = self._lateral_error(index)
        heading_error = float(
            (self.trajectory.heading_rad[index] - self.state.yaw_rad + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )
        half_width = max(
            min(
                self.trajectory.width_left_m[index],
                self.trajectory.width_right_m[index],
            ),
            0.1,
        )
        vehicle = self.config.vehicle
        safety = self.config.safety
        observation = np.asarray(
            [
                np.clip(lateral_error / half_width, -2.0, 2.0),
                heading_error / np.pi,
                self.state.speed_mps / vehicle.max_speed_mps,
                np.clip(self.state.yaw_rate_rad_s / 3.0, -2.0, 2.0),
                self.state.steering_rad / vehicle.max_steer_rad,
                np.clip(self.trajectory.curvature_1pm[index] * 25.0, -2.0, 2.0),
                self.trajectory.speed_mps[index] / vehicle.max_speed_mps,
                self.grip_mu / vehicle.base_grip_mu,
                self.previous_filtered.lateral_offset_m
                / safety.max_residual_offset_m,
                self.previous_filtered.speed_scale_delta
                / max(abs(safety.min_speed_scale_delta), safety.max_speed_scale_delta),
            ],
            dtype=np.float32,
        )
        return observation, {
            "lateral_error_m": lateral_error,
            "heading_error_rad": heading_error,
        }

    def _map_action(self, action: NDArray[np.floating[Any]]) -> ResidualAction:
        values = np.asarray(action, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("action must be a finite array with shape (2,)")
        values = np.clip(values, -1.0, 1.0)
        safety = self.config.safety
        speed_delta = (
            values[1] * safety.max_speed_scale_delta
            if values[1] >= 0
            else -values[1] * safety.min_speed_scale_delta
        )
        return ResidualAction(
            lateral_offset_m=float(values[0] * safety.max_residual_offset_m),
            speed_scale_delta=float(speed_delta),
        )

    def step(self, action: NDArray[np.floating[Any]]) -> CoreStep:
        requested = self._map_action(action)
        index = self._nearest()
        trajectory = self.trajectory
        dt = self.config.simulation.time_step_s

        filtered = self.safety.filter(
            requested,
            SafetyContext(
                baseline_offset_m=float(trajectory.lateral_offset_m[index]),
                width_left_m=float(trajectory.width_left_m[index]),
                width_right_m=float(trajectory.width_right_m[index]),
                vehicle_width_m=self.config.vehicle.width_m,
                baseline_speed_mps=float(trajectory.speed_mps[index]),
                curvature_1pm=float(trajectory.curvature_1pm[index]),
                grip_mu=self.grip_mu,
                dt_s=dt,
            ),
        )
        control, _ = self.controller.control(
            self.state, filtered.target_speed_mps, filtered.lateral_offset_m
        )
        self.state = self.model.step(self.state, control, dt, self.grip_mu)
        self.previous_filtered = ResidualAction(
            filtered.lateral_offset_m, filtered.speed_scale_delta
        )
        self.step_count += 1

        next_index = self._nearest()
        count = trajectory.point_count
        delta_index = (
            next_index - self.previous_index + count // 2
        ) % count - count // 2
        self.previous_index = next_index
        if delta_index >= -2:
            self.unwrapped_index += delta_index

        observation, details = self._observation(next_index)
        total_offset = trajectory.lateral_offset_m[next_index] + details[
            "lateral_error_m"
        ]
        half_vehicle = self.config.vehicle.width_m / 2.0
        inside = bool(
            total_offset <= trajectory.width_left_m[next_index] - half_vehicle
            and total_offset >= -trajectory.width_right_m[next_index] + half_vehicle
        )
        self.off_track_steps = 0 if inside else self.off_track_steps + 1

        mean_segment = trajectory.length_m / count
        forward_m = max(delta_index, 0) * mean_segment
        normalized_lateral = details["lateral_error_m"] / max(
            min(
                trajectory.width_left_m[next_index],
                trajectory.width_right_m[next_index],
            ),
            0.1,
        )
        reward = (
            0.08 * forward_m
            - 0.01
            - 0.04 * normalized_lateral**2
            - 0.02 * (details["heading_error_rad"] / np.pi) ** 2
            - 0.002 * float(filtered.intervened)
        )
        completed = self.unwrapped_index >= count - 2 and self.step_count > count
        off_track = self.off_track_steps * dt >= 0.50
        terminated = bool(completed or off_track)
        truncated = self.step_count >= self.max_steps
        if completed:
            reward += 100.0
        elif off_track:
            reward -= 100.0

        return CoreStep(
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={
                **details,
                "progress_laps": self.unwrapped_index / count,
                "grip_mu": self.grip_mu,
                "inside_track": inside,
                "lap_complete": completed,
                "safety_intervened": filtered.intervened,
                "safety_reasons": filtered.reasons,
                "requested_action": requested,
                "filtered_action": filtered,
            },
        )


try:  # Keep the deterministic core importable without the optional RL stack.
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised in installations without RL extras
    gym = None
    spaces = None


if gym is not None:

    class ResidualRacingEnv(gym.Env):  # type: ignore[misc]
        """Gymnasium adapter around :class:`ResidualRacingCore`."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            trajectory: RacingTrajectory,
            config: AppConfig,
            grip_range: tuple[float, float] | None = None,
        ):
            super().__init__()
            self.core = ResidualRacingCore(trajectory, config, grip_range)
            self.observation_space = spaces.Box(
                low=-2.0,
                high=2.0,
                shape=(ResidualRacingCore.observation_size,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(ResidualRacingCore.action_size,),
                dtype=np.float32,
            )

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[NDArray[np.float32], dict[str, Any]]:
            del options
            super().reset(seed=seed)
            return self.core.reset(seed)

        def step(
            self, action: NDArray[np.float32]
        ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
            result = self.core.step(action)
            return (
                result.observation,
                result.reward,
                result.terminated,
                result.truncated,
                result.info,
            )

else:

    class ResidualRacingEnv:  # pragma: no cover - tiny dependency error shim
        def __init__(self, *_args: Any, **_kwargs: Any):
            raise ImportError(
                "Gymnasium is required for ResidualRacingEnv; "
                "install the project with: pip install -e '.[rl]'"
            )

