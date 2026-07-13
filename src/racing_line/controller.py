"""Baseline trajectory tracking controller for deterministic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, sin

import numpy as np

from .config import SimulationConfig, VehicleConfig
from .trajectory import RacingTrajectory
from .vehicle import VehicleControl, VehicleState, wrap_angle


@dataclass(frozen=True)
class ControllerDiagnostics:
    reference_index: int
    target_index: int
    lookahead_m: float
    lateral_error_m: float
    heading_error_rad: float
    target_speed_mps: float


class TrajectoryController:
    """Pure-pursuit steering plus proportional longitudinal control."""

    def __init__(
        self,
        trajectory: RacingTrajectory,
        vehicle: VehicleConfig,
        simulation: SimulationConfig,
    ):
        self.trajectory = trajectory
        self.vehicle = vehicle
        self.simulation = simulation
        self._last_index = 0

    def reset(self, index: int = 0) -> None:
        self._last_index = int(index) % self.trajectory.point_count

    def nearest_index(self, x_m: float, y_m: float) -> int:
        """Find the nearest point in a continuity-preserving local window."""

        count = self.trajectory.point_count
        half_window = min(max(count // 8, 12), count // 2)
        offsets = np.arange(-half_window, half_window + 1)
        candidates = (self._last_index + offsets) % count
        distances = (
            (self.trajectory.x_m[candidates] - x_m) ** 2
            + (self.trajectory.y_m[candidates] - y_m) ** 2
        )
        result = int(candidates[int(np.argmin(distances))])
        self._last_index = result
        return result

    def _lookahead_index(self, start: int, distance_m: float) -> int:
        lengths = self.trajectory.segment_lengths_m
        accumulated = 0.0
        index = start
        for _ in range(self.trajectory.point_count):
            if accumulated >= distance_m:
                break
            accumulated += lengths[index]
            index = (index + 1) % self.trajectory.point_count
        return index

    def control(
        self,
        state: VehicleState,
        target_speed_mps: float,
        residual_offset_m: float = 0.0,
    ) -> tuple[VehicleControl, ControllerDiagnostics]:
        reference_index = self.nearest_index(state.x_m, state.y_m)
        speed = max(state.speed_mps, 0.0)
        lookahead = (
            self.simulation.lookahead_base_m
            + self.simulation.lookahead_speed_gain_s * speed
        )
        target_index = self._lookahead_index(reference_index, lookahead)

        target_x = (
            self.trajectory.x_m[target_index]
            + residual_offset_m * self.trajectory.normal_x[target_index]
        )
        target_y = (
            self.trajectory.y_m[target_index]
            + residual_offset_m * self.trajectory.normal_y[target_index]
        )
        delta_x = target_x - state.x_m
        delta_y = target_y - state.y_m
        local_y = -sin(state.yaw_rad) * delta_x + cos(state.yaw_rad) * delta_y
        distance_squared = max(delta_x**2 + delta_y**2, 1.0e-6)
        commanded_curvature = 2.0 * local_y / distance_squared
        steering = self.simulation.steering_kp * atan(
            self.vehicle.wheelbase_m * commanded_curvature
        )

        drag_feedforward = self.vehicle.drag_accel_coefficient * speed**2
        acceleration = (
            self.simulation.speed_kp * (target_speed_mps - speed) + drag_feedforward
        )
        control = VehicleControl(steering, acceleration)

        dx = state.x_m - self.trajectory.x_m[reference_index]
        dy = state.y_m - self.trajectory.y_m[reference_index]
        lateral_error = (
            dx * self.trajectory.normal_x[reference_index]
            + dy * self.trajectory.normal_y[reference_index]
        )
        heading_error = wrap_angle(
            self.trajectory.heading_rad[reference_index] - state.yaw_rad
        )
        return control, ControllerDiagnostics(
            reference_index=reference_index,
            target_index=target_index,
            lookahead_m=lookahead,
            lateral_error_m=float(lateral_error),
            heading_error_rad=heading_error,
            target_speed_mps=float(target_speed_mps),
        )

