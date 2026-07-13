"""Single-track vehicle dynamics with a Pacejka-style lateral tyre model."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, atan2, cos, sin, sqrt, tan

import numpy as np

from .config import VehicleConfig


@dataclass
class VehicleState:
    x_m: float
    y_m: float
    yaw_rad: float
    vx_mps: float
    vy_mps: float = 0.0
    yaw_rate_rad_s: float = 0.0
    steering_rad: float = 0.0

    @property
    def speed_mps(self) -> float:
        return float(sqrt(self.vx_mps**2 + self.vy_mps**2))


@dataclass(frozen=True)
class VehicleControl:
    steering_rad: float
    acceleration_mps2: float


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


class SingleTrackModel:
    """A compact dynamic bicycle model suitable for algorithm prototyping.

    The model uses a normalized Pacejka Magic Formula for lateral force and a
    kinematic update below a configurable speed. That low-speed branch avoids
    the slip-angle singularity called out in the feasibility review.
    """

    def __init__(self, config: VehicleConfig):
        self.config = config

    def pacejka_coefficient(self, slip_angle_rad: float, grip_mu: float) -> float:
        """Return signed lateral-force coefficient for a tyre slip angle."""

        cfg = self.config
        bx = cfg.tyre_b * slip_angle_rad
        return float(
            grip_mu
            * sin(cfg.tyre_c * atan(bx - cfg.tyre_e * (bx - atan(bx))))
        )

    def _limited_control(
        self,
        state: VehicleState,
        control: VehicleControl,
        dt_s: float,
        grip_mu: float,
    ) -> tuple[float, float]:
        cfg = self.config
        target_steer = float(
            np.clip(control.steering_rad, -cfg.max_steer_rad, cfg.max_steer_rad)
        )
        max_change = cfg.max_steer_rate_rad_s * dt_s
        steering = state.steering_rad + float(
            np.clip(target_steer - state.steering_rad, -max_change, max_change)
        )

        acceleration = float(
            np.clip(
                control.acceleration_mps2,
                -cfg.max_brake_mps2,
                cfg.max_accel_mps2,
            )
        )
        # A friction-circle approximation reserves acceleration for cornering.
        lateral_accel = state.vx_mps**2 * abs(tan(steering)) / cfg.wheelbase_m
        total_limit = max(grip_mu, 0.05) * 9.81
        longitudinal_limit = sqrt(max(total_limit**2 - lateral_accel**2, 0.0))
        acceleration = float(np.clip(acceleration, -longitudinal_limit, longitudinal_limit))
        return steering, acceleration

    def step(
        self,
        state: VehicleState,
        control: VehicleControl,
        dt_s: float,
        grip_mu: float | None = None,
    ) -> VehicleState:
        """Advance the vehicle one deterministic integration step."""

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        cfg = self.config
        mu = cfg.base_grip_mu if grip_mu is None else float(grip_mu)
        if mu <= 0:
            raise ValueError("grip_mu must be positive")
        steering, acceleration = self._limited_control(state, control, dt_s, mu)

        if state.vx_mps < cfg.low_speed_blend_mps:
            return self._step_kinematic(state, steering, acceleration, dt_s)

        vx = max(state.vx_mps, 0.5)
        vy = state.vy_mps
        yaw_rate = state.yaw_rate_rad_s
        lf = cfg.cg_to_front_m
        lr = cfg.wheelbase_m - lf

        alpha_front = atan2(vy + lf * yaw_rate, vx) - steering
        alpha_rear = atan2(vy - lr * yaw_rate, vx)

        front_load = cfg.mass_kg * 9.81 * lr / cfg.wheelbase_m
        rear_load = cfg.mass_kg * 9.81 * lf / cfg.wheelbase_m
        force_y_front = -front_load * self.pacejka_coefficient(alpha_front, mu)
        force_y_rear = -rear_load * self.pacejka_coefficient(alpha_rear, mu)

        drag = cfg.drag_accel_coefficient * vx**2
        vx_dot = acceleration - drag + yaw_rate * vy
        vy_dot = (
            force_y_front * cos(steering) + force_y_rear
        ) / cfg.mass_kg - yaw_rate * vx
        yaw_accel = (
            lf * force_y_front * cos(steering) - lr * force_y_rear
        ) / cfg.yaw_inertia_kgm2

        # Semi-implicit Euler is noticeably better behaved here than a fully
        # explicit pose update while retaining a very small implementation.
        next_vx = float(np.clip(vx + vx_dot * dt_s, 0.0, cfg.max_speed_mps))
        next_vy = float(vy + vy_dot * dt_s)
        next_yaw_rate = float(yaw_rate + yaw_accel * dt_s)
        next_yaw = wrap_angle(state.yaw_rad + next_yaw_rate * dt_s)
        world_vx = next_vx * cos(next_yaw) - next_vy * sin(next_yaw)
        world_vy = next_vx * sin(next_yaw) + next_vy * cos(next_yaw)

        return VehicleState(
            x_m=state.x_m + world_vx * dt_s,
            y_m=state.y_m + world_vy * dt_s,
            yaw_rad=next_yaw,
            vx_mps=next_vx,
            vy_mps=next_vy,
            yaw_rate_rad_s=next_yaw_rate,
            steering_rad=steering,
        )

    def _step_kinematic(
        self,
        state: VehicleState,
        steering: float,
        acceleration: float,
        dt_s: float,
    ) -> VehicleState:
        cfg = self.config
        speed = float(
            np.clip(
                state.vx_mps + acceleration * dt_s,
                0.0,
                cfg.max_speed_mps,
            )
        )
        lr = cfg.wheelbase_m - cfg.cg_to_front_m
        beta = atan(lr / cfg.wheelbase_m * tan(steering))
        yaw_rate = speed / cfg.wheelbase_m * cos(beta) * tan(steering)
        yaw = wrap_angle(state.yaw_rad + yaw_rate * dt_s)
        return VehicleState(
            x_m=state.x_m + speed * cos(yaw + beta) * dt_s,
            y_m=state.y_m + speed * sin(yaw + beta) * dt_s,
            yaw_rad=yaw,
            vx_mps=speed,
            vy_mps=0.0,
            yaw_rate_rad_s=yaw_rate,
            steering_rad=steering,
        )

