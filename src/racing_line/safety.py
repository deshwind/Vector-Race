"""Stateful projection of learned residual commands into a safe envelope."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .config import SafetyConfig


@dataclass(frozen=True)
class ResidualAction:
    lateral_offset_m: float = 0.0
    speed_scale_delta: float = 0.0


@dataclass(frozen=True)
class SafetyContext:
    baseline_offset_m: float
    width_left_m: float
    width_right_m: float
    vehicle_width_m: float
    baseline_speed_mps: float
    curvature_1pm: float
    grip_mu: float
    dt_s: float


@dataclass(frozen=True)
class FilteredAction:
    lateral_offset_m: float
    speed_scale_delta: float
    target_speed_mps: float
    intervened: bool
    reasons: tuple[str, ...]


class SafetySupervisor:
    """Enforce boundary, action-rate, speed, and friction constraints."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self._previous = ResidualAction()

    def reset(self) -> None:
        self._previous = ResidualAction()

    def filter(self, requested: ResidualAction, context: SafetyContext) -> FilteredAction:
        if context.dt_s <= 0:
            raise ValueError("safety context dt_s must be positive")
        if context.grip_mu <= 0:
            raise ValueError("safety context grip_mu must be positive")

        cfg = self.config
        reasons: list[str] = []

        lateral = float(
            np.clip(
                requested.lateral_offset_m,
                -cfg.max_residual_offset_m,
                cfg.max_residual_offset_m,
            )
        )
        if not np.isclose(lateral, requested.lateral_offset_m):
            reasons.append("residual_offset_limit")

        max_lateral_change = cfg.max_offset_rate_mps * context.dt_s
        rate_limited = self._previous.lateral_offset_m + float(
            np.clip(
                lateral - self._previous.lateral_offset_m,
                -max_lateral_change,
                max_lateral_change,
            )
        )
        if not np.isclose(rate_limited, lateral):
            reasons.append("offset_rate_limit")
        lateral = rate_limited

        half_vehicle = context.vehicle_width_m / 2.0
        minimum_total_offset = (
            -context.width_right_m + half_vehicle + cfg.boundary_margin_m
        )
        maximum_total_offset = (
            context.width_left_m - half_vehicle - cfg.boundary_margin_m
        )
        if minimum_total_offset > maximum_total_offset:
            raise ValueError("track is too narrow for the configured vehicle and margin")
        total_offset = context.baseline_offset_m + lateral
        bounded_total = float(
            np.clip(total_offset, minimum_total_offset, maximum_total_offset)
        )
        if not np.isclose(bounded_total, total_offset):
            reasons.append("track_boundary")
        lateral = bounded_total - context.baseline_offset_m

        speed_delta = float(
            np.clip(
                requested.speed_scale_delta,
                cfg.min_speed_scale_delta,
                cfg.max_speed_scale_delta,
            )
        )
        if not np.isclose(speed_delta, requested.speed_scale_delta):
            reasons.append("speed_scale_limit")
        max_speed_change = cfg.max_speed_scale_rate_per_s * context.dt_s
        rate_limited_speed = self._previous.speed_scale_delta + float(
            np.clip(
                speed_delta - self._previous.speed_scale_delta,
                -max_speed_change,
                max_speed_change,
            )
        )
        if not np.isclose(rate_limited_speed, speed_delta):
            reasons.append("speed_scale_rate_limit")
        speed_delta = rate_limited_speed

        target_speed = max(context.baseline_speed_mps * (1.0 + speed_delta), 0.0)
        absolute_curvature = abs(context.curvature_1pm)
        if absolute_curvature > 1.0e-8:
            friction_speed = sqrt(
                context.grip_mu * 9.81 / absolute_curvature
            ) * cfg.friction_safety_factor
            if target_speed > friction_speed:
                target_speed = friction_speed
                if context.baseline_speed_mps > 1.0e-6:
                    speed_delta = target_speed / context.baseline_speed_mps - 1.0
                reasons.append("friction_envelope")

        self._previous = ResidualAction(lateral, speed_delta)
        return FilteredAction(
            lateral_offset_m=lateral,
            speed_scale_delta=speed_delta,
            target_speed_mps=target_speed,
            intervened=bool(reasons),
            reasons=tuple(dict.fromkeys(reasons)),
        )
