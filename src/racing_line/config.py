"""Validated configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


@dataclass(frozen=True)
class VehicleConfig:
    """Simplified full-scale single-track vehicle parameters.

    Defaults are deliberately approximate because current Formula 1 tyre and
    aero data are proprietary. They are research defaults, not an F1 claim.
    """

    mass_kg: float = 798.0
    yaw_inertia_kgm2: float = 1100.0
    wheelbase_m: float = 3.60
    cg_to_front_m: float = 1.80
    width_m: float = 2.00
    max_steer_rad: float = 0.42
    max_steer_rate_rad_s: float = 1.20
    max_accel_mps2: float = 5.0
    max_brake_mps2: float = 12.0
    max_speed_mps: float = 92.0
    drag_accel_coefficient: float = 0.00045
    base_grip_mu: float = 1.70
    tyre_b: float = 10.0
    tyre_c: float = 1.90
    tyre_e: float = 0.97
    low_speed_blend_mps: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            "mass_kg",
            "yaw_inertia_kgm2",
            "wheelbase_m",
            "cg_to_front_m",
            "width_m",
            "max_steer_rad",
            "max_steer_rate_rad_s",
            "max_accel_mps2",
            "max_brake_mps2",
            "max_speed_mps",
            "base_grip_mu",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"vehicle.{name} must be positive")
        if self.cg_to_front_m >= self.wheelbase_m:
            raise ValueError("vehicle.cg_to_front_m must be shorter than wheelbase_m")
        if self.drag_accel_coefficient < 0:
            raise ValueError("vehicle.drag_accel_coefficient cannot be negative")
        if self.tyre_b <= 0 or self.tyre_c <= 0:
            raise ValueError("vehicle tyre B and C coefficients must be positive")
        if not 0 <= self.tyre_e <= 1:
            raise ValueError("vehicle.tyre_e must be between zero and one")
        if self.low_speed_blend_mps <= 0:
            raise ValueError("vehicle.low_speed_blend_mps must be positive")


@dataclass(frozen=True)
class OptimizerConfig:
    points: int = 220
    safety_margin_m: float = 0.35
    smoothness_weight: float = 0.08
    offset_regularization: float = 1.0e-4
    max_iterations: int = 300

    def __post_init__(self) -> None:
        if self.points < 24:
            raise ValueError("optimizer.points must be at least 24")
        if self.safety_margin_m < 0:
            raise ValueError("optimizer.safety_margin_m cannot be negative")
        if self.smoothness_weight < 0 or self.offset_regularization < 0:
            raise ValueError("optimizer regularization weights cannot be negative")
        if self.max_iterations < 1:
            raise ValueError("optimizer.max_iterations must be positive")


@dataclass(frozen=True)
class SpeedProfileConfig:
    gravity_mps2: float = 9.81
    lateral_safety_factor: float = 0.96
    convergence_tolerance_mps: float = 1.0e-3
    max_iterations: int = 100
    minimum_speed_mps: float = 3.0

    def __post_init__(self) -> None:
        if self.gravity_mps2 <= 0:
            raise ValueError("speed_profile.gravity_mps2 must be positive")
        if not 0 < self.lateral_safety_factor <= 1:
            raise ValueError(
                "speed_profile.lateral_safety_factor must be in (0, 1]"
            )
        if self.convergence_tolerance_mps <= 0:
            raise ValueError(
                "speed_profile.convergence_tolerance_mps must be positive"
            )
        if self.max_iterations < 1:
            raise ValueError("speed_profile.max_iterations must be positive")
        if self.minimum_speed_mps < 0:
            raise ValueError("speed_profile.minimum_speed_mps cannot be negative")


@dataclass(frozen=True)
class SafetyConfig:
    boundary_margin_m: float = 0.25
    max_residual_offset_m: float = 1.50
    max_offset_rate_mps: float = 0.80
    min_speed_scale_delta: float = -0.35
    max_speed_scale_delta: float = 0.10
    max_speed_scale_rate_per_s: float = 0.35
    # Multiplies the speed ceiling; sqrt(0.96) aligns with the baseline
    # profile's 0.96 lateral-acceleration safety factor.
    friction_safety_factor: float = 0.98

    def __post_init__(self) -> None:
        if self.boundary_margin_m < 0:
            raise ValueError("safety.boundary_margin_m cannot be negative")
        if self.max_residual_offset_m <= 0 or self.max_offset_rate_mps <= 0:
            raise ValueError("safety offset limit and rate must be positive")
        if not -1 < self.min_speed_scale_delta <= 0:
            raise ValueError("safety.min_speed_scale_delta must be in (-1, 0]")
        if self.max_speed_scale_delta < 0:
            raise ValueError("safety.max_speed_scale_delta cannot be negative")
        if self.max_speed_scale_rate_per_s <= 0:
            raise ValueError("safety.max_speed_scale_rate_per_s must be positive")
        if not 0 < self.friction_safety_factor <= 1:
            raise ValueError("safety.friction_safety_factor must be in (0, 1]")


@dataclass(frozen=True)
class SimulationConfig:
    time_step_s: float = 0.02
    max_time_s: float = 240.0
    lookahead_base_m: float = 5.0
    lookahead_speed_gain_s: float = 0.24
    speed_kp: float = 1.4
    steering_kp: float = 1.20
    start_speed_mps: float = 8.0
    seed: int = 7

    def __post_init__(self) -> None:
        positive = (
            "time_step_s",
            "max_time_s",
            "lookahead_base_m",
            "speed_kp",
            "steering_kp",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"simulation.{name} must be positive")
        if self.lookahead_speed_gain_s < 0:
            raise ValueError("simulation.lookahead_speed_gain_s cannot be negative")
        if self.start_speed_mps < 0:
            raise ValueError("simulation.start_speed_mps cannot be negative")


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3.0e-4
    batch_size: int = 128
    gamma: float = 0.99
    ent_coef: float = 0.01
    n_steps: int = 2048
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    n_epochs: int = 10
    total_timesteps: int = 250_000
    seed: int = 7

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("ppo.learning_rate must be positive")
        if self.batch_size < 1 or self.n_steps < 1:
            raise ValueError("ppo.batch_size and ppo.n_steps must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("ppo.gamma and ppo.gae_lambda must be in (0, 1]")
        if self.ent_coef < 0:
            raise ValueError("ppo.ent_coef cannot be negative")
        if self.clip_range <= 0:
            raise ValueError("ppo.clip_range must be positive")
        if self.n_epochs < 1 or self.total_timesteps < 1:
            raise ValueError("ppo.n_epochs and total_timesteps must be positive")


@dataclass(frozen=True)
class AppConfig:
    vehicle: VehicleConfig = VehicleConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    speed_profile: SpeedProfileConfig = SpeedProfileConfig()
    safety: SafetyConfig = SafetyConfig()
    simulation: SimulationConfig = SimulationConfig()
    ppo: PPOConfig = PPOConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_silverstone_config() -> AppConfig:
    """Return the validated settings tuned for the bundled Silverstone track.

    The higher spatial resolution preserves the circuit's fast direction
    changes. Its offline clearance and controller gains retain the configured
    runtime edge buffer; the values remain local to this circuit so the legacy
    synthetic demo keeps its original behavior.
    """

    config = AppConfig()
    return replace(
        config,
        optimizer=replace(
            config.optimizer,
            points=600,
            safety_margin_m=0.59,
            offset_regularization=1.0e-6,
        ),
        simulation=replace(
            config.simulation,
            lookahead_base_m=3.0,
            lookahead_speed_gain_s=0.08,
            speed_kp=6.3,
            steering_kp=1.58,
        ),
    )


def make_f1_catalog_config() -> AppConfig:
    """Return conservative shared settings for the bundled F1 catalogue.

    The catalogue combines simplified geographic centrelines from permanent
    and street circuits. A denser resampling grid helps the pure-pursuit
    controller follow sharp layout changes, while a conservative lateral
    acceleration factor keeps the same deterministic controller
    within the provisional track boundaries across all bundled layouts.
    """

    config = AppConfig()
    return replace(
        config,
        optimizer=replace(
            config.optimizer,
            points=600,
            safety_margin_m=1.0,
            offset_regularization=1.0e-4,
        ),
        speed_profile=replace(
            config.speed_profile,
            lateral_safety_factor=0.75,
        ),
        simulation=replace(
            config.simulation,
            max_time_s=400.0,
            lookahead_base_m=3.0,
            lookahead_speed_gain_s=0.08,
            speed_kp=6.3,
            steering_kp=1.58,
            start_speed_mps=5.0,
        ),
    )


T = TypeVar("T")


def _build_dataclass(cls: type[T], values: Mapping[str, Any], prefix: str) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown configuration key(s) under {prefix}: {names}")

    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in values:
            continue
        default_value = getattr(cls(), item.name)
        raw_value = values[item.name]
        if is_dataclass(default_value):
            if not isinstance(raw_value, Mapping):
                raise ValueError(f"{prefix}.{item.name} must be a mapping")
            kwargs[item.name] = _build_dataclass(
                type(default_value), raw_value, f"{prefix}.{item.name}"
            )
        else:
            kwargs[item.name] = raw_value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load an application config, overlaying YAML values on safe defaults."""

    if path is None:
        return AppConfig()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, Mapping):
        raise ValueError("configuration root must be a mapping")
    return _build_dataclass(AppConfig, values, "config")
