"""Optional Stable-Baselines3 PPO training and inference helpers."""

from __future__ import annotations

from collections.abc import Mapping
from math import pi
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig
from .environment import ResidualRacingEnv
from .safety import ResidualAction
from .trajectory import RacingTrajectory


def _ppo_class() -> Any:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - depends on optional stack
        raise ImportError(
            "PPO support is optional. Install it with: pip install -e '.[rl]'"
        ) from exc
    return PPO


def train_ppo(
    trajectory: RacingTrajectory,
    config: AppConfig,
    output_directory: str | Path,
    total_timesteps: int | None = None,
) -> Path:
    """Train the residual policy and save a Stable-Baselines3 checkpoint."""

    PPO = _ppo_class()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    environment = ResidualRacingEnv(trajectory, config)
    ppo = config.ppo
    model = PPO(
        "MlpPolicy",
        environment,
        learning_rate=ppo.learning_rate,
        batch_size=ppo.batch_size,
        gamma=ppo.gamma,
        ent_coef=ppo.ent_coef,
        n_steps=ppo.n_steps,
        clip_range=ppo.clip_range,
        gae_lambda=ppo.gae_lambda,
        n_epochs=ppo.n_epochs,
        seed=ppo.seed,
        tensorboard_log=str(output / "tensorboard"),
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps or ppo.total_timesteps)
    checkpoint = output / "residual_ppo"
    model.save(checkpoint)
    environment.close()
    return checkpoint.with_suffix(".zip")


class PPOPolicy:
    """Load a checkpoint for array inference or lap-simulator residual control.

    Mapping observations use the same ten features and normalization as
    :class:`~racing_line.environment.ResidualRacingCore`.  The simulator must
    provide all state, local-track, and previous-filtered-action fields; no
    neutral defaults are substituted because that would change the policy's
    training distribution.
    """

    _mapping_fields = (
        "lateral_error_m",
        "heading_error_rad",
        "speed_mps",
        "yaw_rate_rad_s",
        "steering_rad",
        "curvature_1pm",
        "reference_speed_mps",
        "grip_mu",
        "reference_half_width_m",
        "previous_filtered_offset_m",
        "previous_filtered_speed_delta",
    )

    def __init__(self, checkpoint: str | Path, config: AppConfig):
        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")
        PPO = _ppo_class()
        self.config = config
        self.model = PPO.load(str(checkpoint))

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Return the checkpoint's deterministic action for an array input."""

        action, _state = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=float)

    def observation_vector(self, observation: Mapping[str, float]) -> np.ndarray:
        """Convert an enriched simulator observation to the training vector."""

        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        missing = [name for name in self._mapping_fields if name not in observation]
        if missing:
            raise ValueError(
                "observation is missing required fields: " + ", ".join(missing)
            )

        values: dict[str, float] = {}
        for name in self._mapping_fields:
            try:
                value = float(observation[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"observation field {name!r} must be numeric") from exc
            if not np.isfinite(value):
                raise ValueError(f"observation field {name!r} must be finite")
            values[name] = value

        if values["reference_half_width_m"] <= 0.0:
            raise ValueError(
                "observation field 'reference_half_width_m' must be positive"
            )
        for name in ("speed_mps", "reference_speed_mps"):
            if values[name] < 0.0:
                raise ValueError(f"observation field {name!r} cannot be negative")
        if values["grip_mu"] <= 0.0:
            raise ValueError("observation field 'grip_mu' must be positive")

        vehicle = self.config.vehicle
        safety = self.config.safety
        half_width = max(values["reference_half_width_m"], 0.1)
        speed_delta_scale = max(
            abs(safety.min_speed_scale_delta), safety.max_speed_scale_delta
        )
        return np.asarray(
            [
                np.clip(values["lateral_error_m"] / half_width, -2.0, 2.0),
                values["heading_error_rad"] / pi,
                values["speed_mps"] / vehicle.max_speed_mps,
                np.clip(values["yaw_rate_rad_s"] / 3.0, -2.0, 2.0),
                values["steering_rad"] / vehicle.max_steer_rad,
                np.clip(values["curvature_1pm"] * 25.0, -2.0, 2.0),
                values["reference_speed_mps"] / vehicle.max_speed_mps,
                values["grip_mu"] / vehicle.base_grip_mu,
                values["previous_filtered_offset_m"]
                / safety.max_residual_offset_m,
                values["previous_filtered_speed_delta"] / speed_delta_scale,
            ],
            dtype=np.float32,
        )

    def __call__(self, observation: Mapping[str, float]) -> ResidualAction:
        """Predict and decode a simulator observation into a residual request."""

        action = self.predict(self.observation_vector(observation))
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError(
                "PPO checkpoint must predict a finite action with shape (2,)"
            )
        action = np.clip(action, -1.0, 1.0)
        safety = self.config.safety
        speed_delta = (
            action[1] * safety.max_speed_scale_delta
            if action[1] >= 0.0
            else -action[1] * safety.min_speed_scale_delta
        )
        return ResidualAction(
            lateral_offset_m=float(action[0] * safety.max_residual_offset_m),
            speed_scale_delta=float(speed_delta),
        )
