"""Small adapter for the pinned official F1TENTH Gym development API.

The simulator dependency is intentionally optional. Its historical ``main``
branch uses legacy OpenAI Gym while the pinned ``dev-humble`` commit uses the
five-return Gymnasium API represented here.
"""

from __future__ import annotations

from typing import Any

import numpy as np


PINNED_COMMIT = "bdaec1420c3b0f103858d289866d0d4e2e597c30"


def create_environment(
    map_name: str,
    *,
    render: bool = False,
    timestep_s: float = 0.01,
    dynamic: bool = False,
) -> Any:
    """Create the pinned single-agent F1TENTH Gymnasium environment.

    ``map_name`` can be a bundled map name or a local map base path supported
    by the pinned simulator. Kinematic dynamics make setup checks easier;
    switch ``dynamic=True`` for single-track validation.
    """

    try:
        import gymnasium as gym
        from f1tenth_gym.envs.dynamic_models import DynamicModel
        from f1tenth_gym.envs.env_config import (
            EnvConfig,
            ObservationConfig,
            SimulationConfig,
        )
        from f1tenth_gym.envs.integrators import IntegratorType
        from f1tenth_gym.envs.observation import ObservationType
    except ImportError as exc:  # pragma: no cover - optional integration
        raise ImportError(
            "F1TENTH support is optional. Install it with: "
            "pip install -e '.[f1tenth]'"
        ) from exc

    model = DynamicModel.ST if dynamic else DynamicModel.KS
    config = EnvConfig(
        map_name=map_name,
        num_agents=1,
        render_enabled=render,
        simulation_config=SimulationConfig(
            timestep=timestep_s,
            integrator_timestep=timestep_s,
            integrator=IntegratorType.RK4,
            dynamics_model=model,
            compute_frenet_frame=True,
            max_laps=1,
        ),
        observation_config=ObservationConfig(
            type=ObservationType.FEATURES,
            features=(
                "pose_x",
                "pose_y",
                "pose_theta",
                "delta",
                "linear_vel_magnitude",
                "frenet_pose",
                "collision",
            ),
        ),
    )
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=config,
        render_mode="human" if render else None,
    )


def clipped_control(environment: Any, steering_rad: float, target_speed_mps: float) -> np.ndarray:
    """Create a correctly ordered and clipped one-agent simulator action."""

    action = np.asarray([[steering_rad, target_speed_mps]], dtype=np.float32)
    return np.clip(action, environment.action_space.low, environment.action_space.high)
