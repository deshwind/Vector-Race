from __future__ import annotations

from math import pi
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from racing_line import rl
from racing_line.config import AppConfig
from racing_line.safety import ResidualAction


class _FakeModel:
    def __init__(self, action: np.ndarray):
        self.action = action
        self.observation: np.ndarray | None = None
        self.deterministic: bool | None = None

    def predict(
        self, observation: np.ndarray, *, deterministic: bool
    ) -> tuple[np.ndarray, None]:
        self.observation = observation
        self.deterministic = deterministic
        return self.action, None


def _policy(
    monkeypatch: pytest.MonkeyPatch, action: np.ndarray | None = None
) -> tuple[rl.PPOPolicy, _FakeModel]:
    model = _FakeModel(
        np.asarray([1.25, -0.5], dtype=float) if action is None else action
    )

    class FakePPO:
        @staticmethod
        def load(checkpoint: str) -> _FakeModel:
            assert checkpoint == str(Path("checkpoint.zip"))
            return model

    monkeypatch.setattr(rl, "_ppo_class", lambda: FakePPO)
    return rl.PPOPolicy("checkpoint.zip", AppConfig()), model


def _observation(**overrides: Any) -> dict[str, float]:
    values = {
        "progress": 0.25,
        "lateral_error_m": 2.0,
        "heading_error_rad": pi / 2.0,
        "speed_mps": 46.0,
        "yaw_rate_rad_s": 7.5,
        "steering_rad": 0.21,
        "curvature_1pm": -0.10,
        "reference_speed_mps": 23.0,
        "grip_mu": 0.85,
        "reference_half_width_m": 4.0,
        "previous_filtered_offset_m": 0.75,
        "previous_filtered_speed_delta": -0.175,
    }
    values.update(overrides)
    return values


def test_mapping_policy_matches_core_normalization_and_action_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, model = _policy(monkeypatch)

    result = policy(_observation())

    assert result == ResidualAction(lateral_offset_m=1.5, speed_scale_delta=-0.175)
    assert model.deterministic is True
    assert model.observation is not None
    assert model.observation.dtype == np.float32
    np.testing.assert_allclose(
        model.observation,
        np.asarray([0.5, 0.5, 0.5, 2.0, 0.5, -2.0, 0.25, 0.5, 0.5, -0.5]),
    )


def test_predict_preserves_direct_array_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, model = _policy(monkeypatch, np.asarray([0.2, 0.3], dtype=np.float32))
    observation = np.zeros(10, dtype=np.float32)

    result = policy.predict(observation)

    np.testing.assert_allclose(result, [0.2, 0.3])
    assert result.dtype == float
    assert model.observation is observation
    assert model.deterministic is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"grip_mu": np.nan}, "'grip_mu' must be finite"),
        ({"grip_mu": 0.0}, "'grip_mu' must be positive"),
        (
            {"reference_half_width_m": 0.0},
            "'reference_half_width_m' must be positive",
        ),
        ({"speed_mps": -1.0}, "'speed_mps' cannot be negative"),
    ],
)
def test_mapping_policy_rejects_invalid_fields(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, float],
    message: str,
) -> None:
    policy, _model = _policy(monkeypatch)

    with pytest.raises(ValueError, match=message):
        policy(_observation(**overrides))


def test_mapping_policy_reports_all_missing_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _model = _policy(monkeypatch)
    observation = _observation()
    del observation["yaw_rate_rad_s"]
    del observation["previous_filtered_offset_m"]

    with pytest.raises(ValueError, match="yaw_rate_rad_s.*previous_filtered_offset_m"):
        policy(observation)


def test_mapping_policy_rejects_invalid_model_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _model = _policy(monkeypatch, np.asarray([[0.0, 0.0]]))

    with pytest.raises(ValueError, match=r"finite action with shape \(2,\)"):
        policy(_observation())
