from __future__ import annotations

from pathlib import Path

import pytest

from racing_line.config import AppConfig, load_config, make_silverstone_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_application_defaults() -> None:
    assert load_config(PROJECT_ROOT / "configs" / "default.yaml") == AppConfig()


def test_silverstone_yaml_matches_bundled_circuit_profile() -> None:
    config = make_silverstone_config()

    assert load_config(PROJECT_ROOT / "configs" / "silverstone.yaml") == config
    assert config.optimizer.points == 600
    assert config.optimizer.safety_margin_m == pytest.approx(0.59)
    assert config.optimizer.offset_regularization == pytest.approx(1.0e-6)
    assert config.simulation.lookahead_base_m == pytest.approx(3.0)
    assert config.simulation.lookahead_speed_gain_s == pytest.approx(0.08)
    assert config.simulation.speed_kp == pytest.approx(6.3)
    assert config.simulation.steering_kp == pytest.approx(1.58)


def test_partial_yaml_overrides_values_and_preserves_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
vehicle:
  mass_kg: 810.5
optimizer:
  points: 96
simulation:
  seed: 123
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.vehicle.mass_kg == pytest.approx(810.5)
    assert config.vehicle.wheelbase_m == AppConfig().vehicle.wheelbase_m
    assert config.optimizer.points == 96
    assert config.optimizer.safety_margin_m == AppConfig().optimizer.safety_margin_m
    assert config.simulation.seed == 123
    assert config.safety == AppConfig().safety


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("vehicle:\n  mystery_parameter: 1\n", "unknown configuration key"),
        ("unknown_section: {}\n", "unknown configuration key"),
        ("vehicle: 42\n", "config.vehicle must be a mapping"),
        ("- vehicle\n- optimizer\n", "configuration root must be a mapping"),
        ("vehicle:\n  mass_kg: 0\n", "vehicle.mass_kg must be positive"),
        ("optimizer:\n  points: 23\n", "optimizer.points must be at least 24"),
    ],
)
def test_invalid_yaml_is_rejected(
    tmp_path: Path, document: str, message: str
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_path)
