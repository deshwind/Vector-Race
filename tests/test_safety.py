from __future__ import annotations

from math import sqrt

import pytest

from racing_line.config import SafetyConfig
from racing_line.safety import ResidualAction, SafetyContext, SafetySupervisor


def _context(**overrides: float) -> SafetyContext:
    values = {
        "baseline_offset_m": 0.0,
        "width_left_m": 6.0,
        "width_right_m": 6.0,
        "vehicle_width_m": 2.0,
        "baseline_speed_mps": 20.0,
        "curvature_1pm": 0.0,
        "grip_mu": 1.7,
        "dt_s": 0.1,
    }
    values.update(overrides)
    return SafetyContext(**values)


def test_filter_enforces_stateful_action_rate_limits_and_reset() -> None:
    supervisor = SafetySupervisor(SafetyConfig())
    requested = ResidualAction(lateral_offset_m=1.0, speed_scale_delta=0.10)

    first = supervisor.filter(requested, _context())
    second = supervisor.filter(requested, _context())

    assert first.lateral_offset_m == pytest.approx(0.08)
    assert first.speed_scale_delta == pytest.approx(0.035)
    assert first.target_speed_mps == pytest.approx(20.7)
    assert "offset_rate_limit" in first.reasons
    assert "speed_scale_rate_limit" in first.reasons
    assert second.lateral_offset_m == pytest.approx(0.16)
    assert second.speed_scale_delta == pytest.approx(0.07)

    supervisor.reset()
    after_reset = supervisor.filter(requested, _context())
    assert after_reset.lateral_offset_m == pytest.approx(first.lateral_offset_m)
    assert after_reset.speed_scale_delta == pytest.approx(first.speed_scale_delta)


def test_filter_keeps_vehicle_inside_asymmetric_track_boundaries() -> None:
    config = SafetyConfig(max_offset_rate_mps=100.0)
    supervisor = SafetySupervisor(config)
    context = _context(
        baseline_offset_m=1.1,
        width_left_m=2.0,
        width_right_m=3.0,
        vehicle_width_m=1.0,
    )

    filtered = supervisor.filter(
        ResidualAction(lateral_offset_m=1.0), context
    )

    maximum_total_offset = 2.0 - 0.5 - config.boundary_margin_m
    assert context.baseline_offset_m + filtered.lateral_offset_m == pytest.approx(
        maximum_total_offset
    )
    assert filtered.reasons == ("track_boundary",)
    assert filtered.intervened


def test_filter_caps_target_speed_at_friction_envelope() -> None:
    config = SafetyConfig(max_speed_scale_rate_per_s=100.0)
    supervisor = SafetySupervisor(config)
    context = _context(
        baseline_speed_mps=50.0,
        curvature_1pm=0.10,
        grip_mu=1.0,
    )

    filtered = supervisor.filter(
        ResidualAction(speed_scale_delta=0.10), context
    )

    expected_speed = sqrt(1.0 * 9.81 / 0.10) * config.friction_safety_factor
    assert filtered.target_speed_mps == pytest.approx(expected_speed)
    assert filtered.speed_scale_delta == pytest.approx(expected_speed / 50.0 - 1.0)
    assert filtered.reasons == ("friction_envelope",)


def test_filter_rejects_impossible_or_invalid_contexts() -> None:
    supervisor = SafetySupervisor(SafetyConfig(max_offset_rate_mps=100.0))

    with pytest.raises(ValueError, match="dt_s must be positive"):
        supervisor.filter(ResidualAction(), _context(dt_s=0.0))
    with pytest.raises(ValueError, match="grip_mu must be positive"):
        supervisor.filter(ResidualAction(), _context(grip_mu=0.0))
    with pytest.raises(ValueError, match="track is too narrow"):
        supervisor.filter(
            ResidualAction(),
            _context(width_left_m=0.5, width_right_m=0.5, vehicle_width_m=2.0),
        )
