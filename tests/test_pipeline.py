from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from racing_line.config import AppConfig, make_silverstone_config
from racing_line.pipeline import build_trajectory
from racing_line.simulation import LapSimulator
from racing_line.track import Track, make_demo_track, make_silverstone_track
from racing_line.trajectory import RacingTrajectory


def _small_config(**speed_overrides: float) -> AppConfig:
    config = AppConfig()
    return replace(
        config,
        optimizer=replace(config.optimizer, points=48, max_iterations=80),
        speed_profile=replace(config.speed_profile, **speed_overrides),
    )


def test_build_preserves_local_grip_in_trajectory_csv(tmp_path: Path) -> None:
    demo = make_demo_track(48)
    local_grip = 1.35 + 0.20 * np.sin(
        np.linspace(0.0, 2.0 * np.pi, demo.point_count, endpoint=False)
    )
    track = Track(
        demo.x_m,
        demo.y_m,
        demo.width_left_m,
        demo.width_right_m,
        local_grip_mu=local_grip,
        name="variable_grip",
    )

    result = build_trajectory(track, _small_config())
    paths = result.save(tmp_path / "build")
    restored = RacingTrajectory.from_csv(paths["trajectory"])

    assert result.track.local_grip_mu is not None
    assert result.trajectory.local_grip_mu is not None
    assert restored.local_grip_mu is not None
    np.testing.assert_allclose(
        result.trajectory.local_grip_mu,
        result.track.local_grip_mu,
    )
    np.testing.assert_allclose(
        restored.local_grip_mu,
        result.trajectory.local_grip_mu,
    )
    assert "grip_mu" in paths["trajectory"].read_text(encoding="utf-8").splitlines()[0]


def test_configured_minimum_speed_rejects_infeasible_profile() -> None:
    with pytest.raises(RuntimeError, match="minimum_speed_mps"):
        build_trajectory(
            make_demo_track(48),
            _small_config(minimum_speed_mps=80.0),
        )


def test_trajectory_csv_without_grip_remains_backward_compatible(tmp_path: Path) -> None:
    trajectory = build_trajectory(make_demo_track(48), _small_config()).trajectory
    path = trajectory.to_csv(tmp_path / "trajectory.csv")

    restored = RacingTrajectory.from_csv(path)

    assert restored.local_grip_mu is None


def test_pipeline_success_reports_optimizer_nonconvergence() -> None:
    config = _small_config()
    config = replace(
        config,
        optimizer=replace(config.optimizer, max_iterations=1),
    )

    result = build_trajectory(make_demo_track(48), config)

    assert not result.optimization.success
    assert not result.success
    assert result.diagnostics["success"] is False


def test_silverstone_profile_builds_and_completes_closed_loop_lap() -> None:
    config = make_silverstone_config()
    result = build_trajectory(make_silverstone_track(), config)
    simulation = LapSimulator(result.trajectory, config).run()
    trajectory_diagnostics = result.diagnostics["trajectory"]

    assert result.success
    assert result.track.point_count == 600
    assert result.trajectory.lap_time_s == pytest.approx(132.03, abs=0.05)
    assert trajectory_diagnostics[
        "estimated_lap_time_improvement_percent"
    ] > 10.0
    assert trajectory_diagnostics[
        "configured_center_to_edge_clearance_m"
    ] == pytest.approx(1.59, abs=1.0e-8)
    assert trajectory_diagnostics["minimum_offset_bound_slack_m"] >= -1.0e-5
    assert simulation.summary.completed
    assert simulation.summary.termination_reason == "lap_complete"
    assert simulation.summary.off_track_events == 0
    assert simulation.summary.minimum_vehicle_edge_clearance_m >= 0.25
    assert simulation.summary.minimum_boundary_margin_slack_m >= 0.0
    assert simulation.summary.simulated_time_s == pytest.approx(139.70, abs=0.1)
