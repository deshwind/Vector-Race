from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from racing_line import cli, rl
from racing_line.config import AppConfig, make_silverstone_config
from racing_line.track import Track, make_demo_track
from racing_line.trajectory import RacingTrajectory


def _trajectory_from_track(track: Track) -> RacingTrajectory:
    segment_lengths = track.segment_lengths_m
    s_m = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1])))
    following_x = np.roll(track.x_m, -1)
    following_y = np.roll(track.y_m, -1)
    heading = np.unwrap(np.arctan2(following_y - track.y_m, following_x - track.x_m))
    return RacingTrajectory(
        s_m=s_m,
        x_m=track.x_m,
        y_m=track.y_m,
        heading_rad=heading,
        curvature_1pm=np.zeros(track.point_count),
        speed_mps=np.full(track.point_count, 12.0),
        lateral_offset_m=np.zeros(track.point_count),
        width_left_m=track.width_left_m,
        width_right_m=track.width_right_m,
    )


def test_simulate_rejects_two_residual_policy_sources() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "simulate",
                "track.csv",
                "trajectory.csv",
                "--grip-aware",
                "--checkpoint",
                "policy.zip",
            ]
        )

    assert exc_info.value.code == 2


def test_bundled_commands_default_to_silverstone_profile() -> None:
    parser = cli.build_parser()

    demo = parser.parse_args(["demo", "--skip-simulation"])
    train = parser.parse_args(["train", "--timesteps", "1"])
    export = parser.parse_args(["export-demo-track"])

    assert demo.circuit == "silverstone"
    assert train.circuit == "silverstone"
    assert export.circuit == "silverstone"
    assert cli._configured(None, circuit=demo.circuit) == make_silverstone_config()


def test_web_command_defaults_to_ten_laps_on_loopback_port() -> None:
    args = cli.build_parser().parse_args(["web"])

    assert args.laps == 10
    assert args.port == 8765
    assert args.no_browser is False
    assert args.function is cli.command_web


def test_web_command_accepts_dashboard_overrides() -> None:
    args = cli.build_parser().parse_args(
        ["web", "--laps", "12", "--port", "9000", "--no-browser"]
    )

    assert args.laps == 12
    assert args.port == 9000
    assert args.no_browser is True


def test_explicit_config_is_not_replaced_by_circuit_profile(tmp_path: Path) -> None:
    config_path = tmp_path / "explicit.yaml"
    config_path.write_text("optimizer:\n  points: 96\n", encoding="utf-8")

    config = cli._configured(config_path, circuit="silverstone")

    assert config.optimizer.points == 96
    assert config.simulation == AppConfig().simulation


def test_synthetic_export_rejects_zero_points() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        cli._built_in_track("synthetic", 0)


def test_simulate_checkpoint_route_uses_track_grip_for_legacy_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    demo = make_demo_track(32)
    grip = np.linspace(1.25, 1.55, demo.point_count)
    track = Track(
        demo.x_m,
        demo.y_m,
        demo.width_left_m,
        demo.width_right_m,
        local_grip_mu=grip,
    )
    trajectory = _trajectory_from_track(track)
    track_path = track.to_csv(tmp_path / "track.csv")
    trajectory_path = trajectory.to_csv(tmp_path / "trajectory.csv")
    checkpoint = tmp_path / "policy.zip"

    policy_arguments: list[tuple[Path, Any]] = []

    @dataclass
    class FakePolicy:
        checkpoint: Path
        config: Any

        def __post_init__(self) -> None:
            policy_arguments.append((self.checkpoint, self.config))

    captured_grip: list[np.ndarray | None] = []

    class FakeSimulationResult:
        summary = SimpleNamespace(completed=True, termination_reason="lap_complete")

        def save(self, _output: Path) -> None:
            return None

    class FakeLapSimulator:
        def __init__(self, replay_trajectory: RacingTrajectory, _config: Any):
            captured_grip.append(replay_trajectory.local_grip_mu)

        def run(self, *, policy: Any) -> FakeSimulationResult:
            assert isinstance(policy, FakePolicy)
            return FakeSimulationResult()

    monkeypatch.setattr(rl, "PPOPolicy", FakePolicy)
    monkeypatch.setattr(cli, "LapSimulator", FakeLapSimulator)
    monkeypatch.setattr(cli, "plot_simulation", lambda *_args, **_kwargs: None)

    args = cli.build_parser().parse_args(
        [
            "simulate",
            str(track_path),
            str(trajectory_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "replay"),
        ]
    )
    status = args.function(args)

    assert status == 0
    assert policy_arguments and policy_arguments[0][0] == checkpoint
    assert captured_grip[0] is not None
    np.testing.assert_allclose(captured_grip[0], grip)
