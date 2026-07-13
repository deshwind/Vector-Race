"""Command-line interface for optimization, simulation, and PPO training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from math import cos, pi
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import AppConfig, load_config, make_silverstone_config
from .pipeline import BuildResult, build_trajectory
from .simulation import GripAwareResidualPolicy, LapSimulator
from .track import Track, make_demo_track, make_silverstone_track
from .trajectory import RacingTrajectory
from .visualization import plot_simulation, plot_speed_profile, plot_trajectory


BUILT_IN_CIRCUITS = ("silverstone", "synthetic")


def _configured(
    path: Path | None,
    points: int | None = None,
    circuit: str | None = None,
) -> AppConfig:
    config = (
        make_silverstone_config()
        if path is None and circuit == "silverstone"
        else load_config(path)
    )
    if points is not None:
        config = replace(
            config,
            optimizer=replace(config.optimizer, points=points),
        )
    return config


def _built_in_track(circuit: str, source_points: int | None = None) -> Track:
    if circuit == "silverstone":
        return make_silverstone_track(source_points)
    if circuit == "synthetic":
        return make_demo_track(180 if source_points is None else source_points)
    raise ValueError(f"unsupported built-in circuit: {circuit}")


def _write_build_outputs(result: BuildResult, output: Path) -> None:
    result.save(output)
    plot_trajectory(result.track, result.trajectory, output / "racing_line.png")
    plot_speed_profile(result.trajectory, output / "speed_profile.png")


def _print_build_summary(result: BuildResult, output: Path) -> None:
    data = result.diagnostics
    summary = {
        "output_directory": str(output.resolve()),
        "track_name": result.track.name,
        "pipeline_success": result.success,
        "optimizer_success": data["optimization"]["success"],
        "optimizer_message": data["optimization"]["message"],
        "speed_profile_converged": data["speed_profile"]["converged"],
        "centerline_speed_profile_converged": data[
            "centerline_speed_profile"
        ]["converged"],
        "circuit_centerline_length_m": round(data["track"]["length_m"], 3),
        "optimized_line_length_m": round(data["trajectory"]["length_m"], 3),
        "estimated_lap_time_s": round(data["trajectory"]["lap_time_s"], 3),
        "centerline_lap_time_s": round(
            data["trajectory"]["centerline_lap_time_s"], 3
        ),
        "estimated_improvement_percent": round(
            data["trajectory"]["estimated_lap_time_improvement_percent"], 2
        ),
        "minimum_speed_kph": round(
            data["trajectory"]["minimum_speed_mps"] * 3.6, 2
        ),
        "maximum_speed_kph": round(
            data["trajectory"]["maximum_speed_mps"] * 3.6, 2
        ),
    }
    print(json.dumps(summary, indent=2))


def command_demo(args: argparse.Namespace) -> int:
    config = _configured(args.config, args.points, args.circuit)
    result = build_trajectory(_built_in_track(args.circuit), config)
    output = args.output or Path("outputs") / args.circuit
    _write_build_outputs(result, output)

    simulation_succeeded = True
    if not args.skip_simulation and result.success:
        simulator = LapSimulator(result.trajectory, config)
        policy = None
        grip_profile = None
        if args.variable_grip:
            nominal = config.vehicle.base_grip_mu
            policy = GripAwareResidualPolicy(nominal)

            def variable_grip_profile(progress: float) -> float:
                return nominal * (
                    0.72 + 0.28 * (0.5 + 0.5 * cos(2.0 * pi * progress))
                )

            grip_profile = variable_grip_profile
        simulation = simulator.run(policy=policy, grip_profile=grip_profile)
        simulation_succeeded = simulation.summary.completed
        simulation.save(output / "simulation")
        plot_simulation(
            result.track,
            result.trajectory,
            simulation,
            output / "simulation" / "driven_path.png",
        )
        print(
            f"Simulation: {simulation.summary.termination_reason}; "
            f"time={simulation.summary.simulated_time_s:.2f}s; "
            f"off-track events={simulation.summary.off_track_events}"
        )
    elif not args.skip_simulation:
        print(
            "Simulation skipped because trajectory generation did not converge; "
            "inspect optimizer_diagnostics.json",
            file=sys.stderr,
        )
    _print_build_summary(result, output)
    return 0 if result.success and simulation_succeeded else 2


def command_optimize(args: argparse.Namespace) -> int:
    config = _configured(args.config, args.points)
    result = build_trajectory(Track.from_csv(args.track), config)
    _write_build_outputs(result, args.output)
    _print_build_summary(result, args.output)
    return 0 if result.success else 2


def command_simulate(args: argparse.Namespace) -> int:
    config = _configured(args.config)
    track = Track.from_csv(args.track)
    trajectory = RacingTrajectory.from_csv(args.trajectory)
    if track.point_count != trajectory.point_count:
        raise ValueError(
            "track and trajectory point counts differ; use resampled_track.csv "
            "created alongside baseline_trajectory.csv"
        )
    if track.local_grip_mu is not None:
        if trajectory.local_grip_mu is None:
            trajectory = replace(
                trajectory,
                local_grip_mu=track.local_grip_mu,
            )
        elif not np.allclose(
            trajectory.local_grip_mu,
            track.local_grip_mu,
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            raise ValueError("track and trajectory local grip profiles differ")

    if args.checkpoint is not None:
        from .rl import PPOPolicy

        policy = PPOPolicy(args.checkpoint, config)
    elif args.grip_aware:
        policy = GripAwareResidualPolicy(config.vehicle.base_grip_mu)
    else:
        policy = None
    result = LapSimulator(trajectory, config).run(policy=policy)
    result.save(args.output)
    plot_simulation(track, trajectory, result, args.output / "driven_path.png")
    print(json.dumps(result.summary.__dict__, indent=2))
    return 0 if result.summary.completed else 2


def command_train(args: argparse.Namespace) -> int:
    from .rl import train_ppo

    circuit = args.circuit if args.track is None else None
    config = _configured(args.config, args.points, circuit)
    track = (
        Track.from_csv(args.track)
        if args.track
        else _built_in_track(args.circuit)
    )
    result = build_trajectory(track, config)
    if not result.success:
        raise RuntimeError(
            "baseline trajectory generation did not converge; inspect the "
            "diagnostics before training"
        )
    _write_build_outputs(result, args.output / "baseline")
    checkpoint = train_ppo(
        result.trajectory,
        config,
        args.output,
        total_timesteps=args.timesteps,
    )
    print(f"Saved PPO checkpoint: {checkpoint.resolve()}")
    return 0


def command_export_demo_track(args: argparse.Namespace) -> int:
    track = _built_in_track(args.circuit, args.points)
    output = args.output or Path("tracks") / f"{args.circuit}.csv"
    destination = track.to_csv(output)
    print(f"Saved demo track: {destination.resolve()}")
    return 0


def command_web(args: argparse.Namespace) -> int:
    """Serve the interactive Silverstone dashboard on this computer."""

    from .webapp import serve_dashboard

    serve_dashboard(
        port=args.port,
        default_laps=args.laps,
        open_browser=not args.no_browser,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="racing-line",
        description="Safe hybrid racing-line optimization research prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="run a bundled circuit demo (Silverstone by default)"
    )
    demo.add_argument("--config", type=Path, help="YAML configuration path")
    demo.add_argument("--output", type=Path)
    demo.add_argument(
        "--circuit",
        choices=BUILT_IN_CIRCUITS,
        default="silverstone",
        help="bundled circuit to run (default: silverstone)",
    )
    demo.add_argument("--points", type=int, help="override optimization point count")
    demo.add_argument("--skip-simulation", action="store_true")
    demo.add_argument(
        "--variable-grip",
        action="store_true",
        help="evaluate the deterministic grip-aware residual policy",
    )
    demo.set_defaults(function=command_demo)

    optimize = subparsers.add_parser(
        "optimize", help="optimize a track supplied as CSV"
    )
    optimize.add_argument("track", type=Path)
    optimize.add_argument("--config", type=Path, help="YAML configuration path")
    optimize.add_argument("--output", type=Path, default=Path("outputs/optimized"))
    optimize.add_argument("--points", type=int, help="override optimization point count")
    optimize.set_defaults(function=command_optimize)

    simulate = subparsers.add_parser(
        "simulate", help="simulate an exported track and trajectory"
    )
    simulate.add_argument("track", type=Path)
    simulate.add_argument("trajectory", type=Path)
    simulate.add_argument("--config", type=Path, help="YAML configuration path")
    simulate.add_argument("--output", type=Path, default=Path("outputs/simulation"))
    replay_policy = simulate.add_mutually_exclusive_group()
    replay_policy.add_argument("--grip-aware", action="store_true")
    replay_policy.add_argument(
        "--checkpoint",
        type=Path,
        help="replay a Stable-Baselines3 PPO checkpoint",
    )
    simulate.set_defaults(function=command_simulate)

    train = subparsers.add_parser("train", help="train the optional PPO residual")
    train_source = train.add_mutually_exclusive_group()
    train_source.add_argument("--track", type=Path, help="custom track CSV")
    train_source.add_argument(
        "--circuit",
        choices=BUILT_IN_CIRCUITS,
        default="silverstone",
        help="bundled circuit when --track is omitted (default: silverstone)",
    )
    train.add_argument("--config", type=Path, help="YAML configuration path")
    train.add_argument("--output", type=Path, default=Path("models/ppo"))
    train.add_argument("--points", type=int, help="override optimization point count")
    train.add_argument("--timesteps", type=int, help="override PPO total timesteps")
    train.set_defaults(function=command_train)

    export = subparsers.add_parser(
        "export-demo-track", help="export a bundled circuit as project-format CSV"
    )
    export.add_argument(
        "--circuit",
        choices=BUILT_IN_CIRCUITS,
        default="silverstone",
    )
    export.add_argument("--points", type=int, help="optional resampled point count")
    export.add_argument("--output", type=Path)
    export.set_defaults(function=command_export_demo_track)

    web = subparsers.add_parser(
        "web",
        help="open the interactive Silverstone simulation dashboard",
    )
    web.add_argument(
        "--laps",
        type=int,
        default=10,
        help="laps to run automatically when the page opens (default: 10)",
    )
    web.add_argument(
        "--port",
        type=int,
        default=8765,
        help="local browser server port (default: 8765)",
    )
    web.add_argument(
        "--no-browser",
        action="store_true",
        help="start the server without opening the browser",
    )
    web.set_defaults(function=command_web)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
