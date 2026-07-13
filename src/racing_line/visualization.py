"""Matplotlib visualisations for trajectories and simulation telemetry."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .simulation import SimulationResult
from .track import Track
from .trajectory import RacingTrajectory


def _track_boundaries(track: Track) -> tuple[np.ndarray, ...]:
    dx = np.roll(track.x_m, -1) - np.roll(track.x_m, 1)
    dy = np.roll(track.y_m, -1) - np.roll(track.y_m, 1)
    norm = np.maximum(np.hypot(dx, dy), 1.0e-12)
    nx = -dy / norm
    ny = dx / norm
    left_x = track.x_m + nx * track.width_left_m
    left_y = track.y_m + ny * track.width_left_m
    right_x = track.x_m - nx * track.width_right_m
    right_y = track.y_m - ny * track.width_right_m
    return left_x, left_y, right_x, right_y


def _closed(values: np.ndarray) -> np.ndarray:
    return np.append(values, values[0])


def _direction(track: Track) -> str:
    twice_area = np.sum(
        track.x_m * np.roll(track.y_m, -1)
        - np.roll(track.x_m, -1) * track.y_m
    )
    return "clockwise" if twice_area < 0.0 else "counter-clockwise"


def _add_source_note(figure: plt.Figure, track_name: str) -> None:
    if "silverstone" in track_name.lower():
        layout_engine = figure.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0.0, 0.045, 1.0, 0.955))
        figure.text(
            0.01,
            0.008,
            "Circuit geometry: TUMFTM racetrack-database; research "
            "reconstruction derived from © OpenStreetMap contributors.",
            fontsize=7,
            color="#4d5560",
        )


def plot_trajectory(
    track: Track,
    trajectory: RacingTrajectory,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    left_x, left_y, right_x, right_y = _track_boundaries(track)

    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    axis.fill(
        np.concatenate([_closed(left_x), _closed(right_x)[::-1]]),
        np.concatenate([_closed(left_y), _closed(right_y)[::-1]]),
        color="#252b35",
        alpha=0.86,
        label="track",
    )
    axis.plot(_closed(left_x), _closed(left_y), color="white", linewidth=1.1)
    axis.plot(_closed(right_x), _closed(right_y), color="white", linewidth=1.1)
    axis.plot(
        _closed(track.x_m),
        _closed(track.y_m),
        color="#8c96a3",
        linestyle="--",
        linewidth=0.8,
        label="centreline",
    )
    scatter = axis.scatter(
        trajectory.x_m,
        trajectory.y_m,
        c=trajectory.speed_mps * 3.6,
        cmap="turbo",
        s=14,
        label="racing line",
        zorder=4,
    )
    colorbar = figure.colorbar(scatter, ax=axis, shrink=0.82)
    colorbar.set_label("Target speed (km/h)")
    axis.scatter(
        trajectory.x_m[0],
        trajectory.y_m[0],
        marker="o",
        s=52,
        color="#62f6a5",
        edgecolor="black",
        zorder=5,
        label="lap origin",
    )
    arrow_index = max(1, min(track.point_count - 1, track.point_count // 80))
    axis.annotate(
        "",
        xy=(track.x_m[arrow_index], track.y_m[arrow_index]),
        xytext=(track.x_m[0], track.y_m[0]),
        arrowprops={"arrowstyle": "->", "color": "#62f6a5", "lw": 2.0},
        zorder=6,
    )
    direction = _direction(track)
    axis.set_title(
        title or f"{track.name} — optimized racing line ({direction})"
    )
    axis.set_xlabel("local x (m)")
    axis.set_ylabel("local y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15)
    axis.legend(loc="best")
    _add_source_note(figure, track.name)
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return destination


def plot_speed_profile(trajectory: RacingTrajectory, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, (speed_axis, curvature_axis) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True
    )
    speed_axis.plot(
        trajectory.s_m,
        trajectory.speed_mps * 3.6,
        color="#e13b46",
        linewidth=1.8,
    )
    speed_axis.set_ylabel("Target speed (km/h)")
    speed_axis.grid(alpha=0.25)
    track_name = str(trajectory.metadata.get("track_name", "Track"))
    speed_axis.set_title(
        f"{track_name} — velocity profile; estimated lap "
        f"{trajectory.lap_time_s:.2f} s"
    )
    curvature_axis.plot(
        trajectory.s_m,
        trajectory.curvature_1pm,
        color="#3579d4",
        linewidth=1.3,
    )
    curvature_axis.axhline(0.0, color="black", linewidth=0.6)
    curvature_axis.set_xlabel("Distance around lap (m)")
    curvature_axis.set_ylabel("Curvature (1/m)")
    curvature_axis.grid(alpha=0.25)
    _add_source_note(figure, track_name)
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return destination


def plot_simulation(
    track: Track,
    trajectory: RacingTrajectory,
    result: SimulationResult,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    left_x, left_y, right_x, right_y = _track_boundaries(track)
    telemetry_x = np.asarray([float(row["x_m"]) for row in result.telemetry])
    telemetry_y = np.asarray([float(row["y_m"]) for row in result.telemetry])

    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    axis.fill(
        np.concatenate([_closed(left_x), _closed(right_x)[::-1]]),
        np.concatenate([_closed(left_y), _closed(right_y)[::-1]]),
        color="#30343d",
        alpha=0.75,
    )
    axis.plot(
        _closed(trajectory.x_m),
        _closed(trajectory.y_m),
        "--",
        color="#e13b46",
        linewidth=1.0,
        label="reference",
    )
    if len(telemetry_x):
        axis.plot(
            telemetry_x,
            telemetry_y,
            color="#4cc9f0",
            linewidth=1.2,
            label="simulated vehicle",
        )
    status = "complete" if result.summary.completed else result.summary.termination_reason
    axis.set_title(
        f"{track.name} — closed-loop simulation; {status} "
        f"({_direction(track)})"
    )
    axis.set_xlabel("local x (m)")
    axis.set_ylabel("local y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15)
    axis.legend(loc="best")
    _add_source_note(figure, track.name)
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return destination
