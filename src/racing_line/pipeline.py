"""End-to-end construction of an optimized, speed-annotated trajectory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig
from .geometry import path_geometry, resample_closed_path
from .optimizer import MinimumCurvatureResult, optimize_minimum_curvature
from .speed_profile import SpeedProfileResult, generate_speed_profile
from .track import Track
from .trajectory import RacingTrajectory


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class BuildResult:
    track: Track
    trajectory: RacingTrajectory
    optimization: MinimumCurvatureResult
    speed_profile: SpeedProfileResult
    centerline_speed_profile: SpeedProfileResult

    @property
    def success(self) -> bool:
        """Whether every numerical stage reported convergence."""

        return bool(
            self.optimization.success
            and self.speed_profile.diagnostics.converged
            and self.centerline_speed_profile.diagnostics.converged
        )

    @property
    def diagnostics(self) -> dict[str, Any]:
        minimum_center_to_edge = float(
            min(
                np.min(
                    self.trajectory.width_left_m
                    - self.trajectory.lateral_offset_m
                ),
                np.min(
                    self.trajectory.width_right_m
                    + self.trajectory.lateral_offset_m
                ),
            )
        )
        configured_center_to_edge_clearance = float(
            np.median(
                np.concatenate(
                    (
                        self.track.width_left_m - self.optimization.upper_bound,
                        self.track.width_right_m + self.optimization.lower_bound,
                    )
                )
            )
        )
        minimum_offset_bound_slack = float(
            min(
                np.min(
                    self.trajectory.lateral_offset_m
                    - self.optimization.lower_bound
                ),
                np.min(
                    self.optimization.upper_bound
                    - self.trajectory.lateral_offset_m
                ),
            )
        )
        return {
            "success": self.success,
            "track": {
                "name": self.track.name,
                "points": self.track.point_count,
                "length_m": self.track.length_m,
            },
            "optimization": {
                "success": self.optimization.success,
                "message": self.optimization.message,
                **asdict(self.optimization.diagnostics),
            },
            "speed_profile": {
                "lap_time_s": self.speed_profile.lap_time_s,
                **asdict(self.speed_profile.diagnostics),
            },
            "centerline_speed_profile": {
                "lap_time_s": self.centerline_speed_profile.lap_time_s,
                **asdict(self.centerline_speed_profile.diagnostics),
            },
            "trajectory": {
                "length_m": self.trajectory.length_m,
                "lap_time_s": self.trajectory.lap_time_s,
                "centerline_lap_time_s": self.centerline_speed_profile.lap_time_s,
                "estimated_lap_time_improvement_s": (
                    self.centerline_speed_profile.lap_time_s
                    - self.trajectory.lap_time_s
                ),
                "estimated_lap_time_improvement_percent": 100.0
                * (
                    self.centerline_speed_profile.lap_time_s
                    - self.trajectory.lap_time_s
                )
                / self.centerline_speed_profile.lap_time_s,
                "minimum_speed_mps": float(np.min(self.trajectory.speed_mps)),
                "maximum_speed_mps": float(np.max(self.trajectory.speed_mps)),
                "maximum_abs_curvature_1pm": float(
                    np.max(np.abs(self.trajectory.curvature_1pm))
                ),
                "minimum_center_to_edge_m": minimum_center_to_edge,
                "configured_center_to_edge_clearance_m": (
                    configured_center_to_edge_clearance
                ),
                "minimum_offset_bound_slack_m": minimum_offset_bound_slack,
                # Backward-compatible alias retained for existing consumers.
                "minimum_boundary_margin_m": minimum_center_to_edge,
            },
        }

    def save(self, directory: str | Path) -> dict[str, Path]:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "track": self.track.to_csv(output / "resampled_track.csv"),
            "trajectory": self.trajectory.to_csv(output / "baseline_trajectory.csv"),
            "diagnostics": output / "optimizer_diagnostics.json",
        }
        paths["diagnostics"].write_text(
            json.dumps(_plain(self.diagnostics), indent=2), encoding="utf-8"
        )
        return paths


def _generate_profile(
    x: np.ndarray,
    y: np.ndarray,
    curvature: np.ndarray,
    local_grip: np.ndarray | None,
    config: AppConfig,
) -> SpeedProfileResult:
    speed_config = config.speed_profile
    result = generate_speed_profile(
        x,
        y,
        curvature=curvature,
        mu=(
            config.vehicle.base_grip_mu
            if local_grip is None
            else local_grip
        ),
        max_speed=config.vehicle.max_speed_mps,
        max_accel=config.vehicle.max_accel_mps2,
        max_brake=config.vehicle.max_brake_mps2,
        gravity=speed_config.gravity_mps2,
        safety_factor=speed_config.lateral_safety_factor,
        max_iterations=speed_config.max_iterations,
        tolerance=speed_config.convergence_tolerance_mps,
        use_friction_circle=True,
    )
    observed_minimum = float(np.min(result.speed))
    configured_minimum = speed_config.minimum_speed_mps
    minimum_tolerance = max(
        1.0e-9,
        speed_config.convergence_tolerance_mps,
    )
    if (
        configured_minimum > 0.0
        and observed_minimum + minimum_tolerance < configured_minimum
    ):
        raise RuntimeError(
            "feasible speed profile falls below "
            f"speed_profile.minimum_speed_mps ({observed_minimum:.3f} < "
            f"{configured_minimum:.3f}); reduce the configured minimum or "
            "change the track/vehicle constraints"
        )
    return result


def build_trajectory(track: Track, config: AppConfig) -> BuildResult:
    """Resample, optimize, and add a feasible cyclic velocity profile."""

    fields: list[np.ndarray] = [track.width_left_m, track.width_right_m]
    if track.local_grip_mu is not None:
        fields.append(track.local_grip_mu)
    resampled = resample_closed_path(
        track.x_m,
        track.y_m,
        config.optimizer.points,
        *fields,
    )
    centre_x, centre_y, left_width, right_width = resampled[:4]
    local_grip = resampled[4] if len(resampled) == 5 else None
    prepared_track = Track(
        x_m=centre_x,
        y_m=centre_y,
        width_left_m=left_width,
        width_right_m=right_width,
        local_grip_mu=local_grip,
        name=track.name,
    )
    centerline_geometry = path_geometry(centre_x, centre_y)
    centerline_speed = _generate_profile(
        centerline_geometry.x,
        centerline_geometry.y,
        centerline_geometry.curvature,
        local_grip,
        config,
    )

    optimizer = config.optimizer
    optimized = optimize_minimum_curvature(
        centre_x,
        centre_y,
        left_width,
        right_width,
        vehicle_width_m=config.vehicle.width_m,
        safety_margin_m=optimizer.safety_margin_m,
        smoothness_weight=optimizer.smoothness_weight,
        offset_regularization=optimizer.offset_regularization,
        max_iterations=optimizer.max_iterations,
    )
    geometry = path_geometry(optimized.x, optimized.y)
    speed = _generate_profile(
        geometry.x,
        geometry.y,
        geometry.curvature,
        local_grip,
        config,
    )
    if np.any(speed.speed <= 0):
        raise RuntimeError("speed-profile solver produced a non-positive speed")

    metadata = {
        "track_name": track.name,
        "optimizer_success": optimized.success,
        "optimizer_message": optimized.message,
        "speed_profile_converged": speed.diagnostics.converged,
    }
    trajectory = RacingTrajectory(
        s_m=geometry.s,
        x_m=geometry.x,
        y_m=geometry.y,
        heading_rad=np.arctan2(geometry.tangent_y, geometry.tangent_x),
        curvature_1pm=geometry.curvature,
        speed_mps=speed.speed,
        lateral_offset_m=optimized.offset,
        width_left_m=prepared_track.width_left_m,
        width_right_m=prepared_track.width_right_m,
        local_grip_mu=prepared_track.local_grip_mu,
        metadata=metadata,
    )
    return BuildResult(
        prepared_track,
        trajectory,
        optimized,
        speed,
        centerline_speed,
    )
