"""Deterministic lap simulation and telemetry capture."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

import numpy as np

from .config import AppConfig
from .controller import TrajectoryController
from .safety import ResidualAction, SafetyContext, SafetySupervisor
from .trajectory import RacingTrajectory
from .vehicle import SingleTrackModel, VehicleState


Policy = Callable[[Mapping[str, float]], ResidualAction]
GripProfile = Callable[[float], float]


class DrivingCondition(Protocol):
    """Runtime condition supplied by the optional race-strategy layer."""

    grip_mu: float
    reference_speed_scale: float
    tyre: str
    tyre_age_laps: float
    weather: str
    track_temp_c: float
    rain_intensity: float
    track_wetness: float


ConditionProfile = Callable[[int, float], DrivingCondition]


@dataclass(frozen=True)
class LapSummary:
    """Aggregate results for a requested run.

    ``lap_time_s`` retains its original meaning: the total elapsed time when
    the requested run completes. For multi-lap runs, ``lap_times_s`` contains
    the duration of each individually completed lap.
    """

    completed: bool
    lap_time_s: float | None
    simulated_time_s: float
    distance_m: float
    mean_speed_mps: float
    max_speed_mps: float
    max_abs_lateral_error_m: float
    minimum_vehicle_edge_clearance_m: float
    minimum_boundary_margin_slack_m: float
    off_track_events: int
    safety_interventions: int
    steps: int
    termination_reason: str
    requested_laps: int = 1
    completed_laps: int = 0
    lap_times_s: tuple[float, ...] = ()
    pit_stops: int = 0
    pit_stop_time_s: float = 0.0


@dataclass
class SimulationResult:
    summary: LapSummary
    telemetry: list[dict[str, float | int | str | bool]]

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        telemetry_path = output / "telemetry.csv"
        summary_path = output / "summary.json"
        if self.telemetry:
            with telemetry_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.telemetry[0]))
                writer.writeheader()
                writer.writerows(self.telemetry)
        else:
            telemetry_path.write_text("", encoding="utf-8")
        summary_path.write_text(
            json.dumps(asdict(self.summary), indent=2), encoding="utf-8"
        )
        return telemetry_path, summary_path


class GripAwareResidualPolicy:
    """Deterministic fallback policy illustrating the residual interface.

    It only reduces the speed reference as grip falls. It does not invent a
    wet racing line, because the review explicitly found insufficient evidence
    for a generic wet-line geometry rule.
    """

    def __init__(self, nominal_grip_mu: float, speed_margin: float = 0.98):
        if nominal_grip_mu <= 0:
            raise ValueError("nominal_grip_mu must be positive")
        if not 0 < speed_margin <= 1:
            raise ValueError("speed_margin must be in (0, 1]")
        self.nominal_grip_mu = nominal_grip_mu
        self.speed_margin = speed_margin

    def __call__(self, observation: Mapping[str, float]) -> ResidualAction:
        grip = max(float(observation["grip_mu"]), 0.05)
        # The small margin covers controller/model mismatch near the theoretical
        # friction ceiling; the safety supervisor remains the final authority.
        scale = self.speed_margin * np.sqrt(grip / self.nominal_grip_mu)
        return ResidualAction(0.0, min(float(scale - 1.0), 0.0))


class LapSimulator:
    def __init__(self, trajectory: RacingTrajectory, config: AppConfig):
        self.trajectory = trajectory
        self.config = config
        self.model = SingleTrackModel(config.vehicle)
        self.controller = TrajectoryController(
            trajectory, config.vehicle, config.simulation
        )
        self.safety = SafetySupervisor(config.safety)

    def initial_state(self) -> VehicleState:
        speed = min(
            self.config.simulation.start_speed_mps,
            float(self.trajectory.speed_mps[0]),
        )
        return VehicleState(
            x_m=float(self.trajectory.x_m[0]),
            y_m=float(self.trajectory.y_m[0]),
            yaw_rad=float(self.trajectory.heading_rad[0]),
            vx_mps=speed,
        )

    def run(
        self,
        policy: Policy | None = None,
        grip_profile: GripProfile | None = None,
        *,
        lap_count: int = 1,
        condition_profile: ConditionProfile | None = None,
        pit_stop_losses_s: Mapping[int, float] | None = None,
    ) -> SimulationResult:
        """Run until all requested laps, an excursion, or the time limit.

        The configured time limit is applied per requested lap. Vehicle,
        controller, safety, and residual-policy state remain continuous across
        lap boundaries.
        """

        if isinstance(lap_count, bool) or not isinstance(lap_count, int):
            raise TypeError("lap_count must be an integer")
        if not 1 <= lap_count <= 100:
            raise ValueError("lap_count must be between 1 and 100")
        if grip_profile is not None and condition_profile is not None:
            raise ValueError(
                "grip_profile and condition_profile cannot be used together"
            )

        pit_losses: dict[int, float] = {}
        for after_lap, raw_loss in (pit_stop_losses_s or {}).items():
            if isinstance(after_lap, bool) or not isinstance(after_lap, int):
                raise TypeError("pit-stop lap numbers must be integers")
            if not 1 <= after_lap < lap_count:
                raise ValueError(
                    "pit-stop lap numbers must be between 1 and lap_count - 1"
                )
            if isinstance(raw_loss, bool):
                raise TypeError("pit-stop losses must be finite numbers")
            loss = float(raw_loss)
            if not np.isfinite(loss) or loss < 0.0:
                raise ValueError("pit-stop losses must be finite and non-negative")
            pit_losses[after_lap] = loss

        dt = self.config.simulation.time_step_s
        max_steps_per_lap = int(np.ceil(self.config.simulation.max_time_s / dt))
        max_steps = max_steps_per_lap * lap_count
        nominal_mu = self.config.vehicle.base_grip_mu
        trajectory_grip = getattr(self.trajectory, "local_grip_mu", None)

        self.controller.reset(0)
        self.safety.reset()
        state = self.initial_state()
        count = self.trajectory.point_count
        previous_index = 0
        unwrapped_index = 0
        travelled = 0.0
        off_track_events = 0
        consecutive_off_track_steps = 0
        interventions = 0
        max_lateral_error = 0.0
        minimum_vehicle_edge_clearance = np.inf
        previous_filtered = ResidualAction()
        telemetry: list[dict[str, float | int | str | bool]] = []
        completed = False
        completed_laps = 0
        lap_times: list[float] = []
        previous_lap_end_time = 0.0
        elapsed_pit_time = 0.0
        pit_stops = 0
        termination = "time_limit"

        for step in range(max_steps):
            index = self.controller.nearest_index(state.x_m, state.y_m)
            delta_index = (index - previous_index + count // 2) % count - count // 2
            # Do not let brief controller oscillations erase accumulated progress.
            if delta_index >= -2:
                unwrapped_index += delta_index
            previous_index = index
            progress = max(unwrapped_index / count, 0.0)
            lap_number = min(completed_laps + 1, lap_count)
            lap_progress = float(np.clip(progress - float(lap_number - 1), 0.0, 1.0))
            condition = (
                condition_profile(lap_number, lap_progress)
                if condition_profile is not None
                else None
            )
            if condition is not None:
                mu = float(condition.grip_mu)
                reference_speed_scale = float(condition.reference_speed_scale)
                if not np.isfinite(mu) or mu <= 0.0:
                    raise ValueError(
                        "condition_profile must return a positive finite grip_mu"
                    )
                if (
                    not np.isfinite(reference_speed_scale)
                    or reference_speed_scale <= 0.0
                ):
                    raise ValueError(
                        "condition_profile must return a positive finite "
                        "reference_speed_scale"
                    )
            elif grip_profile is not None:
                mu = float(grip_profile(progress % 1.0))
                reference_speed_scale = 1.0
            elif trajectory_grip is not None:
                mu = float(trajectory_grip[index])
                reference_speed_scale = 1.0
            else:
                mu = nominal_mu
                reference_speed_scale = 1.0

            baseline_speed = float(
                self.trajectory.speed_mps[index] * reference_speed_scale
            )

            dx = state.x_m - self.trajectory.x_m[index]
            dy = state.y_m - self.trajectory.y_m[index]
            lateral_error = float(
                dx * self.trajectory.normal_x[index]
                + dy * self.trajectory.normal_y[index]
            )
            max_lateral_error = max(max_lateral_error, abs(lateral_error))
            observation = {
                "progress": progress,
                "lateral_error_m": lateral_error,
                "heading_error_rad": float(
                    (self.trajectory.heading_rad[index] - state.yaw_rad + np.pi)
                    % (2.0 * np.pi)
                    - np.pi
                ),
                "speed_mps": state.speed_mps,
                "yaw_rate_rad_s": state.yaw_rate_rad_s,
                "steering_rad": state.steering_rad,
                "reference_speed_mps": baseline_speed,
                "reference_half_width_m": float(
                    min(
                        self.trajectory.width_left_m[index],
                        self.trajectory.width_right_m[index],
                    )
                ),
                "curvature_1pm": float(self.trajectory.curvature_1pm[index]),
                "grip_mu": mu,
                "previous_filtered_offset_m": previous_filtered.lateral_offset_m,
                "previous_filtered_speed_delta": previous_filtered.speed_scale_delta,
            }
            requested = policy(observation) if policy is not None else ResidualAction()
            filtered = self.safety.filter(
                requested,
                SafetyContext(
                    baseline_offset_m=float(self.trajectory.lateral_offset_m[index]),
                    width_left_m=float(self.trajectory.width_left_m[index]),
                    width_right_m=float(self.trajectory.width_right_m[index]),
                    vehicle_width_m=self.config.vehicle.width_m,
                    baseline_speed_mps=baseline_speed,
                    curvature_1pm=float(self.trajectory.curvature_1pm[index]),
                    grip_mu=mu,
                    dt_s=dt,
                ),
            )
            interventions += int(filtered.intervened)
            previous_filtered = ResidualAction(
                filtered.lateral_offset_m, filtered.speed_scale_delta
            )
            control, diagnostics = self.controller.control(
                state,
                filtered.target_speed_mps,
                filtered.lateral_offset_m,
            )
            next_state = self.model.step(state, control, dt, mu)
            travelled += float(
                np.hypot(next_state.x_m - state.x_m, next_state.y_m - state.y_m)
            )

            total_centerline_offset = (
                self.trajectory.lateral_offset_m[index] + lateral_error
            )
            half_vehicle = self.config.vehicle.width_m / 2.0
            vehicle_edge_clearance = float(
                min(
                    self.trajectory.width_left_m[index]
                    - total_centerline_offset
                    - half_vehicle,
                    self.trajectory.width_right_m[index]
                    + total_centerline_offset
                    - half_vehicle,
                )
            )
            minimum_vehicle_edge_clearance = min(
                minimum_vehicle_edge_clearance,
                vehicle_edge_clearance,
            )
            inside = vehicle_edge_clearance >= 0.0
            if not inside:
                off_track_events += 1
                consecutive_off_track_steps += 1
            else:
                consecutive_off_track_steps = 0

            telemetry.append(
                {
                    "time_s": step * dt + elapsed_pit_time,
                    "progress_laps": progress,
                    "lap_number": lap_number,
                    "requested_laps": lap_count,
                    "completed_laps": completed_laps,
                    "reference_index": index,
                    "x_m": state.x_m,
                    "y_m": state.y_m,
                    "yaw_rad": state.yaw_rad,
                    "speed_mps": state.speed_mps,
                    "lateral_error_m": lateral_error,
                    "vehicle_edge_clearance_m": vehicle_edge_clearance,
                    "heading_error_rad": diagnostics.heading_error_rad,
                    "reference_speed_mps": baseline_speed,
                    "target_speed_mps": filtered.target_speed_mps,
                    "steering_rad": control.steering_rad,
                    "acceleration_mps2": control.acceleration_mps2,
                    "requested_offset_m": requested.lateral_offset_m,
                    "filtered_offset_m": filtered.lateral_offset_m,
                    "requested_speed_delta": requested.speed_scale_delta,
                    "filtered_speed_delta": filtered.speed_scale_delta,
                    "grip_mu": mu,
                    "tyre": condition.tyre if condition is not None else "",
                    "tyre_age_laps": (
                        condition.tyre_age_laps if condition is not None else 0.0
                    ),
                    "weather": (condition.weather if condition is not None else ""),
                    "track_temp_c": (
                        condition.track_temp_c if condition is not None else 0.0
                    ),
                    "rain_intensity": (
                        condition.rain_intensity if condition is not None else 0.0
                    ),
                    "track_wetness": (
                        condition.track_wetness if condition is not None else 0.0
                    ),
                    "pit_stop": False,
                    "pit_stop_loss_s": 0.0,
                    "safety_intervened": filtered.intervened,
                    "safety_reasons": ";".join(filtered.reasons),
                    "inside_track": inside,
                }
            )
            state = next_state

            if consecutive_off_track_steps * dt >= 0.50:
                termination = "off_track"
                break

            next_lap_threshold = (completed_laps + 1) * count - 2
            if unwrapped_index >= next_lap_threshold and step > count:
                finished_lap = completed_laps + 1
                lap_end_time = len(telemetry) * dt + elapsed_pit_time
                lap_times.append(lap_end_time - previous_lap_end_time)
                previous_lap_end_time = lap_end_time
                completed_laps += 1
                telemetry[-1]["completed_laps"] = completed_laps

                pit_loss = pit_losses.get(finished_lap, 0.0)
                if pit_loss > 0.0:
                    elapsed_pit_time += pit_loss
                    pit_stops += 1
                    telemetry[-1]["pit_stop"] = True
                    telemetry[-1]["pit_stop_loss_s"] = pit_loss

            if completed_laps >= lap_count:
                completed = True
                termination = "lap_complete"
                break

        simulated_time = len(telemetry) * dt + elapsed_pit_time
        speeds = np.asarray([float(row["speed_mps"]) for row in telemetry])
        summary = LapSummary(
            completed=completed,
            lap_time_s=simulated_time if completed else None,
            simulated_time_s=simulated_time,
            distance_m=travelled,
            mean_speed_mps=float(np.mean(speeds)) if len(speeds) else 0.0,
            max_speed_mps=float(np.max(speeds)) if len(speeds) else 0.0,
            max_abs_lateral_error_m=max_lateral_error,
            minimum_vehicle_edge_clearance_m=float(minimum_vehicle_edge_clearance),
            minimum_boundary_margin_slack_m=float(
                minimum_vehicle_edge_clearance - self.config.safety.boundary_margin_m
            ),
            off_track_events=off_track_events,
            safety_interventions=interventions,
            steps=len(telemetry),
            termination_reason=termination,
            requested_laps=lap_count,
            completed_laps=completed_laps,
            lap_times_s=tuple(lap_times),
            pit_stops=pit_stops,
            pit_stop_time_s=elapsed_pit_time,
        )
        return SimulationResult(summary, telemetry)
