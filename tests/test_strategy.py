from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from racing_line.config import AppConfig
from racing_line.simulation import LapSimulator
from racing_line.strategy import (
    DEFAULT_PIT_STOP_LOSS_S,
    RacePlan,
    TyreStint,
    WeatherPhase,
    parse_race_plan,
    strategy_catalog,
)
from racing_line.trajectory import RacingTrajectory
from racing_line.vehicle import VehicleControl, VehicleState


def _weather(
    start_lap: int = 1,
    condition: str = "dry",
    track_temp_c: float = 30.0,
    rain_intensity: float = 0.0,
) -> dict[str, object]:
    return {
        "start_lap": start_lap,
        "condition": condition,
        "track_temp_c": track_temp_c,
        "rain_intensity": rain_intensity,
    }


def test_parse_race_plan_preserves_legacy_laps_request_with_defaults() -> None:
    plan = parse_race_plan({"laps": 10})

    assert plan.to_request_dict() == {
        "laps": 10,
        "stints": [{"start_lap": 1, "tyre": "medium"}],
        "weather": [_weather()],
    }
    assert plan.pit_stop_count == 0
    assert plan.total_pit_loss_s == 0.0


def test_parse_race_plan_resolves_multi_stint_changing_weather_request() -> None:
    request = {
        "laps": 10,
        "stints": [
            {"start_lap": 1, "tyre": "soft"},
            {"start_lap": 4, "tyre": "hard"},
            {"start_lap": 8, "tyre": "intermediate"},
        ],
        "weather": [
            _weather(1, "dry", 32.0, 0.0),
            _weather(7, "light_rain", 20.0, 0.45),
        ],
    }

    plan = parse_race_plan(request)

    assert plan.to_request_dict() == request
    assert plan.pit_stop_laps == (4, 8)
    assert plan.pit_stop_count == 2
    assert plan.total_pit_loss_s == 2 * DEFAULT_PIT_STOP_LOSS_S
    assert plan.pit_events == (
        {
            "before_lap": 4,
            "from_tyre": "soft",
            "to_tyre": "hard",
            "loss_s": DEFAULT_PIT_STOP_LOSS_S,
        },
        {
            "before_lap": 8,
            "from_tyre": "hard",
            "to_tyre": "intermediate",
            "loss_s": DEFAULT_PIT_STOP_LOSS_S,
        },
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"laps": 3, "stints": [{"start_lap": 2, "tyre": "soft"}]},
            "first stints start_lap must be 1",
        ),
        (
            {
                "laps": 3,
                "weather": [_weather(2, "dry", 30.0, 0.0)],
            },
            "first weather start_lap must be 1",
        ),
        (
            {
                "laps": 5,
                "stints": [
                    {"start_lap": 1, "tyre": "soft"},
                    {"start_lap": 4, "tyre": "hard"},
                    {"start_lap": 3, "tyre": "medium"},
                ],
            },
            "strictly increasing",
        ),
        (
            {
                "laps": 5,
                "weather": [
                    _weather(1),
                    _weather(3, "damp", 24.0, 0.2),
                    _weather(3, "wet", 18.0, 0.7),
                ],
            },
            "strictly increasing",
        ),
        (
            {
                "laps": 3,
                "stints": [
                    {"start_lap": 1, "tyre": "soft"},
                    {"start_lap": 4, "tyre": "hard"},
                ],
            },
            "cannot exceed the race distance",
        ),
        (
            {
                "laps": 3,
                "stints": [
                    {"start_lap": 1, "tyre": "medium"},
                    {"start_lap": 2, "tyre": "medium"},
                ],
            },
            "consecutive stints must use different tyres",
        ),
    ],
)
def test_schedule_start_laps_and_consecutive_tyres_are_validated(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_race_plan(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"laps": True}, "laps must be an integer"),
        ({"laps": 2.0}, "laps must be an integer"),
        (
            {"laps": 2, "stints": [{"start_lap": True, "tyre": "soft"}]},
            "start_lap must be an integer",
        ),
        (
            {"laps": 2, "stints": [{"start_lap": 1, "tyre": "supersoft"}]},
            "stints.*tyre must be one of",
        ),
        (
            {"laps": 2, "weather": [_weather(condition="storm")]},
            "weather.*condition must be one of",
        ),
        (
            {"laps": 2, "weather": [_weather(track_temp_c=float("nan"))]},
            "track_temp_c must be a finite number",
        ),
        (
            {"laps": 2, "weather": [_weather(track_temp_c=71.0)]},
            "track_temp_c must be between -10 and 70",
        ),
        (
            {"laps": 2, "weather": [_weather(rain_intensity=True)]},
            "rain_intensity must be a finite number",
        ),
        (
            {"laps": 2, "weather": [_weather(rain_intensity=1.1)]},
            "rain_intensity must be between 0 and 1",
        ),
    ],
)
def test_strategy_enums_and_numeric_inputs_are_validated(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_race_plan(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"stints": {}}, "stints must be an array"),
        ({"weather": []}, "weather must contain at least one phase"),
        (
            {"stints": [{"start_lap": 1}]},
            "missing stints.* field.*tyre",
        ),
        (
            {"weather": [{**_weather(), "humidity": 80}]},
            "unknown weather.* field.*humidity",
        ),
        ({"laps": 2, "fuel_kg": 100}, "unknown request field.*fuel_kg"),
    ],
)
def test_strategy_request_shapes_and_fields_are_strict(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_race_plan(payload)


def test_tyre_weather_temperature_and_rain_change_effective_grip() -> None:
    dry_soft = RacePlan(
        laps=3,
        stints=(TyreStint(1, "soft"),),
        weather=(WeatherPhase(1, "dry", 33.0, 0.0),),
    )
    dry_hard = replace(dry_soft, stints=(TyreStint(1, "hard"),))
    rain_soft = replace(
        dry_soft,
        weather=(WeatherPhase(1, "heavy_rain", 14.0, 1.0),),
    )
    rain_wet = replace(rain_soft, stints=(TyreStint(1, "wet"),))
    cold_soft = replace(
        dry_soft,
        weather=(WeatherPhase(1, "dry", 5.0, 0.0),),
    )

    dry_soft_condition = dry_soft.condition_at(2, 0.0)
    assert dry_soft_condition.grip_mu > dry_hard.condition_at(2, 0.0).grip_mu
    assert dry_soft_condition.grip_mu > rain_soft.condition_at(2, 0.0).grip_mu
    assert (
        rain_wet.condition_at(2, 0.0).grip_mu > rain_soft.condition_at(2, 0.0).grip_mu
    )
    assert dry_soft_condition.grip_mu > cold_soft.condition_at(2, 0.0).grip_mu


def test_condition_boundaries_reset_tyre_age_and_switch_weather() -> None:
    plan = RacePlan(
        laps=5,
        stints=(TyreStint(1, "soft"), TyreStint(3, "intermediate")),
        weather=(
            WeatherPhase(1, "dry", 32.0, 0.0),
            WeatherPhase(3, "light_rain", 20.0, 0.5),
        ),
    )

    end_of_old_stint = plan.condition_at(2, 1.0)
    start_of_new_stint = plan.condition_at(3, 0.0)

    assert end_of_old_stint.tyre == "soft"
    assert end_of_old_stint.tyre_age_laps == pytest.approx(2.0)
    assert end_of_old_stint.weather == "dry"
    assert start_of_new_stint.tyre == "intermediate"
    assert start_of_new_stint.tyre_age_laps == 0.0
    assert start_of_new_stint.weather == "light_rain"
    assert start_of_new_stint.track_temp_c == 20.0
    assert start_of_new_stint.rain_intensity == 0.5
    assert start_of_new_stint.track_wetness > end_of_old_stint.track_wetness


def test_strategy_catalog_discloses_choices_defaults_and_model_assumptions() -> None:
    catalog = strategy_catalog()

    assert {item["id"] for item in catalog["tyres"]} == {
        "soft",
        "medium",
        "hard",
        "intermediate",
        "wet",
    }
    assert {item["id"] for item in catalog["weather_conditions"]} == {
        "dry",
        "damp",
        "light_rain",
        "wet",
        "heavy_rain",
    }
    assert catalog["pit_stop_loss_s"] == DEFAULT_PIT_STOP_LOSS_S
    assert "not official Formula 1 telemetry" in str(catalog["model_notice"])


def _circular_trajectory(point_count: int = 8) -> RacingTrajectory:
    radius_m = 10.0
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    return RacingTrajectory(
        s_m=radius_m * theta,
        x_m=radius_m * np.cos(theta),
        y_m=radius_m * np.sin(theta),
        heading_rad=theta + np.pi / 2.0,
        curvature_1pm=np.full(point_count, 1.0 / radius_m),
        speed_mps=np.full(point_count, 8.0),
        lateral_offset_m=np.zeros(point_count),
        width_left_m=np.full(point_count, 100.0),
        width_right_m=np.full(point_count, 100.0),
    )


class _IndexSequenceController:
    def __init__(self, indices: list[int]) -> None:
        self.indices = iter(indices)

    def reset(self, _index: int = 0) -> None:
        return None

    def nearest_index(self, _x_m: float, _y_m: float) -> int:
        return next(self.indices)

    def control(
        self,
        _state: VehicleState,
        _target_speed_mps: float,
        _residual_offset_m: float,
    ) -> tuple[VehicleControl, SimpleNamespace]:
        return VehicleControl(0.0, 0.0), SimpleNamespace(heading_error_rad=0.0)


class _ContinuousModel:
    def step(
        self,
        state: VehicleState,
        _control: VehicleControl,
        _dt_s: float,
        _grip_mu: float,
    ) -> VehicleState:
        return replace(state, x_m=state.x_m + 0.001)


def test_simulator_applies_condition_profile_and_accounts_for_pit_loss() -> None:
    plan = RacePlan(
        laps=2,
        stints=(TyreStint(1, "medium"), TyreStint(2, "wet")),
        weather=(
            WeatherPhase(1, "dry", 30.0, 0.0),
            WeatherPhase(2, "heavy_rain", 14.0, 1.0),
        ),
    )
    config = AppConfig()
    config = replace(
        config,
        simulation=replace(
            config.simulation,
            time_step_s=0.1,
            max_time_s=2.0,
            start_speed_mps=8.0,
        ),
    )
    simulator = LapSimulator(_circular_trajectory(), config)
    simulator.controller = _IndexSequenceController(
        [index for _ in range(3) for index in range(8) for _ in range(2)]
    )
    simulator.model = _ContinuousModel()

    result = simulator.run(
        lap_count=2,
        condition_profile=plan.condition_at,
        pit_stop_losses_s={1: plan.pit_stop_loss_s},
    )

    assert result.summary.completed
    assert result.summary.lap_times_s == pytest.approx((1.3, 26.6))
    assert result.summary.simulated_time_s == pytest.approx(27.9)
    assert result.summary.pit_stops == 1
    assert result.summary.pit_stop_time_s == DEFAULT_PIT_STOP_LOSS_S

    pit_index = next(
        index for index, row in enumerate(result.telemetry) if row["pit_stop"]
    )
    pit_row = result.telemetry[pit_index]
    next_row = result.telemetry[pit_index + 1]
    assert pit_row["lap_number"] == 1
    assert pit_row["pit_stop_loss_s"] == DEFAULT_PIT_STOP_LOSS_S
    assert float(next_row["time_s"]) - float(pit_row["time_s"]) == pytest.approx(
        DEFAULT_PIT_STOP_LOSS_S + config.simulation.time_step_s
    )
    assert next_row["lap_number"] == 2
    assert next_row["tyre"] == "wet"
    assert next_row["weather"] == "heavy_rain"
    assert float(next_row["grip_mu"]) != pytest.approx(float(pit_row["grip_mu"]))
