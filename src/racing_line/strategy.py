"""Transparent race-strategy assumptions for the Silverstone dashboard.

The real Formula 1 tyre maps, thermal models, degradation curves, and pit-loss
models are proprietary.  This module therefore implements a deterministic,
editable approximation.  The 2026 Silverstone slick labels use the public
C1/C2/C3 allocation (Hard/Medium/Soft), while every numerical performance
coefficient below is an explicit research assumption rather than official F1
telemetry.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, sqrt


TYRE_IDS = ("hard", "medium", "soft", "intermediate", "wet")
WEATHER_IDS = ("dry", "damp", "light_rain", "wet", "heavy_rain")
DEFAULT_PIT_STOP_LOSS_S = 25.0
REFERENCE_SPEED_MARGIN = 0.97
MODEL_NOTICE = (
    "Calibrated-style research approximation: tyre grip, warm-up, wear, "
    "temperature response, weather response, and pit loss are transparent "
    "estimates, not official Formula 1 telemetry."
)


@dataclass(frozen=True)
class _TyreParameters:
    label: str
    silverstone_compound: str
    color: str
    dry_grip_factor: float
    cold_grip_factor: float
    warmup_laps: float
    wear_per_lap: float
    ideal_track_temp_c: float
    temperature_window_c: float
    temperature_penalty_per_c: float
    wetness_factors: tuple[float, ...]


@dataclass(frozen=True)
class _WeatherParameters:
    label: str
    base_track_wetness: float
    reference_pace_factor: float
    suggested_track_temp_c: float
    suggested_rain_intensity: float


_WETNESS_KNOTS = (0.0, 0.20, 0.45, 0.70, 1.0)

# Factors are sampled at the wetness knots above.  Slicks lose performance as
# water accumulates, the Intermediate peaks around light rain, and the Full Wet
# becomes the best choice as standing water approaches the heavy-rain region.
_TYRES: dict[str, _TyreParameters] = {
    "hard": _TyreParameters(
        label="Hard",
        silverstone_compound="C1",
        color="#f3f4f6",
        dry_grip_factor=0.960,
        cold_grip_factor=0.920,
        warmup_laps=1.50,
        wear_per_lap=0.0020,
        ideal_track_temp_c=42.0,
        temperature_window_c=14.0,
        temperature_penalty_per_c=0.0030,
        wetness_factors=(1.00, 0.89, 0.68, 0.47, 0.30),
    ),
    "medium": _TyreParameters(
        label="Medium",
        silverstone_compound="C2",
        color="#ffd166",
        dry_grip_factor=0.985,
        cold_grip_factor=0.950,
        warmup_laps=1.00,
        wear_per_lap=0.0035,
        ideal_track_temp_c=37.0,
        temperature_window_c=12.0,
        temperature_penalty_per_c=0.0035,
        wetness_factors=(1.00, 0.88, 0.65, 0.44, 0.28),
    ),
    "soft": _TyreParameters(
        label="Soft",
        silverstone_compound="C3",
        color="#ff4057",
        dry_grip_factor=1.000,
        cold_grip_factor=0.980,
        warmup_laps=0.50,
        wear_per_lap=0.0060,
        ideal_track_temp_c=33.0,
        temperature_window_c=10.0,
        temperature_penalty_per_c=0.0040,
        wetness_factors=(1.00, 0.86, 0.61, 0.40, 0.25),
    ),
    "intermediate": _TyreParameters(
        label="Intermediate",
        silverstone_compound="Intermediate",
        color="#61d095",
        dry_grip_factor=0.820,
        cold_grip_factor=0.960,
        warmup_laps=0.50,
        wear_per_lap=0.0050,
        ideal_track_temp_c=22.0,
        temperature_window_c=10.0,
        temperature_penalty_per_c=0.0030,
        wetness_factors=(0.80, 0.96, 1.00, 0.86, 0.62),
    ),
    "wet": _TyreParameters(
        label="Full Wet",
        silverstone_compound="Full Wet",
        color="#4ea8ff",
        dry_grip_factor=0.740,
        cold_grip_factor=0.970,
        warmup_laps=0.40,
        wear_per_lap=0.0040,
        ideal_track_temp_c=16.0,
        temperature_window_c=9.0,
        temperature_penalty_per_c=0.0030,
        wetness_factors=(0.66, 0.78, 0.90, 0.98, 1.00),
    ),
}

_WEATHER: dict[str, _WeatherParameters] = {
    "dry": _WeatherParameters("Dry", 0.00, 1.00, 30.0, 0.00),
    "damp": _WeatherParameters("Damp", 0.18, 0.99, 24.0, 0.15),
    "light_rain": _WeatherParameters("Light rain", 0.42, 0.95, 20.0, 0.40),
    "wet": _WeatherParameters("Wet", 0.70, 0.89, 16.0, 0.70),
    "heavy_rain": _WeatherParameters("Heavy rain", 0.95, 0.80, 14.0, 1.00),
}


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _identifier(value: object, name: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"{name} must be one of: {choices}")
    return value


def _mapping(
    value: object,
    name: str,
    *,
    required: set[str],
    allowed: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - allowed
    if unknown:
        labels = ", ".join(sorted((repr(item) for item in unknown)))
        raise ValueError(f"unknown {name} field(s): {labels}")
    missing = required - set(value)
    if missing:
        labels = ", ".join(sorted(missing))
        raise ValueError(f"missing {name} field(s): {labels}")
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wetness_factor(parameters: _TyreParameters, wetness: float) -> float:
    position = bisect_right(_WETNESS_KNOTS, wetness)
    if position == 0:
        return parameters.wetness_factors[0]
    if position >= len(_WETNESS_KNOTS):
        return parameters.wetness_factors[-1]
    left = position - 1
    right = position
    span = _WETNESS_KNOTS[right] - _WETNESS_KNOTS[left]
    fraction = (wetness - _WETNESS_KNOTS[left]) / span
    return (
        parameters.wetness_factors[left] * (1.0 - fraction)
        + parameters.wetness_factors[right] * fraction
    )


@dataclass(frozen=True)
class TyreStint:
    """A tyre fitted at the beginning of a one-based race lap."""

    start_lap: int
    tyre: str

    def __post_init__(self) -> None:
        start_lap = _integer(self.start_lap, "stint.start_lap")
        if start_lap < 1:
            raise ValueError("stint.start_lap must be at least 1")
        _identifier(self.tyre, "stint.tyre", TYRE_IDS)

    def to_dict(self) -> dict[str, object]:
        return {"start_lap": self.start_lap, "tyre": self.tyre}


@dataclass(frozen=True)
class WeatherPhase:
    """Weather and track inputs taking effect at a one-based race lap."""

    start_lap: int
    condition: str
    track_temp_c: float
    rain_intensity: float

    def __post_init__(self) -> None:
        start_lap = _integer(self.start_lap, "weather.start_lap")
        if start_lap < 1:
            raise ValueError("weather.start_lap must be at least 1")
        _identifier(self.condition, "weather.condition", WEATHER_IDS)
        temperature = _number(self.track_temp_c, "weather.track_temp_c")
        rain = _number(self.rain_intensity, "weather.rain_intensity")
        if not -10.0 <= temperature <= 70.0:
            raise ValueError("weather.track_temp_c must be between -10 and 70")
        if not 0.0 <= rain <= 1.0:
            raise ValueError("weather.rain_intensity must be between 0 and 1")
        object.__setattr__(self, "track_temp_c", temperature)
        object.__setattr__(self, "rain_intensity", rain)

    def to_dict(self) -> dict[str, object]:
        return {
            "start_lap": self.start_lap,
            "condition": self.condition,
            "track_temp_c": self.track_temp_c,
            "rain_intensity": self.rain_intensity,
        }


@dataclass(frozen=True)
class RaceCondition:
    """Resolved physical inputs for one instant of a planned race."""

    grip_mu: float
    reference_speed_scale: float
    tyre: str
    tyre_age_laps: float
    weather: str
    track_temp_c: float
    rain_intensity: float
    track_wetness: float

    def __post_init__(self) -> None:
        positive = {
            "condition.grip_mu": self.grip_mu,
            "condition.reference_speed_scale": self.reference_speed_scale,
        }
        for name, value in positive.items():
            if _number(value, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if _number(self.tyre_age_laps, "condition.tyre_age_laps") < 0.0:
            raise ValueError("condition.tyre_age_laps cannot be negative")
        _identifier(self.tyre, "condition.tyre", TYRE_IDS)
        _identifier(self.weather, "condition.weather", WEATHER_IDS)
        temperature = _number(self.track_temp_c, "condition.track_temp_c")
        rain = _number(self.rain_intensity, "condition.rain_intensity")
        wetness = _number(self.track_wetness, "condition.track_wetness")
        if not -10.0 <= temperature <= 70.0:
            raise ValueError("condition.track_temp_c must be between -10 and 70")
        if not 0.0 <= rain <= 1.0:
            raise ValueError("condition.rain_intensity must be between 0 and 1")
        if not 0.0 <= wetness <= 1.0:
            raise ValueError("condition.track_wetness must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "grip_mu": self.grip_mu,
            "reference_speed_scale": self.reference_speed_scale,
            "tyre": self.tyre,
            "tyre_age_laps": self.tyre_age_laps,
            "weather": self.weather,
            "track_temp_c": self.track_temp_c,
            "rain_intensity": self.rain_intensity,
            "track_wetness": self.track_wetness,
        }


@dataclass(frozen=True)
class RacePlan:
    """Validated race distance, tyre stints, and evolving weather phases."""

    laps: int
    stints: tuple[TyreStint, ...]
    weather: tuple[WeatherPhase, ...]
    pit_stop_loss_s: float = DEFAULT_PIT_STOP_LOSS_S

    def __post_init__(self) -> None:
        laps = _integer(self.laps, "laps")
        if laps < 1:
            raise ValueError("laps must be at least 1")
        stints = tuple(self.stints)
        weather = tuple(self.weather)
        if not stints:
            raise ValueError("stints must contain at least one stint")
        if not weather:
            raise ValueError("weather must contain at least one phase")
        if not all(isinstance(item, TyreStint) for item in stints):
            raise ValueError("stints must contain TyreStint values")
        if not all(isinstance(item, WeatherPhase) for item in weather):
            raise ValueError("weather must contain WeatherPhase values")
        self._validate_start_laps(stints, "stints")
        self._validate_start_laps(weather, "weather")
        for previous, current in zip(stints, stints[1:], strict=False):
            if previous.tyre == current.tyre:
                raise ValueError(
                    "consecutive stints must use different tyres; combine "
                    f"the {current.tyre} stints instead"
                )
        pit_loss = _number(self.pit_stop_loss_s, "pit_stop_loss_s")
        if pit_loss < 0.0:
            raise ValueError("pit_stop_loss_s cannot be negative")
        object.__setattr__(self, "stints", stints)
        object.__setattr__(self, "weather", weather)
        object.__setattr__(self, "pit_stop_loss_s", pit_loss)

    def _validate_start_laps(
        self,
        phases: tuple[TyreStint, ...] | tuple[WeatherPhase, ...],
        name: str,
    ) -> None:
        starts = [item.start_lap for item in phases]
        if starts[0] != 1:
            raise ValueError(f"the first {name} start_lap must be 1")
        if any(start > self.laps for start in starts):
            raise ValueError(f"{name} start_lap cannot exceed the race distance")
        if any(current <= previous for previous, current in zip(starts, starts[1:])):
            raise ValueError(f"{name} start_lap values must be strictly increasing")

    @property
    def pit_stop_laps(self) -> tuple[int, ...]:
        """Laps that begin with the fixed pit-stop time loss."""

        return tuple(stint.start_lap for stint in self.stints[1:])

    @property
    def pit_stop_count(self) -> int:
        return len(self.pit_stop_laps)

    @property
    def total_pit_loss_s(self) -> float:
        return self.pit_stop_count * self.pit_stop_loss_s

    @property
    def pit_events(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "before_lap": current.start_lap,
                "from_tyre": previous.tyre,
                "to_tyre": current.tyre,
                "loss_s": self.pit_stop_loss_s,
            }
            for previous, current in zip(self.stints, self.stints[1:], strict=False)
        )

    def pit_loss_before_lap(self, lap_number: int) -> float:
        lap = _integer(lap_number, "lap_number")
        if not 1 <= lap <= self.laps:
            raise ValueError(f"lap_number must be between 1 and {self.laps}")
        return self.pit_stop_loss_s if lap in self.pit_stop_laps else 0.0

    def _stint_at(self, lap_number: int) -> TyreStint:
        starts = tuple(item.start_lap for item in self.stints)
        return self.stints[bisect_right(starts, lap_number) - 1]

    def _weather_at(self, lap_number: int) -> WeatherPhase:
        starts = tuple(item.start_lap for item in self.weather)
        return self.weather[bisect_right(starts, lap_number) - 1]

    def condition_at(
        self,
        lap_number: int,
        lap_progress: float,
        nominal_mu: float = 1.70,
    ) -> RaceCondition:
        """Resolve grip and pace for a point within a one-based race lap.

        ``lap_progress`` is in ``[0, 1]``.  Tyre warm-up and degradation are
        continuous across a stint and reset when a later stint starts.
        Weather changes take effect at the start lap of each weather phase.
        """

        lap = _integer(lap_number, "lap_number")
        if not 1 <= lap <= self.laps:
            raise ValueError(f"lap_number must be between 1 and {self.laps}")
        progress = _number(lap_progress, "lap_progress")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("lap_progress must be between 0 and 1")
        base_mu = _number(nominal_mu, "nominal_mu")
        if base_mu <= 0.0:
            raise ValueError("nominal_mu must be positive")

        stint = self._stint_at(lap)
        phase = self._weather_at(lap)
        tyre = _TYRES[stint.tyre]
        weather = _WEATHER[phase.condition]
        tyre_age = float(lap - stint.start_lap) + progress

        warmup_fraction = min(tyre_age / tyre.warmup_laps, 1.0)
        warmup_factor = (
            tyre.cold_grip_factor + (1.0 - tyre.cold_grip_factor) * warmup_fraction
        )

        # The named weather state supplies most of the surface estimate while
        # user-controlled rain intensity shifts it within that broad state.
        track_wetness = _clamp(
            0.65 * weather.base_track_wetness + 0.35 * phase.rain_intensity,
            0.0,
            1.0,
        )
        compatibility = _wetness_factor(tyre, track_wetness)

        temperature_error = max(
            abs(phase.track_temp_c - tyre.ideal_track_temp_c)
            - tyre.temperature_window_c,
            0.0,
        )
        temperature_factor = max(
            0.80,
            1.0 - temperature_error * tyre.temperature_penalty_per_c,
        )

        wear_multiplier = 1.0
        if stint.tyre == "intermediate":
            wear_multiplier += max(0.0, 0.45 - track_wetness) * 5.0
        elif stint.tyre == "wet":
            wear_multiplier += max(0.0, 0.65 - track_wetness) * 6.0
        wear_age = max(tyre_age - tyre.warmup_laps, 0.0)
        wear_factor = max(
            0.72,
            1.0 - tyre.wear_per_lap * wear_multiplier * wear_age,
        )

        grip_mu = max(
            base_mu
            * tyre.dry_grip_factor
            * compatibility
            * warmup_factor
            * temperature_factor
            * wear_factor,
            base_mu * 0.20,
        )
        speed_scale = _clamp(
            sqrt(grip_mu / base_mu)
            * weather.reference_pace_factor
            * REFERENCE_SPEED_MARGIN,
            0.40,
            1.02,
        )
        return RaceCondition(
            grip_mu=grip_mu,
            reference_speed_scale=speed_scale,
            tyre=stint.tyre,
            tyre_age_laps=tyre_age,
            weather=phase.condition,
            track_temp_c=phase.track_temp_c,
            rain_intensity=phase.rain_intensity,
            track_wetness=track_wetness,
        )

    def to_request_dict(self) -> dict[str, object]:
        """Return the canonical JSON request representation."""

        return {
            "laps": self.laps,
            "stints": [item.to_dict() for item in self.stints],
            "weather": [item.to_dict() for item in self.weather],
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly plan including derived pit information."""

        result = self.to_request_dict()
        result.update(
            {
                "pit_stop_loss_s": self.pit_stop_loss_s,
                "pit_stop_count": self.pit_stop_count,
                "total_pit_loss_s": self.total_pit_loss_s,
                "pit_events": list(self.pit_events),
                "model_notice": MODEL_NOTICE,
            }
        )
        return result


def _parse_stints(value: object, laps: int) -> tuple[TyreStint, ...]:
    if not isinstance(value, list):
        raise ValueError("stints must be an array")
    if not value:
        raise ValueError("stints must contain at least one stint")
    result: list[TyreStint] = []
    for index, raw_item in enumerate(value):
        name = f"stints[{index}]"
        item = _mapping(
            raw_item,
            name,
            required={"start_lap", "tyre"},
            allowed={"start_lap", "tyre"},
        )
        result.append(
            TyreStint(
                start_lap=_integer(item["start_lap"], f"{name}.start_lap"),
                tyre=_identifier(item["tyre"], f"{name}.tyre", TYRE_IDS),
            )
        )
    plan_stints = tuple(result)
    if any(item.start_lap > laps for item in plan_stints):
        raise ValueError("stints start_lap cannot exceed the race distance")
    return plan_stints


def _parse_weather(value: object, laps: int) -> tuple[WeatherPhase, ...]:
    if not isinstance(value, list):
        raise ValueError("weather must be an array")
    if not value:
        raise ValueError("weather must contain at least one phase")
    result: list[WeatherPhase] = []
    fields = {"start_lap", "condition", "track_temp_c", "rain_intensity"}
    for index, raw_item in enumerate(value):
        name = f"weather[{index}]"
        item = _mapping(
            raw_item,
            name,
            required=fields,
            allowed=fields,
        )
        result.append(
            WeatherPhase(
                start_lap=_integer(item["start_lap"], f"{name}.start_lap"),
                condition=_identifier(
                    item["condition"], f"{name}.condition", WEATHER_IDS
                ),
                track_temp_c=_number(item["track_temp_c"], f"{name}.track_temp_c"),
                rain_intensity=_number(
                    item["rain_intensity"], f"{name}.rain_intensity"
                ),
            )
        )
    plan_weather = tuple(result)
    if any(item.start_lap > laps for item in plan_weather):
        raise ValueError("weather start_lap cannot exceed the race distance")
    return plan_weather


def parse_race_plan(
    payload: Mapping[str, object] | None,
    default_laps: int = 10,
    max_laps: int = 25,
) -> RacePlan:
    """Validate a browser request and return a canonical immutable race plan."""

    maximum = _integer(max_laps, "max_laps")
    if maximum < 1:
        raise ValueError("max_laps must be at least 1")
    fallback_laps = _integer(default_laps, "default_laps")
    if not 1 <= fallback_laps <= maximum:
        raise ValueError("default_laps must be between 1 and max_laps")
    if payload is None:
        request: Mapping[str, object] = {}
    else:
        request = _mapping(
            payload,
            "request",
            required=set(),
            allowed={"laps", "stints", "weather"},
        )
    laps = _integer(request.get("laps", fallback_laps), "laps")
    if not 1 <= laps <= maximum:
        raise ValueError(f"laps must be between 1 and {maximum}")

    if "stints" in request:
        stints = _parse_stints(request["stints"], laps)
    else:
        stints = (TyreStint(1, "medium"),)
    if "weather" in request:
        weather = _parse_weather(request["weather"], laps)
    else:
        weather = (WeatherPhase(1, "dry", 30.0, 0.0),)
    return RacePlan(laps=laps, stints=stints, weather=weather)


def tyre_catalog() -> list[dict[str, object]]:
    """Return JSON-friendly tyre choices and the model's explicit estimates."""

    return [
        {
            "id": tyre_id,
            "label": values.label,
            "silverstone_compound": values.silverstone_compound,
            "color": values.color,
            "dry_grip_factor": values.dry_grip_factor,
            "cold_grip_factor": values.cold_grip_factor,
            "warmup_laps": values.warmup_laps,
            "wear_per_lap": values.wear_per_lap,
            "ideal_track_temp_c": values.ideal_track_temp_c,
            "temperature_window_c": values.temperature_window_c,
            "wetness_curve": [
                {"track_wetness": wetness, "grip_factor": factor}
                for wetness, factor in zip(
                    _WETNESS_KNOTS, values.wetness_factors, strict=True
                )
            ],
        }
        for tyre_id, values in _TYRES.items()
    ]


def weather_catalog() -> list[dict[str, object]]:
    """Return JSON-friendly named weather presets for the strategy editor."""

    return [
        {
            "id": weather_id,
            "label": values.label,
            "base_track_wetness": values.base_track_wetness,
            "reference_pace_factor": values.reference_pace_factor,
            "suggested_track_temp_c": values.suggested_track_temp_c,
            "suggested_rain_intensity": values.suggested_rain_intensity,
        }
        for weather_id, values in _WEATHER.items()
    ]


def strategy_catalog() -> dict[str, object]:
    """Return all choices, defaults, fixed loss, and disclosure for the UI."""

    return {
        "tyres": tyre_catalog(),
        "weather_conditions": weather_catalog(),
        "pit_stop_loss_s": DEFAULT_PIT_STOP_LOSS_S,
        "reference_speed_margin": REFERENCE_SPEED_MARGIN,
        "defaults": {
            "tyre": "medium",
            "condition": "dry",
            "track_temp_c": 30.0,
            "rain_intensity": 0.0,
        },
        "model_notice": MODEL_NOTICE,
    }


__all__ = [
    "DEFAULT_PIT_STOP_LOSS_S",
    "MODEL_NOTICE",
    "RaceCondition",
    "RacePlan",
    "REFERENCE_SPEED_MARGIN",
    "TYRE_IDS",
    "TyreStint",
    "WEATHER_IDS",
    "WeatherPhase",
    "parse_race_plan",
    "strategy_catalog",
    "tyre_catalog",
    "weather_catalog",
]
