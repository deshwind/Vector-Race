"""Explainable tyre-condition estimates for browser simulation results.

The simulator has no access to a team's tyre sensors, construction maps, or
measured tread depth.  This module therefore produces a deterministic review
index from the simulated path, speed, compound, weather, and track-temperature
inputs.  The outputs are deliberately named as estimates and are intended for
scenario comparison rather than real vehicle decisions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import atan2, exp, hypot, isfinite, pi
from typing import Any


MODEL_NOTICE = (
    "Estimated research model: wear and tyre temperatures are inferred from "
    "simulated distance, load, compound, track temperature, and wetness. They "
    "are not measured tread depth, sensor temperatures, or official telemetry."
)


@dataclass(frozen=True)
class _Compound:
    label: str
    base_wear_per_km: float
    ideal_surface_temp_c: float
    temperature_window_c: float
    heat_offset_c: float


_COMPOUNDS: dict[str, _Compound] = {
    "hard": _Compound("Hard", 0.12, 105.0, 15.0, 68.0),
    "medium": _Compound("Medium", 0.17, 100.0, 14.0, 65.0),
    "soft": _Compound("Soft", 0.25, 95.0, 13.0, 62.0),
    "intermediate": _Compound("Intermediate", 0.20, 75.0, 15.0, 52.0),
    "wet": _Compound("Full Wet", 0.18, 65.0, 15.0, 48.0),
}

_WEATHER_WETNESS = {
    "dry": 0.00,
    "damp": 0.18,
    "light_rain": 0.42,
    "wet": 0.70,
    "heavy_rain": 0.95,
}


def _finite(value: object, fallback: float = 0.0) -> float:
    if isinstance(value, bool):
        return fallback
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return result if isfinite(result) else fallback


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _series(value: object) -> Sequence[object]:
    return value if isinstance(value, (list, tuple)) else ()


def _at(values: Sequence[object], index: int, fallback: object = None) -> object:
    return values[index] if index < len(values) else fallback


def _wrapped_angle(value: float) -> float:
    return (value + pi) % (2.0 * pi) - pi


def _track_length(payload: Mapping[str, Any], supplied: float | None) -> float:
    if supplied is not None and isfinite(supplied) and supplied > 0.0:
        return float(supplied)
    track = payload.get("track")
    if isinstance(track, Mapping):
        explicit = _finite(track.get("length_m"), 0.0)
        if explicit > 0.0:
            return explicit
        xs = _series(track.get("x_m"))
        ys = _series(track.get("y_m"))
        count = min(len(xs), len(ys))
        if count > 2:
            points = [(_finite(xs[i]), _finite(ys[i])) for i in range(count)]
            length = sum(
                hypot(
                    points[(index + 1) % count][0] - point[0],
                    points[(index + 1) % count][1] - point[1],
                )
                for index, point in enumerate(points)
            )
            if length > 0.0:
                return length
    return 5000.0


def _wetness(weather: str, rain: float, supplied: float | None) -> float:
    if supplied is not None and isfinite(supplied):
        return _clamp(float(supplied), 0.0, 1.0)
    base = _WEATHER_WETNESS.get(weather, rain)
    return _clamp(0.65 * base + 0.35 * rain, 0.0, 1.0)


def _mismatch(compound: str, wetness: float) -> tuple[float, float, str]:
    """Return wear extra, performance penalty, and a concise condition label."""

    if compound in {"hard", "medium", "soft"}:
        return (
            max(0.0, wetness - 0.10) * 0.20,
            max(0.0, wetness - 0.08) * 52.0,
            "slick tyre on a wet surface",
        )
    if compound == "intermediate":
        dry = max(0.0, 0.35 - wetness)
        heavy = max(0.0, wetness - 0.72)
        return (
            dry * 4.0 + heavy * 0.6,
            dry * 30.0 + heavy * 18.0,
            ("intermediate outside its wetness range"),
        )
    dry = max(0.0, 0.55 - wetness)
    return dry * 5.0, dry * 48.0, "full-wet tyre on a drying surface"


def _risk(wear: float, performance: float, thermal_error: float) -> str:
    score = max(wear, 100.0 - performance, thermal_error * 2.4)
    if score >= 70.0:
        return "critical"
    if score >= 48.0:
        return "high"
    if score >= 25.0:
        return "moderate"
    return "low"


def _impact(contribution: float) -> str:
    if contribution >= 35.0:
        return "high"
    if contribution >= 15.0:
        return "moderate"
    return "low"


def _empty_review(message: str) -> dict[str, Any]:
    return {
        "model_notice": MODEL_NOTICE,
        "review": message,
        "overview": {},
        "laps": [],
        "stints": [],
        "causes": [],
        "recommendations": [],
    }


def analyse_tyre_performance(
    payload: Mapping[str, Any],
    *,
    circuit_length_m: float | None = None,
) -> dict[str, Any]:
    """Return an explainable tyre review for a browser simulation payload."""

    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return _empty_review("Tyre analysis needs simulation telemetry.")

    progress = _series(telemetry.get("progress_laps"))
    speeds = _series(telemetry.get("speed_kph"))
    laps = _series(telemetry.get("lap_number"))
    compounds = _series(telemetry.get("tyre"))
    if not progress or not speeds:
        return _empty_review("Tyre analysis needs distance and speed telemetry.")

    count = min(len(progress), len(speeds))
    length_m = _track_length(payload, circuit_length_m)
    times = _series(telemetry.get("time_s"))
    xs = _series(telemetry.get("x_m"))
    ys = _series(telemetry.get("y_m"))
    target_speeds = _series(telemetry.get("target_speed_kph"))
    tyre_ages = _series(telemetry.get("tyre_age_laps"))
    weather_values = _series(telemetry.get("weather"))
    temperatures = _series(telemetry.get("track_temp_c"))
    rain_values = _series(telemetry.get("rain_intensity"))
    wetness_values = _series(telemetry.get("track_wetness"))

    lap_points: dict[int, dict[str, Any]] = {}
    lap_accumulators: dict[int, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    stint_points: dict[int, list[dict[str, Any]]] = defaultdict(list)
    contributions: dict[str, float] = defaultdict(float)

    previous_progress = _finite(progress[0])
    previous_time = _finite(_at(times, 0), 0.0)
    previous_speed_mps = max(0.0, _finite(speeds[0]) / 3.6)
    previous_heading: float | None = None
    previous_tyre_age = max(0.0, _finite(_at(tyre_ages, 0), 0.0))
    current_compound = str(_at(compounds, 0, "medium") or "medium").lower()
    if current_compound not in _COMPOUNDS:
        current_compound = "medium"
    stint_number = 1
    wear_index = 0.0
    parameters = _COMPOUNDS[current_compound]
    initial_track_temp = _finite(_at(temperatures, 0), 30.0)
    surface_temp = max(
        initial_track_temp + 25.0, parameters.ideal_surface_temp_c - 22.0
    )
    core_temp = max(initial_track_temp + 20.0, parameters.ideal_surface_temp_c - 28.0)
    peak_wetness = 0.0
    peak_lateral_g = 0.0
    peak_longitudinal_g = 0.0
    peak_surface_temp = surface_temp
    minimum_track_temp = initial_track_temp
    maximum_track_temp = initial_track_temp

    for index in range(count):
        compound = str(
            _at(compounds, index, current_compound) or current_compound
        ).lower()
        if compound not in _COMPOUNDS:
            compound = current_compound
        tyre_age = max(0.0, _finite(_at(tyre_ages, index), previous_tyre_age))
        changed = compound != current_compound or tyre_age + 0.15 < previous_tyre_age
        if changed:
            current_compound = compound
            parameters = _COMPOUNDS[current_compound]
            stint_number += 1
            wear_index = 0.0
            track_temp = _finite(_at(temperatures, index), initial_track_temp)
            surface_temp = max(
                track_temp + 25.0,
                parameters.ideal_surface_temp_c - 22.0,
            )
            core_temp = max(
                track_temp + 20.0,
                parameters.ideal_surface_temp_c - 28.0,
            )

        lap_number = max(
            1,
            int(round(_finite(_at(laps, index), int(_finite(progress[index])) + 1))),
        )
        progress_value = _finite(progress[index], previous_progress)
        delta_progress = max(0.0, progress_value - previous_progress)
        distance_m = delta_progress * length_m

        time_value = _finite(_at(times, index), previous_time)
        raw_dt = max(0.0, time_value - previous_time)
        dt = _clamp(raw_dt, 0.05, 4.0) if index else 0.25
        speed_mps = max(0.0, _finite(speeds[index]) / 3.6)
        longitudinal_g = _clamp(
            abs(speed_mps - previous_speed_mps) / max(dt, 0.05) / 9.81,
            0.0,
            3.5,
        )

        heading: float | None = None
        if index and index < len(xs) and index < len(ys):
            dx = _finite(xs[index]) - _finite(xs[index - 1])
            dy = _finite(ys[index]) - _finite(ys[index - 1])
            if hypot(dx, dy) > 0.05:
                heading = atan2(dy, dx)
        lateral_g = 0.0
        if heading is not None and previous_heading is not None and distance_m > 0.5:
            curvature = abs(_wrapped_angle(heading - previous_heading)) / distance_m
            lateral_g = _clamp(speed_mps * speed_mps * curvature / 9.81, 0.0, 6.5)
        if heading is not None:
            previous_heading = heading

        weather = str(_at(weather_values, index, "dry") or "dry")
        rain = _clamp(_finite(_at(rain_values, index), 0.0), 0.0, 1.0)
        supplied_wetness = (
            _finite(wetness_values[index]) if index < len(wetness_values) else None
        )
        track_wetness = _wetness(weather, rain, supplied_wetness)
        track_temp = _finite(_at(temperatures, index), 30.0)
        rain_cooling = track_wetness * (
            10.0 if current_compound in {"intermediate", "wet"} else 20.0
        )
        target_surface_temp = (
            track_temp
            + parameters.heat_offset_c
            + 4.5 * lateral_g
            + 1.5 * longitudinal_g
            - rain_cooling
        )
        surface_alpha = 1.0 - exp(-dt / 16.0)
        core_alpha = 1.0 - exp(-dt / 55.0)
        surface_temp += (target_surface_temp - surface_temp) * surface_alpha
        core_target = 0.72 * surface_temp + 0.28 * (track_temp + 22.0)
        core_temp += (core_target - core_temp) * core_alpha

        lower_temp = parameters.ideal_surface_temp_c - parameters.temperature_window_c
        upper_temp = parameters.ideal_surface_temp_c + parameters.temperature_window_c
        hot_error = max(0.0, surface_temp - upper_temp)
        cold_error = max(0.0, lower_temp - surface_temp)
        lateral_extra = _clamp(max(0.0, lateral_g - 0.65) * 0.18, 0.0, 1.25)
        longitudinal_extra = _clamp(
            max(0.0, longitudinal_g - 0.25) * 0.10,
            0.0,
            0.55,
        )
        hot_extra = _clamp(hot_error * 0.035, 0.0, 1.8)
        cold_extra = _clamp(cold_error * 0.015, 0.0, 0.8)
        mismatch_extra, mismatch_penalty, mismatch_label = _mismatch(
            current_compound,
            track_wetness,
        )

        distance_km = distance_m / 1000.0
        base_wear = parameters.base_wear_per_km * distance_km
        extras = {
            "thermal_hot": hot_extra,
            "thermal_cold": cold_extra,
            "cornering_load": lateral_extra,
            "braking_traction": longitudinal_extra,
            "condition_mismatch": mismatch_extra,
        }
        wear_increment = base_wear * (1.0 + sum(extras.values()))
        wear_index = _clamp(wear_index + wear_increment, 0.0, 100.0)
        contributions["compound_distance"] += base_wear
        for name, multiplier in extras.items():
            contributions[name] += base_wear * multiplier

        temperature_error = hot_error + cold_error
        thermal_penalty = temperature_error * 0.75
        performance = _clamp(
            100.0 - wear_index * 0.48 - thermal_penalty - mismatch_penalty,
            15.0,
            100.0,
        )
        target_speed = max(
            1.0, _finite(_at(target_speeds, index), _finite(speeds[index]))
        )
        speed_shortfall = max(0.0, target_speed - _finite(speeds[index])) / target_speed
        pace_loss = _clamp(
            wear_index * 0.028
            + temperature_error * 0.035
            + mismatch_penalty * 0.045
            + speed_shortfall * 0.25,
            0.0,
            15.0,
        )
        point = {
            "lap": lap_number,
            "compound": current_compound,
            "stint": stint_number,
            "wear_index_pct": round(wear_index, 3),
            "performance_remaining_pct": round(performance, 3),
            "estimated_surface_temp_c": round(surface_temp, 2),
            "estimated_core_temp_c": round(core_temp, 2),
            "pace_loss_s": round(pace_loss, 3),
            "track_temp_c": round(track_temp, 2),
            "track_wetness": round(track_wetness, 4),
            "condition_note": mismatch_label if mismatch_extra > 0.05 else None,
            "risk": _risk(wear_index, performance, temperature_error),
        }
        lap_points[lap_number] = point
        stint_points[stint_number].append(point)
        accumulator = lap_accumulators[lap_number]
        accumulator["samples"] += 1.0
        accumulator["surface_temp"] += surface_temp
        accumulator["core_temp"] += core_temp
        accumulator["wetness"] += track_wetness
        accumulator["lateral_g"] += lateral_g
        accumulator["longitudinal_g"] += longitudinal_g
        accumulator["distance_km"] += distance_km
        accumulator["mismatch"] += mismatch_extra

        peak_wetness = max(peak_wetness, track_wetness)
        peak_lateral_g = max(peak_lateral_g, lateral_g)
        peak_longitudinal_g = max(peak_longitudinal_g, longitudinal_g)
        peak_surface_temp = max(peak_surface_temp, surface_temp)
        minimum_track_temp = min(minimum_track_temp, track_temp)
        maximum_track_temp = max(maximum_track_temp, track_temp)
        previous_progress = progress_value
        previous_time = time_value
        previous_speed_mps = speed_mps
        previous_tyre_age = tyre_age

    ordered_laps: list[dict[str, Any]] = []
    for lap_number, point in sorted(lap_points.items()):
        accumulator = lap_accumulators[lap_number]
        samples = max(1.0, accumulator["samples"])
        result = dict(point)
        result.update(
            {
                "average_surface_temp_c": round(
                    accumulator["surface_temp"] / samples,
                    2,
                ),
                "average_core_temp_c": round(
                    accumulator["core_temp"] / samples,
                    2,
                ),
                "average_track_wetness": round(
                    accumulator["wetness"] / samples,
                    4,
                ),
                "distance_km": round(accumulator["distance_km"], 3),
            }
        )
        ordered_laps.append(result)

    if not ordered_laps:
        return _empty_review(
            "The simulation did not cover enough distance for analysis."
        )

    cliff_lap = next(
        (
            int(point["lap"])
            for point in ordered_laps
            if point["wear_index_pct"] >= 65.0
            or point["performance_remaining_pct"] <= 68.0
        ),
        None,
    )
    stints: list[dict[str, Any]] = []
    for stint_id, points in sorted(stint_points.items()):
        if not points:
            continue
        first = points[0]
        last = points[-1]
        stint_cliff = next(
            (
                int(point["lap"])
                for point in points
                if point["wear_index_pct"] >= 65.0
                or point["performance_remaining_pct"] <= 68.0
            ),
            None,
        )
        end_wear = float(last["wear_index_pct"])
        end_performance = float(last["performance_remaining_pct"])
        stint_risk = max(
            (str(point["risk"]) for point in points),
            key=("low", "moderate", "high", "critical").index,
        )
        review = (
            f"{_COMPOUNDS[str(last['compound'])].label} ended with an estimated "
            f"{end_wear:.1f}% wear index and {end_performance:.1f}% relative "
            "performance remaining."
        )
        stints.append(
            {
                "stint": stint_id,
                "compound": last["compound"],
                "start_lap": int(first["lap"]),
                "end_lap": int(last["lap"]),
                "start_wear_index_pct": 0.0,
                "end_wear_index_pct": round(end_wear, 2),
                "end_performance_remaining_pct": round(end_performance, 2),
                "wear_rate_pct_per_lap": round(
                    end_wear / max(1, int(last["lap"]) - int(first["lap"]) + 1),
                    2,
                ),
                "peak_surface_temp_c": round(
                    max(float(point["estimated_surface_temp_c"]) for point in points),
                    2,
                ),
                "average_core_temp_c": round(
                    sum(float(point["estimated_core_temp_c"]) for point in points)
                    / len(points),
                    2,
                ),
                "pace_loss_s": round(float(last["pace_loss_s"]), 3),
                "projected_cliff_lap": stint_cliff,
                "risk": stint_risk,
                "review": review,
            }
        )

    causes = _cause_review(
        contributions,
        stints,
        peak_surface_temp,
        peak_lateral_g,
        peak_longitudinal_g,
        peak_wetness,
        minimum_track_temp,
        maximum_track_temp,
    )
    recommendations = _strategy_recommendations(
        stints,
        causes,
        ordered_laps,
        cliff_lap,
    )
    peak_wear = max(float(point["wear_index_pct"]) for point in ordered_laps)
    end_performance = float(ordered_laps[-1]["performance_remaining_pct"])
    maximum_pace_loss = max(float(point["pace_loss_s"]) for point in ordered_laps)
    overall_risk = max(
        (str(stint["risk"]) for stint in stints),
        key=("low", "moderate", "high", "critical").index,
    )
    return {
        "model_notice": MODEL_NOTICE,
        "review": (
            f"Estimated tyre condition reached a {peak_wear:.1f}% wear index. "
            "The ranked contributors explain the modelled degradation; risk "
            "labels describe potential conditions, not observed tyre damage."
        ),
        "overview": {
            "peak_wear_index_pct": round(peak_wear, 2),
            "end_performance_remaining_pct": round(end_performance, 2),
            "peak_surface_temp_c": round(peak_surface_temp, 2),
            "estimated_pace_loss_s": round(maximum_pace_loss, 3),
            "projected_cliff_lap": cliff_lap,
            "minimum_track_temp_c": round(minimum_track_temp, 2),
            "maximum_track_temp_c": round(maximum_track_temp, 2),
            "risk": overall_risk,
        },
        "laps": ordered_laps,
        "stints": stints,
        "causes": causes,
        "recommendations": recommendations,
    }


def _cause_review(
    contributions: Mapping[str, float],
    stints: Sequence[Mapping[str, Any]],
    peak_surface_temp: float,
    peak_lateral_g: float,
    peak_longitudinal_g: float,
    peak_wetness: float,
    minimum_track_temp: float,
    maximum_track_temp: float,
) -> list[dict[str, Any]]:
    total = max(sum(max(0.0, value) for value in contributions.values()), 1.0e-9)
    dominant_compound = str(
        max(stints, key=lambda stint: _finite(stint.get("end_wear_index_pct")))[
            "compound"
        ]
    )
    explanations = {
        "compound_distance": (
            "Compound and distance",
            f"The {_COMPOUNDS[dominant_compound].label} compound and distance "
            "travelled created the baseline degradation estimate.",
        ),
        "thermal_hot": (
            "High thermal load",
            f"Estimated surface temperature peaked at {peak_surface_temp:.1f} C; "
            f"the track input reached {maximum_track_temp:.1f} C and time above "
            "the compound window accelerated the wear estimate.",
        ),
        "thermal_cold": (
            "Cold-tyre sliding risk",
            f"With track temperature as low as {minimum_track_temp:.1f} C, "
            "running below the estimated tyre window increased the potential "
            "graining and sliding contribution.",
        ),
        "cornering_load": (
            "Cornering load",
            f"The simulated path reached an estimated {peak_lateral_g:.2f} g "
            "lateral-load proxy in its most demanding sections.",
        ),
        "braking_traction": (
            "Braking and traction",
            f"Longitudinal load peaked at an estimated {peak_longitudinal_g:.2f} g, "
            "adding energy through braking and acceleration zones.",
        ),
        "condition_mismatch": (
            "Tyre-to-surface mismatch",
            f"Track wetness reached {peak_wetness * 100.0:.0f}%; part of the run "
            "placed the selected tyre outside its intended condition range.",
        ),
    }
    causes: list[dict[str, Any]] = []
    for identifier, value in sorted(
        contributions.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        contribution = max(0.0, value) / total * 100.0
        if identifier != "compound_distance" and contribution < 0.5:
            continue
        title, explanation = explanations[identifier]
        causes.append(
            {
                "id": identifier,
                "title": title,
                "explanation": explanation,
                "contribution_pct": round(contribution, 2),
                "impact": _impact(contribution),
            }
        )
    return causes[:5]


def _strategy_recommendations(
    stints: Sequence[Mapping[str, Any]],
    causes: Sequence[Mapping[str, Any]],
    laps: Sequence[Mapping[str, Any]],
    cliff_lap: int | None,
) -> list[dict[str, Any]]:
    if not stints:
        return []
    cause_share = {
        str(cause.get("id")): _finite(cause.get("contribution_pct")) for cause in causes
    }
    worst = max(stints, key=lambda stint: _finite(stint.get("end_wear_index_pct")))
    compound = str(worst.get("compound", "medium"))
    end_wear = _finite(worst.get("end_wear_index_pct"))
    pace_loss = _finite(worst.get("pace_loss_s"))
    recommendations: list[dict[str, Any]] = []
    harder = {"soft": "medium", "medium": "hard"}.get(compound)

    if harder and (end_wear >= 35.0 or cause_share.get("thermal_hot", 0.0) >= 10.0):
        gain = _clamp(
            pace_loss
            * max(1, int(worst.get("end_lap", 1)) - int(worst.get("start_lap", 1)) + 1)
            * 0.30,
            0.2,
            8.0,
        )
        recommendations.append(
            {
                "title": f"Test the {harder.title()} compound",
                "action": f"Replace the {compound.title()} stint with {harder.title()} in a comparison run.",
                "rationale": "The harder option trades some peak grip for lower distance and thermal degradation in this model.",
                "suggested_compound": harder,
                "estimated_gain_s": round(gain, 2),
                "confidence": "medium",
            }
        )

    if cliff_lap is not None:
        suggested_lap = max(int(worst.get("start_lap", 1)) + 1, cliff_lap - 1)
        recommendations.append(
            {
                "title": "Move the stop before the projected cliff",
                "action": f"Trial a tyre change before lap {suggested_lap + 1}.",
                "rationale": f"Relative performance crosses the model's high-degradation threshold around lap {cliff_lap}.",
                "suggested_pit_after_lap": suggested_lap,
                "estimated_gain_s": round(_clamp(pace_loss * 0.75, 0.2, 6.0), 2),
                "confidence": "medium",
            }
        )

    if cause_share.get("condition_mismatch", 0.0) >= 8.0:
        wettest = max(
            laps, key=lambda point: _finite(point.get("average_track_wetness"))
        )
        wetness = _finite(wettest.get("average_track_wetness"))
        if wetness >= 0.68:
            suggested = "wet"
        elif wetness >= 0.25:
            suggested = "intermediate"
        else:
            suggested = "medium"
        recommendations.append(
            {
                "title": "Change at the weather crossover",
                "action": f"Test {suggested.replace('_', ' ').title()} tyres from lap {int(wettest.get('lap', 1))}.",
                "rationale": "The current tyre spent meaningful time outside its estimated wetness range.",
                "suggested_compound": suggested,
                "suggested_start_lap": int(wettest.get("lap", 1)),
                "confidence": "high",
            }
        )

    if cause_share.get("thermal_cold", 0.0) >= 15.0 and compound in {"hard", "medium"}:
        softer = "medium" if compound == "hard" else "soft"
        recommendations.append(
            {
                "title": "Use a faster-warming compound",
                "action": f"Compare the current stint with {softer.title()} tyres.",
                "rationale": "The tyre remained below its estimated working window, increasing warm-up and potential graining losses.",
                "suggested_compound": softer,
                "confidence": "medium",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "title": "Keep the current tyre sequence",
                "action": "Use this run as the baseline and change one strategy input at a time.",
                "rationale": "No high-wear cliff or strong condition mismatch was identified in the completed distance.",
                "confidence": "medium",
            }
        )
    return recommendations[:3]


__all__ = ["MODEL_NOTICE", "analyse_tyre_performance"]
