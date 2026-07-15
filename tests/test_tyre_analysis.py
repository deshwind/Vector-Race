from __future__ import annotations

import json
from copy import deepcopy

import pytest

from racing_line.tyre_analysis import analyse_tyre_performance


def _payload(
    *,
    compound: str = "soft",
    track_temp_c: float = 30.0,
    weather: str = "dry",
    rain: float = 0.0,
) -> dict[str, object]:
    progress = [index / 5.0 for index in range(11)]
    return {
        "track": {"length_m": 5000.0},
        "telemetry": {
            "progress_laps": progress,
            "lap_number": [1] * 6 + [2] * 5,
            "time_s": [index * 10.0 for index in range(11)],
            "x_m": [index * 100.0 for index in range(11)],
            "y_m": [0.0] * 11,
            "speed_kph": [220.0] * 11,
            "target_speed_kph": [225.0] * 11,
            "tyre": [compound] * 11,
            "tyre_age_laps": progress,
            "weather": [weather] * 11,
            "track_temp_c": [track_temp_c] * 11,
            "rain_intensity": [rain] * 11,
        },
        "summary": {
            "requested_laps": 2,
            "completed_laps": 2,
            "lap_times_s": [90.0, 91.0],
        },
    }


def test_tyre_analysis_is_json_safe_and_contains_explainable_sections() -> None:
    result = analyse_tyre_performance(_payload())

    assert len(result["laps"]) == 2
    assert len(result["stints"]) == 1
    assert result["causes"]
    assert result["recommendations"]
    assert "not measured tread depth" in result["model_notice"]
    assert result["overview"]["peak_wear_index_pct"] > 0.0
    json.dumps(result, allow_nan=False)


def test_soft_compound_and_hot_track_raise_the_estimated_wear_index() -> None:
    hard = analyse_tyre_performance(_payload(compound="hard"))
    soft = analyse_tyre_performance(_payload(compound="soft"))
    hot_soft = analyse_tyre_performance(_payload(compound="soft", track_temp_c=65.0))

    assert (
        soft["overview"]["peak_wear_index_pct"]
        > hard["overview"]["peak_wear_index_pct"]
    )
    assert (
        hot_soft["overview"]["peak_wear_index_pct"]
        > soft["overview"]["peak_wear_index_pct"]
    )
    assert any(cause["id"] == "thermal_hot" for cause in hot_soft["causes"])


def test_dry_running_wet_tyre_is_explained_and_changes_strategy_advice() -> None:
    result = analyse_tyre_performance(_payload(compound="wet"))

    mismatch = next(
        cause for cause in result["causes"] if cause["id"] == "condition_mismatch"
    )
    assert mismatch["contribution_pct"] > 0.0
    assert any(
        recommendation.get("suggested_compound") == "medium"
        for recommendation in result["recommendations"]
    )


def test_tyre_change_resets_wear_and_creates_a_second_stint() -> None:
    payload = _payload(compound="soft")
    telemetry = payload["telemetry"]
    assert isinstance(telemetry, dict)
    telemetry["tyre"] = ["soft"] * 6 + ["hard"] * 5
    telemetry["tyre_age_laps"] = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
    ]

    result = analyse_tyre_performance(payload)

    assert [stint["compound"] for stint in result["stints"]] == ["soft", "hard"]
    assert result["laps"][1]["wear_index_pct"] < result["laps"][0]["wear_index_pct"]


@pytest.mark.parametrize("missing", ["telemetry", "progress_laps", "speed_kph"])
def test_missing_telemetry_returns_an_empty_review(missing: str) -> None:
    payload = deepcopy(_payload())
    if missing == "telemetry":
        payload.pop("telemetry")
    else:
        telemetry = payload["telemetry"]
        assert isinstance(telemetry, dict)
        telemetry.pop(missing)

    result = analyse_tyre_performance(payload)

    assert result["laps"] == []
    assert result["stints"] == []
    assert result["recommendations"] == []
