from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from racing_line.simulation import LapSummary, SimulationResult
from racing_line.strategy import parse_race_plan
from racing_line.webapp import (
    SimulationBusyError,
    _static_file,
    _telemetry_rows,
    create_dashboard_server,
    simulation_payload,
    validate_web_laps,
)


@pytest.mark.parametrize("value", [1, 10, 25, "10", " 10 "])
def test_validate_web_laps_accepts_supported_values(value: object) -> None:
    assert validate_web_laps(value) == int(value)


@pytest.mark.parametrize("value", [True, False, None, 0, 26, 1.5, "10.0", "laps"])
def test_validate_web_laps_rejects_unsafe_values(value: object) -> None:
    with pytest.raises(ValueError, match="laps must"):
        validate_web_laps(value)


def test_telemetry_decimation_keeps_endpoints_and_lap_transitions() -> None:
    rows = [
        {
            "time_s": float(index),
            "progress_laps": index / 4.0,
            "x_m": float(index),
            "y_m": 0.0,
            "speed_mps": 20.0,
        }
        for index in range(12)
    ]

    thinned = _telemetry_rows(rows, maximum_points=4)
    retained_times = {row["time_s"] for row in thinned}

    assert len(thinned) < len(rows)
    assert thinned[0] is rows[0]
    assert thinned[-1] is rows[-1]
    assert {3.0, 4.0, 7.0, 8.0} <= retained_times


def test_simulation_payload_projects_browser_schema_and_lap_results() -> None:
    build = SimpleNamespace(
        track=SimpleNamespace(
            name="Silverstone Circuit",
            x_m=np.array([0.0, 10.0, 20.0]),
            y_m=np.array([0.0, 5.0, 0.0]),
            width_left_m=np.array([7.0, 7.5, 7.0]),
            width_right_m=np.array([8.0, 8.5, 8.0]),
        ),
        trajectory=SimpleNamespace(
            x_m=np.array([1.0, 10.0, 19.0]),
            y_m=np.array([0.0, 4.0, 0.0]),
            speed_mps=np.array([50.0, 60.0, 55.0]),
        ),
    )
    progresses = [0.0, 0.4, 0.9, 1.0, 1.5, 1.99]
    telemetry = []
    for index, progress in enumerate(progresses):
        second_lap = progress >= 1.0
        telemetry.append(
            {
                "time_s": float(index),
                "progress_laps": progress,
                "lap_number": 2 if second_lap else 1,
                "x_m": float(index),
                "y_m": float(index * 2),
                "speed_mps": 40.0 + index,
                "vehicle_edge_clearance_m": 0.5 - index * 0.01,
                "grip_mu": 1.65 if second_lap else 1.70,
                "tyre": "hard" if second_lap else "soft",
                "tyre_age_laps": progress - 1.0 if second_lap else progress,
                "weather": "damp" if second_lap else "dry",
                "track_temp_c": 22.0 if second_lap else 32.0,
                "rain_intensity": 0.2 if second_lap else 0.0,
                "pit_stop": index == 2,
            }
        )
    summary = LapSummary(
        completed=True,
        lap_time_s=125.0,
        simulated_time_s=125.0,
        distance_m=11_780.0,
        mean_speed_mps=52.0,
        max_speed_mps=72.0,
        max_abs_lateral_error_m=0.2,
        minimum_vehicle_edge_clearance_m=0.35,
        minimum_boundary_margin_slack_m=0.1,
        off_track_events=0,
        safety_interventions=3,
        steps=len(telemetry),
        termination_reason="lap_complete",
        requested_laps=2,
        completed_laps=2,
        lap_times_s=(62.0, 63.0),
        pit_stops=1,
        pit_stop_time_s=25.0,
    )
    race_plan = parse_race_plan(
        {
            "laps": 2,
            "stints": [
                {"start_lap": 1, "tyre": "soft"},
                {"start_lap": 2, "tyre": "hard"},
            ],
            "weather": [
                {
                    "start_lap": 1,
                    "condition": "dry",
                    "track_temp_c": 32.0,
                    "rain_intensity": 0.0,
                },
                {
                    "start_lap": 2,
                    "condition": "damp",
                    "track_temp_c": 22.0,
                    "rain_intensity": 0.2,
                },
            ],
        }
    )

    payload = simulation_payload(
        build,
        SimulationResult(summary, telemetry),
        requested_laps=2,
        maximum_telemetry_points=100,
        race_plan=race_plan,
    )

    assert payload["track"]["name"] == "Silverstone Circuit"
    assert payload["track"]["x_m"] == [0.0, 10.0, 20.0]
    assert payload["trajectory"]["speed_kph"] == [180.0, 216.0, 198.0]
    assert payload["telemetry"]["lap_number"] == [1, 1, 1, 2, 2, 2]
    assert payload["telemetry"]["clearance_m"] == pytest.approx(
        [0.5, 0.49, 0.48, 0.47, 0.46, 0.45]
    )
    assert payload["telemetry"]["tyre"] == [
        "soft",
        "soft",
        "soft",
        "hard",
        "hard",
        "hard",
    ]
    assert payload["telemetry"]["weather"] == [
        "dry",
        "dry",
        "dry",
        "damp",
        "damp",
        "damp",
    ]
    assert payload["telemetry"]["pit_stop"] == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert payload["summary"] == {
        "completed": True,
        "requested_laps": 2,
        "completed_laps": 2,
        "lap_times_s": [62.0, 63.0],
        "total_time_s": 125.0,
        "fastest_lap_s": 62.0,
        "average_lap_s": 62.5,
        "off_track_events": 0,
        "max_speed_kph": 259.2,
        "mean_speed_kph": 187.20000000000002,
        "minimum_vehicle_edge_clearance_m": 0.35,
        "pit_stops": 1,
        "pit_stop_time_s": 25.0,
        "termination_reason": "lap_complete",
    }
    assert payload["strategy"]["stints"] == [
        {"start_lap": 1, "tyre": "soft"},
        {"start_lap": 2, "tyre": "hard"},
    ]
    assert payload["strategy"]["pit_events"] == [
        {
            "before_lap": 2,
            "from_tyre": "soft",
            "to_tyre": "hard",
            "loss_s": 25.0,
        }
    ]
    assert [item["weather"] for item in payload["strategy"]["lap_conditions"]] == [
        "dry",
        "damp",
    ]


@pytest.mark.parametrize(
    ("name", "marker"),
    [
        ("index.html", b"<!doctype html>"),
        ("app.js", b"/api/simulate"),
        ("styles.css", b"body"),
    ],
)
def test_packaged_dashboard_assets_are_readable(name: str, marker: bytes) -> None:
    assert marker.lower() in _static_file(name).lower()


class _FakeWebService:
    def __init__(self) -> None:
        self.build = SimpleNamespace(track=SimpleNamespace(name="Silverstone Circuit"))
        self.requests: list[dict[str, object]] = []
        self.busy = False

    def simulate(self, request: object) -> dict[str, object]:
        if self.busy:
            raise SimulationBusyError("another simulation is already running")
        if not isinstance(request, Mapping):
            raise ValueError("request must be an object")
        plan = parse_race_plan(request)
        self.requests.append(plan.to_request_dict())
        return {
            "track": {"name": "Silverstone Circuit"},
            "summary": {
                "requested_laps": plan.laps,
                "completed_laps": plan.laps,
            },
            "strategy": plan.to_dict(),
        }


def _read_json(request: str | Request) -> tuple[int, dict[str, object], object]:
    try:
        response = urlopen(request, timeout=2.0)  # noqa: S310 - loopback test server
    except HTTPError as exc:
        return exc.code, json.loads(exc.read()), exc.headers
    with response:
        return response.status, json.loads(response.read()), response.headers


def test_dashboard_server_serves_config_static_ui_and_simulation_api() -> None:
    service = _FakeWebService()
    server = create_dashboard_server(service, port=0, default_laps=10)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        status, config, headers = _read_json(f"{base_url}/api/config")
        assert status == 200
        assert {
            "default_laps": config["default_laps"],
            "maximum_laps": config["maximum_laps"],
            "track_name": config["track_name"],
        } == {
            "default_laps": 10,
            "maximum_laps": 25,
            "track_name": "Silverstone Circuit",
        }
        strategy = config["strategy"]
        assert isinstance(strategy, dict)
        assert strategy["defaults"] == {
            "tyre": "medium",
            "condition": "dry",
            "track_temp_c": 30.0,
            "rain_intensity": 0.0,
        }
        assert len(strategy["tyres"]) == 5
        assert len(strategy["weather_conditions"]) == 5
        assert "default-src 'self'" in headers["Content-Security-Policy"]

        with urlopen(f"{base_url}/", timeout=2.0) as response:  # noqa: S310
            assert response.status == 200
            assert response.headers.get_content_type() == "text/html"
            assert b"Silverstone" in response.read()

        custom_strategy = {
            "laps": 3,
            "stints": [
                {"start_lap": 1, "tyre": "soft"},
                {"start_lap": 3, "tyre": "intermediate"},
            ],
            "weather": [
                {
                    "start_lap": 1,
                    "condition": "dry",
                    "track_temp_c": 31.0,
                    "rain_intensity": 0.0,
                },
                {
                    "start_lap": 3,
                    "condition": "light_rain",
                    "track_temp_c": 19.0,
                    "rain_intensity": 0.5,
                },
            ],
        }
        request = Request(
            f"{base_url}/api/simulate",
            data=json.dumps(custom_strategy).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status, result, _headers = _read_json(request)
        assert status == 200
        assert result["summary"] == {
            "requested_laps": 3,
            "completed_laps": 3,
        }
        assert result["strategy"]["pit_stop_count"] == 1
        assert service.requests == [custom_strategy]

        legacy = Request(
            f"{base_url}/api/simulate",
            data=json.dumps({"laps": 2}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status, result, _headers = _read_json(legacy)
        assert status == 200
        assert result["summary"] == {
            "requested_laps": 2,
            "completed_laps": 2,
        }
        assert service.requests[-1] == {
            "laps": 2,
            "stints": [{"start_lap": 1, "tyre": "medium"}],
            "weather": [
                {
                    "start_lap": 1,
                    "condition": "dry",
                    "track_temp_c": 30.0,
                    "rain_intensity": 0.0,
                }
            ],
        }

        invalid = Request(
            f"{base_url}/api/simulate",
            data=json.dumps(
                {
                    "laps": 3,
                    "weather": [
                        {
                            "start_lap": 1,
                            "condition": "storm",
                            "track_temp_c": 20.0,
                            "rain_intensity": 0.5,
                        }
                    ],
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status, error, _headers = _read_json(invalid)
        assert status == 400
        assert "condition must be one of" in str(error["error"])

        service.busy = True
        busy = Request(
            f"{base_url}/api/simulate",
            data=json.dumps({"laps": 2}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        status, error, _headers = _read_json(busy)
        assert status == 409
        assert error == {"error": "another simulation is already running"}
        service.busy = False

        status, error, _headers = _read_json(f"{base_url}/missing")
        assert status == 404
        assert error == {"error": "not found"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
