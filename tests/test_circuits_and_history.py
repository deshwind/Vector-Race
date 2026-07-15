from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import racing_line.webapp as webapp_module
from racing_line.circuits import (
    DEFAULT_CIRCUIT_ID,
    F1_CIRCUIT_SEASON,
    circuit_info,
    f1_circuit_catalogue,
    make_f1_circuit,
)
from racing_line.config import AppConfig, make_f1_catalog_config
from racing_line.pipeline import build_trajectory
from racing_line.simulation import LapSummary, SimulationResult
from racing_line.strategy import parse_race_plan
from racing_line.web_history import SimulationHistoryStore
from racing_line.webapp import F1WebService


EXPECTED_2026_CIRCUIT_IDS = (
    "au-1953",
    "cn-2004",
    "jp-1962",
    "bh-2002",
    "sa-2021",
    "us-2022",
    "ca-1978",
    "mc-1929",
    "es-1991",
    "at-1969",
    "gb-1948",
    "be-1925",
    "hu-1986",
    "nl-1948",
    "it-1922",
    "es-2026",
    "az-2016",
    "sg-2008",
    "us-2012",
    "mx-1962",
    "br-1940",
    "us-2023",
    "qa-2004",
    "ae-2009",
)


def _fast_catalog_config() -> AppConfig:
    config = make_f1_catalog_config()
    return replace(
        config,
        optimizer=replace(
            config.optimizer,
            points=96,
            max_iterations=120,
        ),
    )


def test_f1_catalogue_is_unique_ordered_and_defaults_to_silverstone() -> None:
    catalogue = f1_circuit_catalogue()

    assert isinstance(catalogue, tuple)
    assert tuple(item.id for item in catalogue) == EXPECTED_2026_CIRCUIT_IDS
    assert len({item.id for item in catalogue}) == 24
    assert len({item.name for item in catalogue}) == 24
    assert F1_CIRCUIT_SEASON == 2026
    assert DEFAULT_CIRCUIT_ID == "gb-1948"
    assert circuit_info(DEFAULT_CIRCUIT_ID).name == "Silverstone Circuit"


@pytest.mark.parametrize("circuit_id", EXPECTED_2026_CIRCUIT_IDS)
def test_every_f1_circuit_loads_valid_geometry(circuit_id: str) -> None:
    info = circuit_info(circuit_id)
    track = make_f1_circuit(circuit_id)

    assert track.name == info.name
    assert track.point_count >= 8
    for values in (
        track.x_m,
        track.y_m,
        track.width_left_m,
        track.width_right_m,
    ):
        assert np.all(np.isfinite(values))
    assert np.all(track.width_left_m > 0.0)
    assert np.all(track.width_right_m > 0.0)
    assert track.length_m == pytest.approx(info.length_m, rel=1.0e-6)


@pytest.mark.parametrize(
    "circuit_id",
    [
        "not-a-circuit",
        "../gb-1948",
        "..\\gb-1948",
        "../../etc/passwd",
        "/gb-1948",
    ],
)
def test_circuit_info_rejects_unknown_and_traversal_ids(circuit_id: str) -> None:
    with pytest.raises(ValueError, match="unknown circuit_id"):
        circuit_info(circuit_id)


@pytest.mark.parametrize("circuit_id", ["gb-1948", "mc-1929", "es-2026"])
def test_catalog_config_builds_representative_trajectories(
    circuit_id: str,
) -> None:
    result = build_trajectory(make_f1_circuit(circuit_id), make_f1_catalog_config())

    assert result.success
    assert result.track.name == circuit_info(circuit_id).name
    assert np.all(np.isfinite(result.trajectory.speed_mps))
    assert np.all(result.trajectory.speed_mps > 0.0)


def test_simulation_history_persists_newest_first_honours_limit_and_clears(
    tmp_path,
) -> None:
    history_path = tmp_path / "nested" / "history.json"
    store = SimulationHistoryStore(history_path, limit=2)
    circuit = circuit_info(DEFAULT_CIRCUIT_ID)
    plan = parse_race_plan({"laps": 1})

    first = store.append(
        circuit=circuit,
        race_plan=plan,
        payload={"summary": {"total_time_s": 61.0}},
    )
    second = store.append(
        circuit=circuit,
        race_plan=plan,
        payload={"summary": {"total_time_s": 60.0}},
    )
    third = store.append(
        circuit=circuit,
        race_plan=plan,
        payload={
            "track": circuit.to_dict(),
            "summary": {"total_time_s": 59.0},
            "telemetry": [{"time_s": 0.0, "speed_mps": 42.0}],
        },
    )

    assert history_path.is_file()
    assert [item["id"] for item in store.list()] == [third["id"], second["id"]]
    assert all(item["details_available"] for item in store.list())
    assert all("telemetry" not in item for item in store.list())
    assert first["id"] not in {item["id"] for item in store.list()}
    assert store.get(first["id"]) is None
    assert not (store.details_path / f"{first['id']}.json.gz").exists()
    assert len(list(store.details_path.glob("*.json.gz"))) == 2

    details = store.get(third["id"])
    assert details is not None
    assert details["telemetry"] == [{"time_s": 0.0, "speed_mps": 42.0}]
    assert details["history"]["id"] == third["id"]
    assert details["details_available"] is True

    # Callers receive independent values; a viewed result cannot mutate history.
    details["telemetry"][0]["speed_mps"] = -1.0
    assert store.get(third["id"])["telemetry"][0]["speed_mps"] == 42.0

    reloaded = SimulationHistoryStore(history_path, limit=2)
    assert [item["id"] for item in reloaded.list()] == [third["id"], second["id"]]
    assert reloaded.clear() == 2
    assert reloaded.list() == []
    assert not history_path.exists()
    assert not reloaded.details_path.exists()
    assert reloaded.clear() == 0


def test_simulation_history_loads_legacy_summary_and_rejects_unsafe_ids(
    tmp_path,
) -> None:
    history_path = tmp_path / "history.json"
    legacy_id = "a" * 32
    history_path.write_text(
        """[
  {
    "id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "created_at": "2026-01-01T00:00:00+00:00",
    "circuit": {"id": "gb-1948", "name": "Silverstone Circuit"},
    "summary": {"total_time_s": 60.0},
    "strategy": {"laps": 1}
  }
]""",
        encoding="utf-8",
    )
    store = SimulationHistoryStore(history_path)

    assert store.list()[0]["details_available"] is False
    details = store.get(legacy_id)
    assert details == {
        "track": {"id": "gb-1948", "name": "Silverstone Circuit"},
        "summary": {"total_time_s": 60.0},
        "strategy": {"laps": 1},
        "history": {
            "id": legacy_id,
            "created_at": "2026-01-01T00:00:00+00:00",
            "circuit": {"id": "gb-1948", "name": "Silverstone Circuit"},
            "summary": {"total_time_s": 60.0},
            "strategy": {"laps": 1},
            "details_available": False,
        },
        "details_available": False,
    }

    for invalid_id in ("", "not-an-id", "../" + legacy_id, legacy_id + "/x"):
        with pytest.raises(ValueError, match="invalid simulation history id"):
            store.get(invalid_id)


class _FastLapSimulator:
    def __init__(self, trajectory, config) -> None:
        self.trajectory = trajectory

    def run(
        self,
        *,
        lap_count: int,
        condition_profile,
        pit_stop_losses_s,
    ) -> SimulationResult:
        summary = LapSummary(
            completed=True,
            lap_time_s=58.0,
            simulated_time_s=58.0,
            distance_m=float(self.trajectory.length_m),
            mean_speed_mps=45.0,
            max_speed_mps=70.0,
            max_abs_lateral_error_m=0.1,
            minimum_vehicle_edge_clearance_m=1.0,
            minimum_boundary_margin_slack_m=0.5,
            off_track_events=0,
            safety_interventions=0,
            steps=1,
            termination_reason="lap_complete",
            requested_laps=lap_count,
            completed_laps=lap_count,
            lap_times_s=tuple(58.0 for _ in range(lap_count)),
        )
        telemetry = [
            {
                "time_s": 0.0,
                "progress_laps": 0.0,
                "lap_number": 1,
                "x_m": float(self.trajectory.x_m[0]),
                "y_m": float(self.trajectory.y_m[0]),
                "speed_mps": 45.0,
            }
        ]
        return SimulationResult(summary, telemetry)


def test_f1_web_service_selects_circuit_and_persists_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(webapp_module, "LapSimulator", _FastLapSimulator)
    history_path = tmp_path / "web-history.json"
    config = _fast_catalog_config()
    service = F1WebService(config=config, history_path=history_path)

    config_payload = service.config_payload(default_laps=3)
    assert config_payload["default_circuit_id"] == DEFAULT_CIRCUIT_ID
    assert config_payload["circuit_season"] == 2026
    assert [item["id"] for item in config_payload["circuits"]] == list(
        EXPECTED_2026_CIRCUIT_IDS
    )

    result = service.simulate({"circuit_id": "es-2026", "laps": 1})

    assert result["track"]["id"] == "es-2026"
    assert result["track"]["name"] == "Circuito de Madring"
    assert result["history"]["circuit"]["id"] == "es-2026"
    assert result["history"]["summary"] == result["summary"]
    assert [item["id"] for item in service.history()] == [result["history"]["id"]]

    restarted = F1WebService(config=config, history_path=history_path)
    assert [item["id"] for item in restarted.history()] == [result["history"]["id"]]
    assert restarted.history()[0]["circuit"]["id"] == "es-2026"
    saved_result = restarted.history_detail(result["history"]["id"])
    assert saved_result is not None
    assert saved_result["track"]["id"] == "es-2026"
    assert saved_result["telemetry"]["speed_kph"] == [162.0]
    assert saved_result["details_available"] is True
