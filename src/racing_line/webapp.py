"""Dependency-free local dashboard for Formula 1 circuit simulations."""

from __future__ import annotations

import copy
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from math import ceil, floor
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .circuits import (
    DEFAULT_CIRCUIT_ID,
    F1_CIRCUIT_SEASON,
    CircuitInfo,
    circuit_info,
    f1_circuit_catalogue,
    make_f1_circuit,
)
from .config import AppConfig, make_f1_catalog_config
from .pipeline import BuildResult, build_trajectory
from .simulation import LapSimulator, SimulationResult
from .strategy import RacePlan, parse_race_plan, strategy_catalog
from .tyre_analysis import analyse_tyre_performance
from .web_history import DEFAULT_HISTORY_LIMIT, SimulationHistoryStore


DEFAULT_WEB_PORT = 8765
DEFAULT_WEB_LAPS = 10
MAX_WEB_LAPS = 25
MAX_REQUEST_BYTES = 16_384
MAX_TELEMETRY_POINTS = 6000
DEFAULT_HISTORY_PATH = Path("outputs") / "simulation_history.json"
_STATIC_TYPES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
}


class SimulationBusyError(RuntimeError):
    """Raised when another browser request is already simulating."""


def validate_web_laps(value: Any) -> int:
    """Return a safe browser lap count or raise a user-facing error."""

    if isinstance(value, bool):
        raise ValueError("laps must be an integer")
    try:
        laps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("laps must be an integer") from exc
    if laps != value and not (isinstance(value, str) and str(laps) == value.strip()):
        raise ValueError("laps must be an integer")
    if not 1 <= laps <= MAX_WEB_LAPS:
        raise ValueError(f"laps must be between 1 and {MAX_WEB_LAPS}")
    return laps


def _float_list(values: Any) -> list[float]:
    return [float(value) for value in values]


def _telemetry_rows(
    rows: list[dict[str, float | int | str | bool]],
    maximum_points: int = MAX_TELEMETRY_POINTS,
) -> list[dict[str, float | int | str | bool]]:
    """Decimate telemetry while retaining endpoints and lap transitions."""

    if len(rows) <= maximum_points:
        return rows
    stride = max(1, ceil(len(rows) / maximum_points))
    indices = set(range(0, len(rows), stride))
    indices.add(len(rows) - 1)
    previous_lap = -1
    previous_signature: tuple[int, str, str] | None = None
    for index, row in enumerate(rows):
        lap = max(0, int(floor(float(row.get("progress_laps", 0.0)))))
        if lap != previous_lap:
            indices.add(index)
            if index:
                indices.add(index - 1)
            previous_lap = lap
        signature = (
            int(row.get("lap_number", lap + 1)),
            str(row.get("tyre", "")),
            str(row.get("weather", "")),
        )
        if signature != previous_signature or bool(row.get("pit_stop", False)):
            indices.add(index)
            if index:
                indices.add(index - 1)
            previous_signature = signature
    return [rows[index] for index in sorted(indices)]


def simulation_payload(
    build: BuildResult,
    simulation: SimulationResult,
    requested_laps: int,
    *,
    maximum_telemetry_points: int = MAX_TELEMETRY_POINTS,
    race_plan: RacePlan | None = None,
) -> dict[str, Any]:
    """Convert numerical results into the stable browser API schema."""

    rows = _telemetry_rows(simulation.telemetry, maximum_telemetry_points)
    summary = simulation.summary
    raw_lap_times = getattr(summary, "lap_times_s", ())
    lap_times = [float(value) for value in raw_lap_times]
    if not lap_times and summary.completed and summary.lap_time_s is not None:
        lap_times = [float(summary.lap_time_s)]
    completed_laps = int(getattr(summary, "completed_laps", len(lap_times)))

    track = build.track
    trajectory = build.trajectory
    telemetry_laps = [
        min(
            requested_laps,
            max(
                1,
                int(
                    row.get(
                        "lap_number",
                        int(floor(float(row["progress_laps"]))) + 1,
                    )
                ),
            ),
        )
        for row in rows
    ]
    strategy = race_plan.to_dict() if race_plan is not None else None
    if race_plan is not None:
        strategy["lap_conditions"] = [
            {
                "lap": lap_number,
                **race_plan.condition_at(lap_number, 0.5).to_dict(),
            }
            for lap_number in range(1, race_plan.laps + 1)
        ]

    payload: dict[str, Any] = {
        "track": {
            "name": track.name,
            "x_m": _float_list(track.x_m),
            "y_m": _float_list(track.y_m),
            "width_left_m": _float_list(track.width_left_m),
            "width_right_m": _float_list(track.width_right_m),
        },
        "trajectory": {
            "s_m": _float_list(trajectory.s_m),
            "x_m": _float_list(trajectory.x_m),
            "y_m": _float_list(trajectory.y_m),
            "speed_kph": _float_list(trajectory.speed_mps * 3.6),
        },
        "telemetry": {
            "x_m": [float(row["x_m"]) for row in rows],
            "y_m": [float(row["y_m"]) for row in rows],
            "time_s": [float(row["time_s"]) for row in rows],
            "progress_laps": [float(row["progress_laps"]) for row in rows],
            "speed_kph": [float(row["speed_mps"]) * 3.6 for row in rows],
            "target_speed_kph": [
                float(
                    row.get(
                        "target_speed_mps",
                        row.get("reference_speed_mps", row["speed_mps"]),
                    )
                )
                * 3.6
                for row in rows
            ],
            "lap_number": telemetry_laps,
            "clearance_m": [
                float(row.get("vehicle_edge_clearance_m", 0.0)) for row in rows
            ],
            "grip_mu": [float(row.get("grip_mu", 0.0)) for row in rows],
            "tyre": [str(row.get("tyre", "")) for row in rows],
            "tyre_age_laps": [float(row.get("tyre_age_laps", 0.0)) for row in rows],
            "weather": [str(row.get("weather", "")) for row in rows],
            "track_temp_c": [float(row.get("track_temp_c", 0.0)) for row in rows],
            "rain_intensity": [float(row.get("rain_intensity", 0.0)) for row in rows],
            "track_wetness": [float(row.get("track_wetness", 0.0)) for row in rows],
            "pit_stop": [bool(row.get("pit_stop", False)) for row in rows],
        },
        "summary": {
            "completed": bool(summary.completed),
            "requested_laps": requested_laps,
            "completed_laps": completed_laps,
            "lap_times_s": lap_times,
            "total_time_s": float(summary.simulated_time_s),
            "fastest_lap_s": min(lap_times) if lap_times else None,
            "average_lap_s": (sum(lap_times) / len(lap_times) if lap_times else None),
            "off_track_events": int(summary.off_track_events),
            "max_speed_kph": float(summary.max_speed_mps) * 3.6,
            "mean_speed_kph": float(summary.mean_speed_mps) * 3.6,
            "minimum_vehicle_edge_clearance_m": float(
                summary.minimum_vehicle_edge_clearance_m
            ),
            "pit_stops": int(getattr(summary, "pit_stops", 0)),
            "pit_stop_time_s": float(getattr(summary, "pit_stop_time_s", 0.0)),
            "termination_reason": summary.termination_reason,
        },
    }
    if strategy is not None:
        payload["strategy"] = strategy
        payload["tyre_analysis"] = analyse_tyre_performance(
            payload,
            circuit_length_m=getattr(trajectory, "length_m", None),
        )
    return payload


class F1WebService:
    """Build circuits lazily, cache numerical runs, and persist run history."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        maximum_telemetry_points: int = MAX_TELEMETRY_POINTS,
        default_circuit_id: str = DEFAULT_CIRCUIT_ID,
        history_path: str | Path = DEFAULT_HISTORY_PATH,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self.config = config or make_f1_catalog_config()
        self.maximum_telemetry_points = maximum_telemetry_points
        self.default_circuit_id = circuit_info(default_circuit_id).id
        self.history_store = SimulationHistoryStore(
            history_path,
            limit=history_limit,
        )
        self._simulation_lock = threading.Lock()
        self._builds: dict[str, BuildResult] = {}
        self._cache: dict[str, dict[str, Any]] = {}

        # Preserve the historic ``build`` attribute for API consumers while
        # making every other circuit an on-demand cost.
        self.build = self._build_circuit(self.default_circuit_id)

    @property
    def circuits(self) -> tuple[CircuitInfo, ...]:
        return f1_circuit_catalogue()

    def _build_circuit(self, circuit_id: str) -> BuildResult:
        circuit = circuit_info(circuit_id)
        cached = self._builds.get(circuit.id)
        if cached is not None:
            return cached
        build = build_trajectory(make_f1_circuit(circuit.id), self.config)
        if not build.success:
            raise RuntimeError(
                f"trajectory generation did not converge for {circuit.name}; "
                "inspect the optimizer configuration"
            )
        self._builds[circuit.id] = build
        return build

    def config_payload(self, default_laps: int) -> dict[str, Any]:
        default = circuit_info(self.default_circuit_id)
        return {
            "default_laps": default_laps,
            "maximum_laps": MAX_WEB_LAPS,
            "default_circuit_id": default.id,
            "track_name": default.name,
            "circuit_season": F1_CIRCUIT_SEASON,
            "circuits": [item.to_dict() for item in self.circuits],
            "strategy": strategy_catalog(),
        }

    def history(self) -> list[dict[str, Any]]:
        return self.history_store.list()

    def history_detail(self, run_id: str) -> dict[str, Any] | None:
        return self.history_store.get(run_id)

    def clear_history(self) -> int:
        return self.history_store.clear()

    def _request_plan(self, request: Any) -> tuple[CircuitInfo, RacePlan]:
        if not isinstance(request, Mapping):
            return (
                circuit_info(self.default_circuit_id),
                parse_race_plan(
                    {"laps": validate_web_laps(request)},
                    max_laps=MAX_WEB_LAPS,
                ),
            )

        values = dict(request)
        raw_circuit_id = values.pop(
            "circuit_id",
            values.pop("circuit", self.default_circuit_id),
        )
        circuit = circuit_info(raw_circuit_id)
        plan = parse_race_plan(values, max_laps=MAX_WEB_LAPS)
        return circuit, plan

    def _record_payload(
        self,
        payload: Mapping[str, Any],
        circuit: CircuitInfo,
        plan: RacePlan,
    ) -> dict[str, Any]:
        result = copy.deepcopy(dict(payload))
        track = result.get("track")
        if isinstance(track, dict):
            track.update(circuit.to_dict())
        record = self.history_store.append(
            circuit=circuit,
            race_plan=plan,
            payload=result,
        )
        result["history"] = record
        return result

    def simulate(self, request: Any) -> dict[str, Any]:
        circuit, plan = self._request_plan(request)
        cache_key = json.dumps(
            {
                "circuit_id": circuit.id,
                **plan.to_request_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return self._record_payload(cached, circuit, plan)
        if not self._simulation_lock.acquire(blocking=False):
            raise SimulationBusyError("another simulation is already running")
        try:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return self._record_payload(cached, circuit, plan)
            build = self._build_circuit(circuit.id)
            pit_losses = {
                stint.start_lap - 1: plan.pit_stop_loss_s for stint in plan.stints[1:]
            }
            simulation = LapSimulator(build.trajectory, self.config).run(
                lap_count=plan.laps,
                condition_profile=lambda lap, progress: plan.condition_at(
                    lap,
                    progress,
                    self.config.vehicle.base_grip_mu,
                ),
                pit_stop_losses_s=pit_losses,
            )
            payload = simulation_payload(
                build,
                simulation,
                plan.laps,
                maximum_telemetry_points=self.maximum_telemetry_points,
                race_plan=plan,
            )
            track = payload.get("track")
            if isinstance(track, dict):
                track.update(circuit.to_dict())
            self._cache[cache_key] = payload
            # Bound memory without evicting the build cache. Each full browser
            # payload contains telemetry, so retaining the most recent eight is
            # sufficient for quick scenario comparisons.
            while len(self._cache) > 8:
                self._cache.pop(next(iter(self._cache)))
            return self._record_payload(payload, circuit, plan)
        finally:
            self._simulation_lock.release()


# Backward-compatible public name retained for callers of version 0.1.
SilverstoneWebService = F1WebService


def _static_file(name: str) -> bytes:
    if name not in _STATIC_TYPES:
        raise FileNotFoundError(name)
    source = resources.files("racing_line").joinpath("web", name)
    return source.read_bytes()


def _handler_factory(
    service: Any,
    default_laps: int,
) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "RacingLineDashboard/0.1"

        def _security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )

        def _bytes_response(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
            *,
            cache_control: str = "no-store",
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self._security_headers()
            self.end_headers()
            self.wfile.write(body)

        def _json_response(
            self,
            status: HTTPStatus,
            value: Mapping[str, Any],
        ) -> None:
            body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
            self._bytes_response(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlsplit(self.path).path
            if path == "/api/config":
                config_provider = getattr(service, "config_payload", None)
                if callable(config_provider):
                    config = config_provider(default_laps)
                else:
                    track_name = service.build.track.name
                    config = {
                        "default_laps": default_laps,
                        "maximum_laps": MAX_WEB_LAPS,
                        "default_circuit_id": DEFAULT_CIRCUIT_ID,
                        "track_name": track_name,
                        "circuit_season": F1_CIRCUIT_SEASON,
                        "circuits": [
                            {
                                "id": DEFAULT_CIRCUIT_ID,
                                "name": track_name,
                                "location": "Silverstone",
                                "country_code": "GB",
                                "length_m": 5891.0,
                            }
                        ],
                        "strategy": strategy_catalog(),
                    }
                self._json_response(
                    HTTPStatus.OK,
                    config,
                )
                return
            if path == "/api/history":
                history_provider = getattr(service, "history", None)
                history = history_provider() if callable(history_provider) else []
                self._json_response(HTTPStatus.OK, {"history": history})
                return
            if path.startswith("/api/history/"):
                run_id = path.removeprefix("/api/history/")
                detail_provider = getattr(service, "history_detail", None)
                try:
                    detail = (
                        detail_provider(run_id) if callable(detail_provider) else None
                    )
                except ValueError:
                    detail = None
                if detail is None:
                    self._json_response(
                        HTTPStatus.NOT_FOUND,
                        {"error": "history run not found"},
                    )
                    return
                self._json_response(HTTPStatus.OK, detail)
                return
            if path == "/favicon.ico":
                self._bytes_response(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
                return
            name = "index.html" if path in ("/", "/index.html") else path[1:]
            try:
                body = _static_file(name)
            except FileNotFoundError:
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._bytes_response(
                HTTPStatus.OK,
                body,
                _STATIC_TYPES[name],
                cache_control="no-cache",
            )

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if urlsplit(self.path).path != "/api/simulate":
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                content_length = -1
            if not 0 < content_length <= MAX_REQUEST_BYTES:
                self._json_response(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "request body is missing or too large"},
                )
                return
            try:
                body = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
                request = {"laps": default_laps, **body}
                payload = service.simulate(request)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except SimulationBusyError as exc:
                self._json_response(HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            except (OSError, RuntimeError):
                self._json_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "simulation failed"},
                )
                return
            self._json_response(HTTPStatus.OK, payload)

        def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if urlsplit(self.path).path != "/api/history":
                self._json_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            clear = getattr(service, "clear_history", None)
            try:
                removed = int(clear()) if callable(clear) else 0
            except OSError:
                self._json_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "history could not be cleared"},
                )
                return
            self._json_response(
                HTTPStatus.OK,
                {"cleared": removed, "history": []},
            )

        def log_message(self, format: str, *args: Any) -> None:
            print(f"dashboard: {format % args}")

    return DashboardHandler


def create_dashboard_server(
    service: Any,
    *,
    port: int = DEFAULT_WEB_PORT,
    default_laps: int = DEFAULT_WEB_LAPS,
) -> ThreadingHTTPServer:
    """Create a loopback-only server; useful for both the CLI and tests."""

    laps = validate_web_laps(default_laps)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        _handler_factory(service, laps),
    )
    server.daemon_threads = True
    return server


def serve_dashboard(
    *,
    port: int = DEFAULT_WEB_PORT,
    default_laps: int = DEFAULT_WEB_LAPS,
    open_browser: bool = True,
) -> None:
    """Build the default circuit and serve the interactive local dashboard."""

    laps = validate_web_laps(default_laps)
    print("Preparing the Formula 1 circuit catalogue...")
    service = F1WebService()
    server = create_dashboard_server(service, port=port, default_laps=laps)
    actual_port = int(server.server_address[1])
    url = f"http://127.0.0.1:{actual_port}/?laps={laps}&autorun=1"
    print(f"Formula 1 racing dashboard: {url}")
    print("Press Ctrl+C to stop the local server.")
    if open_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = [
    "DEFAULT_WEB_LAPS",
    "DEFAULT_WEB_PORT",
    "F1WebService",
    "MAX_WEB_LAPS",
    "SilverstoneWebService",
    "create_dashboard_server",
    "serve_dashboard",
    "simulation_payload",
    "validate_web_laps",
]
