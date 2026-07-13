"""Dependency-free local web dashboard for Silverstone simulations."""

from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from math import ceil, floor
from typing import Any, Mapping
from urllib.parse import urlsplit

from .config import AppConfig, make_silverstone_config
from .pipeline import BuildResult, build_trajectory
from .simulation import LapSimulator, SimulationResult
from .strategy import RacePlan, parse_race_plan, strategy_catalog
from .track import make_silverstone_track


DEFAULT_WEB_PORT = 8765
DEFAULT_WEB_LAPS = 10
MAX_WEB_LAPS = 25
MAX_REQUEST_BYTES = 16_384
MAX_TELEMETRY_POINTS = 6000
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
            "x_m": _float_list(trajectory.x_m),
            "y_m": _float_list(trajectory.y_m),
            "speed_kph": _float_list(trajectory.speed_mps * 3.6),
        },
        "telemetry": {
            "x_m": [float(row["x_m"]) for row in rows],
            "y_m": [float(row["y_m"]) for row in rows],
            "time_s": [float(row["time_s"]) for row in rows],
            "speed_kph": [float(row["speed_mps"]) * 3.6 for row in rows],
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
    return payload


class SilverstoneWebService:
    """Cache the optimized line and serialize deterministic browser runs."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        maximum_telemetry_points: int = MAX_TELEMETRY_POINTS,
    ) -> None:
        self.config = config or make_silverstone_config()
        self.maximum_telemetry_points = maximum_telemetry_points
        self.build = build_trajectory(make_silverstone_track(), self.config)
        if not self.build.success:
            raise RuntimeError(
                "Silverstone trajectory generation did not converge; "
                "inspect the optimizer configuration"
            )
        self._simulation_lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}

    def simulate(self, request: Any) -> dict[str, Any]:
        if isinstance(request, Mapping):
            plan = parse_race_plan(request, max_laps=MAX_WEB_LAPS)
        else:
            plan = parse_race_plan(
                {"laps": validate_web_laps(request)},
                max_laps=MAX_WEB_LAPS,
            )
        cache_key = json.dumps(
            plan.to_request_dict(), sort_keys=True, separators=(",", ":")
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        if not self._simulation_lock.acquire(blocking=False):
            raise SimulationBusyError("another simulation is already running")
        try:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            pit_losses = {
                stint.start_lap - 1: plan.pit_stop_loss_s for stint in plan.stints[1:]
            }
            simulation = LapSimulator(self.build.trajectory, self.config).run(
                lap_count=plan.laps,
                condition_profile=lambda lap, progress: plan.condition_at(
                    lap,
                    progress,
                    self.config.vehicle.base_grip_mu,
                ),
                pit_stop_losses_s=pit_losses,
            )
            payload = simulation_payload(
                self.build,
                simulation,
                plan.laps,
                maximum_telemetry_points=self.maximum_telemetry_points,
                race_plan=plan,
            )
            self._cache.clear()
            self._cache[cache_key] = payload
            return payload
        finally:
            self._simulation_lock.release()


def _static_file(name: str) -> bytes:
    if name not in _STATIC_TYPES:
        raise FileNotFoundError(name)
    source = resources.files("racing_line").joinpath("web", name)
    return source.read_bytes()


def _handler_factory(
    service: SilverstoneWebService,
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
                self._json_response(
                    HTTPStatus.OK,
                    {
                        "default_laps": default_laps,
                        "maximum_laps": MAX_WEB_LAPS,
                        "track_name": service.build.track.name,
                        "strategy": strategy_catalog(),
                    },
                )
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
            except RuntimeError:
                self._json_response(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "simulation failed"},
                )
                return
            self._json_response(HTTPStatus.OK, payload)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"dashboard: {format % args}")

    return DashboardHandler


def create_dashboard_server(
    service: SilverstoneWebService,
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
    """Build Silverstone once and serve the interactive local dashboard."""

    laps = validate_web_laps(default_laps)
    print("Preparing the Silverstone racing line...")
    service = SilverstoneWebService()
    server = create_dashboard_server(service, port=port, default_laps=laps)
    actual_port = int(server.server_address[1])
    url = f"http://127.0.0.1:{actual_port}/?laps={laps}&autorun=1"
    print(f"Silverstone dashboard: {url}")
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
    "MAX_WEB_LAPS",
    "SilverstoneWebService",
    "create_dashboard_server",
    "serve_dashboard",
    "simulation_payload",
    "validate_web_laps",
]
