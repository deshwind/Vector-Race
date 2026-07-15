"""Bundled Formula 1 circuit catalogue and geographic track conversion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import resources
from math import cos, pi
from typing import Any

import numpy as np

from .track import Track


DEFAULT_CIRCUIT_ID = "gb-1948"
F1_CIRCUIT_SEASON = 2026
DEFAULT_HALF_WIDTH_M = 7.0
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class CircuitInfo:
    """Metadata for one selectable Formula 1 circuit."""

    id: str
    name: str
    location: str
    country_code: str
    length_m: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


def _read_data_json(name: str) -> Any:
    source = resources.files("racing_line").joinpath("data", "f1_circuits", name)
    return json.loads(source.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _catalogue_data() -> tuple[tuple[CircuitInfo, ...], dict[str, dict[str, Any]]]:
    locations = _read_data_json(f"f1-locations-{F1_CIRCUIT_SEASON}.json")
    collection = _read_data_json("f1-circuits.geojson")
    if not isinstance(locations, list) or not isinstance(collection, dict):
        raise RuntimeError("bundled Formula 1 circuit catalogue is malformed")

    features: dict[str, dict[str, Any]] = {}
    for feature in collection.get("features", []):
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties", {})
        circuit_id = properties.get("id") if isinstance(properties, dict) else None
        if isinstance(circuit_id, str):
            features[circuit_id] = feature

    catalogue: list[CircuitInfo] = []
    for item in locations:
        if not isinstance(item, dict):
            raise RuntimeError("bundled Formula 1 season catalogue is malformed")
        circuit_id = str(item.get("id", ""))
        feature = features.get(circuit_id)
        if feature is None:
            raise RuntimeError(f"missing circuit geometry for {circuit_id!r}")
        properties = feature.get("properties", {})
        try:
            length_m = float(properties["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid circuit length for {circuit_id!r}") from exc
        catalogue.append(
            CircuitInfo(
                id=circuit_id,
                name=str(item["name"]),
                location=str(item["location"]),
                country_code=circuit_id.split("-", 1)[0].upper(),
                length_m=length_m,
            )
        )

    if len(catalogue) != 24:
        raise RuntimeError(
            f"expected 24 Formula 1 circuits, found {len(catalogue)}"
        )
    if DEFAULT_CIRCUIT_ID not in features:
        raise RuntimeError("default Silverstone circuit geometry is missing")
    return tuple(catalogue), features


def f1_circuit_catalogue() -> tuple[CircuitInfo, ...]:
    """Return the ordered 2026 Formula 1 circuit catalogue."""

    return _catalogue_data()[0]


def circuit_info(circuit_id: str) -> CircuitInfo:
    """Return metadata for ``circuit_id`` or raise a user-facing error."""

    if not isinstance(circuit_id, str) or not circuit_id.strip():
        raise ValueError("circuit_id must be a non-empty string")
    normalized = circuit_id.strip().lower()
    for item in f1_circuit_catalogue():
        if item.id == normalized:
            return item
    raise ValueError(f"unknown circuit_id: {circuit_id}")


def _project_coordinates(coordinates: Any, expected_length_m: float) -> np.ndarray:
    try:
        lon_lat = np.asarray(coordinates, dtype=float)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("circuit geometry contains invalid coordinates") from exc
    if lon_lat.ndim != 2 or lon_lat.shape[1] < 2 or len(lon_lat) < 8:
        raise RuntimeError("circuit geometry must be a LineString with eight points")
    lon_lat = lon_lat[:, :2]
    if not np.all(np.isfinite(lon_lat)):
        raise RuntimeError("circuit geometry contains non-finite coordinates")

    lon0, lat0 = np.mean(lon_lat, axis=0)
    radians = pi / 180.0
    x_m = EARTH_RADIUS_M * cos(lat0 * radians) * (lon_lat[:, 0] - lon0) * radians
    y_m = EARTH_RADIUS_M * (lon_lat[:, 1] - lat0) * radians
    projected = np.column_stack((x_m, y_m))

    # Remove duplicate samples while preserving an explicit closure point for
    # the length calculation. Track will remove that closure point itself.
    keep = np.ones(len(projected), dtype=bool)
    keep[1:] = np.hypot(
        np.diff(projected[:, 0]), np.diff(projected[:, 1])
    ) > 1.0e-3
    projected = projected[keep]
    if len(projected) < 8:
        raise RuntimeError("circuit geometry has too few distinct points")

    is_closed = np.hypot(*(projected[-1] - projected[0])) < 1.0
    length_points = projected if is_closed else np.vstack((projected, projected[0]))
    observed_length = float(
        np.sum(np.hypot(np.diff(length_points[:, 0]), np.diff(length_points[:, 1])))
    )
    if observed_length <= 0.0:
        raise RuntimeError("circuit geometry has zero length")

    # Geographic centreline files are intended for display rather than survey
    # work. Scaling to their supplied circuit length keeps simulator distance
    # and timing comparisons consistent across the catalogue.
    return projected * (expected_length_m / observed_length)


@lru_cache(maxsize=32)
def make_f1_circuit(
    circuit_id: str = DEFAULT_CIRCUIT_ID,
    *,
    half_width_m: float = DEFAULT_HALF_WIDTH_M,
) -> Track:
    """Build a local-metre :class:`Track` for a selectable F1 circuit.

    The source catalogue supplies geographic centrelines and nominal lengths,
    but not surveyed boundary widths. A documented, constant research width is
    therefore used on both sides of the centreline.
    """

    info = circuit_info(circuit_id)
    if isinstance(half_width_m, bool) or not np.isfinite(half_width_m):
        raise ValueError("half_width_m must be a finite number")
    if half_width_m <= 1.0:
        raise ValueError("half_width_m must be greater than one metre")

    feature = _catalogue_data()[1][info.id]
    geometry = feature.get("geometry", {})
    if not isinstance(geometry, dict) or geometry.get("type") != "LineString":
        raise RuntimeError(f"unsupported circuit geometry for {info.id}")
    points = _project_coordinates(geometry.get("coordinates"), info.length_m)
    width = np.full(len(points), float(half_width_m), dtype=float)
    return Track(
        x_m=points[:, 0],
        y_m=points[:, 1],
        width_left_m=width,
        width_right_m=width,
        name=info.name,
    )


__all__ = [
    "CircuitInfo",
    "DEFAULT_CIRCUIT_ID",
    "DEFAULT_HALF_WIDTH_M",
    "F1_CIRCUIT_SEASON",
    "circuit_info",
    "f1_circuit_catalogue",
    "make_f1_circuit",
]
