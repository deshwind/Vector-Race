"""Racing-line optimisation and safe residual-control prototype."""

from .circuits import (
    CircuitInfo,
    f1_circuit_catalogue,
    make_f1_circuit,
)
from .config import (
    AppConfig,
    load_config,
    make_f1_catalog_config,
    make_silverstone_config,
)
from .pipeline import BuildResult, build_trajectory
from .track import Track, make_silverstone_track
from .trajectory import RacingTrajectory

__all__ = [
    "AppConfig",
    "BuildResult",
    "CircuitInfo",
    "RacingTrajectory",
    "Track",
    "build_trajectory",
    "f1_circuit_catalogue",
    "load_config",
    "make_f1_catalog_config",
    "make_f1_circuit",
    "make_silverstone_config",
    "make_silverstone_track",
]
__version__ = "0.1.0"
