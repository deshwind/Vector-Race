"""Racing-line optimisation and safe residual-control prototype."""

from .config import AppConfig, load_config, make_silverstone_config
from .pipeline import BuildResult, build_trajectory
from .track import Track, make_silverstone_track
from .trajectory import RacingTrajectory

__all__ = [
    "AppConfig",
    "BuildResult",
    "RacingTrajectory",
    "Track",
    "build_trajectory",
    "load_config",
    "make_silverstone_config",
    "make_silverstone_track",
]
__version__ = "0.1.0"
