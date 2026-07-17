"""Inspect Robots adapters for Franka arms and OpenPI DROID policy servers.

The package registers embodiment ``franka`` and policy ``openpi``. Both expose
one shared absolute 8-D joint-position contract and remain inert at construction.
"""

from __future__ import annotations

from inspect_robots_franka.config import FrankaConfig, OpenpiConfig
from inspect_robots_franka.embodiment import FrankaEmbodiment
from inspect_robots_franka.operator import OperatorIO
from inspect_robots_franka.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM
from inspect_robots_franka.policy import OpenpiPolicy
from inspect_robots_franka.preflight import build, run_preflight

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("inspect-robots-franka")
except PackageNotFoundError:  # pragma: no cover - only in a non-installed source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "DIM_LABELS",
    "STATE_KEY",
    "TOTAL_DIM",
    "FrankaConfig",
    "FrankaEmbodiment",
    "OpenpiConfig",
    "OpenpiPolicy",
    "OperatorIO",
    "__version__",
    "build",
    "run_preflight",
]
