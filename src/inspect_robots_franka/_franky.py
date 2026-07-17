"""Load the optional franky hardware driver with actionable guidance."""

from __future__ import annotations

from typing import Any

FRANKY_INSTALL_COMMAND = "pip install franky-control"


def _load_franky() -> Any:
    """Import franky or explain its firmware-specific wheel requirement."""
    try:
        import franky
    except ModuleNotFoundError as exc:
        if exc.name != "franky" and not (exc.name or "").startswith("franky."):
            raise
        raise ModuleNotFoundError(
            "franky is the optional Franka hardware driver. Install it with "
            f"`{FRANKY_INSTALL_COMMAND}`. Franky wheels bundle a specific libfranka; "
            "check franky.readthedocs.io and the franky firmware-to-wheel table, then "
            "install the wheel matching the robot firmware.",
            name=exc.name,
        ) from exc
    return franky
