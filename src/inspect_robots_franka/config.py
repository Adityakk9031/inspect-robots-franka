"""Validated configuration and shared spaces for the Franka and openpi pair."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np
import numpy.typing as npt
from inspect_robots.spaces import (
    ActionSemantics,
    Box,
    CameraSpec,
    ObservationSpace,
    StateField,
    StateSpec,
)

from inspect_robots_franka.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM

_T = TypeVar("_T", bound="_FromKwargs")

DEFAULT_CAMERAS: tuple[str, ...] = ("exterior_cam", "wrist_cam")

_FR3_DATASHEET_LOW = (-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159)
_FR3_DATASHEET_HIGH = (2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159)
DEFAULT_JOINT_LOW: tuple[float, ...] = (*(v + 0.05 for v in _FR3_DATASHEET_LOW), 0.0)
DEFAULT_JOINT_HIGH: tuple[float, ...] = (*(v - 0.05 for v in _FR3_DATASHEET_HIGH), 1.0)
DEFAULT_HOME_POSE: tuple[float, ...] = (
    0.0,
    -0.7854,
    0.0,
    -2.3562,
    0.0,
    1.5708,
    0.7854,
    1.0,
)

ACTION_SEMANTICS = ActionSemantics(
    control_mode="joint_pos",
    rotation_repr="none",
    gripper="continuous",
    frame="base",
    dim_labels=DIM_LABELS,
)

STATE_SPEC = StateSpec(
    fields=(StateField(key=STATE_KEY, shape=(TOTAL_DIM,), unit="rad+normalized"),)
)


class _FromKwargs:
    """Build frozen dataclasses from flat CLI-friendly keyword arguments."""

    _FLOAT_TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def from_kwargs(cls: type[_T], **flat: Any) -> _T:
        """Reject unknown keys and parse configured string values into expected types."""
        fields_map = {field.name: field for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
        unknown = set(flat) - set(fields_map)
        if unknown:
            raise TypeError(f"{cls.__name__} got unexpected config keys: {sorted(unknown)}")

        parsed: dict[str, Any] = dict(flat)
        for key in cls._FLOAT_TUPLE_FIELDS & set(parsed):
            value = parsed[key]
            if isinstance(value, str):
                try:
                    parsed[key] = tuple(float(part) for part in value.split(","))
                except ValueError:
                    raise ValueError(
                        f"{key} must be a comma-separated list of numbers, got {value!r}"
                    ) from None

        for field_name, field in fields_map.items():
            if field_name not in parsed or field_name in cls._FLOAT_TUPLE_FIELDS:
                continue
            val = parsed[field_name]
            if not isinstance(val, str):
                continue

            target_type = field.type
            if target_type is int or target_type == "int":
                try:
                    parsed[field_name] = int(val)
                except ValueError:
                    raise ValueError(f"{field_name} must be an integer, got {val!r}") from None
            elif target_type is float or target_type == "float":
                try:
                    parsed[field_name] = float(val)
                except ValueError:
                    raise ValueError(f"{field_name} must be a float, got {val!r}") from None
            elif target_type is bool or target_type == "bool":
                lowered = val.strip().lower()
                if lowered in ("true", "1", "yes"):
                    parsed[field_name] = True
                elif lowered in ("false", "0", "no"):
                    parsed[field_name] = False
                else:
                    raise ValueError(
                        f"{field_name} must be a boolean ('true'/'false'), got {val!r}"
                    )

        return cls(**parsed)



@dataclass(frozen=True)
class FrankaConfig(_FromKwargs):
    """Static hardware, safety, pacing, and camera configuration."""

    _FLOAT_TUPLE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"joint_low", "joint_high", "home_pose", "rest_pose"}
    )

    hostname: str | None = None
    control_hz: float = 15.0
    joint_low: tuple[float, ...] = DEFAULT_JOINT_LOW
    joint_high: tuple[float, ...] = DEFAULT_JOINT_HIGH
    home_pose: tuple[float, ...] = DEFAULT_HOME_POSE
    rest_pose: tuple[float, ...] | None = None
    relative_dynamics_factor: float = 0.15
    gripper_max_width: float = 0.08
    gripper_speed: float = 0.05
    gripper_deadband: float = 0.1
    unattended: bool = False
    exterior_cam_device: str | int | None = None
    wrist_cam_device: str | int | None = None
    cam_height: int = 480
    cam_width: int = 640
    docs_extra: str = ""

    def __post_init__(self) -> None:
        """Reject configurations that violate the fixed 8-D safety contract."""
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and > 0")
        for name in ("joint_low", "joint_high", "home_pose"):
            if len(getattr(self, name)) != TOTAL_DIM:
                raise ValueError(f"{name} must have {TOTAL_DIM} entries")
        if self.rest_pose is not None and len(self.rest_pose) != TOTAL_DIM:
            raise ValueError(f"rest_pose must have {TOTAL_DIM} entries")
        low = self.low
        high = self.high
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("joint_low and joint_high must contain only finite values")
        if np.any(low >= high):
            raise ValueError("joint_low must be below joint_high in every dimension")
        for name in ("home_pose", "rest_pose"):
            pose = getattr(self, name)
            if pose is not None:
                values = np.asarray(pose, dtype=np.float64)
                if not np.all(np.isfinite(values)) or np.any(values < low) or np.any(values > high):
                    raise ValueError(f"{name} must be finite and inside joint_low/joint_high")
        if not np.isfinite(self.gripper_max_width) or self.gripper_max_width <= 0:
            raise ValueError("gripper_max_width must be finite and > 0")
        if not np.isfinite(self.gripper_speed) or self.gripper_speed <= 0:
            raise ValueError("gripper_speed must be finite and > 0")
        if (
            not np.isfinite(self.relative_dynamics_factor)
            or self.relative_dynamics_factor <= 0
            or self.relative_dynamics_factor > 1
        ):
            raise ValueError("relative_dynamics_factor must be finite and in (0, 1]")
        if (
            not np.isfinite(self.gripper_deadband)
            or self.gripper_deadband < 0
            or self.gripper_deadband >= 1
        ):
            raise ValueError("gripper_deadband must be finite and in [0, 1)")
        if self.cam_height < 1 or self.cam_width < 1:
            raise ValueError("cam_height and cam_width must be positive")

    @property
    def low(self) -> npt.NDArray[np.float64]:
        """Return configured lower action bounds as float64."""
        return np.asarray(self.joint_low, dtype=np.float64)

    @property
    def high(self) -> npt.NDArray[np.float64]:
        """Return configured upper action bounds as float64."""
        return np.asarray(self.joint_high, dtype=np.float64)


@dataclass(frozen=True)
class OpenpiConfig(_FromKwargs):
    """Static transport and DROID action-adaptation configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    actions_are_velocity: bool = True
    velocity_action_scale: float = 0.2
    action_horizon: int = 15
    replan_interval: int = 8
    name: str = "openpi"
    resize_px: int = 224

    def __post_init__(self) -> None:
        """Reject invalid transport and chunk metadata before network use."""
        if not self.host:
            raise ValueError("host must not be empty")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port must be an integer in [1, 65535]")
        if not np.isfinite(self.velocity_action_scale) or self.velocity_action_scale <= 0:
            raise ValueError("velocity_action_scale must be finite and > 0")
        for name in ("action_horizon", "replan_interval", "resize_px"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def camera_specs(height: int, width: int) -> tuple[CameraSpec, ...]:
    """Build the two camera declarations at a shared resolution."""
    return tuple(
        CameraSpec(name=name, height=height, width=width, channels=3) for name in DEFAULT_CAMERAS
    )


def action_box(cfg: FrankaConfig | None = None) -> Box:
    """Build the shared absolute joint-position space, optionally with rig bounds."""
    return Box(
        shape=(TOTAL_DIM,),
        low=cfg.low if cfg is not None else None,
        high=cfg.high if cfg is not None else None,
        semantics=ACTION_SEMANTICS,
    )


def observation_space(cfg: FrankaConfig | None = None) -> ObservationSpace:
    """Build the two-camera and packed-proprioception observation contract."""
    height = cfg.cam_height if cfg is not None else 480
    width = cfg.cam_width if cfg is not None else 640
    return ObservationSpace(cameras=camera_specs(height, width), state=STATE_SPEC)
