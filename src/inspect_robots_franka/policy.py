"""OpenPI websocket policy adapter for the pi05-DROID Franka convention.

The transport stays lazy and injectable. The adapter converts DROID gripper
polarity and integrates normalized velocity chunks into absolute joint targets,
so its declared ``joint_pos`` action mode remains honest.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_franka import packing
from inspect_robots_franka.config import OpenpiConfig, action_box, observation_space

OPENPI_CLIENT_INSTALL_COMMAND = (
    'pip install "openpi-client @ '
    "git+https://github.com/Physical-Intelligence/openpi.git"
    '#subdirectory=packages/openpi-client"'
)

OpenpiObservation = Mapping[str, Any]
OpenpiResponse = Mapping[str, Any]
InferFn = Callable[[OpenpiObservation], OpenpiResponse]


def _default_infer(cfg: OpenpiConfig) -> InferFn:  # pragma: no cover - live network transport
    """Build the upstream websocket client and its resize-with-pad transport."""
    try:
        from openpi_client import image_tools, websocket_client_policy
    except ModuleNotFoundError as exc:
        if exc.name != "openpi_client" and not (exc.name or "").startswith("openpi_client."):
            raise
        raise ModuleNotFoundError(
            "The Physical Intelligence openpi-client is git-only. Install it with: "
            f"{OPENPI_CLIENT_INSTALL_COMMAND}",
            name=exc.name,
        ) from exc

    client = websocket_client_policy.WebsocketClientPolicy(
        host=cfg.host,
        port=cfg.port,
        api_key=cfg.api_key,
    )

    def infer(observation: OpenpiObservation) -> OpenpiResponse:
        payload = dict(observation)
        for key in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
            payload[key] = image_tools.resize_with_pad(
                np.asarray(payload[key]), cfg.resize_px, cfg.resize_px
            )
        response: OpenpiResponse = client.infer(payload)
        return response

    return infer


class OpenpiPolicy:
    """Inspect Robots policy for DROID-compatible OpenPI websocket servers."""

    def __init__(
        self,
        config: OpenpiConfig | None = None,
        *,
        infer_fn: InferFn | None = None,
        clock: Callable[[], float] | None = None,
        **flat: Any,
    ) -> None:
        self._cfg = config if config is not None else OpenpiConfig.from_kwargs(**flat)
        self._infer_fn = infer_fn
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._instruction: str | None = None
        self.num_inferences = 0
        self.info = PolicyInfo(
            name=self._cfg.name,
            action_space=action_box(),
            observation_space=observation_space(),
            control_hz=None,
        )
        self.config = PolicyConfig(
            action_horizon=self._cfg.action_horizon,
            replan_interval=self._cfg.replan_interval,
        )

    def _infer(self) -> InferFn:
        """Lazily construct the real websocket inference closure."""
        if self._infer_fn is None:
            self._infer_fn = _default_infer(self._cfg)
        return self._infer_fn

    def reset(self, scene: Scene) -> None:
        """Stash the language instruction and reset inference accounting."""
        self._instruction = scene.instruction
        self.num_inferences = 0

    def act(self, observation: Observation) -> ActionChunk:
        """Adapt one observation and return absolute 8-D joint-position targets."""
        required_cameras = ("exterior_cam", "wrist_cam")
        try:
            exterior, wrist = (observation.images[name] for name in required_cameras)
        except KeyError as exc:
            raise ValueError(f"observation missing camera {exc} required by openpi") from exc
        if packing.STATE_KEY not in observation.state:
            raise ValueError(f"observation missing state key {packing.STATE_KEY!r}")
        state = packing.validate_dim(observation.state[packing.STATE_KEY])

        request: dict[str, Any] = {
            "observation/exterior_image_1_left": np.asarray(exterior, dtype=np.uint8),
            "observation/wrist_image_left": np.asarray(wrist, dtype=np.uint8),
            "observation/joint_position": packing.arm_joints(state),
            "observation/gripper_position": np.asarray(
                [1.0 - packing.gripper(state)], dtype=np.float64
            ),
            "prompt": self._instruction or "",
        }

        started = self._clock()
        response = self._infer()(request)
        elapsed = self._clock() - started
        if "actions" not in response:
            raise ValueError("openpi response missing 'actions'")
        actions = np.asarray(response["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != packing.TOTAL_DIM:
            raise ValueError(
                f"openpi returned actions of shape {actions.shape}; "
                f"expected (N, {packing.TOTAL_DIM})"
            )
        if actions.shape[0] == 0:
            raise ValueError("openpi returned an empty action chunk")
        if not np.isfinite(actions).all():
            raise ValueError("openpi returned non-finite actions")

        adapted = actions.copy()
        if self._cfg.actions_are_velocity:
            velocities = np.clip(adapted[:, : packing.NUM_JOINTS], -1.0, 1.0)
            adapted[:, : packing.NUM_JOINTS] = packing.arm_joints(
                state
            ) + self._cfg.velocity_action_scale * np.cumsum(velocities, axis=0)
        adapted[:, packing.GRIPPER_IDX] = 1.0 - adapted[:, packing.GRIPPER_IDX]
        adapted = adapted[: self._cfg.action_horizon]

        self.num_inferences += 1
        return ActionChunk(
            actions=[Action(data=row.copy()) for row in adapted],
            control_hz=15.0,
            inference_latency_s=elapsed,
        )
