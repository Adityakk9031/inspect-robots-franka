from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
from inspect_robots.policy import PolicyConfig
from inspect_robots.scene import Scene
from inspect_robots.types import Observation

from inspect_robots_franka.config import OpenpiConfig
from inspect_robots_franka.policy import OpenpiPolicy


def _observation(gripper: float = 0.25, joints: np.ndarray | None = None) -> Observation:
    exterior = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    wrist = np.full((2, 3, 3), 17, dtype=np.uint8)
    arm = np.arange(7, dtype=float) / 10 if joints is None else joints
    return Observation(
        images={"exterior_cam": exterior, "wrist_cam": wrist},
        state={"joint_pos": np.concatenate((arm, [gripper]))},
        instruction="observation instruction",
    )


def _infer_returning(actions: np.ndarray, captured: dict[str, Any] | None = None):
    def infer(payload: dict[str, Any]) -> dict[str, np.ndarray]:
        if captured is not None:
            captured.update(payload)
        return {"actions": actions}

    return infer


def test_info_and_config_are_framework_contract_without_api_key() -> None:
    policy = OpenpiPolicy(OpenpiConfig(api_key="secret"))
    assert policy.info.name == "openpi"
    assert policy.info.action_space.dim == 8
    assert policy.info.control_hz is None
    assert policy.config == PolicyConfig(action_horizon=15, replan_interval=8)
    assert "api_key" not in asdict(policy.config)


def test_real_infer_factory_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[OpenpiConfig] = []

    def factory(cfg: OpenpiConfig):
        calls.append(cfg)
        return _infer_returning(np.zeros((1, 8)))

    monkeypatch.setattr("inspect_robots_franka.policy._default_infer", factory)
    policy = OpenpiPolicy(clock=lambda: 0.0)
    assert calls == []
    policy.act(_observation())
    assert calls == [policy._cfg]


def test_request_keys_instruction_and_observation_gripper_polarity() -> None:
    captured: dict[str, Any] = {}
    raw = np.zeros((1, 8))
    policy = OpenpiPolicy(infer_fn=_infer_returning(raw, captured), clock=iter([2.0, 2.4]).__next__)
    policy.reset(Scene(id="s", instruction="pick the mug"))
    chunk = policy.act(_observation(gripper=0.25))
    assert list(captured) == [
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    ]
    assert np.array_equal(
        captured["observation/exterior_image_1_left"], _observation().images["exterior_cam"]
    )
    assert captured["observation/exterior_image_1_left"].dtype == np.uint8
    assert captured["observation/gripper_position"] == pytest.approx([0.75])
    assert captured["prompt"] == "pick the mug"
    assert chunk.inference_latency_s == pytest.approx(0.4)
    assert chunk.control_hz == 15.0
    assert chunk.actions[0].data[7] == pytest.approx(1.0)
    assert policy.num_inferences == 1


def test_velocity_integration_clips_arm_anchors_at_observation_and_leaves_gripper() -> None:
    q_obs = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    raw = np.asarray(
        [
            [2.0, -2.0, 0.5, 0.0, 1.0, -1.0, 0.25, 1.7],
            [1.0, 1.0, -1.0, 0.5, -0.5, 0.0, 2.0, -0.25],
        ]
    )
    policy = OpenpiPolicy(infer_fn=_infer_returning(raw), clock=lambda: 0.0)
    chunk = policy.act(_observation(joints=q_obs))
    expected_velocity = np.clip(raw[:, :7], -1.0, 1.0)
    expected_arm = q_obs + 0.2 * np.cumsum(expected_velocity, axis=0)
    assert np.allclose(chunk.actions[0].data[:7], expected_arm[0])
    assert np.allclose(chunk.actions[1].data[:7], expected_arm[1])
    assert chunk.actions[0].data[7] == pytest.approx(-0.7)
    assert chunk.actions[1].data[7] == pytest.approx(1.25)


def test_position_passthrough_nondefault_scale_and_truncation() -> None:
    positions = np.arange(24, dtype=float).reshape(3, 8) / 10
    direct = OpenpiPolicy(
        OpenpiConfig(actions_are_velocity=False, action_horizon=2),
        infer_fn=_infer_returning(positions),
        clock=lambda: 0.0,
    ).act(_observation())
    assert len(direct) == 2
    assert np.array_equal(direct.actions[0].data[:7], positions[0, :7])
    assert direct.actions[0].data[7] == pytest.approx(1.0 - positions[0, 7])

    velocities = np.ones((1, 8))
    scaled = OpenpiPolicy(
        OpenpiConfig(velocity_action_scale=0.05),
        infer_fn=_infer_returning(velocities),
        clock=lambda: 0.0,
    ).act(_observation(joints=np.zeros(7)))
    assert scaled.actions[0].data[:7] == pytest.approx(np.full(7, 0.05))


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "missing 'actions'"),
        ({"actions": np.zeros((2, 7))}, r"expected \(N, 8\)"),
        ({"actions": np.zeros(8)}, r"expected \(N, 8\)"),
        ({"actions": np.zeros((0, 8))}, "empty action chunk"),
        ({"actions": np.full((1, 8), np.nan)}, "non-finite"),
    ],
)
def test_response_validation(response: dict[str, np.ndarray], message: str) -> None:
    policy = OpenpiPolicy(infer_fn=lambda _payload: response, clock=lambda: 0.0)
    with pytest.raises(ValueError, match=message):
        policy.act(_observation())


def test_helpful_missing_observation_errors_and_reset_counter() -> None:
    infer = _infer_returning(np.zeros((1, 8)))
    policy = OpenpiPolicy(infer_fn=infer, clock=lambda: 0.0)
    with pytest.raises(ValueError, match="missing camera"):
        policy.act(Observation(images={}, state={"joint_pos": np.zeros(8)}))
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="missing state key"):
        policy.act(Observation(images={"exterior_cam": image, "wrist_cam": image}, state={}))
    policy.act(_observation())
    assert policy.num_inferences == 1
    policy.reset(Scene(id="s", instruction="new"))
    assert policy.num_inferences == 0
