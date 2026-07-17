from __future__ import annotations

from typing import Any

from inspect_robots.compat import check_compatibility
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.spaces import ActionSemantics, Box

from inspect_robots_franka.config import action_box, observation_space
from inspect_robots_franka.embodiment import FrankaEmbodiment
from inspect_robots_franka.policy import OpenpiPolicy


class _Policy:
    config = PolicyConfig()

    def __init__(self, info: PolicyInfo) -> None:
        self.info = info

    def reset(self, scene: object) -> None:
        return None

    def act(self, observation: object) -> Any:
        raise AssertionError("not called")


def test_pair_has_zero_errors_and_zero_warnings() -> None:
    report = check_compatibility(OpenpiPolicy(), FrankaEmbodiment())
    assert report.ok is True
    assert report.errors == []
    assert report.warnings == []


def test_builtin_cubepick_reach_is_realizable() -> None:
    task = resolve("task", "cubepick-reach")
    report = check_compatibility(OpenpiPolicy(), FrankaEmbodiment(), task)
    assert report.errors == []


def test_wrong_dimension_is_a_hard_error() -> None:
    info = PolicyInfo(
        name="wrong",
        action_space=Box(shape=(7,), semantics=ActionSemantics(control_mode="joint_pos")),
    )
    report = check_compatibility(_Policy(info), FrankaEmbodiment())  # type: ignore[arg-type]
    assert any(issue.code == "action_dim" for issue in report.errors)


def test_advertised_policy_rate_warns() -> None:
    info = PolicyInfo(
        name="rated",
        action_space=action_box(),
        observation_space=observation_space(),
        control_hz=30.0,
    )
    report = check_compatibility(_Policy(info), FrankaEmbodiment())  # type: ignore[arg-type]
    assert report.ok is True
    assert [issue.code for issue in report.warnings] == ["control_rate"]


def test_velocity_declaring_policy_is_a_control_mode_error() -> None:
    info = PolicyInfo(
        name="dishonest",
        action_space=Box(
            shape=(8,),
            semantics=ActionSemantics(
                control_mode="joint_vel",
                gripper="continuous",
                frame="base",
            ),
        ),
        observation_space=observation_space(),
    )
    report = check_compatibility(_Policy(info), FrankaEmbodiment())  # type: ignore[arg-type]
    assert any(issue.code == "control_mode" for issue in report.errors)
