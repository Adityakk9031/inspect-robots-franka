from __future__ import annotations

import numpy as np
from inspect_robots import eval as robots_eval

from inspect_robots_franka.config import FrankaConfig, OpenpiConfig
from inspect_robots_franka.embodiment import FrankaEmbodiment
from inspect_robots_franka.operator import OperatorIO
from inspect_robots_franka.policy import OpenpiPolicy


class _Driver:
    def __init__(self) -> None:
        self.joints = np.zeros(7)
        self.width = 0.08

    def read_joints(self) -> np.ndarray:
        return self.joints.copy()

    def read_gripper_width(self) -> float:
        return self.width

    def move_joints(self, target: np.ndarray) -> None:
        self.joints = np.asarray(target).copy()

    def move_joints_sync(self, target: np.ndarray) -> None:
        self.joints = np.asarray(target).copy()

    def move_gripper(self, width: float) -> None:
        self.width = width

    def disconnect(self) -> None:
        return None


def _cameras() -> dict[str, np.ndarray]:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"exterior_cam": image, "wrist_cam": image}


def test_full_eval_propagates_success_and_policy_metadata() -> None:
    driver = _Driver()
    policy = OpenpiPolicy(
        OpenpiConfig(actions_are_velocity=False),
        infer_fn=lambda _payload: {"actions": np.asarray([[0.0] * 7 + [0.0]])},
        clock=lambda: 0.0,
    )
    embodiment = FrankaEmbodiment(
        FrankaConfig(hostname="robot"),
        driver_factory=lambda _cfg: driver,
        camera_reader=_cameras,
        operator=OperatorIO(input_fn=lambda _prompt: "yes"),
        poll_end=lambda: True,
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    logs = robots_eval("cubepick-reach", policy, embodiment, sinks=[], seed=0)
    assert len(logs) == 1
    log = logs[0]
    assert log.status == "success"
    assert log.results.metrics["success_at_end"] == 1.0
    assert log.eval.policy_config == {
        "action_horizon": 15,
        "replan_interval": 8,
        "temperature": None,
    }
    assert embodiment.num_steps == 1
