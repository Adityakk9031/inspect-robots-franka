from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from inspect_robots.conformance import device_slots, missing_runtime_requirements
from inspect_robots.embodiment import SELF_PACED
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.task import TaskEnvelope
from inspect_robots.types import Action

from inspect_robots_franka.config import DEFAULT_HOME_POSE, FrankaConfig
from inspect_robots_franka.embodiment import Driver, FrankaEmbodiment, _opencv_camera_reader
from inspect_robots_franka.operator import OperatorIO


class _FakeDriver:
    def __init__(self, events: list[str] | None = None) -> None:
        self.joints = np.zeros(7)
        self.width = 0.02
        self.async_joints: list[np.ndarray] = []
        self.sync_joints: list[np.ndarray] = []
        self.gripper_commands: list[float] = []
        self.disconnect_calls = 0
        self.events = events
        self.fail_sync = False
        self.fail_disconnect = False

    def read_joints(self) -> np.ndarray:
        return self.joints.copy()

    def read_gripper_width(self) -> float:
        return self.width

    def move_joints(self, target: np.ndarray) -> None:
        self.async_joints.append(np.asarray(target).copy())
        self.joints = np.asarray(target).copy()
        if self.events is not None:
            self.events.append("move_joints")

    def move_joints_sync(self, target: np.ndarray) -> None:
        if self.fail_sync:
            raise RuntimeError("park failed")
        self.sync_joints.append(np.asarray(target).copy())
        self.joints = np.asarray(target).copy()
        if self.events is not None:
            self.events.append("move_joints_sync")

    def move_gripper(self, width: float) -> None:
        self.gripper_commands.append(width)
        self.width = width
        if self.events is not None:
            self.events.append("move_gripper")

    def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self.events is not None:
            self.events.append("disconnect")
        if self.fail_disconnect:
            raise RuntimeError("disconnect failed")


def _camera_reader() -> dict[str, np.ndarray]:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    return {"exterior_cam": image, "wrist_cam": image}


def _scene() -> Scene:
    return Scene(id="s", instruction="pick up the cube")


def _embodiment(
    driver: _FakeDriver,
    cfg: FrankaConfig | None = None,
    **kwargs: Any,
) -> FrankaEmbodiment:
    config = cfg or FrankaConfig(hostname="robot", unattended=True)
    return FrankaEmbodiment(
        config,
        driver_factory=lambda _cfg: driver,
        camera_reader=_camera_reader,
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
        **kwargs,
    )


def test_init_is_inert_and_declares_contract() -> None:
    calls = 0

    def factory(_cfg: FrankaConfig) -> Driver:
        nonlocal calls
        calls += 1
        return _FakeDriver()

    embodiment = FrankaEmbodiment(driver_factory=factory)
    assert calls == 0
    assert embodiment.info.name == "franka"
    assert embodiment.info.action_space.dim == 8
    assert embodiment.info.control_hz == 15.0
    assert embodiment.info.capabilities == frozenset({SELF_PACED})


def test_reset_connects_lazily_homes_then_observes_open_positive_width() -> None:
    driver = _FakeDriver()
    driver.width = 0.02
    embodiment = _embodiment(driver)
    observation = embodiment.reset(_scene())
    assert len(driver.sync_joints) == 1
    assert driver.sync_joints[0] == pytest.approx(DEFAULT_HOME_POSE[:7])
    assert driver.gripper_commands == pytest.approx([0.08])
    assert observation.state["joint_pos"][-1] == pytest.approx(1.0)
    assert observation.instruction == "pick up the cube"
    assert observation.state_time == 0.0
    assert set(observation.image_times) == {"exterior_cam", "wrist_cam"}


def test_operator_stand_clear_and_scene_ready_happen_around_homing() -> None:
    events: list[str] = []
    driver = _FakeDriver(events)

    def prompt(text: str) -> str:
        events.append("stand_clear" if "Stand clear" in text else "scene_ready")
        return ""

    embodiment = _embodiment(
        driver,
        FrankaConfig(hostname="robot"),
        operator=OperatorIO(input_fn=prompt, output_fn=lambda _line: None),
        poll_end=lambda: False,
    )
    embodiment.reset(_scene())
    assert events[:4] == ["stand_clear", "move_joints_sync", "move_gripper", "scene_ready"]


def test_step_hard_clamps_without_an_approver_and_paces() -> None:
    driver = _FakeDriver()
    sleeps: list[float] = []
    cfg = FrankaConfig(hostname="robot", unattended=True)
    embodiment = FrankaEmbodiment(
        cfg,
        driver_factory=lambda _cfg: driver,
        camera_reader=_camera_reader,
        sleep_fn=sleeps.append,
        clock=lambda: 0.0,
    )
    embodiment.reset(_scene())
    result = embodiment.step(Action(data=np.full(8, 100.0)))
    assert driver.async_joints[-1] == pytest.approx(cfg.high[:7])
    assert driver.gripper_commands[-1] == pytest.approx(cfg.gripper_max_width)
    assert sleeps == pytest.approx([1.0 / 15.0])
    assert result.terminated is False
    assert embodiment.num_steps == 1


def test_pacing_never_sleeps_negative_time() -> None:
    driver = _FakeDriver()
    times = iter([0.0, 0.0, 1.0, 1.0, 1.0])
    sleeps: list[float] = []
    embodiment = FrankaEmbodiment(
        FrankaConfig(hostname="robot", unattended=True),
        driver_factory=lambda _cfg: driver,
        camera_reader=_camera_reader,
        sleep_fn=sleeps.append,
        clock=times.__next__,
    )
    embodiment.reset(_scene())
    embodiment.step(Action(data=np.asarray(DEFAULT_HOME_POSE)))
    assert sleeps == [0.0]


def test_gripper_deadband_gates_commands_and_denormalizes_asymmetrically() -> None:
    driver = _FakeDriver()
    embodiment = _embodiment(driver)
    embodiment.reset(_scene())
    driver.gripper_commands.clear()
    home = np.asarray(DEFAULT_HOME_POSE)
    embodiment.step(Action(data=home))
    assert driver.gripper_commands == []
    target = home.copy()
    target[-1] = 0.25
    embodiment.step(Action(data=target))
    assert driver.gripper_commands == pytest.approx([0.02])
    embodiment._last_gripper_command = None
    target[-1] = 0.2
    embodiment.step(Action(data=target))
    assert driver.gripper_commands[-1] == pytest.approx(0.016)


def test_default_approver_gripper_ramp_has_bounded_command_cadence() -> None:
    driver = _FakeDriver()
    embodiment = _embodiment(driver)
    embodiment.reset(_scene())
    driver.gripper_commands.clear()
    pose = np.asarray(DEFAULT_HOME_POSE)
    for target in np.arange(0.95, -0.001, -0.05):
        action = pose.copy()
        action[-1] = target
        embodiment.step(Action(data=action))
    assert 5 <= len(driver.gripper_commands) <= 10


@pytest.mark.parametrize("devices", [(None, None), ("/dev/video0", None)])
def test_missing_or_partial_camera_config_fails_before_connect(
    devices: tuple[str | None, str | None],
) -> None:
    calls = 0

    def factory(_cfg: FrankaConfig) -> Driver:
        nonlocal calls
        calls += 1
        return _FakeDriver()

    cfg = FrankaConfig(
        hostname="robot", exterior_cam_device=devices[0], wrist_cam_device=devices[1]
    )
    embodiment = FrankaEmbodiment(cfg, driver_factory=factory)
    with pytest.raises(ConfigError, match="require both"):
        embodiment.reset(_scene())
    assert calls == 0


def test_injected_reader_wins_over_partial_devices_and_hostname_fails_after_camera_check() -> None:
    driver = _FakeDriver()
    cfg = FrankaConfig(hostname="robot", exterior_cam_device="ignored")
    embodiment = FrankaEmbodiment(
        cfg,
        driver_factory=lambda _cfg: driver,
        camera_reader=_camera_reader,
        operator=OperatorIO(input_fn=lambda _prompt: ""),
        sleep_fn=lambda _delay: None,
        clock=lambda: 0.0,
    )
    assert embodiment.reset(_scene()).images.keys() == _camera_reader().keys()

    missing_host = FrankaEmbodiment(camera_reader=_camera_reader)
    with pytest.raises(ConfigError, match="hostname"):
        missing_host.reset(_scene())

    invalid_reader = FrankaEmbodiment(
        FrankaConfig(hostname="robot"),
        driver_factory=lambda _cfg: driver,
        camera_reader="not callable",  # type: ignore[arg-type]
    )
    with pytest.raises(ConfigError, match="must be callable"):
        invalid_reader.reset(_scene())


def test_two_device_config_selects_builtin_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = _FakeDriver()
    calls: list[FrankaConfig] = []

    def builder(cfg: FrankaConfig) -> Callable[[], dict[str, np.ndarray]]:
        calls.append(cfg)
        return _camera_reader

    monkeypatch.setattr("inspect_robots_franka.embodiment._opencv_camera_reader", builder)
    cfg = FrankaConfig(
        hostname="robot",
        unattended=True,
        exterior_cam_device=0,
        wrist_cam_device=1,
    )
    embodiment = FrankaEmbodiment(
        cfg, driver_factory=lambda _cfg: driver, sleep_fn=lambda _delay: None, clock=lambda: 0.0
    )
    embodiment.reset(_scene())
    assert calls == [cfg]


def test_builtin_camera_builder_is_lazy() -> None:
    cfg = FrankaConfig(
        exterior_cam_device="/dev/video0",
        wrist_cam_device="/dev/video1",
    )
    assert callable(_opencv_camera_reader(cfg))


@pytest.mark.parametrize(("verdict", "reason"), [("yes", "success"), ("no", "failure")])
def test_operator_verdict_uses_termination_reason(verdict: str, reason: str) -> None:
    driver = _FakeDriver()
    answers = iter(["", "", verdict])
    embodiment = _embodiment(
        driver,
        FrankaConfig(hostname="robot"),
        operator=OperatorIO(input_fn=lambda _prompt: next(answers)),
        poll_end=lambda: True,
    )
    embodiment.reset(_scene())
    result = embodiment.step(Action(data=np.asarray(DEFAULT_HOME_POSE)))
    assert result.terminated is True
    assert result.termination_reason == reason
    assert result.info == {"operator_confirmed": verdict == "yes"}


def test_unattended_skips_poll_and_prompts() -> None:
    driver = _FakeDriver()
    polled = 0

    def poll() -> bool:
        nonlocal polled
        polled += 1
        return True

    embodiment = _embodiment(driver, poll_end=poll)
    embodiment.reset(_scene())
    result = embodiment.step(Action(data=np.asarray(DEFAULT_HOME_POSE)))
    assert result.terminated is False
    assert polled == 0


def test_close_parks_and_is_idempotent() -> None:
    driver = _FakeDriver()
    rest = (*DEFAULT_HOME_POSE[:-1], 0.25)
    embodiment = _embodiment(
        driver, FrankaConfig(hostname="robot", unattended=True, rest_pose=rest)
    )
    embodiment.reset(_scene())
    driver.sync_joints.clear()
    driver.gripper_commands.clear()
    embodiment.close()
    embodiment.close()
    assert driver.sync_joints == [pytest.approx(rest[:7])]
    assert driver.gripper_commands == []  # park is arm-only: no gripper race with disconnect
    assert driver.disconnect_calls == 1


def test_close_disconnects_and_clears_handle_when_parking_or_disconnect_errors() -> None:
    driver = _FakeDriver()
    embodiment = _embodiment(
        driver,
        FrankaConfig(hostname="robot", unattended=True, rest_pose=DEFAULT_HOME_POSE),
    )
    embodiment.reset(_scene())
    driver.fail_sync = True
    with pytest.raises(RuntimeError, match="park failed"):
        embodiment.close()
    assert driver.disconnect_calls == 1
    embodiment.close()

    driver2 = _FakeDriver()
    embodiment2 = _embodiment(driver2)
    embodiment2.reset(_scene())
    driver2.fail_disconnect = True
    with pytest.raises(RuntimeError, match="disconnect failed"):
        embodiment2.close()
    embodiment2.close()
    assert driver2.disconnect_calls == 1


def test_step_before_reset_and_bad_driver_joint_shape_are_rejected() -> None:
    embodiment = _embodiment(_FakeDriver())
    with pytest.raises(RuntimeError, match="before reset"):
        embodiment.step(Action(data=np.zeros(8)))
    driver = _FakeDriver()
    embodiment = _embodiment(driver)
    embodiment.reset(_scene())
    driver.joints = np.zeros(6)
    with pytest.raises(ValueError, match="driver returned joints"):
        embodiment._observe("instruction")


def test_bind_task_runtime_requirements_and_device_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    embodiment = FrankaEmbodiment()
    embodiment.bind_task(TaskEnvelope(name="task", max_steps=42))
    assert embodiment._bound_max_steps == 42
    slots = device_slots(FrankaEmbodiment)
    assert [slot.arg for slot in slots] == ["exterior_cam_device", "wrist_cam_device"]
    assert {slot.kind for slot in slots} == {"v4l2"}
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    missing = missing_runtime_requirements(FrankaEmbodiment)
    assert missing == {
        "franky": "pip install franky-control",
        "cv2": "pip install opencv-python-headless",
    }


def test_bound_task_horizon_appears_in_operator_status() -> None:
    driver = _FakeDriver()
    output: list[str] = []
    embodiment = _embodiment(
        driver,
        FrankaConfig(hostname="robot"),
        operator=OperatorIO(input_fn=lambda _prompt: "", output_fn=output.append),
        poll_end=lambda: False,
    )
    embodiment.bind_task(TaskEnvelope(name="task", max_steps=30))
    embodiment.reset(_scene())
    assert output == ["Running: press Enter to end the episode, then y/N to score. Max 2s."]


def test_close_without_connection_clears_bound_horizon() -> None:
    embodiment = FrankaEmbodiment()
    embodiment.bind_task(TaskEnvelope(name="task", max_steps=3))
    embodiment.close()
    assert embodiment._bound_max_steps is None
