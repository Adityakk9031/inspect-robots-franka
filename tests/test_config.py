from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_franka.config import (
    ACTION_SEMANTICS,
    DEFAULT_CAMERAS,
    DEFAULT_HOME_POSE,
    DEFAULT_JOINT_HIGH,
    DEFAULT_JOINT_LOW,
    FrankaConfig,
    OpenpiConfig,
    action_box,
    camera_specs,
    observation_space,
)
from inspect_robots_franka.packing import DIM_LABELS, STATE_KEY, TOTAL_DIM


def test_franka_defaults_use_inset_fr3_limits_and_ready_home() -> None:
    cfg = FrankaConfig()
    assert cfg.control_hz == 15.0
    assert cfg.joint_low == DEFAULT_JOINT_LOW
    assert cfg.joint_high == DEFAULT_JOINT_HIGH
    assert cfg.home_pose == DEFAULT_HOME_POSE
    assert cfg.low.shape == cfg.high.shape == (TOTAL_DIM,)
    assert np.all(cfg.low < cfg.high)
    assert np.all(np.asarray(cfg.home_pose) >= cfg.low)
    assert np.all(np.asarray(cfg.home_pose) <= cfg.high)
    assert DEFAULT_JOINT_LOW[0] == pytest.approx(-2.7437 + 0.05)
    assert DEFAULT_JOINT_HIGH[6] == pytest.approx(3.0159 - 0.05)
    assert (DEFAULT_JOINT_LOW[-1], DEFAULT_JOINT_HIGH[-1]) == (0.0, 1.0)


def test_openpi_defaults_and_policy_metadata_values() -> None:
    cfg = OpenpiConfig()
    assert (cfg.host, cfg.port) == ("127.0.0.1", 8000)
    assert cfg.actions_are_velocity is True
    assert cfg.velocity_action_scale == pytest.approx(0.2)
    assert (cfg.action_horizon, cfg.replan_interval, cfg.resize_px) == (15, 8, 224)


def test_from_kwargs_rejects_unknown_and_parses_pose_fields() -> None:
    with pytest.raises(TypeError, match="unexpected config keys"):
        OpenpiConfig.from_kwargs(secret_timeout=1)
    home = ",".join(str(value) for value in DEFAULT_HOME_POSE)
    cfg = FrankaConfig.from_kwargs(home_pose=home, rest_pose=home, control_hz=10.0)
    assert cfg.home_pose == pytest.approx(DEFAULT_HOME_POSE)
    assert cfg.rest_pose == pytest.approx(DEFAULT_HOME_POSE)
    with pytest.raises(ValueError, match="home_pose must be a comma-separated"):
        FrankaConfig.from_kwargs(home_pose="0,bad")
    assert FrankaConfig.from_kwargs(rest_pose=DEFAULT_HOME_POSE).rest_pose == DEFAULT_HOME_POSE


def test_from_kwargs_parses_string_coercion_types() -> None:
    openpi_cfg = OpenpiConfig.from_kwargs(
        name="custom_openpi",
        port="8000",
        action_horizon="15",
        actions_are_velocity="true",
        velocity_action_scale="0.25",
    )
    assert openpi_cfg.name == "custom_openpi"
    assert openpi_cfg.port == 8000
    assert openpi_cfg.action_horizon == 15
    assert openpi_cfg.actions_are_velocity is True
    assert openpi_cfg.velocity_action_scale == 0.25

    franka_cfg = FrankaConfig.from_kwargs(
        hostname="172.16.0.2",
        control_hz="20.0",
        unattended="false",
        cam_height="720",
    )
    assert franka_cfg.hostname == "172.16.0.2"
    assert franka_cfg.control_hz == 20.0
    assert franka_cfg.unattended is False
    assert franka_cfg.cam_height == 720

    with pytest.raises(ValueError, match="port must be an integer"):
        OpenpiConfig.from_kwargs(port="invalid")
    with pytest.raises(ValueError, match="control_hz must be a float"):
        FrankaConfig.from_kwargs(control_hz="invalid")
    with pytest.raises(ValueError, match="unattended must be a boolean"):
        FrankaConfig.from_kwargs(unattended="invalid")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control_hz": 0.0}, "control_hz"),
        ({"joint_low": (0.0,) * 7}, "joint_low must have 8"),
        ({"joint_high": (0.0,) * 7}, "joint_high must have 8"),
        ({"home_pose": (0.0,) * 7}, "home_pose must have 8"),
        ({"rest_pose": (0.0,) * 7}, "rest_pose must have 8"),
        ({"joint_low": (*DEFAULT_JOINT_LOW[:-1], np.nan)}, "only finite"),
        ({"joint_low": DEFAULT_JOINT_HIGH}, "joint_low must be below"),
        ({"home_pose": (*DEFAULT_HOME_POSE[:-1], 2.0)}, "home_pose must be finite"),
        ({"rest_pose": (*DEFAULT_HOME_POSE[:-1], -1.0)}, "rest_pose must be finite"),
        ({"gripper_max_width": 0.0}, "gripper_max_width"),
        ({"gripper_speed": np.inf}, "gripper_speed"),
        ({"relative_dynamics_factor": 0.0}, "relative_dynamics_factor"),
        ({"relative_dynamics_factor": 1.1}, "relative_dynamics_factor"),
        ({"gripper_deadband": -0.1}, "gripper_deadband"),
        ({"gripper_deadband": 1.0}, "gripper_deadband"),
        ({"cam_height": 0}, "cam_height"),
    ],
)
def test_franka_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FrankaConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"host": ""}, "host"),
        ({"port": 0}, "port"),
        ({"port": True}, "port"),
        ({"velocity_action_scale": 0.0}, "velocity_action_scale"),
        ({"action_horizon": 0}, "action_horizon"),
        ({"replan_interval": True}, "replan_interval"),
        ({"resize_px": 0}, "resize_px"),
    ],
)
def test_openpi_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OpenpiConfig(**kwargs)


def test_shared_spaces_have_pinned_semantics_state_and_cameras() -> None:
    cfg = FrankaConfig(cam_height=12, cam_width=16)
    box = action_box(cfg)
    assert box.shape == (8,)
    assert np.array_equal(box.low, cfg.low)
    assert np.array_equal(box.high, cfg.high)
    assert box.semantics is ACTION_SEMANTICS
    assert ACTION_SEMANTICS.control_mode == "joint_pos"
    assert ACTION_SEMANTICS.dim_labels == DIM_LABELS
    assert action_box().low is action_box().high is None
    obs = observation_space(cfg)
    assert obs.camera_names == frozenset(DEFAULT_CAMERAS)
    assert obs.state_keys == frozenset({STATE_KEY})
    assert obs.state is not None
    assert obs.state.fields[0].shape == (8,)
    assert obs.state.fields[0].unit == "rad+normalized"
    specs = camera_specs(12, 16)
    assert [(item.name, item.height, item.width) for item in specs] == [
        ("exterior_cam", 12, 16),
        ("wrist_cam", 12, 16),
    ]
    assert observation_space().cameras[0].height == 480
