from __future__ import annotations

import numpy as np
import pytest

from inspect_robots_franka import packing


def test_constants_and_labels() -> None:
    assert packing.NUM_JOINTS == 7
    assert packing.GRIPPER_IDX == 7
    assert packing.TOTAL_DIM == 8
    assert packing.STATE_KEY == "joint_pos"
    assert packing.DIM_LABELS == (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "gripper",
    )
    assert len(set(packing.DIM_LABELS)) == packing.TOTAL_DIM


def test_validate_dim_accepts_list_and_returns_float64() -> None:
    out = packing.validate_dim(list(range(8)))
    assert np.array_equal(out, np.arange(8))
    assert out.dtype == np.float64


@pytest.mark.parametrize("bad", [np.zeros(7), np.zeros(9), np.zeros((2, 4))])
def test_validate_dim_rejects_wrong_shape(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="expected an 8-D vector"):
        packing.validate_dim(bad)


def test_accessors_return_expected_values_and_arm_copy() -> None:
    vec = np.arange(8, dtype=float)
    arm = packing.arm_joints(vec)
    assert np.array_equal(arm, np.arange(7))
    assert packing.gripper(vec) == 7.0
    arm[0] = 99.0
    assert vec[0] == 0.0
