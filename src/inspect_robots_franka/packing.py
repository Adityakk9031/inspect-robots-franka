"""Canonical 8-D joint-position packing for a Franka arm and hand.

The shared vector is ``[joint1, ..., joint7, gripper]``. Revolute slots are
absolute radians. The final slot is normalized with 0 closed and 1 open. This
module is pure NumPy so importing the package never loads a hardware stack.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NUM_JOINTS = 7
GRIPPER_IDX = NUM_JOINTS
TOTAL_DIM = NUM_JOINTS + 1
DIM_LABELS: tuple[str, ...] = (*(f"joint{i}" for i in range(1, 8)), "gripper")
STATE_KEY = "joint_pos"

Vec = npt.NDArray[np.float64]


def validate_dim(vec: npt.ArrayLike) -> Vec:
    """Return a one-dimensional float vector of length eight.

    Two-dimensional inputs are rejected instead of flattened because flattening
    can silently scramble a robot action's declared packing.
    """
    arr: Vec = np.asarray(vec, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != TOTAL_DIM:
        raise ValueError(f"expected an {TOTAL_DIM}-D vector, got shape {np.shape(vec)}")
    return arr


def arm_joints(vec: npt.ArrayLike) -> Vec:
    """Return a copy of the seven revolute-joint slots."""
    out: Vec = validate_dim(vec)[:NUM_JOINTS].copy()
    return out


def gripper(vec: npt.ArrayLike) -> float:
    """Return the normalized gripper slot as a scalar."""
    return float(validate_dim(vec)[GRIPPER_IDX])
