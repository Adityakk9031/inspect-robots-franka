from __future__ import annotations

from inspect_robots_franka.config import DEFAULT_JOINT_HIGH, DEFAULT_JOINT_LOW, FrankaConfig
from inspect_robots_franka.embodiment import _DOCS, FrankaEmbodiment
from inspect_robots_franka.packing import DIM_LABELS


def test_docs_name_every_dimension_exactly_once_in_bullets() -> None:
    docs = FrankaEmbodiment().info.docs
    assert docs is not None
    bullets = [line for line in docs.splitlines() if line.startswith("- ")]
    for label in DIM_LABELS:
        assert sum(line.startswith(f"- {label}:") for line in bullets) == 1


def test_docs_do_not_leak_numeric_joint_limits() -> None:
    docs = FrankaEmbodiment().info.docs or ""
    for value in (*DEFAULT_JOINT_LOW[:7], *DEFAULT_JOINT_HIGH[:7]):
        assert str(value) not in docs


def test_docs_extra_is_stripped_and_appended_once() -> None:
    embodiment = FrankaEmbodiment(FrankaConfig(docs_extra="  rig note {safe}\n"))
    assert embodiment.info.docs == _DOCS + "\n\nrig note {safe}"
    assert FrankaEmbodiment(FrankaConfig(docs_extra=" \n ")).info.docs == _DOCS
