from __future__ import annotations

import pytest
from inspect_robots.errors import EmbodimentFault

from inspect_robots_franka.operator import OperatorIO, default_poll_end


def _scripted(answers: list[str]):
    seen: list[str] = []

    def input_fn(prompt: str) -> str:
        seen.append(prompt)
        return answers.pop(0)

    return input_fn, seen


def test_wait_ready_reads_prompt() -> None:
    input_fn, seen = _scripted([""])
    OperatorIO(input_fn=input_fn, output_fn=lambda _line: None).wait_ready("ready?")
    assert seen == ["ready?"]


@pytest.mark.parametrize("exception", [EOFError, OSError])
def test_wait_ready_turns_dead_stdin_into_embodiment_fault(
    exception: type[Exception],
) -> None:
    def input_fn(_prompt: str) -> str:
        raise exception("closed")

    with pytest.raises(EmbodimentFault, match=r"FrankaConfig\(unattended=True\)"):
        OperatorIO(input_fn=input_fn).wait_ready()


@pytest.mark.parametrize("answer", ["y", "Yes", "1", "TRUE", "success", "pass"])
def test_confirm_success_affirmative(answer: str) -> None:
    assert OperatorIO(input_fn=lambda _prompt: answer).confirm_success() is True


@pytest.mark.parametrize("answer", ["n", "no", "", "nope"])
def test_confirm_success_negative(answer: str) -> None:
    assert OperatorIO(input_fn=lambda _prompt: answer).confirm_success() is False


def test_default_poll_is_exposed() -> None:
    assert callable(default_poll_end)
