from __future__ import annotations

import builtins
import sys
from types import ModuleType

import pytest

from inspect_robots_franka._franky import FRANKY_INSTALL_COMMAND, _load_franky


def test_loader_returns_imported_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("franky")
    monkeypatch.setitem(sys.modules, "franky", fake)
    assert _load_franky() is fake


def test_loader_gives_install_and_firmware_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def importing(name: str, *args: object, **kwargs: object) -> object:
        if name == "franky":
            raise ModuleNotFoundError("no franky", name="franky")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "franky", raising=False)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ModuleNotFoundError, match="firmware-to-wheel") as caught:
        _load_franky()
    assert FRANKY_INSTALL_COMMAND in str(caught.value)


def test_loader_does_not_mask_nested_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def importing(name: str, *args: object, **kwargs: object) -> object:
        if name == "franky":
            raise ModuleNotFoundError("no nested", name="nested_dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "franky", raising=False)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ModuleNotFoundError) as caught:
        _load_franky()
    assert caught.value.name == "nested_dependency"
