from __future__ import annotations

import io
import sys
import types

import pytest

import retargeter.progress as progress_module
from retargeter.runtime_logging import configure_native_runtime_logging
from retargeter.progress import make_progress


class _TTYStringIO(io.StringIO):
    def __init__(self, *, tty: bool):
        super().__init__()
        self._tty = bool(tty)

    def isatty(self) -> bool:
        return self._tty


class _FakeTqdm:
    def __init__(self, *, total, desc, unit, leave, file, disable, dynamic_ncols, position):
        self.file = file
        self.file.write(f"bar:{desc}:{total}:{unit}:{position}\n")

    def update(self, n=1):
        self.file.write(f"update:{n}\n")

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self.file.write(f"postfix:{values}\n")

    def close(self):
        self.file.write("close\n")

    @staticmethod
    def write(message, file=None):
        file.write(f"stage:{message}\n")


def test_make_progress_auto_on_off_and_stdout_clean(monkeypatch, capsys):
    monkeypatch.setattr(progress_module, "_load_tqdm", lambda: _FakeTqdm)
    tty_stream = _TTYStringIO(tty=True)
    non_tty_stream = _TTYStringIO(tty=False)

    auto_enabled = make_progress("auto", stream=tty_stream)
    auto_enabled.stage("preprocess")
    with auto_enabled.bar(total=2, desc="IK retarget", unit="frame") as bar:
        bar.update(1)
        bar.set_postfix({"ok": 1})

    auto_disabled = make_progress("auto", stream=non_tty_stream)
    auto_disabled.stage("hidden")
    with auto_disabled.bar(total=1, desc="hidden", unit="it") as bar:
        bar.update(1)

    forced = make_progress("on", stream=non_tty_stream)
    forced.stage("forced")

    off = make_progress("off", stream=tty_stream)
    off.stage("off")

    assert "stage:preprocess" in tty_stream.getvalue()
    assert "bar:IK retarget:2:frame:0" in tty_stream.getvalue()
    assert "postfix:{'ok': 1}" in tty_stream.getvalue()
    assert "stage:hidden" not in non_tty_stream.getvalue()
    assert "stage:forced" in non_tty_stream.getvalue()
    assert capsys.readouterr().out == ""


def test_progress_missing_tqdm_raises_only_when_enabled(monkeypatch):
    def missing_tqdm():
        raise RuntimeError("tqdm is required")

    monkeypatch.setattr(progress_module, "_load_tqdm", missing_tqdm)
    disabled = make_progress("auto", stream=_TTYStringIO(tty=False))
    disabled.stage("ignored")

    enabled = make_progress("on", stream=_TTYStringIO(tty=False))
    with pytest.raises(RuntimeError, match="tqdm is required"):
        enabled.stage("visible")


def test_configure_native_runtime_logging_sets_warp_quiet(monkeypatch):
    fake_warp = types.SimpleNamespace(config=types.SimpleNamespace(quiet=False))
    monkeypatch.setitem(sys.modules, "warp", fake_warp)

    configure_native_runtime_logging(quiet=True)
    assert fake_warp.config.quiet is True

    configure_native_runtime_logging(quiet=False)
    assert fake_warp.config.quiet is False
