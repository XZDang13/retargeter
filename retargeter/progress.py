from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, TextIO


class ProgressBar(Protocol):
    def update(self, n: int = 1) -> None: ...

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs: Any) -> None: ...


class ProgressReporter(Protocol):
    enabled: bool
    forced: bool

    def stage(self, message: str) -> None: ...

    def child(self, *, position_offset: int = 1) -> "ProgressReporter": ...

    @contextmanager
    def bar(
        self,
        *,
        total: int | None,
        desc: str,
        unit: str = "it",
        leave: bool = False,
    ) -> Iterator[ProgressBar]: ...


class NullProgressBar:
    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs: Any) -> None:
        return None


class NullProgressReporter:
    enabled = False
    forced = False

    def stage(self, message: str) -> None:
        return None

    def child(self, *, position_offset: int = 1) -> "NullProgressReporter":
        return self

    @contextmanager
    def bar(
        self,
        *,
        total: int | None,
        desc: str,
        unit: str = "it",
        leave: bool = False,
    ) -> Iterator[NullProgressBar]:
        yield NullProgressBar()


class TqdmProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        forced: bool = False,
        stream: TextIO | None = None,
        position: int = 0,
        tqdm_factory=None,
    ):
        self.enabled = bool(enabled)
        self.forced = bool(forced)
        self.stream = stream if stream is not None else sys.stderr
        self.position = int(position)
        self._tqdm_factory = tqdm_factory

    def stage(self, message: str) -> None:
        if not self.enabled:
            return
        tqdm_factory = self._resolve_tqdm()
        write = getattr(tqdm_factory, "write", None)
        if write is None:
            print(message, file=self.stream)
            return
        write(message, file=self.stream)

    def child(self, *, position_offset: int = 1) -> "TqdmProgressReporter":
        return TqdmProgressReporter(
            enabled=self.enabled,
            forced=self.forced,
            stream=self.stream,
            position=self.position + int(position_offset),
            tqdm_factory=self._tqdm_factory,
        )

    @contextmanager
    def bar(
        self,
        *,
        total: int | None,
        desc: str,
        unit: str = "it",
        leave: bool = False,
    ) -> Iterator[ProgressBar]:
        if not self.enabled:
            yield NullProgressBar()
            return
        tqdm_factory = self._resolve_tqdm()
        bar = tqdm_factory(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            file=self.stream,
            disable=False,
            dynamic_ncols=True,
            position=self.position,
        )
        try:
            yield bar
        finally:
            bar.close()

    def _resolve_tqdm(self):
        if self._tqdm_factory is None:
            self._tqdm_factory = _load_tqdm()
        return self._tqdm_factory


def make_progress(mode: str = "auto", *, stream: TextIO | None = None) -> ProgressReporter:
    selected = str(mode).lower()
    if selected not in {"auto", "on", "off"}:
        raise ValueError("progress mode must be one of auto, on, or off.")
    output = stream if stream is not None else sys.stderr
    if selected == "off":
        return NullProgressReporter()
    enabled = selected == "on" or _isatty(output)
    return TqdmProgressReporter(enabled=enabled, forced=selected == "on", stream=output)


def get_progress(progress: ProgressReporter | None) -> ProgressReporter:
    return progress if progress is not None else NullProgressReporter()


def _isatty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if isatty is not None else False


def _load_tqdm():
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required for refine progress. Install it with: python -m pip install tqdm") from exc
    return tqdm
