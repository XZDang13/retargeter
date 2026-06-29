from __future__ import annotations

import fnmatch
from pathlib import Path


def discover_inputs(
    inputs: list[str] | None,
    input_dir: Path | None,
    patterns: list[str],
    recursive: bool,
    input_list: Path | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Path | str]:
    discovered: list[Path | str] = []
    seen: set[str] = set()
    resolved_input_dir = Path(input_dir).expanduser().resolve(strict=False) if input_dir is not None else None

    for value in inputs or []:
        _append_input(value, discovered, seen, input_dir=resolved_input_dir, exclude_patterns=exclude_patterns)

    for value in _read_input_list(input_list):
        _append_input(value, discovered, seen, input_dir=resolved_input_dir, exclude_patterns=exclude_patterns)

    if input_dir is not None:
        directory = Path(input_dir).expanduser()
        if not directory.exists():
            raise FileNotFoundError(f"Batch input directory does not exist: {directory}")
        if not directory.is_dir():
            raise ValueError(f"Batch input path is not a directory: {directory}")
        for path in _discover_from_directory(directory, patterns, recursive):
            _append_input(path, discovered, seen, input_dir=resolved_input_dir, exclude_patterns=exclude_patterns)

    return discovered


def _read_input_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    input_list = Path(path).expanduser()
    values: list[str] = []
    for raw in input_list.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        values.append(value)
    return values


def _discover_from_directory(input_dir: Path, patterns: list[str], recursive: bool) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        matches.extend(path for path in iterator if path.is_file())
    unique = {str(path.resolve(strict=False)): path.resolve(strict=False) for path in matches}
    return [unique[key] for key in sorted(unique)]


def _append_input(
    value: Path | str,
    discovered: list[Path | str],
    seen: set[str],
    *,
    input_dir: Path | None,
    exclude_patterns: list[str] | None,
) -> None:
    if str(value).lower() == "mock":
        discovered.append("mock")
        return

    path = Path(value).expanduser().resolve(strict=False)
    if _is_excluded(path, input_dir=input_dir, exclude_patterns=exclude_patterns):
        return
    key = str(path)
    if key in seen:
        return
    seen.add(key)
    discovered.append(path)


def _is_excluded(path: Path, *, input_dir: Path | None, exclude_patterns: list[str] | None) -> bool:
    if not exclude_patterns:
        return False
    candidates = [path.name, str(path)]
    if input_dir is not None:
        try:
            candidates.append(path.relative_to(input_dir).as_posix())
        except ValueError:
            pass
    return any(fnmatch.fnmatch(candidate, pattern) for candidate in candidates for pattern in exclude_patterns)
