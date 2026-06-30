from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRANSLATION_KEYS = ("transl", "trans", "root_trans_offset")
GLOBAL_ORIENT_KEYS = ("global_orient", "root_orient")
BODY_POSE_KEYS = ("body_pose", "pose_body")
POSE_KEYS = ("poses",)


@dataclass(frozen=True)
class MotionInputInspection:
    path: Path
    is_motion: bool
    reason: str
    frame_count: int | None = None


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


def filter_motion_inputs(
    inputs: list[Path | str],
    *,
    min_frames: int = 2,
) -> list[Path | str]:
    output: list[Path | str] = []
    for value in inputs:
        if str(value).lower() == "mock":
            output.append("mock")
            continue
        inspection = inspect_motion_input(Path(value), min_frames=min_frames)
        if inspection.is_motion:
            output.append(inspection.path)
    return output


def inspect_motion_input(path: Path | str, *, min_frames: int = 2) -> MotionInputInspection:
    input_path = Path(path).expanduser().resolve(strict=False)
    if min_frames <= 0:
        raise ValueError("min_frames must be positive.")
    if not input_path.exists():
        return MotionInputInspection(input_path, False, "missing")
    if not input_path.is_file():
        return MotionInputInspection(input_path, False, "not_file")
    suffix = input_path.suffix.lower()
    try:
        if suffix == ".npz":
            return _inspect_npz_motion(input_path, min_frames=min_frames)
        if suffix == ".npy":
            return _inspect_npy_motion(input_path, min_frames=min_frames)
    except Exception as exc:
        return MotionInputInspection(input_path, False, f"unreadable:{type(exc).__name__}")
    return MotionInputInspection(input_path, False, "unsupported_extension")


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
    return any(
        fnmatch.fnmatch(candidate, pattern) or fnmatch.fnmatch(candidate.lower(), pattern.lower())
        for candidate in candidates
        for pattern in exclude_patterns
    )


def _inspect_npz_motion(path: Path, *, min_frames: int) -> MotionInputInspection:
    with np.load(path, allow_pickle=False) as data:
        shapes = {key: _npz_array_shape(data, key) for key in data.files}

    translation_key, translation_shape = _first_shape(shapes, TRANSLATION_KEYS)
    if translation_shape is None or len(translation_shape) != 2 or translation_shape[1] != 3:
        return MotionInputInspection(path, False, "missing_translation")
    frame_count = int(translation_shape[0])
    if frame_count < min_frames:
        return MotionInputInspection(path, False, "too_few_frames", frame_count=frame_count)

    pose_key, pose_shape = _first_shape(shapes, POSE_KEYS)
    if pose_shape is not None:
        if len(pose_shape) != 2 or pose_shape[0] != frame_count or pose_shape[1] < 66:
            return MotionInputInspection(path, False, f"invalid_pose_shape:{pose_key}", frame_count=frame_count)
        return MotionInputInspection(path, True, "smpl_motion", frame_count=frame_count)

    global_key, global_shape = _first_shape(shapes, GLOBAL_ORIENT_KEYS)
    body_key, body_shape = _first_shape(shapes, BODY_POSE_KEYS)
    if global_shape is None:
        return MotionInputInspection(path, False, "missing_global_orient", frame_count=frame_count)
    if body_shape is None:
        return MotionInputInspection(path, False, "missing_body_pose", frame_count=frame_count)
    if len(global_shape) != 2 or global_shape != translation_shape:
        return MotionInputInspection(path, False, f"invalid_global_orient_shape:{global_key}", frame_count=frame_count)
    if len(body_shape) != 2 or body_shape[0] != frame_count or body_shape[1] < 63 or body_shape[1] % 3 != 0:
        return MotionInputInspection(path, False, f"invalid_body_pose_shape:{body_key}", frame_count=frame_count)
    return MotionInputInspection(path, True, "smpl_motion", frame_count=frame_count)


def _inspect_npy_motion(path: Path, *, min_frames: int) -> MotionInputInspection:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 2 or shape[1] < 69:
        return MotionInputInspection(path, False, "invalid_npy_shape")
    frame_count = int(shape[0])
    if frame_count < min_frames:
        return MotionInputInspection(path, False, "too_few_frames", frame_count=frame_count)
    return MotionInputInspection(path, True, "phuma_motion", frame_count=frame_count)


def _first_shape(shapes: dict[str, tuple[int, ...] | None], keys: tuple[str, ...]) -> tuple[str | None, tuple[int, ...] | None]:
    for key in keys:
        shape = shapes.get(key)
        if shape is not None:
            return key, shape
    return None, None


def _npz_array_shape(data, key: str) -> tuple[int, ...] | None:
    member = f"{key}.npy"
    try:
        with data.zip.open(member) as file:
            version = np.lib.format.read_magic(file)
            shape, _, _ = np.lib.format._read_array_header(file, version)
        return tuple(int(value) for value in shape)
    except Exception:
        try:
            return tuple(int(value) for value in data[key].shape)
        except Exception:
            return None
