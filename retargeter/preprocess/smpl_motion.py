from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class SMPLMotion:
    model_type: Literal["smpl", "smplx"]
    fps: float
    transl: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    betas: np.ndarray | None = None
    gender: str = "neutral"
    left_hand_pose: np.ndarray | None = None
    right_hand_pose: np.ndarray | None = None
    jaw_pose: np.ndarray | None = None
    leye_pose: np.ndarray | None = None
    reye_pose: np.ndarray | None = None
    expression: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def num_frames(self) -> int:
        return int(self.transl.shape[0])

    def validate(self) -> None:
        if self.model_type not in {"smpl", "smplx"}:
            raise ValueError(f"model_type must be 'smpl' or 'smplx', got {self.model_type!r}.")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"fps must be a positive finite value, got {self.fps!r}.")

        transl = _as_array("transl", self.transl)
        global_orient = _as_array("global_orient", self.global_orient)
        body_pose = _as_array("body_pose", self.body_pose)

        if transl.ndim != 2 or transl.shape[1] != 3:
            raise ValueError(f"transl must have shape [T, 3], got {transl.shape}.")
        if global_orient.shape != transl.shape:
            raise ValueError(
                f"global_orient must have shape [T, 3] matching transl, got {global_orient.shape}."
            )
        if body_pose.ndim != 2 or body_pose.shape[0] != transl.shape[0]:
            raise ValueError(
                f"body_pose must have shape [T, J*3] with T={transl.shape[0]}, got {body_pose.shape}."
            )
        if body_pose.shape[1] % 3 != 0:
            raise ValueError(f"body_pose second dimension must be divisible by 3, got {body_pose.shape}.")

        for name in [
            "betas",
            "left_hand_pose",
            "right_hand_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "expression",
        ]:
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = _as_array(name, arr)
            if name != "betas" and arr.shape[0] != transl.shape[0]:
                raise ValueError(f"{name} first dimension must match T={transl.shape[0]}, got {arr.shape}.")
            if name == "betas" and arr.ndim > 1 and arr.shape[0] not in {1, transl.shape[0]}:
                raise ValueError(f"betas must be shape [B], [1, B], or [T, B], got {arr.shape}.")

    def copy(self) -> "SMPLMotion":
        return SMPLMotion(
            model_type=self.model_type,
            fps=float(self.fps),
            transl=self.transl.copy(),
            global_orient=self.global_orient.copy(),
            body_pose=self.body_pose.copy(),
            betas=None if self.betas is None else self.betas.copy(),
            gender=self.gender,
            left_hand_pose=None if self.left_hand_pose is None else self.left_hand_pose.copy(),
            right_hand_pose=None if self.right_hand_pose is None else self.right_hand_pose.copy(),
            jaw_pose=None if self.jaw_pose is None else self.jaw_pose.copy(),
            leye_pose=None if self.leye_pose is None else self.leye_pose.copy(),
            reye_pose=None if self.reye_pose is None else self.reye_pose.copy(),
            expression=None if self.expression is None else self.expression.copy(),
            metadata=copy.deepcopy(self.metadata),
        )


def load_smpl_motion(
    path: str | Path,
    *,
    model_type: Literal["smpl", "smplx"] | None = None,
    fps: float | None = None,
    gender: str | None = None,
) -> SMPLMotion:
    path = Path(path)
    if path.suffix == ".npz":
        return _load_npz_motion(path, model_type=model_type, fps=fps, gender=gender)
    if path.suffix == ".npy":
        return _load_npy_motion(path, model_type=model_type or "smplx", fps=fps, gender=gender or "neutral")
    raise ValueError(f"Unsupported SMPL motion file extension {path.suffix!r}; expected .npz or .npy.")


def _load_npz_motion(
    path: Path,
    *,
    model_type: Literal["smpl", "smplx"] | None,
    fps: float | None,
    gender: str | None,
) -> SMPLMotion:
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        transl = _npz_first(data, "transl", "trans")
        global_orient = _npz_first(data, "global_orient", "root_orient")
        if global_orient is None and "poses" in data:
            global_orient = np.asarray(data["poses"])[:, 0:3]
        body_pose = _npz_first(data, "body_pose", "pose_body")
        if body_pose is None and "poses" in data:
            body_pose = np.asarray(data["poses"])[:, 3:66]
        if transl is None or global_orient is None or body_pose is None:
            raise ValueError(
                f"{path} must contain transl/trans, global_orient/root_orient, and body_pose/pose_body."
            )

        inferred_fps = fps if fps is not None else _optional_scalar(data, "mocap_frame_rate")
        if inferred_fps is None:
            raise ValueError(f"{path} does not contain mocap_frame_rate; pass fps explicitly.")

        surface_model_type = _optional_scalar(data, "surface_model_type")
        inferred_model_type = model_type or _normalize_model_type(surface_model_type) or "smplx"
        inferred_gender = gender or str(_optional_scalar(data, "gender") or "neutral").lower()

        pose_hand = _npz_first(data, "pose_hand")
        left_hand_pose = _npz_first(data, "left_hand_pose")
        right_hand_pose = _npz_first(data, "right_hand_pose")
        if pose_hand is not None and pose_hand.ndim == 2 and pose_hand.shape[1] >= 90:
            left_hand_pose = pose_hand[:, :45] if left_hand_pose is None else left_hand_pose
            right_hand_pose = pose_hand[:, 45:90] if right_hand_pose is None else right_hand_pose

        pose_eye = _npz_first(data, "pose_eye")
        leye_pose = _npz_first(data, "leye_pose", "left_eye_pose")
        reye_pose = _npz_first(data, "reye_pose", "right_eye_pose")
        if pose_eye is not None and pose_eye.ndim == 2 and pose_eye.shape[1] >= 6:
            leye_pose = pose_eye[:, :3] if leye_pose is None else leye_pose
            reye_pose = pose_eye[:, 3:6] if reye_pose is None else reye_pose

        motion = SMPLMotion(
            model_type=inferred_model_type,
            fps=float(inferred_fps),
            transl=np.asarray(transl, dtype=np.float64),
            global_orient=np.asarray(global_orient, dtype=np.float64),
            body_pose=np.asarray(body_pose, dtype=np.float64),
            betas=_optional_array(data, "betas"),
            gender=inferred_gender,
            left_hand_pose=_optional_float_array(left_hand_pose),
            right_hand_pose=_optional_float_array(right_hand_pose),
            jaw_pose=_optional_float_array(_npz_first(data, "jaw_pose", "pose_jaw")),
            leye_pose=_optional_float_array(leye_pose),
            reye_pose=_optional_float_array(reye_pose),
            expression=_optional_float_array(_npz_first(data, "expression", "expressions")),
            metadata={
                "source_path": str(path),
                "source_format": "npz",
                "source_keys": keys,
                "surface_model_type": surface_model_type,
            },
        )
    motion.validate()
    return motion


def _load_npy_motion(
    path: Path,
    *,
    model_type: Literal["smpl", "smplx"],
    fps: float | None,
    gender: str,
) -> SMPLMotion:
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] < 69:
        raise ValueError(f"PHUMA-style .npy motion must have shape [T, >=69], got {data.shape}.")
    if fps is None:
        raise ValueError(f"{path} is a .npy file and does not carry fps; pass fps explicitly.")

    motion = SMPLMotion(
        model_type=model_type,
        fps=float(fps),
        transl=np.asarray(data[:, 0:3], dtype=np.float64),
        global_orient=np.asarray(data[:, 3:6], dtype=np.float64),
        body_pose=np.asarray(data[:, 6:69], dtype=np.float64),
        gender=gender,
        metadata={
            "source_path": str(path),
            "source_format": "phuma_npy",
            "extra_columns": int(max(0, data.shape[1] - 69)),
        },
    )
    motion.validate()
    return motion


def _as_array(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf values.")
    return arr


def _npz_first(data, *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return None


def _optional_array(data, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    return np.asarray(data[key], dtype=np.float64)


def _optional_float_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)


def _optional_scalar(data, key: str):
    if key not in data:
        return None
    value = data[key]
    if np.asarray(value).shape == ():
        return np.asarray(value).item()
    return value


def _normalize_model_type(value) -> Literal["smpl", "smplx"] | None:
    if value is None:
        return None
    normalized = str(value).lower()
    if "smplx" in normalized or "smpl-x" in normalized:
        return "smplx"
    if "smpl" in normalized:
        return "smpl"
    return None

