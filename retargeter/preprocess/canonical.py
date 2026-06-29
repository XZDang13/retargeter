from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


REQUIRED_CANONICAL_BODY_NAMES = [
    "pelvis",
    "chest",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand",
    "right_hand",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
    "left_toe",
    "right_toe",
    "left_heel",
    "right_heel",
]


@dataclass
class CanonicalHumanMotion:
    fps: float
    body_names: list[str]
    body_pos_w: np.ndarray
    body_quat_xyzw: np.ndarray
    vertices_w: np.ndarray | None = None
    mesh_faces: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def num_frames(self) -> int:
        return int(self.body_pos_w.shape[0])

    def num_bodies(self) -> int:
        return int(len(self.body_names))

    def get_body_index(self, name: str) -> int:
        try:
            return self.body_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Body {name!r} is not present in motion.") from exc

    def get_body_pos(self, name: str) -> np.ndarray:
        return self.body_pos_w[:, self.get_body_index(name), :]

    def get_body_quat(self, name: str) -> np.ndarray:
        return self.body_quat_xyzw[:, self.get_body_index(name), :]

    def set_body_pos(self, name: str, pos: np.ndarray) -> None:
        pos = np.asarray(pos)
        expected = (self.num_frames(), 3)
        if pos.shape != expected:
            raise ValueError(f"Position for {name!r} must have shape {expected}, got {pos.shape}.")
        if not np.all(np.isfinite(pos)):
            raise ValueError(f"Position for {name!r} contains NaN or inf values.")
        self.body_pos_w[:, self.get_body_index(name), :] = pos

    def set_body_quat(self, name: str, quat: np.ndarray) -> None:
        quat = np.asarray(quat)
        expected = (self.num_frames(), 4)
        if quat.shape != expected:
            raise ValueError(f"Quaternion for {name!r} must have shape {expected}, got {quat.shape}.")
        if not np.all(np.isfinite(quat)):
            raise ValueError(f"Quaternion for {name!r} contains NaN or inf values.")
        self.body_quat_xyzw[:, self.get_body_index(name), :] = quat

    def validate(self, required_bodies: list[str] | None = None) -> None:
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"fps must be a positive finite value, got {self.fps!r}.")
        if len(self.body_names) != len(set(self.body_names)):
            raise ValueError("body_names must be unique.")

        pos = np.asarray(self.body_pos_w)
        quat = np.asarray(self.body_quat_xyzw)
        if pos.ndim != 3 or pos.shape[2] != 3:
            raise ValueError(f"body_pos_w must have shape [T, H, 3], got {pos.shape}.")
        if quat.ndim != 3 or quat.shape[2] != 4:
            raise ValueError(f"body_quat_xyzw must have shape [T, H, 4], got {quat.shape}.")
        if pos.shape[:2] != quat.shape[:2]:
            raise ValueError(f"body_pos_w and body_quat_xyzw must share [T, H], got {pos.shape} and {quat.shape}.")
        if pos.shape[1] != len(self.body_names):
            raise ValueError(
                f"body_names length must match body dimension H={pos.shape[1]}, got {len(self.body_names)}."
            )
        if not np.all(np.isfinite(pos)):
            raise ValueError("body_pos_w contains NaN or inf values.")
        if not np.all(np.isfinite(quat)):
            raise ValueError("body_quat_xyzw contains NaN or inf values.")

        if self.vertices_w is not None:
            vertices = np.asarray(self.vertices_w)
            if vertices.ndim != 3 or vertices.shape[0] != pos.shape[0] or vertices.shape[2] != 3:
                raise ValueError(f"vertices_w must have shape [T, V, 3] matching T={pos.shape[0]}, got {vertices.shape}.")
            if not np.all(np.isfinite(vertices)):
                raise ValueError("vertices_w contains NaN or inf values.")

        if self.mesh_faces is not None:
            faces = np.asarray(self.mesh_faces)
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(f"mesh_faces must have shape [F, 3], got {faces.shape}.")
            if not np.issubdtype(faces.dtype, np.integer):
                raise ValueError("mesh_faces must contain integer vertex indices.")
            if faces.size and np.min(faces) < 0:
                raise ValueError("mesh_faces contains negative vertex indices.")
            if self.vertices_w is not None and faces.size and np.max(faces) >= self.vertices_w.shape[1]:
                raise ValueError("mesh_faces references vertices outside vertices_w.")

        if required_bodies is not None:
            missing = [name for name in required_bodies if name not in self.body_names]
            if missing:
                raise ValueError(f"Missing required bodies: {missing}.")

    def copy(self) -> "CanonicalHumanMotion":
        return CanonicalHumanMotion(
            fps=float(self.fps),
            body_names=list(self.body_names),
            body_pos_w=self.body_pos_w.copy(),
            body_quat_xyzw=self.body_quat_xyzw.copy(),
            vertices_w=None if self.vertices_w is None else self.vertices_w.copy(),
            mesh_faces=None if self.mesh_faces is None else self.mesh_faces.copy(),
            metadata=copy.deepcopy(self.metadata),
        )
