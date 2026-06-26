from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from retargeter.preprocess import CanonicalHumanMotion
from retargeter.preprocess.lowpass import normalize_quat_xyzw


REQUIRED_SCALER_CONFIG_SECTIONS = [
    "robot",
    "human_root_name",
    "robot_root_name",
    "human_height_assumption",
    "model_height",
    "scales",
    "parents",
    "body_map",
    "offsets",
    "rotation_offsets_xyzw",
]


class HumanToRobotScaler:
    def __init__(self, scaler_config_path: Path | str):
        self.scaler_config_path = Path(scaler_config_path)
        self.config = _load_yaml(self.scaler_config_path)
        _validate_scaler_config(self.config, self.scaler_config_path)

        self.robot = str(self.config["robot"])
        self.human_root_name = str(self.config["human_root_name"])
        self.robot_root_name = str(self.config["robot_root_name"])
        self.human_height_assumption = float(self.config["human_height_assumption"])
        self.model_height = float(self.config["model_height"])
        self.scale_ratio = self.model_height / self.human_height_assumption
        self.scales = {
            name: float(scale) * self.scale_ratio
            for name, scale in self.config["scales"].items()
        }
        self.parents = dict(self.config["parents"])
        self.body_map = dict(self.config["body_map"])
        self.offsets = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in self.config["offsets"].items()
        }
        self.rotation_offsets_xyzw = {
            name: normalize_quat_xyzw(np.asarray(value, dtype=np.float64))
            for name, value in self.config["rotation_offsets_xyzw"].items()
        }

    def scale_motion(self, motion: CanonicalHumanMotion) -> CanonicalHumanMotion:
        motion.validate()
        self._validate_motion_bodies(motion.body_names)
        scaled_pos, scaled_quat = self.scale_frame(motion.body_pos_w, motion.body_quat_xyzw, motion.body_names)
        scaled = CanonicalHumanMotion(
            fps=motion.fps,
            body_names=list(motion.body_names),
            body_pos_w=scaled_pos,
            body_quat_xyzw=scaled_quat,
            vertices_w=None if motion.vertices_w is None else motion.vertices_w.copy(),
            metadata=copy.deepcopy(motion.metadata),
        )
        scaled.metadata["scale"] = {
            "config_path": str(self.scaler_config_path),
            "robot": self.robot,
            "human_root_name": self.human_root_name,
            "robot_root_name": self.robot_root_name,
            "human_height_assumption": self.human_height_assumption,
            "model_height": self.model_height,
            "scale_ratio": self.scale_ratio,
        }
        scaled.validate()
        return scaled

    def scale_frame(
        self,
        body_pos_w: np.ndarray,
        body_quat_xyzw: np.ndarray,
        body_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        body_pos_w = np.asarray(body_pos_w, dtype=np.float64)
        body_quat_xyzw = np.asarray(body_quat_xyzw, dtype=np.float64)
        if body_pos_w.ndim == 2:
            single_frame = True
            pos = body_pos_w[None, ...]
            quat = body_quat_xyzw[None, ...]
        elif body_pos_w.ndim == 3:
            single_frame = False
            pos = body_pos_w
            quat = body_quat_xyzw
        else:
            raise ValueError(f"body_pos_w must have shape [H, 3] or [T, H, 3], got {body_pos_w.shape}.")

        if quat.shape != pos.shape[:2] + (4,):
            raise ValueError(f"body_quat_xyzw must match position shape [T, H, 4], got {quat.shape}.")
        if len(body_names) != pos.shape[1]:
            raise ValueError(f"body_names length {len(body_names)} does not match H={pos.shape[1]}.")

        self._validate_motion_bodies(body_names)
        root_index = body_names.index(self.human_root_name)
        root_pos = pos[:, root_index : root_index + 1, :]
        root_scale = float(self.scales.get(self.human_root_name, 1.0))
        scaled_root_pos = root_pos[:, 0, :] * root_scale

        scaled_pos = pos.copy()
        scaled_quat = normalize_quat_xyzw(quat.copy())
        for body_index, body_name in enumerate(body_names):
            body_scale = float(self.scales.get(body_name, 1.0))
            scaled_pos[:, body_index, :] = scaled_root_pos + (pos[:, body_index, :] - root_pos[:, 0, :]) * body_scale

            rotation_offset = self.rotation_offsets_xyzw.get(body_name)
            if rotation_offset is not None:
                scaled_quat[:, body_index, :] = normalize_quat_xyzw(
                    quat_multiply_xyzw(scaled_quat[:, body_index, :], rotation_offset)
                )

            offset = self.offsets.get(body_name)
            if offset is not None:
                scaled_pos[:, body_index, :] += rotate_vectors_xyzw(scaled_quat[:, body_index, :], offset)

        if single_frame:
            return scaled_pos[0], scaled_quat[0]
        return scaled_pos, scaled_quat

    def required_robot_body_names(self) -> list[str]:
        names = []
        for entry in self.body_map.values():
            robot_name = entry["robot"]
            if robot_name not in names:
                names.append(robot_name)
        return names

    def required_human_body_names(self) -> list[str]:
        names = set(self.scales.keys()) | set(self.parents.keys()) | {self.human_root_name}
        for entry in self.body_map.values():
            names.add(entry["human"])
        return sorted(names)

    def _validate_motion_bodies(self, body_names: list[str]) -> None:
        missing = [name for name in self.required_human_body_names() if name not in body_names]
        if missing:
            raise ValueError(f"Missing required semantic human bodies for scaling: {missing}.")


def quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(np.broadcast_to(b, a.shape), -1, 0)
    out = np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )
    return out


def quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    out = quat.copy()
    out[..., :3] *= -1.0
    return out


def rotate_vectors_xyzw(quat: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = normalize_quat_xyzw(np.asarray(quat, dtype=np.float64))
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape == (3,):
        vectors = np.broadcast_to(vectors, quat.shape[:-1] + (3,))
    vector_quat = np.concatenate([vectors, np.zeros(vectors.shape[:-1] + (1,), dtype=np.float64)], axis=-1)
    rotated = quat_multiply_xyzw(quat_multiply_xyzw(quat, vector_quat), quat_conjugate_xyzw(quat))
    return rotated[..., :3]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load scale configs.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Scale config {path} must contain a YAML mapping.")
    return data


def _validate_scaler_config(config: dict[str, Any], path: Path) -> None:
    missing = [section for section in REQUIRED_SCALER_CONFIG_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Scale config {path} is missing required sections: {missing}.")

    if not isinstance(config["body_map"], dict) or not config["body_map"]:
        raise ValueError(f"Scale config {path} body_map must be a non-empty mapping.")
    for semantic_name, entry in config["body_map"].items():
        if not isinstance(entry, dict) or "human" not in entry or "robot" not in entry:
            raise ValueError(f"body_map entry {semantic_name!r} must contain human and robot names.")

    for name, offset in config["offsets"].items():
        if np.asarray(offset, dtype=np.float64).shape != (3,):
            raise ValueError(f"offsets.{name} must have shape [3].")
    for name, quat in config["rotation_offsets_xyzw"].items():
        if np.asarray(quat, dtype=np.float64).shape != (4,):
            raise ValueError(f"rotation_offsets_xyzw.{name} must have shape [4].")
