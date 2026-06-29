from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .canonical import CanonicalHumanMotion, REQUIRED_CANONICAL_BODY_NAMES
from .smpl_motion import SMPLMotion

try:
    import torch
except ImportError:  # pragma: no cover - SMPL FK requires torch at runtime.
    torch = None

try:
    import smplx
except ImportError:  # pragma: no cover - tested through clear runtime error.
    smplx = None


DEFAULT_SMPLX_BODY_MAPPING = {
    "pelvis": 0,
    "chest": 9,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_hand": 20,
    "right_hand": 21,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
    "left_toe": 60,
    "right_toe": 63,
    "left_heel": 62,
    "right_heel": 65,
}

ORIENTATION_SOURCE_JOINT = {
    "pelvis": 0,
    "chest": 9,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_hand": 20,
    "right_hand": 21,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
    "left_toe": 10,
    "right_toe": 11,
    "left_heel": 10,
    "right_heel": 11,
}

FALLBACK_SMPLX_PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)


class SMPLForwardKinematics:
    def __init__(
        self,
        model_dir: Path | str = Path("assets/body_models"),
        model_type: Literal["smpl", "smplx"] = "smplx",
        gender: str = "neutral",
        device: str = "cpu",
        body_mapping: dict | None = None,
        foot_vertex_config: dict | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.model_type = model_type
        self.gender = gender
        self.device = device
        self.body_mapping = dict(DEFAULT_SMPLX_BODY_MAPPING if body_mapping is None else body_mapping)
        self.foot_vertex_config = foot_vertex_config or {}
        self.model_root = self._resolve_model_root()
        self.model = self._create_model()

    def forward(self, motion: SMPLMotion, return_vertices: bool = True) -> CanonicalHumanMotion:
        motion.validate()
        if motion.model_type != self.model_type:
            raise ValueError(f"FK was created for {self.model_type!r}, but motion has {motion.model_type!r}.")
        if torch is None:
            raise RuntimeError("torch is required for SMPL forward kinematics.")

        params = self._motion_to_torch_params(motion)
        with torch.no_grad():
            output = self.model(**params, return_verts=return_vertices)

        joints = output.joints.detach().cpu().numpy()
        vertices = None
        if return_vertices and getattr(output, "vertices", None) is not None:
            vertices = output.vertices.detach().cpu().numpy()
        mesh_faces = None
        if vertices is not None and getattr(self.model, "faces", None) is not None:
            mesh_faces = np.asarray(self.model.faces, dtype=np.int32).copy()

        global_joint_quats = self._global_joint_quats_xyzw(motion)
        body_pos = np.stack([joints[:, self._body_index(name), :] for name in REQUIRED_CANONICAL_BODY_NAMES], axis=1)
        body_quat = np.stack(
            [global_joint_quats[:, self._orientation_source_joint(name), :] for name in REQUIRED_CANONICAL_BODY_NAMES],
            axis=1,
        )

        if _metadata_says_y_up(motion.metadata):
            body_pos = _convert_points_yup_to_zup(body_pos)
            if vertices is not None:
                vertices = _convert_points_yup_to_zup(vertices)
            body_quat = _convert_quats_yup_to_zup(body_quat)

        canonical = CanonicalHumanMotion(
            fps=motion.fps,
            body_names=list(REQUIRED_CANONICAL_BODY_NAMES),
            body_pos_w=body_pos,
            body_quat_xyzw=body_quat,
            vertices_w=vertices,
            mesh_faces=mesh_faces,
            metadata={
                "source_model_type": motion.model_type,
                "source_gender": motion.gender,
                "quaternion_convention": "xyzw",
                "world_frame": "z_up",
                "foot_vertex_config": self.foot_vertex_config,
            },
        )
        canonical.validate(required_bodies=REQUIRED_CANONICAL_BODY_NAMES)
        return canonical

    def _resolve_model_root(self) -> Path:
        direct_dir = self.model_dir
        nested_dir = self.model_dir / self.model_type
        if nested_dir.exists():
            return self.model_dir
        if direct_dir.name.lower() == self.model_type and direct_dir.exists():
            return direct_dir.parent
        raise FileNotFoundError(
            f"Could not find {self.model_type.upper()} model files. Expected either "
            f"{nested_dir} or a direct {self.model_type} directory at {direct_dir}."
        )

    def _create_model(self):
        if smplx is None:
            raise RuntimeError("The smplx Python package is required for SMPL forward kinematics.")
        try:
            model = smplx.create(
                str(self.model_root),
                model_type=self.model_type,
                gender=self.gender,
                num_pca_comps=45,
                batch_size=1,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create {self.model_type.upper()} model from {self.model_root}. "
                f"Expected model files under {self.model_root / self.model_type}."
            ) from exc
        if hasattr(model, "to"):
            model = model.to(self.device)
        return model

    def _motion_to_torch_params(self, motion: SMPLMotion) -> dict:
        num_frames = motion.num_frames()
        params = {
            "transl": _to_tensor(motion.transl, self.device),
            "global_orient": _to_tensor(motion.global_orient, self.device),
            "body_pose": _to_tensor(motion.body_pose, self.device),
            "betas": _to_tensor(_fit_betas(motion.betas, num_frames, self._num_betas()), self.device),
        }
        optional_map = {
            "left_hand_pose": motion.left_hand_pose,
            "right_hand_pose": motion.right_hand_pose,
            "jaw_pose": motion.jaw_pose,
            "leye_pose": motion.leye_pose,
            "reye_pose": motion.reye_pose,
            "expression": motion.expression,
        }
        for key, value in optional_map.items():
            if value is not None:
                params[key] = _to_tensor(value, self.device)

        if self.model_type == "smplx":
            params.setdefault("left_hand_pose", _to_tensor(np.zeros((num_frames, 45)), self.device))
            params.setdefault("right_hand_pose", _to_tensor(np.zeros((num_frames, 45)), self.device))
            params.setdefault("jaw_pose", _to_tensor(np.zeros((num_frames, 3)), self.device))
            params.setdefault("leye_pose", _to_tensor(np.zeros((num_frames, 3)), self.device))
            params.setdefault("reye_pose", _to_tensor(np.zeros((num_frames, 3)), self.device))
            params.setdefault(
                "expression",
                _to_tensor(np.zeros((num_frames, self._num_expression_coeffs())), self.device),
            )
        return params

    def _num_betas(self) -> int:
        return int(getattr(self.model, "num_betas", 10))

    def _num_expression_coeffs(self) -> int:
        return int(getattr(self.model, "num_expression_coeffs", 10))

    def _body_index(self, name: str) -> int:
        value = self.body_mapping[name]
        if isinstance(value, str):
            try:
                from smplx.joint_names import JOINT_NAMES
            except Exception as exc:
                raise RuntimeError("String body mappings require smplx.joint_names.") from exc
            return int(JOINT_NAMES.index(value))
        return int(value)

    def _orientation_source_joint(self, name: str) -> int:
        return int(ORIENTATION_SOURCE_JOINT[name])

    def _global_joint_quats_xyzw(self, motion: SMPLMotion) -> np.ndarray:
        parents = self._joint_parents()
        joint_count = int(max(max(ORIENTATION_SOURCE_JOINT.values()) + 1, parents.shape[0]))
        if parents.shape[0] < joint_count:
            padded = np.full(joint_count, -1, dtype=np.int64)
            padded[: parents.shape[0]] = parents
            parents = padded
        else:
            parents = parents[:joint_count]

        local = np.zeros((motion.num_frames(), joint_count, 4), dtype=np.float64)
        local[..., 3] = 1.0
        local[:, 0, :] = _axis_angle_to_quat_xyzw(motion.global_orient)
        body_joint_count = min(joint_count - 1, motion.body_pose.shape[1] // 3)
        for joint_idx in range(1, body_joint_count + 1):
            start = (joint_idx - 1) * 3
            local[:, joint_idx, :] = _axis_angle_to_quat_xyzw(motion.body_pose[:, start : start + 3])

        global_quat = local.copy()
        for joint_idx in range(1, joint_count):
            parent_idx = int(parents[joint_idx]) if joint_idx < parents.shape[0] else -1
            if 0 <= parent_idx < joint_idx:
                global_quat[:, joint_idx, :] = _normalize_quats_xyzw(
                    _quat_multiply_xyzw(global_quat[:, parent_idx, :], local[:, joint_idx, :])
                )
        return _normalize_quats_xyzw(global_quat)

    def _joint_parents(self) -> np.ndarray:
        parents = getattr(self.model, "parents", None)
        if parents is None:
            return FALLBACK_SMPLX_PARENTS.copy()
        if hasattr(parents, "detach"):
            parents = parents.detach().cpu().numpy()
        else:
            parents = np.asarray(parents)
        if parents.size == 0:
            return FALLBACK_SMPLX_PARENTS.copy()
        return parents.astype(np.int64, copy=True)


def _to_tensor(array: np.ndarray, device: str):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _fit_betas(betas: np.ndarray | None, num_frames: int, num_betas: int) -> np.ndarray:
    if betas is None:
        return np.zeros((num_frames, num_betas), dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = np.tile(betas[None, :], (num_frames, 1))
    elif betas.shape[0] == 1:
        betas = np.tile(betas, (num_frames, 1))
    elif betas.shape[0] != num_frames:
        raise ValueError(f"betas must be shape [B], [1, B], or [T, B], got {betas.shape}.")

    if betas.shape[1] < num_betas:
        betas = np.pad(betas, ((0, 0), (0, num_betas - betas.shape[1])))
    return betas[:, :num_betas]


def _axis_angle_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.divide(np.sin(half), angle, out=np.full_like(angle, 0.5), where=angle > 1e-12)
    xyz = rotvec * scale
    w = np.cos(half)
    quat = np.concatenate([xyz, w], axis=-1)
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(np.broadcast_to(b, a.shape), -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _normalize_quats_xyzw(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float64)
    norm = np.linalg.norm(quats, axis=-1, keepdims=True)
    return np.divide(quats, norm, out=np.zeros_like(quats), where=norm > 1e-12)


def _metadata_says_y_up(metadata: dict) -> bool:
    for key in ("world_frame", "coordinate_system", "up_axis"):
        value = metadata.get(key)
        if value is not None and str(value).lower() in {"y_up", "y-up", "y", "+y"}:
            return True
    return False


def _convert_points_yup_to_zup(points: np.ndarray) -> np.ndarray:
    converted = points.copy()
    converted[..., 0] = points[..., 0]
    converted[..., 1] = -points[..., 2]
    converted[..., 2] = points[..., 1]
    return converted


def _convert_quats_yup_to_zup(quats: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        return quats
    converter = R.from_euler("x", 90.0, degrees=True).as_matrix()
    flat = quats.reshape(-1, 4)
    matrices = R.from_quat(flat).as_matrix()
    converted = converter @ matrices @ converter.T
    return R.from_matrix(converted).as_quat().reshape(quats.shape)
