from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .newton_backend import IKState, NewtonBackend
from .robot_spec import RobotSpec


JOINT_PRISMATIC = 0
JOINT_REVOLUTE = 1
JOINT_FIXED = 3
JOINT_FREE = 4
SUPPORTED_JOINT_TYPES = {JOINT_PRISMATIC, JOINT_REVOLUTE, JOINT_FIXED, JOINT_FREE}
DEFAULT_POSITION_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class TorchRobotFKResult:
    body_names: list[str]
    body_pos_w: torch.Tensor
    body_quat_xyzw: torch.Tensor


@dataclass(frozen=True)
class _NewtonKinematicTree:
    joint_parent: np.ndarray
    joint_child: np.ndarray
    joint_type: np.ndarray
    joint_q_start: np.ndarray
    joint_qd_start: np.ndarray
    joint_input_index: np.ndarray
    joint_X_p: np.ndarray
    joint_X_c: np.ndarray
    joint_axis: np.ndarray
    output_body_indices: np.ndarray
    body_count: int
    joint_labels: list[str]


class TorchRobotFK(torch.nn.Module):
    """Differentiable Torch forward kinematics matching Newton's tree FK."""

    def __init__(
        self,
        robot_spec: RobotSpec,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        backend: NewtonBackend | None = None,
    ):
        super().__init__()
        self.robot_spec = robot_spec
        self.body_names = list(robot_spec.body_names)
        self.num_dofs = robot_spec.num_dofs

        tree = _extract_newton_tree(robot_spec, backend=backend)
        self.body_count = tree.body_count
        self.joint_count = int(tree.joint_parent.shape[0])
        self.joint_labels = list(tree.joint_labels)

        unsupported = sorted(set(int(v) for v in tree.joint_type.tolist()) - SUPPORTED_JOINT_TYPES)
        if unsupported:
            raise ValueError(f"TorchRobotFK does not support Newton joint types: {unsupported}.")
        for idx, joint_type in enumerate(tree.joint_type.tolist()):
            if int(joint_type) == JOINT_FREE and (
                int(tree.joint_parent[idx]) >= 0 or int(tree.joint_q_start[idx]) != 0
            ):
                raise ValueError("TorchRobotFK supports only the root FREE joint through root_pos/root_quat_xyzw.")

        torch_device = torch.device(device) if device is not None else None
        self.register_buffer("joint_parent", _long_tensor(tree.joint_parent, torch_device), persistent=False)
        self.register_buffer("joint_child", _long_tensor(tree.joint_child, torch_device), persistent=False)
        self.register_buffer("joint_type", _long_tensor(tree.joint_type, torch_device), persistent=False)
        self.register_buffer("joint_q_start", _long_tensor(tree.joint_q_start, torch_device), persistent=False)
        self.register_buffer("joint_qd_start", _long_tensor(tree.joint_qd_start, torch_device), persistent=False)
        self.register_buffer("joint_input_index", _long_tensor(tree.joint_input_index, torch_device), persistent=False)
        self.register_buffer(
            "output_body_indices", _long_tensor(tree.output_body_indices, torch_device), persistent=False
        )

        self.register_buffer("joint_X_p_pos", _float_tensor(tree.joint_X_p[:, :3], dtype, torch_device), persistent=False)
        self.register_buffer("joint_X_p_quat", _float_tensor(tree.joint_X_p[:, 3:7], dtype, torch_device), persistent=False)
        self.register_buffer("joint_X_c_pos", _float_tensor(tree.joint_X_c[:, :3], dtype, torch_device), persistent=False)
        self.register_buffer("joint_X_c_quat", _float_tensor(tree.joint_X_c[:, 3:7], dtype, torch_device), persistent=False)
        self.register_buffer("joint_axis", _float_tensor(tree.joint_axis, dtype, torch_device), persistent=False)

        self._joint_parent = tuple(int(v) for v in tree.joint_parent.tolist())
        self._joint_child = tuple(int(v) for v in tree.joint_child.tolist())
        self._joint_type = tuple(int(v) for v in tree.joint_type.tolist())
        self._joint_qd_start = tuple(int(v) for v in tree.joint_qd_start.tolist())
        self._joint_input_index = tuple(int(v) for v in tree.joint_input_index.tolist())

    def forward(
        self,
        root_pos: torch.Tensor,
        root_quat_xyzw: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> TorchRobotFKResult:
        self._validate_inputs(root_pos, root_quat_xyzw, joint_pos)
        device = root_pos.device
        dtype = root_pos.dtype
        frames = int(root_pos.shape[0])

        root_quat = _quat_normalize(root_quat_xyzw)
        zero_pos = root_pos.new_zeros((frames, 3))
        identity_quat = _identity_quat(frames, dtype=dtype, device=device)

        x_p_pos = self.joint_X_p_pos.to(device=device, dtype=dtype)
        x_p_quat = _quat_normalize(self.joint_X_p_quat.to(device=device, dtype=dtype))
        x_c_pos = self.joint_X_c_pos.to(device=device, dtype=dtype)
        x_c_quat = _quat_normalize(self.joint_X_c_quat.to(device=device, dtype=dtype))
        joint_axis = self.joint_axis.to(device=device, dtype=dtype)
        output_body_indices = self.output_body_indices.to(device=device)

        body_pos: list[torch.Tensor | None] = [None] * self.body_count
        body_quat: list[torch.Tensor | None] = [None] * self.body_count

        for joint_idx in range(self.joint_count):
            parent_idx = self._joint_parent[joint_idx]
            child_idx = self._joint_child[joint_idx]
            joint_type = self._joint_type[joint_idx]

            x_pj_pos = _expand_time(x_p_pos[joint_idx], frames)
            x_pj_quat = _expand_time(x_p_quat[joint_idx], frames)
            if parent_idx >= 0:
                parent_pos = body_pos[parent_idx]
                parent_quat = body_quat[parent_idx]
                if parent_pos is None or parent_quat is None:
                    raise RuntimeError(f"Parent body {parent_idx} has not been evaluated before joint {joint_idx}.")
                x_wpj_pos, x_wpj_quat = _transform_multiply(parent_pos, parent_quat, x_pj_pos, x_pj_quat)
            else:
                x_wpj_pos, x_wpj_quat = x_pj_pos, x_pj_quat

            if joint_type == JOINT_FREE:
                x_j_pos, x_j_quat = root_pos, root_quat
            elif joint_type == JOINT_REVOLUTE:
                input_idx = self._joint_input_index[joint_idx]
                if input_idx < 0:
                    raise RuntimeError(f"Revolute joint {joint_idx} is not mapped to joint_pos.")
                axis = joint_axis[self._joint_qd_start[joint_idx]]
                x_j_pos = zero_pos
                x_j_quat = _quat_from_axis_angle(axis, joint_pos[:, input_idx])
            elif joint_type == JOINT_PRISMATIC:
                input_idx = self._joint_input_index[joint_idx]
                if input_idx < 0:
                    raise RuntimeError(f"Prismatic joint {joint_idx} is not mapped to joint_pos.")
                axis = joint_axis[self._joint_qd_start[joint_idx]]
                x_j_pos = axis.reshape(1, 3) * joint_pos[:, input_idx].reshape(frames, 1)
                x_j_quat = identity_quat
            elif joint_type == JOINT_FIXED:
                x_j_pos = zero_pos
                x_j_quat = identity_quat
            else:  # pragma: no cover - guarded in __init__.
                raise RuntimeError(f"Unsupported joint type {joint_type}.")

            x_wcj_pos, x_wcj_quat = _transform_multiply(x_wpj_pos, x_wpj_quat, x_j_pos, x_j_quat)
            x_cj_inv_pos, x_cj_inv_quat = _transform_inverse(
                _expand_time(x_c_pos[joint_idx], frames),
                _expand_time(x_c_quat[joint_idx], frames),
            )
            child_pos, child_quat = _transform_multiply(x_wcj_pos, x_wcj_quat, x_cj_inv_pos, x_cj_inv_quat)
            body_pos[child_idx] = child_pos
            body_quat[child_idx] = child_quat

        if any(value is None for value in body_pos) or any(value is None for value in body_quat):
            missing = [idx for idx, value in enumerate(body_pos) if value is None]
            raise RuntimeError(f"TorchRobotFK did not evaluate all Newton bodies; missing indices: {missing}.")

        all_body_pos = torch.stack([value for value in body_pos if value is not None], dim=1)
        all_body_quat = torch.stack([value for value in body_quat if value is not None], dim=1)
        return TorchRobotFKResult(
            body_names=list(self.body_names),
            body_pos_w=all_body_pos.index_select(1, output_body_indices),
            body_quat_xyzw=_quat_normalize(all_body_quat.index_select(1, output_body_indices)),
        )

    def _validate_inputs(
        self,
        root_pos: torch.Tensor,
        root_quat_xyzw: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> None:
        if not isinstance(root_pos, torch.Tensor):
            raise TypeError("root_pos must be a torch.Tensor.")
        if not isinstance(root_quat_xyzw, torch.Tensor):
            raise TypeError("root_quat_xyzw must be a torch.Tensor.")
        if not isinstance(joint_pos, torch.Tensor):
            raise TypeError("joint_pos must be a torch.Tensor.")
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"root_pos must have shape [T, 3], got {tuple(root_pos.shape)}.")
        if root_quat_xyzw.ndim != 2 or root_quat_xyzw.shape[1] != 4:
            raise ValueError(f"root_quat_xyzw must have shape [T, 4], got {tuple(root_quat_xyzw.shape)}.")
        if joint_pos.ndim != 2 or joint_pos.shape[1] != self.num_dofs:
            raise ValueError(f"joint_pos must have shape [T, {self.num_dofs}], got {tuple(joint_pos.shape)}.")
        if root_pos.shape[0] != root_quat_xyzw.shape[0] or root_pos.shape[0] != joint_pos.shape[0]:
            raise ValueError("root_pos, root_quat_xyzw, and joint_pos must have the same T dimension.")
        if root_pos.device != root_quat_xyzw.device or root_pos.device != joint_pos.device:
            raise ValueError("root_pos, root_quat_xyzw, and joint_pos must be on the same device.")
        if root_pos.dtype != root_quat_xyzw.dtype or root_pos.dtype != joint_pos.dtype:
            raise ValueError("root_pos, root_quat_xyzw, and joint_pos must have the same dtype.")
        if not torch.is_floating_point(root_pos):
            raise TypeError("root_pos, root_quat_xyzw, and joint_pos must use a floating point dtype.")


def max_position_error_against_newton(
    robot_spec: RobotSpec,
    root_pos: torch.Tensor | np.ndarray,
    root_quat_xyzw: torch.Tensor | np.ndarray,
    joint_pos: torch.Tensor | np.ndarray,
    *,
    pos_tol: float | None = None,
    torch_fk: TorchRobotFK | None = None,
    newton_backend: NewtonBackend | None = None,
) -> float:
    """Return max per-body Euclidean position error and raise if it exceeds pos_tol."""

    effective_tol = _position_tolerance(pos_tol)
    backend = newton_backend or NewtonBackend(robot_spec)
    fk = torch_fk or TorchRobotFK(robot_spec, dtype=torch.float64, backend=backend)

    root_pos_t = _as_torch(root_pos)
    root_quat_t = _as_torch(root_quat_xyzw, dtype=root_pos_t.dtype, device=root_pos_t.device)
    joint_pos_t = _as_torch(joint_pos, dtype=root_pos_t.dtype, device=root_pos_t.device)
    with torch.no_grad():
        torch_result = fk(root_pos_t, root_quat_t, joint_pos_t)

    root_pos_np = _as_numpy(root_pos_t)
    root_quat_np = _as_numpy(root_quat_t)
    joint_pos_np = _as_numpy(joint_pos_t)
    reference_pos = []
    for frame_idx in range(root_pos_np.shape[0]):
        state = IKState(
            root_pos_w=root_pos_np[frame_idx],
            root_quat_xyzw=root_quat_np[frame_idx],
            joint_pos=joint_pos_np[frame_idx],
        )
        reference_pos.append(backend.forward_kinematics(state).body_pos_w)
    reference = np.stack(reference_pos, axis=0)
    predicted = _as_numpy(torch_result.body_pos_w)
    max_error = float(np.linalg.norm(predicted - reference, axis=-1).max(initial=0.0))
    if max_error > effective_tol:
        raise AssertionError(
            f"TorchRobotFK max position error {max_error:.6g} exceeds tolerance {effective_tol:.6g}."
        )
    return max_error


def _extract_newton_tree(robot_spec: RobotSpec, *, backend: NewtonBackend | None = None) -> _NewtonKinematicTree:
    backend = backend or NewtonBackend(robot_spec)
    model = backend.model
    body_name_to_newton_index = dict(getattr(backend, "_body_name_to_newton_index", {}) or {})
    if not body_name_to_newton_index:
        body_name_to_newton_index = _body_name_map_from_labels(getattr(model, "body_label", []))

    missing_bodies = [name for name in robot_spec.body_names if name not in body_name_to_newton_index]
    if missing_bodies:
        raise RuntimeError(f"Newton model is missing RobotSpec bodies: {missing_bodies}.")
    output_body_indices = np.asarray([body_name_to_newton_index[name] for name in robot_spec.body_names], dtype=np.int64)

    joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int64)
    joint_q_start = np.asarray(model.joint_q_start.numpy(), dtype=np.int64)
    joint_qd_start = np.asarray(model.joint_qd_start.numpy(), dtype=np.int64)
    joint_input_index = np.full_like(joint_q_start, -1, dtype=np.int64)

    q_offset = 7 if robot_spec.floating_base else 0
    for joint_idx, joint_kind in enumerate(joint_type.tolist()):
        if int(joint_kind) not in {JOINT_PRISMATIC, JOINT_REVOLUTE}:
            continue
        input_idx = int(joint_q_start[joint_idx]) - q_offset
        if input_idx < 0 or input_idx >= robot_spec.num_dofs:
            raise RuntimeError(
                f"Newton joint {joint_idx} cannot be mapped to RobotSpec joint_pos index {input_idx}."
            )
        joint_input_index[joint_idx] = input_idx

    _validate_actuated_joint_order(robot_spec, model, joint_type, joint_q_start, q_offset)

    return _NewtonKinematicTree(
        joint_parent=np.asarray(model.joint_parent.numpy(), dtype=np.int64),
        joint_child=np.asarray(model.joint_child.numpy(), dtype=np.int64),
        joint_type=joint_type,
        joint_q_start=joint_q_start,
        joint_qd_start=joint_qd_start,
        joint_input_index=joint_input_index,
        joint_X_p=np.asarray(model.joint_X_p.numpy(), dtype=np.float64),
        joint_X_c=np.asarray(model.joint_X_c.numpy(), dtype=np.float64),
        joint_axis=np.asarray(model.joint_axis.numpy(), dtype=np.float64),
        output_body_indices=output_body_indices,
        body_count=int(model.body_count),
        joint_labels=[str(label) for label in getattr(model, "joint_label", [])],
    )


def _validate_actuated_joint_order(
    robot_spec: RobotSpec,
    model: Any,
    joint_type: np.ndarray,
    joint_q_start: np.ndarray,
    q_offset: int,
) -> None:
    labels = [str(label).rstrip("/").split("/")[-1] for label in getattr(model, "joint_label", [])]
    indexed_names: list[tuple[int, str]] = []
    for joint_idx, joint_kind in enumerate(joint_type.tolist()):
        if int(joint_kind) not in {JOINT_PRISMATIC, JOINT_REVOLUTE}:
            continue
        input_idx = int(joint_q_start[joint_idx]) - q_offset
        if 0 <= input_idx < robot_spec.num_dofs and joint_idx < len(labels):
            indexed_names.append((input_idx, labels[joint_idx]))
    newton_order = [name for _, name in sorted(indexed_names)]
    if newton_order != robot_spec.actuated_joints:
        raise RuntimeError(
            "RobotSpec actuated_joints must match Newton coordinate order. "
            f"RobotSpec={robot_spec.actuated_joints}, Newton={newton_order}."
        )


def _body_name_map_from_labels(labels: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, label in enumerate(labels):
        name = str(label).rstrip("/").split("/")[-1]
        mapping[name] = idx
    return mapping


def _position_tolerance(pos_tol: float | None) -> float:
    raw = os.environ.get("RETARGETER_TORCH_FK_POS_TOL", str(DEFAULT_POSITION_TOLERANCE)) if pos_tol is None else pos_tol
    value = float(raw)
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError(f"pos_tol must be positive and finite, got {raw!r}.")
    return value


def _as_torch(
    value: torch.Tensor | np.ndarray,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        target_device = value.device if device is None else device
        return value.to(dtype=dtype, device=target_device)
    if device is None:
        return torch.as_tensor(value, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _long_tensor(array: np.ndarray, device: torch.device | None) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.long, device=device)


def _float_tensor(array: np.ndarray, dtype: torch.dtype, device: torch.device | None) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def _expand_time(value: torch.Tensor, frames: int) -> torch.Tensor:
    return value.reshape(1, -1).expand(frames, -1)


def _identity_quat(frames: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    quat = torch.zeros((frames, 4), dtype=dtype, device=device)
    quat[:, 3] = 1.0
    return quat


def _quat_normalize(quat: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(eps)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    xyz = -quat[..., :3]
    w = quat[..., 3:4]
    return torch.cat((xyz, w), dim=-1)


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = lhs.unbind(dim=-1)
    x2, y2, z2, w2 = rhs.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat[..., :3]
    quat_w = quat[..., 3:4]
    uv = torch.cross(quat_xyz, vector, dim=-1)
    uuv = torch.cross(quat_xyz, uv, dim=-1)
    return vector + 2.0 * (quat_w * uv + uuv)


def _quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half_angle = 0.5 * angle
    sin_half = torch.sin(half_angle).reshape(-1, 1)
    quat_xyz = axis.reshape(1, 3) * sin_half
    quat_w = torch.cos(half_angle).reshape(-1, 1)
    return _quat_normalize(torch.cat((quat_xyz, quat_w), dim=-1))


def _transform_multiply(
    lhs_pos: torch.Tensor,
    lhs_quat: torch.Tensor,
    rhs_pos: torch.Tensor,
    rhs_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = lhs_pos + _quat_rotate(lhs_quat, rhs_pos)
    quat = _quat_normalize(_quat_multiply(lhs_quat, rhs_quat))
    return pos, quat


def _transform_inverse(pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quat_inv = _quat_conjugate(_quat_normalize(quat))
    pos_inv = _quat_rotate(quat_inv, -pos)
    return pos_inv, quat_inv
