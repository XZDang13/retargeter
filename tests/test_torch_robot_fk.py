from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from retargeter.newton import (
    IKState,
    NewtonBackend,
    RobotSpec,
    RetargetedMotion,
    TorchRobotFK,
    max_position_error_against_newton,
)


G1_29_ROBOT = Path("retargeter/newton/configs/g1_29_robot.yaml")
G1_23_ROBOT = Path("retargeter/newton/configs/g1_23_robot.yaml")


def test_torch_robot_fk_matches_newton_on_random_g1_29_poses():
    spec = _load_real_newton_spec(G1_29_ROBOT)
    root_pos, root_quat, joint_pos = _random_retarget_inputs(spec, frames=6, seed=29)

    error = max_position_error_against_newton(spec, root_pos, root_quat, joint_pos, pos_tol=_position_tolerance())

    assert error <= _position_tolerance()


def test_torch_robot_fk_matches_newton_on_random_g1_23_poses():
    spec = _load_real_newton_spec(G1_23_ROBOT)
    root_pos, root_quat, joint_pos = _random_retarget_inputs(spec, frames=5, seed=23)

    error = max_position_error_against_newton(spec, root_pos, root_quat, joint_pos, pos_tol=_position_tolerance())

    assert error <= _position_tolerance()


def test_torch_robot_fk_is_differentiable():
    spec = _load_real_newton_spec(G1_29_ROBOT)
    root_pos, root_quat, joint_pos = _random_retarget_inputs(spec, frames=4, seed=7)
    fk = TorchRobotFK(spec, dtype=torch.float64)
    root_pos_t = torch.tensor(root_pos, dtype=torch.float64, requires_grad=True)
    root_quat_t = torch.tensor(root_quat, dtype=torch.float64, requires_grad=True)
    joint_pos_t = torch.tensor(joint_pos, dtype=torch.float64, requires_grad=True)

    result = fk(root_pos_t, root_quat_t, joint_pos_t)
    loss = result.body_pos_w.square().mean() + result.body_quat_xyzw.square().mean()
    loss.backward()

    assert root_pos_t.grad is not None
    assert root_quat_t.grad is not None
    assert joint_pos_t.grad is not None
    assert torch.isfinite(root_pos_t.grad).all()
    assert torch.isfinite(root_quat_t.grad).all()
    assert torch.isfinite(joint_pos_t.grad).all()


def test_torch_robot_fk_matches_retargeted_motion_reference_body_positions():
    spec = _load_real_newton_spec(G1_29_ROBOT)
    root_pos, root_quat, joint_pos = _random_retarget_inputs(spec, frames=4, seed=11)
    backend = NewtonBackend(spec)
    body_pos_w, body_quat_xyzw = _newton_body_sequence(backend, root_pos, root_quat, joint_pos)
    retargeted_motion = RetargetedMotion(
        fps=30.0,
        robot=spec.robot,
        joint_names=list(spec.actuated_joints),
        root_pos_w=root_pos,
        root_quat_xyzw=root_quat,
        joint_pos=joint_pos,
        joint_vel=np.zeros_like(joint_pos),
        body_names=list(spec.body_names),
        body_pos_w=body_pos_w,
        body_quat_xyzw=body_quat_xyzw,
        success=np.ones(root_pos.shape[0], dtype=bool),
    )
    retargeted_motion.validate()
    fk = TorchRobotFK(spec, dtype=torch.float64, backend=backend)

    result = fk(
        torch.as_tensor(retargeted_motion.root_pos_w, dtype=torch.float64),
        torch.as_tensor(retargeted_motion.root_quat_xyzw, dtype=torch.float64),
        torch.as_tensor(retargeted_motion.joint_pos, dtype=torch.float64),
    )
    error = np.linalg.norm(result.body_pos_w.detach().cpu().numpy() - retargeted_motion.body_pos_w, axis=-1).max()

    assert float(error) <= _position_tolerance()


def test_torch_robot_fk_rejects_invalid_input_shapes():
    spec = _load_real_newton_spec(G1_29_ROBOT)
    fk = TorchRobotFK(spec)
    root_pos = torch.zeros((2, 3), dtype=torch.float32)
    root_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    joint_pos = torch.zeros((2, spec.num_dofs), dtype=torch.float32)

    with pytest.raises(ValueError, match="root_pos"):
        fk(torch.zeros(3), root_quat, joint_pos)
    with pytest.raises(ValueError, match="root_quat_xyzw"):
        fk(root_pos, torch.zeros((2, 3)), joint_pos)
    with pytest.raises(ValueError, match="joint_pos"):
        fk(root_pos, root_quat, torch.zeros((2, spec.num_dofs + 1)))
    with pytest.raises(ValueError, match="same T dimension"):
        fk(root_pos[:1], root_quat, joint_pos)


def test_max_position_error_rejects_invalid_tolerance():
    spec = _load_real_newton_spec(G1_29_ROBOT)
    root_pos, root_quat, joint_pos = _random_retarget_inputs(spec, frames=1, seed=5)

    with pytest.raises(ValueError, match="pos_tol"):
        max_position_error_against_newton(spec, root_pos, root_quat, joint_pos, pos_tol=0.0)


def _load_real_newton_spec(path: Path) -> RobotSpec:
    pytest.importorskip("newton")
    spec = RobotSpec.from_yaml(path)
    if not spec.model_path.exists():
        pytest.skip(f"local robot asset is unavailable: {spec.model_path}")
    return spec


def _position_tolerance() -> float:
    return float(os.environ.get("RETARGETER_TORCH_FK_POS_TOL", "1e-4"))


def _random_retarget_inputs(spec: RobotSpec, *, frames: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    root_pos = np.zeros((frames, 3), dtype=np.float64)
    root_pos[:, :2] = rng.uniform(-0.15, 0.15, size=(frames, 2))
    root_pos[:, 2] = rng.uniform(0.65, 0.95, size=frames)

    root_quat = rng.normal(size=(frames, 4))
    root_quat /= np.linalg.norm(root_quat, axis=-1, keepdims=True)

    center = 0.5 * (spec.joint_lower_rad + spec.joint_upper_rad)
    half_span = 0.25 * (spec.joint_upper_rad - spec.joint_lower_rad)
    joint_pos = center.reshape(1, -1) + rng.uniform(-1.0, 1.0, size=(frames, spec.num_dofs)) * half_span
    return root_pos, root_quat.astype(np.float64), joint_pos.astype(np.float64)


def _newton_body_sequence(
    backend: NewtonBackend,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    body_pos_w = []
    body_quat_xyzw = []
    for frame_idx in range(root_pos.shape[0]):
        state = IKState(
            root_pos_w=root_pos[frame_idx],
            root_quat_xyzw=root_quat[frame_idx],
            joint_pos=joint_pos[frame_idx],
        )
        body_state = backend.forward_kinematics(state)
        body_pos_w.append(body_state.body_pos_w)
        body_quat_xyzw.append(body_state.body_quat_xyzw)
    return np.stack(body_pos_w, axis=0), np.stack(body_quat_xyzw, axis=0)
