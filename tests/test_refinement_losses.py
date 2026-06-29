from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from retargeter.newton import RobotSpec, RetargetedMotion, TorchRobotFKResult
from retargeter.refinement import (
    delta_regularization_loss,
    grounding_loss,
    joint_feasibility_loss,
    motion_fidelity_loss,
    skating_loss,
    smoothness_loss,
    total_refinement_loss,
)


def test_motion_fidelity_loss_is_differentiable_and_does_not_mutate_retargeted():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    before = _retargeted_array_copies(retargeted)
    refined_fk, refined_root_pos, refined_joint_pos = _make_refined_fk_inputs(retargeted)

    loss, metrics = motion_fidelity_loss(retargeted, refined_fk, refined_joint_pos, refined_root_pos, {"motion_fidelity": {"body_names": retargeted.body_names}})
    loss.backward()

    assert loss.ndim == 0
    _assert_metrics(metrics)
    _assert_finite_grad(refined_fk.body_pos_w, refined_fk.body_quat_xyzw, refined_root_pos, refined_joint_pos)
    _assert_retargeted_unchanged(retargeted, before)


def test_joint_feasibility_loss_penalizes_limit_and_velocity_violations():
    spec = _make_robot_spec()
    joint_pos_ok = torch.zeros((4, spec.num_dofs), dtype=torch.float64, requires_grad=True)
    joint_vel_ok = torch.zeros((3, spec.num_dofs), dtype=torch.float64, requires_grad=True)

    ok_loss, ok_metrics = joint_feasibility_loss(joint_pos_ok, joint_vel_ok, spec, {"joint_feasibility": {"weight": 1.0, "velocity_weight": 1.0}})

    joint_pos_bad = torch.tensor([[0.0, 0.0], [2.0, -3.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    joint_vel_bad = torch.tensor([[0.0, 0.0], [3.0, -4.0], [0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    bad_loss, bad_metrics = joint_feasibility_loss(
        joint_pos_bad,
        joint_vel_bad,
        spec,
        {"joint_feasibility": {"weight": 1.0, "velocity_weight": 1.0, "joint_range_margin": 1.0}},
    )
    bad_loss.backward()

    assert torch.isclose(ok_loss, torch.zeros_like(ok_loss))
    assert bad_loss > ok_loss
    assert bad_metrics["joint_limit"] > ok_metrics["joint_limit"]
    assert bad_metrics["joint_velocity"] > ok_metrics["joint_velocity"]
    _assert_finite_grad(joint_pos_bad, joint_vel_bad)


def test_grounding_loss_uses_fractional_contact_scores_and_local_offsets():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined_fk, _, _ = _make_refined_fk_inputs(retargeted, body_pos_offset=0.0)
    ones = {"left_foot": torch.ones(retargeted.num_frames(), dtype=torch.float64)}
    half = {"left_foot": torch.full((retargeted.num_frames(),), 0.5, dtype=torch.float64)}
    config = {"grounding": {"weight": 1.0, "contact_points": {"left_foot": {"body": "left_ankle_roll_link"}}}}

    full_loss, _ = grounding_loss(refined_fk, ones, 0.0, spec, config)
    half_loss, _ = grounding_loss(refined_fk, half, 0.0, spec, config)
    zero_loss, _ = grounding_loss(refined_fk, {"left_foot": torch.zeros(retargeted.num_frames())}, 0.0, spec, config)

    assert torch.allclose(half_loss, full_loss * 0.5)
    assert torch.allclose(zero_loss, torch.zeros_like(zero_loss))

    quat = refined_fk.body_quat_xyzw.detach().clone()
    left_ankle_idx = refined_fk.body_names.index("left_ankle_roll_link")
    quat[:, left_ankle_idx, :] = torch.tensor([0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)], dtype=torch.float64)
    rotated_fk = TorchRobotFKResult(
        body_names=refined_fk.body_names,
        body_pos_w=refined_fk.body_pos_w.detach().clone().requires_grad_(True),
        body_quat_xyzw=quat.requires_grad_(True),
    )
    offset_config = {
        "grounding": {
            "weight": 1.0,
            "contact_points": {"left_foot": {"body": "left_ankle_roll_link", "local_pos": [1.0, 0.0, 0.0]}},
        }
    }
    identity_loss, _ = grounding_loss(refined_fk, ones, 0.0, spec, offset_config)
    rotated_loss, _ = grounding_loss(rotated_fk, ones, 0.0, spec, offset_config)
    rotated_loss.backward()

    assert not torch.allclose(identity_loss, rotated_loss)
    _assert_finite_grad(rotated_fk.body_pos_w, rotated_fk.body_quat_xyzw)


def test_skating_loss_uses_fractional_contact_scores_without_dropping_frames():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined_fk, _, _ = _make_refined_fk_inputs(retargeted, body_pos_offset=0.0)
    left_ankle_idx = refined_fk.body_names.index("left_ankle_roll_link")
    with torch.no_grad():
        refined_fk.body_pos_w[:, left_ankle_idx, 0] = torch.arange(retargeted.num_frames(), dtype=torch.float64)

    config = {"fps": 1.0, "skating": {"weight": 1.0, "contact_points": {"left_foot": "left_ankle_roll_link"}}}
    ones = {"left_foot": torch.ones(retargeted.num_frames(), dtype=torch.float64)}
    quarter = {"left_foot": torch.full((retargeted.num_frames(),), 0.25, dtype=torch.float64)}
    zeros = {"left_foot": torch.zeros(retargeted.num_frames(), dtype=torch.float64)}

    full_loss, _ = skating_loss(refined_fk, ones, spec, config)
    quarter_loss, _ = skating_loss(refined_fk, quarter, spec, config)
    zero_loss, _ = skating_loss(refined_fk, zeros, spec, config)

    assert torch.allclose(quarter_loss, full_loss * 0.25)
    assert torch.allclose(zero_loss, torch.zeros_like(zero_loss))
    assert full_loss > 0.0


def test_skating_loss_is_not_diluted_by_non_contact_frames():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined_fk, _, _ = _make_refined_fk_inputs(retargeted, body_pos_offset=0.0)
    left_ankle_idx = refined_fk.body_names.index("left_ankle_roll_link")
    with torch.no_grad():
        refined_fk.body_pos_w[:, left_ankle_idx, 0] = torch.arange(retargeted.num_frames(), dtype=torch.float64)

    config = {"fps": 1.0, "skating": {"weight": 1.0, "contact_points": {"left_foot": "left_ankle_roll_link"}}}
    full_contact = {"left_foot": torch.ones(retargeted.num_frames(), dtype=torch.float64)}
    sparse_contact = {"left_foot": torch.zeros(retargeted.num_frames(), dtype=torch.float64)}
    sparse_contact["left_foot"][1] = 1.0

    full_loss, _ = skating_loss(refined_fk, full_contact, spec, config)
    sparse_loss, _ = skating_loss(refined_fk, sparse_contact, spec, config)

    assert torch.allclose(sparse_loss, full_loss)


def test_smoothness_loss_zero_for_constant_velocity_and_positive_for_curvature():
    t = torch.arange(5, dtype=torch.float64).reshape(-1, 1)
    root_linear = torch.cat((t, t * 0.0, t * 0.0), dim=1).requires_grad_(True)
    joint_linear = torch.cat((t, -t), dim=1).requires_grad_(True)
    config = {"fps": 1.0, "smoothness": {"weight": 1.0}}

    zero_loss, _ = smoothness_loss(root_linear, joint_linear, config)

    root_curved = root_linear.detach().clone()
    root_curved[2, 0] += 1.0
    root_curved.requires_grad_(True)
    joint_curved = joint_linear.detach().clone()
    joint_curved[2, 1] -= 1.0
    joint_curved.requires_grad_(True)
    curved_loss, _ = smoothness_loss(root_curved, joint_curved, config)
    curved_loss.backward()

    assert torch.allclose(zero_loss, torch.zeros_like(zero_loss))
    assert curved_loss > 0.0
    _assert_finite_grad(root_curved, joint_curved)


def test_delta_regularization_loss_is_differentiable():
    root_delta = torch.full((4, 3), 0.1, dtype=torch.float64, requires_grad=True)
    joint_delta = torch.full((4, 2), -0.2, dtype=torch.float64, requires_grad=True)

    loss, metrics = delta_regularization_loss(
        root_delta,
        joint_delta,
        {"delta_regularization": {"weight": 2.0, "root_weight": 3.0, "joint_weight": 5.0}},
    )
    loss.backward()

    assert loss > 0.0
    _assert_metrics(metrics)
    _assert_finite_grad(root_delta, joint_delta)


def test_total_refinement_loss_matches_component_sum():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined_fk, refined_root_pos, refined_joint_pos = _make_refined_fk_inputs(retargeted)
    refined_joint_vel = torch.diff(refined_joint_pos, dim=0) * retargeted.fps
    root_delta = refined_root_pos - torch.as_tensor(retargeted.root_pos_w, dtype=refined_root_pos.dtype)
    joint_delta = refined_joint_pos - torch.as_tensor(retargeted.joint_pos, dtype=refined_joint_pos.dtype)
    contact_score = {"left_foot": torch.linspace(0.0, 1.0, retargeted.num_frames(), dtype=torch.float64)}
    config = _all_loss_config(retargeted)

    total, metrics = total_refinement_loss(
        retargeted,
        refined_fk,
        refined_joint_pos,
        refined_root_pos,
        refined_joint_vel,
        root_delta,
        joint_delta,
        contact_score,
        0.0,
        spec,
        config,
    )
    components = [
        motion_fidelity_loss(retargeted, refined_fk, refined_joint_pos, refined_root_pos, config)[0],
        joint_feasibility_loss(refined_joint_pos, refined_joint_vel, spec, config)[0],
        grounding_loss(refined_fk, contact_score, 0.0, spec, config)[0],
        skating_loss(refined_fk, contact_score, spec, config)[0],
        smoothness_loss(refined_root_pos, refined_joint_pos, config)[0],
        delta_regularization_loss(root_delta, joint_delta, config)[0],
    ]

    assert torch.allclose(total, sum(components))
    assert torch.allclose(metrics["loss"], total.detach())
    assert "motion_fidelity/loss" in metrics
    assert "joint_feasibility/loss" in metrics


def test_refinement_losses_raise_clear_errors_for_invalid_inputs():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined_fk, refined_root_pos, refined_joint_pos = _make_refined_fk_inputs(retargeted)

    with pytest.raises(ValueError, match="missing"):
        motion_fidelity_loss(retargeted, refined_fk, refined_joint_pos, refined_root_pos, {"motion_fidelity": {"body_names": ["missing"]}})
    with pytest.raises(ValueError, match="contact_score"):
        grounding_loss(refined_fk, {"left_foot": torch.ones(retargeted.num_frames() + 1)}, 0.0, spec, {})
    with pytest.raises(ValueError, match="missing from RobotSpec"):
        grounding_loss(
            refined_fk,
            {"left_foot": torch.ones(retargeted.num_frames())},
            0.0,
            spec,
            {"grounding": {"contact_points": {"left_foot": {"body": "missing_body"}}}},
        )


def _make_robot_spec() -> RobotSpec:
    spec = RobotSpec(
        robot="test_g1",
        model_path=Path("dummy.usda"),
        model_format="usd",
        floating_base=True,
        root_body="pelvis",
        body_names=[
            "pelvis",
            "left_ankle_roll_link",
            "left_toe_link",
            "right_ankle_roll_link",
            "right_toe_link",
            "torso_link",
        ],
        actuated_joints=["joint_a", "joint_b"],
        joint_lower_rad=np.array([-1.0, -2.0], dtype=np.float64),
        joint_upper_rad=np.array([1.0, 2.0], dtype=np.float64),
        velocity_limits_rad_s=np.array([1.0, 2.0], dtype=np.float64),
        default_joint_pos=np.zeros(2, dtype=np.float64),
        metadata={},
    )
    spec.validate()
    return spec


def _make_retargeted(spec: RobotSpec, frames: int = 5) -> RetargetedMotion:
    root_pos = np.zeros((frames, 3), dtype=np.float64)
    root_pos[:, 0] = np.linspace(0.0, 0.4, frames)
    root_pos[:, 2] = 0.8
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.10, 0.08, -0.75],
            [0.22, 0.08, -0.80],
            [0.10, -0.08, -0.75],
            [0.22, -0.08, -0.80],
            [0.0, 0.0, 0.35],
        ],
        dtype=np.float64,
    )
    body_pos = root_pos[:, None, :] + offsets[None, :, :]
    body_quat = np.zeros((frames, len(spec.body_names), 4), dtype=np.float64)
    body_quat[..., 3] = 1.0
    joint_pos = np.zeros((frames, spec.num_dofs), dtype=np.float64)
    joint_vel = np.zeros_like(joint_pos)
    motion = RetargetedMotion(
        fps=30.0,
        robot=spec.robot,
        joint_names=list(spec.actuated_joints),
        root_pos_w=root_pos,
        root_quat_xyzw=np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64), (frames, 1)),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_names=list(spec.body_names),
        body_pos_w=body_pos,
        body_quat_xyzw=body_quat,
        success=np.ones(frames, dtype=bool),
    )
    motion.validate()
    return motion


def _make_refined_fk_inputs(
    retargeted: RetargetedMotion,
    *,
    body_pos_offset: float = 0.01,
) -> tuple[TorchRobotFKResult, torch.Tensor, torch.Tensor]:
    body_pos = torch.tensor(retargeted.body_pos_w + body_pos_offset, dtype=torch.float64, requires_grad=True)
    body_quat_np = retargeted.body_quat_xyzw.copy()
    body_quat_np[..., 0] = 0.05
    body_quat_np[..., 3] = np.sqrt(1.0 - 0.05**2)
    body_quat = torch.tensor(body_quat_np, dtype=torch.float64, requires_grad=True)
    refined_root_pos = torch.tensor(retargeted.root_pos_w + 0.02, dtype=torch.float64, requires_grad=True)
    refined_joint_pos = torch.tensor(retargeted.joint_pos + 0.03, dtype=torch.float64, requires_grad=True)
    return TorchRobotFKResult(list(retargeted.body_names), body_pos, body_quat), refined_root_pos, refined_joint_pos


def _retargeted_array_copies(retargeted: RetargetedMotion) -> dict[str, np.ndarray]:
    return {
        "root_pos_w": retargeted.root_pos_w.copy(),
        "root_quat_xyzw": retargeted.root_quat_xyzw.copy(),
        "joint_pos": retargeted.joint_pos.copy(),
        "joint_vel": retargeted.joint_vel.copy(),
        "body_pos_w": retargeted.body_pos_w.copy(),
        "body_quat_xyzw": retargeted.body_quat_xyzw.copy(),
        "success": retargeted.success.copy(),
    }


def _assert_retargeted_unchanged(retargeted: RetargetedMotion, before: dict[str, np.ndarray]) -> None:
    for name, expected in before.items():
        actual = getattr(retargeted, name)
        assert np.array_equal(actual, expected), name


def _assert_metrics(metrics: dict[str, torch.Tensor]) -> None:
    assert metrics
    for value in metrics.values():
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
        assert not value.requires_grad
        assert torch.isfinite(value)


def _assert_finite_grad(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def _all_loss_config(retargeted: RetargetedMotion) -> dict:
    return {
        "fps": retargeted.fps,
        "motion_fidelity": {
            "body_names": retargeted.body_names,
            "body_pos_weight": 1.0,
            "local_body_pos_weight": 1.0,
            "body_quat_weight": 0.1,
            "root_pos_weight": 1.0,
            "joint_pos_weight": 0.01,
        },
        "joint_feasibility": {"weight": 1.0, "velocity_weight": 1.0},
        "grounding": {"weight": 1.0, "contact_points": {"left_foot": "left_ankle_roll_link"}},
        "skating": {"weight": 1.0, "contact_points": {"left_foot": "left_ankle_roll_link"}},
        "smoothness": {"weight": 1.0},
        "delta_regularization": {"weight": 1.0},
    }
