from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from retargeter.newton import RobotSpec, RetargetedMotion, TorchRobotFKResult
from retargeter.preprocess import CanonicalHumanMotion, FootContactResult, PreprocessResult
from retargeter.refinement import RefinedMotion, TorchMotionRefiner, run_refinement, run_refinement_batch


def test_torch_motion_refiner_outputs_valid_refined_motion_and_preserves_inputs():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    preprocess = _make_preprocess(retargeted, with_contact=True)
    before = _retargeted_array_copies(retargeted)
    log_records = []
    refiner = TorchMotionRefiner(spec, FakeTorchFK(spec), config=_base_config(iterations=8), log_fn=log_records.append)

    refined = refiner.refine(retargeted, preprocess)

    assert isinstance(refined, RefinedMotion)
    refined.validate()
    assert refined.num_frames() == retargeted.num_frames()
    assert refined.root_pos_w.shape == retargeted.root_pos_w.shape
    assert refined.joint_pos.shape == retargeted.joint_pos.shape
    assert refined.body_pos_w.shape == (retargeted.num_frames(), len(spec.body_names), 3)
    assert np.max(np.abs(refined.root_delta)) <= 0.2 + 1e-6
    assert np.max(np.abs(refined.joint_delta)) <= 0.2 + 1e-6
    assert refined.loss_curve[0]["iteration"] == 0
    assert refined.loss_curve[-1]["iteration"] == 8
    assert log_records == refined.loss_curve
    assert refined.quality_metrics["contact_available"] is True
    assert refined.quality_metrics["iteration_count"] == 8
    assert "final/loss" in refined.quality_metrics
    _assert_retargeted_unchanged(retargeted, before)


def test_torch_motion_refiner_reduces_simple_grounding_loss():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec, root_height=0.12)
    preprocess = _make_preprocess(retargeted, with_contact=True)
    config = _base_config(iterations=40, lr=0.2)
    config["motion_fidelity"] = _zero_motion_fidelity_config(retargeted)
    config["grounding"] = {"weight": 1.0, "contact_points": {"left_foot": "left_ankle_roll_link"}}
    config["skating"] = {"weight": 0.0, "contact_points": {"left_foot": "left_ankle_roll_link"}}
    refiner = TorchMotionRefiner(spec, FakeTorchFK(spec), config=config)

    refined = refiner.refine(retargeted, preprocess)

    assert refined.quality_metrics["final_loss"] <= refined.quality_metrics["initial_loss"]
    left_ankle_idx = refined.body_names.index("left_ankle_roll_link")
    assert np.mean(refined.body_pos_w[:, left_ankle_idx, 2] ** 2) < np.mean(retargeted.body_pos_w[:, left_ankle_idx, 2] ** 2)


def test_torch_motion_refiner_missing_contact_runs_with_zero_contact_losses():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    preprocess = _make_preprocess(retargeted, with_contact=False)
    refined = run_refinement(retargeted, preprocess, spec, FakeTorchFK(spec), config=_base_config(iterations=2))

    assert refined.quality_metrics["contact_available"] is False
    assert refined.quality_metrics["final/grounding/loss"] == 0.0
    assert refined.quality_metrics["final/skating/loss"] == 0.0
    refined.validate()


def test_run_refinement_batch_pads_and_unpads_variable_length_clips():
    spec = _make_robot_spec()
    first = _make_retargeted(spec, frames=5)
    second = _make_retargeted(spec, frames=3, root_height=0.14)
    preprocess = [_make_preprocess(first, with_contact=True), _make_preprocess(second, with_contact=True)]

    refined = run_refinement_batch(
        [first, second],
        preprocess,
        spec,
        FakeTorchFK(spec),
        config=_base_config(iterations=2),
    )

    assert [motion.num_frames() for motion in refined] == [5, 3]
    assert all(isinstance(motion, RefinedMotion) for motion in refined)
    assert [motion.quality_metrics["batch_valid_frame_count"] for motion in refined] == [5, 3]
    assert all(motion.quality_metrics["batch_size"] == 2 for motion in refined)
    assert all(motion.metadata["source"] == "BatchedTorchMotionRefiner" for motion in refined)


def test_torch_motion_refiner_lbfgs_path_runs():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    preprocess = _make_preprocess(retargeted, with_contact=True)
    config = _base_config(iterations=2)
    config["refiner"]["lbfgs_enabled"] = True
    config["refiner"]["lbfgs_max_iter"] = 2
    config["refiner"]["lbfgs_lr"] = 0.5

    refined = TorchMotionRefiner(spec, FakeTorchFK(spec), config=config).refine(retargeted, preprocess)

    assert refined.quality_metrics["lbfgs_enabled"] is True
    assert refined.loss_curve[-1]["phase"] == "lbfgs"
    refined.validate()


def test_torch_motion_refiner_rejects_invalid_inputs_and_config():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    preprocess = _make_preprocess(retargeted, with_contact=True)
    bad_preprocess = _make_preprocess(_make_retargeted(spec, frames=retargeted.num_frames() + 1), with_contact=True)

    with pytest.raises(ValueError, match="frames"):
        TorchMotionRefiner(spec, FakeTorchFK(spec), config=_base_config(iterations=1)).refine(retargeted, bad_preprocess)
    with pytest.raises(ValueError, match="max_root_delta"):
        TorchMotionRefiner(spec, FakeTorchFK(spec), config={"refiner": {"max_root_delta": -0.1}})
    with pytest.raises(ValueError, match="dtype"):
        TorchMotionRefiner(spec, FakeTorchFK(spec), config={"refiner": {"dtype": "float16"}})
    bad_retargeted = _make_retargeted(spec)
    bad_retargeted.joint_names = ["bad_a", "bad_b"]
    with pytest.raises(ValueError, match="joint_names"):
        TorchMotionRefiner(spec, FakeTorchFK(spec), config=_base_config(iterations=1)).refine(bad_retargeted, preprocess)


class FakeTorchFK(torch.nn.Module):
    def __init__(self, robot_spec: RobotSpec):
        super().__init__()
        self.robot_spec = robot_spec
        self.body_names = list(robot_spec.body_names)

    def forward(self, root_pos: torch.Tensor, root_quat_xyzw: torch.Tensor, joint_pos: torch.Tensor) -> TorchRobotFKResult:
        offsets = torch.as_tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.08, 0.0],
                [0.2, 0.08, -0.02],
                [0.1, -0.08, 0.0],
                [0.2, -0.08, -0.02],
                [0.0, 0.0, 0.3],
            ],
            dtype=root_pos.dtype,
            device=root_pos.device,
        )
        body_pos = root_pos[:, None, :] + offsets[None, :, :]
        body_pos = body_pos.clone()
        body_pos[:, 1, 2] = body_pos[:, 1, 2] + 0.25 * joint_pos[:, 0]
        body_pos[:, 3, 2] = body_pos[:, 3, 2] + 0.25 * joint_pos[:, 1]
        body_quat = root_quat_xyzw[:, None, :].expand(-1, len(self.body_names), -1)
        return TorchRobotFKResult(body_names=list(self.body_names), body_pos_w=body_pos, body_quat_xyzw=body_quat)


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
        joint_lower_rad=np.array([-1.0, -1.0], dtype=np.float64),
        joint_upper_rad=np.array([1.0, 1.0], dtype=np.float64),
        velocity_limits_rad_s=np.array([20.0, 20.0], dtype=np.float64),
        default_joint_pos=np.zeros(2, dtype=np.float64),
        metadata={},
    )
    spec.validate()
    return spec


def _make_retargeted(spec: RobotSpec, *, frames: int = 5, root_height: float = 0.1) -> RetargetedMotion:
    root_pos = np.zeros((frames, 3), dtype=np.float64)
    root_pos[:, 0] = np.linspace(0.0, 0.1, frames)
    root_pos[:, 2] = root_height
    root_quat = np.zeros((frames, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    joint_pos = np.zeros((frames, spec.num_dofs), dtype=np.float64)
    joint_vel = np.zeros_like(joint_pos)
    fk = FakeTorchFK(spec)
    with torch.no_grad():
        fk_result = fk(
            torch.as_tensor(root_pos, dtype=torch.float64),
            torch.as_tensor(root_quat, dtype=torch.float64),
            torch.as_tensor(joint_pos, dtype=torch.float64),
        )
    motion = RetargetedMotion(
        fps=30.0,
        robot=spec.robot,
        joint_names=list(spec.actuated_joints),
        root_pos_w=root_pos,
        root_quat_xyzw=root_quat,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_names=list(fk_result.body_names),
        body_pos_w=fk_result.body_pos_w.numpy(),
        body_quat_xyzw=fk_result.body_quat_xyzw.numpy(),
        success=np.ones(frames, dtype=bool),
    )
    motion.validate()
    return motion


def _make_preprocess(retargeted: RetargetedMotion, *, with_contact: bool) -> PreprocessResult:
    motion = CanonicalHumanMotion(
        fps=retargeted.fps,
        body_names=["pelvis"],
        body_pos_w=retargeted.root_pos_w[:, None, :].copy(),
        body_quat_xyzw=retargeted.root_quat_xyzw[:, None, :].copy(),
    )
    if with_contact:
        score = np.ones(retargeted.num_frames(), dtype=np.float64)
        contact = FootContactResult(
            contact_score={"left_foot": score},
            contact_binary={"left_foot": np.ones(retargeted.num_frames(), dtype=bool)},
            foot_height={"left_foot": retargeted.body_pos_w[:, retargeted.body_names.index("left_ankle_roll_link"), 2]},
            foot_speed={"left_foot": np.zeros(retargeted.num_frames(), dtype=np.float64)},
            ground_height=0.0,
        )
    else:
        contact = None
    return PreprocessResult(
        motion=motion,
        ground=None,
        contact=contact,
        warnings=[],
        metadata={"normalized_ground_height": 0.0, "contact_available": contact is not None},
    )


def _base_config(*, iterations: int, lr: float = 0.05) -> dict:
    return {
        "refiner": {
            "iterations": iterations,
            "lr": lr,
            "log_interval": 2,
            "max_root_delta": 0.2,
            "max_joint_delta": 0.2,
            "dtype": "float64",
        },
        "motion_fidelity": _zero_motion_fidelity_config_body(),
        "joint_feasibility": {"weight": 0.0, "velocity_weight": 0.0},
        "grounding": {"weight": 1.0, "contact_points": {"left_foot": "left_ankle_roll_link"}},
        "skating": {"weight": 0.0, "contact_points": {"left_foot": "left_ankle_roll_link"}},
        "smoothness": {"weight": 0.0},
        "delta_regularization": {"weight": 0.0},
    }


def _zero_motion_fidelity_config(retargeted: RetargetedMotion) -> dict:
    config = _zero_motion_fidelity_config_body()
    config["body_names"] = list(retargeted.body_names)
    return config


def _zero_motion_fidelity_config_body() -> dict:
    return {
        "body_pos_weight": 0.0,
        "local_body_pos_weight": 0.0,
        "body_quat_weight": 0.0,
        "root_pos_weight": 0.0,
        "joint_pos_weight": 0.0,
    }


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
        assert np.array_equal(getattr(retargeted, name), expected), name
