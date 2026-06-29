from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retargeter.newton import RobotSpec, RetargetedMotion
from retargeter.preprocess import FootContactResult
from retargeter.refinement import RefinedMotion, RefinementQualityReport, evaluate_refinement_quality


def test_refinement_quality_identical_motion_is_valid_and_jsonable():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined = _make_refined(retargeted)
    before = _motion_copies(retargeted, refined)

    report = evaluate_refinement_quality(retargeted, refined, spec)

    assert isinstance(report, RefinementQualityReport)
    assert report.valid is True
    assert report.failures == []
    assert report.metrics["body_pos_deviation_max_m"] == pytest.approx(0.0)
    assert report.metrics["joint_pos_deviation_max_rad"] == pytest.approx(0.0)
    assert report.metrics["skating_gate_evaluated"] is False
    assert report.to_dict()["valid"] is True
    _assert_unchanged(retargeted, refined, before)


def test_refinement_quality_reports_nonfinite_values_without_raising():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined = _make_refined(retargeted)
    before = _motion_copies(retargeted, refined)
    refined.root_pos_w[1, 0] = np.nan

    report = evaluate_refinement_quality(retargeted, refined, spec)

    assert report.valid is False
    assert "nonfinite_values" in report.failures
    assert report.metrics["refined_nonfinite_count"] == 1
    _assert_unchanged(retargeted, refined, before, refined_changed=True)


def test_refinement_quality_fails_joint_limit_and_large_motion_deviation():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined = _make_refined(retargeted)
    refined.joint_pos[2, 0] = spec.joint_upper_rad[0] + 0.05

    joint_report = evaluate_refinement_quality(retargeted, refined, spec, config={"quality": {"max_joint_deviation_rad": 10.0}})

    assert joint_report.valid is False
    assert "joint_limit_violation" in joint_report.failures
    assert joint_report.metrics["joint_limit_violation_count"] == 1
    assert joint_report.metrics["joint_limit_worst_joint"] == "joint_a"

    deviated = _make_refined(retargeted)
    deviated.root_pos_w[:, 0] += 0.2
    deviation_report = evaluate_refinement_quality(retargeted, deviated, spec)

    assert deviation_report.valid is False
    assert "root_position_deviation" in deviation_report.failures
    assert deviation_report.metrics["root_pos_deviation_max_m"] > deviation_report.thresholds["max_root_pos_deviation_m"]


def test_refinement_quality_soft_contact_scales_height_and_skating_metrics():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined = _make_refined(retargeted)
    ankle_idx = refined.body_names.index("left_ankle_roll_link")
    refined.body_pos_w[:, ankle_idx, 2] += 0.1
    refined.body_pos_w[:, ankle_idx, 0] += np.linspace(0.0, 0.05, refined.num_frames())

    ones = {"left_foot": np.ones(refined.num_frames(), dtype=np.float64)}
    half = {"left_foot": np.full(refined.num_frames(), 0.5, dtype=np.float64)}
    config = _loose_quality_config()

    full = evaluate_refinement_quality(retargeted, refined, spec, config=config, contact_score=ones)
    scaled = evaluate_refinement_quality(retargeted, refined, spec, config=config, contact_score=half)

    assert scaled.metrics["refined_weighted_foot_height_mean_m"] == pytest.approx(
        full.metrics["refined_weighted_foot_height_mean_m"] * 0.5
    )
    assert scaled.metrics["refined_weighted_skating_m_s"] == pytest.approx(
        full.metrics["refined_weighted_skating_m_s"] * 0.5
    )


def test_refinement_quality_skating_gate_uses_contact_scores_only_when_available():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined = _make_refined(retargeted)
    ankle_idx = refined.body_names.index("left_ankle_roll_link")
    refined.body_pos_w[:, ankle_idx, 0] += np.linspace(0.0, 0.05, refined.num_frames())
    config = _loose_quality_config()

    contact_report = evaluate_refinement_quality(
        retargeted,
        refined,
        spec,
        config=config,
        contact_score={"left_foot": np.ones(refined.num_frames(), dtype=np.float64)},
    )
    missing_contact_report = evaluate_refinement_quality(retargeted, refined, spec, config=config)

    assert contact_report.valid is False
    assert "skating_not_improved" in contact_report.failures
    assert contact_report.metrics["skating_gate_evaluated"] is True
    assert missing_contact_report.valid is True
    assert missing_contact_report.metrics["refined_weighted_skating_m_s"] > 0.0
    assert missing_contact_report.metrics["skating_gate_evaluated"] is False


def test_refinement_quality_penetration_worsening_fails():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined = _make_refined(retargeted)
    ankle_idx = refined.body_names.index("left_ankle_roll_link")
    refined.body_pos_w[:, ankle_idx, 2] = -0.02

    report = evaluate_refinement_quality(retargeted, refined, spec, config=_loose_quality_config(), ground_height=0.0)

    assert report.valid is False
    assert "penetration_worsened" in report.failures
    assert report.metrics["penetration_worsening_m"] > report.thresholds["penetration_worsening_tolerance_m"]


def test_refinement_quality_dynamics_are_finite_for_normal_and_short_sequences():
    spec = _make_robot_spec()
    for frames in (6, 2):
        retargeted = _make_retargeted(spec, frames=frames)
        refined = _make_refined(retargeted)
        refined.root_pos_w[:, 0] = np.linspace(0.0, 0.1, frames) ** 2

        report = evaluate_refinement_quality(retargeted, refined, spec, config=_loose_quality_config())

        for key, value in report.metrics.items():
            if "acceleration" in key or "jerk" in key:
                assert np.isfinite(value)


def test_refinement_quality_uses_contact_object_ground_height_and_local_offsets():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined = _make_refined(retargeted)
    contact = FootContactResult(
        contact_score={"left_foot": np.ones(refined.num_frames(), dtype=np.float64)},
        contact_binary={"left_foot": np.ones(refined.num_frames(), dtype=bool)},
        foot_height={},
        foot_speed={},
        ground_height=0.1,
    )
    config = {
        "quality": {
            "contact_points": {"left_foot": {"body": "left_ankle_roll_link", "local_pos": [0.0, 0.0, 0.05]}},
            "penetration_worsening_tolerance_m": 1.0,
        }
    }

    report = evaluate_refinement_quality(retargeted, refined, spec, config=config, contact_score=contact)

    assert report.metrics["ground_height"] == pytest.approx(0.1)
    assert report.metrics["refined_weighted_foot_height_mean_m"] == pytest.approx(0.05)


def test_refinement_quality_rejects_structural_mismatches_and_bad_contact_scores():
    spec = _make_robot_spec()
    retargeted = _make_retargeted(spec)
    refined = _make_refined(retargeted)

    bad_frames = _make_refined(_make_retargeted(spec, frames=retargeted.num_frames() + 1))
    with pytest.raises(ValueError, match="frames"):
        evaluate_refinement_quality(retargeted, bad_frames, spec)

    bad_joints = _make_refined(retargeted)
    bad_joints.joint_names = ["bad_a", "bad_b"]
    with pytest.raises(ValueError, match="joint_names"):
        evaluate_refinement_quality(retargeted, bad_joints, spec)

    with pytest.raises(ValueError, match="missing requested names"):
        evaluate_refinement_quality(retargeted, refined, spec, config={"quality": {"body_names": ["missing_body"]}})

    with pytest.raises(ValueError, match="contact_score"):
        evaluate_refinement_quality(retargeted, refined, spec, contact_score={"left_foot": np.ones(retargeted.num_frames() + 1)})


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
        velocity_limits_rad_s=np.array([2.0, 2.0], dtype=np.float64),
        default_joint_pos=np.zeros(2, dtype=np.float64),
        metadata={},
    )
    spec.validate()
    return spec


def _make_retargeted(spec: RobotSpec, *, frames: int = 6) -> RetargetedMotion:
    root_pos = np.zeros((frames, 3), dtype=np.float64)
    root_pos[:, 2] = 0.2
    root_quat = np.zeros((frames, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    joint_pos = np.zeros((frames, spec.num_dofs), dtype=np.float64)
    joint_vel = np.zeros_like(joint_pos)
    body_pos = root_pos[:, None, :] + _body_offsets()[None, :, :]
    body_quat = root_quat[:, None, :].repeat(len(spec.body_names), axis=1)
    motion = RetargetedMotion(
        fps=30.0,
        robot=spec.robot,
        joint_names=list(spec.actuated_joints),
        root_pos_w=root_pos,
        root_quat_xyzw=root_quat,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_names=list(spec.body_names),
        body_pos_w=body_pos,
        body_quat_xyzw=body_quat,
        success=np.ones(frames, dtype=bool),
    )
    motion.validate()
    return motion


def _make_refined(retargeted: RetargetedMotion) -> RefinedMotion:
    motion = RefinedMotion(
        fps=retargeted.fps,
        robot=retargeted.robot,
        joint_names=list(retargeted.joint_names),
        root_pos_w=retargeted.root_pos_w.copy(),
        root_quat_xyzw=retargeted.root_quat_xyzw.copy(),
        joint_pos=retargeted.joint_pos.copy(),
        joint_vel=retargeted.joint_vel.copy(),
        body_names=list(retargeted.body_names),
        body_pos_w=retargeted.body_pos_w.copy(),
        body_quat_xyzw=retargeted.body_quat_xyzw.copy(),
        root_delta=np.zeros_like(retargeted.root_pos_w),
        joint_delta=np.zeros_like(retargeted.joint_pos),
        metadata={"ground_height": 0.0},
    )
    motion.validate()
    return motion


def _body_offsets() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.08, -0.2],
            [0.2, 0.08, -0.2],
            [0.1, -0.08, -0.2],
            [0.2, -0.08, -0.2],
            [0.0, 0.0, 0.3],
        ],
        dtype=np.float64,
    )


def _loose_quality_config() -> dict:
    return {
        "quality": {
            "max_body_pos_deviation_m": 10.0,
            "mean_body_pos_deviation_m": 10.0,
            "max_root_pos_deviation_m": 10.0,
            "max_joint_deviation_rad": 10.0,
        }
    }


def _motion_copies(retargeted: RetargetedMotion, refined: RefinedMotion) -> dict[str, np.ndarray]:
    return {
        "retargeted_root": retargeted.root_pos_w.copy(),
        "retargeted_joint": retargeted.joint_pos.copy(),
        "retargeted_body": retargeted.body_pos_w.copy(),
        "refined_root": refined.root_pos_w.copy(),
        "refined_joint": refined.joint_pos.copy(),
        "refined_body": refined.body_pos_w.copy(),
    }


def _assert_unchanged(
    retargeted: RetargetedMotion,
    refined: RefinedMotion,
    before: dict[str, np.ndarray],
    *,
    refined_changed: bool = False,
) -> None:
    np.testing.assert_array_equal(retargeted.root_pos_w, before["retargeted_root"])
    np.testing.assert_array_equal(retargeted.joint_pos, before["retargeted_joint"])
    np.testing.assert_array_equal(retargeted.body_pos_w, before["retargeted_body"])
    if not refined_changed:
        np.testing.assert_array_equal(refined.root_pos_w, before["refined_root"])
        np.testing.assert_array_equal(refined.joint_pos, before["refined_joint"])
        np.testing.assert_array_equal(refined.body_pos_w, before["refined_body"])
