from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import make_canonical_motion
from retargeter.newton import RobotSpec, build_regularization_objectives, build_target_objectives
from retargeter.newton.postprocess import apply_ik_postprocess, clamp_joint_limits, clamp_joint_velocity
from retargeter.scale import BodyIKTarget, IKTargetSet, IKTargetBuilder


G1_29_ROBOT = Path("retargeter/newton/configs/g1_29_robot.yaml")
G1_29_SCALER = Path("retargeter/scale/configs/g1_29_scaler.yaml")
G1_29_TARGETS = Path("retargeter/scale/configs/g1_29_ik_targets.yaml")


def test_build_target_objectives_preserves_target_weights_and_confidence():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    motion = make_canonical_motion(num_frames=3)
    target_set = IKTargetBuilder(G1_29_SCALER, G1_29_TARGETS).build(motion, 1, "full_body_tracking")

    descriptors = build_target_objectives(target_set, spec)

    kinds = [descriptor.kind for descriptor in descriptors]
    assert kinds.count("position") == len(target_set.active_position_targets())
    assert kinds.count("rotation") == len(target_set.active_rotation_targets())
    pelvis_pos = next(d for d in descriptors if d.kind == "position" and d.semantic_name == "pelvis")
    assert pelvis_pos.weight == target_set.get_target("pelvis").pos_weight
    assert pelvis_pos.confidence == 1.0


def test_local_foot_target_objective_preserves_body_local_point():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    target_set = IKTargetSet(
        pass_name="full_body_tracking",
        targets=[
            BodyIKTarget(
                semantic_name="left_toe",
                human_body_name="left_toe",
                robot_body_name="left_ankle_roll_link",
                target_pos_w=np.zeros(3),
                target_quat_xyzw=None,
                pos_weight=1.0,
                rot_weight=0.0,
                robot_local_pos=np.array([0.10, 0.0, 0.0]),
            )
        ],
    )

    descriptors = build_target_objectives(target_set, spec)

    left_toe = next(d for d in descriptors if d.kind == "position" and d.semantic_name == "left_toe")
    assert left_toe.body_name == "left_ankle_roll_link"
    assert np.allclose(left_toe.body_local_pos, [0.10, 0.0, 0.0])


def test_target_objectives_raise_for_missing_robot_body():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    target_set = IKTargetSet(
        pass_name="full_body_tracking",
        targets=[
            BodyIKTarget(
                semantic_name="bad",
                human_body_name="pelvis",
                robot_body_name="missing_link",
                target_pos_w=np.zeros(3),
                target_quat_xyzw=None,
                pos_weight=1.0,
                rot_weight=0.0,
            )
        ],
    )

    with pytest.raises(ValueError, match="missing required bodies"):
        build_target_objectives(target_set, spec)


def test_regularization_objectives_include_joint_limit_posture_and_smooth():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    previous = np.full(spec.num_dofs, 0.1)

    descriptors = build_regularization_objectives(
        spec,
        joint_limit_weight=10.0,
        posture_weight=0.05,
        smooth_weight=5.5,
        damping_weight=0.0,
        previous_joint_pos=previous,
    )

    assert [descriptor.kind for descriptor in descriptors] == ["joint_limit", "posture", "smooth"]
    assert np.allclose(descriptors[-1].target, previous)


def test_joint_limit_and_velocity_clamps_are_deterministic():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    q = spec.joint_upper_rad + 1.0

    clamped, changed, violation = clamp_joint_limits(q, spec)

    assert changed
    assert violation > 0.9
    assert np.allclose(clamped, spec.joint_upper_rad)

    previous = np.zeros(spec.num_dofs)
    fast = np.full(spec.num_dofs, 10.0)
    vel_clamped, vel_changed, max_vel = clamp_joint_velocity(fast, previous, 0.1, spec, velocity_scale=0.5)

    assert vel_changed
    assert max_vel == 100.0
    assert np.all(vel_clamped <= spec.velocity_limits_rad_s * 0.05 + 1e-12)


def test_apply_ik_postprocess_preserves_shape():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    q = spec.joint_upper_rad + 0.25

    out, report = apply_ik_postprocess(
        q,
        spec,
        previous_joint_pos=np.zeros(spec.num_dofs),
        dt=1.0 / 30.0,
        clamp_limits=True,
        clamp_velocity=True,
    )

    assert out.shape == (spec.num_dofs,)
    assert report.joint_limit_clamped
    assert report.velocity_clamped
