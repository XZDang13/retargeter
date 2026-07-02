from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import make_canonical_motion
from retargeter.newton import (
    RobotSpec,
    build_regularization_objectives,
    build_self_collision_objectives,
    build_target_objectives,
)
from retargeter.newton.newton_backend import NewtonBackend, _native_objective_layout_key
from retargeter.newton.objectives import IKObjectiveDescriptor
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


def test_newton_backend_binds_regularization_as_native_objectives():
    pytest.importorskip("newton")
    pytest.importorskip("warp")
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    backend = NewtonBackend(spec)
    backend._ensure_loaded()
    descriptors = [
        IKObjectiveDescriptor(kind="posture", weight=0.05, target=spec.default_joint_pos.copy()),
        IKObjectiveDescriptor(kind="smooth", weight=5.5, target=np.full(spec.num_dofs, 0.1)),
    ]

    native, bindings = backend._build_bound_native_objectives(descriptors)

    assert [binding.kind for binding in bindings] == ["posture", "smooth"]
    assert [objective.residual_dim() for objective in native] == [spec.num_dofs, spec.num_dofs]
    assert _native_objective_layout_key(descriptors) == (("posture",), ("smooth",))


def test_self_collision_config_validation_and_layout_key():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    config = {
        "enabled": True,
        "weight": 3.0,
        "margin_m": 0.06,
        "pairs": [
            {
                "name": "left_hand_torso",
                "point_body": "left_wrist_yaw_link",
                "obstacle": {"shape": "sphere", "body": "torso_link", "radius_m": 0.14},
            }
        ],
    }

    descriptors = build_self_collision_objectives(spec, config)

    assert [descriptor.kind for descriptor in descriptors] == ["self_collision"]
    assert descriptors[0].self_collision_pairs[0].name == "left_hand_torso"
    assert _native_objective_layout_key(descriptors)[0][0] == "self_collision"

    bad = dict(config)
    bad["pairs"] = [
        {
            "name": "bad_body",
            "point_body": "missing_link",
            "obstacle": {"shape": "sphere", "body": "torso_link", "radius_m": 0.14},
        }
    ]
    with pytest.raises(ValueError, match="missing required bodies"):
        build_self_collision_objectives(spec, bad)

    bad_radius = dict(config)
    bad_radius["pairs"] = [
        {
            "name": "bad_radius",
            "point_body": "left_wrist_yaw_link",
            "obstacle": {"shape": "sphere", "body": "torso_link", "radius_m": 0.0},
        }
    ]
    with pytest.raises(ValueError, match="radius"):
        build_self_collision_objectives(spec, bad_radius)

    bad_empty = dict(config)
    bad_empty["pairs"] = []
    with pytest.raises(ValueError, match="non-empty list"):
        build_self_collision_objectives(spec, bad_empty)


def test_self_collision_objective_residuals_are_batched():
    pytest.importorskip("newton")
    wp = pytest.importorskip("warp")
    from newton._src.sim.ik.ik_common import IKJacobianType
    from retargeter.newton.objectives import SelfCollisionPairSpec
    from retargeter.newton.self_collision_objectives import IKSelfCollisionObjective

    device = wp.get_device("cpu")
    pair = SelfCollisionPairSpec(
        name="point_sphere",
        point_body="point",
        obstacle_body="obstacle",
        obstacle_shape="sphere",
        obstacle_radius_m=0.20,
        margin_m=0.10,
    )
    objective = IKSelfCollisionObjective((pair,), body_name_to_index={"point": 0, "obstacle": 1}, weight=2.0)
    objective.set_batch_layout(total_residuals=1, residual_offset=0, n_batch=2)
    objective.bind_device(device)
    objective.init_buffers(model=None, jacobian_mode=IKJacobianType.AUTODIFF)

    identity = wp.quat(0.0, 0.0, 0.0, 1.0)
    body_q = wp.array(
        [
            [wp.transform(wp.vec3(0.25, 0.0, 0.0), identity), wp.transform(wp.vec3(0.0, 0.0, 0.0), identity)],
            [wp.transform(wp.vec3(0.50, 0.0, 0.0), identity), wp.transform(wp.vec3(0.0, 0.0, 0.0), identity)],
        ],
        dtype=wp.transform,
        device=device,
    )
    residuals = wp.zeros((2, 1), dtype=wp.float32, device=device)
    joint_q = wp.zeros((2, 1), dtype=wp.float32, device=device)
    problem_idx = wp.array(np.asarray([0, 1], dtype=np.int32), dtype=wp.int32, device=device)

    objective.compute_residuals(body_q, joint_q, None, residuals, 0, problem_idx)
    wp.synchronize()

    values = residuals.numpy()
    assert values[0, 0] == pytest.approx(0.10, abs=1e-5)
    assert values[1, 0] == pytest.approx(0.0, abs=1e-6)


def test_newton_backend_binds_self_collision_objective():
    pytest.importorskip("newton")
    pytest.importorskip("warp")
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    backend = NewtonBackend(spec)
    backend._ensure_loaded()
    descriptors = build_self_collision_objectives(
        spec,
        {
            "enabled": True,
            "weight": 3.0,
            "margin_m": 0.06,
            "pairs": [
                {
                    "name": "left_hand_torso",
                    "point_body": "left_wrist_yaw_link",
                    "obstacle": {"shape": "sphere", "body": "torso_link", "radius_m": 0.14},
                }
            ],
        },
    )

    native, bindings = backend._build_bound_native_objectives(descriptors)

    assert [binding.kind for binding in bindings] == ["self_collision"]
    assert [objective.residual_dim() for objective in native] == [1]


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
