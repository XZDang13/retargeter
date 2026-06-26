from __future__ import annotations

from pathlib import Path

import numpy as np

from retargeter.newton import RobotSpec
from retargeter.scale import Stage1TargetBuilder


G1_29_ROBOT = Path("retargeter/newton/configs/g1_29_robot.yaml")
G1_23_ROBOT = Path("retargeter/newton/configs/g1_23_robot.yaml")
G1_29_STAGE = Path("retargeter/scale/configs/g1_29_stage1_targets.yaml")
G1_23_STAGE = Path("retargeter/scale/configs/g1_23_stage1_targets.yaml")
G1_29_SCALER = Path("retargeter/scale/configs/g1_29_scaler.yaml")
G1_23_SCALER = Path("retargeter/scale/configs/g1_23_scaler.yaml")


def test_g1_29_robot_spec_loads_explicit_names_and_limits():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)

    assert spec.robot == "unitree_g1_29"
    assert spec.model_path.exists()
    assert spec.num_dofs == 29
    assert "head_link" not in spec.body_names
    assert "left_wrist_yaw_link" in spec.body_names
    assert spec.joint_lower_rad.shape == (29,)
    assert np.isclose(spec.joint_lower_rad[0], np.deg2rad(-144.99843))
    assert np.all(spec.default_joint_pos >= spec.joint_lower_rad)
    assert np.all(spec.default_joint_pos <= spec.joint_upper_rad)


def test_g1_23_robot_spec_loads_wrist_roll_body_names():
    spec = RobotSpec.from_yaml(G1_23_ROBOT)

    assert spec.robot == "unitree_g1_23"
    assert spec.num_dofs == 23
    assert "left_wrist_roll" in spec.body_names
    assert "right_wrist_roll" in spec.body_names
    assert "head_link" not in spec.body_names


def test_robot_specs_cover_stage_target_required_bodies():
    cases = [
        (RobotSpec.from_yaml(G1_29_ROBOT), Stage1TargetBuilder(G1_29_SCALER, G1_29_STAGE)),
        (RobotSpec.from_yaml(G1_23_ROBOT), Stage1TargetBuilder(G1_23_SCALER, G1_23_STAGE)),
    ]

    for spec, builder in cases:
        spec.require_body_names(builder.required_robot_body_names("stage1a"))
        spec.require_body_names(builder.required_robot_body_names("stage1b"))
