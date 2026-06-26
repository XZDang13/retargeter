from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from conftest import make_canonical_motion
from retargeter.preprocess import FootContactResult
from retargeter.scale import Stage1TargetBuilder


G1_29_SCALER = Path("retargeter/scale/configs/g1_29_scaler.yaml")
G1_23_SCALER = Path("retargeter/scale/configs/g1_23_scaler.yaml")
G1_29_TARGETS = Path("retargeter/scale/configs/g1_29_stage1_targets.yaml")
G1_23_TARGETS = Path("retargeter/scale/configs/g1_23_stage1_targets.yaml")


def test_g1_29_stage1a_builder_returns_expected_targets():
    motion = make_canonical_motion(num_frames=5)
    builder = Stage1TargetBuilder(G1_29_SCALER, G1_29_TARGETS)

    target_set = builder.build(motion, frame_idx=2, stage_name="stage1a")

    assert target_set.stage_name == "stage1a"
    assert [target.semantic_name for target in target_set.targets] == [
        "pelvis",
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_hip",
        "right_knee",
        "right_ankle",
        "chest",
        "left_shoulder",
        "left_elbow",
        "left_hand",
        "right_shoulder",
        "right_elbow",
        "right_hand",
    ]
    assert target_set.get_target("left_hand").robot_body_name == "left_wrist_yaw_link"
    assert target_set.get_target("right_ankle").robot_body_name == "right_ankle_roll_link"
    assert target_set.get_target("right_ankle").human_body_name == "right_ankle"
    assert target_set.get_target("right_ankle").target_pos_w.shape == (3,)
    assert target_set.get_target("right_ankle").target_quat_xyzw is None


def test_gmr_style_hand_and_shoulder_links_for_both_g1_variants():
    motion = make_canonical_motion(num_frames=3)
    target_29 = Stage1TargetBuilder(G1_29_SCALER, G1_29_TARGETS).build(motion, 0, "stage1a")
    target_23 = Stage1TargetBuilder(G1_23_SCALER, G1_23_TARGETS).build(motion, 0, "stage1a")

    assert target_29.get_target("left_hand").rot_weight == 10.0
    assert target_29.get_target("left_hand").target_pos_w is None
    assert target_29.get_target("left_hand").target_quat_xyzw.shape == (4,)
    assert target_23.get_target("left_hand").robot_body_name == "left_rubber_hand_link"
    assert target_23.get_target("right_hand").robot_body_name == "right_rubber_hand_link"
    assert target_29.get_target("left_shoulder").robot_body_name == "left_shoulder_yaw_link"
    assert target_23.get_target("right_shoulder").robot_body_name == "right_shoulder_yaw_link"


def test_stage1b_includes_gmr_style_limb_targets():
    motion = make_canonical_motion(num_frames=3)
    target_set = Stage1TargetBuilder(G1_29_SCALER, G1_29_TARGETS).build(motion, 0, "stage1b")

    assert target_set.get_target("left_hip").robot_body_name == "left_hip_roll_link"
    assert target_set.get_target("right_hip").robot_body_name == "right_hip_roll_link"
    assert target_set.get_target("left_ankle").human_body_name == "left_ankle"
    assert target_set.get_target("right_ankle").human_body_name == "right_ankle"
    assert target_set.get_target("left_shoulder").robot_body_name == "left_shoulder_yaw_link"
    assert target_set.get_target("right_shoulder").robot_body_name == "right_shoulder_yaw_link"


def test_stage_required_robot_bodies_exclude_non_target_head_link():
    builder = Stage1TargetBuilder(G1_29_SCALER, G1_29_TARGETS)

    assert "head_link" not in builder.required_robot_body_names("stage1a")
    assert "head_link" not in builder.required_robot_body_names("stage1b")


def test_contact_scores_increase_stance_foot_weights():
    motion = make_canonical_motion(num_frames=4)
    contact = FootContactResult(
        contact_score={
            "left_foot": np.array([0.0, 0.75, 0.0, 0.0]),
            "right_foot": np.array([0.0, 0.25, 0.0, 0.0]),
        },
        contact_binary={},
        foot_height={},
        foot_speed={},
        ground_height=0.0,
    )
    builder = Stage1TargetBuilder(G1_29_SCALER, G1_29_TARGETS)

    target_set = builder.build(motion, frame_idx=1, stage_name="stage1a", contact_result=contact)

    assert target_set.get_target("left_ankle").pos_weight == 57.5
    assert target_set.get_target("left_ankle").rot_weight == 0.0
    assert target_set.get_target("left_ankle").confidence == 0.75
    assert target_set.get_target("right_ankle").pos_weight == 52.5


def test_config_robot_body_names_match_local_usd_assets():
    expected = {
        G1_29_SCALER: _usd_defined_names(Path("assets/robots/unitree_g1/g1_29_dof_rubber_hand/Payload/Geometry.usda")),
        G1_23_SCALER: _usd_defined_names(Path("assets/robots/unitree_g1/g1_23_dof_rubber_hand/Payload/Geometry.usda")),
    }

    for scaler_path, usd_names in expected.items():
        builder = Stage1TargetBuilder(
            scaler_path,
            G1_29_TARGETS if scaler_path == G1_29_SCALER else G1_23_TARGETS,
        )
        missing = [name for name in builder.required_robot_body_names() if name not in usd_names]
        assert missing == []


def _usd_defined_names(path: Path) -> set[str]:
    text = path.read_text()
    return set(re.findall(r'def\s+(?:\w+\s+)?"([^"]+)"', text))
