from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from conftest import make_canonical_motion
from retargeter.preprocess import FootContactResult
from retargeter.scale import IKTargetBuilder


G1_29_SCALER = Path("retargeter/scale/configs/g1_29_scaler.yaml")
G1_23_SCALER = Path("retargeter/scale/configs/g1_23_scaler.yaml")
G1_29_TARGETS = Path("retargeter/scale/configs/g1_29_ik_targets.yaml")
G1_23_TARGETS = Path("retargeter/scale/configs/g1_23_ik_targets.yaml")


def test_g1_29_full_body_tracking_builder_returns_expected_targets():
    motion = make_canonical_motion(num_frames=5)
    builder = IKTargetBuilder(G1_29_SCALER, G1_29_TARGETS)

    target_set = builder.build(motion, frame_idx=2, pass_name="full_body_tracking")

    assert target_set.pass_name == "full_body_tracking"
    assert [target.semantic_name for target in target_set.targets] == [
        "pelvis",
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_foot",
        "left_toe",
        "left_heel",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_foot",
        "right_toe",
        "right_heel",
        "chest",
        "left_shoulder",
        "left_elbow",
        "left_hand",
        "right_shoulder",
        "right_elbow",
        "right_hand",
    ]
    assert target_set.get_target("left_hand").robot_body_name == "left_wrist_yaw_link"
    assert target_set.get_target("right_ankle").robot_body_name == "right_ankle_pitch_link"
    assert target_set.get_target("right_ankle").human_body_name == "right_ankle"
    assert target_set.get_target("right_ankle").target_pos_w.shape == (3,)
    assert target_set.get_target("right_ankle").target_quat_xyzw is None


def test_gmr_style_hands_and_phuma_soma_style_shoulders_for_both_g1_variants():
    motion = make_canonical_motion(num_frames=3)
    target_29 = IKTargetBuilder(G1_29_SCALER, G1_29_TARGETS).build(motion, 0, "full_body_tracking")
    target_23 = IKTargetBuilder(G1_23_SCALER, G1_23_TARGETS).build(motion, 0, "full_body_tracking")

    assert target_29.get_target("left_hand").pos_weight == 10.0
    assert target_29.get_target("left_hand").rot_weight == 5.0
    assert target_29.get_target("left_hand").target_pos_w.shape == (3,)
    assert target_29.get_target("left_hand").target_quat_xyzw.shape == (4,)
    assert target_23.get_target("left_hand").robot_body_name == "left_rubber_hand_link"
    assert target_23.get_target("right_hand").robot_body_name == "right_rubber_hand_link"
    assert target_29.get_target("left_shoulder").robot_body_name == "left_shoulder_roll_link"
    assert target_23.get_target("right_shoulder").robot_body_name == "right_shoulder_roll_link"


def test_full_body_tracking_includes_gmr_style_limb_targets():
    motion = make_canonical_motion(num_frames=3)
    target_set = IKTargetBuilder(G1_29_SCALER, G1_29_TARGETS).build(motion, 0, "full_body_tracking")

    assert target_set.get_target("left_hip").robot_body_name == "left_hip_roll_link"
    assert target_set.get_target("right_hip").robot_body_name == "right_hip_roll_link"
    assert target_set.get_target("left_ankle").human_body_name == "left_ankle"
    assert target_set.get_target("left_ankle").robot_body_name == "left_ankle_pitch_link"
    assert target_set.get_target("right_ankle").human_body_name == "right_ankle"
    assert target_set.get_target("right_ankle").robot_body_name == "right_ankle_pitch_link"
    assert target_set.get_target("left_shoulder").robot_body_name == "left_shoulder_roll_link"
    assert target_set.get_target("right_shoulder").robot_body_name == "right_shoulder_roll_link"


def test_full_body_tracking_uses_skeleton_foot_targets_and_position_only_contact_keypoints():
    motion = make_canonical_motion(num_frames=3)
    target_set = IKTargetBuilder(G1_29_SCALER, G1_29_TARGETS).build(motion, 0, "full_body_tracking")

    left_foot = target_set.get_target("left_foot")
    assert left_foot.human_body_name == "left_foot"
    assert left_foot.robot_body_name == "left_toe_link"
    assert left_foot.robot_local_pos is None
    assert left_foot.pos_weight == 100.0
    assert left_foot.rot_weight == 5.0
    assert left_foot.target_pos_w.shape == (3,)
    assert left_foot.target_quat_xyzw.shape == (4,)
    assert left_foot.metadata["position_source"] == {"type": "body", "body": "left_foot"}
    assert left_foot.metadata["rotation_source"] == {"type": "body", "body": "left_foot"}

    left_toe = target_set.get_target("left_toe")
    assert left_toe.human_body_name == "left_toe"
    assert left_toe.robot_body_name == "left_ankle_roll_link"
    assert np.allclose(left_toe.robot_local_pos, [0.142362, 0.000048, -0.034586])
    assert left_toe.pos_weight == 0.0
    assert left_toe.rot_weight == 0.0
    assert left_toe.target_pos_w is None
    assert left_toe.target_quat_xyzw is None
    assert target_set.get_target("right_foot").robot_body_name == "right_toe_link"
    assert np.allclose(target_set.get_target("right_heel").robot_local_pos, [-0.065841, -0.000051, -0.034624])


def test_default_scaler_configs_do_not_use_mesh_sole_sources():
    for scaler_path in (G1_23_SCALER, G1_29_SCALER):
        text = scaler_path.read_text()
        assert "foot_sole" not in text
        assert "human_rotation_source" not in text


def test_stage_required_robot_bodies_exclude_non_target_head_link():
    builder = IKTargetBuilder(G1_29_SCALER, G1_29_TARGETS)

    assert "head_link" not in builder.required_robot_body_names("full_body_tracking")


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
    builder = IKTargetBuilder(G1_29_SCALER, G1_29_TARGETS)

    target_set = builder.build(motion, frame_idx=1, pass_name="full_body_tracking", contact_result=contact)

    assert target_set.get_target("left_foot").pos_weight == 115.0
    assert target_set.get_target("left_foot").rot_weight == 5.0
    assert target_set.get_target("left_foot").confidence == 0.75
    assert target_set.get_target("right_foot").pos_weight == 105.0
    assert target_set.get_target("left_ankle").pos_weight == 50.0
    assert target_set.get_target("left_ankle").confidence == 1.0


def test_contact_scores_activate_position_only_toe_heel_keypoints():
    motion = make_canonical_motion(num_frames=4)
    contact = FootContactResult(
        contact_score={
            "left_toe": np.array([0.0, 0.80, 0.0, 0.0]),
            "left_heel": np.array([0.0, 0.50, 0.0, 0.0]),
            "right_toe": np.array([0.0, 0.25, 0.0, 0.0]),
            "right_heel": np.array([0.0, 0.00, 0.0, 0.0]),
        },
        contact_binary={},
        foot_height={},
        foot_speed={},
        ground_height=0.0,
    )
    builder = IKTargetBuilder(G1_29_SCALER, G1_29_TARGETS)

    target_set = builder.build(motion, frame_idx=1, pass_name="full_body_tracking", contact_result=contact)

    assert target_set.get_target("left_toe").pos_weight == 40.0
    assert target_set.get_target("left_toe").rot_weight == 0.0
    assert target_set.get_target("left_toe").target_pos_w.shape == (3,)
    assert target_set.get_target("left_toe").target_quat_xyzw is None
    assert target_set.get_target("left_toe").confidence == 0.8
    assert target_set.get_target("left_heel").pos_weight == 25.0
    assert target_set.get_target("right_toe").pos_weight == 12.5
    assert target_set.get_target("right_heel").pos_weight == 0.0
    assert target_set.get_target("right_heel").target_pos_w.shape == (3,)
    assert target_set.get_target("right_heel").metadata["can_activate_position"] is True
    assert target_set.get_target("left_foot").pos_weight == 100.0
    assert target_set.get_target("left_foot").confidence == 1.0


def test_config_robot_body_names_match_local_usd_assets():
    expected = {
        G1_29_SCALER: _usd_defined_names(Path("assets/robots/unitree_g1/g1_29_dof/Payload/Geometry.usda")),
        G1_23_SCALER: _usd_defined_names(Path("assets/robots/unitree_g1/g1_23_dof/Payload/Geometry.usda")),
    }

    for scaler_path, usd_names in expected.items():
        builder = IKTargetBuilder(
            scaler_path,
            G1_29_TARGETS if scaler_path == G1_29_SCALER else G1_23_TARGETS,
        )
        missing = [name for name in builder.required_robot_body_names() if name not in usd_names]
        assert missing == []


def _usd_defined_names(path: Path) -> set[str]:
    text = path.read_text()
    return set(re.findall(r'def\s+(?:\w+\s+)?"([^"]+)"', text))
