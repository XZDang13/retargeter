from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import make_canonical_motion
from retargeter.scale import HumanToRobotScaler
from retargeter.scale.human_to_robot_scaler import rotate_vectors_xyzw


G1_29_SCALER = Path("retargeter/scale/configs/g1_29_scaler.yaml")


def test_g1_29_scaler_preserves_shape_order_and_does_not_mutate_input():
    motion = make_canonical_motion(num_frames=4)
    original_pos = motion.body_pos_w.copy()
    original_quat = motion.body_quat_xyzw.copy()

    scaled = HumanToRobotScaler(G1_29_SCALER).scale_motion(motion)

    assert scaled.num_frames() == motion.num_frames()
    assert scaled.body_names == motion.body_names
    assert scaled.body_pos_w.shape == motion.body_pos_w.shape
    assert scaled.body_quat_xyzw.shape == motion.body_quat_xyzw.shape
    assert np.allclose(motion.body_pos_w, original_pos)
    assert np.allclose(motion.body_quat_xyzw, original_quat)
    assert np.allclose(np.linalg.norm(scaled.body_quat_xyzw, axis=-1), 1.0)
    assert scaled.metadata["scale"]["robot"] == "unitree_g1_29"


def test_scaler_applies_deterministic_scale_and_local_offset():
    motion = make_canonical_motion(num_frames=1)
    scaler = HumanToRobotScaler(G1_29_SCALER)

    scaled_once = scaler.scale_motion(motion)
    scaled_twice = scaler.scale_motion(motion)
    pelvis_idx = motion.get_body_index("pelvis")
    left_knee_idx = motion.get_body_index("left_knee")
    root_scale = scaler.scales["pelvis"]
    knee_scale = scaler.scales["left_knee"]

    expected_left_knee = (
        motion.body_pos_w[0, pelvis_idx] * root_scale
        + (motion.body_pos_w[0, left_knee_idx] - motion.body_pos_w[0, pelvis_idx]) * knee_scale
        + rotate_vectors_xyzw(scaler.rotation_offsets_xyzw["left_knee"], scaler.offsets["left_knee"])
    )

    assert np.allclose(scaled_once.body_pos_w, scaled_twice.body_pos_w)
    assert np.allclose(scaled_once.body_quat_xyzw, scaled_twice.body_quat_xyzw)
    assert np.allclose(scaled_once.body_pos_w[0, left_knee_idx], expected_left_knee)
    assert not np.allclose(scaled_once.body_pos_w[0, pelvis_idx], motion.body_pos_w[0, pelvis_idx])


def test_scaler_required_robot_bodies_excludes_fixed_head_visual():
    required = HumanToRobotScaler(G1_29_SCALER).required_robot_body_names()

    assert "head_link" not in required
    assert "torso_link" in required


def test_scaler_missing_semantic_body_raises_clear_error():
    motion = make_canonical_motion(num_frames=2)
    remove_idx = motion.get_body_index("left_toe")
    motion.body_names.pop(remove_idx)
    motion.body_pos_w = np.delete(motion.body_pos_w, remove_idx, axis=1)
    motion.body_quat_xyzw = np.delete(motion.body_quat_xyzw, remove_idx, axis=1)

    with pytest.raises(ValueError, match="Missing required semantic human bodies"):
        HumanToRobotScaler(G1_29_SCALER).scale_motion(motion)
