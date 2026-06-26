from __future__ import annotations

import numpy as np
import pytest

from retargeter.preprocess import SMPLMotion, load_smpl_motion


def test_smpl_motion_validate_and_copy():
    motion = SMPLMotion(
        model_type="smplx",
        fps=30.0,
        transl=np.zeros((5, 3)),
        global_orient=np.zeros((5, 3)),
        body_pose=np.zeros((5, 63)),
        betas=np.zeros(10),
        metadata={"nested": {"value": 1}},
    )
    motion.validate()

    copied = motion.copy()
    copied.transl[0, 0] = 10.0
    copied.metadata["nested"]["value"] = 2

    assert motion.transl[0, 0] == 0.0
    assert motion.metadata["nested"]["value"] == 1


def test_smpl_motion_validation_rejects_bad_shapes():
    motion = SMPLMotion(
        model_type="smplx",
        fps=30.0,
        transl=np.zeros((5, 3)),
        global_orient=np.zeros((4, 3)),
        body_pose=np.zeros((5, 63)),
    )
    with pytest.raises(ValueError, match="global_orient"):
        motion.validate()


def test_load_npz_motion_maps_amass_style_keys(tmp_path):
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        trans=np.ones((4, 3), dtype=np.float32),
        root_orient=np.zeros((4, 3), dtype=np.float32),
        pose_body=np.zeros((4, 63), dtype=np.float32),
        mocap_frame_rate=np.array(60.0),
        gender=np.array("neutral"),
        surface_model_type=np.array("smplx"),
        betas=np.zeros(16, dtype=np.float32),
        pose_hand=np.zeros((4, 90), dtype=np.float32),
        pose_jaw=np.zeros((4, 3), dtype=np.float32),
        pose_eye=np.zeros((4, 6), dtype=np.float32),
    )

    motion = load_smpl_motion(path)

    assert motion.model_type == "smplx"
    assert motion.fps == 60.0
    assert motion.transl.shape == (4, 3)
    assert motion.body_pose.shape == (4, 63)
    assert motion.left_hand_pose.shape == (4, 45)
    assert motion.reye_pose.shape == (4, 3)


def test_load_phuma_npy_motion_requires_explicit_fps(tmp_path):
    path = tmp_path / "motion.npy"
    np.save(path, np.zeros((6, 69), dtype=np.float32))

    with pytest.raises(ValueError, match="fps"):
        load_smpl_motion(path)

    motion = load_smpl_motion(path, fps=30.0)
    assert motion.transl.shape == (6, 3)
    assert motion.global_orient.shape == (6, 3)
    assert motion.body_pose.shape == (6, 63)

