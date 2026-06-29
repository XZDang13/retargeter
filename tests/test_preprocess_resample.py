from __future__ import annotations

import numpy as np
import pytest

from retargeter.preprocess import SMPLMotion, resample_smpl_motion


def test_resample_smpl_motion_downsamples_to_target_fps():
    motion = _make_motion(num_frames=7, fps=60.0)

    resampled = resample_smpl_motion(motion, target_fps=30.0)

    assert resampled.fps == 30.0
    assert resampled.num_frames() == 4
    assert np.allclose(resampled.transl[:, 0], [0.0, 2.0, 4.0, 6.0])
    assert resampled.metadata["resample"] == {
        "resampled": True,
        "source_fps": 60.0,
        "target_fps": 30.0,
        "source_frame_count": 7,
        "target_frame_count": 4,
    }


def test_resample_smpl_motion_handles_non_integer_ratio_with_finite_rotations():
    motion = _make_motion(num_frames=6, fps=50.0)

    resampled = resample_smpl_motion(motion, target_fps=30.0)

    assert resampled.fps == 30.0
    assert resampled.num_frames() == 4
    assert resampled.transl.shape == (4, 3)
    assert resampled.global_orient.shape == (4, 3)
    assert resampled.body_pose.shape == (4, 63)
    assert np.all(np.isfinite(resampled.global_orient))
    assert np.all(np.isfinite(resampled.body_pose))


def test_resample_smpl_motion_resamples_optional_fields_and_frame_betas():
    motion = _make_motion(num_frames=5, fps=40.0)
    motion.betas = np.arange(5 * 16, dtype=np.float64).reshape(5, 16)
    motion.left_hand_pose = np.zeros((5, 45), dtype=np.float64)
    motion.right_hand_pose = np.zeros((5, 45), dtype=np.float64)
    motion.jaw_pose = np.zeros((5, 3), dtype=np.float64)
    motion.leye_pose = np.zeros((5, 3), dtype=np.float64)
    motion.reye_pose = np.zeros((5, 3), dtype=np.float64)
    motion.expression = np.arange(5 * 10, dtype=np.float64).reshape(5, 10)
    motion.left_hand_pose[:, 2::3] = np.linspace(0.0, 0.5, 5)[:, None]

    resampled = resample_smpl_motion(motion, target_fps=20.0)

    assert resampled.num_frames() == 3
    assert resampled.betas.shape == (3, 16)
    assert resampled.left_hand_pose.shape == (3, 45)
    assert resampled.right_hand_pose.shape == (3, 45)
    assert resampled.jaw_pose.shape == (3, 3)
    assert resampled.leye_pose.shape == (3, 3)
    assert resampled.reye_pose.shape == (3, 3)
    assert resampled.expression.shape == (3, 10)
    assert np.all(np.isfinite(resampled.left_hand_pose))


def test_resample_smpl_motion_preserves_static_betas():
    motion = _make_motion(num_frames=5, fps=40.0)
    motion.betas = np.arange(16, dtype=np.float64)
    assert resample_smpl_motion(motion, target_fps=20.0).betas.shape == (16,)

    motion.betas = np.arange(16, dtype=np.float64)[None, :]
    assert resample_smpl_motion(motion, target_fps=20.0).betas.shape == (1, 16)


def test_resample_smpl_motion_rejects_invalid_target_fps():
    motion = _make_motion(num_frames=5, fps=40.0)

    with pytest.raises(ValueError, match="target_fps"):
        resample_smpl_motion(motion, target_fps=0.0)


def _make_motion(num_frames: int, fps: float) -> SMPLMotion:
    frame = np.arange(num_frames, dtype=np.float64)
    transl = np.stack([frame, frame * 0.5, np.zeros_like(frame)], axis=1)
    global_orient = np.zeros((num_frames, 3), dtype=np.float64)
    global_orient[:, 2] = np.linspace(0.0, np.pi / 3.0, num_frames)
    body_pose = np.zeros((num_frames, 63), dtype=np.float64)
    body_pose[:, 2] = np.linspace(0.0, np.pi / 6.0, num_frames)
    return SMPLMotion(
        model_type="smplx",
        fps=fps,
        transl=transl,
        global_orient=global_orient,
        body_pose=body_pose,
        betas=np.zeros(16, dtype=np.float64),
        gender="neutral",
    )
