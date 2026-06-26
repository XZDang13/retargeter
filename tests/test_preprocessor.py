from __future__ import annotations

import numpy as np

from conftest import make_canonical_motion
from retargeter.preprocess import ContactConfig, GroundConfig, LowPassConfig, MotionPreprocessor, PreprocessConfig


def test_preprocessor_preserves_length_normalizes_ground_and_estimates_contact():
    motion = make_canonical_motion(num_frames=60, foot_z=0.03)
    config = PreprocessConfig(
        lowpass=LowPassConfig(enabled=True),
        ground=GroundConfig(enabled=True),
        contact=ContactConfig(enabled=True, smooth_contact=False),
    )

    result = MotionPreprocessor(config).process(motion)

    assert result.motion.num_frames() == motion.num_frames()
    assert result.ground is not None
    assert result.contact is not None
    assert abs(result.motion.get_body_pos("left_foot")[:, 2].mean()) <= 1e-6
    assert "left_foot" in result.contact.contact_score
    assert "left_foot" in result.contact.contact_binary
    assert result.metadata["lowpass_applied"] is True
    assert result.metadata["contact_available"] is True
    assert "left_foot" in result.metadata["contact_ratio"]


def test_preprocessor_can_disable_optional_steps():
    motion = make_canonical_motion(num_frames=20, foot_z=0.03)
    config = PreprocessConfig(
        lowpass=LowPassConfig(enabled=False),
        ground=GroundConfig(enabled=False),
        contact=ContactConfig(enabled=False),
    )

    result = MotionPreprocessor(config).process(motion)

    assert result.motion.num_frames() == motion.num_frames()
    assert result.ground is None
    assert result.contact is None
    assert result.metadata["lowpass_applied"] is False
    assert result.metadata["contact_available"] is False

