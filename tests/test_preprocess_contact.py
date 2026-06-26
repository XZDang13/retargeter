from __future__ import annotations

import numpy as np

from conftest import make_canonical_motion
from retargeter.preprocess import ContactConfig, FootContactEstimator


def test_static_foot_on_ground_has_high_contact_score():
    motion = make_canonical_motion(num_frames=12, foot_z=0.0)

    result = FootContactEstimator(ContactConfig(smooth_contact=False)).estimate(motion, ground_height=0.0)

    assert np.all(result.contact_score["left_foot"] > 0.9)
    assert np.all(result.contact_binary["left_foot"])


def test_airborne_foot_has_low_contact_score():
    motion = make_canonical_motion(num_frames=12, foot_z=0.3)

    result = FootContactEstimator(ContactConfig(smooth_contact=False)).estimate(motion, ground_height=0.0)

    assert np.all(result.contact_score["left_foot"] < 0.1)
    assert not np.any(result.contact_binary["left_foot"])


def test_fast_sliding_foot_near_ground_lowers_contact_score():
    static_motion = make_canonical_motion(num_frames=12, foot_z=0.0)
    sliding_motion = make_canonical_motion(num_frames=12, foot_z=0.0)
    pos = sliding_motion.get_body_pos("left_foot").copy()
    pos[:, 0] = np.linspace(0.0, 1.0, pos.shape[0])
    sliding_motion.set_body_pos("left_foot", pos)

    config = ContactConfig(smooth_contact=False)
    static_result = FootContactEstimator(config).estimate(static_motion, ground_height=0.0)
    sliding_result = FootContactEstimator(config).estimate(sliding_motion, ground_height=0.0)

    assert sliding_result.contact_score["left_foot"].mean() < static_result.contact_score["left_foot"].mean()


def test_slight_penetration_is_clamped_for_scoring():
    motion = make_canonical_motion(num_frames=12, foot_z=-0.01)

    result = FootContactEstimator(ContactConfig(smooth_contact=False)).estimate(motion, ground_height=0.0)

    assert np.all(np.isfinite(result.contact_score["left_foot"]))
    assert np.all(result.contact_score["left_foot"] > 0.9)

