from __future__ import annotations

import numpy as np

from conftest import make_canonical_motion
from retargeter.preprocess import GroundConfig, GroundPlaneEstimator
from retargeter.preprocess.canonical import CanonicalHumanMotion


SMALL_VERTEX_CONFIG = {
    "left_toe_indices": [0, 1],
    "left_heel_indices": [2, 3],
    "right_toe_indices": [4, 5],
    "right_heel_indices": [6, 7],
}


def test_majority_vote_estimates_standing_foot_height_from_vertices():
    motion = make_canonical_motion(num_frames=20, foot_z=0.03, include_vertices=True)
    config = GroundConfig(foot_vertex_indices=SMALL_VERTEX_CONFIG)

    estimate = GroundPlaneEstimator(config).estimate(motion)

    assert estimate.method == "majority_vote"
    assert abs(estimate.ground_height - 0.03) <= 0.005
    assert estimate.confidence > 0.0
    assert estimate.metadata["source"] == "vertices"


def test_jump_frames_do_not_dominate_majority_vote():
    motion = make_canonical_motion(num_frames=20, foot_z=0.03, include_vertices=False)
    for name in ["left_foot", "right_foot", "left_toe", "right_toe", "left_heel", "right_heel"]:
        pos = motion.get_body_pos(name).copy()
        pos[-5:, 2] = 0.8
        motion.set_body_pos(name, pos)

    estimate = GroundPlaneEstimator(GroundConfig()).estimate(motion)

    assert abs(estimate.ground_height - 0.03) <= 0.005
    assert estimate.metadata["source"] == "bodies"


def test_missing_vertices_falls_back_to_body_positions():
    motion = make_canonical_motion(num_frames=10, foot_z=0.04, include_vertices=False)

    estimate = GroundPlaneEstimator(GroundConfig()).estimate(motion)

    assert abs(estimate.ground_height - 0.04) <= 0.005
    assert estimate.metadata["source"] == "bodies"


def test_no_foot_data_returns_fixed_ground_low_confidence():
    motion = CanonicalHumanMotion(
        fps=30.0,
        body_names=["pelvis"],
        body_pos_w=np.zeros((5, 1, 3)),
        body_quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (5, 1, 1)),
    )

    estimate = GroundPlaneEstimator(GroundConfig(fixed_ground_height=0.2)).estimate(motion)

    assert estimate.ground_height == 0.2
    assert estimate.confidence == 0.0
    assert estimate.method == "fixed_fallback"

