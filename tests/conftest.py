from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retargeter.preprocess import CanonicalHumanMotion, REQUIRED_CANONICAL_BODY_NAMES


def make_canonical_motion(
    num_frames: int = 60,
    fps: float = 30.0,
    foot_z: float = 0.03,
    include_vertices: bool = False,
) -> CanonicalHumanMotion:
    body_names = list(REQUIRED_CANONICAL_BODY_NAMES)
    body_pos = np.zeros((num_frames, len(body_names), 3), dtype=np.float64)
    body_quat = np.zeros((num_frames, len(body_names), 4), dtype=np.float64)
    body_quat[..., 3] = 1.0

    default_heights = {
        "pelvis": 0.9,
        "chest": 1.3,
        "head": 1.6,
        "left_shoulder": 1.35,
        "right_shoulder": 1.35,
        "left_elbow": 1.1,
        "right_elbow": 1.1,
        "left_hand": 0.9,
        "right_hand": 0.9,
        "left_hip": 0.85,
        "right_hip": 0.85,
        "left_knee": 0.45,
        "right_knee": 0.45,
        "left_ankle": foot_z,
        "right_ankle": foot_z,
        "left_foot": foot_z,
        "right_foot": foot_z,
        "left_toe": foot_z,
        "right_toe": foot_z,
        "left_heel": foot_z,
        "right_heel": foot_z,
    }

    for i, name in enumerate(body_names):
        body_pos[:, i, 0] = 0.1 * i
        body_pos[:, i, 2] = default_heights[name]

    vertices = None
    if include_vertices:
        vertices = np.zeros((num_frames, 8, 3), dtype=np.float64)
        vertices[:, :, 2] = 1.0
        vertices[:, 0:2, 2] = foot_z
        vertices[:, 2:4, 2] = foot_z
        vertices[:, 4:6, 2] = foot_z
        vertices[:, 6:8, 2] = foot_z

    return CanonicalHumanMotion(
        fps=fps,
        body_names=body_names,
        body_pos_w=body_pos,
        body_quat_xyzw=body_quat,
        vertices_w=vertices,
    )
