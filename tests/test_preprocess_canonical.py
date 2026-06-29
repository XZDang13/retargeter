from __future__ import annotations

import numpy as np
import pytest

from conftest import make_canonical_motion
from retargeter.preprocess import REQUIRED_CANONICAL_BODY_NAMES


def test_canonical_motion_accessors_and_setters():
    motion = make_canonical_motion(num_frames=5)
    motion.validate(required_bodies=REQUIRED_CANONICAL_BODY_NAMES)

    left_foot = motion.get_body_pos("left_foot")
    assert left_foot.shape == (5, 3)

    new_pos = np.ones((5, 3))
    motion.set_body_pos("left_foot", new_pos)
    assert np.allclose(motion.get_body_pos("left_foot"), new_pos)

    new_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (5, 1))
    motion.set_body_quat("left_foot", new_quat)
    assert np.allclose(motion.get_body_quat("left_foot"), new_quat)


def test_canonical_motion_copy_is_deep():
    motion = make_canonical_motion(num_frames=5)
    motion.metadata["nested"] = {"value": 1}

    copied = motion.copy()
    copied.body_pos_w[0, 0, 0] = 9.0
    copied.metadata["nested"]["value"] = 2

    assert motion.body_pos_w[0, 0, 0] != 9.0
    assert motion.metadata["nested"]["value"] == 1


def test_canonical_motion_required_body_validation():
    motion = make_canonical_motion(num_frames=5)
    motion.body_names = motion.body_names[:-1]
    motion.body_pos_w = motion.body_pos_w[:, :-1, :]
    motion.body_quat_xyzw = motion.body_quat_xyzw[:, :-1, :]

    with pytest.raises(ValueError, match="Missing required bodies"):
        motion.validate(required_bodies=REQUIRED_CANONICAL_BODY_NAMES)

