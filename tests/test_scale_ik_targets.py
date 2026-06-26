from __future__ import annotations

import numpy as np
import pytest

from retargeter.scale import BodyIKTarget, IKTargetSet


def test_ik_target_set_validation_and_active_filters():
    targets = [
        BodyIKTarget(
            semantic_name="pelvis",
            human_body_name="pelvis",
            robot_body_name="pelvis",
            target_pos_w=np.array([0.0, 0.0, 1.0]),
            target_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            pos_weight=1.0,
            rot_weight=0.0,
        ),
        BodyIKTarget(
            semantic_name="chest",
            human_body_name="chest",
            robot_body_name="torso_link",
            target_pos_w=None,
            target_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            pos_weight=0.0,
            rot_weight=1.0,
        ),
    ]
    target_set = IKTargetSet(stage_name="stage1a", targets=targets)

    target_set.validate()

    assert target_set.get_target("pelvis").robot_body_name == "pelvis"
    assert [target.semantic_name for target in target_set.active_position_targets()] == ["pelvis"]
    assert [target.semantic_name for target in target_set.active_rotation_targets()] == ["chest"]


def test_ik_target_set_rejects_duplicate_semantic_name():
    target = BodyIKTarget(
        semantic_name="pelvis",
        human_body_name="pelvis",
        robot_body_name="pelvis",
        target_pos_w=np.zeros(3),
        target_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pos_weight=1.0,
        rot_weight=1.0,
    )
    target_set = IKTargetSet(stage_name="stage1a", targets=[target, target])

    with pytest.raises(ValueError, match="Duplicate"):
        target_set.validate()

