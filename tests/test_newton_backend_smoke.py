from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retargeter.newton import IKState, NewtonBackend, RobotSpec


G1_29_ROBOT = Path("retargeter/newton/configs/g1_29_robot.yaml")


def test_optional_real_newton_backend_loads_g1_29_and_runs_fk():
    pytest.importorskip("newton")
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    if not spec.model_path.exists():
        pytest.skip("local Unitree G1 29 USD asset is unavailable")

    backend = NewtonBackend(spec)
    state = IKState(
        root_pos_w=np.array([0.0, 0.0, 0.793]),
        root_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        joint_pos=spec.default_joint_pos.copy(),
    )

    body_state = backend.forward_kinematics(state)

    assert body_state.body_pos_w.shape == (len(spec.body_names), 3)
    assert body_state.body_quat_xyzw.shape == (len(spec.body_names), 4)
