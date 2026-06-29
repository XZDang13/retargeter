from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retargeter.newton import RobotBodyState, IKRetargetFrameResult, export_retargeted_motion, load_retargeted_motion_npz
from retargeter.newton import retargeted_motion_from_frames


def test_retargeted_motion_export_npz_and_sidecars_round_trip(tmp_path: Path):
    body_state = RobotBodyState(
        body_names=["pelvis", "torso_link"],
        body_pos_w=np.zeros((2, 3)),
        body_quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1)),
    )
    frames = [
        IKRetargetFrameResult(
            frame_idx=i,
            robot="unitree_g1_29",
            root_pos_w=np.array([0.0, 0.0, 0.8 + i * 0.01]),
            root_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            joint_names=["j0", "j1"],
            joint_pos=np.array([0.1 * i, 0.2 * i]),
            joint_vel=np.array([0.1, 0.2]),
            body_state=body_state,
            success=True,
            diagnostics={"frame": i},
        )
        for i in range(3)
    ]
    motion = retargeted_motion_from_frames(frames, fps=30.0, metadata={"source": "synthetic"})

    result = export_retargeted_motion(
        motion,
        tmp_path / "retargeted.npz",
        metadata_path=tmp_path / "retargeted_meta.json",
        quality_path=tmp_path / "retargeted_quality.json",
    )
    loaded = load_retargeted_motion_npz(tmp_path / "retargeted.npz")

    assert Path(result["npz_path"]).exists()
    assert loaded.num_frames() == 3
    assert loaded.joint_names == ["j0", "j1"]
    assert loaded.root_quat_xyzw.shape == (3, 4)
    assert json.loads((tmp_path / "retargeted_meta.json").read_text())["robot"] == "unitree_g1_29"
    assert json.loads((tmp_path / "retargeted_quality.json").read_text())["success_ratio"] == 1.0
