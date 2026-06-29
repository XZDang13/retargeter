from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from retargeter.refinement import RefinedMotion, export_refined_motion, load_refined_motion_npz


def test_refined_motion_export_npz_and_sidecars_round_trip(tmp_path: Path):
    motion = RefinedMotion(
        fps=30.0,
        robot="unitree_g1_29",
        joint_names=["j0", "j1"],
        root_pos_w=np.zeros((3, 3), dtype=np.float64),
        root_quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1)),
        joint_pos=np.zeros((3, 2), dtype=np.float64),
        joint_vel=np.ones((3, 2), dtype=np.float64),
        body_names=["pelvis", "torso_link"],
        body_pos_w=np.zeros((3, 2, 3), dtype=np.float64),
        body_quat_xyzw=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 2, 1)),
        root_delta=np.zeros((3, 3), dtype=np.float64),
        joint_delta=np.zeros((3, 2), dtype=np.float64),
        loss_curve=[{"iteration": 0, "phase": "adam", "loss": 1.0}],
        quality_metrics={"final_loss": 0.5, "iteration_count": 0},
        metadata={"source": "synthetic"},
    )

    result = export_refined_motion(
        motion,
        tmp_path / "refined.npz",
        metadata_path=tmp_path / "refined_meta.json",
        quality_path=tmp_path / "refined_quality.json",
    )
    loaded = load_refined_motion_npz(tmp_path / "refined.npz")

    assert Path(result["npz_path"]).exists()
    assert loaded.num_frames() == 3
    assert loaded.joint_names == ["j0", "j1"]
    assert loaded.body_pos_w.shape == (3, 2, 3)
    assert json.loads((tmp_path / "refined_meta.json").read_text())["metadata"]["source"] == "synthetic"
    quality = json.loads((tmp_path / "refined_quality.json").read_text())
    assert quality["quality_metrics"]["final_loss"] == 0.5
    assert quality["loss_curve"][0]["iteration"] == 0
