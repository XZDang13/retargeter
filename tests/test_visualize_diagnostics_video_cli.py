from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from conftest import make_canonical_motion
from retargeter.cli.visualize_stage1 import main as visualize_main
from retargeter.newton import IKState, RobotSpec, Stage1Motion, export_stage1_motion
from retargeter.preprocess import FootContactResult, PreprocessResult
from retargeter.visualize import (
    export_canonical_human_motion_npz,
    load_canonical_human_motion_npz,
    plot_contact_scores,
    plot_foot_height_and_speed,
    plot_frame_success,
    plot_ik_errors,
    plot_joint_limit_margin,
    plot_joint_positions,
    plot_joint_velocities,
    plot_root_height,
)


G1_29_ROBOT = Path("retargeter/newton/configs/g1_29_robot.yaml")


def test_diagnostic_plots_write_png_files(tmp_path: Path):
    human = make_canonical_motion(num_frames=6)
    preprocess = _make_preprocess_result(human)
    stage1 = _make_stage1_motion(num_frames=6)
    spec = RobotSpec.from_yaml(G1_29_ROBOT)

    paths = [
        plot_contact_scores(preprocess, tmp_path / "contact.png"),
        plot_foot_height_and_speed(preprocess, tmp_path / "foot.png"),
        plot_ik_errors(stage1, tmp_path / "ik.png"),
        plot_joint_positions(stage1, tmp_path / "q.png"),
        plot_joint_velocities(stage1, tmp_path / "qd.png"),
        plot_joint_limit_margin(stage1, spec, tmp_path / "margin.png"),
        plot_root_height(stage1, tmp_path / "root.png"),
        plot_frame_success(stage1, tmp_path / "success.png"),
    ]

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_load_canonical_human_motion_npz_round_trip(tmp_path: Path):
    motion = make_canonical_motion(num_frames=2)
    path = tmp_path / "human.npz"
    _write_human_npz(path, motion)

    loaded = load_canonical_human_motion_npz(path)

    assert loaded.num_frames() == 2
    assert loaded.body_names == motion.body_names
    assert np.allclose(loaded.body_pos_w, motion.body_pos_w)


def test_export_canonical_human_motion_npz_round_trips_mesh(tmp_path: Path):
    motion = make_canonical_motion(num_frames=2, include_vertices=True)
    motion.mesh_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    path = tmp_path / "human_mesh.npz"

    export_canonical_human_motion_npz(motion, path, require_mesh=True)
    loaded = load_canonical_human_motion_npz(path)

    assert loaded.vertices_w is not None
    assert loaded.mesh_faces is not None
    assert np.allclose(loaded.vertices_w, motion.vertices_w)
    assert np.array_equal(loaded.mesh_faces, motion.mesh_faces)


def test_visualize_cli_replay_and_diagnostics_modes(tmp_path: Path):
    human = make_canonical_motion(num_frames=2)
    human_mesh = make_canonical_motion(num_frames=2, include_vertices=True)
    human_mesh.mesh_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    stage1 = _make_stage1_motion(num_frames=2)
    human_path = tmp_path / "human.npz"
    human_mesh_path = tmp_path / "human_mesh.npz"
    stage1_path = tmp_path / "stage1.npz"
    config_path = tmp_path / "vis.yaml"
    config_path.write_text("viewer: file\nfps: 5\n", encoding="utf-8")
    _write_human_npz(human_path, human, include_contact=True)
    export_canonical_human_motion_npz(human_mesh, human_mesh_path, require_mesh=True)
    export_stage1_motion(stage1, stage1_path)

    replay_dir = tmp_path / "replay"
    assert visualize_main(
        [
            "--stage1",
            str(stage1_path),
            "--output",
            str(replay_dir),
            "--mode",
            "replay",
            "--config",
            str(config_path),
            "--robot-spec",
            str(G1_29_ROBOT),
        ],
        backend=FakeReplayBackend(RobotSpec.from_yaml(G1_29_ROBOT)),
        viewer_factory=fake_viewer_factory,
    ) == 0
    assert (replay_dir / "newton_replay.json").exists()

    overlay_created = []

    def overlay_viewer_factory(viewer, options):
        fake = FakeViewer(options.get("output_path"))
        overlay_created.append(fake)
        return fake

    overlay_dir = tmp_path / "overlay_replay"
    assert visualize_main(
        [
            "--human",
            str(human_mesh_path),
            "--stage1",
            str(stage1_path),
            "--output",
            str(overlay_dir),
            "--mode",
            "replay",
            "--config",
            str(config_path),
            "--robot-spec",
            str(G1_29_ROBOT),
            "--human-offset",
            "1,2,3",
        ],
        backend=FakeReplayBackend(RobotSpec.from_yaml(G1_29_ROBOT)),
        viewer_factory=overlay_viewer_factory,
    ) == 0
    assert (overlay_dir / "newton_replay.json").exists()
    assert len(overlay_created[0].meshes) == 2

    diagnostics_dir = tmp_path / "diagnostics"
    assert visualize_main(
        [
            "--human",
            str(human_path),
            "--stage1",
            str(stage1_path),
            "--output",
            str(diagnostics_dir),
            "--mode",
            "diagnostics",
            "--robot-spec",
            str(G1_29_ROBOT),
        ]
    ) == 0
    assert (diagnostics_dir / "contact_scores.png").exists()
    assert (diagnostics_dir / "joint_positions.png").exists()
    assert (diagnostics_dir / "joint_limit_margin.png").exists()


class FakeReplayBackend:
    def __init__(self, robot_spec: RobotSpec):
        self.robot_spec = robot_spec
        self.model = "fake-model"

    def make_newton_state(self, state: IKState):
        return {"root": state.root_pos_w.copy(), "joint_pos": state.joint_pos.copy()}


class FakeViewer:
    def __init__(self, output_path: Path | None):
        self.output_path = output_path
        self.model = None
        self.meshes = []

    def set_model(self, model):
        self.model = model

    def is_running(self):
        return True

    def should_step(self):
        return True

    def begin_frame(self, time):
        pass

    def log_state(self, state):
        pass

    def log_mesh(self, name, points, indices, **kwargs):
        self.meshes.append((name, _as_numpy(points), _as_numpy(indices), dict(kwargs)))

    def end_frame(self):
        pass

    def close(self):
        if self.output_path is not None:
            self.output_path.write_text("fake replay\n", encoding="utf-8")


def fake_viewer_factory(viewer, options):
    return FakeViewer(options.get("output_path"))


def _write_human_npz(path: Path, motion, include_contact: bool = False) -> None:
    payload = {
        "fps": np.asarray(motion.fps),
        "body_names": np.asarray(motion.body_names),
        "body_pos_w": motion.body_pos_w,
        "body_quat_xyzw": motion.body_quat_xyzw,
    }
    if include_contact:
        score = np.linspace(0.0, 1.0, motion.num_frames())
        payload.update(
            {
                "contact_score_left_foot": score,
                "contact_binary_left_foot": score > 0.5,
                "foot_height_left_foot": score * 0.02,
                "foot_speed_left_foot": score * 0.1,
                "contact_score_right_foot": score[::-1],
                "contact_binary_right_foot": score[::-1] > 0.5,
                "foot_height_right_foot": score[::-1] * 0.02,
                "foot_speed_right_foot": score[::-1] * 0.1,
                "ground_height": np.asarray(0.0),
            }
        )
    np.savez_compressed(path, **payload)


def _make_stage1_motion(num_frames: int = 5) -> Stage1Motion:
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    body_pos = np.zeros((num_frames, len(spec.body_names), 3), dtype=np.float64)
    body_quat = np.zeros((num_frames, len(spec.body_names), 4), dtype=np.float64)
    body_quat[..., 3] = 1.0
    for idx, _ in enumerate(spec.body_names):
        body_pos[:, idx, 0] = 0.03 * idx
        body_pos[:, idx, 1] = 0.02 * (idx % 3)
        body_pos[:, idx, 2] = 0.05 + 0.02 * idx
    joint_pos = np.zeros((num_frames, spec.num_dofs), dtype=np.float64)
    joint_vel = np.zeros_like(joint_pos)
    root_pos = body_pos[:, spec.body_names.index("pelvis"), :]
    root_quat = np.zeros((num_frames, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    return Stage1Motion(
        fps=30.0,
        robot=spec.robot,
        joint_names=list(spec.actuated_joints),
        root_pos_w=root_pos,
        root_quat_xyzw=root_quat,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_names=list(spec.body_names),
        body_pos_w=body_pos,
        body_quat_xyzw=body_quat,
        success=np.ones((num_frames,), dtype=bool),
        diagnostics=[
            {"stage1a": {"cost": float(i)}, "stage1b": {"cost": float(i) * 0.5}}
            for i in range(num_frames)
        ],
    )


def _make_preprocess_result(motion):
    score = np.linspace(0.0, 1.0, motion.num_frames())
    contact = FootContactResult(
        contact_score={"left_foot": score, "right_foot": score[::-1]},
        contact_binary={"left_foot": score > 0.5, "right_foot": score[::-1] > 0.5},
        foot_height={"left_foot": score * 0.02, "right_foot": score[::-1] * 0.02},
        foot_speed={"left_foot": score * 0.1, "right_foot": score[::-1] * 0.1},
        ground_height=0.0,
    )
    return PreprocessResult(motion=motion, ground=None, contact=contact, warnings=[])


def _as_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
