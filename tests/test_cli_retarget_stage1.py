from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import make_canonical_motion
from retargeter.cli import retarget_stage1
from retargeter.newton import BackendSolveResult, IKState, NewtonSolveSettings, RobotBodyState, RobotSpec


class MockBackend:
    def __init__(self, robot_spec: RobotSpec):
        self.robot_spec = robot_spec
        self.calls = []

    def solve_ik(self, seed_state: IKState, objectives, settings: NewtonSolveSettings) -> BackendSolveResult:
        call_idx = len(self.calls)
        self.calls.append({"seed": seed_state.copy(), "objectives": list(objectives), "settings": settings})
        q = seed_state.joint_pos.copy() + 0.001 * (call_idx + 1)
        return BackendSolveResult(
            state=IKState(
                root_pos_w=seed_state.root_pos_w.copy(),
                root_quat_xyzw=seed_state.root_quat_xyzw.copy(),
                joint_pos=q,
            ),
            success=True,
            cost=float(call_idx),
            iterations=settings.iterations,
            diagnostics={"mock_call_idx": call_idx},
        )

    def forward_kinematics(self, state: IKState) -> RobotBodyState:
        body_pos = np.zeros((len(self.robot_spec.body_names), 3), dtype=np.float64)
        body_quat = np.zeros((len(self.robot_spec.body_names), 4), dtype=np.float64)
        body_quat[:, 3] = 1.0
        body_pos[:] = state.root_pos_w
        body_pos[:, 2] += np.linspace(0.0, 0.6, len(self.robot_spec.body_names))
        return RobotBodyState(list(self.robot_spec.body_names), body_pos, body_quat)

    @property
    def model(self):
        return "mock-newton-model"

    def make_newton_state(self, state: IKState):
        return {"root": state.root_pos_w.copy(), "joint_pos": state.joint_pos.copy()}


class MockNewtonViewer:
    def __init__(self):
        self.model = None
        self.states = []
        self.closed = False

    def set_model(self, model):
        self.model = model

    def is_running(self):
        return True

    def should_step(self):
        return True

    def begin_frame(self, time):
        self.time = time

    def log_state(self, state):
        self.states.append(state)

    def log_mesh(self, name, points, indices, **kwargs):
        pass

    def end_frame(self):
        pass

    def close(self):
        self.closed = True


def mock_viewer_factory(viewer, options):
    output_path = options.get("output_path")
    if output_path is not None:
        Path(output_path).write_text("mock newton replay\n", encoding="utf-8")
    return MockNewtonViewer()


def test_retarget_stage1_mock_mode_end_to_end_with_visualization(tmp_path: Path):
    vis_config = tmp_path / "vis.yaml"
    vis_config.write_text("output_resolution: [220, 160]\nfps: 5\n", encoding="utf-8")
    output = tmp_path / "mock_stage1"

    exit_code = retarget_stage1.main(
        [
            "--input",
            "mock",
            "--robot",
            "unitree_g1_29",
            "--output",
            str(output),
            "--mock-frames",
            "3",
            "--visualize",
            "1",
            "--visualize-config",
            str(vis_config),
            "--visualize-fps",
            "5",
        ],
        backend_factory=MockBackend,
        viewer_backend_factory=MockBackend,
        viewer_factory=mock_viewer_factory,
    )

    assert exit_code == 0
    expected = [
        "motion.npz",
        "meta.yaml",
        "quality.json",
        "newton_replay.json",
        "contact_scores.png",
        "foot_height_speed.png",
        "ik_errors.png",
        "joint_positions.png",
        "joint_velocities.png",
        "joint_limit_margin.png",
        "root_height.png",
        "frame_success.png",
    ]
    for name in expected:
        path = output / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    meta = yaml.safe_load((output / "meta.yaml").read_text(encoding="utf-8"))
    assert meta["robot"] == "unitree_g1_29"
    assert meta["frame_count"] == 3
    assert meta["metadata"]["source"]["mock_mode"] is True


def test_retarget_stage1_infers_g1_23_configs():
    args = retarget_stage1._build_parser().parse_args(
        [
            "--input",
            "mock",
            "--robot",
            "g1_23",
            "--output",
            "unused",
        ]
    )

    robot = retarget_stage1.normalize_robot_name(args.robot)
    configs = retarget_stage1.resolve_pipeline_configs(args, robot)

    assert robot == "unitree_g1_23"
    assert configs["scaler_config"].name == "g1_23_scaler.yaml"
    assert configs["target_config"].name == "g1_23_stage1_targets.yaml"
    assert configs["newton_config"].name == "g1_23_newton_stage1.yaml"


def test_retarget_stage1_passes_selected_target_builder_to_solver(tmp_path: Path):
    args = retarget_stage1._build_parser().parse_args(
        [
            "--input",
            "mock",
            "--robot",
            "unitree_g1_29",
            "--output",
            str(tmp_path / "out"),
            "--mock-frames",
            "2",
            "--visualize",
            "0",
        ]
    )

    result = retarget_stage1.run_stage1_pipeline(args, backend_factory=MockBackend)

    assert result["solver"].target_builder is result["target_builder"]
    assert result["stage1_motion"].num_frames() == 2


def test_retarget_stage1_human_output_exports_preprocessed_mesh(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "motion.npz"
    input_path.write_bytes(b"unused")
    human_output = tmp_path / "human.npz"

    def fake_real_input_pipeline(args, preprocess_config):
        motion = make_canonical_motion(num_frames=2, include_vertices=True)
        motion.mesh_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        preprocess_result = retarget_stage1.MotionPreprocessor(preprocess_config).process(motion)
        return motion, preprocess_result, {"input": str(args.input), "mock_mode": False, "smpl_fk_applied": True}

    monkeypatch.setattr(retarget_stage1, "run_real_input_pipeline", fake_real_input_pipeline)
    args = retarget_stage1._build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--robot",
            "g1_23",
            "--output",
            str(tmp_path / "out"),
            "--human-output",
            str(human_output),
        ]
    )

    result = retarget_stage1.run_stage1_pipeline(args, backend_factory=MockBackend)

    assert result["human_path"] == human_output
    with np.load(human_output, allow_pickle=False) as data:
        assert "vertices_w" in data
        assert "mesh_faces" in data
        assert data["vertices_w"].shape[:2] == (2, 8)
        assert np.array_equal(data["mesh_faces"], np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32))


def test_retarget_stage1_human_output_requires_real_vertices(tmp_path: Path):
    args = retarget_stage1._build_parser().parse_args(
        [
            "--input",
            "mock",
            "--output",
            str(tmp_path / "out"),
            "--human-output",
            str(tmp_path / "human.npz"),
        ]
    )

    with pytest.raises(ValueError, match="requires a real SMPL/SMPL-X input"):
        retarget_stage1.run_stage1_pipeline(args, backend_factory=MockBackend)


def test_retarget_stage1_real_mode_missing_input_raises_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Input motion file does not exist"):
        retarget_stage1.main(
            [
                "--input",
                str(tmp_path / "missing.npz"),
                "--output",
                str(tmp_path / "out"),
            ],
            backend_factory=MockBackend,
        )


def test_retarget_stage1_real_mode_npy_without_fps_raises(tmp_path: Path):
    path = tmp_path / "motion.npy"
    np.save(path, np.zeros((2, 69), dtype=np.float64))

    with pytest.raises(ValueError, match="does not carry fps"):
        retarget_stage1.main(
            [
                "--input",
                str(path),
                "--output",
                str(tmp_path / "out"),
            ],
            backend_factory=MockBackend,
        )


def test_retarget_stage1_real_mode_missing_smpl_model_dir_raises(tmp_path: Path):
    path = tmp_path / "motion.npz"
    np.savez_compressed(
        path,
        transl=np.zeros((2, 3), dtype=np.float64),
        global_orient=np.zeros((2, 3), dtype=np.float64),
        body_pose=np.zeros((2, 63), dtype=np.float64),
        mocap_frame_rate=np.asarray(30.0),
    )

    with pytest.raises(FileNotFoundError, match="SMPL model directory does not exist"):
        retarget_stage1.main(
            [
                "--input",
                str(path),
                "--smpl-model-dir",
                str(tmp_path / "missing_models"),
                "--output",
                str(tmp_path / "out"),
            ],
            backend_factory=MockBackend,
        )
