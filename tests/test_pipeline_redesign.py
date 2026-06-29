from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from retargeter.cli import online, refine, viewer
from retargeter.newton import BackendSolveResult, IKState, NewtonSolveSettings, RobotBodyState, RobotSpec
from retargeter.pipeline import OnlineRetargeter, RefinePipeline, ViewerPipeline
from retargeter.refinement import RefinedMotion, export_refined_motion
from retargeter.newton import export_retargeted_motion
from retargeter.newton import RetargetedMotion
from retargeter.newton import TorchRobotFKResult


G1_29_ROBOT = Path("retargeter/newton/configs/g1_29_robot.yaml")


class MockBackend:
    def __init__(self, robot_spec: RobotSpec):
        self.robot_spec = robot_spec
        self.calls = []

    @property
    def model(self):
        return "mock-newton-model"

    def solve_ik(self, seed_state: IKState, objectives, settings: NewtonSolveSettings) -> BackendSolveResult:
        call_idx = len(self.calls)
        self.calls.append({"seed": seed_state.copy(), "objectives": list(objectives), "settings": settings})
        q = seed_state.joint_pos.copy() + 0.01 * (call_idx + 1)
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
        body_pos[:, 2] += _mock_robot_body_z_offsets(self.robot_spec.body_names)
        return RobotBodyState(list(self.robot_spec.body_names), body_pos, body_quat)

    def make_newton_state(self, state: IKState):
        return {"root": state.root_pos_w.copy(), "joint_pos": state.joint_pos.copy()}


class FakeTorchFK(torch.nn.Module):
    def __init__(self, robot_spec: RobotSpec, *, x_offset: float = 0.0):
        super().__init__()
        self.robot_spec = robot_spec
        self.body_names = list(robot_spec.body_names)
        self.x_offset = float(x_offset)

    def forward(self, root_pos: torch.Tensor, root_quat_xyzw: torch.Tensor, joint_pos: torch.Tensor) -> TorchRobotFKResult:
        offsets = torch.zeros((len(self.body_names), 3), dtype=root_pos.dtype, device=root_pos.device)
        offsets[:, 0] = self.x_offset
        offsets[:, 2] = torch.as_tensor(_mock_robot_body_z_offsets(self.body_names), dtype=root_pos.dtype, device=root_pos.device)
        body_pos = root_pos[:, None, :] + offsets[None, :, :]
        body_quat = root_quat_xyzw[:, None, :].expand(-1, len(self.body_names), -1)
        return TorchRobotFKResult(body_names=list(self.body_names), body_pos_w=body_pos, body_quat_xyzw=body_quat)


class FakeViewer:
    def __init__(self, output_path: Path | None):
        self.output_path = output_path
        self.model = None
        self.states = []
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
        self.states.append(state)

    def log_mesh(self, name, points, indices, **kwargs):
        self.meshes.append((name, points, indices, kwargs))

    def end_frame(self):
        pass

    def close(self):
        if self.output_path is not None:
            self.output_path.write_text("fake replay\n", encoding="utf-8")


def fake_viewer_factory(viewer_kind, options):
    return FakeViewer(options.get("output_path"))


def _mock_robot_body_z_offsets(body_names) -> np.ndarray:
    offsets = []
    for name in body_names:
        if name == "pelvis":
            offsets.append(0.0)
        elif "toe" in name or "ankle" in name:
            offsets.append(-0.72)
        elif "knee" in name:
            offsets.append(-0.40)
        elif "hip" in name:
            offsets.append(-0.12)
        elif "waist" in name:
            offsets.append(0.16)
        elif "torso" in name:
            offsets.append(0.38)
        elif "shoulder" in name:
            offsets.append(0.42)
        elif "elbow" in name:
            offsets.append(0.20)
        elif "wrist" in name or "hand" in name:
            offsets.append(0.05)
        else:
            offsets.append(0.0)
    return np.asarray(offsets, dtype=np.float64)


def test_online_retargeter_step_uses_warm_start_and_reset():
    backend_instances = []

    def factory(spec):
        backend = MockBackend(spec)
        backend_instances.append(backend)
        return backend

    retargeter = OnlineRetargeter(robot="g1_29", backend_factory=factory)
    motion = _make_canonical_motion(frames=2)

    first = retargeter.step(motion, 0)
    retargeter.step(motion, 1)

    backend = backend_instances[0]
    assert len(backend.calls) == 4
    np.testing.assert_allclose(backend.calls[2]["seed"].joint_pos, first.joint_pos)

    retargeter.reset()
    retargeter.step(motion, 0)
    assert retargeter.runner.frame_count == 1


def test_online_cli_writes_online_layout_only(tmp_path: Path):
    output = tmp_path / "online"
    assert online.main(
        ["--input", "mock", "--output", str(output), "--mock-frames", "2"],
        backend_factory=MockBackend,
    ) == 0

    assert (output / "online_motion.npz").exists()
    assert (output / "online_meta.yaml").exists()
    assert (output / "online_quality.json").exists()
    assert not (output / "final_motion.npz").exists()
    assert not (output / "retargeted_motion.npz").exists()


def test_refine_cli_writes_final_training_layout(tmp_path: Path):
    output = tmp_path / "refine"
    assert refine.main(
        [
            "--input",
            "mock",
            "--output",
            str(output),
            "--mock-frames",
            "2",
            "--refinement-iterations",
            "0",
            "--refinement-dtype",
            "float64",
            "--refinement-lbfgs",
            "0",
        ],
        backend_factory=MockBackend,
        refinement_fk_factory=lambda spec: FakeTorchFK(spec),
    ) == 0

    for name in (
        "retargeted_motion.npz",
        "retargeted_meta.yaml",
        "retargeted_quality.json",
        "final_motion.npz",
        "final_meta.yaml",
        "final_quality.json",
    ):
        assert (output / name).exists(), name
    assert not (output / "motion.npz").exists()
    assert not (output / "refinement_motion.npz").exists()


def test_refine_cli_batch_writes_per_input_layout_and_manifest(tmp_path: Path):
    output = tmp_path / "batch"
    assert refine.main(
        [
            "--inputs",
            "mock",
            "mock",
            "--output",
            str(output),
            "--mock-frames",
            "2",
            "--refinement-iterations",
            "0",
            "--refinement-dtype",
            "float64",
            "--refinement-lbfgs",
            "0",
        ],
        backend_factory=MockBackend,
        refinement_fk_factory=lambda spec: FakeTorchFK(spec),
    ) == 0

    for item_dir in (output / "mock", output / "mock__2"):
        for name in (
            "retargeted_motion.npz",
            "retargeted_meta.yaml",
            "retargeted_quality.json",
            "final_motion.npz",
            "final_meta.yaml",
            "final_quality.json",
        ):
            assert (item_dir / name).exists(), f"{item_dir / name}"

    manifest = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pipeline"] == "refine_batch"
    assert manifest["input_count"] == 2
    assert manifest["success_count"] == 2
    assert manifest["failure_count"] == 0
    assert [Path(item["output_dir"]).name for item in manifest["items"]] == ["mock", "mock__2"]
    assert [item["status"] for item in manifest["items"]] == ["success", "success"]


def test_refine_cli_batch_records_failure_and_continues(tmp_path: Path):
    output = tmp_path / "batch_partial"
    missing = tmp_path / "missing.npz"
    assert refine.main(
        [
            "--inputs",
            str(missing),
            "mock",
            "--output",
            str(output),
            "--mock-frames",
            "2",
            "--refinement-iterations",
            "0",
            "--refinement-dtype",
            "float64",
            "--refinement-lbfgs",
            "0",
        ],
        backend_factory=MockBackend,
        refinement_fk_factory=lambda spec: FakeTorchFK(spec),
    ) == 1

    assert (output / "mock" / "final_motion.npz").exists()
    manifest = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_count"] == 2
    assert manifest["success_count"] == 1
    assert manifest["failure_count"] == 1
    assert [item["status"] for item in manifest["items"]] == ["failed", "success"]
    assert manifest["items"][0]["error_type"] == "FileNotFoundError"
    assert manifest["items"][1]["error"] is None


def test_refine_cli_batch_input_list_dry_run_and_summary_csv(tmp_path: Path):
    output = tmp_path / "batch_plan"
    summary_csv = output / "summary.csv"
    input_list = tmp_path / "inputs.txt"
    input_list.write_text("\n# comment\nmock\nmock\n", encoding="utf-8")

    assert refine.main(
        [
            "--input-list",
            str(input_list),
            "--output",
            str(output),
            "--dry-run",
            "--summary-csv",
            str(summary_csv),
        ],
        backend_factory=MockBackend,
        refinement_fk_factory=lambda spec: FakeTorchFK(spec),
    ) == 0

    manifest = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
    assert [Path(item["output_dir"]).name for item in manifest["items"]] == ["mock", "mock__2"]
    assert [item["status"] for item in manifest["items"]] == ["pending", "pending"]
    assert summary_csv.exists()
    assert summary_csv.read_text(encoding="utf-8").splitlines()[0].startswith("input,output_dir,status")
    assert not (output / "mock").exists()


def test_refine_cli_batch_preserve_tree_dry_run(tmp_path: Path):
    input_dir = tmp_path / "data"
    nested = input_dir / "a"
    nested.mkdir(parents=True)
    walk = nested / "walk.npz"
    walk.touch()
    output = tmp_path / "batch_tree"

    assert refine.main(
        [
            "--input-dir",
            str(input_dir),
            "--input-pattern",
            "*.npz",
            "--recursive",
            "--preserve-tree",
            "--output",
            str(output),
            "--dry-run",
        ]
    ) == 0

    manifest = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_count"] == 1
    assert Path(manifest["items"][0]["output_dir"]) == output / "a" / "walk"
    assert manifest["items"][0]["status"] == "pending"


def test_refine_pipeline_run_batch_returns_lightweight_results(tmp_path: Path):
    result = RefinePipeline(
        robot="g1_29",
        backend_factory=MockBackend,
        refinement_fk_factory=lambda spec: FakeTorchFK(spec),
    ).run_batch(
        input_paths=["mock", "mock"],
        output_dir=tmp_path / "api_batch",
        mock_frames=2,
        refinement_config={"refiner": {"iterations": 0, "dtype": "float64", "lbfgs_enabled": False}},
    )

    assert result.success_count == 2
    assert result.failure_count == 0
    assert result.manifest_path.exists()
    assert [item.output_dir.name for item in result.items] == ["mock", "mock__2"]
    assert [item.frame_count for item in result.items] == [2, 2]
    assert all(item.success for item in result.items)
    assert all((item.output_dir / "final_motion.npz").exists() for item in result.items)


def test_refine_rejects_invalid_quality_unless_allowed(tmp_path: Path):
    with pytest.raises(RuntimeError, match="RefinementQualityReport"):
        RefinePipeline(
            robot="g1_29",
            backend_factory=MockBackend,
            refinement_fk_factory=lambda spec: FakeTorchFK(spec, x_offset=1.0),
        ).run(
            input_path="mock",
            output_dir=tmp_path / "invalid",
            mock_frames=2,
            refinement_config={"refiner": {"iterations": 0, "dtype": "float64"}},
        )

    result = RefinePipeline(
        robot="g1_29",
        backend_factory=MockBackend,
        refinement_fk_factory=lambda spec: FakeTorchFK(spec, x_offset=1.0),
    ).run(
        input_path="mock",
        output_dir=tmp_path / "allowed",
        mock_frames=2,
        refinement_config={"refiner": {"iterations": 0, "dtype": "float64"}},
        allow_invalid=True,
    )
    assert result.quality_report.valid is False
    assert (tmp_path / "allowed" / "final_motion.npz").exists()


def test_refine_physical_feasibility_failure_exports_invalid_quality_and_allow_invalid(tmp_path: Path):
    config = {
        "refiner": {"iterations": 0, "dtype": "float64", "lbfgs_enabled": False},
        "physical_feasibility": {"fail_on_pelvis_height": True, "min_pelvis_height_m": 10.0},
    }
    invalid_output = tmp_path / "physical_invalid"

    with pytest.raises(RuntimeError, match="RefinementQualityReport"):
        RefinePipeline(
            robot="g1_29",
            backend_factory=MockBackend,
            refinement_fk_factory=lambda spec: FakeTorchFK(spec),
        ).run(
            input_path="mock",
            output_dir=invalid_output,
            mock_frames=2,
            refinement_config=config,
        )

    invalid_quality = json.loads((invalid_output / "final_quality.json").read_text(encoding="utf-8"))
    invalid_report = invalid_quality["quality_report"]
    assert invalid_report["valid"] is False
    assert "pelvis_height_too_low" in invalid_report["failures"]

    config_path = tmp_path / "physical_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    allowed_output = tmp_path / "physical_allowed"
    assert refine.main(
        [
            "--input",
            "mock",
            "--output",
            str(allowed_output),
            "--mock-frames",
            "2",
            "--refinement-config",
            str(config_path),
            "--allow-invalid",
        ],
        backend_factory=MockBackend,
        refinement_fk_factory=lambda spec: FakeTorchFK(spec),
    ) == 0
    allowed_quality = json.loads((allowed_output / "final_quality.json").read_text(encoding="utf-8"))
    assert allowed_quality["valid"] is False
    assert "pelvis_height_too_low" in allowed_quality["quality_report"]["failures"]


def test_refine_cli_batch_physical_feasibility_records_invalid_items(tmp_path: Path):
    config_path = tmp_path / "physical_config.json"
    config_path.write_text(
        json.dumps(
            {
                "refiner": {"iterations": 0, "dtype": "float64", "lbfgs_enabled": False},
                "physical_feasibility": {"fail_on_pelvis_height": True, "min_pelvis_height_m": 10.0},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "physical_batch"

    assert refine.main(
        [
            "--inputs",
            "mock",
            "mock",
            "--output",
            str(output),
            "--mock-frames",
            "2",
            "--refinement-config",
            str(config_path),
        ],
        backend_factory=MockBackend,
        refinement_fk_factory=lambda spec: FakeTorchFK(spec),
    ) == 1

    manifest = json.loads((output / "batch_manifest.json").read_text(encoding="utf-8"))
    assert manifest["success_count"] == 0
    assert manifest["failure_count"] == 2
    assert [item["status"] for item in manifest["items"]] == ["invalid", "invalid"]
    assert all("pelvis_height_too_low" in item["error"] for item in manifest["items"])


def test_viewer_pipeline_loads_refine_directory_refinement_without_success_field(tmp_path: Path):
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    output = tmp_path / "refine"
    output.mkdir()
    refined = _make_refinement_motion(spec, frames=3)
    export_refined_motion(refined, output / "final_motion.npz")

    result = ViewerPipeline().replay(
        input_path=output,
        output_dir=tmp_path / "viewer",
        viewer="file",
        backend=MockBackend(spec),
        viewer_factory=fake_viewer_factory,
    )

    assert result.motion_path == output / "final_motion.npz"
    assert result.replay_result.frame_count == 3
    assert (tmp_path / "viewer" / "newton_replay.json").exists()


def test_viewer_cli_auto_loads_online_directory(tmp_path: Path):
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    output = tmp_path / "online"
    output.mkdir()
    retargeted = _make_retargeted_motion(spec, frames=2)
    export_retargeted_motion(retargeted, output / "online_motion.npz")

    assert viewer.main(
        ["--input", str(output), "--output", str(tmp_path / "viewer"), "--viewer", "file"],
        backend=MockBackend(spec),
        viewer_factory=fake_viewer_factory,
    ) == 0
    assert (tmp_path / "viewer" / "newton_replay.json").exists()


def _make_canonical_motion(frames: int):
    from retargeter.pipeline import make_mock_canonical_motion

    return make_mock_canonical_motion(num_frames=frames, fps=30.0)


def _make_retargeted_motion(spec: RobotSpec, frames: int) -> RetargetedMotion:
    root_pos = np.zeros((frames, 3), dtype=np.float64)
    root_quat = np.zeros((frames, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    joint_pos = np.zeros((frames, spec.num_dofs), dtype=np.float64)
    body_pos = np.zeros((frames, len(spec.body_names), 3), dtype=np.float64)
    body_quat = np.zeros((frames, len(spec.body_names), 4), dtype=np.float64)
    body_quat[..., 3] = 1.0
    return RetargetedMotion(
        fps=30.0,
        robot=spec.robot,
        joint_names=list(spec.actuated_joints),
        root_pos_w=root_pos,
        root_quat_xyzw=root_quat,
        joint_pos=joint_pos,
        joint_vel=np.zeros_like(joint_pos),
        body_names=list(spec.body_names),
        body_pos_w=body_pos,
        body_quat_xyzw=body_quat,
        success=np.ones((frames,), dtype=bool),
    )


def _make_refinement_motion(spec: RobotSpec, frames: int) -> RefinedMotion:
    retargeted = _make_retargeted_motion(spec, frames)
    return RefinedMotion(
        fps=retargeted.fps,
        robot=retargeted.robot,
        joint_names=list(retargeted.joint_names),
        root_pos_w=retargeted.root_pos_w.copy(),
        root_quat_xyzw=retargeted.root_quat_xyzw.copy(),
        joint_pos=retargeted.joint_pos.copy(),
        joint_vel=retargeted.joint_vel.copy(),
        body_names=list(retargeted.body_names),
        body_pos_w=retargeted.body_pos_w.copy(),
        body_quat_xyzw=retargeted.body_quat_xyzw.copy(),
        root_delta=np.zeros_like(retargeted.root_pos_w),
        joint_delta=np.zeros_like(retargeted.joint_pos),
    )
