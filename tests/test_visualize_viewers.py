from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import make_canonical_motion
from retargeter.newton import IKState, RobotSpec, RetargetedMotion
from retargeter.visualize import (
    motion_frame_to_ik_state,
    replay_motion_with_newton,
    validate_replay_motion_for_robot,
)


G1_29_ROBOT = Path("retargeter/newton/configs/g1_29_robot.yaml")


class FakeReplayBackend:
    def __init__(self, robot_spec: RobotSpec):
        self.robot_spec = robot_spec
        self.model = "fake-model"
        self.states: list[IKState] = []

    def make_newton_state(self, state: IKState):
        self.states.append(state.copy())
        return {"root": state.root_pos_w.copy(), "q": state.joint_pos.copy()}


class FakeViewer:
    def __init__(self, output_path: Path | None = None):
        self.output_path = output_path
        self.model = None
        self.logged = []
        self.meshes = []
        self.times = []
        self.closed = False

    def set_model(self, model):
        self.model = model

    def is_running(self):
        return True

    def should_step(self):
        return True

    def begin_frame(self, time):
        self.times.append(time)

    def log_state(self, state):
        self.logged.append(state)

    def log_mesh(self, name, points, indices, **kwargs):
        self.meshes.append(
            {
                "name": name,
                "points": _as_numpy(points),
                "indices": _as_numpy(indices),
                "kwargs": dict(kwargs),
            }
        )

    def end_frame(self):
        pass

    def close(self):
        if self.output_path is not None:
            self.output_path.write_text("fake replay\n", encoding="utf-8")
        self.closed = True


def test_motion_frame_to_ik_state_validates_order_and_copies_arrays():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    motion = _make_retargeted_motion(spec, num_frames=3)

    state = motion_frame_to_ik_state(motion, spec, 1)

    assert np.allclose(state.root_pos_w, motion.root_pos_w[1])
    assert np.allclose(state.joint_pos, motion.joint_pos[1])
    state.joint_pos[:] = 10.0
    assert not np.allclose(motion.joint_pos[1], 10.0)


def test_motion_frame_to_ik_state_rejects_bad_frame():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    motion = _make_retargeted_motion(spec, num_frames=1)

    with pytest.raises(IndexError):
        motion_frame_to_ik_state(motion, spec, 2)


def test_validate_replay_motion_for_robot_rejects_joint_mismatch():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    motion = _make_retargeted_motion(spec, num_frames=1)
    motion.joint_names = list(reversed(motion.joint_names))

    with pytest.raises(ValueError, match="joint_names"):
        validate_replay_motion_for_robot(motion, spec)


def test_replay_motion_uses_newton_viewer_api(tmp_path: Path):
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    motion = _make_retargeted_motion(spec, num_frames=4)
    backend = FakeReplayBackend(spec)
    created = []

    def viewer_factory(viewer, options):
        fake = FakeViewer(options["output_path"])
        created.append((viewer, options, fake))
        return fake

    output = tmp_path / "replay.json"
    result = replay_motion_with_newton(
        motion,
        spec,
        viewer="file",
        output_path=output,
        fps=20,
        start_frame=1,
        end_frame=4,
        backend=backend,
        viewer_factory=viewer_factory,
    )

    assert result.viewer == "file"
    assert result.frame_count == 3
    assert result.output_path == output
    assert output.exists()
    assert len(backend.states) == 3
    assert created[0][2].model == "fake-model"
    assert len(created[0][2].logged) == 3
    assert created[0][2].closed is True


def test_replay_motion_logs_human_mesh_with_offset_and_time_sync():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    motion = _make_retargeted_motion(spec, num_frames=3)
    motion.fps = 10.0
    backend = FakeReplayBackend(spec)
    created = []

    human = make_canonical_motion(num_frames=6, fps=20.0)
    vertices = np.zeros((6, 4, 3), dtype=np.float64)
    for frame_idx in range(vertices.shape[0]):
        vertices[frame_idx, :, 0] = float(frame_idx)
    human.vertices_w = vertices
    human.mesh_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    def viewer_factory(viewer, options):
        fake = FakeViewer(options["output_path"])
        created.append(fake)
        return fake

    result = replay_motion_with_newton(
        motion,
        spec,
        viewer="null",
        backend=backend,
        viewer_factory=viewer_factory,
        human_motion=human,
        human_offset=(1.0, 2.0, 3.0),
    )

    viewer = created[0]
    assert result.frame_count == 3
    assert len(viewer.logged) == 3
    assert len(viewer.meshes) == 3
    assert [mesh["name"] for mesh in viewer.meshes] == ["human/smplx_mesh"] * 3
    expected_human_indices = [0, 2, 4]
    for mesh, expected_idx in zip(viewer.meshes, expected_human_indices, strict=True):
        assert np.allclose(mesh["points"][:, 0], expected_idx + 1.0)
        assert np.allclose(mesh["points"][:, 1], 2.0)
        assert np.allclose(mesh["points"][:, 2], 3.0)
        assert np.array_equal(mesh["indices"], np.asarray([0, 1, 2, 0, 2, 3], dtype=np.int32))
        assert mesh["kwargs"]["backface_culling"] is False
        assert len(mesh["kwargs"]["color"]) == 3


def test_replay_motion_rejects_human_without_mesh():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    motion = _make_retargeted_motion(spec, num_frames=1)
    human = make_canonical_motion(num_frames=1)

    with pytest.raises(ValueError, match="vertices_w and mesh_faces"):
        replay_motion_with_newton(
            motion,
            spec,
            viewer="null",
            backend=FakeReplayBackend(spec),
            viewer_factory=lambda viewer, options: FakeViewer(options["output_path"]),
            human_motion=human,
        )


def _make_retargeted_motion(spec: RobotSpec, num_frames: int = 5) -> RetargetedMotion:
    joint_pos = np.zeros((num_frames, spec.num_dofs), dtype=np.float64)
    for idx in range(spec.num_dofs):
        joint_pos[:, idx] = 0.01 * idx
    joint_vel = np.zeros_like(joint_pos)
    root_pos = np.zeros((num_frames, 3), dtype=np.float64)
    root_pos[:, 2] = 0.8
    root_quat = np.zeros((num_frames, 4), dtype=np.float64)
    root_quat[:, 3] = 1.0
    body_pos = np.zeros((num_frames, len(spec.body_names), 3), dtype=np.float64)
    body_quat = np.zeros((num_frames, len(spec.body_names), 4), dtype=np.float64)
    body_quat[..., 3] = 1.0
    return RetargetedMotion(
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
    )


def _as_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
