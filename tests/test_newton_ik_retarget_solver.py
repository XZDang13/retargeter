from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import make_canonical_motion
from retargeter.newton import (
    BackendSolveResult,
    IKState,
    NewtonSolveSettings,
    OnlineIKRetargetRunner,
    RobotBodyState,
    RobotSpec,
    SequenceIKRetargetRunner,
    NewtonIKRetargetSolver,
)


G1_29_NEWTON = Path("retargeter/newton/configs/g1_29_newton_ik.yaml")
G1_29_ROBOT = Path("retargeter/newton/configs/g1_29_robot.yaml")


class MockBackend:
    def __init__(self, robot_spec: RobotSpec, *, fail_call_indices: set[int] | None = None, joint_delta: float = 0.01):
        self.robot_spec = robot_spec
        self.fail_call_indices = fail_call_indices or set()
        self.joint_delta = joint_delta
        self.calls = []

    def solve_ik(self, seed_state: IKState, objectives, settings: NewtonSolveSettings) -> BackendSolveResult:
        call_index = len(self.calls)
        self.calls.append(
            {
                "seed": seed_state.copy(),
                "objectives": list(objectives),
                "settings": settings,
            }
        )
        success = call_index not in self.fail_call_indices
        q = seed_state.joint_pos.copy() + self.joint_delta * (call_index + 1)
        return BackendSolveResult(
            state=IKState(
                root_pos_w=seed_state.root_pos_w.copy(),
                root_quat_xyzw=seed_state.root_quat_xyzw.copy(),
                joint_pos=q,
            ),
            success=success,
            cost=float(call_index),
            iterations=settings.iterations,
            diagnostics={"mock_call_index": call_index},
        )

    def forward_kinematics(self, state: IKState) -> RobotBodyState:
        body_pos = np.zeros((len(self.robot_spec.body_names), 3), dtype=np.float64)
        body_quat = np.zeros((len(self.robot_spec.body_names), 4), dtype=np.float64)
        body_quat[:, 3] = 1.0
        body_pos[:] = state.root_pos_w
        body_pos[:, 2] += np.linspace(0.0, 0.5, len(self.robot_spec.body_names))
        return RobotBodyState(list(self.robot_spec.body_names), body_pos, body_quat)


class MockReusableSolver:
    def __init__(self, backend: "MockReusableBackend"):
        self.backend = backend

    def compatible(self, objectives, settings: NewtonSolveSettings) -> bool:
        return True

    def solve(self, seed_state: IKState, objectives, settings: NewtonSolveSettings) -> BackendSolveResult:
        call_index = len(self.backend.reusable_calls)
        self.backend.reusable_calls.append(
            {
                "seed": seed_state.copy(),
                "objectives": list(objectives),
                "settings": settings,
            }
        )
        q = seed_state.joint_pos.copy() + self.backend.joint_delta * (call_index + 1)
        return BackendSolveResult(
            state=IKState(
                root_pos_w=seed_state.root_pos_w.copy(),
                root_quat_xyzw=seed_state.root_quat_xyzw.copy(),
                joint_pos=q,
            ),
            success=True,
            cost=float(call_index),
            iterations=settings.iterations,
            diagnostics={"mock_reusable_call_index": call_index, "reused_solver": True},
        )


class MockReusableBackend(MockBackend):
    def __init__(self, robot_spec: RobotSpec, *, joint_delta: float = 0.01):
        super().__init__(robot_spec, joint_delta=joint_delta)
        self.reusable_builds = 0
        self.reusable_calls = []

    def create_reusable_solver(self, objectives, settings: NewtonSolveSettings) -> MockReusableSolver:
        self.reusable_builds += 1
        return MockReusableSolver(self)


def test_ik_retarget_solver_single_frame_uses_full_body_tracking_only():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    backend = MockBackend(spec)
    solver = NewtonIKRetargetSolver(G1_29_NEWTON, backend=backend)
    motion = make_canonical_motion(num_frames=4)

    result = solver.solve_frame(motion, frame_idx=1)

    assert result.success
    assert result.joint_pos.shape == (29,)
    assert result.body_state.body_pos_w.shape == (len(spec.body_names), 3)
    assert len(backend.calls) == 1
    assert backend.calls[0]["settings"].iterations == 8
    assert result.diagnostics["full_body_tracking"]["success"]
    assert result.diagnostics["target_counts"] == {"full_body_tracking": 20}


def test_ik_retarget_solver_warm_starts_from_previous_result():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    backend = MockBackend(spec)
    solver = NewtonIKRetargetSolver(G1_29_NEWTON, backend=backend)
    motion = make_canonical_motion(num_frames=3)

    first = solver.solve_frame(motion, frame_idx=0)
    second = solver.solve_frame(motion, frame_idx=1, previous_result=first)

    assert second.success
    assert np.allclose(backend.calls[1]["seed"].joint_pos, first.joint_pos)
    assert np.any(second.joint_vel != 0.0)


def test_ik_retarget_solver_falls_back_without_dropping_frame():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    backend = MockBackend(spec, fail_call_indices={0})
    solver = NewtonIKRetargetSolver(G1_29_NEWTON, backend=backend)
    motion = make_canonical_motion(num_frames=2)

    result = solver.solve_frame(motion, frame_idx=0)

    assert not result.success
    assert result.diagnostics["fallback_used"]
    assert result.frame_idx == 0
    assert result.joint_pos.shape == (29,)


def test_sequence_runner_keeps_100_frames_and_calls_one_solve_per_frame():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    backend = MockBackend(spec, joint_delta=0.001)
    solver = NewtonIKRetargetSolver(G1_29_NEWTON, backend=backend)
    runner = SequenceIKRetargetRunner(solver)
    motion = make_canonical_motion(num_frames=100)

    retargeted_motion = runner.run(motion)

    assert retargeted_motion.num_frames() == 100
    assert retargeted_motion.joint_pos.shape == (100, 29)
    assert retargeted_motion.body_pos_w.shape == (100, len(spec.body_names), 3)
    assert np.all(retargeted_motion.success)
    assert len(backend.calls) == 100


def test_sequence_runner_reuses_backend_solver_when_available():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    backend = MockReusableBackend(spec, joint_delta=0.001)
    solver = NewtonIKRetargetSolver(G1_29_NEWTON, backend=backend)
    motion = make_canonical_motion(num_frames=5)

    retargeted_motion = SequenceIKRetargetRunner(solver).run(motion)

    assert retargeted_motion.num_frames() == 5
    assert backend.reusable_builds == 1
    assert len(backend.reusable_calls) == 5
    assert backend.calls == []
    assert retargeted_motion.diagnostics[0]["full_body_tracking"]["diagnostics"]["reused_solver"] is True


def test_online_runner_steps_and_resets_state():
    spec = RobotSpec.from_yaml(G1_29_ROBOT)
    backend = MockBackend(spec)
    solver = NewtonIKRetargetSolver(G1_29_NEWTON, backend=backend)
    runner = OnlineIKRetargetRunner(solver)
    motion = make_canonical_motion(num_frames=3)

    first = runner.step(motion, 0)
    second = runner.step(motion, 1)

    assert runner.frame_count == 2
    assert np.allclose(backend.calls[1]["seed"].joint_pos, first.joint_pos)
    assert second.frame_idx == 1

    runner.reset()
    assert runner.previous_result is None
    assert runner.frame_count == 0
