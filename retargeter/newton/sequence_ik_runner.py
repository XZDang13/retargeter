from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from retargeter.preprocess import CanonicalHumanMotion, FootContactResult
from retargeter.progress import ProgressReporter, get_progress

from .ik_retarget_solver import IKRetargetFrameResult, NewtonIKRetargetSolver


@dataclass
class RetargetedMotion:
    fps: float
    robot: str
    joint_names: list[str]
    root_pos_w: np.ndarray
    root_quat_xyzw: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_names: list[str]
    body_pos_w: np.ndarray
    body_quat_xyzw: np.ndarray
    success: np.ndarray
    diagnostics: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    def validate(self) -> None:
        t = self.num_frames()
        d = len(self.joint_names)
        b = len(self.body_names)
        checks = [
            ("root_pos_w", self.root_pos_w, (t, 3)),
            ("root_quat_xyzw", self.root_quat_xyzw, (t, 4)),
            ("joint_pos", self.joint_pos, (t, d)),
            ("joint_vel", self.joint_vel, (t, d)),
            ("body_pos_w", self.body_pos_w, (t, b, 3)),
            ("body_quat_xyzw", self.body_quat_xyzw, (t, b, 4)),
            ("success", self.success, (t,)),
        ]
        for name, value, expected in checks:
            arr = np.asarray(value)
            if arr.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {arr.shape}.")
            if name != "success" and not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or inf values.")
        if self.fps <= 0.0 or not np.isfinite(self.fps):
            raise ValueError(f"fps must be positive and finite, got {self.fps!r}.")


class SequenceIKRetargetRunner:
    def __init__(self, solver: NewtonIKRetargetSolver):
        self.solver = solver

    def run(
        self,
        motion: CanonicalHumanMotion,
        *,
        contact_result: FootContactResult | None = None,
        progress: ProgressReporter | None = None,
    ) -> RetargetedMotion:
        motion.validate()
        reporter = get_progress(progress)
        frame_results: list[IKRetargetFrameResult] = []
        previous: IKRetargetFrameResult | None = None
        with reporter.bar(total=motion.num_frames(), desc="IK retarget", unit="frame") as bar:
            for frame_idx in range(motion.num_frames()):
                result = self.solver.solve_frame(
                    motion,
                    frame_idx,
                    contact_result=contact_result,
                    previous_result=previous,
                )
                frame_results.append(result)
                previous = result
                bar.update(1)
        return retargeted_motion_from_frames(frame_results, fps=float(motion.fps), metadata={"source_frames": motion.num_frames()})


def retargeted_motion_from_frames(
    frame_results: list[IKRetargetFrameResult],
    *,
    fps: float,
    metadata: dict | None = None,
) -> RetargetedMotion:
    if not frame_results:
        raise ValueError("frame_results must be non-empty.")

    joint_names = list(frame_results[0].joint_names)
    body_names = list(frame_results[0].body_state.body_names)
    robot = frame_results[0].robot
    for result in frame_results:
        if result.robot != robot:
            raise ValueError("All frame results must use the same robot.")
        if result.joint_names != joint_names:
            raise ValueError("All frame results must use the same joint order.")
        if result.body_state.body_names != body_names:
            raise ValueError("All frame results must use the same body order.")

    retargeted_motion = RetargetedMotion(
        fps=float(fps),
        robot=robot,
        joint_names=joint_names,
        root_pos_w=np.stack([result.root_pos_w for result in frame_results], axis=0),
        root_quat_xyzw=np.stack([result.root_quat_xyzw for result in frame_results], axis=0),
        joint_pos=np.stack([result.joint_pos for result in frame_results], axis=0),
        joint_vel=np.stack([result.joint_vel for result in frame_results], axis=0),
        body_names=body_names,
        body_pos_w=np.stack([result.body_state.body_pos_w for result in frame_results], axis=0),
        body_quat_xyzw=np.stack([result.body_state.body_quat_xyzw for result in frame_results], axis=0),
        success=np.asarray([result.success for result in frame_results], dtype=bool),
        diagnostics=[dict(result.diagnostics) for result in frame_results],
        metadata=dict(metadata or {}),
    )
    retargeted_motion.validate()
    return retargeted_motion
