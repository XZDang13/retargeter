from __future__ import annotations

from retargeter.preprocess import CanonicalHumanMotion, FootContactResult
from retargeter.scale import IKTargetSet

from .stage1_solver import Stage1FrameResult, Stage1NewtonSolver


class OnlineStage1Runner:
    def __init__(self, solver: Stage1NewtonSolver):
        self.solver = solver
        self.previous_result: Stage1FrameResult | None = None
        self.frame_count = 0

    def reset(self) -> None:
        self.previous_result = None
        self.frame_count = 0

    def step(
        self,
        motion: CanonicalHumanMotion,
        frame_idx: int,
        *,
        contact_result: FootContactResult | None = None,
    ) -> Stage1FrameResult:
        result = self.solver.solve_frame(
            motion,
            frame_idx,
            contact_result=contact_result,
            previous_result=self.previous_result,
        )
        self.previous_result = result
        self.frame_count += 1
        return result

    def step_targets(
        self,
        stage1a_targets: IKTargetSet,
        stage1b_targets: IKTargetSet,
        *,
        fps: float,
        frame_idx: int | None = None,
    ) -> Stage1FrameResult:
        result = self.solver.solve_target_sets(
            stage1a_targets,
            stage1b_targets,
            frame_idx=self.frame_count if frame_idx is None else int(frame_idx),
            fps=float(fps),
            previous_result=self.previous_result,
        )
        self.previous_result = result
        self.frame_count += 1
        return result
