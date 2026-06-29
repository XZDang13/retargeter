from __future__ import annotations

from retargeter.preprocess import CanonicalHumanMotion, FootContactResult
from retargeter.scale import IKTargetSet

from .ik_retarget_solver import IKRetargetFrameResult, NewtonIKRetargetSolver


class OnlineIKRetargetRunner:
    def __init__(self, solver: NewtonIKRetargetSolver):
        self.solver = solver
        self.previous_result: IKRetargetFrameResult | None = None
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
    ) -> IKRetargetFrameResult:
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
        coarse_targets: IKTargetSet,
        tracking_targets: IKTargetSet,
        *,
        fps: float,
        frame_idx: int | None = None,
    ) -> IKRetargetFrameResult:
        result = self.solver.solve_target_sets(
            coarse_targets,
            tracking_targets,
            frame_idx=self.frame_count if frame_idx is None else int(frame_idx),
            fps=float(fps),
            previous_result=self.previous_result,
        )
        self.previous_result = result
        self.frame_count += 1
        return result
