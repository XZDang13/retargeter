from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class BodyIKTarget:
    semantic_name: str
    human_body_name: str
    robot_body_name: str
    target_pos_w: np.ndarray | None
    target_quat_xyzw: np.ndarray | None
    pos_weight: float
    rot_weight: float
    robot_local_pos: np.ndarray | None = None
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.semantic_name:
            raise ValueError("semantic_name must be non-empty.")
        if not self.human_body_name:
            raise ValueError("human_body_name must be non-empty.")
        if not self.robot_body_name:
            raise ValueError("robot_body_name must be non-empty.")
        if self.pos_weight < 0.0:
            raise ValueError(f"pos_weight must be non-negative for {self.semantic_name!r}.")
        if self.rot_weight < 0.0:
            raise ValueError(f"rot_weight must be non-negative for {self.semantic_name!r}.")
        if not np.isfinite(self.confidence) or self.confidence < 0.0:
            raise ValueError(f"confidence must be finite and non-negative for {self.semantic_name!r}.")

        if self.target_pos_w is not None:
            pos = np.asarray(self.target_pos_w)
            if pos.shape != (3,):
                raise ValueError(f"target_pos_w for {self.semantic_name!r} must have shape [3], got {pos.shape}.")
            if not np.all(np.isfinite(pos)):
                raise ValueError(f"target_pos_w for {self.semantic_name!r} contains NaN or inf values.")
            if self.robot_local_pos is not None:
                local_pos = np.asarray(self.robot_local_pos)
                if local_pos.shape != (3,):
                    raise ValueError(
                        f"robot_local_pos for {self.semantic_name!r} must have shape [3], got {local_pos.shape}."
                    )
                if not np.all(np.isfinite(local_pos)):
                    raise ValueError(f"robot_local_pos for {self.semantic_name!r} contains NaN or inf values.")

        if self.target_quat_xyzw is not None:
            quat = np.asarray(self.target_quat_xyzw)
            if quat.shape != (4,):
                raise ValueError(f"target_quat_xyzw for {self.semantic_name!r} must have shape [4], got {quat.shape}.")
            if not np.all(np.isfinite(quat)):
                raise ValueError(f"target_quat_xyzw for {self.semantic_name!r} contains NaN or inf values.")
            norm = np.linalg.norm(quat)
            if norm < 1e-8:
                raise ValueError(f"target_quat_xyzw for {self.semantic_name!r} has near-zero norm.")


@dataclass
class IKTargetSet:
    pass_name: Literal["coarse_alignment", "full_body_tracking"]
    targets: list[BodyIKTarget]
    metadata: dict = field(default_factory=dict)

    def active_position_targets(self) -> list[BodyIKTarget]:
        return [target for target in self.targets if target.target_pos_w is not None and target.pos_weight > 0.0]

    def active_rotation_targets(self) -> list[BodyIKTarget]:
        return [target for target in self.targets if target.target_quat_xyzw is not None and target.rot_weight > 0.0]

    def get_target(self, semantic_name: str) -> BodyIKTarget:
        for target in self.targets:
            if target.semantic_name == semantic_name:
                return target
        raise KeyError(f"Target {semantic_name!r} is not present in {self.pass_name}.")

    def validate(self) -> None:
        if self.pass_name not in {"coarse_alignment", "full_body_tracking"}:
            raise ValueError(
                "pass_name must be 'coarse_alignment' or 'full_body_tracking' "
                f"got {self.pass_name!r}."
            )
        seen = set()
        for target in self.targets:
            target.validate()
            if target.semantic_name in seen:
                raise ValueError(f"Duplicate target semantic_name {target.semantic_name!r}.")
            seen.add(target.semantic_name)
