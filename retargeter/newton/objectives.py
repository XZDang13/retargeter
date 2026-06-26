from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from retargeter.preprocess.lowpass import normalize_quat_xyzw
from retargeter.scale import IKTargetSet

from .robot_spec import RobotSpec


ObjectiveKind = Literal["position", "rotation", "joint_limit", "posture", "smooth", "damping"]


@dataclass(frozen=True)
class IKObjectiveDescriptor:
    kind: ObjectiveKind
    weight: float
    body_name: str | None = None
    semantic_name: str | None = None
    target: np.ndarray | None = None
    confidence: float = 1.0

    def validate(self, robot_spec: RobotSpec) -> None:
        if self.weight < 0.0 or not np.isfinite(self.weight):
            raise ValueError(f"{self.kind} objective weight must be finite and non-negative.")
        if self.confidence < 0.0 or not np.isfinite(self.confidence):
            raise ValueError(f"{self.kind} objective confidence must be finite and non-negative.")

        if self.kind in {"position", "rotation"}:
            if not self.body_name:
                raise ValueError(f"{self.kind} objective must define body_name.")
            robot_spec.require_body_names([self.body_name])
            if self.target is None:
                raise ValueError(f"{self.kind} objective must define a target.")

        if self.kind == "position":
            target = np.asarray(self.target, dtype=np.float64)
            if target.shape != (3,):
                raise ValueError(f"position target for {self.body_name!r} must have shape [3], got {target.shape}.")
            if not np.all(np.isfinite(target)):
                raise ValueError(f"position target for {self.body_name!r} contains NaN or inf values.")

        if self.kind == "rotation":
            target = np.asarray(self.target, dtype=np.float64)
            if target.shape != (4,):
                raise ValueError(f"rotation target for {self.body_name!r} must have shape [4], got {target.shape}.")
            if not np.all(np.isfinite(target)):
                raise ValueError(f"rotation target for {self.body_name!r} contains NaN or inf values.")
            if np.linalg.norm(target) < 1e-8:
                raise ValueError(f"rotation target for {self.body_name!r} has near-zero norm.")

        if self.kind in {"posture", "smooth", "damping"}:
            if self.target is None:
                raise ValueError(f"{self.kind} objective must define a target joint vector.")
            target = np.asarray(self.target, dtype=np.float64)
            if target.shape != (robot_spec.num_dofs,):
                raise ValueError(f"{self.kind} target must have shape [{robot_spec.num_dofs}], got {target.shape}.")
            if not np.all(np.isfinite(target)):
                raise ValueError(f"{self.kind} target contains NaN or inf values.")


def build_target_objectives(target_set: IKTargetSet, robot_spec: RobotSpec) -> list[IKObjectiveDescriptor]:
    target_set.validate()
    descriptors: list[IKObjectiveDescriptor] = []

    for target in target_set.targets:
        robot_spec.require_body_names([target.robot_body_name])
        confidence = float(target.confidence)
        if target.target_pos_w is not None and target.pos_weight > 0.0:
            descriptors.append(
                IKObjectiveDescriptor(
                    kind="position",
                    weight=float(target.pos_weight),
                    body_name=target.robot_body_name,
                    semantic_name=target.semantic_name,
                    target=np.asarray(target.target_pos_w, dtype=np.float64).copy(),
                    confidence=confidence,
                )
            )
        if target.target_quat_xyzw is not None and target.rot_weight > 0.0:
            descriptors.append(
                IKObjectiveDescriptor(
                    kind="rotation",
                    weight=float(target.rot_weight),
                    body_name=target.robot_body_name,
                    semantic_name=target.semantic_name,
                    target=normalize_quat_xyzw(np.asarray(target.target_quat_xyzw, dtype=np.float64)).copy(),
                    confidence=confidence,
                )
            )

    for descriptor in descriptors:
        descriptor.validate(robot_spec)
    return descriptors


def build_regularization_objectives(
    robot_spec: RobotSpec,
    *,
    joint_limit_weight: float,
    posture_weight: float,
    smooth_weight: float,
    damping_weight: float,
    default_joint_pos: np.ndarray | None = None,
    previous_joint_pos: np.ndarray | None = None,
) -> list[IKObjectiveDescriptor]:
    descriptors: list[IKObjectiveDescriptor] = []

    if joint_limit_weight > 0.0:
        descriptors.append(IKObjectiveDescriptor(kind="joint_limit", weight=float(joint_limit_weight)))

    if posture_weight > 0.0:
        target = robot_spec.default_joint_pos if default_joint_pos is None else np.asarray(default_joint_pos, dtype=np.float64)
        descriptors.append(IKObjectiveDescriptor(kind="posture", weight=float(posture_weight), target=target.copy()))

    if smooth_weight > 0.0 and previous_joint_pos is not None:
        target = np.asarray(previous_joint_pos, dtype=np.float64)
        descriptors.append(IKObjectiveDescriptor(kind="smooth", weight=float(smooth_weight), target=target.copy()))

    if damping_weight > 0.0 and previous_joint_pos is not None:
        target = np.asarray(previous_joint_pos, dtype=np.float64)
        descriptors.append(IKObjectiveDescriptor(kind="damping", weight=float(damping_weight), target=target.copy()))

    for descriptor in descriptors:
        descriptor.validate(robot_spec)
    return descriptors
