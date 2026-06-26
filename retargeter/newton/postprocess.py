from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .robot_spec import RobotSpec


@dataclass
class PostprocessReport:
    joint_limit_clamped: bool = False
    velocity_clamped: bool = False
    joint_limit_violation_before: float = 0.0
    max_velocity_before: float = 0.0
    metadata: dict = field(default_factory=dict)


def clamp_joint_limits(joint_pos: np.ndarray, robot_spec: RobotSpec) -> tuple[np.ndarray, bool, float]:
    q = np.asarray(joint_pos, dtype=np.float64)
    if q.shape != (robot_spec.num_dofs,):
        raise ValueError(f"joint_pos must have shape [{robot_spec.num_dofs}], got {q.shape}.")

    below = np.maximum(robot_spec.joint_lower_rad - q, 0.0)
    above = np.maximum(q - robot_spec.joint_upper_rad, 0.0)
    violation = float(np.max(np.maximum(below, above))) if q.size else 0.0
    clamped = np.clip(q, robot_spec.joint_lower_rad, robot_spec.joint_upper_rad)
    changed = bool(np.any(np.abs(clamped - q) > 1e-12))
    return clamped, changed, violation


def clamp_joint_velocity(
    joint_pos: np.ndarray,
    previous_joint_pos: np.ndarray | None,
    dt: float,
    robot_spec: RobotSpec,
    *,
    velocity_scale: float = 1.0,
) -> tuple[np.ndarray, bool, float]:
    q = np.asarray(joint_pos, dtype=np.float64)
    if q.shape != (robot_spec.num_dofs,):
        raise ValueError(f"joint_pos must have shape [{robot_spec.num_dofs}], got {q.shape}.")
    if previous_joint_pos is None:
        return q.copy(), False, 0.0
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError(f"dt must be positive and finite, got {dt!r}.")

    prev = np.asarray(previous_joint_pos, dtype=np.float64)
    if prev.shape != (robot_spec.num_dofs,):
        raise ValueError(f"previous_joint_pos must have shape [{robot_spec.num_dofs}], got {prev.shape}.")

    velocity = (q - prev) / dt
    max_velocity_before = float(np.max(np.abs(velocity))) if q.size else 0.0
    limit = robot_spec.velocity_limits_rad_s * float(velocity_scale)
    delta_limit = limit * dt
    clamped = np.minimum(np.maximum(q, prev - delta_limit), prev + delta_limit)
    changed = bool(np.any(np.abs(clamped - q) > 1e-12))
    return clamped, changed, max_velocity_before


def apply_stage1_postprocess(
    joint_pos: np.ndarray,
    robot_spec: RobotSpec,
    *,
    previous_joint_pos: np.ndarray | None,
    dt: float,
    clamp_limits: bool = True,
    clamp_velocity: bool = True,
    velocity_scale: float = 1.0,
) -> tuple[np.ndarray, PostprocessReport]:
    q = np.asarray(joint_pos, dtype=np.float64).copy()
    report = PostprocessReport()

    if clamp_limits:
        q, changed, violation = clamp_joint_limits(q, robot_spec)
        report.joint_limit_clamped = changed
        report.joint_limit_violation_before = violation

    if clamp_velocity:
        q, changed, max_velocity = clamp_joint_velocity(
            q,
            previous_joint_pos,
            dt,
            robot_spec,
            velocity_scale=velocity_scale,
        )
        report.velocity_clamped = changed
        report.max_velocity_before = max_velocity
        if clamp_limits:
            q, changed_again, violation = clamp_joint_limits(q, robot_spec)
            report.joint_limit_clamped = report.joint_limit_clamped or changed_again
            report.joint_limit_violation_before = max(report.joint_limit_violation_before, violation)

    return q, report
