from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from retargeter.newton import RobotSpec, RetargetedMotion

from .losses import DEFAULT_CONTACT_POINTS
from .refiner import RefinedMotion


DEFAULT_QUALITY_CONFIG: dict[str, float | bool] = {
    "max_body_pos_deviation_m": 0.20,
    "mean_body_pos_deviation_m": 0.08,
    "max_root_pos_deviation_m": 0.10,
    "max_joint_deviation_rad": 0.35,
    "joint_limit_tolerance_rad": 1.0e-6,
    "joint_velocity_tolerance_rad_s": 1.0e-6,
    "fail_on_joint_velocity_violation": False,
    "penetration_worsening_tolerance_m": 0.005,
    "skating_worsening_tolerance_m_s": 0.005,
    "skating_min_improvement_m_s": 0.0,
}
DEFAULT_PHYSICAL_FEASIBILITY_CONFIG: dict[str, float | bool] = {
    "enabled": True,
    "max_joint_acceleration_rad_s2": 800.0,
    "max_joint_jerk_rad_s3": 40000.0,
    "max_foot_penetration_m": 0.03,
    "max_weighted_foot_height_m": 0.18,
    "fail_on_pelvis_height": False,
    "min_pelvis_height_m": 0.30,
    "fail_on_support_unavailable": True,
    "support_contact_threshold": 0.5,
    "support_max_foot_height_m": 0.08,
    "max_unsupported_duration_s": 1.0,
    "max_unsupported_fraction": 0.35,
    "bos_margin_m": 0.35,
    "max_bos_violation_fraction": 0.50,
}
EPS = 1.0e-12


@dataclass
class RefinementQualityReport:
    valid: bool
    metrics: dict[str, Any]
    failures: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "metrics": _jsonable(self.metrics),
            "failures": list(self.failures),
            "thresholds": _jsonable(self.thresholds),
            "metadata": _jsonable(self.metadata),
        }


def evaluate_refinement_quality(
    retargeted: RetargetedMotion,
    refined: RefinedMotion,
    robot_spec: RobotSpec,
    config: Mapping[str, Any] | None = None,
    contact_score=None,
    ground_height: float | None = None,
) -> RefinementQualityReport:
    """Evaluate refined motion against the fixed retargeted reference."""

    _validate_structure(retargeted, refined, robot_spec)
    thresholds = _quality_thresholds(config)
    physical_thresholds = _physical_feasibility_thresholds(config)
    report_thresholds = {**thresholds, **physical_thresholds}
    body_names = _body_names(retargeted, refined, config)
    contact_specs = _contact_point_specs(config)
    scores = _contact_score_mapping(contact_score)
    contact_available = scores is not None and bool(scores)
    resolved_ground = _ground_height(refined, contact_score, ground_height)

    retargeted_body_idx = _indices_for_names(retargeted.body_names, body_names, "retargeted.body_names")
    refined_body_idx = _indices_for_names(refined.body_names, body_names, "refined.body_names")

    metrics: dict[str, Any] = {
        "num_frames": int(refined.num_frames()),
        "fps": float(refined.fps),
        "compared_body_count": int(len(body_names)),
        "contact_available": bool(contact_available),
        "ground_height": float(resolved_ground),
    }
    failures: list[str] = []

    metrics.update(_nonfinite_metrics(retargeted, refined))
    if int(metrics["nonfinite_count"]) > 0:
        failures.append("nonfinite_values")

    metrics.update(_motion_deviation_metrics(retargeted, refined, retargeted_body_idx, refined_body_idx))
    metrics.update(_joint_feasibility_metrics(refined, robot_spec, thresholds))

    if metrics["joint_limit_violation_count"] > 0:
        failures.append("joint_limit_violation")
    if thresholds["fail_on_joint_velocity_violation"] and metrics["joint_velocity_violation_count"] > 0:
        failures.append("joint_velocity_violation")

    if metrics["body_pos_deviation_max_m"] > thresholds["max_body_pos_deviation_m"]:
        failures.append("body_position_deviation")
    if metrics["body_pos_deviation_mean_m"] > thresholds["mean_body_pos_deviation_m"]:
        failures.append("body_position_deviation_mean")
    if metrics["root_pos_deviation_max_m"] > thresholds["max_root_pos_deviation_m"]:
        failures.append("root_position_deviation")
    if metrics["joint_pos_deviation_max_rad"] > thresholds["max_joint_deviation_rad"]:
        failures.append("joint_position_deviation")

    contact_points = _contact_points(retargeted, refined, robot_spec, contact_specs, scores)
    metrics.update(_contact_metrics(contact_points, float(resolved_ground), float(refined.fps), contact_available))
    if metrics["penetration_worsening_m"] > thresholds["penetration_worsening_tolerance_m"]:
        failures.append("penetration_worsened")
    if contact_available:
        min_improvement = thresholds["skating_min_improvement_m_s"]
        tolerance = thresholds["skating_worsening_tolerance_m_s"]
        if metrics["skating_improvement_m_s"] < min_improvement - tolerance:
            failures.append("skating_not_improved")

    metrics.update(_dynamics_metrics(refined, float(refined.fps)))
    metrics.update(_support_metrics(refined, contact_points, float(resolved_ground), physical_thresholds, contact_available))
    metrics["physical_feasibility_enabled"] = bool(physical_thresholds["physical_feasibility_enabled"])
    failures.extend(_physical_feasibility_failures(metrics, physical_thresholds))
    failures = _unique_failures(failures)

    return RefinementQualityReport(
        valid=not failures,
        metrics=metrics,
        failures=failures,
        thresholds=report_thresholds,
        metadata={
            "source": "evaluate_refinement_quality",
            "robot": refined.robot,
            "body_names": list(body_names),
            "contact_regions": [point.region for point in contact_points],
            "contact_point_count": int(len(contact_points)),
        },
    )

@dataclass(frozen=True)
class _ContactPoint:
    region: str
    retargeted_pos_w: np.ndarray
    refined_pos_w: np.ndarray
    score: np.ndarray


def _validate_structure(retargeted: RetargetedMotion, refined: RefinedMotion, robot_spec: RobotSpec) -> None:
    t = int(retargeted.num_frames())
    if t != int(refined.num_frames()):
        raise ValueError(f"RetargetedMotion has {t} frames but RefinedMotion has {refined.num_frames()}.")
    if retargeted.robot != refined.robot:
        raise ValueError(f"RetargetedMotion robot {retargeted.robot!r} does not match RefinedMotion robot {refined.robot!r}.")
    if retargeted.robot != robot_spec.robot:
        raise ValueError(f"RetargetedMotion robot {retargeted.robot!r} does not match RobotSpec {robot_spec.robot!r}.")
    if retargeted.joint_names != refined.joint_names:
        raise ValueError("RetargetedMotion and RefinedMotion joint_names must match exactly.")
    if refined.joint_names != robot_spec.actuated_joints:
        raise ValueError("RefinedMotion joint_names must exactly match RobotSpec actuated_joints.")
    if not np.isfinite(float(refined.fps)) or float(refined.fps) <= 0.0:
        raise ValueError(f"RefinedMotion fps must be positive and finite, got {refined.fps!r}.")

    d = len(retargeted.joint_names)
    _require_shape(retargeted.root_pos_w, "retargeted.root_pos_w", (t, 3))
    _require_shape(retargeted.root_quat_xyzw, "retargeted.root_quat_xyzw", (t, 4))
    _require_shape(retargeted.joint_pos, "retargeted.joint_pos", (t, d))
    _require_shape(retargeted.joint_vel, "retargeted.joint_vel", (t, d))
    _require_shape(refined.root_pos_w, "refined.root_pos_w", (t, 3))
    _require_shape(refined.root_quat_xyzw, "refined.root_quat_xyzw", (t, 4))
    _require_shape(refined.joint_pos, "refined.joint_pos", (t, d))
    _require_shape(refined.joint_vel, "refined.joint_vel", (t, d))
    _require_shape(retargeted.body_pos_w, "retargeted.body_pos_w", (t, len(retargeted.body_names), 3))
    _require_shape(retargeted.body_quat_xyzw, "retargeted.body_quat_xyzw", (t, len(retargeted.body_names), 4))
    _require_shape(refined.body_pos_w, "refined.body_pos_w", (t, len(refined.body_names), 3))
    _require_shape(refined.body_quat_xyzw, "refined.body_quat_xyzw", (t, len(refined.body_names), 4))


def _motion_deviation_metrics(
    retargeted: RetargetedMotion,
    refined: RefinedMotion,
    retargeted_body_idx: list[int],
    refined_body_idx: list[int],
) -> dict[str, float]:
    retargeted_body = np.asarray(retargeted.body_pos_w[:, retargeted_body_idx, :], dtype=np.float64)
    refined_body = np.asarray(refined.body_pos_w[:, refined_body_idx, :], dtype=np.float64)
    body_l2 = _safe_norm(refined_body - retargeted_body, axis=-1)
    local_l2 = _safe_norm(
        (refined_body - np.asarray(refined.root_pos_w, dtype=np.float64)[:, None, :])
        - (retargeted_body - np.asarray(retargeted.root_pos_w, dtype=np.float64)[:, None, :]),
        axis=-1,
    )
    root_l2 = _safe_norm(np.asarray(refined.root_pos_w, dtype=np.float64) - np.asarray(retargeted.root_pos_w, dtype=np.float64), axis=-1)
    joint_abs = np.abs(np.asarray(refined.joint_pos, dtype=np.float64) - np.asarray(retargeted.joint_pos, dtype=np.float64))
    quat_error = _quat_error(
        np.asarray(refined.body_quat_xyzw[:, refined_body_idx, :], dtype=np.float64),
        np.asarray(retargeted.body_quat_xyzw[:, retargeted_body_idx, :], dtype=np.float64),
    )
    return {
        "body_pos_deviation_mean_m": _finite_mean(body_l2),
        "body_pos_deviation_max_m": _finite_max(body_l2),
        "local_body_pos_deviation_mean_m": _finite_mean(local_l2),
        "local_body_pos_deviation_max_m": _finite_max(local_l2),
        "root_pos_deviation_mean_m": _finite_mean(root_l2),
        "root_pos_deviation_max_m": _finite_max(root_l2),
        "joint_pos_deviation_mean_rad": _finite_mean(joint_abs),
        "joint_pos_deviation_max_rad": _finite_max(joint_abs),
        "body_quat_deviation_mean": _finite_mean(quat_error),
        "body_quat_deviation_max": _finite_max(quat_error),
    }


def _joint_feasibility_metrics(
    refined: RefinedMotion,
    robot_spec: RobotSpec,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    q = np.asarray(refined.joint_pos, dtype=np.float64)
    dq = np.asarray(refined.joint_vel, dtype=np.float64)
    lower = np.asarray(robot_spec.joint_lower_rad, dtype=np.float64).reshape(1, -1)
    upper = np.asarray(robot_spec.joint_upper_rad, dtype=np.float64).reshape(1, -1)
    velocity_limit = np.asarray(robot_spec.velocity_limits_rad_s, dtype=np.float64).reshape(1, -1)
    joint_tol = float(thresholds["joint_limit_tolerance_rad"])
    velocity_tol = float(thresholds["joint_velocity_tolerance_rad_s"])

    lower_violation = np.maximum(lower - q, 0.0)
    upper_violation = np.maximum(q - upper, 0.0)
    joint_violation = np.maximum(lower_violation, upper_violation)
    joint_mask = np.isfinite(q) & ((q < lower - joint_tol) | (q > upper + joint_tol))

    velocity_violation = np.maximum(np.abs(dq) - velocity_limit, 0.0)
    velocity_mask = np.isfinite(dq) & (np.abs(dq) > velocity_limit + velocity_tol)

    return {
        "joint_limit_violation_count": int(np.count_nonzero(joint_mask)),
        "joint_limit_violation_max_rad": _finite_max(joint_violation),
        "joint_limit_worst_joint": _worst_joint_name(joint_violation, refined.joint_names),
        "joint_velocity_violation_count": int(np.count_nonzero(velocity_mask)),
        "joint_velocity_violation_max_rad_s": _finite_max(velocity_violation),
        "joint_velocity_worst_joint": _worst_joint_name(velocity_violation, refined.joint_names),
    }


def _contact_metrics(
    contact_points: list[_ContactPoint],
    ground_height: float,
    fps: float,
    contact_available: bool,
) -> dict[str, Any]:
    if not contact_points:
        return {
            "contact_point_count": 0,
            "contact_weight_mean": 0.0,
            "retargeted_weighted_foot_height_mean_m": 0.0,
            "refined_weighted_foot_height_mean_m": 0.0,
            "refined_weighted_foot_height_max_m": 0.0,
            "retargeted_penetration_max_m": 0.0,
            "refined_penetration_max_m": 0.0,
            "retargeted_penetration_mean_m": 0.0,
            "refined_penetration_mean_m": 0.0,
            "penetration_worsening_m": 0.0,
            "retargeted_weighted_skating_m_s": 0.0,
            "refined_weighted_skating_m_s": 0.0,
            "skating_improvement_m_s": 0.0,
            "skating_gate_evaluated": bool(contact_available),
        }

    retargeted_height = []
    refined_height = []
    retargeted_penetration = []
    refined_penetration = []
    retargeted_skating = []
    refined_skating = []
    height_weights = []
    skating_weights = []
    for point in contact_points:
        score = point.score
        s1_height = point.retargeted_pos_w[:, 2] - ground_height
        s2_height = point.refined_pos_w[:, 2] - ground_height
        retargeted_height.append(np.abs(s1_height) * score)
        refined_height.append(np.abs(s2_height) * score)
        retargeted_penetration.append(np.maximum(ground_height - point.retargeted_pos_w[:, 2], 0.0))
        refined_penetration.append(np.maximum(ground_height - point.refined_pos_w[:, 2], 0.0))
        height_weights.append(score)
        if point.refined_pos_w.shape[0] >= 2:
            velocity_weight = score[:-1]
            retargeted_skating.append(_horizontal_speed(point.retargeted_pos_w, fps) * velocity_weight)
            refined_skating.append(_horizontal_speed(point.refined_pos_w, fps) * velocity_weight)
            skating_weights.append(velocity_weight)

    retargeted_pen = np.stack(retargeted_penetration, axis=1)
    refined_pen = np.stack(refined_penetration, axis=1)
    retargeted_skate = np.stack(retargeted_skating, axis=1) if retargeted_skating else np.zeros((0, 0), dtype=np.float64)
    refined_skate = np.stack(refined_skating, axis=1) if refined_skating else np.zeros((0, 0), dtype=np.float64)
    return {
        "contact_point_count": int(len(contact_points)),
        "contact_weight_mean": _finite_mean(np.stack(height_weights, axis=1)),
        "retargeted_weighted_foot_height_mean_m": _finite_mean(np.stack(retargeted_height, axis=1)),
        "refined_weighted_foot_height_mean_m": _finite_mean(np.stack(refined_height, axis=1)),
        "refined_weighted_foot_height_max_m": _finite_max(np.stack(refined_height, axis=1)),
        "retargeted_penetration_max_m": _finite_max(retargeted_pen),
        "refined_penetration_max_m": _finite_max(refined_pen),
        "retargeted_penetration_mean_m": _finite_mean(retargeted_pen),
        "refined_penetration_mean_m": _finite_mean(refined_pen),
        "penetration_worsening_m": _finite_max(refined_pen) - _finite_max(retargeted_pen),
        "retargeted_weighted_skating_m_s": _finite_mean(retargeted_skate),
        "refined_weighted_skating_m_s": _finite_mean(refined_skate),
        "skating_improvement_m_s": _finite_mean(retargeted_skate) - _finite_mean(refined_skate),
        "skating_gate_evaluated": bool(contact_available),
        "skating_contact_weight_mean": _finite_mean(np.stack(skating_weights, axis=1)) if skating_weights else 0.0,
    }


def _dynamics_metrics(refined: RefinedMotion, fps: float) -> dict[str, float]:
    root_acc = _finite_difference_stats(np.asarray(refined.root_pos_w, dtype=np.float64), fps, order=2, vector_norm=True)
    root_jerk = _finite_difference_stats(np.asarray(refined.root_pos_w, dtype=np.float64), fps, order=3, vector_norm=True)
    joint_acc = _finite_difference_stats(np.asarray(refined.joint_pos, dtype=np.float64), fps, order=2, vector_norm=False)
    joint_jerk = _finite_difference_stats(np.asarray(refined.joint_pos, dtype=np.float64), fps, order=3, vector_norm=False)
    body_acc = _finite_difference_stats(np.asarray(refined.body_pos_w, dtype=np.float64), fps, order=2, vector_norm=True)
    body_jerk = _finite_difference_stats(np.asarray(refined.body_pos_w, dtype=np.float64), fps, order=3, vector_norm=True)
    return {
        "root_acceleration_mean_m_s2": root_acc[0],
        "root_acceleration_max_m_s2": root_acc[1],
        "root_jerk_mean_m_s3": root_jerk[0],
        "root_jerk_max_m_s3": root_jerk[1],
        "joint_acceleration_mean_rad_s2": joint_acc[0],
        "joint_acceleration_max_rad_s2": joint_acc[1],
        "joint_jerk_mean_rad_s3": joint_jerk[0],
        "joint_jerk_max_rad_s3": joint_jerk[1],
        "body_acceleration_mean_m_s2": body_acc[0],
        "body_acceleration_max_m_s2": body_acc[1],
        "body_jerk_mean_m_s3": body_jerk[0],
        "body_jerk_max_m_s3": body_jerk[1],
    }


def _support_metrics(
    refined: RefinedMotion,
    contact_points: list[_ContactPoint],
    ground_height: float,
    thresholds: Mapping[str, Any],
    contact_available: bool,
) -> dict[str, Any]:
    pelvis = _pelvis_or_root_pos(refined)
    pelvis_height = np.asarray(pelvis[:, 2], dtype=np.float64)
    frames = int(refined.num_frames())
    if not contact_available:
        return {
            "support_evaluated": False,
            "pelvis_height_min_m": _finite_min(pelvis_height),
            "support_frame_count": 0,
            "unsupported_frame_count": 0,
            "unsupported_fraction": 0.0,
            "unsupported_max_duration_s": 0.0,
            "bos_violation_frame_count": 0,
            "bos_violation_fraction": 0.0,
            "bos_violation_max_distance_m": 0.0,
        }

    contact_threshold = float(thresholds["support_contact_threshold"])
    max_foot_height = float(thresholds["support_max_foot_height_m"])
    support_mask = np.zeros((frames,), dtype=bool)
    bos_violation = np.zeros((frames,), dtype=bool)
    bos_distance = np.zeros((frames,), dtype=np.float64)

    for frame_idx in range(frames):
        support_xy = []
        for point in contact_points:
            height = float(point.refined_pos_w[frame_idx, 2] - ground_height)
            if float(point.score[frame_idx]) >= contact_threshold and abs(height) <= max_foot_height:
                support_xy.append(point.refined_pos_w[frame_idx, :2])
        if not support_xy:
            continue
        support_mask[frame_idx] = True
        distance = _support_distance_xy(pelvis[frame_idx, :2], np.asarray(support_xy, dtype=np.float64))
        bos_distance[frame_idx] = distance
        bos_violation[frame_idx] = distance > float(thresholds["bos_margin_m"])

    unsupported = ~support_mask
    supported_count = int(np.count_nonzero(support_mask))
    return {
        "support_evaluated": True,
        "pelvis_height_min_m": _finite_min(pelvis_height),
        "support_frame_count": supported_count,
        "unsupported_frame_count": int(np.count_nonzero(unsupported)),
        "unsupported_fraction": float(np.count_nonzero(unsupported) / max(frames, 1)),
        "unsupported_max_duration_s": _max_true_run_duration(unsupported, float(refined.fps)),
        "bos_violation_frame_count": int(np.count_nonzero(bos_violation)),
        "bos_violation_fraction": float(np.count_nonzero(bos_violation) / max(supported_count, 1)),
        "bos_violation_max_distance_m": _finite_max(bos_distance[support_mask]) if supported_count else 0.0,
    }


def _physical_feasibility_failures(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[str]:
    if not bool(thresholds["physical_feasibility_enabled"]):
        return []

    failures: list[str] = []
    if int(metrics["joint_velocity_violation_count"]) > 0:
        failures.append("joint_velocity_violation")
    if float(metrics["joint_acceleration_max_rad_s2"]) > float(thresholds["max_joint_acceleration_rad_s2"]):
        failures.append("joint_acceleration_violation")
    if float(metrics["joint_jerk_max_rad_s3"]) > float(thresholds["max_joint_jerk_rad_s3"]):
        failures.append("joint_jerk_violation")
    if float(metrics["refined_penetration_max_m"]) > float(thresholds["max_foot_penetration_m"]):
        failures.append("foot_penetration")
    if (
        bool(metrics["contact_available"])
        and float(metrics["refined_weighted_foot_height_max_m"]) > float(thresholds["max_weighted_foot_height_m"])
    ):
        failures.append("foot_floating")
    if bool(thresholds["fail_on_pelvis_height"]) and float(metrics["pelvis_height_min_m"]) < float(thresholds["min_pelvis_height_m"]):
        failures.append("pelvis_height_too_low")
    if bool(metrics["support_evaluated"]):
        if (
            bool(thresholds["fail_on_support_unavailable"])
            and float(metrics["unsupported_max_duration_s"]) > float(thresholds["max_unsupported_duration_s"])
        ):
            failures.append("support_unavailable")
        if float(metrics["bos_violation_fraction"]) > float(thresholds["max_bos_violation_fraction"]):
            failures.append("base_of_support_violation")
    return failures


def _unique_failures(failures: list[str]) -> list[str]:
    seen = set()
    unique = []
    for failure in failures:
        if failure in seen:
            continue
        seen.add(failure)
        unique.append(failure)
    return unique


def _contact_points(
    retargeted: RetargetedMotion,
    refined: RefinedMotion,
    robot_spec: RobotSpec,
    specs: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Any] | None,
) -> list[_ContactPoint]:
    regions = list(scores.keys()) if scores is not None else list(specs.keys())
    points: list[_ContactPoint] = []
    for region in regions:
        if region not in specs:
            continue
        spec = specs[region]
        body = str(spec["body"])
        if not robot_spec.has_body(body):
            raise ValueError(f"contact point region {region!r} references body {body!r} missing from RobotSpec.")
        if body not in retargeted.body_names:
            raise ValueError(f"contact point region {region!r} references body {body!r} missing from retargeted.")
        if body not in refined.body_names:
            raise ValueError(f"contact point region {region!r} references body {body!r} missing from refined.")
        local_pos = np.asarray(spec.get("local_pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        if local_pos.shape != (3,):
            raise ValueError(f"contact point local_pos for region {region!r} must have shape [3], got {local_pos.shape}.")
        score = np.ones((refined.num_frames(),), dtype=np.float64)
        if scores is not None:
            score = _contact_score_array(scores[region], refined.num_frames(), region)
        retargeted_idx = retargeted.body_names.index(body)
        refined_idx = refined.body_names.index(body)
        points.append(
            _ContactPoint(
                region=str(region),
                retargeted_pos_w=_body_point_w(retargeted.body_pos_w[:, retargeted_idx, :], retargeted.body_quat_xyzw[:, retargeted_idx, :], local_pos),
                refined_pos_w=_body_point_w(refined.body_pos_w[:, refined_idx, :], refined.body_quat_xyzw[:, refined_idx, :], local_pos),
                score=score,
            )
        )
    return points


def _quality_thresholds(config: Mapping[str, Any] | None) -> dict[str, Any]:
    thresholds = copy.deepcopy(DEFAULT_QUALITY_CONFIG)
    for key in list(thresholds.keys()):
        value = _cfg(config, "quality", key, thresholds[key])
        thresholds[key] = bool(value) if isinstance(thresholds[key], bool) else _finite_float(value, f"quality.{key}")
    for key, value in thresholds.items():
        if isinstance(value, float) and value < 0.0:
            raise ValueError(f"quality.{key} must be non-negative, got {value!r}.")
    return thresholds


def _physical_feasibility_thresholds(config: Mapping[str, Any] | None) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    for key, default in DEFAULT_PHYSICAL_FEASIBILITY_CONFIG.items():
        value = _cfg(config, "physical_feasibility", key, default)
        out_key = "physical_feasibility_enabled" if key == "enabled" else key
        thresholds[out_key] = bool(value) if isinstance(default, bool) else _finite_float(value, f"physical_feasibility.{key}")
    for key, value in thresholds.items():
        if key != "physical_feasibility_enabled" and isinstance(value, float) and value < 0.0:
            raise ValueError(f"physical_feasibility.{key} must be non-negative, got {value!r}.")
    return thresholds


def _body_names(retargeted: RetargetedMotion, refined: RefinedMotion, config: Mapping[str, Any] | None) -> list[str]:
    configured = _cfg(config, "quality", "body_names", None)
    if configured is not None:
        names = [str(name) for name in configured]
    else:
        refined_names = set(refined.body_names)
        names = [name for name in retargeted.body_names if name in refined_names]
    if not names:
        raise ValueError("quality body_names resolved to an empty list.")
    _indices_for_names(retargeted.body_names, names, "retargeted.body_names")
    _indices_for_names(refined.body_names, names, "refined.body_names")
    return names


def _contact_point_specs(config: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    specs = {region: dict(value) for region, value in DEFAULT_CONTACT_POINTS.items()}
    for overrides in (_lookup(config, "contact_points", None), _cfg(config, "quality", "contact_points", None)):
        if overrides is None:
            continue
        if not isinstance(overrides, Mapping):
            raise TypeError("quality.contact_points must be a mapping.")
        for region, raw in overrides.items():
            if isinstance(raw, str):
                specs[str(region)] = {"body": raw}
            elif isinstance(raw, Mapping):
                if "body" not in raw:
                    raise ValueError(f"quality.contact_points.{region} must define body.")
                entry = {"body": str(raw["body"])}
                if "local_pos" in raw:
                    entry["local_pos"] = raw["local_pos"]
                specs[str(region)] = entry
            else:
                raise TypeError(f"quality.contact_points.{region} must be a body string or mapping.")
    return specs


def _contact_score_mapping(contact_score) -> Mapping[str, Any] | None:
    if contact_score is None:
        return None
    if hasattr(contact_score, "contact_score"):
        contact_score = contact_score.contact_score
    if not isinstance(contact_score, Mapping):
        raise TypeError("contact_score must be a mapping or object with a contact_score mapping.")
    return contact_score


def _ground_height(refined: RefinedMotion, contact_score, ground_height: float | None) -> float:
    if ground_height is not None:
        return _finite_float(ground_height, "ground_height")
    if contact_score is not None and hasattr(contact_score, "ground_height"):
        return _finite_float(contact_score.ground_height, "contact_score.ground_height")
    if "ground_height" in refined.metadata:
        return _finite_float(refined.metadata["ground_height"], "refined.metadata['ground_height']")
    return 0.0


def _nonfinite_metrics(retargeted: RetargetedMotion, refined: RefinedMotion) -> dict[str, int]:
    retargeted_count = sum(_nonfinite_count(value) for value in _numeric_arrays(retargeted))
    refined_count = sum(_nonfinite_count(value) for value in _numeric_arrays(refined))
    return {
        "retargeted_nonfinite_count": int(retargeted_count),
        "refined_nonfinite_count": int(refined_count),
        "nonfinite_count": int(retargeted_count + refined_count),
    }


def _numeric_arrays(motion: RetargetedMotion | RefinedMotion) -> list[np.ndarray]:
    return [
        np.asarray(motion.root_pos_w),
        np.asarray(motion.root_quat_xyzw),
        np.asarray(motion.joint_pos),
        np.asarray(motion.joint_vel),
        np.asarray(motion.body_pos_w),
        np.asarray(motion.body_quat_xyzw),
    ]


def _body_point_w(body_pos: np.ndarray, body_quat: np.ndarray, local_pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(body_pos, dtype=np.float64)
    if np.all(local_pos == 0.0):
        return pos.copy()
    quat = _quat_normalize(np.asarray(body_quat, dtype=np.float64))
    return pos + _quat_rotate_xyzw(quat, np.broadcast_to(local_pos.reshape(1, 3), pos.shape))


def _pelvis_or_root_pos(refined: RefinedMotion) -> np.ndarray:
    if "pelvis" in refined.body_names:
        return np.asarray(refined.body_pos_w[:, refined.body_names.index("pelvis"), :], dtype=np.float64)
    return np.asarray(refined.root_pos_w, dtype=np.float64)


def _support_distance_xy(point_xy: np.ndarray, support_xy: np.ndarray) -> float:
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    support = _unique_points_xy(np.asarray(support_xy, dtype=np.float64))
    if support.shape[0] == 0 or not np.all(np.isfinite(point)):
        return 0.0
    if support.shape[0] == 1:
        return float(np.linalg.norm(point - support[0]))
    hull = _convex_hull_xy(support)
    if hull.shape[0] == 1:
        return float(np.linalg.norm(point - hull[0]))
    if hull.shape[0] == 2:
        return _point_segment_distance_xy(point, hull[0], hull[1])
    if _point_in_convex_polygon_xy(point, hull):
        return 0.0
    distances = [
        _point_segment_distance_xy(point, hull[idx], hull[(idx + 1) % hull.shape[0]])
        for idx in range(hull.shape[0])
    ]
    return float(min(distances)) if distances else 0.0


def _unique_points_xy(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    finite = points[np.all(np.isfinite(points), axis=1)]
    if finite.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    rounded = np.round(finite, decimals=9)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return finite[np.sort(indices)]


def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    ordered = sorted((float(x), float(y)) for x, y in points)
    if len(ordered) <= 1:
        return np.asarray(ordered, dtype=np.float64).reshape(-1, 2)

    def cross(origin, a, b) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= EPS:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= EPS:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _point_in_convex_polygon_xy(point: np.ndarray, polygon: np.ndarray) -> bool:
    for idx in range(polygon.shape[0]):
        a = polygon[idx]
        b = polygon[(idx + 1) % polygon.shape[0]]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if cross < -EPS:
            return False
    return True


def _point_segment_distance_xy(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    segment = b - a
    denom = float(np.dot(segment, segment))
    if denom <= EPS:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, segment) / denom, 0.0, 1.0))
    projection = a + t * segment
    return float(np.linalg.norm(point - projection))


def _max_true_run_duration(mask: np.ndarray, fps: float) -> float:
    values = np.asarray(mask, dtype=bool)
    max_run = 0
    current = 0
    for value in values:
        if value:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return float(max_run / max(float(fps), EPS))


def _finite_difference_stats(values: np.ndarray, fps: float, *, order: int, vector_norm: bool) -> tuple[float, float]:
    if values.shape[0] <= order:
        return 0.0, 0.0
    diff = np.diff(values, n=order, axis=0) * (float(fps) ** order)
    if vector_norm:
        series = _safe_norm(diff, axis=-1)
    else:
        series = np.abs(diff)
    return _finite_mean(series), _finite_max(series)


def _quat_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_n = _quat_normalize(pred)
    target_n = _quat_normalize(target)
    dot = np.abs(np.sum(pred_n * target_n, axis=-1))
    return 1.0 - np.clip(dot, 0.0, 1.0)


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, EPS)


def _quat_rotate_xyzw(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat_xyz = quat[..., :3]
    quat_w = quat[..., 3:4]
    uv = np.cross(quat_xyz, vector, axis=-1)
    uuv = np.cross(quat_xyz, uv, axis=-1)
    return vector + 2.0 * (quat_w * uv + uuv)


def _horizontal_speed(pos_w: np.ndarray, fps: float) -> np.ndarray:
    if pos_w.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(pos_w[:, :2], axis=0), axis=1) * float(fps)


def _contact_score_array(value, frames: int, region: str) -> np.ndarray:
    score = np.asarray(value, dtype=np.float64)
    if score.shape != (frames,):
        raise ValueError(f"contact_score[{region!r}] must have shape ({frames},), got {score.shape}.")
    if not np.all(np.isfinite(score)):
        raise ValueError(f"contact_score[{region!r}] contains NaN or inf values.")
    return np.clip(score, 0.0, 1.0)


def _safe_norm(value: np.ndarray, axis: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    finite = np.all(np.isfinite(value), axis=axis)
    norm = np.linalg.norm(np.where(np.isfinite(value), value, 0.0), axis=axis)
    return np.where(finite, norm, np.nan)


def _finite_mean(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def _finite_max(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite))


def _finite_min(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.min(finite))


def _worst_joint_name(violation: np.ndarray, joint_names: list[str]) -> str | None:
    finite = np.where(np.isfinite(violation), violation, 0.0)
    if finite.size == 0 or float(np.max(finite)) <= 0.0:
        return None
    return joint_names[int(np.argmax(np.max(finite, axis=0)))]


def _indices_for_names(available: list[str], requested: list[str], label: str) -> list[int]:
    lookup = {name: idx for idx, name in enumerate(available)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise ValueError(f"{label} is missing requested names: {missing}.")
    return [lookup[name] for name in requested]


def _require_shape(value: np.ndarray, name: str, expected: tuple[int, ...]) -> None:
    shape = np.asarray(value).shape
    if shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {shape}.")


def _nonfinite_count(value: np.ndarray) -> int:
    return int(np.size(value) - np.count_nonzero(np.isfinite(np.asarray(value))))


def _finite_float(value, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _cfg(config: Mapping[str, Any] | None, section: str, key: str, default):
    section_config = _lookup(config, section, {})
    if isinstance(section_config, Mapping) and key in section_config:
        return section_config[key]
    section_key = f"{section}_{key}"
    if _has_key(config, section_key):
        return _lookup(config, section_key, default)
    return _lookup(config, key, default)


def _lookup(config: Mapping[str, Any] | None, key: str, default):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _has_key(config: Mapping[str, Any] | None, key: str) -> bool:
    if config is None:
        return False
    if isinstance(config, Mapping):
        return key in config
    return hasattr(config, key)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
