from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from retargeter.newton import RobotSpec, RetargetedMotion, TorchRobotFKResult


DEFAULT_FPS = 30.0
EPS = 1.0e-12

DEFAULT_CONTACT_POINTS = {
    "left_foot": {"body": "left_ankle_roll_link"},
    "right_foot": {"body": "right_ankle_roll_link"},
    "left_toe": {"body": "left_toe_link"},
    "right_toe": {"body": "right_toe_link"},
    "left_heel": {"body": "left_ankle_roll_link"},
    "right_heel": {"body": "right_ankle_roll_link"},
}


def motion_fidelity_loss(
    retargeted: RetargetedMotion,
    refined_fk: TorchRobotFKResult,
    refined_joint_pos: torch.Tensor,
    refined_root_pos: torch.Tensor,
    config: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Track the retargeted robot motion while preserving root-local body shape."""

    _validate_fk_result(refined_fk)
    _validate_tensor_shape(refined_root_pos, "refined_root_pos", ndim=2, trailing=(3,))
    _validate_tensor_shape(refined_joint_pos, "refined_joint_pos", ndim=2)
    frames = _validate_time_match(
        refined_fk.body_pos_w,
        refined_root_pos,
        "refined_fk.body_pos_w",
        "refined_root_pos",
    )
    _validate_time_match(refined_fk.body_pos_w, refined_joint_pos, "refined_fk.body_pos_w", "refined_joint_pos")

    if retargeted.num_frames() != frames:
        raise ValueError(f"retargeted has {retargeted.num_frames()} frames but refined tensors have {frames}.")
    if refined_joint_pos.shape[1] != len(retargeted.joint_names):
        raise ValueError(
            f"refined_joint_pos must have {len(retargeted.joint_names)} joints, got {refined_joint_pos.shape[1]}."
        )

    body_names = _motion_body_names(retargeted, refined_fk, config)
    retargeted_indices = _indices_for_names(retargeted.body_names, body_names, "retargeted.body_names")
    refined_indices = _indices_for_names(refined_fk.body_names, body_names, "refined_fk.body_names")

    target_body_pos = _retargeted_tensor(retargeted.body_pos_w[:, retargeted_indices, :], refined_fk.body_pos_w)
    target_body_quat = _retargeted_tensor(retargeted.body_quat_xyzw[:, retargeted_indices, :], refined_fk.body_quat_xyzw)
    target_root = _retargeted_tensor(retargeted.root_pos_w, refined_root_pos)
    target_joint_pos = _retargeted_tensor(retargeted.joint_pos, refined_joint_pos)

    body_pos = refined_fk.body_pos_w[:, refined_indices, :]
    body_quat = refined_fk.body_quat_xyzw[:, refined_indices, :]

    body_pos_loss = torch.mean(torch.abs(body_pos - target_body_pos))
    body_local_loss = torch.mean(
        torch.abs((body_pos - refined_root_pos[:, None, :]) - (target_body_pos - target_root[:, None, :]))
    )
    body_quat_loss = _sign_invariant_quat_loss(body_quat, target_body_quat)
    root_pos_loss = torch.mean(torch.abs(refined_root_pos - target_root))
    joint_pos_loss = torch.mean(torch.abs(refined_joint_pos - target_joint_pos))

    body_pos_weight = _cfg_float(config, "motion_fidelity", "body_pos_weight", 1.0)
    local_body_pos_weight = _cfg_float(config, "motion_fidelity", "local_body_pos_weight", 1.0)
    body_quat_weight = _cfg_float(config, "motion_fidelity", "body_quat_weight", 0.1)
    root_pos_weight = _cfg_float(config, "motion_fidelity", "root_pos_weight", 1.0)
    joint_pos_weight = _cfg_float(config, "motion_fidelity", "joint_pos_weight", 0.01)

    loss = (
        body_pos_weight * body_pos_loss
        + local_body_pos_weight * body_local_loss
        + body_quat_weight * body_quat_loss
        + root_pos_weight * root_pos_loss
        + joint_pos_weight * joint_pos_loss
    )
    metrics = _metrics(
        loss=loss,
        body_pos=body_pos_loss,
        local_body_pos=body_local_loss,
        body_quat=body_quat_loss,
        root_pos=root_pos_loss,
        joint_pos=joint_pos_loss,
        body_count=_scalar_like(float(len(body_names)), loss),
    )
    return loss, metrics


def joint_feasibility_loss(
    refined_joint_pos: torch.Tensor,
    refined_joint_vel: torch.Tensor,
    robot_spec: RobotSpec,
    config: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize joint position and velocity limit violations."""

    _validate_tensor_shape(refined_joint_pos, "refined_joint_pos", ndim=2)
    _validate_tensor_shape(refined_joint_vel, "refined_joint_vel", ndim=2)
    if refined_joint_pos.shape[1] != robot_spec.num_dofs:
        raise ValueError(f"refined_joint_pos must have {robot_spec.num_dofs} joints, got {refined_joint_pos.shape[1]}.")
    if refined_joint_vel.shape[1] != robot_spec.num_dofs:
        raise ValueError(f"refined_joint_vel must have {robot_spec.num_dofs} joints, got {refined_joint_vel.shape[1]}.")
    if refined_joint_vel.shape[0] not in {refined_joint_pos.shape[0], max(refined_joint_pos.shape[0] - 1, 0)}:
        raise ValueError(
            "refined_joint_vel must have T or T-1 frames relative to refined_joint_pos; "
            f"got {refined_joint_vel.shape[0]} vs {refined_joint_pos.shape[0]}."
        )

    margin = _cfg_float(config, "joint_feasibility", "joint_range_margin", 0.98)
    if margin <= 0.0 or margin > 1.0 or not np.isfinite(margin):
        raise ValueError(f"joint_range_margin must be in (0, 1], got {margin!r}.")

    lower_np = np.asarray(robot_spec.joint_lower_rad, dtype=np.float64)
    upper_np = np.asarray(robot_spec.joint_upper_rad, dtype=np.float64)
    center = 0.5 * (lower_np + upper_np)
    half_span = 0.5 * (upper_np - lower_np) * margin
    lower = _as_tensor(center - half_span, refined_joint_pos)
    upper = _as_tensor(center + half_span, refined_joint_pos)
    velocity_limits = _as_tensor(
        np.asarray(robot_spec.velocity_limits_rad_s, dtype=np.float64) * margin,
        refined_joint_pos,
    )

    joint_violation = torch.relu(lower.reshape(1, -1) - refined_joint_pos) + torch.relu(
        refined_joint_pos - upper.reshape(1, -1)
    )
    joint_limit_loss = joint_violation.mean()
    if refined_joint_vel.numel() == 0:
        velocity_loss = _zero_like(refined_joint_pos)
    else:
        velocity_loss = torch.relu(torch.abs(refined_joint_vel) - velocity_limits.reshape(1, -1)).mean()

    joint_weight = _cfg_float(config, "joint_feasibility", "weight", 1000.0)
    velocity_weight = _cfg_float(config, "joint_feasibility", "velocity_weight", 1000.0)
    loss = joint_weight * joint_limit_loss + velocity_weight * velocity_loss
    metrics = _metrics(loss=loss, joint_limit=joint_limit_loss, joint_velocity=velocity_loss)
    return loss, metrics


def grounding_loss(
    refined_fk: TorchRobotFKResult,
    contact_score,
    ground_height: float,
    robot_spec: RobotSpec,
    config: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep soft-contact foot points on the ground plane."""

    _validate_fk_result(refined_fk)
    contact_points = _contact_points_w(refined_fk, contact_score, robot_spec, config, "grounding")
    if not contact_points:
        zero = _zero_like(refined_fk.body_pos_w)
        return zero, _metrics(loss=zero, contact_weight_mean=zero, contact_point_count=zero)

    terms = []
    weights = []
    for point, score in contact_points:
        height = point[:, 2] - float(ground_height)
        terms.append(height.square() * score)
        weights.append(score)
    raw_loss = torch.stack(terms, dim=1).mean()
    weight_mean = torch.stack(weights, dim=1).mean()
    weight = _cfg_float(config, "grounding", "weight", 10.0)
    loss = weight * raw_loss
    metrics = _metrics(
        loss=loss,
        grounding=raw_loss,
        contact_weight_mean=weight_mean,
        contact_point_count=_scalar_like(float(len(contact_points)), loss),
    )
    return loss, metrics


def skating_loss(
    refined_fk: TorchRobotFKResult,
    contact_score,
    robot_spec: RobotSpec,
    config: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize horizontal contact-point velocity using soft contact scores."""

    _validate_fk_result(refined_fk)
    contact_points = _contact_points_w(refined_fk, contact_score, robot_spec, config, "skating")
    if refined_fk.body_pos_w.shape[0] < 2 or not contact_points:
        zero = _zero_like(refined_fk.body_pos_w)
        return zero, _metrics(loss=zero, skating=zero, contact_weight_mean=zero, contact_point_count=zero)

    dt = _dt_from_config(config)
    per_point_losses = []
    weights = []
    for point, score in contact_points:
        horizontal_velocity = (point[1:, :2] - point[:-1, :2]) / dt
        speed = torch.linalg.vector_norm(horizontal_velocity, dim=-1)
        velocity_weight = score[:-1]
        weighted_speed = speed * velocity_weight
        positive_contact_count = (velocity_weight > 0.0).to(speed.dtype).sum()
        per_point_losses.append(
            torch.where(
                positive_contact_count > 0.0,
                weighted_speed.sum() / (positive_contact_count + EPS),
                _zero_like(speed),
            )
        )
        weights.append(velocity_weight)
    raw_loss = torch.stack(per_point_losses).sum()
    weight_mean = torch.stack(weights, dim=1).mean()
    weight = _cfg_float(config, "skating", "weight", 0.02)
    loss = weight * raw_loss
    metrics = _metrics(
        loss=loss,
        skating=raw_loss,
        contact_weight_mean=weight_mean,
        contact_point_count=_scalar_like(float(len(contact_points)), loss),
    )
    return loss, metrics


def smoothness_loss(
    refined_root_pos: torch.Tensor,
    refined_joint_pos: torch.Tensor,
    config: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PHUMA-style velocity curvature penalty for root and joint trajectories."""

    _validate_tensor_shape(refined_root_pos, "refined_root_pos", ndim=2, trailing=(3,))
    _validate_tensor_shape(refined_joint_pos, "refined_joint_pos", ndim=2)
    _validate_time_match(refined_root_pos, refined_joint_pos, "refined_root_pos", "refined_joint_pos")

    dt = _dt_from_config(config)
    root_raw = _velocity_curvature_l1(refined_root_pos, dt)
    joint_raw = _velocity_curvature_l1(refined_joint_pos, dt)
    root_weight = _cfg_float(config, "smoothness", "root_weight", 1.0)
    joint_weight = _cfg_float(config, "smoothness", "joint_weight", 1.0)
    weight = _cfg_float(config, "smoothness", "weight", 0.05)
    loss = weight * (root_weight * root_raw + joint_weight * joint_raw)
    metrics = _metrics(loss=loss, root=root_raw, joint=joint_raw)
    return loss, metrics


def delta_regularization_loss(
    root_delta: torch.Tensor,
    joint_delta: torch.Tensor,
    config: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep refined root and joint deltas small."""

    _validate_tensor_shape(root_delta, "root_delta", ndim=2, trailing=(3,))
    _validate_tensor_shape(joint_delta, "joint_delta", ndim=2)
    _validate_time_match(root_delta, joint_delta, "root_delta", "joint_delta")

    root_raw = root_delta.square().mean()
    joint_raw = joint_delta.square().mean()
    root_weight = _cfg_float(config, "delta_regularization", "root_weight", 1.0)
    joint_weight = _cfg_float(config, "delta_regularization", "joint_weight", 1.0)
    weight = _cfg_float(config, "delta_regularization", "weight", 1.0)
    loss = weight * (root_weight * root_raw + joint_weight * joint_raw)
    metrics = _metrics(loss=loss, root=root_raw, joint=joint_raw)
    return loss, metrics


def total_refinement_loss(
    retargeted: RetargetedMotion,
    refined_fk: TorchRobotFKResult,
    refined_joint_pos: torch.Tensor,
    refined_root_pos: torch.Tensor,
    refined_joint_vel: torch.Tensor,
    root_delta: torch.Tensor,
    joint_delta: torch.Tensor,
    contact_score,
    ground_height: float,
    robot_spec: RobotSpec,
    config: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the weighted refined objective and merged metrics."""

    timed_config = _with_retargeted_timing(config, retargeted)
    motion_loss, motion_metrics = motion_fidelity_loss(retargeted, refined_fk, refined_joint_pos, refined_root_pos, timed_config)
    feasibility_loss, feasibility_metrics = joint_feasibility_loss(refined_joint_pos, refined_joint_vel, robot_spec, timed_config)
    ground_loss, ground_metrics = grounding_loss(refined_fk, contact_score, ground_height, robot_spec, timed_config)
    skate_loss, skate_metrics = skating_loss(refined_fk, contact_score, robot_spec, timed_config)
    smooth_loss, smooth_metrics = smoothness_loss(refined_root_pos, refined_joint_pos, timed_config)
    delta_loss, delta_metrics = delta_regularization_loss(root_delta, joint_delta, timed_config)

    total = motion_loss + feasibility_loss + ground_loss + skate_loss + smooth_loss + delta_loss
    metrics = {"loss": _metric(total)}
    metrics.update(_prefix_metrics("motion_fidelity", motion_metrics))
    metrics.update(_prefix_metrics("joint_feasibility", feasibility_metrics))
    metrics.update(_prefix_metrics("grounding", ground_metrics))
    metrics.update(_prefix_metrics("skating", skate_metrics))
    metrics.update(_prefix_metrics("smoothness", smooth_metrics))
    metrics.update(_prefix_metrics("delta_regularization", delta_metrics))
    return total, metrics

def _motion_body_names(
    retargeted: RetargetedMotion,
    refined_fk: TorchRobotFKResult,
    config: Mapping[str, Any] | None,
) -> list[str]:
    configured = _cfg(config, "motion_fidelity", "body_names", None)
    if configured is not None:
        names = [str(name) for name in configured]
    else:
        refined_names = set(refined_fk.body_names)
        names = [name for name in retargeted.body_names if name in refined_names]
    if not names:
        raise ValueError("motion_fidelity body_names resolved to an empty list.")
    missing_retargeted = [name for name in names if name not in retargeted.body_names]
    missing_refined = [name for name in names if name not in refined_fk.body_names]
    if missing_retargeted:
        raise ValueError(f"motion_fidelity body_names missing from retargeted: {missing_retargeted}.")
    if missing_refined:
        raise ValueError(f"motion_fidelity body_names missing from refined_fk: {missing_refined}.")
    return names


def _contact_points_w(
    refined_fk: TorchRobotFKResult,
    contact_score,
    robot_spec: RobotSpec,
    config: Mapping[str, Any] | None,
    section: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    scores = _contact_score_mapping(contact_score)
    specs = _contact_point_specs(config, section)
    body_index = {name: idx for idx, name in enumerate(refined_fk.body_names)}
    points: list[tuple[torch.Tensor, torch.Tensor]] = []
    for region, raw_score in scores.items():
        if region not in specs:
            continue
        point_spec = specs[region]
        body = str(point_spec["body"])
        if not robot_spec.has_body(body):
            raise ValueError(f"contact point region {region!r} references body {body!r} missing from RobotSpec.")
        if body not in body_index:
            raise ValueError(f"contact point region {region!r} references body {body!r} missing from refined_fk.")
        score = _contact_score_tensor(raw_score, refined_fk.body_pos_w, region)
        pos = refined_fk.body_pos_w[:, body_index[body], :]
        local_pos = _as_tensor(point_spec.get("local_pos", [0.0, 0.0, 0.0]), refined_fk.body_pos_w)
        if local_pos.shape != (3,):
            raise ValueError(f"contact point local_pos for region {region!r} must have shape [3], got {tuple(local_pos.shape)}.")
        if torch.any(local_pos != 0.0):
            quat = refined_fk.body_quat_xyzw[:, body_index[body], :]
            pos = pos + _quat_rotate_xyzw(_quat_normalize(quat), local_pos.reshape(1, 3).expand_as(pos))
        points.append((pos, score))
    return points


def _contact_score_mapping(contact_score) -> Mapping[str, Any]:
    if hasattr(contact_score, "contact_score"):
        contact_score = contact_score.contact_score
    if not isinstance(contact_score, Mapping):
        raise TypeError("contact_score must be a mapping or object with a contact_score mapping.")
    return contact_score


def _contact_point_specs(config: Mapping[str, Any] | None, section: str) -> dict[str, dict[str, Any]]:
    specs = {region: dict(value) for region, value in DEFAULT_CONTACT_POINTS.items()}
    for overrides in (_lookup(config, "contact_points", None), _cfg(config, section, "contact_points", None)):
        if overrides is None:
            continue
        if not isinstance(overrides, Mapping):
            raise TypeError("contact_points must be a mapping.")
        for region, raw in overrides.items():
            if isinstance(raw, str):
                specs[str(region)] = {"body": raw}
            elif isinstance(raw, Mapping):
                if "body" not in raw:
                    raise ValueError(f"contact_points.{region} must define body.")
                entry = {"body": str(raw["body"])}
                if "local_pos" in raw:
                    entry["local_pos"] = raw["local_pos"]
                specs[str(region)] = entry
            else:
                raise TypeError(f"contact_points.{region} must be a body string or mapping.")
    return specs


def _contact_score_tensor(value, like: torch.Tensor, region: str) -> torch.Tensor:
    score = _as_tensor(value, like)
    if score.ndim != 1:
        raise ValueError(f"contact_score[{region!r}] must have shape [T], got {tuple(score.shape)}.")
    if score.shape[0] != like.shape[0]:
        raise ValueError(f"contact_score[{region!r}] has {score.shape[0]} frames but refined has {like.shape[0]}.")
    if not torch.isfinite(score).all():
        raise ValueError(f"contact_score[{region!r}] contains NaN or inf values.")
    return score.clamp(0.0, 1.0)


def _sign_invariant_quat_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = _quat_normalize(pred)
    target_n = _quat_normalize(target)
    dot = torch.sum(pred_n * target_n, dim=-1).abs().clamp(max=1.0)
    return torch.mean(1.0 - dot)


def _velocity_curvature_l1(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[0] < 4:
        return _zero_like(values)
    velocity = torch.diff(values, n=1, dim=0) / dt
    last = velocity[0:-2]
    this = velocity[1:-1]
    next_value = velocity[2:]
    return torch.mean(torch.abs(this - 0.5 * (last + next_value)))


def _dt_from_config(config: Mapping[str, Any] | None, retargeted: RetargetedMotion | None = None) -> float:
    dt = _lookup(config, "dt", None)
    if dt is not None:
        value = float(dt)
    else:
        fps = _lookup(config, "fps", None)
        if fps is None and retargeted is not None:
            fps = retargeted.fps
        if fps is None:
            fps = DEFAULT_FPS
        fps_value = float(fps)
        if fps_value <= 0.0 or not np.isfinite(fps_value):
            raise ValueError(f"fps must be positive and finite, got {fps!r}.")
        value = 1.0 / fps_value
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError(f"dt must be positive and finite, got {dt!r}.")
    return value


def _with_retargeted_timing(config: Mapping[str, Any] | None, retargeted: RetargetedMotion) -> Mapping[str, Any] | None:
    if _has_key(config, "dt") or _has_key(config, "fps"):
        return config
    if isinstance(config, Mapping):
        merged = dict(config)
        merged["fps"] = retargeted.fps
        return merged
    if config is None:
        return {"fps": retargeted.fps}
    return config


def _cfg_float(config: Mapping[str, Any] | None, section: str, key: str, default: float) -> float:
    value = float(_cfg(config, section, key, default))
    if not np.isfinite(value):
        raise ValueError(f"{section}.{key} must be finite, got {value!r}.")
    return value


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


def _validate_fk_result(refined_fk: TorchRobotFKResult) -> None:
    if not isinstance(refined_fk.body_names, list) or not refined_fk.body_names:
        raise ValueError("refined_fk.body_names must be a non-empty list.")
    _validate_tensor_shape(refined_fk.body_pos_w, "refined_fk.body_pos_w", ndim=3, trailing=(len(refined_fk.body_names), 3))
    _validate_tensor_shape(
        refined_fk.body_quat_xyzw,
        "refined_fk.body_quat_xyzw",
        ndim=3,
        trailing=(len(refined_fk.body_names), 4),
    )
    _validate_time_match(refined_fk.body_pos_w, refined_fk.body_quat_xyzw, "refined_fk.body_pos_w", "refined_fk.body_quat_xyzw")


def _validate_tensor_shape(tensor: torch.Tensor, name: str, *, ndim: int, trailing: tuple[int, ...] | None = None) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {tuple(tensor.shape)}.")
    if trailing is not None and tuple(tensor.shape[-len(trailing) :]) != trailing:
        raise ValueError(f"{name} must end with shape {trailing}, got {tuple(tensor.shape)}.")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must use a floating point dtype.")


def _validate_time_match(lhs: torch.Tensor, rhs: torch.Tensor, lhs_name: str, rhs_name: str) -> int:
    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError(f"{lhs_name} and {rhs_name} must have the same T dimension, got {lhs.shape[0]} and {rhs.shape[0]}.")
    return int(lhs.shape[0])


def _indices_for_names(available: list[str], requested: list[str], label: str) -> list[int]:
    lookup = {name: idx for idx, name in enumerate(available)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise ValueError(f"{label} is missing requested names: {missing}.")
    return [lookup[name] for name in requested]


def _retargeted_tensor(value: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=like.dtype, device=like.device)


def _as_tensor(value, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _metrics(**values: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: _metric(value) for name, value in values.items()}


def _metric(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("metrics must be tensors.")
    if value.ndim != 0:
        value = value.mean()
    return value.detach().clone()


def _prefix_metrics(prefix: str, metrics: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}


def _scalar_like(value: float, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(EPS)


def _quat_rotate_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat[..., :3]
    quat_w = quat[..., 3:4]
    uv = torch.cross(quat_xyz, vector, dim=-1)
    uuv = torch.cross(quat_xyz, uv, dim=-1)
    return vector + 2.0 * (quat_w * uv + uuv)
