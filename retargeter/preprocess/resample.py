from __future__ import annotations

import copy

import numpy as np

from .smpl_motion import SMPLMotion


def resample_smpl_motion(motion: SMPLMotion, target_fps: float) -> SMPLMotion:
    motion.validate()
    target_fps = _validate_target_fps(target_fps)
    source_fps = float(motion.fps)
    source_frames = motion.num_frames()

    if np.isclose(source_fps, target_fps, rtol=1e-9, atol=1e-9):
        out = motion.copy()
        _set_resample_metadata(out, source_fps, target_fps, source_frames, source_frames, resampled=False)
        return out

    target_times = _target_times(source_frames, source_fps, target_fps)
    source_times = np.arange(source_frames, dtype=np.float64) / source_fps

    out = SMPLMotion(
        model_type=motion.model_type,
        fps=target_fps,
        transl=_interp_linear(motion.transl, source_times, target_times),
        global_orient=_interp_axis_angle(motion.global_orient, source_times, target_times),
        body_pose=_interp_axis_angle(motion.body_pose, source_times, target_times),
        betas=_resample_betas(motion.betas, source_times, target_times, source_frames),
        gender=motion.gender,
        left_hand_pose=_interp_optional_axis_angle(motion.left_hand_pose, source_times, target_times),
        right_hand_pose=_interp_optional_axis_angle(motion.right_hand_pose, source_times, target_times),
        jaw_pose=_interp_optional_axis_angle(motion.jaw_pose, source_times, target_times),
        leye_pose=_interp_optional_axis_angle(motion.leye_pose, source_times, target_times),
        reye_pose=_interp_optional_axis_angle(motion.reye_pose, source_times, target_times),
        expression=_interp_optional_linear(motion.expression, source_times, target_times),
        metadata=copy.deepcopy(motion.metadata),
    )
    _set_resample_metadata(out, source_fps, target_fps, source_frames, out.num_frames(), resampled=True)
    out.validate()
    return out


def _validate_target_fps(target_fps: float) -> float:
    value = float(target_fps)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"target_fps must be positive and finite, got {target_fps!r}.")
    return value


def _target_times(source_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if source_frames <= 1:
        return np.zeros((source_frames,), dtype=np.float64)
    duration = (source_frames - 1) / source_fps
    target_count = max(1, int(np.floor(duration * target_fps + 1e-9)) + 1)
    times = np.arange(target_count, dtype=np.float64) / target_fps
    return np.minimum(times, duration)


def _interp_optional_linear(
    values: np.ndarray | None,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray | None:
    if values is None:
        return None
    return _interp_linear(values, source_times, target_times)


def _interp_linear(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != source_times.shape[0]:
        raise ValueError(f"Per-frame data must have T={source_times.shape[0]}, got {values.shape}.")
    if values.shape[0] <= 1:
        return values.copy()
    flat = values.reshape(values.shape[0], -1)
    out = np.empty((target_times.shape[0], flat.shape[1]), dtype=np.float64)
    for dim in range(flat.shape[1]):
        out[:, dim] = np.interp(target_times, source_times, flat[:, dim])
    return out.reshape((target_times.shape[0],) + values.shape[1:])


def _resample_betas(
    betas: np.ndarray | None,
    source_times: np.ndarray,
    target_times: np.ndarray,
    source_frames: int,
) -> np.ndarray | None:
    if betas is None:
        return None
    values = np.asarray(betas, dtype=np.float64)
    if values.ndim == 1 or (values.ndim >= 2 and values.shape[0] == 1):
        return values.copy()
    if values.shape[0] != source_frames:
        raise ValueError(f"betas must be shape [B], [1, B], or [T, B], got {values.shape}.")
    return _interp_linear(values, source_times, target_times)


def _interp_optional_axis_angle(
    values: np.ndarray | None,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray | None:
    if values is None:
        return None
    return _interp_axis_angle(values, source_times, target_times)


def _interp_axis_angle(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != source_times.shape[0]:
        raise ValueError(f"Axis-angle data must have T={source_times.shape[0]}, got {values.shape}.")
    if values.shape[-1] % 3 != 0:
        raise ValueError(f"Axis-angle data last dimension must be divisible by 3, got {values.shape}.")
    if values.shape[0] <= 1:
        return values.copy()

    original_shape = values.shape
    rotvec = values.reshape(values.shape[0], -1, 3)
    quats = _axis_angle_to_quat_xyzw(rotvec)

    indices = np.searchsorted(source_times, target_times, side="right") - 1
    indices = np.clip(indices, 0, source_times.shape[0] - 2)
    next_indices = indices + 1
    span = source_times[next_indices] - source_times[indices]
    alpha = np.divide(
        target_times - source_times[indices],
        span,
        out=np.zeros_like(target_times),
        where=span > 1e-12,
    )

    interp = _slerp_xyzw(quats[indices], quats[next_indices], alpha)
    return _quat_to_axis_angle_xyzw(interp).reshape((target_times.shape[0],) + original_shape[1:])


def _axis_angle_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.divide(np.sin(half), angle, out=np.full_like(angle, 0.5), where=angle > 1e-12)
    quat = np.concatenate([rotvec * scale, np.cos(half)], axis=-1)
    return _normalize_quat_xyzw(quat)


def _quat_to_axis_angle_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_xyzw(quat)
    quat = np.where(quat[..., 3:4] < 0.0, -quat, quat)
    xyz = quat[..., :3]
    w = np.clip(quat[..., 3:4], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.divide(angle, sin_half, out=np.full_like(angle, 2.0), where=sin_half > 1e-12)
    return xyz * scale


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = _normalize_quat_xyzw(q0)
    q1 = _normalize_quat_xyzw(q1)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape[0] != q0.shape[0]:
        raise ValueError(f"alpha length must match target frames {q0.shape[0]}, got {alpha.shape}.")
    alpha = alpha.reshape((q0.shape[0],) + (1,) * (q0.ndim - 1))
    alpha = np.broadcast_to(alpha, q0.shape[:-1] + (1,)).copy()
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)

    linear = dot > 0.9995
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.divide(
        np.sin(theta_0 - theta),
        sin_theta_0,
        out=(1.0 - alpha).copy(),
        where=sin_theta_0 > 1e-12,
    )
    s1 = np.divide(
        sin_theta,
        sin_theta_0,
        out=alpha.copy(),
        where=sin_theta_0 > 1e-12,
    )
    slerped = s0 * q0 + s1 * q1
    lerped = (1.0 - alpha) * q0 + alpha * q1
    return _normalize_quat_xyzw(np.where(linear, lerped, slerped))


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    out = np.divide(quat, norm, out=np.zeros_like(quat), where=norm > 1e-12)
    invalid = norm <= 1e-12
    if np.any(invalid):
        out[..., 3] = np.where(invalid[..., 0], 1.0, out[..., 3])
    return out


def _set_resample_metadata(
    motion: SMPLMotion,
    source_fps: float,
    target_fps: float,
    source_frames: int,
    target_frames: int,
    *,
    resampled: bool,
) -> None:
    motion.metadata["resample"] = {
        "resampled": bool(resampled),
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "source_frame_count": int(source_frames),
        "target_frame_count": int(target_frames),
    }
