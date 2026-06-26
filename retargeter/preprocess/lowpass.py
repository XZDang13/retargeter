from __future__ import annotations

import warnings

import numpy as np

from .canonical import CanonicalHumanMotion
from .config import LowPassConfig


def lowpass_positions(positions, fps: float, cutoff_hz: float, order: int, mode: str):
    return _lowpass_array(np.asarray(positions, dtype=np.float64), fps, cutoff_hz, order, mode)


def lowpass_quaternions_xyzw(quats, fps: float, cutoff_hz: float, order: int, mode: str):
    quats = normalize_quat_xyzw(enforce_quat_sign_continuity_xyzw(np.asarray(quats, dtype=np.float64)))
    filtered = _lowpass_array(quats, fps, cutoff_hz, order, mode)
    return normalize_quat_xyzw(filtered)


def enforce_quat_sign_continuity_xyzw(quats):
    out = np.asarray(quats, dtype=np.float64).copy()
    if out.shape[-1] != 4:
        raise ValueError(f"Quaternions must have last dimension 4, got {out.shape}.")
    if out.shape[0] <= 1:
        return out
    for i in range(1, out.shape[0]):
        dot = np.sum(out[i] * out[i - 1], axis=-1)
        mask = dot < 0.0
        out[i][mask] *= -1.0
    return out


def normalize_quat_xyzw(quats):
    out = np.asarray(quats, dtype=np.float64).copy()
    if out.shape[-1] != 4:
        raise ValueError(f"Quaternions must have last dimension 4, got {out.shape}.")
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    invalid = norms < 1e-12
    out = np.divide(out, norms, out=np.zeros_like(out), where=~invalid)
    if np.any(invalid):
        out[..., 3] = np.where(invalid[..., 0], 1.0, out[..., 3])
    return out


class MotionLowPassFilter:
    def __init__(self, config: LowPassConfig):
        self.config = config

    def apply(self, motion: CanonicalHumanMotion) -> CanonicalHumanMotion:
        if not self.config.enabled:
            return motion.copy()

        filtered = motion.copy()
        pos = filtered.body_pos_w.copy()
        quat = filtered.body_quat_xyzw.copy()
        for body_index, name in enumerate(filtered.body_names):
            position_cutoff = (
                self.config.root_position_cutoff_hz
                if name == "pelvis" or body_index == 0
                else self.config.position_cutoff_hz
            )
            rotation_cutoff = (
                self.config.root_rotation_cutoff_hz
                if name == "pelvis" or body_index == 0
                else self.config.rotation_cutoff_hz
            )
            pos[:, body_index, :] = lowpass_positions(
                pos[:, body_index, :],
                filtered.fps,
                position_cutoff,
                self.config.order,
                self.config.mode,
            )
            quat[:, body_index, :] = lowpass_quaternions_xyzw(
                quat[:, body_index, :],
                filtered.fps,
                rotation_cutoff,
                self.config.order,
                self.config.mode,
            )

        filtered.body_pos_w = pos
        filtered.body_quat_xyzw = quat
        filtered.metadata["lowpass_applied"] = True
        return filtered


def _lowpass_array(data: np.ndarray, fps: float, cutoff_hz: float, order: int, mode: str) -> np.ndarray:
    if data.shape[0] <= 1 or cutoff_hz <= 0.0 or fps <= 0.0:
        return data.copy()

    nyquist = 0.5 * fps
    if cutoff_hz >= nyquist:
        return data.copy()

    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        warnings.warn("scipy is unavailable; using causal exponential low-pass fallback.", RuntimeWarning)
        return _exponential_smooth(data, fps, cutoff_hz)

    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    if mode in {"offline_zero_phase", "zero_phase"}:
        padlen = 3 * max(len(a), len(b))
        if data.shape[0] <= padlen:
            warnings.warn(
                f"Sequence length {data.shape[0]} is too short for zero-phase filtfilt padlen {padlen}; "
                "using causal exponential low-pass fallback.",
                RuntimeWarning,
            )
            return _exponential_smooth(data, fps, cutoff_hz)
        return filtfilt(b, a, data, axis=0)

    warnings.warn(f"Unsupported low-pass mode {mode!r}; using causal exponential fallback.", RuntimeWarning)
    return _exponential_smooth(data, fps, cutoff_hz)


def _exponential_smooth(data: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    out = data.copy()
    if out.shape[0] <= 1 or cutoff_hz <= 0.0 or fps <= 0.0:
        return out
    dt = 1.0 / fps
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    for i in range(1, out.shape[0]):
        out[i] = alpha * out[i] + (1.0 - alpha) * out[i - 1]
    return out

