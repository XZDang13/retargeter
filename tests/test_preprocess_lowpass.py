from __future__ import annotations

import numpy as np

from retargeter.preprocess.lowpass import (
    enforce_quat_sign_continuity_xyzw,
    lowpass_positions,
    lowpass_quaternions_xyzw,
)


def test_lowpass_positions_smooths_noisy_trajectory():
    fps = 30.0
    t = np.arange(120) / fps
    clean = np.sin(2.0 * np.pi * 1.0 * t)
    noisy = clean + 0.25 * np.sin(2.0 * np.pi * 10.0 * t)
    positions = np.stack([noisy, np.zeros_like(noisy), np.zeros_like(noisy)], axis=1)

    filtered = lowpass_positions(positions, fps=fps, cutoff_hz=3.0, order=4, mode="offline_zero_phase")

    assert filtered.shape == positions.shape
    assert np.std(np.diff(filtered[:, 0])) < np.std(np.diff(positions[:, 0]))


def test_quaternion_filter_preserves_norms_and_continuity():
    quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (40, 1))
    quats[::2] *= -1.0

    continuous = enforce_quat_sign_continuity_xyzw(quats)
    assert np.all(np.sum(continuous[1:] * continuous[:-1], axis=1) >= 0.0)

    filtered = lowpass_quaternions_xyzw(quats, fps=30.0, cutoff_hz=6.0, order=4, mode="offline_zero_phase")
    assert filtered.shape == quats.shape
    assert np.allclose(np.linalg.norm(filtered, axis=1), 1.0, atol=1e-6)


def test_lowpass_short_sequence_uses_fallback_without_crashing():
    positions = np.random.default_rng(0).normal(size=(5, 3))
    filtered = lowpass_positions(positions, fps=30.0, cutoff_hz=3.0, order=4, mode="offline_zero_phase")

    assert filtered.shape == positions.shape
    assert np.all(np.isfinite(filtered))

