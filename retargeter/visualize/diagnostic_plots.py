from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from retargeter.newton import RetargetedMotion, RobotSpec
from retargeter.preprocess import PreprocessResult


def plot_contact_scores(preprocess_result: PreprocessResult, output_path: Path | str) -> Path:
    _require_contact(preprocess_result)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    for region, values in preprocess_result.contact.contact_score.items():
        ax.plot(np.asarray(values), label=region)
    ax.set_title("Contact scores")
    ax.set_xlabel("frame")
    ax.set_ylabel("score")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    return _save(fig, output_path)


def plot_foot_height_and_speed(preprocess_result: PreprocessResult, output_path: Path | str) -> Path:
    _require_contact(preprocess_result)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=120, sharex=True)
    for region, values in preprocess_result.contact.foot_height.items():
        axes[0].plot(np.asarray(values), label=region)
    for region, values in preprocess_result.contact.foot_speed.items():
        axes[1].plot(np.asarray(values), label=region)
    axes[0].set_title("Foot height")
    axes[0].set_ylabel("height m")
    axes[1].set_title("Foot speed")
    axes[1].set_ylabel("speed m/s")
    axes[1].set_xlabel("frame")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    return _save(fig, output_path)


def plot_ik_errors(retargeted_motion: RetargetedMotion, output_path: Path | str) -> Path:
    retargeted_motion.validate()
    coarse_cost = _diagnostic_series(retargeted_motion, ("coarse_alignment", "cost"))
    tracking_cost = _diagnostic_series(retargeted_motion, ("full_body_tracking", "cost"))
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    if coarse_cost is not None:
        ax.plot(coarse_cost, label="coarse alignment cost")
    if tracking_cost is not None:
        ax.plot(tracking_cost, label="full body tracking cost")
    if coarse_cost is None and tracking_cost is None:
        ax.plot(np.zeros(retargeted_motion.num_frames()), label="no cost diagnostics")
    ax.set_title("IK diagnostics")
    ax.set_xlabel("frame")
    ax.set_ylabel("cost")
    ax.legend(loc="best")
    return _save(fig, output_path)


def plot_joint_positions(retargeted_motion: RetargetedMotion, output_path: Path | str) -> Path:
    retargeted_motion.validate()
    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    for idx, name in enumerate(retargeted_motion.joint_names):
        ax.plot(retargeted_motion.joint_pos[:, idx], linewidth=0.8, label=name)
    ax.set_title("Joint positions")
    ax.set_xlabel("frame")
    ax.set_ylabel("rad")
    _legend_if_small(ax, retargeted_motion.joint_names)
    return _save(fig, output_path)


def plot_joint_velocities(retargeted_motion: RetargetedMotion, output_path: Path | str) -> Path:
    retargeted_motion.validate()
    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    for idx, name in enumerate(retargeted_motion.joint_names):
        ax.plot(retargeted_motion.joint_vel[:, idx], linewidth=0.8, label=name)
    ax.set_title("Joint velocities")
    ax.set_xlabel("frame")
    ax.set_ylabel("rad/s")
    _legend_if_small(ax, retargeted_motion.joint_names)
    return _save(fig, output_path)


def plot_joint_limit_margin(retargeted_motion: RetargetedMotion, robot_spec: RobotSpec, output_path: Path | str) -> Path:
    retargeted_motion.validate()
    if retargeted_motion.joint_names != robot_spec.actuated_joints:
        raise ValueError("retargeted_motion joint_names must match robot_spec actuated_joints.")
    lower_margin = retargeted_motion.joint_pos - robot_spec.joint_lower_rad.reshape(1, -1)
    upper_margin = robot_spec.joint_upper_rad.reshape(1, -1) - retargeted_motion.joint_pos
    margin = np.minimum(lower_margin, upper_margin)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    for idx, name in enumerate(retargeted_motion.joint_names):
        ax.plot(margin[:, idx], linewidth=0.8, label=name)
    ax.axhline(0.0, color="red", linewidth=1.0)
    ax.set_title("Joint limit margin")
    ax.set_xlabel("frame")
    ax.set_ylabel("rad to nearest limit")
    _legend_if_small(ax, retargeted_motion.joint_names)
    return _save(fig, output_path)


def plot_root_height(retargeted_motion: RetargetedMotion, output_path: Path | str) -> Path:
    retargeted_motion.validate()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    ax.plot(retargeted_motion.root_pos_w[:, 2])
    ax.set_title("Root height")
    ax.set_xlabel("frame")
    ax.set_ylabel("z m")
    return _save(fig, output_path)


def plot_frame_success(retargeted_motion: RetargetedMotion, output_path: Path | str) -> Path:
    retargeted_motion.validate()
    fig, ax = plt.subplots(figsize=(10, 3), dpi=120)
    ax.step(np.arange(retargeted_motion.num_frames()), retargeted_motion.success.astype(int), where="mid")
    ax.set_title("Frame success")
    ax.set_xlabel("frame")
    ax.set_ylabel("success")
    ax.set_ylim(-0.1, 1.1)
    return _save(fig, output_path)


def _require_contact(preprocess_result: PreprocessResult) -> None:
    if preprocess_result.contact is None:
        raise ValueError("preprocess_result.contact is required for this plot.")


def _save(fig, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _legend_if_small(ax, names: list[str]) -> None:
    if len(names) <= 12:
        ax.legend(loc="best")


def _diagnostic_series(retargeted_motion: RetargetedMotion, path: tuple[str, ...]) -> np.ndarray | None:
    values: list[float] = []
    found = False
    for diagnostics in retargeted_motion.diagnostics:
        cursor = diagnostics
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if cursor is None:
            values.append(np.nan)
        else:
            found = True
            values.append(float(cursor) if cursor is not None else np.nan)
    return np.asarray(values, dtype=np.float64) if found else None
