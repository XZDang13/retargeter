"""BVH reader matching the server's (Sources/MotionIO/BVH.swift): centimetres -> metres, Y-up / +Z forward ->
the canonical Z-up / +X forward frame, channels composed left to right (R = R1 R2 R3), quaternions w x y z."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Skeleton:
    name: str
    joint_names: list
    parents: np.ndarray       # (J,) int32, -1 for the root
    rest_offsets: np.ndarray  # (J, 3) float32, metres, canonical frame


@dataclass
class Clip:
    skeleton: Skeleton
    fps: float
    joint_pos: np.ndarray     # (F, J, 3) float32, local translations (rest offsets unless the joint has position channels)
    joint_quat: np.ndarray    # (F, J, 4) float32, local, w x y z

    @property
    def frames(self) -> int:
        return len(self.joint_pos)


def _axis_quat(axis: int, angle: np.ndarray) -> np.ndarray:
    q = np.zeros(angle.shape + (4,))
    q[..., 0] = np.cos(angle / 2)
    q[..., 1 + axis] = np.sin(angle / 2)
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _basis(p: np.ndarray) -> np.ndarray:
    """(x, y, z) -> (z, x, y): Y-up / +Z forward to Z-up / +X forward (a proper rotation)."""
    return p[..., [2, 0, 1]]


def load_bvh(path: str, unit_scale: float = 0.01, y_up: bool = True, skeleton_name: str = "soma77") -> Clip:
    tokens = open(path).read().split()
    at = 0
    names, parents, offsets, channels = [], [], [], []
    stack = []

    def take():
        nonlocal at
        at += 1
        return tokens[at - 1]

    while True:
        word = take()
        if word in ("ROOT", "JOINT"):
            names.append(take())
            parents.append(stack[-1] if stack else -1)
            stack.append(len(names) - 1)
        elif word == "End":
            take()  # Site
            depth = 0
            while True:
                w = take()
                depth += w == "{"
                depth -= w == "}"
                if depth == 0:
                    break
        elif word == "OFFSET":
            offsets.append([float(take()) for _ in range(3)])
        elif word == "CHANNELS":
            channels.append([take() for _ in range(int(take()))])
        elif word == "}":
            stack.pop()
        elif word == "MOTION":
            break
    assert take() == "Frames:"
    frames = int(take())
    assert take() == "Frame" and take() == "Time:"
    fps = 1.0 / float(take())
    values = np.array(tokens[at:at + frames * sum(len(c) for c in channels)], dtype=np.float64).reshape(frames, -1)

    J = len(names)
    offsets = np.array(offsets) * unit_scale
    quat = np.tile(np.array([1.0, 0, 0, 0]), (frames, J, 1))
    pos = np.tile(offsets, (frames, 1, 1))
    column = 0
    for j, chans in enumerate(channels):
        has_position = False
        for c in chans:
            v = values[:, column]
            column += 1
            if c.endswith("position"):
                if not has_position:
                    pos[:, j] = 0
                    has_position = True
                pos[:, j, "XYZ".index(c[0])] = v * unit_scale
            else:
                quat[:, j] = quat_mul(quat[:, j], _axis_quat("XYZ".index(c[0]), np.radians(v)))
    if y_up:
        offsets, pos = _basis(offsets), _basis(pos)
        quat = np.concatenate([quat[..., :1], _basis(quat[..., 1:])], axis=-1)
    skeleton = Skeleton(skeleton_name, names, np.array(parents, np.int32), offsets.astype(np.float32))
    return Clip(skeleton, fps, pos.astype(np.float32), quat.astype(np.float32))
