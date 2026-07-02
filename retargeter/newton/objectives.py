from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from retargeter.preprocess.lowpass import normalize_quat_xyzw
from retargeter.scale import IKTargetSet

from .robot_spec import RobotSpec


ObjectiveKind = Literal["position", "rotation", "joint_limit", "posture", "smooth", "damping", "self_collision"]
SelfCollisionShape = Literal["sphere", "capsule"]


@dataclass(frozen=True)
class SelfCollisionPairSpec:
    name: str
    point_body: str
    obstacle_body: str
    obstacle_shape: SelfCollisionShape
    obstacle_radius_m: float
    margin_m: float
    point_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obstacle_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obstacle_point_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obstacle_point_b: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self, robot_spec: RobotSpec) -> None:
        if not self.name:
            raise ValueError("self_collision pair name must be non-empty.")
        if self.obstacle_shape not in {"sphere", "capsule"}:
            raise ValueError(f"self_collision pair {self.name!r} has unsupported obstacle shape {self.obstacle_shape!r}.")
        robot_spec.require_body_names([self.point_body, self.obstacle_body])
        if self.obstacle_radius_m <= 0.0 or not np.isfinite(self.obstacle_radius_m):
            raise ValueError(f"self_collision pair {self.name!r} obstacle_radius_m must be positive and finite.")
        if self.margin_m <= 0.0 or not np.isfinite(self.margin_m):
            raise ValueError(f"self_collision pair {self.name!r} margin_m must be positive and finite.")
        for field_name in ("point_local_pos", "obstacle_center", "obstacle_point_a", "obstacle_point_b"):
            value = np.asarray(getattr(self, field_name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"self_collision pair {self.name!r} {field_name} must be a finite vec3.")
        if self.obstacle_shape == "capsule":
            a = np.asarray(self.obstacle_point_a, dtype=np.float64)
            b = np.asarray(self.obstacle_point_b, dtype=np.float64)
            if np.linalg.norm(a - b) < 1e-6:
                raise ValueError(f"self_collision pair {self.name!r} capsule endpoints must be distinct.")

    def layout_key(self) -> tuple:
        return (
            self.name,
            self.point_body,
            _round_vec3(self.point_local_pos),
            self.obstacle_body,
            self.obstacle_shape,
            _round_vec3(self.obstacle_center),
            _round_vec3(self.obstacle_point_a),
            _round_vec3(self.obstacle_point_b),
            round(float(self.obstacle_radius_m), 10),
            round(float(self.margin_m), 10),
        )


@dataclass(frozen=True)
class IKObjectiveDescriptor:
    kind: ObjectiveKind
    weight: float
    body_name: str | None = None
    semantic_name: str | None = None
    target: np.ndarray | None = None
    body_local_pos: np.ndarray | None = None
    confidence: float = 1.0
    self_collision_pairs: tuple[SelfCollisionPairSpec, ...] | None = None

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
            if self.body_local_pos is not None:
                local_pos = np.asarray(self.body_local_pos, dtype=np.float64)
                if local_pos.shape != (3,):
                    raise ValueError(
                        f"position local point for {self.body_name!r} must have shape [3], got {local_pos.shape}."
                    )
                if not np.all(np.isfinite(local_pos)):
                    raise ValueError(f"position local point for {self.body_name!r} contains NaN or inf values.")

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

        if self.kind == "self_collision":
            if self.weight <= 0.0:
                raise ValueError("self_collision objective weight must be positive.")
            if not self.self_collision_pairs:
                raise ValueError("self_collision objective must define at least one pair.")
            names = set()
            for pair in self.self_collision_pairs:
                pair.validate(robot_spec)
                if pair.name in names:
                    raise ValueError(f"duplicate self_collision pair name {pair.name!r}.")
                names.add(pair.name)


def build_target_objectives(target_set: IKTargetSet, robot_spec: RobotSpec) -> list[IKObjectiveDescriptor]:
    target_set.validate()
    descriptors: list[IKObjectiveDescriptor] = []

    for target in target_set.targets:
        robot_spec.require_body_names([target.robot_body_name])
        confidence = float(target.confidence)
        can_activate_position = bool(target.metadata.get("can_activate_position", False))
        if target.target_pos_w is not None and (target.pos_weight > 0.0 or can_activate_position):
            descriptors.append(
                IKObjectiveDescriptor(
                    kind="position",
                    weight=float(target.pos_weight),
                    body_name=target.robot_body_name,
                    semantic_name=target.semantic_name,
                    target=np.asarray(target.target_pos_w, dtype=np.float64).copy(),
                    body_local_pos=None
                    if target.robot_local_pos is None
                    else np.asarray(target.robot_local_pos, dtype=np.float64).copy(),
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


def build_self_collision_objectives(robot_spec: RobotSpec, config: dict | None) -> list[IKObjectiveDescriptor]:
    section = dict(config or {})
    if not bool(section.get("enabled", False)):
        return []
    weight = _finite_positive_float(section.get("weight", 4.0), "self_collision.weight")
    default_margin = _finite_positive_float(section.get("margin_m", 0.08), "self_collision.margin_m")
    raw_pairs = section.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("self_collision.pairs must be a non-empty list when self-collision is enabled.")

    pairs = tuple(_parse_self_collision_pair(raw_pair, default_margin, index) for index, raw_pair in enumerate(raw_pairs))
    descriptor = IKObjectiveDescriptor(kind="self_collision", weight=weight, self_collision_pairs=pairs)
    descriptor.validate(robot_spec)
    return [descriptor]


def summarize_self_collision_clearance(
    body_names: list[str],
    body_pos_w: np.ndarray,
    body_quat_xyzw: np.ndarray,
    pairs: tuple[SelfCollisionPairSpec, ...],
) -> dict:
    if not pairs:
        return {"enabled": False, "pair_count": 0}
    name_to_index = {name: idx for idx, name in enumerate(body_names)}
    pos = np.asarray(body_pos_w, dtype=np.float64)
    quat = np.asarray(body_quat_xyzw, dtype=np.float64)
    clearances: list[float] = []
    active = 0
    worst_pair = None
    worst_clearance = np.inf
    worst_margin = 0.0
    for pair in pairs:
        point_idx = name_to_index[pair.point_body]
        obstacle_idx = name_to_index[pair.obstacle_body]
        point = pos[point_idx] + _quat_rotate_xyzw(quat[point_idx], np.asarray(pair.point_local_pos, dtype=np.float64))
        obstacle_pos = pos[obstacle_idx]
        obstacle_quat = quat[obstacle_idx]
        if pair.obstacle_shape == "sphere":
            center = obstacle_pos + _quat_rotate_xyzw(obstacle_quat, np.asarray(pair.obstacle_center, dtype=np.float64))
        else:
            a = obstacle_pos + _quat_rotate_xyzw(obstacle_quat, np.asarray(pair.obstacle_point_a, dtype=np.float64))
            b = obstacle_pos + _quat_rotate_xyzw(obstacle_quat, np.asarray(pair.obstacle_point_b, dtype=np.float64))
            center = _closest_point_on_segment(point, a, b)
        clearance = float(np.linalg.norm(point - center) - float(pair.obstacle_radius_m))
        clearances.append(clearance)
        if clearance < pair.margin_m:
            active += 1
        if clearance < worst_clearance:
            worst_clearance = clearance
            worst_pair = pair.name
            worst_margin = float(pair.margin_m)
    violation = max(0.0, worst_margin - worst_clearance) if np.isfinite(worst_clearance) else 0.0
    return {
        "enabled": True,
        "pair_count": int(len(pairs)),
        "active_pair_count": int(active),
        "min_clearance_m": float(np.min(clearances)) if clearances else None,
        "worst_pair": worst_pair,
        "worst_pair_clearance_m": None if not np.isfinite(worst_clearance) else float(worst_clearance),
        "worst_pair_margin_m": float(worst_margin),
        "worst_pair_violation_m": float(violation),
    }


def _parse_self_collision_pair(raw_pair, default_margin: float, index: int) -> SelfCollisionPairSpec:
    if not isinstance(raw_pair, dict):
        raise TypeError(f"self_collision.pairs[{index}] must be a mapping.")
    obstacle = raw_pair.get("obstacle")
    if not isinstance(obstacle, dict):
        raise ValueError(f"self_collision.pairs[{index}].obstacle must be a mapping.")
    shape = str(obstacle.get("shape", "sphere")).lower()
    if shape not in {"sphere", "capsule"}:
        raise ValueError(f"self_collision.pairs[{index}].obstacle.shape must be 'sphere' or 'capsule'.")
    name = str(raw_pair.get("name", f"pair_{index}"))
    point_body = str(raw_pair.get("point_body", ""))
    obstacle_body = str(obstacle.get("body", ""))
    if not point_body:
        raise ValueError(f"self_collision pair {name!r} must define point_body.")
    if not obstacle_body:
        raise ValueError(f"self_collision pair {name!r} obstacle must define body.")
    return SelfCollisionPairSpec(
        name=name,
        point_body=point_body,
        point_local_pos=_vec3_tuple(raw_pair.get("point_local_pos", [0.0, 0.0, 0.0]), f"self_collision pair {name!r} point_local_pos"),
        obstacle_body=obstacle_body,
        obstacle_shape=shape,
        obstacle_radius_m=_finite_positive_float(obstacle.get("radius_m"), f"self_collision pair {name!r} obstacle.radius_m"),
        margin_m=_finite_positive_float(raw_pair.get("margin_m", default_margin), f"self_collision pair {name!r} margin_m"),
        obstacle_center=_vec3_tuple(obstacle.get("center", [0.0, 0.0, 0.0]), f"self_collision pair {name!r} obstacle.center"),
        obstacle_point_a=_vec3_tuple(obstacle.get("point_a", [0.0, 0.0, 0.0]), f"self_collision pair {name!r} obstacle.point_a"),
        obstacle_point_b=_vec3_tuple(obstacle.get("point_b", [0.0, 0.0, 0.0]), f"self_collision pair {name!r} obstacle.point_b"),
    )


def _finite_positive_float(value, name: str) -> float:
    result = float(value)
    if result <= 0.0 or not np.isfinite(result):
        raise ValueError(f"{name} must be positive and finite.")
    return result


def _vec3_tuple(value, name: str) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite vec3.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _round_vec3(value: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(round(float(item), 10) for item in value)


def _quat_rotate_xyzw(quat_xyzw: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = normalize_quat_xyzw(np.asarray(quat_xyzw, dtype=np.float64))
    v = np.asarray(vector, dtype=np.float64)
    q_xyz = q[:3]
    q_w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _closest_point_on_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return a.copy()
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return a + t * ab
