from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from retargeter.preprocess import CanonicalHumanMotion, FootContactResult
from retargeter.preprocess.lowpass import normalize_quat_xyzw

from .human_to_robot_scaler import HumanToRobotScaler
from .ik_targets import BodyIKTarget, IKTargetSet


IK_PASS_NAMES = ("coarse_alignment", "full_body_tracking")
IKPassName = Literal["coarse_alignment", "full_body_tracking"]


class IKTargetBuilder:
    def __init__(
        self,
        scaler_config_path: Path | str,
        target_config_path: Path | str,
    ):
        self.scaler = HumanToRobotScaler(scaler_config_path)
        self.target_config_path = Path(target_config_path)
        self.target_config = _load_yaml(self.target_config_path)
        _validate_target_config(self.target_config, self.target_config_path)

    def build(
        self,
        motion: CanonicalHumanMotion,
        frame_idx: int,
        pass_name: IKPassName,
        contact_result: FootContactResult | None = None,
    ) -> IKTargetSet:
        pass_name = _canonical_pass_name(pass_name)
        if frame_idx < 0 or frame_idx >= motion.num_frames():
            raise IndexError(f"frame_idx {frame_idx} is outside motion length {motion.num_frames()}.")

        scaled_motion = self.scaler.scale_motion(motion)
        pass_weights = self.target_config[pass_name]
        targets: list[BodyIKTarget] = []
        modulation = self.target_config.get("contact_weight_modulation", {})

        for semantic_name, weights in pass_weights.items():
            if semantic_name not in self.scaler.body_map:
                raise ValueError(f"No body_map entry for target semantic name {semantic_name!r}.")
            map_entry = self.scaler.body_map[semantic_name]
            human_body_name = map_entry["human"]
            robot_local_pos = _optional_vec3(map_entry.get("robot_local_pos"), field_name=f"{semantic_name}.robot_local_pos")
            body_index = scaled_motion.get_body_index(human_body_name)
            pos_weight = float(weights.get("pos_weight", 0.0))
            rot_weight = float(weights.get("rot_weight", 0.0))
            confidence = 1.0
            position_source_metadata: dict[str, Any] = {"type": "body", "body": human_body_name}
            rotation_source_metadata: dict[str, Any] = {"type": "body", "body": human_body_name}

            if contact_result is not None and modulation.get("enabled", False):
                region_config = modulation.get("regions", {})
                for region, entry in region_config.items():
                    if entry.get("target") != semantic_name:
                        continue
                    score = _contact_score(contact_result, str(entry.get("source_region", region)), frame_idx)
                    if score > 0.0:
                        confidence = score
                        pos_weight += score * float(entry.get("extra_pos_weight", 0.0))
                        rot_weight += score * float(entry.get("extra_rot_weight", 0.0))

            target_pos_w = None
            if pos_weight > 0.0:
                target_pos_w, position_source_metadata = self._target_position_w(
                    motion,
                    scaled_motion,
                    frame_idx,
                    human_body_name=human_body_name,
                    map_entry=map_entry,
                )

            target_quat_xyzw = None
            if rot_weight > 0.0:
                target_quat_xyzw, rotation_source_metadata = self._target_quat_xyzw(
                    motion,
                    scaled_motion,
                    frame_idx,
                    human_body_name=human_body_name,
                    map_entry=map_entry,
                )

            target = BodyIKTarget(
                semantic_name=semantic_name,
                human_body_name=human_body_name,
                robot_body_name=map_entry["robot"],
                target_pos_w=target_pos_w,
                target_quat_xyzw=target_quat_xyzw,
                pos_weight=pos_weight,
                rot_weight=rot_weight,
                robot_local_pos=robot_local_pos,
                confidence=confidence,
                metadata={
                    "frame_idx": frame_idx,
                    "robot": self.scaler.robot,
                    "target_config_path": str(self.target_config_path),
                    "position_source": position_source_metadata,
                    "rotation_source": rotation_source_metadata,
                },
            )
            targets.append(target)

        target_set = IKTargetSet(
            pass_name=pass_name,
            targets=targets,
            metadata={
                "frame_idx": frame_idx,
                "robot": self.scaler.robot,
                "scaler_config_path": str(self.scaler.scaler_config_path),
                "target_config_path": str(self.target_config_path),
                "required_robot_body_names": self.required_robot_body_names(pass_name),
            },
        )
        target_set.validate()
        return target_set

    def required_robot_body_names(self, pass_name: IKPassName | None = None) -> list[str]:
        if pass_name is None:
            pass_names = IK_PASS_NAMES
        else:
            pass_names = (_canonical_pass_name(pass_name),)

        names: list[str] = []
        for active_pass in pass_names:
            for semantic_name in self.target_config[active_pass]:
                map_entry = self.scaler.body_map.get(semantic_name)
                if map_entry is None:
                    continue
                robot_name = map_entry["robot"]
                if robot_name not in names:
                    names.append(robot_name)
        return names

    def _target_position_w(
        self,
        motion: CanonicalHumanMotion,
        scaled_motion: CanonicalHumanMotion,
        frame_idx: int,
        *,
        human_body_name: str,
        map_entry: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        source_config = map_entry.get("human_position_source")
        if source_config is None:
            body_index = scaled_motion.get_body_index(human_body_name)
            return scaled_motion.body_pos_w[frame_idx, body_index].copy(), {"type": "body", "body": human_body_name}
        if not isinstance(source_config, dict):
            raise ValueError(f"human_position_source for {human_body_name!r} must be a mapping.")

        source_type = str(source_config.get("type", "body"))
        if source_type == "body":
            body = str(source_config.get("body", human_body_name))
            body_index = scaled_motion.get_body_index(body)
            return scaled_motion.body_pos_w[frame_idx, body_index].copy(), {"type": "body", "body": body}
        if source_type != "foot_sole":
            raise ValueError(f"Unsupported human_position_source type {source_type!r} for {human_body_name!r}.")

        points_w, metadata = _foot_sole_points_w(motion, source_config)
        if points_w is None:
            body_index = scaled_motion.get_body_index(human_body_name)
            fallback = {"type": "body", "body": human_body_name, "fallback_reason": metadata.get("fallback_reason")}
            return scaled_motion.body_pos_w[frame_idx, body_index].copy(), fallback

        scale_body = str(source_config.get("scale_body", human_body_name))
        scaled_points = _scale_points_like_body(self.scaler, motion, points_w, scale_body)
        return scaled_points[frame_idx].copy(), metadata

    def _target_quat_xyzw(
        self,
        motion: CanonicalHumanMotion,
        scaled_motion: CanonicalHumanMotion,
        frame_idx: int,
        *,
        human_body_name: str,
        map_entry: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        body_index = scaled_motion.get_body_index(human_body_name)
        source_config = map_entry.get("human_rotation_source")
        if source_config is None:
            return scaled_motion.body_quat_xyzw[frame_idx, body_index].copy(), {"type": "body", "body": human_body_name}
        if not isinstance(source_config, dict):
            raise ValueError(f"human_rotation_source for {human_body_name!r} must be a mapping.")

        source_type = str(source_config.get("type", "body"))
        if source_type == "body":
            body = str(source_config.get("body", human_body_name))
            body_index = scaled_motion.get_body_index(body)
            return scaled_motion.body_quat_xyzw[frame_idx, body_index].copy(), {"type": "body", "body": body}
        if source_type not in {"foot_sole_frame", "foot_sole_yaw"}:
            raise ValueError(f"Unsupported human_rotation_source type {source_type!r} for {human_body_name!r}.")

        toe_w, heel_w, metadata = _foot_sole_toe_heel_points_w(motion, source_config)
        if toe_w is None or heel_w is None:
            fallback = {"type": "body", "body": human_body_name, "fallback_reason": metadata.get("fallback_reason")}
            return scaled_motion.body_quat_xyzw[frame_idx, body_index].copy(), fallback

        forward = toe_w[frame_idx] - heel_w[frame_idx]
        if not np.all(np.isfinite(forward)) or np.linalg.norm(forward) < 1e-8:
            fallback = {"type": "body", "body": human_body_name, "fallback_reason": "degenerate_foot_heading"}
            return scaled_motion.body_quat_xyzw[frame_idx, body_index].copy(), fallback

        yaw_offset_deg = float(source_config.get("yaw_offset_deg", 0.0))
        if not np.isfinite(yaw_offset_deg):
            raise ValueError("foot sole rotation yaw_offset_deg must be finite.")
        if source_type == "foot_sole_yaw":
            forward_xy = forward[:2]
            if np.linalg.norm(forward_xy) < 1e-8:
                fallback = {"type": "body", "body": human_body_name, "fallback_reason": "degenerate_foot_heading"}
                return scaled_motion.body_quat_xyzw[frame_idx, body_index].copy(), fallback
            yaw = float(np.arctan2(forward_xy[1], forward_xy[0]) + np.deg2rad(yaw_offset_deg))
            quat = _yaw_quat_xyzw(yaw)
        else:
            quat = _foot_sole_frame_quat_xyzw(forward, yaw_offset_deg=yaw_offset_deg)
            if quat is None:
                fallback = {"type": "body", "body": human_body_name, "fallback_reason": "degenerate_foot_frame"}
                return scaled_motion.body_quat_xyzw[frame_idx, body_index].copy(), fallback
        return quat, {
            "type": source_type,
            "side": metadata["side"],
            "lower_fraction": metadata["lower_fraction"],
            "yaw_offset_deg": yaw_offset_deg,
        }


def _contact_score(contact_result: FootContactResult, region: str, frame_idx: int) -> float:
    if region not in contact_result.contact_score:
        return 0.0
    scores = np.asarray(contact_result.contact_score[region], dtype=np.float64)
    if frame_idx < 0 or frame_idx >= scores.shape[0]:
        return 0.0
    if not np.isfinite(scores[frame_idx]):
        return 0.0
    return float(np.clip(scores[frame_idx], 0.0, 1.0))


def _optional_vec3(value, *, field_name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{field_name} must have shape [3], got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field_name} contains NaN or inf values.")
    return arr.copy()


def _foot_sole_points_w(
    motion: CanonicalHumanMotion,
    source_config: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    toe, heel, metadata = _foot_sole_toe_heel_points_w(motion, source_config)
    if toe is None or heel is None:
        return None, metadata

    along = _foot_sole_along(source_config)
    height_offset = float(source_config.get("height_offset", 0.0))
    if not np.isfinite(height_offset):
        raise ValueError("foot_sole height_offset must be finite.")

    points = heel + float(along) * (toe - heel)
    if height_offset != 0.0:
        points = points.copy()
        points[:, 2] += height_offset
    metadata = {
        **metadata,
        "type": "foot_sole",
        "along": float(along),
        "height_offset": height_offset,
        "scale_body": str(source_config.get("scale_body", f"{metadata['side']}_foot")),
    }
    return points, metadata


def _foot_sole_toe_heel_points_w(
    motion: CanonicalHumanMotion,
    source_config: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    side = str(source_config.get("side", ""))
    if side not in {"left", "right"}:
        raise ValueError("foot_sole human_position_source side must be 'left' or 'right'.")
    if motion.vertices_w is None:
        return None, None, {"fallback_reason": "missing_vertices"}

    vertex_config = motion.metadata.get("foot_vertex_config", {})
    if not isinstance(vertex_config, dict):
        return None, None, {"fallback_reason": "missing_foot_vertex_config"}
    vertex_count = int(motion.vertices_w.shape[1])
    toe_indices = _valid_vertex_indices(vertex_config.get(f"{side}_toe_indices", []), vertex_count)
    heel_indices = _valid_vertex_indices(vertex_config.get(f"{side}_heel_indices", []), vertex_count)
    if toe_indices.size == 0 or heel_indices.size == 0:
        return None, None, {"fallback_reason": "missing_foot_vertices"}

    lower_fraction = float(source_config.get("lower_fraction", 0.35))
    if lower_fraction <= 0.0 or not np.isfinite(lower_fraction):
        raise ValueError("foot_sole lower_fraction must be positive and finite.")
    lower_fraction = min(lower_fraction, 1.0)

    toe = _lower_vertex_region_points(motion.vertices_w, toe_indices, lower_fraction)
    heel = _lower_vertex_region_points(motion.vertices_w, heel_indices, lower_fraction)
    return toe, heel, {
        "side": side,
        "lower_fraction": lower_fraction,
    }


def _foot_sole_along(source_config: dict[str, Any]) -> float:
    if "along" in source_config:
        along = float(source_config["along"])
    else:
        point = str(source_config.get("point", "toe")).lower()
        point_to_along = {
            "heel": 0.0,
            "back": 0.0,
            "mid": 0.5,
            "middle": 0.5,
            "center": 0.5,
            "toe": 1.0,
            "front": 1.0,
        }
        if point not in point_to_along:
            raise ValueError(f"Unsupported foot_sole point {point!r}.")
        along = point_to_along[point]
    if not np.isfinite(along):
        raise ValueError("foot_sole along must be finite.")
    return float(along)


def _lower_vertex_region_points(vertices_w: np.ndarray, indices: np.ndarray, lower_fraction: float) -> np.ndarray:
    points = np.asarray(vertices_w[:, indices, :], dtype=np.float64)
    count = max(1, int(np.ceil(points.shape[1] * lower_fraction)))
    order = np.argsort(points[:, :, 2], axis=1)[:, :count]
    lower_points = np.take_along_axis(points, order[:, :, None], axis=1)
    return lower_points.mean(axis=1)


def _valid_vertex_indices(indices, vertex_count: int) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64)
    if arr.size == 0:
        return arr
    return arr[(arr >= 0) & (arr < vertex_count)]


def _yaw_quat_xyzw(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return normalize_quat_xyzw(np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float64))


def _foot_sole_frame_quat_xyzw(forward: np.ndarray, *, yaw_offset_deg: float = 0.0) -> np.ndarray | None:
    x_axis = np.asarray(forward, dtype=np.float64)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8 or not np.isfinite(x_norm):
        return None
    x_axis = x_axis / x_norm

    if yaw_offset_deg != 0.0:
        yaw_offset = np.deg2rad(float(yaw_offset_deg))
        x_axis = rotate_z_xyzw(_yaw_quat_xyzw(yaw_offset), x_axis)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(world_up, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-8 or not np.isfinite(y_norm):
        return None
    y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8 or not np.isfinite(z_norm):
        return None
    z_axis = z_axis / z_norm

    matrix = np.column_stack([x_axis, y_axis, z_axis])
    return _matrix_to_quat_xyzw(matrix)


def rotate_z_xyzw(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = normalize_quat_xyzw(np.asarray(quat, dtype=np.float64))
    v = np.asarray(vector, dtype=np.float64)
    q_xyz = q[:3]
    uv = np.cross(q_xyz, v)
    uuv = np.cross(q_xyz, uv)
    return v + 2.0 * (q[3] * uv + uuv)


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"rotation matrix must have shape [3, 3], got {m.shape}.")
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = normalize_quat_xyzw(np.array([qx, qy, qz, qw], dtype=np.float64))
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def _scale_points_like_body(
    scaler: HumanToRobotScaler,
    motion: CanonicalHumanMotion,
    points_w: np.ndarray,
    scale_body_name: str,
) -> np.ndarray:
    points = np.asarray(points_w, dtype=np.float64)
    expected = (motion.num_frames(), 3)
    if points.shape != expected:
        raise ValueError(f"position source points must have shape {expected}, got {points.shape}.")
    root_index = motion.get_body_index(scaler.human_root_name)
    root_pos = motion.body_pos_w[:, root_index, :]
    root_scale = float(scaler.scales.get(scaler.human_root_name, 1.0))
    point_scale = float(scaler.scales.get(scale_body_name, 1.0))
    return root_pos * root_scale + (points - root_pos) * point_scale


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load target configs.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Target config {path} must contain a YAML mapping.")
    return data


def _validate_target_config(config: dict[str, Any], path: Path) -> None:
    missing = [pass_name for pass_name in IK_PASS_NAMES if pass_name not in config]
    if missing:
        raise ValueError(f"Target config {path} is missing required IK passes: {missing}.")
    for pass_name in IK_PASS_NAMES:
        stage = config[pass_name]
        if not isinstance(stage, dict) or not stage:
            raise ValueError(f"{pass_name} in {path} must be a non-empty mapping.")
        for semantic_name, weights in stage.items():
            if "pos_weight" not in weights or "rot_weight" not in weights:
                raise ValueError(f"{pass_name}.{semantic_name} must define pos_weight and rot_weight.")


def _canonical_pass_name(pass_name: str) -> str:
    if pass_name not in IK_PASS_NAMES:
        raise ValueError(f"IK pass name must be one of {sorted(IK_PASS_NAMES)}, got {pass_name!r}.")
    return pass_name
