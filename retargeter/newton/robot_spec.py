from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ROBOT_CONFIG_SECTIONS = [
    "robot",
    "model",
    "root_body",
    "body_names",
    "actuated_joints",
    "joint_limits_deg",
    "velocity_limits_rad_s",
    "default_joint_pos",
]


@dataclass(frozen=True)
class RobotSpec:
    robot: str
    model_path: Path
    model_format: str
    floating_base: bool
    root_body: str
    body_names: list[str]
    actuated_joints: list[str]
    joint_lower_rad: np.ndarray
    joint_upper_rad: np.ndarray
    velocity_limits_rad_s: np.ndarray
    default_joint_pos: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RobotSpec":
        config_path = Path(path)
        config = _load_yaml(config_path)
        _validate_config(config, config_path)

        model = config["model"]
        model_path = _resolve_path(str(model["path"]), config_path)
        model_format = str(model["format"]).lower()
        floating_base = bool(model.get("floating_base", True))

        robot = str(config["robot"])
        root_body = str(config["root_body"])
        body_names = [str(name) for name in config["body_names"]]
        actuated_joints = [str(name) for name in config["actuated_joints"]]

        lower: list[float] = []
        upper: list[float] = []
        default: list[float] = []
        for joint_name in actuated_joints:
            limits = config["joint_limits_deg"][joint_name]
            lower.append(np.deg2rad(float(limits[0])))
            upper.append(np.deg2rad(float(limits[1])))
            default.append(float(config["default_joint_pos"].get(joint_name, 0.0)))

        velocity_config = config["velocity_limits_rad_s"]
        default_velocity = float(velocity_config.get("default", np.inf))
        joint_velocity = velocity_config.get("joints", {}) or {}
        velocity = [float(joint_velocity.get(joint_name, default_velocity)) for joint_name in actuated_joints]

        spec = cls(
            robot=robot,
            model_path=model_path,
            model_format=model_format,
            floating_base=floating_base,
            root_body=root_body,
            body_names=body_names,
            actuated_joints=actuated_joints,
            joint_lower_rad=np.asarray(lower, dtype=np.float64),
            joint_upper_rad=np.asarray(upper, dtype=np.float64),
            velocity_limits_rad_s=np.asarray(velocity, dtype=np.float64),
            default_joint_pos=np.asarray(default, dtype=np.float64),
            metadata={
                "config_path": str(config_path),
                "model_path": str(model_path),
                "model_format": model_format,
                "source_limits": "usd_degrees",
            },
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.robot:
            raise ValueError("robot must be non-empty.")
        if self.model_format not in {"usd", "mjcf", "urdf"}:
            raise ValueError(f"Unsupported model_format {self.model_format!r}.")
        if not self.root_body:
            raise ValueError("root_body must be non-empty.")
        if self.root_body not in self.body_names:
            raise ValueError(f"root_body {self.root_body!r} is not listed in body_names.")
        if len(self.body_names) != len(set(self.body_names)):
            raise ValueError("body_names must be unique.")
        if len(self.actuated_joints) != len(set(self.actuated_joints)):
            raise ValueError("actuated_joints must be unique.")
        if self.num_dofs == 0:
            raise ValueError("actuated_joints must be non-empty.")

        expected = (self.num_dofs,)
        for name, value in [
            ("joint_lower_rad", self.joint_lower_rad),
            ("joint_upper_rad", self.joint_upper_rad),
            ("velocity_limits_rad_s", self.velocity_limits_rad_s),
            ("default_joint_pos", self.default_joint_pos),
        ]:
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {arr.shape}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or inf values.")

        if np.any(self.joint_lower_rad >= self.joint_upper_rad):
            raise ValueError("Each joint lower limit must be less than its upper limit.")
        if np.any(self.velocity_limits_rad_s <= 0.0):
            raise ValueError("velocity_limits_rad_s must be positive.")
        if np.any(self.default_joint_pos < self.joint_lower_rad) or np.any(self.default_joint_pos > self.joint_upper_rad):
            raise ValueError("default_joint_pos must lie inside joint limits.")

    @property
    def num_dofs(self) -> int:
        return len(self.actuated_joints)

    @property
    def joint_name_to_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.actuated_joints)}

    @property
    def body_name_to_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.body_names)}

    def require_body_names(self, body_names: list[str]) -> None:
        missing = [name for name in body_names if name not in self.body_names]
        if missing:
            raise ValueError(f"Robot spec {self.robot!r} is missing required bodies: {missing}.")

    def has_body(self, body_name: str) -> bool:
        return body_name in self.body_name_to_index


def load_robot_spec(path: Path | str) -> RobotSpec:
    return RobotSpec.from_yaml(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Newton robot configs.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Robot config {path} must contain a YAML mapping.")
    return data


def _validate_config(config: dict[str, Any], path: Path) -> None:
    missing = [section for section in REQUIRED_ROBOT_CONFIG_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Robot config {path} is missing required sections: {missing}.")

    model = config["model"]
    if not isinstance(model, dict) or "path" not in model or "format" not in model:
        raise ValueError(f"Robot config {path} model must define path and format.")

    for section in ("body_names", "actuated_joints"):
        if not isinstance(config[section], list) or not config[section]:
            raise ValueError(f"Robot config {path} {section} must be a non-empty list.")

    joint_limits = config["joint_limits_deg"]
    if not isinstance(joint_limits, dict):
        raise ValueError(f"Robot config {path} joint_limits_deg must be a mapping.")
    for joint_name in config["actuated_joints"]:
        if joint_name not in joint_limits:
            raise ValueError(f"Robot config {path} has no joint_limits_deg entry for {joint_name!r}.")
        limits = np.asarray(joint_limits[joint_name], dtype=np.float64)
        if limits.shape != (2,):
            raise ValueError(f"joint_limits_deg.{joint_name} must have shape [2].")

    velocity_limits = config["velocity_limits_rad_s"]
    if not isinstance(velocity_limits, dict) or "default" not in velocity_limits:
        raise ValueError(f"Robot config {path} velocity_limits_rad_s must define default.")

    if not isinstance(config["default_joint_pos"], dict):
        raise ValueError(f"Robot config {path} default_joint_pos must be a mapping.")


def _resolve_path(raw_path: str, config_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return (config_path.parent / path).resolve()
