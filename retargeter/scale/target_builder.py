from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from retargeter.preprocess import CanonicalHumanMotion, FootContactResult

from .human_to_robot_scaler import HumanToRobotScaler
from .ik_targets import BodyIKTarget, IKTargetSet


class Stage1TargetBuilder:
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
        stage_name: Literal["stage1a", "stage1b"],
        contact_result: FootContactResult | None = None,
    ) -> IKTargetSet:
        if stage_name not in {"stage1a", "stage1b"}:
            raise ValueError(f"stage_name must be 'stage1a' or 'stage1b', got {stage_name!r}.")
        if frame_idx < 0 or frame_idx >= motion.num_frames():
            raise IndexError(f"frame_idx {frame_idx} is outside motion length {motion.num_frames()}.")

        scaled_motion = self.scaler.scale_motion(motion)
        stage_weights = self.target_config[stage_name]
        targets: list[BodyIKTarget] = []
        modulation = self.target_config.get("contact_weight_modulation", {})

        for semantic_name, weights in stage_weights.items():
            if semantic_name not in self.scaler.body_map:
                raise ValueError(f"No body_map entry for target semantic name {semantic_name!r}.")
            map_entry = self.scaler.body_map[semantic_name]
            human_body_name = map_entry["human"]
            body_index = scaled_motion.get_body_index(human_body_name)
            pos_weight = float(weights.get("pos_weight", 0.0))
            rot_weight = float(weights.get("rot_weight", 0.0))
            confidence = 1.0

            if contact_result is not None and modulation.get("enabled", False):
                region_config = modulation.get("regions", {})
                for region, entry in region_config.items():
                    if entry.get("target") != semantic_name:
                        continue
                    score = _contact_score(contact_result, region, frame_idx)
                    confidence = score
                    pos_weight += score * float(entry.get("extra_pos_weight", 0.0))
                    rot_weight += score * float(entry.get("extra_rot_weight", 0.0))

            target = BodyIKTarget(
                semantic_name=semantic_name,
                human_body_name=human_body_name,
                robot_body_name=map_entry["robot"],
                target_pos_w=None if pos_weight <= 0.0 else scaled_motion.body_pos_w[frame_idx, body_index].copy(),
                target_quat_xyzw=None if rot_weight <= 0.0 else scaled_motion.body_quat_xyzw[frame_idx, body_index].copy(),
                pos_weight=pos_weight,
                rot_weight=rot_weight,
                confidence=confidence,
                metadata={
                    "frame_idx": frame_idx,
                    "robot": self.scaler.robot,
                    "target_config_path": str(self.target_config_path),
                },
            )
            targets.append(target)

        target_set = IKTargetSet(
            stage_name=stage_name,
            targets=targets,
            metadata={
                "frame_idx": frame_idx,
                "robot": self.scaler.robot,
                "scaler_config_path": str(self.scaler.scaler_config_path),
                "target_config_path": str(self.target_config_path),
                "required_robot_body_names": self.required_robot_body_names(stage_name),
            },
        )
        target_set.validate()
        return target_set

    def required_robot_body_names(self, stage_name: Literal["stage1a", "stage1b"] | None = None) -> list[str]:
        if stage_name is None:
            stage_names = ("stage1a", "stage1b")
        else:
            if stage_name not in {"stage1a", "stage1b"}:
                raise ValueError(f"stage_name must be 'stage1a' or 'stage1b', got {stage_name!r}.")
            stage_names = (stage_name,)

        names: list[str] = []
        for active_stage in stage_names:
            for semantic_name in self.target_config[active_stage]:
                map_entry = self.scaler.body_map.get(semantic_name)
                if map_entry is None:
                    continue
                robot_name = map_entry["robot"]
                if robot_name not in names:
                    names.append(robot_name)
        return names


def _contact_score(contact_result: FootContactResult, region: str, frame_idx: int) -> float:
    if region not in contact_result.contact_score:
        return 0.0
    scores = np.asarray(contact_result.contact_score[region], dtype=np.float64)
    if frame_idx < 0 or frame_idx >= scores.shape[0]:
        return 0.0
    if not np.isfinite(scores[frame_idx]):
        return 0.0
    return float(np.clip(scores[frame_idx], 0.0, 1.0))


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
    missing = [stage for stage in ("stage1a", "stage1b") if stage not in config]
    if missing:
        raise ValueError(f"Target config {path} is missing required stages: {missing}.")
    for stage_name in ("stage1a", "stage1b"):
        stage = config[stage_name]
        if not isinstance(stage, dict) or not stage:
            raise ValueError(f"{stage_name} in {path} must be a non-empty mapping.")
        for semantic_name, weights in stage.items():
            if "pos_weight" not in weights or "rot_weight" not in weights:
                raise ValueError(f"{stage_name}.{semantic_name} must define pos_weight and rot_weight.")
