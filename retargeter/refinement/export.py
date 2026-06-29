from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .quality import RefinementQualityReport
from .refiner import RefinedMotion


def export_refined_motion(
    motion: RefinedMotion,
    output_path: Path | str,
    *,
    metadata_path: Path | str | None = None,
    quality_path: Path | str | None = None,
    quality_report: RefinementQualityReport | None = None,
) -> dict[str, Any]:
    motion.validate()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        fps=np.asarray(motion.fps, dtype=np.float64),
        robot=np.asarray(motion.robot),
        joint_names=np.asarray(motion.joint_names),
        root_pos_w=motion.root_pos_w,
        root_quat_xyzw=motion.root_quat_xyzw,
        joint_pos=motion.joint_pos,
        joint_vel=motion.joint_vel,
        body_names=np.asarray(motion.body_names),
        body_pos_w=motion.body_pos_w,
        body_quat_xyzw=motion.body_quat_xyzw,
        root_delta=motion.root_delta,
        joint_delta=motion.joint_delta,
    )

    metadata = {
        "robot": motion.robot,
        "fps": float(motion.fps),
        "frame_count": motion.num_frames(),
        "joint_names": list(motion.joint_names),
        "body_names": list(motion.body_names),
        "metadata": _to_jsonable(motion.metadata),
        "quality_metrics": _to_jsonable(motion.quality_metrics),
    }
    quality = {
        "frame_count": motion.num_frames(),
        "quality_metrics": _to_jsonable(motion.quality_metrics),
        "loss_curve": _to_jsonable(motion.loss_curve),
    }
    if quality_report is not None:
        quality["quality_report"] = quality_report.to_dict()
        quality["valid"] = bool(quality_report.valid)

    if metadata_path is not None:
        _write_metadata(Path(metadata_path), metadata)
    if quality_path is not None:
        _write_json(Path(quality_path), quality)

    return {"npz_path": str(output), "metadata": metadata, "quality": quality}


def load_refined_motion_npz(path: Path | str) -> RefinedMotion:
    data = np.load(Path(path), allow_pickle=False)
    motion = RefinedMotion(
        fps=float(data["fps"]),
        robot=str(data["robot"]),
        joint_names=[str(name) for name in data["joint_names"].tolist()],
        root_pos_w=np.asarray(data["root_pos_w"], dtype=np.float64),
        root_quat_xyzw=np.asarray(data["root_quat_xyzw"], dtype=np.float64),
        joint_pos=np.asarray(data["joint_pos"], dtype=np.float64),
        joint_vel=np.asarray(data["joint_vel"], dtype=np.float64),
        body_names=[str(name) for name in data["body_names"].tolist()],
        body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float64),
        body_quat_xyzw=np.asarray(data["body_quat_xyzw"], dtype=np.float64),
        root_delta=np.asarray(data["root_delta"], dtype=np.float64),
        joint_delta=np.asarray(data["joint_delta"], dtype=np.float64),
        metadata={"loaded_from": str(path)},
    )
    motion.validate()
    return motion

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to write YAML metadata files.") from exc
        path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        return
    _write_json(path, payload)


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
