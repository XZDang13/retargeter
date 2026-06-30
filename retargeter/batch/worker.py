from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from retargeter.progress import ProgressReporter

from .manifest import BatchItemRecord


@dataclass
class RefineBatchTask:
    input_path: Path | str
    output_dir: Path
    robot: str = "unitree_g1_29"
    model_type: str | None = None
    fps: float | None = None
    target_fps: float | None = None
    gender: str | None = None
    smpl_model_dir: Path | str = Path("assets/body_models")
    device: str = "cpu"
    refinement_device: str | None = None
    mock_frames: int = 120
    return_vertices: bool = True
    export_human: bool = True
    export_retargeted: bool = False
    allow_invalid: bool = False
    preprocess_config: Path | None = None
    scaler_config: Path | None = None
    target_config: Path | None = None
    newton_config: Path | None = None
    refinement_config: dict[str, Any] = field(default_factory=dict)


TaskProcessor = Callable[[RefineBatchTask], BatchItemRecord]


def process_refine_batch_task(task: RefineBatchTask) -> BatchItemRecord:
    return _process_refine_batch_task(task)


def make_refine_task_processor(
    *,
    backend_factory=None,
    refinement_fk_factory=None,
    progress: ProgressReporter | None = None,
) -> TaskProcessor:
    def _processor(task: RefineBatchTask) -> BatchItemRecord:
        return _process_refine_batch_task(
            task,
            backend_factory=backend_factory,
            refinement_fk_factory=refinement_fk_factory,
            progress=progress,
        )

    return _processor


def _process_refine_batch_task(
    task: RefineBatchTask,
    *,
    backend_factory=None,
    refinement_fk_factory=None,
    progress: ProgressReporter | None = None,
) -> BatchItemRecord:
    start = time.perf_counter()
    try:
        from retargeter.pipeline import RefinePipeline

        pipeline = RefinePipeline(
            robot=task.robot,
            preprocess_config=task.preprocess_config,
            scaler_config=task.scaler_config,
            target_config=task.target_config,
            newton_config=task.newton_config,
            backend_factory=backend_factory,
            refinement_fk_factory=refinement_fk_factory,
        )
        result = pipeline.run(
            input_path=task.input_path,
            output_dir=task.output_dir,
            model_type=task.model_type,
            fps=task.fps,
            target_fps=task.target_fps,
            gender=task.gender,
            smpl_model_dir=task.smpl_model_dir,
            device=task.device,
            mock_frames=task.mock_frames,
            return_vertices=task.return_vertices,
            export_human=task.export_human,
            export_retargeted=task.export_retargeted,
            refinement_config=_refinement_config_for_task(task),
            allow_invalid=task.allow_invalid,
            progress=progress,
        )
        runtime = time.perf_counter() - start
        valid = bool(result.quality_report.valid)
        return BatchItemRecord(
            input=str(task.input_path),
            output_dir=str(task.output_dir),
            status="success" if valid else "invalid",
            frame_count=result.final_motion.num_frames(),
            fps=float(result.final_motion.fps),
            runtime_sec=runtime,
            quality_valid=valid,
            quality_summary=_quality_summary_from_report(result.quality_report),
            paths={key: str(value) for key, value in result.paths.items()},
            error_type=None,
            error=None,
        )
    except Exception as exc:
        runtime = time.perf_counter() - start
        if _looks_like_invalid_quality(exc):
            return _record_from_existing_outputs(task, status="invalid", runtime_sec=runtime, exc=exc)
        return BatchItemRecord(
            input=str(task.input_path),
            output_dir=str(task.output_dir),
            status="failed",
            runtime_sec=runtime,
            paths=_existing_standard_paths(task.output_dir),
            error_type=type(exc).__name__,
            error=str(exc),
        )


def _refinement_config_for_task(task: RefineBatchTask) -> dict[str, Any]:
    config = copy.deepcopy(dict(task.refinement_config or {}))
    if task.refinement_device is not None:
        refiner = config.get("refiner", {})
        if refiner is None:
            refiner = {}
        if not isinstance(refiner, dict):
            raise TypeError("refinement config 'refiner' must be a mapping.")
        refiner = dict(refiner)
        refiner["device"] = task.refinement_device
        config["refiner"] = refiner
    return config


def _looks_like_invalid_quality(exc: Exception) -> bool:
    message = str(exc)
    return "Refine output failed RefinementQualityReport" in message or "RefinementQualityReport" in message


def _record_from_existing_outputs(
    task: RefineBatchTask,
    *,
    status: str,
    runtime_sec: float | None,
    exc: Exception | None = None,
) -> BatchItemRecord:
    output_dir = Path(task.output_dir)
    quality = _load_json(output_dir / "final_quality.json")
    motion_info = _load_motion_info(output_dir / "final_motion.npz")
    quality_report = quality.get("quality_report", {}) if isinstance(quality, dict) else {}
    valid = quality.get("valid", quality_report.get("valid")) if isinstance(quality, dict) else None
    return BatchItemRecord(
        input=str(task.input_path),
        output_dir=str(task.output_dir),
        status=status,
        frame_count=_first_int(motion_info.get("frame_count"), quality.get("frame_count")),
        fps=_first_float(motion_info.get("fps"), _nested_get(quality_report, "metrics", "fps")),
        runtime_sec=runtime_sec,
        quality_valid=None if valid is None else bool(valid),
        quality_summary=_quality_summary_from_dict(quality_report),
        paths=_existing_standard_paths(output_dir),
        error_type=None if exc is None else "RefinementQualityReport",
        error=None if exc is None else str(exc),
    )


def _existing_standard_paths(output_dir: Path) -> dict[str, str]:
    names = {
        "retargeted_motion": "retargeted_motion.npz",
        "retargeted_metadata": "retargeted_meta.yaml",
        "retargeted_quality": "retargeted_quality.json",
        "final_motion": "final_motion.npz",
        "final_metadata": "final_meta.yaml",
        "final_quality": "final_quality.json",
        "human": "human.npz",
    }
    return {key: str(output_dir / name) for key, name in names.items() if (output_dir / name).exists()}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_motion_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            frames = int(data["joint_pos"].shape[0]) if "joint_pos" in data.files else None
            fps = float(data["fps"]) if "fps" in data.files else None
        return {"frame_count": frames, "fps": fps}
    except Exception:
        return {}


def _quality_summary_from_report(report) -> dict[str, Any]:
    return {
        "valid": bool(report.valid),
        "failures": list(report.failures),
        "metrics": _compact_metrics(dict(report.metrics)),
    }


def _quality_summary_from_dict(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    return {
        "valid": report.get("valid"),
        "failures": list(report.get("failures", [])),
        "metrics": _compact_metrics(dict(report.get("metrics", {}))),
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "num_frames",
        "fps",
        "body_pos_deviation_max_m",
        "body_pos_deviation_mean_m",
        "root_pos_deviation_max_m",
        "joint_pos_deviation_max_rad",
        "penetration_worsening_m",
        "skating_improvement_m_s",
    ]
    return {key: _jsonable(metrics[key]) for key in keys if key in metrics}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is not None:
            return int(value)
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None


def _nested_get(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
