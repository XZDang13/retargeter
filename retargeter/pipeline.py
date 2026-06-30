from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from retargeter.newton import (
    OnlineIKRetargetRunner,
    RobotSpec,
    SequenceIKRetargetRunner,
    IKRetargetFrameResult,
    RetargetedMotion,
    NewtonIKRetargetSolver,
    TorchRobotFK,
    export_retargeted_motion,
    load_newton_ik_config,
    retargeted_motion_from_frames,
)
from retargeter.newton.newton_backend import IKBackend
from retargeter.preprocess import (
    CanonicalHumanMotion,
    FootContactResult,
    MotionPreprocessor,
    PreprocessResult,
    REQUIRED_CANONICAL_BODY_NAMES,
    load_preprocess_config,
    run_smpl_preprocess,
)
from retargeter.progress import ProgressReporter, get_progress
from retargeter.scale import IKTargetSet, IKTargetBuilder
from retargeter.refinement import (
    RefinedMotion,
    RefinementQualityReport,
    evaluate_refinement_quality,
    export_refined_motion,
    run_refinement,
)
from retargeter.visualize import (
    NewtonReplayResult,
    default_human_path_for_replay_input,
    export_canonical_human_motion_npz,
    load_canonical_human_motion_npz,
    load_replay_motion_npz,
    replay_motion_with_newton,
    resolve_replay_motion_path,
)


ROBOT_DEFAULTS = {
    "unitree_g1_29": {
        "scaler_config": Path("retargeter/scale/configs/g1_29_scaler.yaml"),
        "target_config": Path("retargeter/scale/configs/g1_29_ik_targets.yaml"),
        "newton_config": Path("retargeter/newton/configs/g1_29_newton_ik.yaml"),
        "robot_spec": Path("retargeter/newton/configs/g1_29_robot.yaml"),
    },
    "unitree_g1_23": {
        "scaler_config": Path("retargeter/scale/configs/g1_23_scaler.yaml"),
        "target_config": Path("retargeter/scale/configs/g1_23_ik_targets.yaml"),
        "newton_config": Path("retargeter/newton/configs/g1_23_newton_ik.yaml"),
        "robot_spec": Path("retargeter/newton/configs/g1_23_robot.yaml"),
    },
}
ROBOT_ALIASES = {
    "g1_29": "unitree_g1_29",
    "g1_23": "unitree_g1_23",
}
DEFAULT_PREPROCESS_CONFIG = Path("retargeter/preprocess/configs/default_preprocess.yaml")


BackendFactory = Callable[[RobotSpec], IKBackend]
RefinementFKFactory = Callable[[RobotSpec], object]


@dataclass(frozen=True)
class PipelineConfigPaths:
    preprocess_config: Path
    scaler_config: Path
    target_config: Path
    newton_config: Path
    robot_spec: Path


@dataclass
class OnlinePipelineResult:
    motion: RetargetedMotion
    canonical_motion: CanonicalHumanMotion
    preprocess_result: PreprocessResult
    paths: dict[str, Path]


@dataclass
class RefinePipelineResult:
    retargeted_motion: RetargetedMotion
    final_motion: RefinedMotion
    canonical_motion: CanonicalHumanMotion
    preprocess_result: PreprocessResult
    quality_report: RefinementQualityReport
    paths: dict[str, Path]


@dataclass
class RefineBatchItemResult:
    input_path: Path | str
    output_dir: Path
    success: bool
    paths: dict[str, Path]
    frame_count: int | None
    quality_valid: bool | None
    error_type: str | None = None
    error: str | None = None


@dataclass
class RefineBatchResult:
    items: list[RefineBatchItemResult]
    manifest_path: Path
    success_count: int
    failure_count: int
    results_csv_path: Path | None = None


@dataclass
class ViewerPipelineResult:
    replay_result: NewtonReplayResult
    motion_path: Path
    human_path: Path | None


class OnlineRetargeter:
    """Stateful IK retargeter for frame-by-frame online use."""

    def __init__(
        self,
        *,
        robot: str = "unitree_g1_29",
        preprocess_config: Path | None = None,
        scaler_config: Path | None = None,
        target_config: Path | None = None,
        newton_config: Path | None = None,
        backend: IKBackend | None = None,
        backend_factory: BackendFactory | None = None,
    ):
        self.robot = normalize_robot_name(robot)
        self.config_paths = resolve_pipeline_config_paths(
            self.robot,
            preprocess_config=preprocess_config,
            scaler_config=scaler_config,
            target_config=target_config,
            newton_config=newton_config,
        )
        self.preprocess_config = load_preprocess_config(self.config_paths.preprocess_config)
        self.target_builder = IKTargetBuilder(self.config_paths.scaler_config, self.config_paths.target_config)
        self.robot_spec = _robot_spec_from_newton_config(self.config_paths.newton_config)
        chosen_backend = backend
        if chosen_backend is None and backend_factory is not None:
            chosen_backend = backend_factory(self.robot_spec)
        self.solver = NewtonIKRetargetSolver(
            self.config_paths.newton_config,
            backend=chosen_backend,
            target_builder=self.target_builder,
        )
        validate_robot_choices(self.robot, self.target_builder, self.solver)
        self.runner = OnlineIKRetargetRunner(self.solver)

    def reset(self) -> None:
        self.runner.reset()

    def step(
        self,
        motion: CanonicalHumanMotion,
        frame_idx: int,
        *,
        contact_result: FootContactResult | None = None,
    ) -> IKRetargetFrameResult:
        return self.runner.step(motion, frame_idx, contact_result=contact_result)

    def step_targets(
        self,
        tracking_targets: IKTargetSet,
        *,
        fps: float,
        frame_idx: int | None = None,
    ) -> IKRetargetFrameResult:
        return self.runner.step_targets(tracking_targets, fps=fps, frame_idx=frame_idx)

    def run_motion(
        self,
        motion: CanonicalHumanMotion,
        *,
        contact_result: FootContactResult | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RetargetedMotion:
        motion.validate()
        self.reset()
        frame_results = [self.step(motion, frame_idx, contact_result=contact_result) for frame_idx in range(motion.num_frames())]
        output = retargeted_motion_from_frames(frame_results, fps=float(motion.fps), metadata=dict(metadata or {}))
        output.metadata.update({"pipeline": "online", "robot": self.robot})
        return output


class RefinePipeline:
    """Offline training-data pipeline: preprocess, IK retarget, refine, quality report."""

    def __init__(
        self,
        *,
        robot: str = "unitree_g1_29",
        preprocess_config: Path | None = None,
        scaler_config: Path | None = None,
        target_config: Path | None = None,
        newton_config: Path | None = None,
        backend_factory: BackendFactory | None = None,
        refinement_fk_factory: RefinementFKFactory | None = None,
    ):
        self.robot = normalize_robot_name(robot)
        self.config_paths = resolve_pipeline_config_paths(
            self.robot,
            preprocess_config=preprocess_config,
            scaler_config=scaler_config,
            target_config=target_config,
            newton_config=newton_config,
        )
        self.backend_factory = backend_factory
        self.refinement_fk_factory = refinement_fk_factory

    def run(
        self,
        *,
        input_path: Path | str,
        output_dir: Path | str,
        model_type: str | None = None,
        fps: float | None = None,
        target_fps: float | None = None,
        gender: str | None = None,
        smpl_model_dir: Path | str = Path("assets/body_models"),
        device: str = "cpu",
        mock_frames: int = 120,
        return_vertices: bool = True,
        export_human: bool = True,
        export_retargeted: bool = False,
        refinement_config: Mapping[str, Any] | None = None,
        allow_invalid: bool = False,
        progress: ProgressReporter | None = None,
    ) -> RefinePipelineResult:
        reporter = get_progress(progress)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        preprocess_config = load_preprocess_config(self.config_paths.preprocess_config)
        reporter.stage(f"Preprocess {input_path}")
        canonical_motion, preprocess_result, source_metadata = load_pipeline_input(
            input_path,
            preprocess_config,
            model_type=model_type,
            fps=fps,
            target_fps=target_fps,
            gender=gender,
            smpl_model_dir=smpl_model_dir,
            device=device,
            mock_frames=mock_frames,
            return_vertices=return_vertices,
        )

        human_path: Path | None = None
        if export_human:
            reporter.stage("Export human motion")
            human_path = export_canonical_human_motion_npz(
                preprocess_result.motion,
                output_dir / "human.npz",
                preprocess_result=preprocess_result,
                require_mesh=False,
            )

        target_builder = IKTargetBuilder(self.config_paths.scaler_config, self.config_paths.target_config)
        robot_spec = _robot_spec_from_newton_config(self.config_paths.newton_config)
        backend = self.backend_factory(robot_spec) if self.backend_factory is not None else None
        solver = NewtonIKRetargetSolver(self.config_paths.newton_config, backend=backend, target_builder=target_builder)
        validate_robot_choices(self.robot, target_builder, solver)
        reporter.stage("IK retarget")
        retargeted_motion = SequenceIKRetargetRunner(solver).run(
            preprocess_result.motion,
            contact_result=preprocess_result.contact,
            progress=reporter,
        )
        retargeted_motion.metadata.update(
            _retargeted_metadata(
                source_metadata=source_metadata,
                robot=self.robot,
                config_paths=self.config_paths,
                preprocess_result=preprocess_result,
                pipeline="refine_retarget",
            )
        )

        refinement_cfg = copy.deepcopy(dict(refinement_config or {}))
        torch_fk = self.refinement_fk_factory(robot_spec) if self.refinement_fk_factory is not None else TorchRobotFK(robot_spec)
        reporter.stage("Refinement")
        final_motion = run_refinement(retargeted_motion, preprocess_result, robot_spec, torch_fk, config=refinement_cfg, progress=reporter)
        reporter.stage("Quality evaluation")
        quality_report = evaluate_refinement_quality(
            retargeted_motion,
            final_motion,
            robot_spec,
            config=refinement_cfg,
            contact_score=preprocess_result.contact,
        )
        final_motion.metadata.update(
            {
                "pipeline": "refine",
                "retargeted_motion_path": "retargeted_motion.npz" if export_retargeted else None,
                "retargeted_motion_exported": bool(export_retargeted),
                "refinement_config": copy.deepcopy(refinement_cfg),
                "refinement_quality_valid": bool(quality_report.valid),
                "rejected": bool((not quality_report.valid) and not allow_invalid),
                "training_motion": bool(quality_report.valid),
            }
        )
        final_output_dir = output_dir if quality_report.valid or allow_invalid else output_dir / "rejected"

        paths = {
            "final_motion": final_output_dir / "final_motion.npz",
            "final_metadata": final_output_dir / "final_meta.yaml",
            "final_quality": final_output_dir / "final_quality.json",
        }
        if export_retargeted:
            paths.update(
                {
                    "retargeted_motion": output_dir / "retargeted_motion.npz",
                    "retargeted_metadata": output_dir / "retargeted_meta.yaml",
                    "retargeted_quality": output_dir / "retargeted_quality.json",
                }
            )
        if human_path is not None:
            paths["human"] = human_path

        reporter.stage("Export refine outputs")
        if not export_retargeted:
            _clear_top_level_refine_retargeted_outputs(output_dir)
        if final_output_dir != output_dir:
            _clear_top_level_refine_final_outputs(output_dir)
        if export_retargeted:
            export_retargeted_motion(
                retargeted_motion,
                paths["retargeted_motion"],
                metadata_path=paths["retargeted_metadata"],
                quality_path=paths["retargeted_quality"],
            )
        export_refined_motion(
            final_motion,
            paths["final_motion"],
            metadata_path=paths["final_metadata"],
            quality_path=paths["final_quality"],
            quality_report=quality_report,
        )

        result = RefinePipelineResult(
            retargeted_motion=retargeted_motion,
            final_motion=final_motion,
            canonical_motion=canonical_motion,
            preprocess_result=preprocess_result,
            quality_report=quality_report,
            paths=paths,
        )
        return result

    def run_batch(
        self,
        *,
        input_paths: Sequence[Path | str],
        output_dir: Path | str,
        model_type: str | None = None,
        fps: float | None = None,
        target_fps: float | None = None,
        gender: str | None = None,
        smpl_model_dir: Path | str = Path("assets/body_models"),
        device: str = "cpu",
        mock_frames: int = 120,
        return_vertices: bool = True,
        export_human: bool = True,
        export_retargeted: bool = False,
        refinement_config: Mapping[str, Any] | None = None,
        allow_invalid: bool = False,
        fail_fast: bool = False,
        workers: int = 1,
        resume: bool = False,
        skip_existing: bool = False,
        overwrite: bool = False,
        input_dir: Path | str | None = None,
        preserve_tree: bool = False,
        dry_run: bool = False,
        summary_csv: Path | str | None = None,
        worker_devices: Sequence[str] | None = None,
        refinement_device: str | None = None,
        native_batch: bool = True,
        batch_size: int = 8,
        preprocess_workers: int = 1,
        batch_order: str = "length",
        progress: ProgressReporter | None = None,
    ) -> RefineBatchResult:
        reporter = get_progress(progress)
        inputs = list(input_paths)
        if not inputs:
            raise ValueError("run_batch requires at least one input path.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if preprocess_workers <= 0:
            raise ValueError("preprocess_workers must be positive.")
        if batch_order not in {"input", "length"}:
            raise ValueError("batch_order must be 'input' or 'length'.")

        if workers > 1 and (self.backend_factory is not None or self.refinement_fk_factory is not None):
            raise ValueError("Injected backend/refinement factories are only supported for workers=1.")
        if preprocess_workers > 1 and not native_batch:
            raise ValueError("preprocess_workers is only supported with native_batch=True.")

        from retargeter.batch.manifest import summarize, write_pass_reject_csv, write_summary_csv
        from retargeter.batch.native import NativeBatchRefineRunner
        from retargeter.batch.runner import BatchRefineRunner, build_refine_batch_tasks
        from retargeter.batch.worker import make_refine_task_processor, process_refine_batch_task

        tasks = build_refine_batch_tasks(
            inputs,
            output_dir,
            robot=self.robot,
            model_type=model_type,
            fps=fps,
            target_fps=target_fps,
            gender=gender,
            smpl_model_dir=smpl_model_dir,
            device=device,
            refinement_device=refinement_device,
            mock_frames=mock_frames,
            return_vertices=return_vertices,
            export_human=export_human,
            export_retargeted=export_retargeted,
            allow_invalid=allow_invalid,
            preprocess_config=self.config_paths.preprocess_config,
            scaler_config=self.config_paths.scaler_config,
            target_config=self.config_paths.target_config,
            newton_config=self.config_paths.newton_config,
            refinement_config=dict(refinement_config or {}),
            input_dir=Path(input_dir) if input_dir is not None else None,
            preserve_tree=preserve_tree,
            worker_devices=worker_devices,
        )
        use_native_batch = bool(native_batch) and self.backend_factory is None and self.refinement_fk_factory is None
        if preprocess_workers > 1 and not use_native_batch:
            raise ValueError("preprocess_workers is only supported by the native batch runner.")
        if use_native_batch:
            runner = NativeBatchRefineRunner(
                manifest_path=Path(output_dir) / "batch_manifest.json",
                robot=self.robot,
                allow_invalid=allow_invalid,
            )
            manifest = runner.run(
                tasks,
                batch_size=batch_size,
                fail_fast=fail_fast,
                resume=resume,
                skip_existing=skip_existing,
                overwrite=overwrite,
                dry_run=dry_run,
                preprocess_workers=preprocess_workers,
                batch_order=batch_order,
                progress=reporter,
            )
        else:
            nested_progress = reporter.child(position_offset=1) if workers == 1 and reporter.enabled and reporter.forced else None
            task_processor = (
                make_refine_task_processor(
                    backend_factory=self.backend_factory,
                    refinement_fk_factory=self.refinement_fk_factory,
                    progress=nested_progress,
                )
                if self.backend_factory is not None or self.refinement_fk_factory is not None or nested_progress is not None
                else process_refine_batch_task
            )
            runner = BatchRefineRunner(
                manifest_path=Path(output_dir) / "batch_manifest.json",
                robot=self.robot,
                allow_invalid=allow_invalid,
                task_processor=task_processor,
            )
            manifest = runner.run(
                tasks,
                workers=workers,
                fail_fast=fail_fast,
                resume=resume,
                skip_existing=skip_existing,
                overwrite=overwrite,
                dry_run=dry_run,
                progress=reporter,
            )
        if summary_csv is not None:
            write_summary_csv(summary_csv, manifest)
        results_csv_path = write_pass_reject_csv(Path(output_dir) / "batch_results.csv", manifest)

        summary = summarize(manifest)
        return RefineBatchResult(
            items=[_batch_record_to_pipeline_result_item(item, allow_invalid=allow_invalid) for item in manifest.items],
            manifest_path=runner.manifest_path,
            success_count=summary["success_count"],
            failure_count=summary["failure_count"],
            results_csv_path=results_csv_path,
        )


class ViewerPipeline:
    """Unified viewer for online and refine outputs."""

    def replay(
        self,
        *,
        input_path: Path | str,
        output_dir: Path | str | None = None,
        human_path: Path | str | None = None,
        robot_spec_path: Path | str | None = None,
        viewer: str = "file",
        fps: float | None = None,
        loop: bool = False,
        max_loops: int | None = 1,
        realtime: bool = False,
        port: int = 8080,
        share: bool = False,
        replay_name: str | None = None,
        human_offset: np.ndarray | tuple[float, float, float] | list[float] | None = None,
        backend=None,
        viewer_factory=None,
    ) -> ViewerPipelineResult:
        motion_path = resolve_replay_motion_path(input_path)
        motion = load_replay_motion_npz(motion_path)
        robot_spec = RobotSpec.from_yaml(robot_spec_path or default_robot_spec_path(motion.robot))
        explicit_human_path = human_path is not None
        resolved_human_path = Path(human_path) if explicit_human_path else default_human_path_for_replay_input(input_path)
        human_motion = _load_human_motion_for_replay(resolved_human_path, explicit=explicit_human_path)

        viewer_kind = str(viewer).lower()
        output = _viewer_output_path(input_path, output_dir, viewer_kind, replay_name)
        replay_result = replay_motion_with_newton(
            motion,
            robot_spec,
            viewer=viewer_kind,
            output_path=output,
            fps=fps,
            loop=loop,
            max_loops=max_loops,
            realtime=realtime,
            port=port,
            share=share,
            backend=backend,
            viewer_factory=viewer_factory,
            human_motion=human_motion,
            human_offset=human_offset,
        )
        return ViewerPipelineResult(
            replay_result=replay_result,
            motion_path=motion_path,
            human_path=resolved_human_path,
        )


def run_online_cli_pipeline(
    *,
    input_path: Path | str,
    output_dir: Path | str,
    robot: str = "unitree_g1_29",
    model_type: str | None = None,
    fps: float | None = None,
    target_fps: float | None = None,
    gender: str | None = None,
    smpl_model_dir: Path | str = Path("assets/body_models"),
    device: str = "cpu",
    mock_frames: int = 120,
    return_vertices: bool = False,
    preprocess_config: Path | None = None,
    scaler_config: Path | None = None,
    target_config: Path | None = None,
    newton_config: Path | None = None,
    backend_factory: BackendFactory | None = None,
) -> OnlinePipelineResult:
    robot = normalize_robot_name(robot)
    paths = resolve_pipeline_config_paths(
        robot,
        preprocess_config=preprocess_config,
        scaler_config=scaler_config,
        target_config=target_config,
        newton_config=newton_config,
    )
    pre_cfg = load_preprocess_config(paths.preprocess_config)
    canonical_motion, preprocess_result, source_metadata = load_pipeline_input(
        input_path,
        pre_cfg,
        model_type=model_type,
        fps=fps,
        target_fps=target_fps,
        gender=gender,
        smpl_model_dir=smpl_model_dir,
        device=device,
        mock_frames=mock_frames,
        return_vertices=return_vertices,
    )
    retargeter = OnlineRetargeter(
        robot=robot,
        preprocess_config=paths.preprocess_config,
        scaler_config=paths.scaler_config,
        target_config=paths.target_config,
        newton_config=paths.newton_config,
        backend_factory=backend_factory,
    )
    motion = retargeter.run_motion(
        preprocess_result.motion,
        contact_result=preprocess_result.contact,
        metadata=_retargeted_metadata(
            source_metadata=source_metadata,
            robot=robot,
            config_paths=paths,
            preprocess_result=preprocess_result,
            pipeline="online",
        ),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "online_motion": output_dir / "online_motion.npz",
        "online_metadata": output_dir / "online_meta.yaml",
        "online_quality": output_dir / "online_quality.json",
    }
    export_retargeted_motion(
        motion,
        written["online_motion"],
        metadata_path=written["online_metadata"],
        quality_path=written["online_quality"],
    )
    return OnlinePipelineResult(
        motion=motion,
        canonical_motion=canonical_motion,
        preprocess_result=preprocess_result,
        paths=written,
    )


def load_pipeline_input(
    input_path: Path | str,
    preprocess_config,
    *,
    model_type: str | None,
    fps: float | None,
    target_fps: float | None,
    gender: str | None,
    smpl_model_dir: Path | str,
    device: str,
    mock_frames: int,
    return_vertices: bool,
) -> tuple[CanonicalHumanMotion, PreprocessResult, dict[str, Any]]:
    if str(input_path).lower() == "mock":
        mock_fps = _mock_motion_fps(fps=fps, target_fps=target_fps)
        canonical_motion = make_mock_canonical_motion(num_frames=int(mock_frames), fps=mock_fps)
        preprocess_result = MotionPreprocessor(preprocess_config).process(canonical_motion)
        preprocess_result.contact = make_mock_contact_result(preprocess_result.motion)
        preprocess_result.metadata["contact_available"] = True
        preprocess_result.metadata["mock_contact"] = True
        return canonical_motion, preprocess_result, {
            "input": "mock",
            "mock_mode": True,
            "model_type": None,
            "smpl_fk_applied": False,
            "target_fps": mock_fps if target_fps is not None else None,
        }

    output = run_smpl_preprocess(
        input_path,
        preprocess_config,
        model_type=model_type,  # type: ignore[arg-type]
        fps=fps,
        gender=gender,
        smpl_model_dir=smpl_model_dir,
        device=device,
        return_vertices=return_vertices,
        target_fps=target_fps,
    )
    return output.canonical_motion, output.preprocess_result, output.source_metadata


def load_refinement_config_file(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Refinement config file does not exist: {config_path}")
    if config_path.suffix.lower() == ".json":
        data = json.loads(config_path.read_text(encoding="utf-8") or "{}")
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load refinement YAML config files.") from exc
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Refinement config {config_path} must contain a mapping.")
    return dict(data)


def refinement_config_with_overrides(
    config: Mapping[str, Any] | None,
    *,
    iterations: int | None = None,
    lr: float | None = None,
    log_interval: int | None = None,
    max_root_delta: float | None = None,
    max_joint_delta: float | None = None,
    device: str | None = None,
    dtype: str | None = None,
    lbfgs_enabled: bool | None = None,
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(config or {}))
    overrides = {
        "iterations": iterations,
        "lr": lr,
        "log_interval": log_interval,
        "max_root_delta": max_root_delta,
        "max_joint_delta": max_joint_delta,
        "device": device,
        "dtype": dtype,
        "lbfgs_enabled": lbfgs_enabled,
    }
    refiner_overrides = {key: value for key, value in overrides.items() if value is not None}
    if refiner_overrides:
        section = merged.get("refiner", {})
        if section is None:
            section = {}
        if not isinstance(section, dict):
            raise TypeError("refinement config 'refiner' must be a mapping.")
        section = dict(section)
        section.update(refiner_overrides)
        merged["refiner"] = section
    return merged


def normalize_robot_name(robot: str) -> str:
    normalized = ROBOT_ALIASES.get(str(robot), str(robot))
    if normalized not in ROBOT_DEFAULTS:
        raise ValueError(f"Unsupported robot {robot!r}; expected one of {sorted(ROBOT_DEFAULTS)}.")
    return normalized


def resolve_pipeline_config_paths(
    robot: str,
    *,
    preprocess_config: Path | None = None,
    scaler_config: Path | None = None,
    target_config: Path | None = None,
    newton_config: Path | None = None,
) -> PipelineConfigPaths:
    robot = normalize_robot_name(robot)
    defaults = ROBOT_DEFAULTS[robot]
    chosen_newton = Path(newton_config or defaults["newton_config"])
    return PipelineConfigPaths(
        preprocess_config=Path(preprocess_config or DEFAULT_PREPROCESS_CONFIG),
        scaler_config=Path(scaler_config or defaults["scaler_config"]),
        target_config=Path(target_config or defaults["target_config"]),
        newton_config=chosen_newton,
        robot_spec=Path(defaults["robot_spec"]),
    )


def default_robot_spec_path(robot: str) -> Path:
    return Path(ROBOT_DEFAULTS[normalize_robot_name(robot)]["robot_spec"])


def _load_human_motion_for_replay(path: Path | None, *, explicit: bool) -> CanonicalHumanMotion | None:
    if path is None:
        return None
    motion = load_canonical_human_motion_npz(path)
    if explicit or (motion.vertices_w is not None and motion.mesh_faces is not None):
        return motion
    return None


def validate_robot_choices(robot: str, target_builder: IKTargetBuilder, solver: NewtonIKRetargetSolver) -> None:
    if target_builder.scaler.robot != robot:
        raise ValueError(f"Scaler config robot {target_builder.scaler.robot!r} does not match requested robot {robot!r}.")
    if solver.robot_spec.robot != robot:
        raise ValueError(f"Newton config robot {solver.robot_spec.robot!r} does not match requested robot {robot!r}.")


def make_mock_canonical_motion(num_frames: int = 120, fps: float = 30.0) -> CanonicalHumanMotion:
    if num_frames <= 0:
        raise ValueError("mock_frames must be positive.")
    body_names = list(REQUIRED_CANONICAL_BODY_NAMES)
    pos = np.zeros((num_frames, len(body_names), 3), dtype=np.float64)
    quat = np.zeros((num_frames, len(body_names), 4), dtype=np.float64)
    quat[..., 3] = 1.0
    offsets = {
        "pelvis": [0.0, 0.0, 0.90],
        "chest": [0.0, 0.0, 1.30],
        "head": [0.0, 0.0, 1.60],
        "left_shoulder": [0.0, 0.18, 1.35],
        "right_shoulder": [0.0, -0.18, 1.35],
        "left_elbow": [0.05, 0.35, 1.12],
        "right_elbow": [0.05, -0.35, 1.12],
        "left_hand": [0.08, 0.48, 0.92],
        "right_hand": [0.08, -0.48, 0.92],
        "left_hip": [0.0, 0.09, 0.84],
        "right_hip": [0.0, -0.09, 0.84],
        "left_knee": [0.02, 0.10, 0.45],
        "right_knee": [0.02, -0.10, 0.45],
        "left_ankle": [0.04, 0.11, 0.03],
        "right_ankle": [0.04, -0.11, 0.03],
        "left_foot": [0.04, 0.11, 0.03],
        "right_foot": [0.04, -0.11, 0.03],
        "left_toe": [0.17, 0.11, 0.03],
        "right_toe": [0.17, -0.11, 0.03],
        "left_heel": [-0.06, 0.11, 0.03],
        "right_heel": [-0.06, -0.11, 0.03],
    }
    phase = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False)
    root = np.stack(
        [np.linspace(0.0, 0.20, num_frames), 0.02 * np.sin(phase), 0.02 * np.sin(2.0 * phase)],
        axis=1,
    )
    for idx, name in enumerate(body_names):
        pos[:, idx, :] = root + np.asarray(offsets[name], dtype=np.float64)
    left_swing = 0.10 * np.sin(phase)
    right_swing = -left_swing
    for name, swing in [("left_hand", left_swing), ("left_elbow", left_swing * 0.5)]:
        pos[:, body_names.index(name), 0] += swing
    for name, swing in [("right_hand", right_swing), ("right_elbow", right_swing * 0.5)]:
        pos[:, body_names.index(name), 0] += swing
    return CanonicalHumanMotion(
        fps=float(fps),
        body_names=body_names,
        body_pos_w=pos,
        body_quat_xyzw=quat,
        vertices_w=None,
        metadata={"source": "mock", "world_frame": "z_up"},
    )


def make_mock_contact_result(motion: CanonicalHumanMotion) -> FootContactResult:
    t = motion.num_frames()
    phase = np.arange(t)
    left_score = (0.5 + 0.5 * np.sin(2.0 * np.pi * phase / max(8, t // 4))).astype(np.float64)
    right_score = 1.0 - left_score
    score = {
        "left_foot": left_score,
        "right_foot": right_score,
        "left_toe": left_score,
        "right_toe": right_score,
        "left_heel": left_score,
        "right_heel": right_score,
    }
    binary = {name: values >= 0.5 for name, values in score.items()}
    foot_height = {}
    foot_speed = {}
    for region in score:
        if region in motion.body_names:
            body_pos = motion.get_body_pos(region)
            foot_height[region] = body_pos[:, 2].copy()
        else:
            foot_height[region] = np.zeros(t, dtype=np.float64)
        foot_speed[region] = np.zeros(t, dtype=np.float64)
    return FootContactResult(
        contact_score=score,
        contact_binary=binary,
        foot_height=foot_height,
        foot_speed=foot_speed,
        ground_height=0.0,
        metadata={"source": "mock", "regions": list(score)},
    )


def _robot_spec_from_newton_config(newton_config_path: Path) -> RobotSpec:
    newton_config = load_newton_ik_config(newton_config_path)
    return RobotSpec.from_yaml(_resolve_config_relative_path(newton_config_path, str(newton_config["robot_config"])))


def _resolve_config_relative_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return (config_path.parent / path).resolve()


def _retargeted_metadata(
    *,
    source_metadata: Mapping[str, Any],
    robot: str,
    config_paths: PipelineConfigPaths,
    preprocess_result: PreprocessResult,
    pipeline: str,
) -> dict[str, Any]:
    return {
        "pipeline": pipeline,
        "robot": robot,
        "config_paths": {
            "preprocess_config": str(config_paths.preprocess_config),
            "scaler_config": str(config_paths.scaler_config),
            "target_config": str(config_paths.target_config),
            "newton_config": str(config_paths.newton_config),
        },
        "frame_count": preprocess_result.motion.num_frames(),
        "fps": float(preprocess_result.motion.fps),
        "preprocess_warnings": list(preprocess_result.warnings),
        "contact_available": preprocess_result.contact is not None,
        "preprocess_metadata": dict(preprocess_result.metadata),
        "source": dict(source_metadata),
    }


def _clear_top_level_refine_final_outputs(output_dir: Path) -> None:
    for name in ("final_motion.npz", "final_meta.yaml", "final_quality.json"):
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def _clear_top_level_refine_retargeted_outputs(output_dir: Path) -> None:
    for name in ("retargeted_motion.npz", "retargeted_meta.yaml", "retargeted_quality.json"):
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def _batch_record_to_pipeline_result_item(record, *, allow_invalid: bool) -> RefineBatchItemResult:
    return RefineBatchItemResult(
        input_path=record.input,
        output_dir=Path(record.output_dir),
        success=record.status in {"success", "skipped"},
        paths={key: Path(value) for key, value in record.paths.items()},
        frame_count=record.frame_count,
        quality_valid=record.quality_valid,
        error_type=record.error_type,
        error=record.error,
    )


def _next_batch_output_dir(output_dir: Path, input_path: Path | str, used_names: dict[str, int]) -> Path:
    base_name = _safe_batch_item_name(input_path)
    next_count = used_names.get(base_name, 0) + 1
    used_names[base_name] = next_count
    item_name = base_name if next_count == 1 else f"{base_name}__{next_count}"
    return output_dir / item_name


def _safe_batch_item_name(input_path: Path | str) -> str:
    raw = str(input_path)
    if raw.lower() == "mock":
        stem = "mock"
    else:
        stem = Path(input_path).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return safe or "item"


def _write_refine_batch_manifest(
    path: Path,
    *,
    robot: str,
    inputs: Sequence[Path | str],
    items: Sequence[RefineBatchItemResult],
    success_count: int,
    failure_count: int,
) -> None:
    payload = {
        "pipeline": "refine_batch",
        "robot": robot,
        "input_count": len(inputs),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
        "items": [_refine_batch_item_to_manifest(item) for item in items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _refine_batch_item_to_manifest(item: RefineBatchItemResult) -> dict[str, Any]:
    return {
        "input": str(item.input_path),
        "output_dir": str(item.output_dir),
        "status": "success" if item.success else "failed",
        "paths": {key: str(value) for key, value in item.paths.items()},
        "quality_valid": item.quality_valid,
        "frame_count": item.frame_count,
        "error_type": item.error_type,
        "error": item.error,
    }


def _mock_motion_fps(*, fps: float | None, target_fps: float | None) -> float:
    if target_fps is not None:
        target = float(target_fps)
        if target <= 0.0 or not np.isfinite(target):
            raise ValueError(f"target_fps must be positive and finite, got {target_fps!r}.")
        return target
    return float(fps or 30.0)


def _viewer_output_path(
    input_path: Path | str,
    output_dir: Path | str | None,
    viewer_kind: str,
    replay_name: str | None,
) -> Path | None:
    if viewer_kind not in {"file", "usd"}:
        return None
    base = Path(output_dir) if output_dir is not None else _default_viewer_output_dir(input_path)
    base.mkdir(parents=True, exist_ok=True)
    if replay_name is not None:
        return base / replay_name
    return base / ("newton_replay.json" if viewer_kind == "file" else "newton_replay.usd")


def _default_viewer_output_dir(input_path: Path | str) -> Path:
    path = Path(input_path)
    if path.is_dir():
        return path / "viewer"
    return path.parent / "viewer"
