from __future__ import annotations

import copy
import concurrent.futures
import json
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from retargeter.newton import BatchSequenceIKRetargetRunner, NewtonIKRetargetSolver, TorchRobotFK
from retargeter.progress import ProgressReporter, get_progress
from retargeter.refinement import evaluate_refinement_quality, export_refined_motion, run_refinement, run_refinement_batch
from retargeter.scale import IKTargetBuilder

from .manifest import BatchItemRecord, BatchManifest, load_manifest, save_manifest, summarize, update_item
from .worker import RefineBatchTask, _quality_summary_from_report, _refinement_config_for_task


@dataclass
class _PreparedRefineItem:
    task: RefineBatchTask
    canonical_motion: Any
    preprocess_result: Any
    source_metadata: dict[str, Any]
    human_path: Path | None
    start_time: float


class NativeBatchRefineRunner:
    def __init__(
        self,
        *,
        manifest_path: Path | str,
        robot: str = "unitree_g1_29",
        allow_invalid: bool = False,
        backend_factory=None,
        refinement_fk_factory=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.robot = robot
        self.allow_invalid = bool(allow_invalid)
        self.backend_factory = backend_factory
        self.refinement_fk_factory = refinement_fk_factory

    def run(
        self,
        tasks: Sequence[RefineBatchTask],
        *,
        batch_size: int = 8,
        fail_fast: bool = False,
        resume: bool = False,
        skip_existing: bool = False,
        overwrite: bool = False,
        dry_run: bool = False,
        preprocess_workers: int = 1,
        progress: ProgressReporter | None = None,
    ) -> BatchManifest:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if preprocess_workers <= 0:
            raise ValueError("preprocess_workers must be positive.")
        reporter = get_progress(progress)
        task_list = list(tasks)
        manifest = _initial_manifest(
            self.manifest_path,
            task_list,
            robot=self.robot,
            allow_invalid=self.allow_invalid,
            resume=resume,
            skip_existing=skip_existing,
            overwrite=overwrite,
        )
        save_manifest(self.manifest_path, manifest)
        if dry_run:
            reporter.stage(f"Batch dry run planned {len(task_list)} item(s)")
            return manifest

        runnable = [task for task in task_list if _record_for_task(manifest, task).status == "pending"]
        if preprocess_workers > 1:
            return self._run_with_parallel_preprocess(
                manifest,
                runnable,
                batch_size=batch_size,
                preprocess_workers=preprocess_workers,
                fail_fast=fail_fast,
                reporter=reporter,
            )

        stop_after_chunk = False
        with reporter.bar(total=len(runnable), desc="Batch refine", unit="clip") as bar:
            bar.set_postfix(_native_progress_postfix(manifest, batch_size=batch_size), refresh=False)
            for start in range(0, len(runnable), batch_size):
                if stop_after_chunk:
                    break
                chunk = runnable[start : start + batch_size]
                for task in chunk:
                    update_item(manifest, _running_record(task))
                save_manifest(self.manifest_path, manifest)
                bar.set_postfix(_native_progress_postfix(manifest, batch_size=batch_size, current=chunk), refresh=False)
                for result in self._process_chunk(chunk, progress=reporter.child(position_offset=1) if reporter.enabled and reporter.forced else None):
                    update_item(manifest, result)
                    save_manifest(self.manifest_path, manifest)
                    bar.update(1)
                    bar.set_postfix(_native_progress_postfix(manifest, batch_size=batch_size), refresh=False)
                    if fail_fast and result.status == "failed":
                        stop_after_chunk = True
                if stop_after_chunk:
                    remaining = runnable[start + len(chunk) :]
                    for task in remaining:
                        update_item(manifest, _skipped_record(task, reason="fail_fast"))
                    save_manifest(self.manifest_path, manifest)
                    if remaining:
                        bar.update(len(remaining))
                        bar.set_postfix(_native_progress_postfix(manifest, batch_size=batch_size), refresh=False)
        return manifest

    def _run_with_parallel_preprocess(
        self,
        manifest: BatchManifest,
        runnable: list[RefineBatchTask],
        *,
        batch_size: int,
        preprocess_workers: int,
        fail_fast: bool,
        reporter: ProgressReporter,
    ) -> BatchManifest:
        stop_after_chunk = False
        context = multiprocessing.get_context("spawn")
        with reporter.bar(total=len(runnable), desc="Batch refine", unit="clip") as bar:
            bar.set_postfix(
                _native_progress_postfix(manifest, batch_size=batch_size, preprocess_workers=preprocess_workers),
                refresh=False,
            )
            with concurrent.futures.ProcessPoolExecutor(max_workers=preprocess_workers, mp_context=context) as executor:
                for start in range(0, len(runnable), batch_size):
                    if stop_after_chunk:
                        break
                    chunk = runnable[start : start + batch_size]
                    for task in chunk:
                        update_item(manifest, _running_record(task))
                    save_manifest(self.manifest_path, manifest)
                    bar.set_postfix(
                        _native_progress_postfix(
                            manifest,
                            batch_size=batch_size,
                            preprocess_workers=preprocess_workers,
                            current=chunk,
                        ),
                        refresh=False,
                    )
                    for result in self._process_chunk_with_parallel_preprocess(
                        chunk,
                        executor=executor,
                        progress=reporter.child(position_offset=1) if reporter.enabled and reporter.forced else None,
                    ):
                        update_item(manifest, result)
                        save_manifest(self.manifest_path, manifest)
                        bar.update(1)
                        bar.set_postfix(
                            _native_progress_postfix(
                                manifest,
                                batch_size=batch_size,
                                preprocess_workers=preprocess_workers,
                            ),
                            refresh=False,
                        )
                        if fail_fast and result.status == "failed":
                            stop_after_chunk = True
                    if stop_after_chunk:
                        remaining = runnable[start + len(chunk) :]
                        for task in remaining:
                            update_item(manifest, _skipped_record(task, reason="fail_fast"))
                        save_manifest(self.manifest_path, manifest)
                        if remaining:
                            bar.update(len(remaining))
                            bar.set_postfix(
                                _native_progress_postfix(
                                    manifest,
                                    batch_size=batch_size,
                                    preprocess_workers=preprocess_workers,
                                ),
                                refresh=False,
                            )
        return manifest

    def _process_chunk(
        self,
        tasks: Sequence[RefineBatchTask],
        *,
        progress: ProgressReporter | None,
    ) -> list[BatchItemRecord]:
        prepared: list[_PreparedRefineItem] = []
        records: dict[tuple[str, str], BatchItemRecord] = {}
        for task in tasks:
            start = time.perf_counter()
            try:
                prepared.append(_prepare_refine_item(task, start_time=start, progress=progress))
            except Exception as exc:
                records[_task_key(task)] = _failed_record(task, start, exc)

        if prepared:
            for record in self._process_prepared_items(prepared, progress=progress):
                records[_record_key(record)] = record

        output = []
        for task in tasks:
            record = records.get(_task_key(task))
            if record is None:
                record = BatchItemRecord(
                    input=str(task.input_path),
                    output_dir=str(task.output_dir),
                    status="failed",
                    error_type="RuntimeError",
                    error="native batch item did not produce a result",
                )
            output.append(record)
        return output

    def _process_chunk_with_parallel_preprocess(
        self,
        tasks: Sequence[RefineBatchTask],
        *,
        executor: concurrent.futures.ProcessPoolExecutor,
        progress: ProgressReporter | None,
    ) -> list[BatchItemRecord]:
        prepared: list[_PreparedRefineItem] = []
        records: dict[tuple[str, str], BatchItemRecord] = {}
        futures: dict[concurrent.futures.Future[_PreparedRefineItem], tuple[RefineBatchTask, float]] = {}
        for task in tasks:
            start = time.perf_counter()
            futures[executor.submit(_prepare_refine_item_worker, task, start)] = (task, start)

        for future in concurrent.futures.as_completed(futures):
            task, start = futures[future]
            try:
                prepared.append(future.result())
            except Exception as exc:
                records[_task_key(task)] = _failed_record(task, start, exc)

        if prepared:
            order = {_task_key(task): index for index, task in enumerate(tasks)}
            prepared.sort(key=lambda item: order[_task_key(item.task)])
            for record in self._process_prepared_items(prepared, progress=progress):
                records[_record_key(record)] = record

        output = []
        for task in tasks:
            record = records.get(_task_key(task))
            if record is None:
                record = BatchItemRecord(
                    input=str(task.input_path),
                    output_dir=str(task.output_dir),
                    status="failed",
                    error_type="RuntimeError",
                    error="native batch item did not produce a result",
                )
            output.append(record)
        return output

    def _process_prepared_items(
        self,
        items: list[_PreparedRefineItem],
        *,
        progress: ProgressReporter | None,
    ) -> list[BatchItemRecord]:
        try:
            retargeted = _run_batched_ik(items, self.robot, backend_factory=self.backend_factory, progress=progress)
        except Exception as exc:
            return [_failed_record(item.task, item.start_time, exc) for item in items]

        records: list[BatchItemRecord] = []
        index_by_id = {id(item): index for index, item in enumerate(items)}
        for group in _group_by_refinement_config(items):
            group_indices = [index_by_id[id(item)] for item in group]
            group_retargeted = [retargeted[index] for index in group_indices]
            for item, motion in zip(group, group_retargeted):
                motion.metadata.update(
                    _retargeted_metadata_for_item(
                        item,
                        robot=self.robot,
                    )
                )
            try:
                refined = _run_batched_refinement(
                    group,
                    group_retargeted,
                    refinement_fk_factory=self.refinement_fk_factory,
                    progress=progress,
                )
            except Exception:
                refined = _run_refinement_fallback(
                    group,
                    group_retargeted,
                    refinement_fk_factory=self.refinement_fk_factory,
                    progress=progress,
                )
            for item, retargeted_motion, refined_motion in zip(group, group_retargeted, refined):
                if isinstance(refined_motion, BatchItemRecord):
                    records.append(refined_motion)
                    continue
                try:
                    records.append(_export_item(item, retargeted_motion, refined_motion))
                except Exception as exc:
                    records.append(_failed_record(item.task, item.start_time, exc))
        return records


def _prepare_refine_item(
    task: RefineBatchTask,
    *,
    start_time: float,
    progress: ProgressReporter | None,
) -> _PreparedRefineItem:
    from retargeter.pipeline import load_pipeline_input
    from retargeter.preprocess import load_preprocess_config
    from retargeter.visualize import export_canonical_human_motion_npz

    reporter = get_progress(progress)
    Path(task.output_dir).mkdir(parents=True, exist_ok=True)
    preprocess_config = load_preprocess_config(task.preprocess_config)
    reporter.stage(f"Preprocess {task.input_path}")
    canonical_motion, preprocess_result, source_metadata = load_pipeline_input(
        task.input_path,
        preprocess_config,
        model_type=task.model_type,
        fps=task.fps,
        target_fps=task.target_fps,
        gender=task.gender,
        smpl_model_dir=task.smpl_model_dir,
        device=task.device,
        mock_frames=task.mock_frames,
        return_vertices=task.return_vertices,
    )
    human_path = None
    if task.export_human:
        reporter.stage("Export human motion")
        human_path = export_canonical_human_motion_npz(
            preprocess_result.motion,
            Path(task.output_dir) / "human.npz",
            preprocess_result=preprocess_result,
            require_mesh=False,
        )
    return _PreparedRefineItem(
        task=task,
        canonical_motion=canonical_motion,
        preprocess_result=preprocess_result,
        source_metadata=dict(source_metadata),
        human_path=human_path,
        start_time=start_time,
    )


def _prepare_refine_item_worker(task: RefineBatchTask, start_time: float) -> _PreparedRefineItem:
    return _prepare_refine_item(task, start_time=start_time, progress=None)


def _run_batched_ik(
    items: Sequence[_PreparedRefineItem],
    robot: str,
    *,
    backend_factory=None,
    progress: ProgressReporter | None,
):
    from retargeter.pipeline import validate_robot_choices

    first = items[0].task
    target_builder = IKTargetBuilder(first.scaler_config, first.target_config)
    robot_spec = _robot_spec_from_task(first)
    backend = backend_factory(robot_spec) if backend_factory is not None else None
    solver = NewtonIKRetargetSolver(first.newton_config, backend=backend, target_builder=target_builder)
    validate_robot_choices(robot, target_builder, solver)
    return BatchSequenceIKRetargetRunner(solver).run(
        [item.preprocess_result.motion for item in items],
        contact_results=[item.preprocess_result.contact for item in items],
        progress=progress,
    )


def _run_batched_refinement(
    items: Sequence[_PreparedRefineItem],
    retargeted,
    *,
    refinement_fk_factory=None,
    progress: ProgressReporter | None,
):
    first = items[0].task
    robot_spec = _robot_spec_from_task(first)
    torch_fk = refinement_fk_factory(robot_spec) if refinement_fk_factory is not None else TorchRobotFK(robot_spec)
    return run_refinement_batch(
        retargeted,
        [item.preprocess_result for item in items],
        robot_spec,
        torch_fk,
        config=_refinement_config_for_task(first),
        progress=progress,
    )


def _run_refinement_fallback(
    items: Sequence[_PreparedRefineItem],
    retargeted,
    *,
    refinement_fk_factory=None,
    progress: ProgressReporter | None,
) -> list[Any]:
    outputs: list[Any] = []
    for item, motion in zip(items, retargeted):
        try:
            robot_spec = _robot_spec_from_task(item.task)
            torch_fk = refinement_fk_factory(robot_spec) if refinement_fk_factory is not None else TorchRobotFK(robot_spec)
            outputs.append(
                run_refinement(
                    motion,
                    item.preprocess_result,
                    robot_spec,
                    torch_fk,
                    config=_refinement_config_for_task(item.task),
                    progress=progress,
                )
            )
        except Exception as exc:
            outputs.append(_failed_record(item.task, item.start_time, exc))
    return outputs


def _export_item(item: _PreparedRefineItem, retargeted_motion, final_motion) -> BatchItemRecord:
    from retargeter.pipeline import _clear_top_level_refine_final_outputs, _clear_top_level_refine_retargeted_outputs
    from retargeter.newton import export_retargeted_motion

    task = item.task
    robot_spec = _robot_spec_from_task(task)
    refinement_cfg = _refinement_config_for_task(task)
    quality_report = evaluate_refinement_quality(
        retargeted_motion,
        final_motion,
        robot_spec,
        config=refinement_cfg,
        contact_score=item.preprocess_result.contact,
    )
    final_motion.metadata.update(
        {
            "pipeline": "refine",
            "retargeted_motion_path": "retargeted_motion.npz" if task.export_retargeted else None,
            "retargeted_motion_exported": bool(task.export_retargeted),
            "refinement_config": copy.deepcopy(refinement_cfg),
            "refinement_quality_valid": bool(quality_report.valid),
            "rejected": bool((not quality_report.valid) and not task.allow_invalid),
            "training_motion": bool(quality_report.valid),
        }
    )
    output_dir = Path(task.output_dir)
    final_output_dir = output_dir if quality_report.valid or task.allow_invalid else output_dir / "rejected"
    paths = {
        "final_motion": final_output_dir / "final_motion.npz",
        "final_metadata": final_output_dir / "final_meta.yaml",
        "final_quality": final_output_dir / "final_quality.json",
    }
    if task.export_retargeted:
        paths.update(
            {
                "retargeted_motion": output_dir / "retargeted_motion.npz",
                "retargeted_metadata": output_dir / "retargeted_meta.yaml",
                "retargeted_quality": output_dir / "retargeted_quality.json",
            }
        )
    if item.human_path is not None:
        paths["human"] = item.human_path

    if not task.export_retargeted:
        _clear_top_level_refine_retargeted_outputs(output_dir)
    if final_output_dir != output_dir:
        _clear_top_level_refine_final_outputs(output_dir)
    if task.export_retargeted:
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
    runtime = time.perf_counter() - item.start_time
    valid = bool(quality_report.valid)
    return BatchItemRecord(
        input=str(task.input_path),
        output_dir=str(task.output_dir),
        status="success" if valid else "invalid",
        frame_count=final_motion.num_frames(),
        fps=float(final_motion.fps),
        runtime_sec=runtime,
        quality_valid=valid,
        quality_summary=_quality_summary_from_report(quality_report),
        paths={key: str(value) for key, value in paths.items()},
    )


def _retargeted_metadata_for_item(item: _PreparedRefineItem, *, robot: str) -> dict[str, Any]:
    from retargeter.pipeline import PipelineConfigPaths, _retargeted_metadata

    task = item.task
    return _retargeted_metadata(
        source_metadata=item.source_metadata,
        robot=robot,
        config_paths=PipelineConfigPaths(
            preprocess_config=Path(task.preprocess_config),
            scaler_config=Path(task.scaler_config),
            target_config=Path(task.target_config),
            newton_config=Path(task.newton_config),
            robot_spec=Path(task.newton_config),
        ),
        preprocess_result=item.preprocess_result,
        pipeline="refine_retarget",
    )


def _group_by_refinement_config(items: Sequence[_PreparedRefineItem]) -> list[list[_PreparedRefineItem]]:
    groups: dict[str, list[_PreparedRefineItem]] = {}
    for item in items:
        key = json.dumps(_refinement_config_for_task(item.task), sort_keys=True, default=str)
        groups.setdefault(key, []).append(item)
    return list(groups.values())


def _robot_spec_from_task(task: RefineBatchTask):
    from retargeter.pipeline import _robot_spec_from_newton_config

    return _robot_spec_from_newton_config(Path(task.newton_config))


def _initial_manifest(
    manifest_path: Path,
    tasks: list[RefineBatchTask],
    *,
    robot: str,
    allow_invalid: bool,
    resume: bool,
    skip_existing: bool,
    overwrite: bool,
) -> BatchManifest:
    existing = load_manifest(manifest_path) if resume and manifest_path.exists() and not overwrite else None
    existing_by_key = {_record_key(item): item for item in (existing.items if existing is not None else [])}
    items: list[BatchItemRecord] = []
    for task in tasks:
        key = _task_key(task)
        existing_item = existing_by_key.get(key)
        if existing_item is not None and existing_item.status in {"success", "invalid", "skipped"}:
            items.append(existing_item)
            continue
        if skip_existing and not overwrite and _task_outputs_exist(task):
            from .worker import _record_from_existing_outputs

            items.append(_record_from_existing_outputs(task, status="skipped", runtime_sec=0.0))
            continue
        items.append(_pending_record(task))
    return BatchManifest(robot=robot, items=items, allow_invalid=allow_invalid)


def _failed_record(task: RefineBatchTask, start_time: float, exc: Exception) -> BatchItemRecord:
    return BatchItemRecord(
        input=str(task.input_path),
        output_dir=str(task.output_dir),
        status="failed",
        runtime_sec=time.perf_counter() - start_time,
        paths=_existing_standard_paths(Path(task.output_dir)),
        error_type=type(exc).__name__,
        error=str(exc),
    )


def _pending_record(task: RefineBatchTask) -> BatchItemRecord:
    return BatchItemRecord(input=str(task.input_path), output_dir=str(task.output_dir), status="pending")


def _running_record(task: RefineBatchTask) -> BatchItemRecord:
    return BatchItemRecord(input=str(task.input_path), output_dir=str(task.output_dir), status="running")


def _skipped_record(task: RefineBatchTask, *, reason: str) -> BatchItemRecord:
    return BatchItemRecord(
        input=str(task.input_path),
        output_dir=str(task.output_dir),
        status="skipped",
        error_type="Skipped",
        error=reason,
    )


def _record_for_task(manifest: BatchManifest, task: RefineBatchTask) -> BatchItemRecord:
    key = _task_key(task)
    for item in manifest.items:
        if _record_key(item) == key:
            return item
    raise KeyError(key)


def _task_outputs_exist(task: RefineBatchTask) -> bool:
    output_dir = Path(task.output_dir)
    return (output_dir / "final_motion.npz").exists() and (output_dir / "final_quality.json").exists()


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


def _native_progress_postfix(
    manifest: BatchManifest,
    *,
    batch_size: int,
    preprocess_workers: int | None = None,
    current: Sequence[RefineBatchTask] | None = None,
) -> dict[str, int | str]:
    summary = summarize(manifest)
    postfix: dict[str, int | str] = {
        "ok": summary.get("success", 0),
        "invalid": summary.get("invalid", 0),
        "failed": summary.get("failed", 0),
        "skipped": summary.get("skipped", 0),
        "batch": int(batch_size),
    }
    if preprocess_workers is not None:
        postfix["prep"] = int(preprocess_workers)
    if current:
        postfix["item"] = Path(str(current[0].input_path)).stem if str(current[0].input_path).lower() != "mock" else "mock"
    return postfix


def _task_key(task: RefineBatchTask) -> tuple[str, str]:
    return str(task.input_path), str(task.output_dir)


def _record_key(item: BatchItemRecord) -> tuple[str, str]:
    return item.input, item.output_dir
