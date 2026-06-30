from __future__ import annotations

import concurrent.futures
import multiprocessing
import re
from pathlib import Path
from typing import Sequence

from retargeter.progress import ProgressReporter, get_progress

from .manifest import BatchItemRecord, BatchManifest, load_manifest, save_manifest, summarize, update_item
from .worker import RefineBatchTask, TaskProcessor, process_refine_batch_task


class BatchRefineRunner:
    def __init__(
        self,
        *,
        manifest_path: Path | str,
        robot: str = "unitree_g1_29",
        allow_invalid: bool = False,
        task_processor: TaskProcessor = process_refine_batch_task,
    ):
        self.manifest_path = Path(manifest_path)
        self.robot = robot
        self.allow_invalid = bool(allow_invalid)
        self.task_processor = task_processor

    def run(
        self,
        tasks: Sequence[RefineBatchTask],
        workers: int = 1,
        fail_fast: bool = False,
        resume: bool = False,
        skip_existing: bool = False,
        overwrite: bool = False,
        dry_run: bool = False,
        progress: ProgressReporter | None = None,
    ) -> BatchManifest:
        if workers <= 0:
            raise ValueError("workers must be positive.")
        reporter = get_progress(progress)
        task_list = list(tasks)
        manifest = self._initial_manifest(
            task_list,
            resume=resume,
            skip_existing=skip_existing,
            overwrite=overwrite,
        )
        save_manifest(self.manifest_path, manifest)
        if dry_run:
            reporter.stage(f"Batch dry run planned {len(task_list)} item(s)")
            return manifest

        runnable = [task for task in task_list if self._record_for_task(manifest, task).status == "pending"]
        if workers == 1:
            return self._run_sequential(manifest, runnable, fail_fast=fail_fast, progress=reporter)
        return self._run_parallel(manifest, runnable, workers=workers, fail_fast=fail_fast, progress=reporter)

    def _initial_manifest(
        self,
        tasks: list[RefineBatchTask],
        *,
        resume: bool,
        skip_existing: bool,
        overwrite: bool,
    ) -> BatchManifest:
        existing = load_manifest(self.manifest_path) if resume and self.manifest_path.exists() and not overwrite else None
        existing_by_key = {_record_key(item): item for item in (existing.items if existing is not None else [])}
        items: list[BatchItemRecord] = []
        for task in tasks:
            key = _task_key(task)
            existing_item = existing_by_key.get(key)
            if existing_item is not None and _resume_skips(existing_item, allow_invalid=self.allow_invalid):
                items.append(existing_item)
                continue
            if skip_existing and not overwrite and _task_outputs_exist(task):
                items.append(_skipped_existing_record(task))
                continue
            items.append(_pending_record(task))
        return BatchManifest(robot=self.robot, items=items, allow_invalid=self.allow_invalid)

    def _run_sequential(
        self,
        manifest: BatchManifest,
        tasks: list[RefineBatchTask],
        *,
        fail_fast: bool,
        progress: ProgressReporter,
    ) -> BatchManifest:
        with progress.bar(total=len(tasks), desc="Batch refine", unit="clip") as bar:
            bar.set_postfix(_progress_postfix(manifest, workers=1), refresh=False)
            for index, task in enumerate(tasks):
                update_item(manifest, _running_record(task))
                save_manifest(self.manifest_path, manifest)
                bar.set_postfix(_progress_postfix(manifest, workers=1, current=task), refresh=False)
                result = self.task_processor(task)
                update_item(manifest, result)
                save_manifest(self.manifest_path, manifest)
                bar.update(1)
                bar.set_postfix(_progress_postfix(manifest, workers=1), refresh=False)
                if fail_fast and _is_blocking_failure(result, allow_invalid=self.allow_invalid):
                    remaining = tasks[index + 1 :]
                    self._skip_remaining(manifest, remaining, reason="fail_fast")
                    save_manifest(self.manifest_path, manifest)
                    if remaining:
                        bar.update(len(remaining))
                        bar.set_postfix(_progress_postfix(manifest, workers=1), refresh=False)
                    break
        return manifest

    def _run_parallel(
        self,
        manifest: BatchManifest,
        tasks: list[RefineBatchTask],
        *,
        workers: int,
        fail_fast: bool,
        progress: ProgressReporter,
    ) -> BatchManifest:
        pending = list(tasks)
        running: dict[concurrent.futures.Future[BatchItemRecord], RefineBatchTask] = {}
        context = multiprocessing.get_context("spawn")
        with progress.bar(total=len(tasks), desc="Batch refine", unit="clip") as bar:
            bar.set_postfix(_progress_postfix(manifest, workers=workers), refresh=False)
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                self._submit_until_full(executor, pending, running, manifest, workers)
                bar.set_postfix(_progress_postfix(manifest, workers=workers), refresh=False)
                while running:
                    done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
                    stop = False
                    for future in done:
                        task = running.pop(future)
                        result = _future_result_or_failure(future, task)
                        update_item(manifest, result)
                        save_manifest(self.manifest_path, manifest)
                        bar.update(1)
                        bar.set_postfix(_progress_postfix(manifest, workers=workers), refresh=False)
                        if fail_fast and _is_blocking_failure(result, allow_invalid=self.allow_invalid):
                            stop = True
                            break
                    if stop:
                        remaining = len(running) + len(pending)
                        self._finish_after_parallel_fail_fast(manifest, running, pending)
                        save_manifest(self.manifest_path, manifest)
                        if remaining:
                            bar.update(remaining)
                            bar.set_postfix(_progress_postfix(manifest, workers=workers), refresh=False)
                        break
                    self._submit_until_full(executor, pending, running, manifest, workers)
                    bar.set_postfix(_progress_postfix(manifest, workers=workers), refresh=False)
        return manifest

    def _submit_until_full(
        self,
        executor: concurrent.futures.ProcessPoolExecutor,
        pending: list[RefineBatchTask],
        running: dict[concurrent.futures.Future[BatchItemRecord], RefineBatchTask],
        manifest: BatchManifest,
        workers: int,
    ) -> None:
        while pending and len(running) < workers:
            task = pending.pop(0)
            update_item(manifest, _running_record(task))
            save_manifest(self.manifest_path, manifest)
            running[executor.submit(self.task_processor, task)] = task

    def _finish_after_parallel_fail_fast(
        self,
        manifest: BatchManifest,
        running: dict[concurrent.futures.Future[BatchItemRecord], RefineBatchTask],
        pending: list[RefineBatchTask],
    ) -> None:
        still_running: dict[concurrent.futures.Future[BatchItemRecord], RefineBatchTask] = {}
        for future, task in list(running.items()):
            if future.cancel():
                update_item(manifest, _skipped_record(task, reason="fail_fast"))
            else:
                still_running[future] = task
        running.clear()
        for task in pending:
            update_item(manifest, _skipped_record(task, reason="fail_fast"))
        pending.clear()
        for future, task in still_running.items():
            result = _future_result_or_failure(future, task)
            update_item(manifest, result)

    def _skip_remaining(self, manifest: BatchManifest, tasks: Sequence[RefineBatchTask], *, reason: str) -> None:
        for task in tasks:
            update_item(manifest, _skipped_record(task, reason=reason))

    def _record_for_task(self, manifest: BatchManifest, task: RefineBatchTask) -> BatchItemRecord:
        key = _task_key(task)
        for item in manifest.items:
            if _record_key(item) == key:
                return item
        raise KeyError(key)


def build_refine_batch_tasks(
    input_paths: Sequence[Path | str],
    output_dir: Path | str,
    *,
    robot: str = "unitree_g1_29",
    model_type: str | None = None,
    fps: float | None = None,
    target_fps: float | None = None,
    gender: str | None = None,
    smpl_model_dir: Path | str = Path("assets/body_models"),
    device: str = "cpu",
    refinement_device: str | None = None,
    mock_frames: int = 120,
    return_vertices: bool = True,
    export_human: bool = True,
    export_retargeted: bool = False,
    allow_invalid: bool = False,
    preprocess_config: Path | None = None,
    scaler_config: Path | None = None,
    target_config: Path | None = None,
    newton_config: Path | None = None,
    refinement_config: dict | None = None,
    input_dir: Path | None = None,
    preserve_tree: bool = False,
    worker_devices: Sequence[str] | None = None,
) -> list[RefineBatchTask]:
    output_root = Path(output_dir)
    resolved_input_dir = Path(input_dir).expanduser().resolve(strict=False) if input_dir is not None else None
    used_names: dict[str, int] = {}
    config_has_refinement_device = _refinement_config_has_device(refinement_config)
    tasks: list[RefineBatchTask] = []
    for index, input_path in enumerate(input_paths):
        task_device = worker_devices[index % len(worker_devices)] if worker_devices else device
        task_refinement_device = refinement_device
        if task_refinement_device is None and worker_devices and not config_has_refinement_device:
            task_refinement_device = task_device
        tasks.append(
            RefineBatchTask(
                input_path=input_path,
                output_dir=_task_output_dir(
                    output_root,
                    input_path,
                    used_names,
                    input_dir=resolved_input_dir,
                    preserve_tree=preserve_tree,
                ),
                robot=robot,
                model_type=model_type,
                fps=fps,
                target_fps=target_fps,
                gender=gender,
                smpl_model_dir=smpl_model_dir,
                device=task_device,
                refinement_device=task_refinement_device,
                mock_frames=mock_frames,
                return_vertices=return_vertices,
                export_human=export_human,
                export_retargeted=export_retargeted,
                allow_invalid=allow_invalid,
                preprocess_config=preprocess_config,
                scaler_config=scaler_config,
                target_config=target_config,
                newton_config=newton_config,
                refinement_config=dict(refinement_config or {}),
            )
        )
    return tasks


def _task_output_dir(
    output_root: Path,
    input_path: Path | str,
    used_names: dict[str, int],
    *,
    input_dir: Path | None,
    preserve_tree: bool,
) -> Path:
    if preserve_tree and input_dir is not None and str(input_path).lower() != "mock":
        path = Path(input_path).expanduser().resolve(strict=False)
        try:
            rel = path.relative_to(input_dir).with_suffix("")
        except ValueError:
            rel = Path(_safe_batch_item_name(input_path))
        return _dedupe_output_dir(output_root, rel, used_names)
    return _dedupe_output_dir(output_root, Path(_safe_batch_item_name(input_path)), used_names)


def _dedupe_output_dir(output_root: Path, relative: Path, used_names: dict[str, int]) -> Path:
    key = relative.as_posix()
    next_count = used_names.get(key, 0) + 1
    used_names[key] = next_count
    if next_count == 1:
        return output_root / relative
    return output_root / relative.parent / f"{relative.name}__{next_count}"


def _safe_batch_item_name(input_path: Path | str) -> str:
    raw = str(input_path)
    if raw.lower() == "mock":
        stem = "mock"
    else:
        stem = Path(input_path).stem
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return safe or "item"


def _refinement_config_has_device(refinement_config: dict | None) -> bool:
    if not isinstance(refinement_config, dict):
        return False
    refiner = refinement_config.get("refiner")
    return isinstance(refiner, dict) and refiner.get("device") is not None


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


def _skipped_existing_record(task: RefineBatchTask) -> BatchItemRecord:
    from .worker import _record_from_existing_outputs

    return _record_from_existing_outputs(task, status="skipped", runtime_sec=0.0)


def _task_outputs_exist(task: RefineBatchTask) -> bool:
    output_dir = Path(task.output_dir)
    return (output_dir / "final_motion.npz").exists() and (output_dir / "final_quality.json").exists()


def _resume_skips(item: BatchItemRecord, *, allow_invalid: bool) -> bool:
    if item.status in {"success", "invalid", "skipped"}:
        return True
    return False


def _is_blocking_failure(item: BatchItemRecord, *, allow_invalid: bool) -> bool:
    return item.status == "failed"


def _future_result_or_failure(
    future: concurrent.futures.Future[BatchItemRecord],
    task: RefineBatchTask,
) -> BatchItemRecord:
    try:
        return future.result()
    except Exception as exc:
        return BatchItemRecord(
            input=str(task.input_path),
            output_dir=str(task.output_dir),
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )


def _progress_postfix(
    manifest: BatchManifest,
    *,
    workers: int,
    current: RefineBatchTask | None = None,
) -> dict[str, int | str]:
    summary = summarize(manifest)
    postfix: dict[str, int | str] = {
        "ok": summary.get("success", 0),
        "invalid": summary.get("invalid", 0),
        "failed": summary.get("failed", 0),
        "skipped": summary.get("skipped", 0),
        "workers": int(workers),
    }
    if current is not None:
        postfix["item"] = _safe_batch_item_name(current.input_path)
    return postfix


def _task_key(task: RefineBatchTask) -> tuple[str, str]:
    return str(task.input_path), str(task.output_dir)


def _record_key(item: BatchItemRecord) -> tuple[str, str]:
    return item.input, item.output_dir
