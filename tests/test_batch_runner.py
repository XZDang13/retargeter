from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from retargeter.batch.manifest import BatchItemRecord, BatchManifest, save_manifest, write_summary_csv
from retargeter.batch.runner import BatchRefineRunner, build_refine_batch_tasks
from retargeter.batch.worker import RefineBatchTask


class RecordingProgressBar:
    def __init__(self, *, total, desc, unit):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.updates: list[int] = []
        self.postfixes: list[dict] = []

    def update(self, n: int = 1) -> None:
        self.updates.append(int(n))

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs) -> None:
        values = dict(ordered_dict or {})
        values.update(kwargs)
        self.postfixes.append(values)


class RecordingProgress:
    enabled = True
    forced = False

    def __init__(self):
        self.stages: list[str] = []
        self.bars: list[RecordingProgressBar] = []

    def stage(self, message: str) -> None:
        self.stages.append(message)

    def child(self, *, position_offset: int = 1):
        return self

    @contextmanager
    def bar(self, *, total, desc, unit="it", leave=False):
        bar = RecordingProgressBar(total=total, desc=desc, unit=unit)
        self.bars.append(bar)
        yield bar


def fake_batch_worker(task: RefineBatchTask) -> BatchItemRecord:
    Path(task.output_dir).mkdir(parents=True, exist_ok=True)
    if "fail" in str(task.input_path):
        return BatchItemRecord(
            input=str(task.input_path),
            output_dir=str(task.output_dir),
            status="failed",
            error_type="RuntimeError",
            error="planned failure",
        )
    return BatchItemRecord(
        input=str(task.input_path),
        output_dir=str(task.output_dir),
        status="success",
        frame_count=2,
        fps=30.0,
        runtime_sec=0.01,
        quality_valid=True,
        paths={"final_motion": str(Path(task.output_dir) / "final_motion.npz")},
    )


def test_build_refine_batch_tasks_default_and_preserve_tree(tmp_path: Path):
    input_dir = tmp_path / "input"
    nested = input_dir / "a"
    nested.mkdir(parents=True)
    walk = nested / "walk.npz"
    walk.touch()

    default_tasks = build_refine_batch_tasks(["mock", "mock"], tmp_path / "out")
    assert [task.output_dir.name for task in default_tasks] == ["mock", "mock__2"]

    tree_tasks = build_refine_batch_tasks(
        [walk.resolve(), walk.resolve()],
        tmp_path / "out_tree",
        input_dir=input_dir,
        preserve_tree=True,
    )
    assert tree_tasks[0].output_dir == tmp_path / "out_tree" / "a" / "walk"
    assert tree_tasks[1].output_dir == tmp_path / "out_tree" / "a" / "walk__2"


def test_build_refine_batch_tasks_gpu_assignment_respects_explicit_refinement_device(tmp_path: Path):
    worker_device_tasks = build_refine_batch_tasks(
        ["a", "b"],
        tmp_path / "out_worker",
        worker_devices=["cuda:0", "cuda:1"],
    )
    assert [task.device for task in worker_device_tasks] == ["cpu", "cpu"]
    assert [task.refinement_device for task in worker_device_tasks] == ["cuda:0", "cuda:1"]

    tasks = build_refine_batch_tasks(
        ["a", "b"],
        tmp_path / "out",
        refinement_config={"refiner": {"device": "cpu"}},
        worker_devices=["cuda:0", "cuda:1"],
    )
    assert [task.device for task in tasks] == ["cpu", "cpu"]
    assert [task.refinement_device for task in tasks] == [None, None]

    explicit_tasks = build_refine_batch_tasks(
        ["a"],
        tmp_path / "out_explicit",
        refinement_device="cuda:2",
        refinement_config={"refiner": {"device": "cpu"}},
        worker_devices=["cuda:0"],
    )
    assert explicit_tasks[0].device == "cpu"
    assert explicit_tasks[0].refinement_device == "cuda:2"


def test_batch_refine_runner_sequential_failure_continue_and_incremental_manifest(tmp_path: Path, monkeypatch):
    writes: list[list[str]] = []
    real_save_manifest = save_manifest

    def recording_save_manifest(path, manifest):
        writes.append([item.status for item in manifest.items])
        return real_save_manifest(path, manifest)

    monkeypatch.setattr("retargeter.batch.runner.save_manifest", recording_save_manifest)
    tasks = build_refine_batch_tasks(["ok", "fail", "ok2"], tmp_path / "out")
    manifest = BatchRefineRunner(
        manifest_path=tmp_path / "out" / "batch_manifest.json",
        task_processor=fake_batch_worker,
    ).run(tasks, workers=1, fail_fast=False)

    assert [item.status for item in manifest.items] == ["success", "failed", "success"]
    assert any("running" in statuses for statuses in writes)
    payload = json.loads((tmp_path / "out" / "batch_manifest.json").read_text(encoding="utf-8"))
    assert payload["failure_count"] == 1


def test_batch_refine_runner_progress_updates_for_sequential_fail_fast(tmp_path: Path):
    progress = RecordingProgress()
    tasks = build_refine_batch_tasks(["fail", "ok", "ok2"], tmp_path / "out")
    manifest = BatchRefineRunner(
        manifest_path=tmp_path / "out" / "batch_manifest.json",
        task_processor=fake_batch_worker,
    ).run(tasks, workers=1, fail_fast=True, progress=progress)

    assert [item.status for item in manifest.items] == ["failed", "skipped", "skipped"]
    assert len(progress.bars) == 1
    bar = progress.bars[0]
    assert bar.desc == "Batch refine"
    assert bar.total == 3
    assert sum(bar.updates) == 3
    assert bar.postfixes[-1]["failed"] == 1
    assert bar.postfixes[-1]["skipped"] == 2


def test_batch_refine_runner_fail_fast_skips_remaining(tmp_path: Path):
    tasks = build_refine_batch_tasks(["fail", "ok", "ok2"], tmp_path / "out")
    manifest = BatchRefineRunner(
        manifest_path=tmp_path / "out" / "batch_manifest.json",
        task_processor=fake_batch_worker,
    ).run(tasks, workers=1, fail_fast=True)

    assert [item.status for item in manifest.items] == ["failed", "skipped", "skipped"]


def test_batch_refine_runner_resume_and_skip_existing(tmp_path: Path):
    output = tmp_path / "out"
    tasks = build_refine_batch_tasks(["done", "existing"], output)
    save_manifest(
        output / "batch_manifest.json",
        BatchManifest(items=[BatchItemRecord(input=str(tasks[0].input_path), output_dir=str(tasks[0].output_dir), status="success")]),
    )
    _write_existing_outputs(tasks[1].output_dir)

    manifest = BatchRefineRunner(
        manifest_path=output / "batch_manifest.json",
        task_processor=fake_batch_worker,
    ).run(tasks, workers=1, resume=True, skip_existing=True)

    assert [item.status for item in manifest.items] == ["success", "skipped"]
    assert manifest.items[1].frame_count == 4
    assert manifest.items[1].fps == 60.0


def test_batch_refine_runner_dry_run_and_summary_csv(tmp_path: Path):
    tasks = build_refine_batch_tasks(["mock"], tmp_path / "out")
    runner = BatchRefineRunner(manifest_path=tmp_path / "out" / "batch_manifest.json", task_processor=fake_batch_worker)
    manifest = runner.run(tasks, workers=1, dry_run=True)
    csv_path = write_summary_csv(tmp_path / "out" / "summary.csv", manifest)

    assert [item.status for item in manifest.items] == ["pending"]
    assert csv_path.exists()
    assert not (tmp_path / "out" / "mock").exists()


def test_batch_refine_runner_parallel_with_fake_worker(tmp_path: Path):
    tasks = build_refine_batch_tasks(["ok", "ok2"], tmp_path / "out")
    manifest = BatchRefineRunner(
        manifest_path=tmp_path / "out" / "batch_manifest.json",
        task_processor=fake_batch_worker,
    ).run(tasks, workers=2)

    assert sorted(item.status for item in manifest.items) == ["success", "success"]


def test_batch_refine_runner_progress_updates_for_parallel_parent_completions(tmp_path: Path):
    progress = RecordingProgress()
    tasks = build_refine_batch_tasks(["ok", "ok2"], tmp_path / "out")
    manifest = BatchRefineRunner(
        manifest_path=tmp_path / "out" / "batch_manifest.json",
        task_processor=fake_batch_worker,
    ).run(tasks, workers=2, progress=progress)

    assert sorted(item.status for item in manifest.items) == ["success", "success"]
    assert len(progress.bars) == 1
    bar = progress.bars[0]
    assert bar.total == 2
    assert sum(bar.updates) == 2
    assert bar.postfixes[-1]["ok"] == 2
    assert bar.postfixes[-1]["workers"] == 2


def _write_existing_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "final_motion.npz", fps=np.asarray(60.0), joint_pos=np.zeros((4, 2)))
    (output_dir / "final_quality.json").write_text(
        json.dumps(
            {
                "frame_count": 4,
                "valid": True,
                "quality_report": {"valid": True, "metrics": {"fps": 60.0, "num_frames": 4}, "failures": []},
            }
        ),
        encoding="utf-8",
    )
