from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


BATCH_STATUSES = {"pending", "running", "success", "invalid", "failed", "skipped"}
SUMMARY_CSV_COLUMNS = [
    "input",
    "output_dir",
    "status",
    "frame_count",
    "fps",
    "runtime_sec",
    "quality_valid",
    "error_type",
    "error",
]
PASS_REJECT_CSV_COLUMNS = [
    "input",
    "output_dir",
    "decision",
    "status",
    "quality_valid",
    "failures",
    "frame_count",
    "fps",
    "error_type",
    "error",
]


@dataclass
class BatchItemRecord:
    input: str
    output_dir: str
    status: str = "pending"
    frame_count: int | None = None
    fps: float | None = None
    runtime_sec: float | None = None
    quality_valid: bool | None = None
    quality_summary: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in BATCH_STATUSES:
            raise ValueError(f"Unsupported batch item status {self.status!r}.")


@dataclass
class BatchManifest:
    pipeline: str = "refine_batch"
    robot: str = "unitree_g1_29"
    items: list[BatchItemRecord] = field(default_factory=list)
    allow_invalid: bool = False


def load_manifest(path: Path | str) -> BatchManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [BatchItemRecord(**item) for item in payload.get("items", [])]
    return BatchManifest(
        pipeline=str(payload.get("pipeline", "refine_batch")),
        robot=str(payload.get("robot", "unitree_g1_29")),
        items=items,
        allow_invalid=bool(payload.get("allow_invalid", False)),
    )


def save_manifest(path: Path | str, manifest: BatchManifest) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _manifest_to_dict(manifest)
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, output)
    return output


def update_item(manifest: BatchManifest, item: BatchItemRecord) -> BatchManifest:
    key = _item_key(item)
    for idx, existing in enumerate(manifest.items):
        if _item_key(existing) == key:
            manifest.items[idx] = item
            return manifest
    manifest.items.append(item)
    return manifest


def summarize(manifest: BatchManifest) -> dict[str, int]:
    counts = {status: 0 for status in sorted(BATCH_STATUSES)}
    for item in manifest.items:
        counts[item.status] = counts.get(item.status, 0) + 1
    blocking_failure_count = counts.get("failed", 0)
    summary = dict(counts)
    summary.update(
        {
            "input_count": len(manifest.items),
            "success_count": counts.get("success", 0),
            "failure_count": blocking_failure_count,
            "blocking_failure_count": blocking_failure_count,
        }
    )
    return summary


def write_summary_csv(path: Path | str, manifest: BatchManifest) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for item in manifest.items:
            writer.writerow({column: _csv_value(getattr(item, column)) for column in SUMMARY_CSV_COLUMNS})
    return output


def write_pass_reject_csv(path: Path | str, manifest: BatchManifest) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PASS_REJECT_CSV_COLUMNS)
        writer.writeheader()
        for item in manifest.items:
            writer.writerow(
                {
                    "input": item.input,
                    "output_dir": item.output_dir,
                    "decision": _pass_reject_decision(item),
                    "status": item.status,
                    "quality_valid": _csv_value(item.quality_valid),
                    "failures": ";".join(str(value) for value in _quality_failures(item)),
                    "frame_count": _csv_value(item.frame_count),
                    "fps": _csv_value(item.fps),
                    "error_type": _csv_value(item.error_type),
                    "error": _csv_value(item.error),
                }
            )
    return output


def _manifest_to_dict(manifest: BatchManifest) -> dict[str, Any]:
    summary = summarize(manifest)
    return {
        "pipeline": manifest.pipeline,
        "robot": manifest.robot,
        "allow_invalid": bool(manifest.allow_invalid),
        "input_count": summary["input_count"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "blocking_failure_count": summary["blocking_failure_count"],
        "status_counts": {status: summary[status] for status in sorted(BATCH_STATUSES)},
        "items": [_item_to_dict(item) for item in manifest.items],
    }


def _item_to_dict(item: BatchItemRecord) -> dict[str, Any]:
    return {
        "input": item.input,
        "output_dir": item.output_dir,
        "status": item.status,
        "frame_count": item.frame_count,
        "fps": item.fps,
        "runtime_sec": item.runtime_sec,
        "quality_valid": item.quality_valid,
        "quality_summary": item.quality_summary,
        "paths": item.paths,
        "error_type": item.error_type,
        "error": item.error,
    }


def _item_key(item: BatchItemRecord) -> tuple[str, str]:
    return item.input, item.output_dir


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _pass_reject_decision(item: BatchItemRecord) -> str:
    if item.status == "success":
        return "pass"
    if item.status in {"invalid", "failed"}:
        return "reject"
    if item.status == "skipped":
        if item.quality_valid is True:
            return "pass"
        if item.quality_valid is False:
            return "reject"
        return "skipped"
    return item.status


def _quality_failures(item: BatchItemRecord) -> list[Any]:
    summary = item.quality_summary
    if not isinstance(summary, dict):
        return []
    failures = summary.get("failures", [])
    return list(failures) if isinstance(failures, list) else []
