from __future__ import annotations

import json
from pathlib import Path

import pytest

from retargeter.batch.gpu_pool import assign_device, parse_gpu_ids
from retargeter.batch.manifest import (
    BatchItemRecord,
    BatchManifest,
    load_manifest,
    save_manifest,
    summarize,
    update_item,
    write_summary_csv,
)


def test_manifest_round_trip_update_summary_and_csv(tmp_path: Path):
    manifest = BatchManifest(
        robot="unitree_g1_29",
        items=[
            BatchItemRecord(input="a", output_dir="out/a", status="pending"),
            BatchItemRecord(input="b", output_dir="out/b", status="failed", error_type="ValueError", error="bad"),
        ],
    )
    update_item(manifest, BatchItemRecord(input="a", output_dir="out/a", status="success", frame_count=3, fps=30.0))

    path = tmp_path / "batch_manifest.json"
    save_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["success_count"] == 1
    assert payload["failure_count"] == 1

    loaded = load_manifest(path)
    assert [item.status for item in loaded.items] == ["success", "failed"]
    assert summarize(loaded)["blocking_failure_count"] == 1

    csv_path = write_summary_csv(tmp_path / "summary.csv", loaded)
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "input,output_dir,status,frame_count,fps,runtime_sec,quality_valid,error_type,error"
    assert "a,out/a,success,3,30.0" in lines[1]


def test_gpu_pool_parsing_and_assignment():
    assert parse_gpu_ids(None) == []
    assert parse_gpu_ids("") == []
    assert parse_gpu_ids("cpu") == []
    assert parse_gpu_ids("0,1,2") == [0, 1, 2]
    assert [assign_device(i, [0, 1], 2) for i in range(5)] == ["cuda:0", "cuda:0", "cuda:1", "cuda:1", "cuda:0"]
    assert assign_device(3, [], 2) == "cpu"

    with pytest.raises(ValueError):
        parse_gpu_ids("0,nope")
    with pytest.raises(ValueError):
        assign_device(0, [0], 0)
