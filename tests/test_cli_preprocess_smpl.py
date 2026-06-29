from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from conftest import make_canonical_motion
from retargeter.cli import preprocess_smpl
from retargeter.preprocess import MotionPreprocessor, SMPLPreprocessOutput


def test_preprocess_smpl_cli_exports_human_motion_and_metadata(tmp_path: Path):
    input_path = tmp_path / "motion.npz"
    input_path.write_bytes(b"unused")
    output_dir = tmp_path / "preprocessed"

    exit_code = preprocess_smpl.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--model-type",
            "smplx",
            "--target-fps",
            "30",
        ],
        preprocess_runner=_fake_preprocess_runner,
    )

    assert exit_code == 0
    human_path = output_dir / "human.npz"
    meta_path = output_dir / "meta.yaml"
    assert human_path.exists()
    assert meta_path.exists()

    with np.load(human_path, allow_pickle=False) as data:
        assert data["body_pos_w"].shape[:2] == (3, 21)
        assert "vertices_w" in data
        assert "mesh_faces" in data
        assert "contact_score_left_foot" in data

    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert meta["input"] == str(input_path)
    assert meta["frame_count"] == 3
    assert meta["vertices_exported"] is True
    assert meta["source"]["model_type"] == "smplx"
    assert meta["source"]["return_vertices"] is True
    assert meta["source"]["target_fps"] == 30.0
    assert meta["preprocess_metadata"]["contact_available"] is True


def test_preprocess_smpl_cli_can_skip_vertices(tmp_path: Path):
    input_path = tmp_path / "motion.npz"
    input_path.write_bytes(b"unused")
    output_dir = tmp_path / "preprocessed"

    result = preprocess_smpl.run_preprocess_cli(
        preprocess_smpl._build_parser().parse_args(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_dir),
                "--no-vertices",
            ]
        ),
        preprocess_runner=_fake_preprocess_runner,
    )

    with np.load(result["motion_path"], allow_pickle=False) as data:
        assert "body_pos_w" in data
        assert "vertices_w" not in data
        assert "mesh_faces" not in data

    meta = yaml.safe_load(result["metadata_path"].read_text(encoding="utf-8"))
    assert meta["vertices_exported"] is False
    assert meta["source"]["return_vertices"] is False


def _fake_preprocess_runner(input_path, preprocess_config, **kwargs) -> SMPLPreprocessOutput:
    motion = make_canonical_motion(
        num_frames=3,
        fps=float(kwargs.get("target_fps") or 30.0),
        include_vertices=bool(kwargs["return_vertices"]),
    )
    if motion.vertices_w is not None:
        motion.mesh_faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    preprocess_result = MotionPreprocessor(preprocess_config).process(motion)
    return SMPLPreprocessOutput(
        canonical_motion=motion,
        preprocess_result=preprocess_result,
        source_metadata={
            "input": str(input_path),
            "mock_mode": False,
            "model_type": kwargs.get("model_type") or "smplx",
            "gender": kwargs.get("gender") or "neutral",
            "smpl_model_dir": str(kwargs["smpl_model_dir"]),
            "smpl_fk_applied": True,
            "return_vertices": bool(kwargs["return_vertices"]),
            "target_fps": kwargs.get("target_fps"),
        },
    )
