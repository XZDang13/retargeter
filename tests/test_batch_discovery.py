from __future__ import annotations

from pathlib import Path

import numpy as np

from retargeter.batch.discovery import discover_inputs, filter_motion_inputs, inspect_motion_input


def test_discover_inputs_patterns_recursive_excludes_and_dedupe(tmp_path: Path):
    root = tmp_path / "data"
    nested = root / "nested"
    nested.mkdir(parents=True)
    walk = root / "walk.npz"
    run = nested / "run.npy"
    skip = nested / "skip.npz"
    text = nested / "note.txt"
    for path in (walk, run, skip, text):
        path.write_text("x", encoding="utf-8")

    result = discover_inputs(
        inputs=[str(walk), str(walk), "mock", "mock"],
        input_dir=root,
        patterns=["*.npz", "*.npy"],
        recursive=True,
        exclude_patterns=["skip.npz", "nested/skip.npz"],
    )

    assert result.count("mock") == 2
    paths = [Path(item) for item in result if item != "mock"]
    assert paths == [walk.resolve(), run.resolve()]


def test_discover_inputs_input_list_preserves_order_and_comments(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npy"
    first.touch()
    second.touch()
    input_list = tmp_path / "inputs.txt"
    input_list.write_text(
        f"\n# comment\n{second}\nmock\n{first}\n{second}\n",
        encoding="utf-8",
    )

    result = discover_inputs(
        inputs=None,
        input_dir=None,
        patterns=["*.npz", "*.npy"],
        recursive=False,
        input_list=input_list,
    )

    assert result == [second.resolve(), "mock", first.resolve()]


def test_discover_inputs_nonrecursive_only_matches_top_level(tmp_path: Path):
    root = tmp_path / "data"
    nested = root / "nested"
    nested.mkdir(parents=True)
    top = root / "top.npz"
    child = nested / "child.npz"
    top.touch()
    child.touch()

    result = discover_inputs(
        inputs=None,
        input_dir=root,
        patterns=["*.npz"],
        recursive=False,
    )

    assert result == [top.resolve()]


def test_inspect_motion_input_rejects_neutral_template_and_accepts_motion(tmp_path: Path):
    neutral = tmp_path / "neutral_stagei.npz"
    motion = tmp_path / "clip_stageii.npz"
    short = tmp_path / "short.npz"
    phuma = tmp_path / "clip.npy"
    np.savez_compressed(
        neutral,
        gender=np.asarray("neutral"),
        surface_model_type=np.asarray("smplx"),
        markers_latent=np.zeros((41, 3)),
        latent_labels=np.asarray(["marker"] * 41),
        betas=np.zeros(16),
    )
    np.savez_compressed(
        motion,
        trans=np.zeros((10, 3)),
        poses=np.zeros((10, 165)),
        mocap_frame_rate=np.asarray(120.0),
    )
    np.savez_compressed(
        short,
        trans=np.zeros((1, 3)),
        poses=np.zeros((1, 165)),
        mocap_frame_rate=np.asarray(120.0),
    )
    np.save(phuma, np.zeros((5, 69)))

    neutral_info = inspect_motion_input(neutral)
    motion_info = inspect_motion_input(motion)
    short_info = inspect_motion_input(short)
    phuma_info = inspect_motion_input(phuma)

    assert neutral_info.is_motion is False
    assert neutral_info.reason == "missing_translation"
    assert motion_info.is_motion is True
    assert motion_info.frame_count == 10
    assert short_info.is_motion is False
    assert short_info.reason == "too_few_frames"
    assert phuma_info.is_motion is True
    assert phuma_info.frame_count == 5


def test_filter_motion_inputs_preserves_mock_and_filters_non_motion(tmp_path: Path):
    neutral = tmp_path / "neutral_stagei.npz"
    motion = tmp_path / "clip_stageii.npz"
    np.savez_compressed(neutral, betas=np.zeros(16), markers_latent=np.zeros((41, 3)))
    np.savez_compressed(motion, trans=np.zeros((2, 3)), root_orient=np.zeros((2, 3)), pose_body=np.zeros((2, 63)))

    assert filter_motion_inputs([neutral.resolve(), "mock", motion.resolve()]) == ["mock", motion.resolve()]
