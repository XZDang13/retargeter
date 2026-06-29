from __future__ import annotations

from pathlib import Path

from retargeter.batch.discovery import discover_inputs


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
