from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from retargeter.preprocess import SMPLPreprocessOutput, load_preprocess_config, run_smpl_preprocess
from retargeter.preprocess.config import DEFAULT_PREPROCESS_CONFIG_PATH
from retargeter.visualize import export_canonical_human_motion_npz, plot_contact_scores, plot_foot_height_and_speed


PreprocessRunner = Callable[..., SMPLPreprocessOutput]


def main(argv: list[str] | None = None, *, preprocess_runner: PreprocessRunner = run_smpl_preprocess) -> int:
    args = _build_parser().parse_args(argv)
    result = run_preprocess_cli(args, preprocess_runner=preprocess_runner)
    for key in ("motion_path", "metadata_path"):
        print(result[key])
    for path in result["diagnostic_paths"]:
        print(path)
    return 0


def run_preprocess_cli(
    args: argparse.Namespace,
    *,
    preprocess_runner: PreprocessRunner = run_smpl_preprocess,
) -> dict:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_config_path = Path(args.preprocess_config or DEFAULT_PREPROCESS_CONFIG_PATH)
    preprocess_config = load_preprocess_config(preprocess_config_path)
    preprocess_output = preprocess_runner(
        args.input,
        preprocess_config,
        model_type=args.model_type,
        fps=args.fps,
        gender=args.gender,
        smpl_model_dir=args.smpl_model_dir,
        device=args.device,
        return_vertices=not args.no_vertices,
        target_fps=args.target_fps,
    )

    motion_path = output_dir / "human.npz"
    export_canonical_human_motion_npz(
        preprocess_output.preprocess_result.motion,
        motion_path,
        preprocess_result=preprocess_output.preprocess_result,
        require_mesh=not args.no_vertices,
    )

    metadata_path = output_dir / "meta.yaml"
    _write_preprocess_metadata(
        metadata_path,
        preprocess_output,
        input_path=args.input,
        preprocess_config_path=preprocess_config_path,
        motion_path=motion_path,
        no_vertices=bool(args.no_vertices),
    )

    diagnostic_paths: list[Path] = []
    if bool(args.diagnostics) and preprocess_output.preprocess_result.contact is not None:
        diagnostic_paths = [
            plot_contact_scores(preprocess_output.preprocess_result, output_dir / "contact_scores.png"),
            plot_foot_height_and_speed(preprocess_output.preprocess_result, output_dir / "foot_height_speed.png"),
        ]

    return {
        "motion_path": motion_path,
        "metadata_path": metadata_path,
        "diagnostic_paths": diagnostic_paths,
        "canonical_motion": preprocess_output.canonical_motion,
        "preprocess_result": preprocess_output.preprocess_result,
        "source_metadata": preprocess_output.source_metadata,
    }


def _write_preprocess_metadata(
    path: Path,
    preprocess_output: SMPLPreprocessOutput,
    *,
    input_path: Path | str,
    preprocess_config_path: Path,
    motion_path: Path,
    no_vertices: bool,
) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write preprocess metadata.") from exc

    motion = preprocess_output.preprocess_result.motion
    payload = {
        "input": str(input_path),
        "motion_path": str(motion_path),
        "preprocess_config": str(preprocess_config_path),
        "fps": float(motion.fps),
        "frame_count": motion.num_frames(),
        "body_names": list(motion.body_names),
        "vertices_exported": motion.vertices_w is not None,
        "mesh_faces_exported": motion.mesh_faces is not None,
        "no_vertices": bool(no_vertices),
        "source": dict(preprocess_output.source_metadata),
        "preprocess_warnings": list(preprocess_output.preprocess_result.warnings),
        "preprocess_metadata": dict(preprocess_output.preprocess_result.metadata),
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess SMPL/SMPL-X motion into canonical human motion.")
    parser.add_argument("--input", type=Path, required=True, help="Input SMPL/SMPL-X .npz or PHUMA-style .npy path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for human.npz and meta.yaml.")
    parser.add_argument("--model-type", choices=["smpl", "smplx"], default=None)
    parser.add_argument("--smpl-model-dir", type=Path, default=Path("assets/body_models"))
    parser.add_argument("--preprocess-config", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Input/source FPS override.")
    parser.add_argument("--target-fps", type=float, default=None, help="Optional output FPS after SMPL parameter resampling.")
    parser.add_argument("--gender", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-vertices", action="store_true")
    parser.add_argument("--diagnostics", type=int, choices=[0, 1], default=0)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
