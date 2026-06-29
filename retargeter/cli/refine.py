from __future__ import annotations

import argparse
from pathlib import Path

from retargeter.pipeline import RefinePipeline, load_refinement_config_file, refinement_config_with_overrides


def main(argv: list[str] | None = None, *, backend_factory=None, refinement_fk_factory=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = refinement_config_with_overrides(
        load_refinement_config_file(args.refinement_config),
        iterations=args.refinement_iterations,
        lr=args.refinement_lr,
        log_interval=args.refinement_log_interval,
        max_root_delta=args.refinement_max_root_delta,
        max_joint_delta=args.refinement_max_joint_delta,
        device=args.refinement_device,
        dtype=args.refinement_dtype,
        lbfgs_enabled=None if args.refinement_lbfgs is None else bool(int(args.refinement_lbfgs)),
    )
    pipeline = RefinePipeline(
        robot=args.robot,
        preprocess_config=args.preprocess_config,
        scaler_config=args.scaler_config,
        target_config=args.target_config,
        newton_config=args.newton_config,
        backend_factory=backend_factory,
        refinement_fk_factory=refinement_fk_factory,
    )
    if args.inputs is not None or args.input_dir is not None:
        input_paths = _batch_inputs_from_args(args)
        if not input_paths:
            parser.error("batch mode did not find any inputs")
        batch_result = pipeline.run_batch(
            input_paths=input_paths,
            output_dir=args.output,
            model_type=args.model_type,
            fps=args.fps,
            target_fps=args.target_fps,
            gender=args.gender,
            smpl_model_dir=args.smpl_model_dir,
            device=args.device,
            mock_frames=args.mock_frames,
            return_vertices=not args.no_vertices,
            export_human=not args.no_human_output,
            refinement_config=config,
            allow_invalid=bool(args.allow_invalid),
            fail_fast=bool(args.fail_fast),
        )
        print(batch_result.manifest_path)
        return 0 if batch_result.failure_count == 0 else 1

    result = pipeline.run(
        input_path=args.input,
        output_dir=args.output,
        model_type=args.model_type,
        fps=args.fps,
        target_fps=args.target_fps,
        gender=args.gender,
        smpl_model_dir=args.smpl_model_dir,
        device=args.device,
        mock_frames=args.mock_frames,
        return_vertices=not args.no_vertices,
        export_human=not args.no_human_output,
        refinement_config=config,
        allow_invalid=bool(args.allow_invalid),
    )
    for key in ("retargeted_motion", "retargeted_metadata", "retargeted_quality", "final_motion", "final_metadata", "final_quality", "human"):
        path = result.paths.get(key)
        if path is not None:
            print(path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline IK retargeting + refinement for training data.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Input SMPL/SMPL-X .npz/.npy path, or 'mock'.")
    input_group.add_argument("--inputs", nargs="+", help="Batch input SMPL/SMPL-X .npz/.npy paths, or 'mock'.")
    input_group.add_argument("--input-dir", type=Path, default=None, help="Batch input directory.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="unitree_g1_29")
    parser.add_argument("--model-type", choices=["smpl", "smplx"], default=None)
    parser.add_argument("--smpl-model-dir", type=Path, default=Path("assets/body_models"))
    parser.add_argument("--fps", type=float, default=None, help="Input/source FPS override.")
    parser.add_argument("--target-fps", type=float, default=None, help="Optional output FPS after resampling.")
    parser.add_argument("--gender", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mock-frames", type=int, default=120)
    parser.add_argument("--no-vertices", action="store_true")
    parser.add_argument("--no-human-output", action="store_true", help="Skip human.npz export.")
    parser.add_argument("--allow-invalid", action="store_true", help="Allow invalid refinement quality reports.")
    parser.add_argument("--preprocess-config", type=Path, default=None)
    parser.add_argument("--scaler-config", type=Path, default=None)
    parser.add_argument("--target-config", type=Path, default=None)
    parser.add_argument("--newton-config", type=Path, default=None)
    parser.add_argument("--refinement-config", type=Path, default=None)
    parser.add_argument("--refinement-iterations", type=int, default=None)
    parser.add_argument("--refinement-lr", type=float, default=None)
    parser.add_argument("--refinement-log-interval", type=int, default=None)
    parser.add_argument("--refinement-max-root-delta", type=float, default=None)
    parser.add_argument("--refinement-max-joint-delta", type=float, default=None)
    parser.add_argument("--refinement-device", default=None)
    parser.add_argument("--refinement-dtype", choices=["float32", "float64"], default=None)
    parser.add_argument("--refinement-lbfgs", type=int, choices=[0, 1], default=None)
    parser.add_argument("--input-pattern", action="append", default=None, help="Batch --input-dir glob pattern. Repeatable.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover --input-dir files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop a batch run after the first failed item.")
    return parser


def _batch_inputs_from_args(args: argparse.Namespace) -> list[Path | str]:
    if args.inputs is not None:
        return list(args.inputs)
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Batch input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Batch input path is not a directory: {input_dir}")

    patterns = args.input_pattern or ["*.npz", "*.npy"]
    discovered: list[Path] = []
    for pattern in patterns:
        matches = input_dir.rglob(pattern) if args.recursive else input_dir.glob(pattern)
        discovered.extend(path for path in matches if path.is_file())

    unique = {str(path): path for path in discovered}
    return [unique[key] for key in sorted(unique)]


if __name__ == "__main__":
    raise SystemExit(main())
