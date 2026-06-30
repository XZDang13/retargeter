from __future__ import annotations

import argparse
from pathlib import Path

from retargeter.batch import assign_device, discover_inputs, parse_gpu_ids
from retargeter.pipeline import RefinePipeline, load_refinement_config_file, refinement_config_with_overrides
from retargeter.progress import make_progress


def main(argv: list[str] | None = None, *, backend_factory=None, refinement_fk_factory=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_input_mode(parser, args)
    progress = make_progress(args.progress)
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
    if _is_batch_mode(args):
        if args.workers > 1 and (backend_factory is not None or refinement_fk_factory is not None):
            parser.error("Injected backend/refinement factories are only supported with --workers 1.")
        input_paths = _batch_inputs_from_args(args)
        if not input_paths:
            parser.error("batch mode did not find any inputs")
        gpu_ids = parse_gpu_ids(args.gpu_ids)
        assigned_devices = (
            [assign_device(index, gpu_ids, args.processes_per_gpu) for index in range(max(args.workers, 1))] if gpu_ids else None
        )
        worker_devices = None if args.native_batch else assigned_devices
        refinement_device = assigned_devices[0] if args.native_batch and assigned_devices else None
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
            export_retargeted=bool(args.save_retargeted),
            refinement_config=config,
            allow_invalid=bool(args.allow_invalid),
            fail_fast=bool(args.fail_fast),
            workers=args.workers,
            resume=bool(args.resume),
            skip_existing=bool(args.skip_existing),
            overwrite=bool(args.overwrite),
            input_dir=args.input_dir,
            preserve_tree=bool(args.preserve_tree),
            dry_run=bool(args.dry_run),
            summary_csv=args.summary_csv,
            worker_devices=worker_devices,
            refinement_device=refinement_device,
            native_batch=bool(args.native_batch),
            batch_size=int(args.batch_size),
            preprocess_workers=int(args.preprocess_workers),
            progress=progress,
        )
        print(batch_result.manifest_path)
        if batch_result.results_csv_path is not None:
            print(batch_result.results_csv_path)
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
        export_retargeted=bool(args.save_retargeted),
        refinement_config=config,
        allow_invalid=bool(args.allow_invalid),
        progress=progress,
    )
    for key in ("retargeted_motion", "retargeted_metadata", "retargeted_quality", "final_motion", "final_metadata", "final_quality", "human"):
        path = result.paths.get(key)
        if path is not None:
            print(path)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline IK retargeting + refinement for training data.")
    parser.add_argument("--input", help="Input SMPL/SMPL-X .npz/.npy path, or 'mock'.")
    parser.add_argument("--inputs", nargs="+", help="Batch input SMPL/SMPL-X .npz/.npy paths, or 'mock'.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Batch input directory.")
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
    parser.add_argument("--save-retargeted", action="store_true", help="Save IK-stage retargeted outputs for debugging.")
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Write invalid final outputs in the normal output layout instead of the rejected/ subtree.",
    )
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
    parser.add_argument(
        "--input-pattern",
        action="append",
        default=None,
        help="Batch --input-dir glob pattern. Repeatable; defaults to auto-detecting *.npz files.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively discover --input-dir files.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop a batch run after the first failed item.")
    parser.add_argument("--workers", type=int, default=1, help="Number of batch worker processes.")
    parser.add_argument("--batch-size", type=int, default=8, help="Native batch microbatch size.")
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help="Number of native batch preprocessing worker processes.",
    )
    parser.set_defaults(native_batch=True)
    parser.add_argument("--native-batch", dest="native_batch", action="store_true", help="Use native solver-level batch mode.")
    parser.add_argument(
        "--no-native-batch",
        dest="native_batch",
        action="store_false",
        help="Use the legacy per-item worker batch path.",
    )
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids for batch workers, e.g. 0,1,2.")
    parser.add_argument("--processes-per-gpu", type=int, default=1, help="Batch worker slots per GPU id.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing batch_manifest.json.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip items that already have final outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess items even when resume/skip-existing would skip.")
    parser.add_argument("--input-list", type=Path, default=None, help="Text file containing batch inputs, one per line.")
    parser.add_argument("--exclude-pattern", action="append", default=None, help="Exclude discovered batch inputs by fnmatch pattern.")
    parser.add_argument("--preserve-tree", action="store_true", help="Preserve --input-dir relative paths under --output.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned batch manifest without retargeting.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Optional CSV summary path for batch runs.")
    parser.add_argument("--progress", choices=["auto", "on", "off"], default="auto", help="Show tqdm progress on stderr.")
    return parser


def _batch_inputs_from_args(args: argparse.Namespace) -> list[Path | str]:
    return discover_inputs(
        inputs=args.inputs,
        input_dir=args.input_dir,
        patterns=args.input_pattern or ["*.npz"],
        recursive=bool(args.recursive),
        input_list=args.input_list,
        exclude_patterns=args.exclude_pattern,
    )


def _is_batch_mode(args: argparse.Namespace) -> bool:
    return args.inputs is not None or args.input_dir is not None or args.input_list is not None


def _validate_input_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    batch_mode = _is_batch_mode(args)
    if args.input is not None and batch_mode:
        parser.error("--input cannot be combined with batch sources.")
    if args.input is None and not batch_mode:
        parser.error("one of --input, --inputs, --input-dir, or --input-list is required.")
    if args.workers <= 0:
        parser.error("--workers must be positive.")
    if args.processes_per_gpu <= 0:
        parser.error("--processes-per-gpu must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.preprocess_workers <= 0:
        parser.error("--preprocess-workers must be positive.")
    if batch_mode and not args.native_batch and args.preprocess_workers != 1:
        parser.error("--preprocess-workers is only supported with native batch mode. Use --workers with --no-native-batch.")


if __name__ == "__main__":
    raise SystemExit(main())
