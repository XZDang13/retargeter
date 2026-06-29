from __future__ import annotations

import argparse
from pathlib import Path

from retargeter.pipeline import run_online_cli_pipeline


def main(argv: list[str] | None = None, *, backend_factory=None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_online_cli_pipeline(
        input_path=args.input,
        output_dir=args.output,
        robot=args.robot,
        model_type=args.model_type,
        fps=args.fps,
        target_fps=args.target_fps,
        gender=args.gender,
        smpl_model_dir=args.smpl_model_dir,
        device=args.device,
        mock_frames=args.mock_frames,
        return_vertices=not args.no_vertices,
        preprocess_config=args.preprocess_config,
        scaler_config=args.scaler_config,
        target_config=args.target_config,
        newton_config=args.newton_config,
        backend_factory=backend_factory,
    )
    for key in ("online_motion", "online_metadata", "online_quality"):
        print(result.paths[key])
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run online IK retargeting.")
    parser.add_argument("--input", required=True, help="Input SMPL/SMPL-X .npz/.npy path, or 'mock'.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", default="unitree_g1_29")
    parser.add_argument("--model-type", choices=["smpl", "smplx"], default=None)
    parser.add_argument("--smpl-model-dir", type=Path, default=Path("assets/body_models"))
    parser.add_argument("--fps", type=float, default=None, help="Input/source FPS override.")
    parser.add_argument("--target-fps", type=float, default=None, help="Optional online output FPS after resampling.")
    parser.add_argument("--gender", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mock-frames", type=int, default=120)
    parser.add_argument("--no-vertices", action="store_true")
    parser.add_argument("--preprocess-config", type=Path, default=None)
    parser.add_argument("--scaler-config", type=Path, default=None)
    parser.add_argument("--target-config", type=Path, default=None)
    parser.add_argument("--newton-config", type=Path, default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
