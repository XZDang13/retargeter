from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from retargeter.pipeline import ViewerPipeline


def main(argv: list[str] | None = None, *, backend=None, viewer_factory=None) -> int:
    args = _build_parser().parse_args(argv)
    realtime = _resolve_realtime(args.viewer, args.realtime)
    result = ViewerPipeline().replay(
        input_path=args.input,
        output_dir=args.output,
        human_path=args.human,
        robot_spec_path=args.robot_spec,
        viewer=args.viewer,
        fps=args.fps,
        loop=bool(args.loop),
        max_loops=None if bool(args.loop) else 1,
        realtime=realtime,
        port=args.port,
        share=bool(args.share),
        replay_name=args.replay_name,
        human_offset=parse_human_offset(args.human_offset),
        backend=backend,
        viewer_factory=viewer_factory,
    )
    if result.replay_result.output_path is not None:
        print(result.replay_result.output_path)
    return 0


def parse_human_offset(raw: str | None):
    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    if len(parts) != 3:
        raise ValueError("--human-offset must be comma-separated X,Y,Z.")
    values = tuple(float(part) for part in parts)
    if not np.all(np.isfinite(values)):
        raise ValueError("--human-offset values must be finite.")
    return values


def _resolve_realtime(viewer: str, raw: int | None) -> bool:
    if raw is not None:
        return bool(raw)
    return str(viewer).lower() in {"gl", "viser"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View online or refined retargeting outputs.")
    parser.add_argument("--input", type=Path, required=True, help="Output directory or motion npz.")
    parser.add_argument("--output", type=Path, default=None, help="Replay artifact directory for file/usd viewers.")
    parser.add_argument("--human", type=Path, default=None, help="Optional human.npz mesh overlay.")
    parser.add_argument("--robot-spec", type=Path, default=None)
    parser.add_argument("--viewer", choices=["file", "usd", "viser", "gl", "null"], default="file")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop", type=int, choices=[0, 1], default=0)
    parser.add_argument("--realtime", type=int, choices=[0, 1], default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--share", type=int, choices=[0, 1], default=0)
    parser.add_argument("--replay-name", default=None)
    parser.add_argument("--human-offset", default=None, help="Comma-separated X,Y,Z offset. Default: 0,1.25,0.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
