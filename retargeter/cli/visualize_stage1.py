from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from retargeter.newton import RobotSpec, load_stage1_motion_npz
from retargeter.visualize import (
    load_canonical_human_motion_npz,
    load_preprocess_result_npz,
    load_vis_config,
    plot_contact_scores,
    plot_foot_height_and_speed,
    plot_frame_success,
    plot_ik_errors,
    plot_joint_limit_margin,
    plot_joint_positions,
    plot_joint_velocities,
    plot_root_height,
    replay_stage1_motion_with_newton,
)


DEFAULT_ROBOT_SPEC = Path("retargeter/newton/configs/g1_29_robot.yaml")


def main(argv: list[str] | None = None, *, backend=None, viewer_factory=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_visualize_stage1(args, backend=backend, viewer_factory=viewer_factory)


def run_visualize_stage1(args: argparse.Namespace, *, backend=None, viewer_factory=None) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = {}
    if args.fps is not None:
        overrides["fps"] = int(args.fps)
    if args.viewer is not None:
        overrides["viewer"] = args.viewer
    if args.port is not None:
        overrides["port"] = int(args.port)
    overrides["share"] = bool(args.share)
    overrides["loop"] = bool(args.loop)
    overrides["realtime"] = bool(args.realtime)
    config = load_vis_config(args.config, overrides=overrides)

    human_motion = load_canonical_human_motion_npz(args.human) if args.human else None
    preprocess_result = load_preprocess_result_npz(args.human, human_motion) if human_motion is not None else None
    stage1_motion = load_stage1_motion_npz(args.stage1) if args.stage1 else None

    mode = args.mode
    written: list[Path] = []

    if mode in {"replay", "all"}:
        if stage1_motion is None:
            raise ValueError("--stage1 is required for Newton replay visualization.")
        robot_spec_path = args.robot_spec or DEFAULT_ROBOT_SPEC
        robot_spec = RobotSpec.from_yaml(robot_spec_path)
        viewer_kind = str(config.get("viewer", "file")).lower()
        output_path = None
        if viewer_kind == "file":
            output_path = output_dir / (args.replay_name or str(config.get("replay_name", "newton_replay.json")))
        elif viewer_kind == "usd":
            output_path = output_dir / (args.replay_name or "newton_replay.usd")
        replay_result = replay_stage1_motion_with_newton(
            stage1_motion,
            robot_spec,
            viewer=viewer_kind,
            output_path=output_path,
            fps=float(config.get("fps", stage1_motion.fps)),
            loop=bool(config.get("loop", False)) if viewer_kind in {"viser", "gl"} else False,
            max_loops=None if viewer_kind in {"viser", "gl"} and bool(config.get("loop", False)) else 1,
            realtime=bool(config.get("realtime", False)),
            port=int(config.get("port", 8080)),
            share=bool(config.get("share", False)),
            backend=backend,
            viewer_factory=viewer_factory,
            human_motion=human_motion,
            human_offset=parse_human_offset(args.human_offset),
        )
        if replay_result.output_path is not None:
            written.append(replay_result.output_path)

    if mode in {"diagnostics", "all"}:
        if stage1_motion is None and preprocess_result is None:
            raise ValueError("--stage1 or --human with contact arrays is required for diagnostics.")
        written.extend(_write_diagnostics(output_dir, stage1_motion, preprocess_result, args.robot_spec))

    for path in written:
        print(path)
    return 0


def _write_diagnostics(output_dir: Path, stage1_motion, preprocess_result, robot_spec_path: str | None) -> list[Path]:
    paths: list[Path] = []
    if preprocess_result is not None and preprocess_result.contact is not None:
        paths.append(plot_contact_scores(preprocess_result, output_dir / "contact_scores.png"))
        paths.append(plot_foot_height_and_speed(preprocess_result, output_dir / "foot_height_speed.png"))
    if stage1_motion is not None:
        paths.append(plot_ik_errors(stage1_motion, output_dir / "ik_errors.png"))
        paths.append(plot_joint_positions(stage1_motion, output_dir / "joint_positions.png"))
        paths.append(plot_joint_velocities(stage1_motion, output_dir / "joint_velocities.png"))
        paths.append(plot_root_height(stage1_motion, output_dir / "root_height.png"))
        paths.append(plot_frame_success(stage1_motion, output_dir / "frame_success.png"))
        if robot_spec_path is not None:
            robot_spec = RobotSpec.from_yaml(robot_spec_path)
            paths.append(plot_joint_limit_margin(stage1_motion, robot_spec, output_dir / "joint_limit_margin.png"))
    return paths


def parse_human_offset(raw: str | None):
    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    if len(parts) != 3:
        raise ValueError("--human-offset must be comma-separated X,Y,Z.")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--human-offset must contain three numeric values.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("--human-offset values must be finite.")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Stage 1 retargeting outputs.")
    parser.add_argument("--human", type=Path, default=None, help="CanonicalHumanMotion npz path.")
    parser.add_argument("--stage1", type=Path, default=None, help="Stage1Motion npz path.")
    parser.add_argument("--meta", type=Path, default=None, help="Reserved metadata path for future compatibility.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--mode", choices=["replay", "diagnostics", "all"], default="replay")
    parser.add_argument("--config", type=Path, default=None, help="Visualization config override YAML.")
    parser.add_argument("--fps", type=int, default=None, help="Override output video FPS.")
    parser.add_argument("--viewer", choices=["file", "usd", "viser", "gl", "null"], default=None)
    parser.add_argument("--port", type=int, default=None, help="Port for Newton ViewerViser.")
    parser.add_argument("--share", type=int, choices=[0, 1], default=0, help="Request a public Viser share URL.")
    parser.add_argument("--loop", type=int, choices=[0, 1], default=0, help="Loop interactive Newton replay.")
    parser.add_argument("--realtime", type=int, choices=[0, 1], default=0, help="Sleep between frames while replaying.")
    parser.add_argument("--replay-name", default=None, help="Output replay filename for file/usd viewers.")
    parser.add_argument("--robot-spec", type=Path, default=None, help="RobotSpec YAML for Newton replay and diagnostics.")
    parser.add_argument(
        "--human-offset",
        default=None,
        help="Comma-separated X,Y,Z offset for replay human mesh overlay. Default: 0,1.25,0.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
