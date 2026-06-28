from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from retargeter.newton import (
    RobotSpec,
    SequenceStage1Runner,
    Stage1Motion,
    Stage1NewtonSolver,
    export_stage1_motion,
    load_stage1_newton_config,
)
from retargeter.newton.newton_backend import IKBackend
from retargeter.preprocess import (
    CanonicalHumanMotion,
    FootContactResult,
    MotionPreprocessor,
    REQUIRED_STAGE1_BODY_NAMES,
    load_preprocess_config,
    load_smpl_motion,
)
from retargeter.preprocess.smpl_fk import SMPLForwardKinematics
from retargeter.scale import Stage1TargetBuilder
from retargeter.visualize import (
    export_canonical_human_motion_npz,
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


ROBOT_DEFAULTS = {
    "unitree_g1_29": {
        "scaler_config": Path("retargeter/scale/configs/g1_29_scaler.yaml"),
        "target_config": Path("retargeter/scale/configs/g1_29_stage1_targets.yaml"),
        "newton_config": Path("retargeter/newton/configs/g1_29_newton_stage1.yaml"),
    },
    "unitree_g1_23": {
        "scaler_config": Path("retargeter/scale/configs/g1_23_scaler.yaml"),
        "target_config": Path("retargeter/scale/configs/g1_23_stage1_targets.yaml"),
        "newton_config": Path("retargeter/newton/configs/g1_23_newton_stage1.yaml"),
    },
}
ROBOT_ALIASES = {
    "g1_29": "unitree_g1_29",
    "g1_23": "unitree_g1_23",
}
DEFAULT_PREPROCESS_CONFIG = Path("retargeter/preprocess/configs/default_preprocess.yaml")


BackendFactory = Callable[[RobotSpec], IKBackend]


def main(
    argv: list[str] | None = None,
    *,
    backend_factory: BackendFactory | None = None,
    viewer_backend_factory: BackendFactory | None = None,
    viewer_factory=None,
) -> int:
    args = _build_parser().parse_args(argv)
    result = run_stage1_pipeline(
        args,
        backend_factory=backend_factory,
        viewer_backend_factory=viewer_backend_factory,
        viewer_factory=viewer_factory,
    )
    for key in ("motion_path", "metadata_path", "quality_path", "human_path"):
        path = result.get(key)
        if path is not None:
            print(path)
    for path in result.get("visualization_paths", []):
        print(path)
    return 0


def run_stage1_pipeline(
    args: argparse.Namespace,
    *,
    backend_factory: BackendFactory | None = None,
    viewer_backend_factory: BackendFactory | None = None,
    viewer_factory=None,
) -> dict:
    robot = normalize_robot_name(args.robot)
    config_paths = resolve_pipeline_configs(args, robot)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_config = load_preprocess_config(config_paths["preprocess_config"])
    input_is_mock = str(args.input).lower() == "mock"
    human_output_path = getattr(args, "human_output", None)
    if human_output_path is not None:
        if input_is_mock:
            raise ValueError("--human-output requires a real SMPL/SMPL-X input with vertices.")
        if bool(getattr(args, "no_vertices", False)):
            raise ValueError("--human-output requires vertices; do not use --no-vertices.")

    if input_is_mock:
        canonical_motion = make_mock_canonical_motion(
            num_frames=int(args.mock_frames),
            fps=float(args.fps or 30.0),
        )
        preprocess_result = MotionPreprocessor(preprocess_config).process(canonical_motion)
        mock_contact = make_mock_contact_result(preprocess_result.motion)
        preprocess_result.contact = mock_contact
        preprocess_result.metadata["contact_available"] = True
        preprocess_result.metadata["mock_contact"] = True
        source_metadata = {
            "input": "mock",
            "mock_mode": True,
            "model_type": None,
            "smpl_fk_applied": False,
        }
    else:
        canonical_motion, preprocess_result, source_metadata = run_real_input_pipeline(args, preprocess_config)

    human_path: Path | None = None
    if human_output_path is not None:
        human_path = export_canonical_human_motion_npz(
            preprocess_result.motion,
            human_output_path,
            preprocess_result=preprocess_result,
            require_mesh=True,
        )

    target_builder = Stage1TargetBuilder(config_paths["scaler_config"], config_paths["target_config"])
    newton_config = load_stage1_newton_config(config_paths["newton_config"])
    robot_config_path = resolve_config_relative_path(config_paths["newton_config"], str(newton_config["robot_config"]))
    robot_spec = RobotSpec.from_yaml(robot_config_path)
    backend = backend_factory(robot_spec) if backend_factory is not None else None
    solver = Stage1NewtonSolver(
        config_paths["newton_config"],
        backend=backend,
        target_builder=target_builder,
    )

    validate_pipeline_robot_choices(robot, target_builder, solver)
    stage1_motion = SequenceStage1Runner(solver).run(
        preprocess_result.motion,
        contact_result=preprocess_result.contact,
    )
    attach_pipeline_metadata(
        stage1_motion,
        source_metadata=source_metadata,
        robot=robot,
        config_paths=config_paths,
        preprocess_result=preprocess_result,
    )

    motion_path = output_dir / "motion.npz"
    metadata_path = output_dir / "meta.yaml"
    quality_path = output_dir / "quality.json"
    export_stage1_motion(
        stage1_motion,
        motion_path,
        metadata_path=metadata_path,
        quality_path=quality_path,
    )

    visualization_paths: list[Path] = []
    if bool(int(args.visualize)):
        visualization_paths = write_visualizations(
            preprocess_result.motion,
            preprocess_result,
            stage1_motion,
            robot_spec,
            output_dir,
            vis_config_path=args.visualize_config,
            fps_override=args.visualize_fps,
            viewer_override=args.newton_viewer,
            port_override=args.newton_viewer_port,
            share=bool(args.newton_viewer_share),
            backend=viewer_backend_factory(robot_spec) if viewer_backend_factory is not None else None,
            viewer_factory=viewer_factory,
        )

    return {
        "motion_path": motion_path,
        "metadata_path": metadata_path,
        "quality_path": quality_path,
        "human_path": human_path,
        "visualization_paths": visualization_paths,
        "canonical_motion": canonical_motion,
        "preprocess_result": preprocess_result,
        "stage1_motion": stage1_motion,
        "target_builder": target_builder,
        "solver": solver,
    }


def run_real_input_pipeline(args: argparse.Namespace, preprocess_config) -> tuple[CanonicalHumanMotion, object, dict]:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input motion file does not exist: {input_path}")

    smpl_motion = load_smpl_motion(
        input_path,
        model_type=args.model_type,
        fps=args.fps,
        gender=args.gender,
    )
    model_dir = Path(args.smpl_model_dir)
    validate_smpl_model_dir(model_dir, smpl_motion.model_type)

    try:
        fk = SMPLForwardKinematics(
            model_dir=model_dir,
            model_type=smpl_motion.model_type,
            gender=args.gender or smpl_motion.gender,
            device=args.device,
            foot_vertex_config=preprocess_config.ground.foot_vertex_indices,
        )
        canonical_motion = fk.forward(smpl_motion, return_vertices=not args.no_vertices)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to run SMPL forward kinematics for {input_path}. "
            f"Check --smpl-model-dir, --model-type, --gender, and installed smplx/torch packages."
        ) from exc

    preprocess_result = MotionPreprocessor(preprocess_config).process(canonical_motion)
    source_metadata = {
        "input": str(input_path),
        "mock_mode": False,
        "model_type": smpl_motion.model_type,
        "gender": smpl_motion.gender,
        "smpl_model_dir": str(model_dir),
        "smpl_fk_applied": True,
    }
    return canonical_motion, preprocess_result, source_metadata


def normalize_robot_name(robot: str) -> str:
    normalized = ROBOT_ALIASES.get(str(robot), str(robot))
    if normalized not in ROBOT_DEFAULTS:
        raise ValueError(f"Unsupported robot {robot!r}; expected one of {sorted(ROBOT_DEFAULTS)}.")
    return normalized


def resolve_pipeline_configs(args: argparse.Namespace, robot: str) -> dict[str, Path]:
    defaults = ROBOT_DEFAULTS[robot]
    return {
        "preprocess_config": Path(args.preprocess_config or DEFAULT_PREPROCESS_CONFIG),
        "scaler_config": Path(args.scaler_config or defaults["scaler_config"]),
        "target_config": Path(args.target_config or defaults["target_config"]),
        "newton_config": Path(args.newton_config or defaults["newton_config"]),
    }


def resolve_config_relative_path(config_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return (config_path.parent / path).resolve()


def validate_pipeline_robot_choices(robot: str, target_builder: Stage1TargetBuilder, solver: Stage1NewtonSolver) -> None:
    if target_builder.scaler.robot != robot:
        raise ValueError(
            f"Scaler config robot {target_builder.scaler.robot!r} does not match requested robot {robot!r}."
        )
    if solver.robot_spec.robot != robot:
        raise ValueError(f"Newton config robot {solver.robot_spec.robot!r} does not match requested robot {robot!r}.")


def validate_smpl_model_dir(model_dir: Path, model_type: str) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"SMPL model directory does not exist: {model_dir}")
    nested = model_dir / model_type
    if nested.exists():
        return
    if model_dir.name.lower() == model_type and model_dir.is_dir():
        return
    raise FileNotFoundError(
        f"Could not find {model_type.upper()} model files. Expected {nested} or a direct {model_type} directory."
    )


def make_mock_canonical_motion(num_frames: int = 120, fps: float = 30.0) -> CanonicalHumanMotion:
    if num_frames <= 0:
        raise ValueError("mock_frames must be positive.")
    body_names = list(REQUIRED_STAGE1_BODY_NAMES)
    pos = np.zeros((num_frames, len(body_names), 3), dtype=np.float64)
    quat = np.zeros((num_frames, len(body_names), 4), dtype=np.float64)
    quat[..., 3] = 1.0

    offsets = {
        "pelvis": [0.0, 0.0, 0.90],
        "chest": [0.0, 0.0, 1.30],
        "head": [0.0, 0.0, 1.60],
        "left_shoulder": [0.0, 0.18, 1.35],
        "right_shoulder": [0.0, -0.18, 1.35],
        "left_elbow": [0.05, 0.35, 1.12],
        "right_elbow": [0.05, -0.35, 1.12],
        "left_hand": [0.08, 0.48, 0.92],
        "right_hand": [0.08, -0.48, 0.92],
        "left_hip": [0.0, 0.09, 0.84],
        "right_hip": [0.0, -0.09, 0.84],
        "left_knee": [0.02, 0.10, 0.45],
        "right_knee": [0.02, -0.10, 0.45],
        "left_ankle": [0.04, 0.11, 0.03],
        "right_ankle": [0.04, -0.11, 0.03],
        "left_foot": [0.04, 0.11, 0.03],
        "right_foot": [0.04, -0.11, 0.03],
        "left_toe": [0.17, 0.11, 0.03],
        "right_toe": [0.17, -0.11, 0.03],
        "left_heel": [-0.06, 0.11, 0.03],
        "right_heel": [-0.06, -0.11, 0.03],
    }

    phase = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False)
    root = np.stack(
        [
            np.linspace(0.0, 0.20, num_frames),
            0.02 * np.sin(phase),
            0.02 * np.sin(2.0 * phase),
        ],
        axis=1,
    )
    for idx, name in enumerate(body_names):
        pos[:, idx, :] = root + np.asarray(offsets[name], dtype=np.float64)

    left_swing = 0.10 * np.sin(phase)
    right_swing = -left_swing
    for name, swing in [("left_hand", left_swing), ("left_elbow", left_swing * 0.5)]:
        pos[:, body_names.index(name), 0] += swing
    for name, swing in [("right_hand", right_swing), ("right_elbow", right_swing * 0.5)]:
        pos[:, body_names.index(name), 0] += swing

    return CanonicalHumanMotion(
        fps=float(fps),
        body_names=body_names,
        body_pos_w=pos,
        body_quat_xyzw=quat,
        vertices_w=None,
        metadata={"source": "mock", "world_frame": "z_up"},
    )


def make_mock_contact_result(motion: CanonicalHumanMotion) -> FootContactResult:
    t = motion.num_frames()
    phase = np.arange(t)
    left_score = (0.5 + 0.5 * np.sin(2.0 * np.pi * phase / max(8, t // 4))).astype(np.float64)
    right_score = 1.0 - left_score
    score = {
        "left_foot": left_score,
        "right_foot": right_score,
        "left_toe": left_score,
        "right_toe": right_score,
        "left_heel": left_score,
        "right_heel": right_score,
    }
    binary = {name: values >= 0.5 for name, values in score.items()}
    foot_height = {}
    foot_speed = {}
    for region in score:
        if region in motion.body_names:
            body_pos = motion.get_body_pos(region)
            foot_height[region] = body_pos[:, 2].copy()
            foot_speed[region] = np.zeros(t, dtype=np.float64)
        else:
            foot_height[region] = np.zeros(t, dtype=np.float64)
            foot_speed[region] = np.zeros(t, dtype=np.float64)
    return FootContactResult(
        contact_score=score,
        contact_binary=binary,
        foot_height=foot_height,
        foot_speed=foot_speed,
        ground_height=0.0,
        metadata={"source": "mock", "regions": list(score)},
    )


def attach_pipeline_metadata(
    stage1_motion: Stage1Motion,
    *,
    source_metadata: dict,
    robot: str,
    config_paths: dict[str, Path],
    preprocess_result,
) -> None:
    stage1_motion.metadata.update(
        {
            "pipeline": "stage1",
            "robot": robot,
            "config_paths": {key: str(path) for key, path in config_paths.items()},
            "frame_count": stage1_motion.num_frames(),
            "fps": float(stage1_motion.fps),
            "success_ratio": float(np.mean(stage1_motion.success)) if stage1_motion.num_frames() else 0.0,
            "preprocess_warnings": list(preprocess_result.warnings),
            "contact_available": preprocess_result.contact is not None,
            "preprocess_metadata": dict(preprocess_result.metadata),
            "source": source_metadata,
        }
    )


def write_visualizations(
    human_motion: CanonicalHumanMotion,
    preprocess_result,
    stage1_motion: Stage1Motion,
    robot_spec: RobotSpec,
    output_dir: Path,
    *,
    vis_config_path: Path | None,
    fps_override: int | None,
    viewer_override: str | None,
    port_override: int | None,
    share: bool,
    backend=None,
    viewer_factory=None,
) -> list[Path]:
    overrides = {}
    if fps_override is not None:
        overrides["fps"] = int(fps_override)
    if viewer_override is not None:
        overrides["viewer"] = str(viewer_override)
    if port_override is not None:
        overrides["port"] = int(port_override)
    if share:
        overrides["share"] = True
    config = load_vis_config(vis_config_path, overrides=overrides)
    viewer_kind = str(config.get("viewer", "file")).lower()
    replay_path: Path | None = None
    if viewer_kind == "file":
        replay_path = output_dir / str(config.get("replay_name", "newton_replay.json"))
    elif viewer_kind == "usd":
        replay_path = output_dir / "newton_replay.usd"

    replay_result = replay_stage1_motion_with_newton(
        stage1_motion,
        robot_spec,
        viewer=viewer_kind,
        output_path=replay_path,
        fps=float(config.get("fps", stage1_motion.fps)),
        loop=bool(config.get("loop", False)) if viewer_kind in {"viser", "gl"} else False,
        max_loops=None if viewer_kind in {"viser", "gl"} and bool(config.get("loop", False)) else 1,
        realtime=bool(config.get("realtime", False)),
        port=int(config.get("port", 8080)),
        share=bool(config.get("share", False)),
        backend=backend,
        viewer_factory=viewer_factory,
    )

    paths: list[Path] = []
    if replay_result.output_path is not None:
        paths.append(replay_result.output_path)

    diagnostics = [
        plot_ik_errors(stage1_motion, output_dir / "ik_errors.png"),
        plot_joint_positions(stage1_motion, output_dir / "joint_positions.png"),
        plot_joint_velocities(stage1_motion, output_dir / "joint_velocities.png"),
        plot_joint_limit_margin(stage1_motion, robot_spec, output_dir / "joint_limit_margin.png"),
        plot_root_height(stage1_motion, output_dir / "root_height.png"),
        plot_frame_success(stage1_motion, output_dir / "frame_success.png"),
    ]
    if preprocess_result.contact is not None:
        diagnostics.insert(0, plot_foot_height_and_speed(preprocess_result, output_dir / "foot_height_speed.png"))
        diagnostics.insert(0, plot_contact_scores(preprocess_result, output_dir / "contact_scores.png"))
    return paths + diagnostics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 1 SMPL/SMPL-X to humanoid Newton retargeting.")
    parser.add_argument("--input", required=True, help="Input SMPL/SMPL-X .npz/.npy path, or 'mock'.")
    parser.add_argument("--model-type", choices=["smpl", "smplx"], default=None)
    parser.add_argument("--smpl-model-dir", type=Path, default=Path("assets/body_models"))
    parser.add_argument("--robot", default="unitree_g1_29")
    parser.add_argument("--preprocess-config", type=Path, default=None)
    parser.add_argument("--scaler-config", type=Path, default=None)
    parser.add_argument("--target-config", type=Path, default=None)
    parser.add_argument("--newton-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualize", type=int, choices=[0, 1], default=0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--gender", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mock-frames", type=int, default=120)
    parser.add_argument("--no-vertices", action="store_true")
    parser.add_argument(
        "--human-output",
        type=Path,
        default=None,
        help="Optional canonical human replay npz output with SMPL-X vertices and mesh faces.",
    )
    parser.add_argument("--visualize-config", type=Path, default=None)
    parser.add_argument("--visualize-fps", type=int, default=None)
    parser.add_argument("--newton-viewer", choices=["file", "usd", "viser", "gl", "null"], default=None)
    parser.add_argument("--newton-viewer-port", type=int, default=None)
    parser.add_argument("--newton-viewer-share", type=int, choices=[0, 1], default=0)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
