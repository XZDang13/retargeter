from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from retargeter.newton import IKState, NewtonBackend, RobotSpec, Stage1Motion
from retargeter.preprocess import CanonicalHumanMotion


NEWTON_VIEWER_KINDS = {"file", "usd", "viser", "gl", "null"}
DEFAULT_HUMAN_MESH_OFFSET = np.asarray([0.0, 1.25, 0.0], dtype=np.float64)
DEFAULT_HUMAN_MESH_COLOR = (0.58, 0.72, 0.82)


class ReplayBackend(Protocol):
    robot_spec: RobotSpec
    model: Any

    def make_newton_state(self, state: IKState) -> Any:
        ...


ViewerFactory = Callable[[str, dict[str, Any]], Any]


@dataclass(frozen=True)
class NewtonReplayResult:
    viewer: str
    frame_count: int
    fps: float
    output_path: Path | None = None
    url: str | None = None


@dataclass(frozen=True)
class HumanMeshOverlay:
    motion: CanonicalHumanMotion
    offset: np.ndarray
    indices: Any
    wp: Any
    color: tuple[float, float, float]


def stage1_frame_to_ik_state(motion: Stage1Motion, robot_spec: RobotSpec, frame_idx: int) -> IKState:
    """Convert one Stage1Motion frame into the IKState expected by NewtonBackend."""
    validate_stage1_motion_for_robot(motion, robot_spec)
    if frame_idx < 0 or frame_idx >= motion.num_frames():
        raise IndexError(f"frame_idx {frame_idx} is outside [0, {motion.num_frames()}).")
    return IKState(
        root_pos_w=np.asarray(motion.root_pos_w[frame_idx], dtype=np.float64).copy(),
        root_quat_xyzw=np.asarray(motion.root_quat_xyzw[frame_idx], dtype=np.float64).copy(),
        joint_pos=np.asarray(motion.joint_pos[frame_idx], dtype=np.float64).copy(),
    )


def validate_stage1_motion_for_robot(motion: Stage1Motion, robot_spec: RobotSpec) -> None:
    motion.validate()
    if motion.robot != robot_spec.robot:
        raise ValueError(f"Stage1Motion robot {motion.robot!r} does not match RobotSpec {robot_spec.robot!r}.")
    if motion.joint_names != robot_spec.actuated_joints:
        raise ValueError("Stage1Motion joint_names must exactly match RobotSpec actuated_joints.")


def replay_stage1_motion_with_newton(
    motion: Stage1Motion,
    robot_spec: RobotSpec,
    *,
    viewer: str = "file",
    output_path: Path | str | None = None,
    fps: float | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    loop: bool = False,
    max_loops: int | None = 1,
    realtime: bool = False,
    port: int = 8080,
    share: bool = False,
    close_viewer: bool = True,
    backend: ReplayBackend | None = None,
    viewer_factory: ViewerFactory | None = None,
    human_motion: CanonicalHumanMotion | None = None,
    human_offset: np.ndarray | tuple[float, float, float] | list[float] | None = None,
    human_mesh_color: tuple[float, float, float] = DEFAULT_HUMAN_MESH_COLOR,
) -> NewtonReplayResult:
    """Send Stage1Motion frames through Newton's real viewer API.

    Use ``viewer='file'`` or ``viewer='usd'`` for deterministic artifacts, and
    ``viewer='viser'`` or ``viewer='gl'`` for interactive inspection.
    """
    viewer = str(viewer).lower()
    if viewer not in NEWTON_VIEWER_KINDS:
        raise ValueError(f"Unsupported Newton viewer {viewer!r}; expected one of {sorted(NEWTON_VIEWER_KINDS)}.")
    validate_stage1_motion_for_robot(motion, robot_spec)
    frame_indices = _frame_indices(motion, start_frame=start_frame, end_frame=end_frame)
    effective_fps = float(fps or motion.fps)
    if effective_fps <= 0.0 or not np.isfinite(effective_fps):
        raise ValueError(f"fps must be positive and finite, got {effective_fps!r}.")

    output = Path(output_path) if output_path is not None else None
    if viewer in {"file", "usd"} and output is None:
        output = Path("stage1_newton_replay.json" if viewer == "file" else "stage1_newton_replay.usd")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    human_overlay = _prepare_human_mesh_overlay(
        human_motion,
        offset=human_offset,
        color=human_mesh_color,
    )
    replay_backend = backend or NewtonBackend(robot_spec, load_visual_shapes=True, add_ground_plane=True)
    newton_viewer = _make_viewer(
        viewer,
        output_path=output,
        fps=effective_fps,
        frame_count=len(frame_indices),
        port=port,
        share=share,
        viewer_factory=viewer_factory,
    )
    newton_viewer.set_model(replay_backend.model)

    frames_written = 0
    cursor = 0
    loops_completed = 0
    try:
        while newton_viewer.is_running():
            if not newton_viewer.should_step():
                if realtime:
                    time.sleep(min(1.0 / effective_fps, 0.01))
                continue

            frame_idx = frame_indices[cursor]
            ik_state = stage1_frame_to_ik_state(motion, robot_spec, frame_idx)
            native_state = replay_backend.make_newton_state(ik_state)
            newton_viewer.begin_frame(frames_written / effective_fps)
            newton_viewer.log_state(native_state)
            _log_replay_scalars(newton_viewer, frame_idx, ik_state)
            if human_overlay is not None:
                robot_time_s = frame_idx / float(motion.fps)
                _log_human_mesh(newton_viewer, human_overlay, robot_time_s)
            newton_viewer.end_frame()
            frames_written += 1

            cursor += 1
            if cursor >= len(frame_indices):
                loops_completed += 1
                if not loop or (max_loops is not None and loops_completed >= max_loops):
                    break
                cursor = 0

            if realtime:
                time.sleep(1.0 / effective_fps)
    except KeyboardInterrupt:
        pass
    finally:
        if close_viewer:
            close = getattr(newton_viewer, "close", None)
            if close is not None:
                close()

    return NewtonReplayResult(
        viewer=viewer,
        frame_count=frames_written,
        fps=effective_fps,
        output_path=output,
        url=getattr(newton_viewer, "url", None),
    )


def _prepare_human_mesh_overlay(
    motion: CanonicalHumanMotion | None,
    *,
    offset: np.ndarray | tuple[float, float, float] | list[float] | None,
    color: tuple[float, float, float],
) -> HumanMeshOverlay | None:
    if motion is None:
        return None
    motion.validate()
    if motion.vertices_w is None or motion.mesh_faces is None:
        raise ValueError("Human mesh replay requires vertices_w and mesh_faces.")
    if motion.num_frames() <= 0:
        raise ValueError("Human mesh replay requires at least one human frame.")

    try:
        import warp as wp
    except ImportError as exc:  # pragma: no cover - runtime dependency for Newton viewers.
        raise RuntimeError("Warp is required for human mesh replay.") from exc

    offset_arr = _human_offset_array(offset)
    color_tuple = _human_mesh_color(color)
    faces = np.asarray(motion.mesh_faces, dtype=np.int32)
    indices = wp.array(faces.reshape(-1), dtype=wp.int32)
    return HumanMeshOverlay(motion=motion, offset=offset_arr, indices=indices, wp=wp, color=color_tuple)


def _log_human_mesh(newton_viewer, overlay: HumanMeshOverlay, robot_time_s: float) -> None:
    log_mesh = getattr(newton_viewer, "log_mesh", None)
    if log_mesh is None:
        raise RuntimeError("Selected Newton viewer does not support human mesh replay.")

    human_idx = _human_frame_index(robot_time_s, overlay.motion.fps, overlay.motion.num_frames())
    vertices = np.asarray(overlay.motion.vertices_w[human_idx], dtype=np.float32).copy()
    vertices += overlay.offset.astype(np.float32, copy=False)
    points = overlay.wp.array(vertices, dtype=overlay.wp.vec3)
    log_mesh(
        "human/smplx_mesh",
        points,
        overlay.indices,
        color=overlay.color,
        backface_culling=False,
    )


def _human_frame_index(robot_time_s: float, human_fps: float, frame_count: int) -> int:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if not np.isfinite(robot_time_s):
        raise ValueError(f"robot_time_s must be finite, got {robot_time_s!r}.")
    if not np.isfinite(human_fps) or human_fps <= 0.0:
        raise ValueError(f"human_fps must be positive and finite, got {human_fps!r}.")
    idx = int(round(float(robot_time_s) * float(human_fps)))
    return int(np.clip(idx, 0, frame_count - 1))


def _human_offset_array(offset: np.ndarray | tuple[float, float, float] | list[float] | None) -> np.ndarray:
    arr = DEFAULT_HUMAN_MESH_OFFSET if offset is None else np.asarray(offset, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"human_offset must have shape [3], got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("human_offset contains NaN or inf values.")
    return arr.astype(np.float64, copy=True)


def _human_mesh_color(color: tuple[float, float, float]) -> tuple[float, float, float]:
    arr = np.asarray(color, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"human_mesh_color must be three finite floats, got {color!r}.")
    return tuple(float(value) for value in arr)


def _log_replay_scalars(newton_viewer, frame_idx: int, state: IKState) -> None:
    log_scalar = getattr(newton_viewer, "log_scalar", None)
    if log_scalar is None:
        return
    try:
        log_scalar("stage1/frame", int(frame_idx))
        log_scalar("stage1/root_x", float(state.root_pos_w[0]))
        log_scalar("stage1/root_y", float(state.root_pos_w[1]))
        log_scalar("stage1/root_z", float(state.root_pos_w[2]))
    except Exception:
        return


def record_stage1_newton_replay(
    motion: Stage1Motion,
    robot_spec: RobotSpec,
    output_path: Path | str,
    *,
    fps: float | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    backend: ReplayBackend | None = None,
    viewer_factory: ViewerFactory | None = None,
) -> NewtonReplayResult:
    """Write a Newton ViewerFile replay that can be loaded by Newton tools."""
    return replay_stage1_motion_with_newton(
        motion,
        robot_spec,
        viewer="file",
        output_path=output_path,
        fps=fps,
        start_frame=start_frame,
        end_frame=end_frame,
        loop=False,
        realtime=False,
        backend=backend,
        viewer_factory=viewer_factory,
    )


def _frame_indices(motion: Stage1Motion, *, start_frame: int, end_frame: int | None) -> list[int]:
    total = motion.num_frames()
    end = total if end_frame is None else int(end_frame)
    start = int(start_frame)
    if start < 0 or start >= total:
        raise IndexError(f"start_frame {start} is outside [0, {total}).")
    if end <= start or end > total:
        raise IndexError(f"end_frame {end} must be in ({start}, {total}].")
    return list(range(start, end))


def _make_viewer(
    viewer: str,
    *,
    output_path: Path | None,
    fps: float,
    frame_count: int,
    port: int,
    share: bool,
    viewer_factory: ViewerFactory | None,
):
    if viewer_factory is not None:
        return viewer_factory(
            viewer,
            {
                "output_path": output_path,
                "fps": fps,
                "frame_count": frame_count,
                "port": port,
                "share": share,
            },
        )

    try:
        from newton import viewer as newton_viewer
    except ImportError as exc:
        raise RuntimeError("Newton is required for Stage 1 replay visualization.") from exc

    if viewer == "file":
        return newton_viewer.ViewerFile(str(output_path))
    if viewer == "usd":
        return newton_viewer.ViewerUSD(str(output_path), fps=int(round(fps)), num_frames=frame_count)
    if viewer == "viser":
        try:
            return newton_viewer.ViewerViser(port=int(port), share=bool(share), verbose=True)
        except ImportError as exc:
            raise RuntimeError(
                "Newton ViewerViser requires the optional 'viser' package. "
                "Install it or use '--viewer gl' for Newton's native GL viewer."
            ) from exc
    if viewer == "gl":
        return newton_viewer.ViewerGL()
    if viewer == "null":
        return newton_viewer.ViewerNull(num_frames=frame_count)
    raise AssertionError(f"Unhandled viewer kind {viewer!r}.")
