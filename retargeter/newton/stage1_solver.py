from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retargeter.preprocess import CanonicalHumanMotion, FootContactResult
from retargeter.preprocess.lowpass import normalize_quat_xyzw
from retargeter.scale import IKTargetSet, Stage1TargetBuilder

from .newton_backend import BackendSolveResult, IKBackend, IKState, NewtonBackend, NewtonSolveSettings, RobotBodyState
from .objectives import build_regularization_objectives, build_target_objectives
from .postprocess import PostprocessReport, apply_stage1_postprocess
from .robot_spec import RobotSpec


@dataclass
class Stage1FrameResult:
    frame_idx: int
    robot: str
    root_pos_w: np.ndarray
    root_quat_xyzw: np.ndarray
    joint_names: list[str]
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_state: RobotBodyState
    success: bool
    diagnostics: dict = field(default_factory=dict)

    def state(self) -> IKState:
        return IKState(
            root_pos_w=np.asarray(self.root_pos_w, dtype=np.float64).copy(),
            root_quat_xyzw=normalize_quat_xyzw(self.root_quat_xyzw).copy(),
            joint_pos=np.asarray(self.joint_pos, dtype=np.float64).copy(),
        )


class Stage1NewtonSolver:
    def __init__(
        self,
        stage_config_path: Path | str,
        *,
        robot_config_path: Path | str | None = None,
        backend: IKBackend | None = None,
        target_builder: Stage1TargetBuilder | None = None,
    ):
        self.stage_config_path = Path(stage_config_path)
        self.config = _load_yaml(self.stage_config_path)
        _validate_stage_config(self.config, self.stage_config_path)

        resolved_robot_config = Path(robot_config_path) if robot_config_path is not None else _resolve_path(
            str(self.config["robot_config"]),
            self.stage_config_path,
        )
        self.robot_spec = RobotSpec.from_yaml(resolved_robot_config)
        self.backend = backend if backend is not None else NewtonBackend(self.robot_spec)

        if getattr(self.backend, "robot_spec", self.robot_spec).robot != self.robot_spec.robot:
            raise ValueError("backend robot_spec does not match solver robot_spec.")

        self.target_builder = target_builder
        if self.target_builder is None:
            self.target_builder = Stage1TargetBuilder(
                _resolve_path(str(self.config["scaler_config"]), self.stage_config_path),
                _resolve_path(str(self.config["target_config"]), self.stage_config_path),
            )

        self.robot_spec.require_body_names(self.target_builder.required_robot_body_names("stage1a"))
        self.robot_spec.require_body_names(self.target_builder.required_robot_body_names("stage1b"))

    def solve_frame(
        self,
        motion: CanonicalHumanMotion,
        frame_idx: int,
        *,
        contact_result: FootContactResult | None = None,
        previous_result: Stage1FrameResult | None = None,
    ) -> Stage1FrameResult:
        stage1a_targets = self.target_builder.build(motion, frame_idx, "stage1a", contact_result=contact_result)
        stage1b_targets = self.target_builder.build(motion, frame_idx, "stage1b", contact_result=contact_result)
        return self.solve_target_sets(
            stage1a_targets,
            stage1b_targets,
            frame_idx=frame_idx,
            fps=float(motion.fps),
            previous_result=previous_result,
        )

    def solve_target_sets(
        self,
        stage1a_targets: IKTargetSet,
        stage1b_targets: IKTargetSet,
        *,
        frame_idx: int,
        fps: float,
        previous_result: Stage1FrameResult | None = None,
    ) -> Stage1FrameResult:
        if stage1a_targets.stage_name != "stage1a":
            raise ValueError("stage1a_targets must have stage_name 'stage1a'.")
        if stage1b_targets.stage_name != "stage1b":
            raise ValueError("stage1b_targets must have stage_name 'stage1b'.")
        stage1a_targets.validate()
        stage1b_targets.validate()
        if fps <= 0.0 or not np.isfinite(fps):
            raise ValueError(f"fps must be positive and finite, got {fps!r}.")

        previous_state = previous_result.state() if previous_result is not None else None
        seed_state = self._initial_state(stage1a_targets, previous_state)
        previous_joint_pos = previous_state.joint_pos if previous_state is not None else None
        dt = 1.0 / float(fps)

        stage1a_result = self._solve_stage(
            stage1a_targets,
            seed_state,
            stage_name="stage1a",
            previous_joint_pos=previous_joint_pos,
            dt=dt,
        )
        stage1a_state = stage1a_result.state
        if not stage1a_result.success and self._fallback_on_failure:
            stage1a_state = seed_state

        stage1b_result = self._solve_stage(
            stage1b_targets,
            stage1a_state,
            stage_name="stage1b",
            previous_joint_pos=previous_joint_pos,
            dt=dt,
        )
        final_state = stage1b_result.state
        if not stage1b_result.success and self._fallback_on_failure:
            final_state = stage1a_state

        final_state = final_state.copy()
        joint_vel = np.zeros(self.robot_spec.num_dofs, dtype=np.float64)
        if previous_joint_pos is not None:
            joint_vel = (final_state.joint_pos - previous_joint_pos) / dt

        body_state = self.backend.forward_kinematics(final_state)
        success = bool(stage1a_result.success and stage1b_result.success)
        diagnostics = {
            "stage1a": _backend_result_diagnostics(stage1a_result),
            "stage1b": _backend_result_diagnostics(stage1b_result),
            "fallback_used": bool((not success) and self._fallback_on_failure),
            "target_counts": {
                "stage1a": len(stage1a_targets.targets),
                "stage1b": len(stage1b_targets.targets),
            },
        }

        return Stage1FrameResult(
            frame_idx=int(frame_idx),
            robot=self.robot_spec.robot,
            root_pos_w=final_state.root_pos_w.copy(),
            root_quat_xyzw=normalize_quat_xyzw(final_state.root_quat_xyzw).copy(),
            joint_names=list(self.robot_spec.actuated_joints),
            joint_pos=final_state.joint_pos.copy(),
            joint_vel=joint_vel,
            body_state=body_state,
            success=success,
            diagnostics=diagnostics,
        )

    def _solve_stage(
        self,
        target_set: IKTargetSet,
        seed_state: IKState,
        *,
        stage_name: str,
        previous_joint_pos: np.ndarray | None,
        dt: float,
    ) -> BackendSolveResult:
        objectives = build_target_objectives(target_set, self.robot_spec)
        objectives += build_regularization_objectives(
            self.robot_spec,
            joint_limit_weight=self._objective_weight("joint_limit_weight"),
            posture_weight=self._objective_weight("posture_weight"),
            smooth_weight=self._objective_weight("smooth_weight"),
            damping_weight=self._objective_weight("damping_weight"),
            previous_joint_pos=previous_joint_pos,
        )
        settings = self._solve_settings(stage_name)
        result = self.backend.solve_ik(seed_state, objectives, settings)

        q, report = apply_stage1_postprocess(
            result.state.joint_pos,
            self.robot_spec,
            previous_joint_pos=previous_joint_pos,
            dt=dt,
            clamp_limits=bool(self.config["postprocess"].get("clamp_joint_limits", True)),
            clamp_velocity=bool(self.config["postprocess"].get("clamp_velocity", True)),
            velocity_scale=float(self.config["postprocess"].get("velocity_limit_scale", 1.0)),
        )
        state = IKState(
            root_pos_w=result.state.root_pos_w.copy(),
            root_quat_xyzw=normalize_quat_xyzw(result.state.root_quat_xyzw).copy(),
            joint_pos=q,
        )
        diagnostics = dict(result.diagnostics)
        diagnostics["postprocess"] = _postprocess_report_dict(report)
        return BackendSolveResult(
            state=state,
            success=bool(result.success),
            cost=result.cost,
            iterations=result.iterations,
            diagnostics=diagnostics,
        )

    def _initial_state(self, target_set: IKTargetSet, previous_state: IKState | None) -> IKState:
        if previous_state is not None:
            root_pos = previous_state.root_pos_w.copy()
            root_quat = previous_state.root_quat_xyzw.copy()
            joint_pos = previous_state.joint_pos.copy()
        else:
            root_pos = np.zeros(3, dtype=np.float64)
            root_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            joint_pos = self.robot_spec.default_joint_pos.copy()

        try:
            pelvis_target = target_set.get_target("pelvis")
        except KeyError:
            pelvis_target = None
        if pelvis_target is not None:
            if pelvis_target.target_pos_w is not None:
                root_pos = np.asarray(pelvis_target.target_pos_w, dtype=np.float64).copy()
            if pelvis_target.target_quat_xyzw is not None:
                root_quat = normalize_quat_xyzw(pelvis_target.target_quat_xyzw).copy()

        state = IKState(root_pos_w=root_pos, root_quat_xyzw=root_quat, joint_pos=joint_pos)
        state.validate(self.robot_spec)
        return state

    def _solve_settings(self, stage_name: str) -> NewtonSolveSettings:
        solver_config = self.config["solver"]
        iterations_config = solver_config.get("iterations", {})
        iterations = int(iterations_config.get(stage_name, solver_config.get("default_iterations", 24)))
        return NewtonSolveSettings(
            iterations=iterations,
            step_size=float(solver_config.get("step_size", 1.0)),
            optimizer=str(solver_config.get("optimizer", "lm")),
            jacobian_mode=str(solver_config.get("jacobian_mode", "analytic")),
            lambda_initial=float(solver_config.get("lambda_initial", 0.1)),
        )

    def _objective_weight(self, name: str) -> float:
        return float(self.config["objectives"].get(name, 0.0))

    @property
    def _fallback_on_failure(self) -> bool:
        return bool(self.config["solver"].get("fallback_on_failure", True))


def load_stage1_newton_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    config = _load_yaml(config_path)
    _validate_stage_config(config, config_path)
    return config


def _backend_result_diagnostics(result: BackendSolveResult) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "cost": result.cost,
        "iterations": int(result.iterations),
        "diagnostics": dict(result.diagnostics),
    }


def _postprocess_report_dict(report: PostprocessReport) -> dict[str, Any]:
    return {
        "joint_limit_clamped": bool(report.joint_limit_clamped),
        "velocity_clamped": bool(report.velocity_clamped),
        "joint_limit_violation_before": float(report.joint_limit_violation_before),
        "max_velocity_before": float(report.max_velocity_before),
        "metadata": dict(report.metadata),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Newton stage configs.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Newton stage config {path} must contain a YAML mapping.")
    return data


def _validate_stage_config(config: dict[str, Any], path: Path) -> None:
    required = ["robot_config", "scaler_config", "target_config", "solver", "objectives", "postprocess"]
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError(f"Newton stage config {path} is missing required sections: {missing}.")

    iterations = config["solver"].get("iterations", {})
    if not isinstance(iterations, dict) or "stage1a" not in iterations or "stage1b" not in iterations:
        raise ValueError(f"Newton stage config {path} solver.iterations must define stage1a and stage1b.")


def _resolve_path(raw_path: str, config_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return (config_path.parent / path).resolve()
