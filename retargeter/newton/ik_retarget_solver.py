from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retargeter.preprocess import CanonicalHumanMotion, FootContactResult
from retargeter.preprocess.lowpass import normalize_quat_xyzw
from retargeter.scale import IKTargetBuilder, IKTargetSet

from .newton_backend import BackendSolveResult, IKBackend, IKState, NewtonBackend, NewtonSolveSettings, RobotBodyState
from .objectives import build_regularization_objectives, build_self_collision_objectives, build_target_objectives
from .objectives import summarize_self_collision_clearance
from .postprocess import PostprocessReport, apply_ik_postprocess
from .robot_spec import RobotSpec


IK_PASS_NAMES = ("full_body_tracking",)


@dataclass
class IKRetargetFrameResult:
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


class NewtonIKRetargetSolver:
    def __init__(
        self,
        stage_config_path: Path | str,
        *,
        robot_config_path: Path | str | None = None,
        backend: IKBackend | None = None,
        target_builder: IKTargetBuilder | None = None,
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
        self._reusable_ik_solver = None
        self._reusable_ik_batch_solvers = {}
        self._self_collision_objectives = build_self_collision_objectives(
            self.robot_spec,
            self.config.get("self_collision"),
        )
        self._self_collision_pairs = (
            self._self_collision_objectives[0].self_collision_pairs if self._self_collision_objectives else ()
        )

        self.target_builder = target_builder
        if self.target_builder is None:
            self.target_builder = IKTargetBuilder(
                _resolve_path(str(self.config["scaler_config"]), self.stage_config_path),
                _resolve_path(str(self.config["target_config"]), self.stage_config_path),
            )

        self.robot_spec.require_body_names(self.target_builder.required_robot_body_names("full_body_tracking"))

    def solve_frame(
        self,
        motion: CanonicalHumanMotion,
        frame_idx: int,
        *,
        contact_result: FootContactResult | None = None,
        previous_result: IKRetargetFrameResult | None = None,
    ) -> IKRetargetFrameResult:
        tracking_targets = self.target_builder.build(motion, frame_idx, "full_body_tracking", contact_result=contact_result)
        return self.solve_target_sets(
            tracking_targets,
            frame_idx=frame_idx,
            fps=float(motion.fps),
            previous_result=previous_result,
        )

    def solve_target_sets(
        self,
        tracking_targets: IKTargetSet,
        *,
        frame_idx: int,
        fps: float,
        previous_result: IKRetargetFrameResult | None = None,
    ) -> IKRetargetFrameResult:
        tracking_targets = _target_set_with_pass_name(tracking_targets, "full_body_tracking")
        tracking_targets.validate()
        if fps <= 0.0 or not np.isfinite(fps):
            raise ValueError(f"fps must be positive and finite, got {fps!r}.")

        previous_state = previous_result.state() if previous_result is not None else None
        previous_joint_pos = previous_state.joint_pos if previous_state is not None else None
        dt = 1.0 / float(fps)

        fallback_used = False
        seed_state = self._initial_state(tracking_targets, previous_state)

        tracking_result = self._solve_pass(
            tracking_targets,
            seed_state,
            pass_name="full_body_tracking",
            previous_joint_pos=previous_joint_pos,
            dt=dt,
        )
        final_state = tracking_result.state
        if not tracking_result.success and self._fallback_on_failure:
            final_state = seed_state
            fallback_used = True

        final_state = final_state.copy()
        joint_vel = np.zeros(self.robot_spec.num_dofs, dtype=np.float64)
        if previous_joint_pos is not None:
            joint_vel = (final_state.joint_pos - previous_joint_pos) / dt

        body_state = self.backend.forward_kinematics(final_state)
        success = bool(tracking_result.success)
        diagnostics = {
            "full_body_tracking": _backend_result_diagnostics(tracking_result),
            "fallback_used": bool(fallback_used),
            "target_counts": {
                "full_body_tracking": len(tracking_targets.targets),
            },
        }
        self_collision = self._self_collision_diagnostics(body_state)
        if self_collision is not None:
            diagnostics["self_collision"] = self_collision

        return IKRetargetFrameResult(
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

    def _solve_pass(
        self,
        target_set: IKTargetSet,
        seed_state: IKState,
        *,
        pass_name: str,
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
        objectives += self._self_collision_objectives
        settings = self._solve_settings(pass_name)
        result = self._solve_backend_ik(seed_state, objectives, settings)

        q, report = apply_ik_postprocess(
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

    def _solve_backend_ik(
        self,
        seed_state: IKState,
        objectives,
        settings: NewtonSolveSettings,
    ) -> BackendSolveResult:
        factory = getattr(self.backend, "create_reusable_solver", None)
        if factory is None:
            return self.backend.solve_ik(seed_state, objectives, settings)

        reusable = self._reusable_ik_solver
        try:
            if reusable is None or not reusable.compatible(objectives, settings):
                reusable = factory(objectives, settings)
                self._reusable_ik_solver = reusable
            return reusable.solve(seed_state, objectives, settings)
        except Exception:
            self._reusable_ik_solver = None
            return self.backend.solve_ik(seed_state, objectives, settings)

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

    def _solve_settings(self, pass_name: str) -> NewtonSolveSettings:
        solver_config = self.config["solver"]
        iterations_config = solver_config.get("iterations", {})
        iterations = int(iterations_config.get(pass_name, solver_config.get("default_iterations", 24)))
        return NewtonSolveSettings(
            iterations=iterations,
            step_size=float(solver_config.get("step_size", 1.0)),
            optimizer=str(solver_config.get("optimizer", "lm")),
            jacobian_mode=str(solver_config.get("jacobian_mode", "analytic")),
            lambda_initial=float(solver_config.get("lambda_initial", 0.1)),
        )

    def _objective_weight(self, name: str) -> float:
        return float(self.config["objectives"].get(name, 0.0))

    def _self_collision_diagnostics(self, body_state: RobotBodyState) -> dict | None:
        if not self._self_collision_pairs:
            return None
        return summarize_self_collision_clearance(
            body_state.body_names,
            body_state.body_pos_w,
            body_state.body_quat_xyzw,
            self._self_collision_pairs,
        )

    @property
    def _fallback_on_failure(self) -> bool:
        return bool(self.config["solver"].get("fallback_on_failure", True))


def load_newton_ik_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    config = _load_yaml(config_path)
    _validate_stage_config(config, config_path)
    return config


def _canonical_pass_name(pass_name: str) -> str:
    if pass_name not in IK_PASS_NAMES:
        raise ValueError(f"IK pass name must be one of {sorted(IK_PASS_NAMES)}, got {pass_name!r}.")
    return pass_name


def _target_set_with_pass_name(target_set: IKTargetSet, expected_pass_name: str) -> IKTargetSet:
    actual = _canonical_pass_name(target_set.pass_name)
    if actual != expected_pass_name:
        raise ValueError(f"{expected_pass_name}_targets must have pass_name {expected_pass_name!r}.")
    if target_set.pass_name == actual:
        return target_set
    return IKTargetSet(targets=list(target_set.targets), pass_name=actual, metadata=dict(target_set.metadata))


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
        raise RuntimeError("PyYAML is required to load Newton IK configs.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Newton IK config {path} must contain a YAML mapping.")
    return data


def _validate_stage_config(config: dict[str, Any], path: Path) -> None:
    required = ["robot_config", "scaler_config", "target_config", "solver", "objectives", "postprocess"]
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError(f"Newton IK config {path} is missing required sections: {missing}.")

    if "pass_mode" in config["solver"]:
        raise ValueError(
            f"Newton IK config {path} uses deprecated solver.pass_mode. "
            "IK retargeting is single-pass full_body_tracking only."
        )
    iterations = config["solver"].get("iterations", {})
    if not isinstance(iterations, dict):
        raise ValueError(f"Newton IK config {path} solver.iterations must be a mapping.")
    normalized = {_canonical_pass_name(name): value for name, value in iterations.items()}
    if "full_body_tracking" not in normalized:
        raise ValueError(f"Newton IK config {path} solver.iterations must define full_body_tracking.")
    config["solver"]["iterations"] = normalized


def _resolve_path(raw_path: str, config_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return (config_path.parent / path).resolve()
