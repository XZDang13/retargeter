from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retargeter.preprocess import CanonicalHumanMotion, FootContactResult
from retargeter.preprocess.lowpass import normalize_quat_xyzw
from retargeter.scale import IKTargetBuilder, IKTargetSet

from .newton_backend import BackendSolveResult, IKBackend, IKState, NewtonBackend, NewtonSolveSettings, RobotBodyState
from .objectives import build_regularization_objectives, build_target_objectives
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

    def solve_frames_batch(
        self,
        motions: list[CanonicalHumanMotion],
        frame_idx: int,
        *,
        contact_results: list[FootContactResult | None] | None = None,
        previous_results: list[IKRetargetFrameResult | None] | None = None,
    ) -> list[IKRetargetFrameResult]:
        if not motions:
            return []
        contacts = contact_results if contact_results is not None else [None] * len(motions)
        previous = previous_results if previous_results is not None else [None] * len(motions)
        if len(contacts) != len(motions) or len(previous) != len(motions):
            raise ValueError("motions, contact_results, and previous_results must have the same length.")
        target_sets = [
            self.target_builder.build(motion, frame_idx, "full_body_tracking", contact_result=contact)
            for motion, contact in zip(motions, contacts)
        ]
        return self.solve_target_sets_batch(
            target_sets,
            frame_idx=frame_idx,
            fps_values=[float(motion.fps) for motion in motions],
            previous_results=previous,
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

    def solve_target_sets_batch(
        self,
        tracking_targets: list[IKTargetSet],
        *,
        frame_idx: int,
        fps_values: list[float],
        previous_results: list[IKRetargetFrameResult | None] | None = None,
    ) -> list[IKRetargetFrameResult]:
        if not tracking_targets:
            return []
        if len(fps_values) != len(tracking_targets):
            raise ValueError("fps_values must match tracking_targets length.")
        previous = previous_results if previous_results is not None else [None] * len(tracking_targets)
        if len(previous) != len(tracking_targets):
            raise ValueError("previous_results must match tracking_targets length.")

        prepared: list[dict[str, Any]] = []
        groups: dict[tuple[tuple[Any, ...], ...], list[int]] = {}
        for index, (target_set, fps, previous_result) in enumerate(zip(tracking_targets, fps_values, previous)):
            target_set = _target_set_with_pass_name(target_set, "full_body_tracking")
            target_set.validate()
            if fps <= 0.0 or not np.isfinite(fps):
                raise ValueError(f"fps must be positive and finite, got {fps!r}.")
            previous_state = previous_result.state() if previous_result is not None else None
            previous_joint_pos = previous_state.joint_pos if previous_state is not None else None
            seed_state = self._initial_state(target_set, previous_state)
            objectives = build_target_objectives(target_set, self.robot_spec)
            objectives += build_regularization_objectives(
                self.robot_spec,
                joint_limit_weight=self._objective_weight("joint_limit_weight"),
                posture_weight=self._objective_weight("posture_weight"),
                smooth_weight=self._objective_weight("smooth_weight"),
                damping_weight=self._objective_weight("damping_weight"),
                previous_joint_pos=previous_joint_pos,
            )
            prepared.append(
                {
                    "target_set": target_set,
                    "fps": float(fps),
                    "previous_joint_pos": previous_joint_pos,
                    "seed_state": seed_state,
                    "objectives": objectives,
                }
            )
            groups.setdefault(_objective_batch_key(objectives), []).append(index)

        pass_results: list[BackendSolveResult | None] = [None] * len(prepared)
        settings = self._solve_settings("full_body_tracking")
        for indices in groups.values():
            if len(indices) <= 1:
                idx = indices[0]
                pass_results[idx] = self._solve_pass(
                    prepared[idx]["target_set"],
                    prepared[idx]["seed_state"],
                    pass_name="full_body_tracking",
                    previous_joint_pos=prepared[idx]["previous_joint_pos"],
                    dt=1.0 / prepared[idx]["fps"],
                )
                continue
            seed_states = [prepared[idx]["seed_state"] for idx in indices]
            objectives_by_problem = [prepared[idx]["objectives"] for idx in indices]
            raw_results = self._solve_backend_ik_batch(seed_states, objectives_by_problem, settings)
            for local_index, idx in enumerate(indices):
                result = raw_results[local_index]
                q, report = apply_ik_postprocess(
                    result.state.joint_pos,
                    self.robot_spec,
                    previous_joint_pos=prepared[idx]["previous_joint_pos"],
                    dt=1.0 / prepared[idx]["fps"],
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
                pass_results[idx] = BackendSolveResult(
                    state=state,
                    success=bool(result.success),
                    cost=result.cost,
                    iterations=result.iterations,
                    diagnostics=diagnostics,
                )

        frame_results: list[IKRetargetFrameResult] = []
        for idx, result in enumerate(pass_results):
            if result is None:
                raise RuntimeError("Internal error: missing batched IK result.")
            previous_joint_pos = prepared[idx]["previous_joint_pos"]
            dt = 1.0 / prepared[idx]["fps"]
            final_state = result.state
            fallback_used = False
            if not result.success and self._fallback_on_failure:
                final_state = prepared[idx]["seed_state"]
                fallback_used = True
            final_state = final_state.copy()
            joint_vel = np.zeros(self.robot_spec.num_dofs, dtype=np.float64)
            if previous_joint_pos is not None:
                joint_vel = (final_state.joint_pos - previous_joint_pos) / dt
            body_state = self.backend.forward_kinematics(final_state)
            frame_results.append(
                IKRetargetFrameResult(
                    frame_idx=int(frame_idx),
                    robot=self.robot_spec.robot,
                    root_pos_w=final_state.root_pos_w.copy(),
                    root_quat_xyzw=normalize_quat_xyzw(final_state.root_quat_xyzw).copy(),
                    joint_names=list(self.robot_spec.actuated_joints),
                    joint_pos=final_state.joint_pos.copy(),
                    joint_vel=joint_vel,
                    body_state=body_state,
                    success=bool(result.success),
                    diagnostics={
                        "full_body_tracking": _backend_result_diagnostics(result),
                        "fallback_used": bool(fallback_used),
                        "target_counts": {
                            "full_body_tracking": len(prepared[idx]["target_set"].targets),
                        },
                    },
                )
            )
        return frame_results

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

    def _solve_backend_ik_batch(
        self,
        seed_states: list[IKState],
        objectives_by_problem: list[list[Any]],
        settings: NewtonSolveSettings,
    ) -> list[BackendSolveResult]:
        factory = getattr(self.backend, "create_reusable_batch_solver", None)
        if factory is None:
            batch_solver = getattr(self.backend, "solve_ik_batch", None)
            if batch_solver is None:
                return [
                    self._solve_backend_ik(seed_state, objectives, settings)
                    for seed_state, objectives in zip(seed_states, objectives_by_problem)
                ]
            return batch_solver(seed_states, objectives_by_problem, settings)

        cache = getattr(self, "_reusable_ik_batch_solvers", None)
        if cache is None:
            cache = {}
            self._reusable_ik_batch_solvers = cache
        key = (len(seed_states), _objective_batch_key(objectives_by_problem[0]), _settings_batch_key(settings))
        reusable = cache.get(key)
        try:
            if reusable is None or not reusable.compatible(objectives_by_problem, settings):
                reusable = factory(objectives_by_problem, settings)
                cache[key] = reusable
            return reusable.solve(seed_states, objectives_by_problem, settings)
        except Exception:
            cache.pop(key, None)
            batch_solver = getattr(self.backend, "solve_ik_batch", None)
            if batch_solver is not None:
                return batch_solver(seed_states, objectives_by_problem, settings)
            return [
                self._solve_backend_ik(seed_state, objectives, settings)
                for seed_state, objectives in zip(seed_states, objectives_by_problem)
            ]

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


def _objective_batch_key(objectives) -> tuple[tuple[Any, ...], ...]:
    key: list[tuple[Any, ...]] = []
    for objective in objectives:
        if objective.kind == "position":
            key.append(
                (
                    objective.kind,
                    objective.body_name,
                    objective.semantic_name,
                    _float_tuple(objective.body_local_pos, 3),
                    round(float(objective.weight), 10),
                )
            )
        elif objective.kind == "rotation":
            key.append(
                (
                    objective.kind,
                    objective.body_name,
                    objective.semantic_name,
                    round(float(objective.weight), 10),
                )
            )
        elif objective.kind in {"joint_limit", "posture", "smooth", "damping"}:
            key.append((objective.kind, round(float(objective.weight), 10)))
        else:
            key.append((objective.kind, round(float(objective.weight), 10)))
    return tuple(key)


def _settings_batch_key(settings: NewtonSolveSettings) -> tuple[str, str, float]:
    return (str(settings.optimizer), str(settings.jacobian_mode), float(settings.lambda_initial))


def _float_tuple(value: np.ndarray | None, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,):
        return None
    return tuple(round(float(item), 10) for item in arr)


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
