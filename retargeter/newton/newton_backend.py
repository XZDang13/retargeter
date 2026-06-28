from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from retargeter.preprocess.lowpass import normalize_quat_xyzw

from .objectives import IKObjectiveDescriptor
from .robot_spec import RobotSpec


@dataclass
class IKState:
    root_pos_w: np.ndarray
    root_quat_xyzw: np.ndarray
    joint_pos: np.ndarray

    def validate(self, robot_spec: RobotSpec) -> None:
        root_pos = np.asarray(self.root_pos_w, dtype=np.float64)
        root_quat = np.asarray(self.root_quat_xyzw, dtype=np.float64)
        joint_pos = np.asarray(self.joint_pos, dtype=np.float64)
        if root_pos.shape != (3,):
            raise ValueError(f"root_pos_w must have shape [3], got {root_pos.shape}.")
        if root_quat.shape != (4,):
            raise ValueError(f"root_quat_xyzw must have shape [4], got {root_quat.shape}.")
        if joint_pos.shape != (robot_spec.num_dofs,):
            raise ValueError(f"joint_pos must have shape [{robot_spec.num_dofs}], got {joint_pos.shape}.")
        if not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(root_quat)) or not np.all(np.isfinite(joint_pos)):
            raise ValueError("IKState contains NaN or inf values.")

    def copy(self) -> "IKState":
        return IKState(
            root_pos_w=np.asarray(self.root_pos_w, dtype=np.float64).copy(),
            root_quat_xyzw=normalize_quat_xyzw(np.asarray(self.root_quat_xyzw, dtype=np.float64)).copy(),
            joint_pos=np.asarray(self.joint_pos, dtype=np.float64).copy(),
        )


@dataclass
class RobotBodyState:
    body_names: list[str]
    body_pos_w: np.ndarray
    body_quat_xyzw: np.ndarray

    def validate(self) -> None:
        pos = np.asarray(self.body_pos_w, dtype=np.float64)
        quat = np.asarray(self.body_quat_xyzw, dtype=np.float64)
        if pos.shape != (len(self.body_names), 3):
            raise ValueError(f"body_pos_w must have shape [{len(self.body_names)}, 3], got {pos.shape}.")
        if quat.shape != (len(self.body_names), 4):
            raise ValueError(f"body_quat_xyzw must have shape [{len(self.body_names)}, 4], got {quat.shape}.")
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
            raise ValueError("RobotBodyState contains NaN or inf values.")


@dataclass
class NewtonSolveSettings:
    iterations: int = 24
    step_size: float = 1.0
    optimizer: str = "lm"
    jacobian_mode: str = "analytic"
    lambda_initial: float = 0.1


@dataclass
class BackendSolveResult:
    state: IKState
    success: bool
    cost: float | None = None
    iterations: int = 0
    diagnostics: dict = field(default_factory=dict)


class IKBackend(Protocol):
    robot_spec: RobotSpec

    def solve_ik(
        self,
        seed_state: IKState,
        objectives: list[IKObjectiveDescriptor],
        settings: NewtonSolveSettings,
    ) -> BackendSolveResult:
        ...

    def forward_kinematics(self, state: IKState) -> RobotBodyState:
        ...


class NewtonBackend:
    """Small adapter around Newton's IK API.

    All direct Newton/Warp imports stay inside this class so unit tests can use
    a mock backend without importing GPU/runtime packages.
    """

    def __init__(
        self,
        robot_spec: RobotSpec,
        *,
        load_visual_shapes: bool = False,
        add_ground_plane: bool = False,
        ground_height: float = 0.0,
    ):
        self.robot_spec = robot_spec
        self.load_visual_shapes = load_visual_shapes
        self.add_ground_plane = add_ground_plane
        self.ground_height = float(ground_height)
        self._newton = None
        self._wp = None
        self._model = None
        self._state = None
        self._body_name_to_newton_index: dict[str, int] = {}

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    def solve_ik(
        self,
        seed_state: IKState,
        objectives: list[IKObjectiveDescriptor],
        settings: NewtonSolveSettings,
    ) -> BackendSolveResult:
        self._ensure_loaded()
        assert self._newton is not None and self._wp is not None and self._model is not None
        seed_state.validate(self.robot_spec)
        for objective in objectives:
            objective.validate(self.robot_spec)

        seed = self._apply_seed_bias(seed_state, objectives)
        full_q = self._state_to_full_q(seed)
        joint_q_in = self._wp.array(full_q[None, :].astype(np.float32), dtype=self._wp.float32)
        joint_q_out = self._wp.array(full_q[None, :].astype(np.float32), dtype=self._wp.float32)

        native_objectives = self._build_native_objectives(objectives)
        diagnostics = {
            "objective_count": len(objectives),
            "native_objective_count": len(native_objectives),
            "regularization_seed_bias": True,
        }

        if not native_objectives:
            return BackendSolveResult(
                state=seed,
                success=True,
                cost=0.0,
                iterations=0,
                diagnostics=diagnostics,
            )

        try:
            solver = self._newton.ik.IKSolver(
                self._model,
                1,
                native_objectives,
                optimizer=settings.optimizer,
                jacobian_mode=settings.jacobian_mode,
                lambda_initial=settings.lambda_initial,
            )
            solver.step(joint_q_in, joint_q_out, iterations=int(settings.iterations), step_size=float(settings.step_size))
            solved_full_q = joint_q_out.numpy()[0].astype(np.float64)
            state = self._full_q_to_state(solved_full_q)
            costs = solver.costs.numpy()
            cost = float(np.min(costs)) if costs.size else None
            return BackendSolveResult(
                state=state,
                success=True,
                cost=cost,
                iterations=int(settings.iterations),
                diagnostics=diagnostics,
            )
        except Exception as exc:  # pragma: no cover - exercised only with real Newton failures.
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return BackendSolveResult(
                state=seed,
                success=False,
                cost=None,
                iterations=0,
                diagnostics=diagnostics,
            )

    def forward_kinematics(self, state: IKState) -> RobotBodyState:
        self._ensure_loaded()
        assert self._newton is not None and self._wp is not None and self._model is not None
        state.validate(self.robot_spec)

        fk_state = self.make_newton_state(state)
        body_q = fk_state.body_q.numpy().astype(np.float64)

        pos = np.zeros((len(self.robot_spec.body_names), 3), dtype=np.float64)
        quat = np.zeros((len(self.robot_spec.body_names), 4), dtype=np.float64)
        quat[:, 3] = 1.0
        for idx, body_name in enumerate(self.robot_spec.body_names):
            newton_idx = self._body_name_to_newton_index.get(body_name)
            if newton_idx is None:
                continue
            pos[idx] = body_q[newton_idx, :3]
            quat[idx] = normalize_quat_xyzw(body_q[newton_idx, 3:7])

        body_state = RobotBodyState(list(self.robot_spec.body_names), pos, quat)
        body_state.validate()
        return body_state

    def state_to_full_q(self, state: IKState) -> np.ndarray:
        """Return a full Newton joint_q vector for an actuated Stage 1 state."""
        self._ensure_loaded()
        state.validate(self.robot_spec)
        return self._state_to_full_q(state)

    def make_newton_state(self, state: IKState):
        """Run Newton FK and return a native Newton State for viewer/replay use."""
        self._ensure_loaded()
        assert self._newton is not None and self._wp is not None and self._model is not None
        state.validate(self.robot_spec)

        full_q = self._state_to_full_q(state)
        joint_q = self._wp.array(full_q.astype(np.float32), dtype=self._wp.float32)
        joint_qd = self._wp.zeros(self._model.joint_dof_count, dtype=self._wp.float32)
        fk_state = self._model.state()
        self._newton.eval_fk(self._model, joint_q, joint_qd, fk_state)
        self._wp.synchronize()
        return fk_state

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import newton
            import warp as wp
        except ImportError as exc:  # pragma: no cover - depends on optional Newton install.
            raise RuntimeError("Newton and Warp are required to use NewtonBackend.") from exc

        builder = newton.ModelBuilder()
        if self.robot_spec.model_format == "usd":
            info = builder.add_usd(
                str(self.robot_spec.model_path),
                floating=self.robot_spec.floating_base,
                collapse_fixed_joints=False,
                load_visual_shapes=self.load_visual_shapes,
                skip_mesh_approximation=True,
            )
        elif self.robot_spec.model_format == "mjcf":
            info = builder.add_mjcf(str(self.robot_spec.model_path), floating=self.robot_spec.floating_base)
        elif self.robot_spec.model_format == "urdf":
            info = builder.add_urdf(str(self.robot_spec.model_path), floating=self.robot_spec.floating_base)
        else:
            raise ValueError(f"Unsupported model_format {self.robot_spec.model_format!r}.")

        if self.add_ground_plane:
            builder.add_ground_plane(height=self.ground_height, label="ground")

        self._newton = newton
        self._wp = wp
        self._model = builder.finalize(requires_grad=True)
        self._state = self._model.state()
        self._body_name_to_newton_index = _body_name_map_from_import_info(info)

        missing = [name for name in self.robot_spec.body_names if name not in self._body_name_to_newton_index]
        if missing:
            raise RuntimeError(f"Newton model is missing robot spec bodies: {missing}.")

    def _build_native_objectives(self, objectives: list[IKObjectiveDescriptor]):
        assert self._newton is not None and self._wp is not None and self._model is not None
        native = []
        for objective in objectives:
            if objective.kind == "position":
                body_index = self._body_name_to_newton_index[objective.body_name or ""]
                local_pos = (
                    np.zeros(3, dtype=np.float64)
                    if objective.body_local_pos is None
                    else np.asarray(objective.body_local_pos, dtype=np.float64)
                )
                target = self._wp.array(np.asarray([objective.target], dtype=np.float32), dtype=self._wp.vec3)
                native.append(
                    self._newton.ik.IKObjectivePosition(
                        body_index,
                        self._wp.vec3(float(local_pos[0]), float(local_pos[1]), float(local_pos[2])),
                        target,
                        weight=float(objective.weight),
                    )
                )
            elif objective.kind == "rotation":
                body_index = self._body_name_to_newton_index[objective.body_name or ""]
                target = self._wp.array(np.asarray([objective.target], dtype=np.float32), dtype=self._wp.vec4)
                native.append(
                    self._newton.ik.IKObjectiveRotation(
                        body_index,
                        self._wp.quat(0.0, 0.0, 0.0, 1.0),
                        target,
                        weight=float(objective.weight),
                    )
                )
            elif objective.kind == "joint_limit":
                native.append(
                    self._newton.ik.IKObjectiveJointLimit(
                        self._model.joint_limit_lower,
                        self._model.joint_limit_upper,
                        weight=float(objective.weight),
                    )
                )
        return native

    def _apply_seed_bias(self, seed_state: IKState, objectives: list[IKObjectiveDescriptor]) -> IKState:
        q = np.asarray(seed_state.joint_pos, dtype=np.float64).copy()
        total_weight = 1.0
        for objective in objectives:
            if objective.kind not in {"posture", "smooth", "damping"}:
                continue
            target = np.asarray(objective.target, dtype=np.float64)
            weight = float(objective.weight)
            q += target * weight
            total_weight += weight
        q /= total_weight
        return IKState(
            root_pos_w=np.asarray(seed_state.root_pos_w, dtype=np.float64).copy(),
            root_quat_xyzw=normalize_quat_xyzw(seed_state.root_quat_xyzw).copy(),
            joint_pos=q,
        )

    def _state_to_full_q(self, state: IKState) -> np.ndarray:
        assert self._model is not None
        full_q = self._model.joint_q.numpy().astype(np.float64).copy()
        if self.robot_spec.floating_base:
            full_q[:3] = np.asarray(state.root_pos_w, dtype=np.float64)
            full_q[3:7] = normalize_quat_xyzw(state.root_quat_xyzw)
            start = 7
        else:
            start = 0
        end = start + self.robot_spec.num_dofs
        if end > full_q.shape[0]:
            raise RuntimeError(
                f"Robot spec has {self.robot_spec.num_dofs} DoFs but Newton model has "
                f"{full_q.shape[0] - start} actuated coordinates."
            )
        full_q[start:end] = np.asarray(state.joint_pos, dtype=np.float64)
        return full_q

    def _full_q_to_state(self, full_q: np.ndarray) -> IKState:
        if self.robot_spec.floating_base:
            root_pos = full_q[:3]
            root_quat = full_q[3:7]
            start = 7
        else:
            root_pos = np.zeros(3, dtype=np.float64)
            root_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            start = 0
        joint_pos = full_q[start : start + self.robot_spec.num_dofs]
        return IKState(
            root_pos_w=root_pos.copy(),
            root_quat_xyzw=normalize_quat_xyzw(root_quat).copy(),
            joint_pos=joint_pos.copy(),
        )


def _body_name_map_from_import_info(info) -> dict[str, int]:
    if not isinstance(info, dict):
        return {}
    path_body_map = info.get("path_body_map", {})
    name_to_index: dict[str, int] = {}
    for path, index in path_body_map.items():
        name = str(path).rstrip("/").split("/")[-1]
        name_to_index[name] = int(index)
    return name_to_index
