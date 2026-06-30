from __future__ import annotations

import numpy as np

import newton.ik as ik
import warp as wp

from newton._src.sim.ik.ik_common import IKJacobianType


@wp.kernel
def _joint_target_residuals(
    joint_q: wp.array2d(dtype=wp.float32),
    target_q: wp.array2d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    coord_start: int,
    residual_start: int,
    problem_idx: wp.array1d(dtype=wp.int32),
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, local_dof = wp.tid()
    target_problem = problem_idx[problem]
    residuals[problem, residual_start + local_dof] = (
        joint_q[problem, coord_start + local_dof] - target_q[target_problem, local_dof]
    ) * weight[0]


@wp.kernel
def _joint_target_jacobian(
    n_dofs: int,
    weight: wp.array1d(dtype=wp.float32),
    dof_start: int,
    residual_start: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, local_dof = wp.tid()
    if local_dof < n_dofs:
        jacobian[problem, residual_start + local_dof, dof_start + local_dof] = weight[0]


@wp.kernel
def _update_joint_target_weight(
    value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),
):
    out_weight[0] = value


@wp.kernel
def _update_joint_target(
    problem_index: int,
    values: wp.array1d(dtype=wp.float32),
    target_q: wp.array2d(dtype=wp.float32),
):
    local_dof = wp.tid()
    target_q[problem_index, local_dof] = values[local_dof]


class IKJointTargetObjective(ik.IKObjective):
    """Penalize actuated joint-coordinate deviation from per-problem targets."""

    def __init__(
        self,
        target_q: np.ndarray,
        *,
        weight: float,
        coord_start: int,
        dof_start: int,
    ):
        super().__init__()
        target = np.asarray(target_q, dtype=np.float32)
        if target.ndim != 2:
            raise ValueError(f"target_q must have shape [problem_count, num_dofs], got {target.shape}.")
        self.target_q_np = target.copy()
        self.target_q = None
        self.n_dofs = int(target.shape[1])
        self.coord_start = int(coord_start)
        self.dof_start = int(dof_start)
        self._weight = wp.array([float(weight)], dtype=wp.float32)
        self._autodiff_rows = None

    def bind_device(self, device):
        super().bind_device(device)

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        self.target_q = wp.array(self.target_q_np, dtype=wp.float32, device=self.device)
        if jacobian_mode == IKJacobianType.AUTODIFF:
            rows = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for problem in range(self.n_batch):
                for local_dof in range(self.n_dofs):
                    rows[problem, self.residual_offset + local_dof] = 1.0
            self._autodiff_rows = wp.array(rows.flatten(), dtype=wp.float32, device=self.device)

    def residual_dim(self):
        return self.n_dofs

    def supports_analytic(self):
        return True

    def set_target(self, problem_index: int, target_q: np.ndarray) -> None:
        if self.target_q is None:
            self.target_q_np[int(problem_index)] = np.asarray(target_q, dtype=np.float32)
            return
        values = np.asarray(target_q, dtype=np.float32)
        if values.shape != (self.n_dofs,):
            raise ValueError(f"target_q must have shape [{self.n_dofs}], got {values.shape}.")
        wp.launch(
            _update_joint_target,
            dim=self.n_dofs,
            inputs=[int(problem_index), wp.array(values, dtype=wp.float32, device=self.device)],
            outputs=[self.target_q],
            device=self.device,
        )

    def set_weight(self, value: float) -> None:
        wp.launch(_update_joint_target_weight, dim=1, inputs=[float(value)], outputs=[self._weight], device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = joint_q.shape[0]
        wp.launch(
            _joint_target_residuals,
            dim=[count, self.n_dofs],
            inputs=[joint_q, self.target_q, self._weight, self.coord_start, start_idx, problem_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        if self._autodiff_rows is not None:
            tape.backward(grads={tape.outputs[0]: self._autodiff_rows})
        self.compute_jacobian_analytic(None, None, model, jacobian, None, start_idx)

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        wp.launch(
            _joint_target_jacobian,
            dim=[self.n_batch, self.n_dofs],
            inputs=[self.n_dofs, self._weight, self.dof_start, start_idx],
            outputs=[jacobian],
            device=self.device,
        )
