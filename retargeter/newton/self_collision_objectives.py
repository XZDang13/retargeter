from __future__ import annotations

import numpy as np

import newton.ik as ik
import warp as wp

from newton._src.sim.ik.ik_common import IKJacobianType

from .objectives import SelfCollisionPairSpec

SPHERE_SHAPE = 0
CAPSULE_SHAPE = 1


@wp.kernel
def _self_collision_residuals(
    body_q: wp.array2d(dtype=wp.transform),
    point_body_indices: wp.array1d(dtype=wp.int32),
    point_local_pos: wp.array1d(dtype=wp.vec3),
    obstacle_body_indices: wp.array1d(dtype=wp.int32),
    obstacle_shapes: wp.array1d(dtype=wp.int32),
    obstacle_centers: wp.array1d(dtype=wp.vec3),
    obstacle_point_a: wp.array1d(dtype=wp.vec3),
    obstacle_point_b: wp.array1d(dtype=wp.vec3),
    obstacle_radii: wp.array1d(dtype=wp.float32),
    margins: wp.array1d(dtype=wp.float32),
    weight: wp.array1d(dtype=wp.float32),
    residual_start: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    problem, pair_idx = wp.tid()
    point_tf = body_q[problem, point_body_indices[pair_idx]]
    obstacle_tf = body_q[problem, obstacle_body_indices[pair_idx]]

    point_w = wp.transform_point(point_tf, point_local_pos[pair_idx])
    center_w = wp.transform_point(obstacle_tf, obstacle_centers[pair_idx])

    if obstacle_shapes[pair_idx] == CAPSULE_SHAPE:
        a_w = wp.transform_point(obstacle_tf, obstacle_point_a[pair_idx])
        b_w = wp.transform_point(obstacle_tf, obstacle_point_b[pair_idx])
        ab = b_w - a_w
        denom = wp.dot(ab, ab)
        t = wp.float32(0.0)
        if denom > 1.0e-8:
            t = wp.clamp(wp.dot(point_w - a_w, ab) / denom, 0.0, 1.0)
        center_w = a_w + t * ab

    delta = point_w - center_w
    distance = wp.sqrt(wp.dot(delta, delta) + 1.0e-12)
    clearance = distance - obstacle_radii[pair_idx]
    violation = wp.max(wp.float32(0.0), margins[pair_idx] - clearance)
    residuals[problem, residual_start + pair_idx] = weight[0] * violation


@wp.kernel
def _self_collision_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),
    n_dofs: int,
    residual_start: int,
    pair_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_idx, dof_idx = wp.tid()
    if dof_idx < n_dofs:
        jacobian[problem_idx, residual_start + pair_idx, dof_idx] = q_grad[problem_idx, dof_idx]


@wp.func
def _point_velocity(joint_s: wp.spatial_vector, point_w: wp.vec3):
    v_orig = wp.vec3(joint_s[0], joint_s[1], joint_s[2])
    omega = wp.vec3(joint_s[3], joint_s[4], joint_s[5])
    return v_orig + wp.cross(omega, point_w)


@wp.kernel
def _self_collision_jac_analytic(
    body_q: wp.array2d(dtype=wp.transform),
    joint_s_s: wp.array2d(dtype=wp.spatial_vector),
    point_body_indices: wp.array1d(dtype=wp.int32),
    point_local_pos: wp.array1d(dtype=wp.vec3),
    obstacle_body_indices: wp.array1d(dtype=wp.int32),
    obstacle_shapes: wp.array1d(dtype=wp.int32),
    obstacle_centers: wp.array1d(dtype=wp.vec3),
    obstacle_point_a: wp.array1d(dtype=wp.vec3),
    obstacle_point_b: wp.array1d(dtype=wp.vec3),
    obstacle_radii: wp.array1d(dtype=wp.float32),
    margins: wp.array1d(dtype=wp.float32),
    point_affects_dof: wp.array2d(dtype=wp.int32),
    obstacle_affects_dof: wp.array2d(dtype=wp.int32),
    weight: wp.array1d(dtype=wp.float32),
    n_dofs: int,
    residual_start: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem, pair_idx, dof_idx = wp.tid()
    if dof_idx >= n_dofs:
        return

    point_tf = body_q[problem, point_body_indices[pair_idx]]
    obstacle_tf = body_q[problem, obstacle_body_indices[pair_idx]]

    point_w = wp.transform_point(point_tf, point_local_pos[pair_idx])
    center_w = wp.transform_point(obstacle_tf, obstacle_centers[pair_idx])
    t = wp.float32(0.0)
    a_w = center_w
    b_w = center_w
    if obstacle_shapes[pair_idx] == CAPSULE_SHAPE:
        a_w = wp.transform_point(obstacle_tf, obstacle_point_a[pair_idx])
        b_w = wp.transform_point(obstacle_tf, obstacle_point_b[pair_idx])
        ab = b_w - a_w
        denom = wp.dot(ab, ab)
        if denom > 1.0e-8:
            t = wp.clamp(wp.dot(point_w - a_w, ab) / denom, 0.0, 1.0)
        center_w = a_w + t * ab

    delta = point_w - center_w
    distance = wp.sqrt(wp.dot(delta, delta) + 1.0e-12)
    clearance = distance - obstacle_radii[pair_idx]
    if clearance >= margins[pair_idx]:
        jacobian[problem, residual_start + pair_idx, dof_idx] = 0.0
        return

    normal = delta / distance
    joint_s = joint_s_s[problem, dof_idx]
    grad = wp.float32(0.0)

    if point_affects_dof[pair_idx, dof_idx] != 0:
        point_vel = _point_velocity(joint_s, point_w)
        grad = grad - weight[0] * wp.dot(normal, point_vel)

    if obstacle_affects_dof[pair_idx, dof_idx] != 0:
        center_vel = _point_velocity(joint_s, center_w)
        if obstacle_shapes[pair_idx] == CAPSULE_SHAPE:
            a_vel = _point_velocity(joint_s, a_w)
            b_vel = _point_velocity(joint_s, b_w)
            center_vel = (1.0 - t) * a_vel + t * b_vel
        grad = grad + weight[0] * wp.dot(normal, center_vel)

    jacobian[problem, residual_start + pair_idx, dof_idx] = grad


@wp.kernel
def _update_weight(
    value: wp.float32,
    out_weight: wp.array1d(dtype=wp.float32),
):
    out_weight[0] = value


class IKSelfCollisionObjective(ik.IKObjective):
    """Penalize selected robot point/proxy pairs that are inside a clearance margin."""

    def __init__(
        self,
        pairs: tuple[SelfCollisionPairSpec, ...],
        *,
        body_name_to_index: dict[str, int],
        weight: float,
    ):
        super().__init__()
        if not pairs:
            raise ValueError("IKSelfCollisionObjective requires at least one pair.")
        self.pairs = tuple(pairs)
        self.pair_names = [pair.name for pair in self.pairs]
        self.weight = float(weight)
        self._weight = None
        self._e_arrays = None

        point_indices = []
        obstacle_indices = []
        shapes = []
        point_offsets = []
        obstacle_centers = []
        obstacle_a = []
        obstacle_b = []
        radii = []
        margins = []
        for pair in self.pairs:
            point_indices.append(int(body_name_to_index[pair.point_body]))
            obstacle_indices.append(int(body_name_to_index[pair.obstacle_body]))
            shapes.append(SPHERE_SHAPE if pair.obstacle_shape == "sphere" else CAPSULE_SHAPE)
            point_offsets.append(pair.point_local_pos)
            obstacle_centers.append(pair.obstacle_center)
            obstacle_a.append(pair.obstacle_point_a)
            obstacle_b.append(pair.obstacle_point_b)
            radii.append(pair.obstacle_radius_m)
            margins.append(pair.margin_m)

        self.point_body_indices_np = np.asarray(point_indices, dtype=np.int32)
        self.obstacle_body_indices_np = np.asarray(obstacle_indices, dtype=np.int32)
        self.obstacle_shapes_np = np.asarray(shapes, dtype=np.int32)
        self.point_local_pos_np = np.asarray(point_offsets, dtype=np.float32)
        self.obstacle_centers_np = np.asarray(obstacle_centers, dtype=np.float32)
        self.obstacle_point_a_np = np.asarray(obstacle_a, dtype=np.float32)
        self.obstacle_point_b_np = np.asarray(obstacle_b, dtype=np.float32)
        self.obstacle_radii_np = np.asarray(radii, dtype=np.float32)
        self.margins_np = np.asarray(margins, dtype=np.float32)

        self.point_body_indices = None
        self.obstacle_body_indices = None
        self.obstacle_shapes = None
        self.point_local_pos = None
        self.obstacle_centers = None
        self.obstacle_point_a = None
        self.obstacle_point_b = None
        self.obstacle_radii = None
        self.margins = None
        self.point_affects_dof = None
        self.obstacle_affects_dof = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        self._weight = wp.array([self.weight], dtype=wp.float32, device=self.device)
        self.point_body_indices = wp.array(self.point_body_indices_np, dtype=wp.int32, device=self.device)
        self.obstacle_body_indices = wp.array(self.obstacle_body_indices_np, dtype=wp.int32, device=self.device)
        self.obstacle_shapes = wp.array(self.obstacle_shapes_np, dtype=wp.int32, device=self.device)
        self.point_local_pos = wp.array(self.point_local_pos_np, dtype=wp.vec3, device=self.device)
        self.obstacle_centers = wp.array(self.obstacle_centers_np, dtype=wp.vec3, device=self.device)
        self.obstacle_point_a = wp.array(self.obstacle_point_a_np, dtype=wp.vec3, device=self.device)
        self.obstacle_point_b = wp.array(self.obstacle_point_b_np, dtype=wp.vec3, device=self.device)
        self.obstacle_radii = wp.array(self.obstacle_radii_np, dtype=wp.float32, device=self.device)
        self.margins = wp.array(self.margins_np, dtype=wp.float32, device=self.device)

        if jacobian_mode == IKJacobianType.AUTODIFF:
            self._e_arrays = []
            for pair_idx in range(self.residual_dim()):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for problem_idx in range(self.n_batch):
                    e[problem_idx, self.residual_offset + pair_idx] = 1.0
                self._e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))
        elif jacobian_mode == IKJacobianType.ANALYTIC:
            point_affects, obstacle_affects = _pair_affects_dof(
                model,
                self.point_body_indices_np,
                self.obstacle_body_indices_np,
            )
            self.point_affects_dof = wp.array(point_affects, dtype=wp.int32, device=self.device)
            self.obstacle_affects_dof = wp.array(obstacle_affects, dtype=wp.int32, device=self.device)
        else:
            raise ValueError(f"Unsupported Jacobian mode for IKSelfCollisionObjective: {jacobian_mode!r}.")

    def residual_dim(self):
        return len(self.pairs)

    def supports_analytic(self):
        return True

    def set_weight(self, value: float) -> None:
        self.weight = float(value)
        if self._weight is None:
            return
        wp.launch(_update_weight, dim=1, inputs=[float(value)], outputs=[self._weight], device=self.device)

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        count = body_q.shape[0]
        wp.launch(
            _self_collision_residuals,
            dim=[count, self.residual_dim()],
            inputs=[
                body_q,
                self.point_body_indices,
                self.point_local_pos,
                self.obstacle_body_indices,
                self.obstacle_shapes,
                self.obstacle_centers,
                self.obstacle_point_a,
                self.obstacle_point_b,
                self.obstacle_radii,
                self.margins,
                self._weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        self._require_batch_layout()
        if self._e_arrays is None:
            raise RuntimeError("IKSelfCollisionObjective buffers are not initialized.")
        n_dofs = model.joint_dof_count
        for pair_idx, e_array in enumerate(self._e_arrays):
            tape.backward(grads={tape.outputs[0]: e_array})
            q_grad = tape.gradients[dq_dof]
            wp.launch(
                _self_collision_jac_fill,
                dim=[self.n_batch, n_dofs],
                inputs=[q_grad, n_dofs, start_idx, pair_idx],
                outputs=[jacobian],
                device=self.device,
            )
            tape.zero()

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        if self.point_affects_dof is None or self.obstacle_affects_dof is None:
            raise RuntimeError("IKSelfCollisionObjective analytic buffers are not initialized.")
        n_dofs = model.joint_dof_count
        wp.launch(
            _self_collision_jac_analytic,
            dim=[body_q.shape[0], self.residual_dim(), n_dofs],
            inputs=[
                body_q,
                joint_S_s,
                self.point_body_indices,
                self.point_local_pos,
                self.obstacle_body_indices,
                self.obstacle_shapes,
                self.obstacle_centers,
                self.obstacle_point_a,
                self.obstacle_point_b,
                self.obstacle_radii,
                self.margins,
                self.point_affects_dof,
                self.obstacle_affects_dof,
                self._weight,
                n_dofs,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )


def _pair_affects_dof(model, point_body_indices: np.ndarray, obstacle_body_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point_affects = _bodies_affect_dof(model, point_body_indices)
    obstacle_affects = _bodies_affect_dof(model, obstacle_body_indices)
    return point_affects, obstacle_affects


def _bodies_affect_dof(model, body_indices: np.ndarray) -> np.ndarray:
    joint_qd_start_np = model.joint_qd_start.numpy()
    dof_count = int(joint_qd_start_np[-1])
    dof_to_joint_np = np.empty(dof_count, dtype=np.int32)
    for joint_index in range(len(joint_qd_start_np) - 1):
        dof_to_joint_np[joint_qd_start_np[joint_index] : joint_qd_start_np[joint_index + 1]] = joint_index

    joint_child_np = model.joint_child.numpy()
    body_to_joint_np = np.full(model.body_count, -1, dtype=np.int32)
    for joint_index in range(model.joint_count):
        child = int(joint_child_np[joint_index])
        if child != -1:
            body_to_joint_np[child] = joint_index

    joint_parent_np = model.joint_parent.numpy()
    affects = np.zeros((len(body_indices), dof_count), dtype=np.int32)
    for pair_index, body_index in enumerate(body_indices):
        ancestors = np.zeros(model.joint_count, dtype=bool)
        body = int(body_index)
        while body != -1:
            joint_index = int(body_to_joint_np[body])
            if joint_index == -1:
                break
            ancestors[joint_index] = True
            body = int(joint_parent_np[joint_index])
        affects[pair_index] = ancestors[dof_to_joint_np].astype(np.int32)
    return affects
