#!/usr/bin/env python3
"""Bake the SOMA body surface for the BONES-SEED uniform skeleton.

Produces `surface.npz` for our Swift LBS (BodySurface):
  rest_vertices  (V, 3) f32  bind-pose vertices, metres, Z-up (+X forward)
  bind_rot       (J, 4) f32  bind-pose world joint rotations, wxyz, Z-up
  bind_pos       (J, 3) f32  bind-pose world joint positions, metres, Z-up
  weight_joints  (V, K) i32  top-K skinning influences (joint index)
  weight_values  (V, K) f32  normalised weights
  triangles      (F, 3) i32
  joint_names    (J,)   str  must equal the BVH joint order
  vertex_excluded (V,)  u1   1 for inner geometry (mouth bag, eye bags) that never touches anything
  tpose_rot      (J, 4) f32  T-pose world joint rotations (wxyz, Z-up), in the BVH joint-frame convention
  tpose_pos      (J, 3) f32  T-pose world joint positions (metres, Z-up), Hips above the origin

and an alignment fixture (`--fixture`) with SOMA-X's joints and vertices for a few frames of a BVH clip, so the
Swift side can check BVH FK + our LBS against the reference implementation.

numpy only: SOMA-X's low-LOD MHR body model (SOMALayer, identity_model_type="mhr", lod="low", no procedural
transforms or correctives) is reproduced from the fixed arrays in assets/body_models/soma/soma_x_model.npz, exported
from SOMA-X by tools/export_soma_x_model.py, which also checks this against SOMA-X itself. Run with the tooling venv
(build/tools/venv), e.g.
  build/tools/venv/bin/python -I tools/bake_soma_surface.py \
      --shape assets/body_models/soma/seed_soma_shapes/soma_base_fit_mhr_params.npz \
      --bvh ~/Downloads/soma_uniform/bvh/230322/sweep_floor_side_walk_R_004__A298.bvh \
      --out assets/body_models/soma/surface_lo.npz --fixture build/fixtures/soma_lbs_reference.npz
"""

import argparse
import os
import sys

import numpy as np

# Y-up (x, y, z) -> Z-up (z, x, y): same basis change as Core.YUpToZUp.
BASIS = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
CM_TO_M = 0.01  # BVH files are centimetres; SOMALayer outputs metres (its default output_unit).


def parse_bvh(path):
    """Returns joint names, parents, local rotation matrices (T, J, 3, 3), local translations (T, J, 3), fps.

    Native units and basis (cm, Y-up)."""
    with open(path) as f:
        text = f.read()
    head, motion = text.split("MOTION", 1)
    tokens = head.split()
    names, parents, offsets, channels = [], [], [], []
    stack, i, end_site = [], 0, False
    while i < len(tokens):
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            names.append(tokens[i + 1])
            parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3))
            channels.append([])
            i += 2
            continue
        if t == "End":
            end_site = True
            i += 2
            continue
        if t == "{":
            stack.append(None if end_site else len(names) - 1)
        elif t == "}":
            stack.pop()
            end_site = False
        elif t == "OFFSET":
            if not end_site and stack and stack[-1] is not None:
                offsets[stack[-1]] = np.array([float(x) for x in tokens[i + 1:i + 4]])
            i += 4
            continue
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            channels[stack[-1]] = tokens[i + 2:i + 2 + n]
            i += 2 + n
            continue
        i += 1
    # Hierarchy parents index the stack of joints (None = end site).
    parents = [p if p is not None else -1 for p in parents]

    lines = motion.strip().splitlines()
    frames = int(lines[0].split()[1])
    frame_time = float(lines[1].split()[2])
    data = np.array([[float(x) for x in line.split()] for line in lines[2:2 + frames]])

    J = len(names)
    rot = np.tile(np.eye(3), (frames, J, 1, 1))
    trans = np.tile(np.stack(offsets)[None], (frames, 1, 1))
    col = 0
    for j in range(J):
        has_pos = False
        for c in channels[j]:
            v = data[:, col]
            col += 1
            if c.endswith("position"):
                if not has_pos:
                    trans[:, j] = 0
                    has_pos = True
                trans[:, j, "XYZ".index(c[0])] = v
            else:
                a = np.radians(v)
                ca, sa = np.cos(a), np.sin(a)
                m = np.tile(np.eye(3), (frames, 1, 1))
                axis = "XYZ".index(c[0])
                o = [k for k in range(3) if k != axis]
                m[:, o[0], o[0]] = ca
                m[:, o[0], o[1]] = -sa
                m[:, o[1], o[0]] = sa
                m[:, o[1], o[1]] = ca
                if axis == 1:  # rotation about Y has the sign pattern transposed
                    m[:, o[0], o[1]] = sa
                    m[:, o[1], o[0]] = -sa
                rot[:, j] = rot[:, j] @ m
    return names, parents, rot, trans, 1.0 / frame_time


def fk(parents, rot, trans):
    T, J = rot.shape[:2]
    world_r = np.zeros_like(rot)
    world_p = np.zeros_like(trans)
    for j in range(J):
        p = parents[j]
        if p < 0:
            world_r[:, j] = rot[:, j]
            world_p[:, j] = trans[:, j]
        else:
            world_r[:, j] = world_r[:, p] @ rot[:, j]
            world_p[:, j] = world_p[:, p] + np.einsum("tij,tj->ti", world_r[:, p], trans[:, j])
    return world_r, world_p


# MARK: SOMA-X body model (numpy port; array names as in tools/export_soma_x_model.py)

def rodrigues(axis_angle):
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    v = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    k = v / np.maximum(angle, 1e-12)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - np.swapaxes(K, -1, -2)
    a = angle[..., None]
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _qmul(q1, q2):
    """Quaternion product, (x, y, z, w) as in MHR."""
    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def _qrot(q, v):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    av = np.cross(q[..., :3], v)
    return v + 2 * (av * q[..., 3:] + np.cross(q[..., :3], av))


def _state_mul(a, b):
    """MHR skeleton states (t xyz, q xyzw, s): a * b."""
    qa = a[..., 3:7] / np.maximum(np.linalg.norm(a[..., 3:7], axis=-1, keepdims=True), 1e-12)
    qb = b[..., 3:7] / np.maximum(np.linalg.norm(b[..., 3:7], axis=-1, keepdims=True), 1e-12)
    return np.concatenate([a[..., :3] + a[..., 7:] * _qrot(qa, b[..., :3]), _qmul(qa, qb), a[..., 7:] * b[..., 7:]], -1)


def mhr_rest_shape(m, identity, scale):
    """MHR at the zero pose with the identity's body-part scales: (V_mhr, 3) centimetres, Y-up."""
    template = m["mhr_base_shape"] + np.einsum("nvd,n->vd", m["mhr_shape_vectors"].astype(np.float64), identity)
    params = (m["mhr_scale_transform"].astype(np.float64) @ scale).reshape(-1, 7)
    roll, pitch, yaw = (params[:, 3:6] * 0.5).T
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    q = np.stack([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
                  cr * cp * cy + sr * sp * sy], -1)
    state = np.concatenate([params[:, :3] + m["mhr_translation_offsets"], _qmul(m["mhr_prerotations"], q),
                            np.exp(params[:, 6:7] * np.log(2.0))], -1)
    # Prefix multiplication, level by level: child <- parent * child.
    start = 0
    for size in m["mhr_pmi_sizes"]:
        child, parent = m["mhr_pmi"][:, start:start + size]
        state[child] = _state_mul(state[parent], state[child])
        start += size
    joint = _state_mul(state, m["mhr_inverse_bind"].astype(np.float64))[m["mhr_skin_joint"]]
    v = m["mhr_skin_vertex"]
    posed = joint[:, :3] + _qrot(joint[:, 3:7], joint[:, 7:] * template[v])
    out = np.zeros_like(template)
    np.add.at(out, v, posed * m["mhr_skin_weight"][:, None])
    return out


def mhr_to_soma(m, vertices):
    """MHR vertices -> SOMA low-LOD rest shape (metres, Y-up): barycentric transfer, then the Laplacian fill."""
    f = m["transfer_faces"]
    apex = vertices[f[:, 0]] + np.cross(vertices[f[:, 1]] - vertices[f[:, 0]], vertices[f[:, 2]] - vertices[f[:, 0]])
    tet_vertices = np.concatenate([vertices, apex])
    tets = m["transfer_tets"][m["transfer_face"]]
    out = np.einsum("tkd,tk->td", tet_vertices[tets], m["transfer_bary"].astype(np.float64))
    free, anchor = m["lap_free"], m["lap_anchor"]
    fg = np.zeros((len(free), len(anchor)))
    np.add.at(fg, tuple(m["lap_fg_index"]), m["lap_fg_value"])
    out[free] = np.linalg.solve(m["lap_ff"], m["lap_btilde"] - fg @ out[anchor])
    return out * m["unit_to_m"]


def _align(a, b):
    """Rotation R with R b ~ a (SOMA-X's "auto" rotation fit: Newton-Schulz polar factor, Kabsch for reflections)."""
    H = a.T @ b
    if len(a) >= 2:
        na, nb = np.cross(a[0], a[1]), np.cross(b[0], b[1])
        if np.linalg.norm(na) > 1e-9 and np.linalg.norm(nb) > 1e-9:
            va = na * (np.linalg.norm(a[0]) / (np.linalg.norm(na) + 1e-8))
            vb = nb * (np.linalg.norm(b[0]) / (np.linalg.norm(nb) + 1e-8))
            H = H + np.outer(va, vb)
    prior_scale = max(np.abs(H).sum(axis=1).max(), 1e-8)
    volume = abs(np.linalg.det(H)) / prior_scale ** 3
    H = H + 0.05 * np.clip((1e-6 - volume) / 1e-6, 0, 1) * prior_scale * np.eye(3)
    if np.linalg.det(H) < 0:
        U, _, Vt = np.linalg.svd(H)
        return U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    R = H / max(np.abs(H).sum(axis=1).max(), 1e-8)
    for _ in range(30):
        R = R @ (3 * np.eye(3) - R.T @ R) * 0.5
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]
    return R


def _rodrigues_align(target, source):
    """Smallest rotation taking `source` onto `target`'s direction."""
    a = source / max(np.linalg.norm(source), 1e-8)
    b = target / max(np.linalg.norm(target), 1e-8)
    c = np.clip(a @ b, -1, 1)
    if c < -1 + 1e-6:
        u = np.cross(a, [0.0, 1.0, 0.0] if abs(a[0]) > 0.6 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        return 2 * np.outer(u, u) - np.eye(3)
    k = np.cross(a, b)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def fit_skeleton(m, rest):
    """SOMA-X SkeletonTransfer.fit (warp variant): joint positions by regression, rotations by alignment."""
    J = len(m["parents"])
    rbf = np.zeros((J, rest.shape[0]))
    np.add.at(rbf, tuple(m["rbf_index"]), m["rbf_value"])
    t = rbf @ rest
    bind_world = m["bind_world"].astype(np.float64)
    bind_pos, R0 = bind_world[:, :3, 3], bind_world[:, :3, :3]
    parents = m["parents"]
    children = [[] for _ in range(J)]
    for i in range(1, J):
        children[parents[i]].append(i)
    weights, bind_shape = m["skin_weights"], m["bind_shape"].astype(np.float64)
    R = R0.copy()
    for i in range(1, J):
        if not children[i]:
            continue
        vids = np.nonzero(weights[:, i] > 0.01)[0]
        R_init = _align(rest[vids] - t[i], bind_shape[vids] - bind_pos[i])
        source = (R_init @ (bind_pos[children[i]] - bind_pos[i]).T).T
        target = t[children[i]] - t[i]
        align = _rodrigues_align(target[0], source[0]) if len(children[i]) == 1 else _align(target, source)
        R[i] = align @ R_init @ R0[i]
    for i in range(1, J):
        if not children[i]:
            R[i] = R[parents[i]]   # end joints follow their parent (parents come first)
    valid = np.allclose(np.swapaxes(R, -1, -2) @ R, np.eye(3), atol=1e-4) and np.all(np.linalg.det(R) > 0)
    assert valid, "skeleton fit produced an invalid rotation"
    bind = np.tile(np.eye(4), (J, 1, 1))
    bind[:, :3, :3], bind[:, :3, 3] = R, t
    return bind


def identity_rig(m, identity, scale):
    """Rest shape (V, 3) and bind transforms (J, 4, 4) of an identity: metres, Y-up, SOMA's low LOD."""
    rest = mhr_to_soma(m, mhr_rest_shape(m, np.asarray(identity, np.float64).ravel(), np.asarray(scale, np.float64).ravel()))
    return rest, fit_skeleton(m, rest)


def forward_kinematics(m, bind, rotations, translation, absolute):
    """World transforms (B, J, 4, 4) of the bind skeleton posed by local rotations (B, J-1, 3, 3) of the joints after
    the virtual root and the Hips translation (B, 3); without `absolute`, rotations are relative to the T-pose."""
    parents = m["parents"]
    B, J = rotations.shape[0], len(parents)
    local_r = np.concatenate([np.tile(np.eye(3), (B, 1, 1, 1)), rotations], axis=1)
    if not absolute:
        orient = m["t_pose_orient"].astype(np.float64)
        local_r = np.swapaxes(orient[parents], -1, -2)[None] @ local_r @ orient[None]
    inv = np.linalg.inv(bind)
    local_t = np.einsum("jab,jb->ja", inv[parents], np.concatenate([bind[:, :3, 3], np.ones((J, 1))], 1))[:, :3]
    local_t[0] = bind[0, :3, 3]
    local_t = np.tile(local_t, (B, 1, 1))
    local_t[:, 1] = translation
    world = np.tile(np.eye(4), (B, J, 1, 1))
    for j in range(J):
        local = np.tile(np.eye(4), (B, 1, 1))
        local[:, :3, :3], local[:, :3, 3] = local_r[:, j], local_t[:, j]
        world[:, j] = local if parents[j] == j else world[:, parents[j]] @ local
    return world


def skin(m, rest, bind, world):
    """Linear blend skinning with SOMA's full low-LOD weights: (B, V, 3)."""
    skin_t = world @ np.linalg.inv(bind)[None]
    per_joint = np.einsum("bjik,vk->bvji", skin_t[..., :3, :], np.concatenate([rest, np.ones((len(rest), 1))], 1))
    return np.einsum("bvji,vj->bvi", per_joint, m["skin_weights"].astype(np.float64))


def matrix_to_quat_wxyz(m):
    """Batched rotation matrix -> quaternion (w, x, y, z)."""
    m = np.asarray(m, dtype=np.float64)
    q = np.zeros(m.shape[:-2] + (4,))
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q[..., 0] = np.sqrt(np.maximum(0, 1 + tr)) / 2
    q[..., 1] = np.sqrt(np.maximum(0, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2
    q[..., 2] = np.sqrt(np.maximum(0, 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2
    q[..., 3] = np.sqrt(np.maximum(0, 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2
    q[..., 1] = np.copysign(q[..., 1], m[..., 2, 1] - m[..., 1, 2])
    q[..., 2] = np.copysign(q[..., 2], m[..., 0, 2] - m[..., 2, 0])
    q[..., 3] = np.copysign(q[..., 3], m[..., 1, 0] - m[..., 0, 1])
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def to_zup_points(p_m):
    return p_m @ BASIS.T


def to_zup_rotations(r):
    return BASIS @ r @ BASIS.T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets",
                                                    "body_models", "soma", "soma_x_model.npz"),
                    help="SOMA-X model arrays (tools/export_soma_x_model.py)")
    ap.add_argument("--shape", required=True, help="BONES-SEED soma_base_fit_mhr_params.npz")
    ap.add_argument("--bvh", required=True, help="a SOMA uniform BVH clip for the alignment check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture", help="write SOMA-X reference joints/vertices for a few frames")
    ap.add_argument("--max-influences", type=int, default=8)
    ap.add_argument("--frames", default="0,100,400")
    args = ap.parse_args()

    model = dict(np.load(args.model))
    shape = np.load(args.shape)
    rest, bind = identity_rig(model, shape["identity_params"], shape["scale_params"])

    names, parents, rot, trans, fps = parse_bvh(args.bvh)
    if list(model["joint_names"]) != names:
        sys.exit(f"joint order mismatch:\nSOMA {list(model['joint_names'])}\nBVH  {names}")

    frames = [int(f) for f in args.frames.split(",")]
    # Root (index 0) is identity at the origin in SEED clips; SOMA's pose input starts at Hips.
    assert np.allclose(rot[:, 0], np.eye(3)) and np.allclose(trans[:, 0], 0), "non-identity Root not supported"
    trans = trans * CM_TO_M  # work in metres from here on
    soma_transforms = forward_kinematics(model, bind, rot[frames, 1:], trans[frames, 1], absolute=True)  # (B, 78, 4, 4)
    soma_joints = soma_transforms[:, 1:, :3, 3]                     # (B, 77, 3) m, Y-up
    soma_vertices = skin(model, rest, bind, soma_transforms)        # (B, V, 3)

    # 1) Alignment: SOMA-X joints vs our BVH forward kinematics.
    world_r, world_p = fk(parents, rot[frames], trans[frames])
    per_joint = np.linalg.norm(soma_joints - world_p[:, 1:], axis=-1).max(axis=0)
    rot_err = np.abs(soma_transforms[:, :, :3, :3] - world_r).max()
    groups = {
        "legs+feet": [n for n in names if any(k in n for k in ("Leg", "Shin", "Foot", "Toe"))],
        "arms+hands": [n for n in names if any(k in n for k in ("Shoulder", "Arm", "Hand"))],
        "spine+head": [n for n in names if any(k in n for k in ("Spine", "Chest", "Neck", "Head", "Jaw", "Eye"))],
    }
    group_err = {g: max(per_joint[names.index(n) - 1] for n in members) for g, members in groups.items()}
    joint_err = group_err["legs+feet"]
    print(f"alignment (SOMA-X FK vs BVH FK): rotations max |dR| {rot_err:.3g}; joint position max error "
          + ", ".join(f"{g} {e * 100:.2f} cm" for g, e in group_err.items()))

    # 2) Skinning weights SOMA-X uses for this LOD (dense V x J).
    weights = model["skin_weights"].astype(np.float64)
    if weights.shape[0] != rest.shape[0]:
        sys.exit(f"weights {weights.shape} do not match rest shape {rest.shape}")

    K = args.max_influences
    order = np.argsort(-weights, axis=1)[:, :K]
    top = np.take_along_axis(weights, order, axis=1)
    dropped = 1 - top.sum(axis=1)
    top = top / top.sum(axis=1, keepdims=True)

    # Our LBS in native units, to check the bake before converting.
    def lbs(world4):
        skin = world4 @ np.linalg.inv(bind)[None]  # (B, J, 4, 4)
        homo = np.concatenate([rest, np.ones((rest.shape[0], 1))], axis=1)
        per = np.einsum("bjik,vk->bvji", skin, homo)[..., :3]  # (B, V, J, 3)
        gathered = np.take_along_axis(per, order[None, :, :, None].repeat(len(frames), 0).repeat(3, -1), axis=2)
        return (gathered * top[None, :, :, None]).sum(axis=2)

    ours = lbs(soma_transforms)
    lbs_err = np.abs(ours - soma_vertices).max()
    print(f"LBS (top-{K}) vs SOMA-X vertices: max error {lbs_err * 100:.4g} cm; max dropped weight {dropped.max():.3g}")

    faces = model["triangles"]

    # T-pose of this identity (relative pose = identity), the reference for robot rotation offsets.
    tpose_t = forward_kinematics(model, bind, np.tile(np.eye(3), (1, len(names) - 1, 1, 1)), np.zeros((1, 3)),
                                 absolute=False)[0]
    tpose_t[:, :3, 3] -= np.array([0.0, tpose_t[1:, 1, 3].min(), 0.0])  # lowest joint on y = 0

    # Inner geometry (mouth bag, eye bags), from SOMA's semantic segments.
    excluded = model["vertex_excluded"].astype(bool)
    print(f"excluded inner vertices: {int(excluded.sum())}")

    # Convert to our frame.
    bind_r = to_zup_rotations(bind[:, :3, :3])
    out_data = dict(
        rest_vertices=to_zup_points(rest).astype(np.float32),
        bind_rot=matrix_to_quat_wxyz(bind_r).astype(np.float32),
        bind_pos=to_zup_points(bind[:, :3, 3]).astype(np.float32),
        weight_joints=order.astype(np.int32),
        weight_values=top.astype(np.float32),
        triangles=faces.astype(np.int32),
        joint_names=np.array(names),
        vertex_excluded=excluded.astype(np.uint8),
        tpose_rot=matrix_to_quat_wxyz(to_zup_rotations(tpose_t[:, :3, :3])).astype(np.float32),
        tpose_pos=to_zup_points(tpose_t[:, :3, 3]).astype(np.float32),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out_data)
    print(f"wrote {args.out}: {rest.shape[0]} vertices, {faces.shape[0]} triangles, {len(names)} joints")

    if args.fixture:
        os.makedirs(os.path.dirname(os.path.abspath(args.fixture)), exist_ok=True)
        np.savez(args.fixture,
                 bvh=np.array(os.path.abspath(args.bvh)),
                 frames=np.array(frames, dtype=np.int32),
                 joints=to_zup_points(soma_joints).astype(np.float32),
                 transform_rot=matrix_to_quat_wxyz(to_zup_rotations(soma_transforms[..., :3, :3])).astype(np.float32),
                 transform_pos=to_zup_points(soma_transforms[..., :3, 3]).astype(np.float32),
                 vertices=to_zup_points(soma_vertices).astype(np.float32))
        print(f"wrote {args.fixture}")

    # The SEED shape fit and the uniform BVH skeleton differ by a few mm per bone (~1.5 cm accumulated at the
    # head). Our LBS skins with BVH FK transforms, so the surface follows the BVH skeleton; we gate on the
    # contact-relevant legs/feet and on LBS fidelity.
    ok = joint_err < 0.01 and lbs_err < 1e-3  # metres
    print("alignment OK" if ok else "alignment FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
