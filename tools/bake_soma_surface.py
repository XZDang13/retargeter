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

and an alignment fixture (`--fixture`) with SOMA-X's own joints and vertices for a few frames of a
BVH clip, so the Swift side can check BVH FK + our LBS against the reference implementation.

Run with the tooling venv (build/tools/venv), e.g.
  build/tools/venv/bin/python -I tools/bake_soma_surface.py \
      --soma-x build/tools/SOMA-X/assets \
      --shape assets/body_models/soma/seed_soma_shapes/soma_base_fit_mhr_params.npz \
      --bvh ~/Downloads/soma_uniform/bvh/230322/sweep_floor_side_walk_R_004__A298.bvh \
      --out assets/body_models/soma/surface_lo.npz --fixture build/fixtures/soma_lbs_reference.npz
"""

import argparse
import os
import sys

import numpy as np
import torch

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
    ap.add_argument("--soma-x", required=True, help="SOMA-X assets directory")
    ap.add_argument("--shape", required=True, help="BONES-SEED soma_base_fit_mhr_params.npz")
    ap.add_argument("--bvh", required=True, help="a SOMA uniform BVH clip for the alignment check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture", help="write SOMA-X reference joints/vertices for a few frames")
    ap.add_argument("--lod", default="low", choices=["low", "mid"])
    ap.add_argument("--max-influences", type=int, default=8)
    ap.add_argument("--frames", default="0,100,400")
    args = ap.parse_args()

    from soma.body.soma import SOMALayer

    layer = SOMALayer(data_root=args.soma_x, device="cpu", identity_model_type="mhr", lod=args.lod,
                      enable_procedural_transforms=False)
    shape = np.load(args.shape)
    identity = torch.from_numpy(shape["identity_params"]).float()
    scale = torch.from_numpy(shape["scale_params"]).float()

    names, parents, rot, trans, fps = parse_bvh(args.bvh)
    if list(layer.public_joint_names) != names:
        sys.exit(f"joint order mismatch:\nSOMA {list(layer.public_joint_names)}\nBVH  {names}")

    frames = [int(f) for f in args.frames.split(",")]
    # Root (index 0) is identity at the origin in SEED clips; SOMA's pose input starts at Hips.
    assert np.allclose(rot[:, 0], np.eye(3)) and np.allclose(trans[:, 0], 0), "non-identity Root not supported"
    trans = trans * CM_TO_M  # work in metres from here on
    poses = torch.from_numpy(rot[frames, 1:]).float()
    transl = torch.from_numpy(trans[frames, 1]).float()

    with torch.no_grad():
        out = layer(poses, identity.expand(len(frames), -1), scale.expand(len(frames), -1), transl=transl,
                    pose2rot=False, apply_correctives=False, absolute_pose=True)
        rest = layer._cached_rest_shape[0].double().numpy()
        bind = layer._cached_bind_transforms_world[0].double().numpy()
    soma_joints = out.joints.double().numpy()        # (B, 77, 3) m, Y-up
    soma_vertices = out.vertices.double().numpy()    # (B, V, 3)
    soma_transforms = out.transforms.double().numpy()  # (B, 78, 4, 4)

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

    # 2) Skinning weights actually used by SOMA-X for this LOD (dense V x J).
    weights = layer.batched_skinning.skinning_weights if hasattr(layer.batched_skinning, "skinning_weights") else None
    if weights is None:
        weights = layer.public_skinning_weights
    weights = weights.to_dense() if weights.is_sparse or weights.layout != torch.strided else weights
    weights = weights.double().numpy()
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

    faces = layer.faces.numpy() if hasattr(layer.faces, "numpy") else np.asarray(layer.faces)

    # T-pose of this identity (relative pose = identity), the reference for robot rotation offsets.
    with torch.no_grad():
        tpose = layer(torch.zeros(1, len(names) - 1, 3), identity, scale, pose2rot=True, apply_correctives=False)
    tpose_t = tpose.transforms[0].double().numpy()
    tpose_t[:, :3, 3] -= np.array([0.0, tpose.joints[0].double().numpy()[:, 1].min(), 0.0])  # lowest joint on y = 0

    # Inner geometry, from SOMA's semantic segments (mid-LOD vertex ids; low LOD maps through lod_mid_to_low).
    neutral = np.load(os.path.join(args.soma_x, "SOMA_neutral.npz"))
    inner_mid = np.concatenate([neutral["segment_mouth_bag"], neutral["segment_eye_bags"]])
    if rest.shape[0] == neutral["mean"].shape[0]:
        excluded = np.isin(np.arange(rest.shape[0]), inner_mid)
    else:
        excluded = np.isin(neutral["lod_mid_to_low"], inner_mid)
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
