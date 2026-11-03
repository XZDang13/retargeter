#!/usr/bin/env python3
"""Export the fixed (identity-independent) part of SOMA-X's low-LOD MHR body model for the torch-free bake.

tools/bake_soma_surface.py builds the SOMA body surface of the BONES-SEED identity with numpy alone from the arrays
written here: the MHR identity model (LOD6 blend shapes, skeleton, skinning), the MHR -> SOMA topology transfer
(barycentric correspondence and the Laplacian fill of the inner face geometry), SOMA's skeleton transfer (joint
regressor, rotation-fit structure) and the low-LOD SOMA rig (skinning weights, joint orient, triangles). SOMA-X and
MHR are Apache-2.0 (see assets/body_models/soma/SOMA-X_NOTICE.md); nothing here comes from BONES-SEED.

Maintainers only, when moving to another SOMA-X version: needs SOMA-X itself and torch (tools/requirements-soma.txt),
e.g. in a scratch venv with SOMA-X installed from its pinned source:
  python tools/export_soma_x_model.py --soma-x <SOMA-X>/assets --commit <sha> --out assets/body_models/soma/soma_x_model.npz
With --check-shape (a BONES-SEED soma_base_fit_mhr_params.npz) it then compares the numpy bake against SOMA-X.
"""

import argparse
import os
import sys

import numpy as np
import torch


def export(soma_x, commit):
    from soma.body.soma import SOMALayer

    layer = SOMALayer(data_root=soma_x, device="cpu", identity_model_type="mhr", lod="low",
                      enable_procedural_transforms=False)
    im = layer.identity_model
    mhr = (im.identity_model.module if hasattr(im.identity_model, "module") else im.identity_model).character_torch
    interp, lap, skel, skin = im._to_soma_interp, im._laplacian_mesh, layer.skeleton_transfer, layer.batched_skinning
    # The bake reproduces exactly this configuration; fail loudly if a SOMA-X update changes it.
    assert interp.tet_normal_scale == "area" and lap.constraint_mode == "hard" and lap.order == 1
    assert skel.use_warp_for_rotations and skel.rotation_method == "auto" and skel.skip_endjoints
    assert not skel.freeze_rotations and not skel.skip_inverse_lbs and skel._first_joint == 1
    assert torch.equal(skel.skinning_weights, layer.skinning_weights) and skin.global_translation_joint_idx == 1
    assert torch.equal(layer.public_transform_joint_indices, torch.arange(len(layer.public_joint_names)))

    f32 = lambda t: t.detach().cpu().double().numpy().astype(np.float32)
    i32 = lambda t: t.detach().cpu().numpy().astype(np.int32)
    n_id = im.num_identity_coeffs
    lfg = lap.L_FG.to_sparse_coo().coalesce()
    rbf = skel.sparse_rbf_matrix.to_sparse_coo().coalesce()
    chol = lap._chol_factor.double()
    return dict(
        soma_x_commit=np.array(commit),
        joint_names=np.array(layer.public_joint_names),
        # MHR identity model (centimetres, Y-up): rest = base + shape_vectors . identity, posed by the scale
        # parameters (model parameters 136:204) through the parameter transform, FK and LBS.
        mhr_shape_vectors=f32(mhr.blend_shape.shape_vectors[:n_id]),
        mhr_base_shape=f32(mhr.blend_shape.base_shape),
        mhr_scale_transform=f32(mhr.parameter_transform.parameter_transform[:, 136:204]),
        mhr_pmi=i32(mhr.skeleton.pmi),
        mhr_pmi_sizes=np.array([65, 56, 62, 83], dtype=np.int32),
        mhr_prerotations=f32(mhr.skeleton.joint_prerotations),
        mhr_translation_offsets=f32(mhr.skeleton.joint_translation_offsets),
        mhr_inverse_bind=f32(mhr.linear_blend_skinning.inverse_bind_pose),
        mhr_skin_joint=i32(mhr.linear_blend_skinning.skin_indices_flattened),
        mhr_skin_weight=f32(mhr.linear_blend_skinning.skin_weights_flattened),
        mhr_skin_vertex=i32(mhr.linear_blend_skinning.vert_indices_flattened),
        # MHR -> SOMA (low LOD): barycentric coordinates in tetrahedra fabricated on the MHR faces.
        transfer_faces=i32(interp.F_src_torch),
        transfer_tets=i32(interp.F_src_tet),
        transfer_face=i32(interp.face_ids),
        transfer_bary=f32(interp.bary_coords),
        unit_to_m=np.float64(im._unit_conversion),
        # Laplacian fill of the vertices MHR has no geometry for: L_FF x = btilde - L_FG x_G.
        lap_free=i32(lap.vid_unknown),
        lap_anchor=i32(lap.vid_constrained),
        lap_ff=(lap._hard_chol_sign * chol @ chol.T).numpy(),
        lap_fg_index=i32(lfg.indices()),
        lap_fg_value=lfg.values().double().numpy(),
        lap_btilde=lap.btilde.double().numpy(),
        # Skeleton transfer: joint positions = regressor . rest; rotations aligned as in SkeletonTransfer.
        rbf_index=i32(rbf.indices()),
        rbf_value=rbf.values().double().numpy(),
        parents=np.array(skel.joint_parent_ids, dtype=np.int32),
        bind_world=f32(skel.bind_world_transforms),
        bind_shape=f32(skel.bind_shape),
        skin_weights=f32(skel.skinning_weights),
        t_pose_orient=f32(layer._t_pose_orient),
        triangles=i32(layer.faces),
        vertex_excluded=np.isin(np.arange(skel.bind_shape.shape[0]), i32(layer.excluded_vert_ids)).astype(np.uint8),
    ), layer


def check(layer, model, shape_file):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bake_soma_surface as bake

    shape = np.load(shape_file)
    identity = torch.from_numpy(shape["identity_params"]).float()
    scale = torch.from_numpy(shape["scale_params"]).float()
    with torch.no_grad():
        layer.prepare_identity(identity, scale, repose_to_bind_pose=False)
    ref_rest = layer._cached_rest_shape[0].detach().double().numpy()
    ref_bind = layer._cached_bind_transforms_world[0].detach().double().numpy()
    rest, bind = bake.identity_rig(model, shape["identity_params"], shape["scale_params"])
    rng = np.random.default_rng(0)
    rot = np.stack([bake.rodrigues(rng.normal(scale=0.4, size=(len(model["parents"]) - 1, 3))) for _ in range(3)])
    transl = rng.normal(scale=0.2, size=(3, 3))
    with torch.no_grad():
        out = layer.pose(torch.from_numpy(rot).float(), transl=torch.from_numpy(transl).float(), pose2rot=False,
                         apply_correctives=False, absolute_pose=True)
    world = bake.forward_kinematics(model, bind, rot, transl, absolute=True)
    with torch.no_grad():   # relative pose (the T-pose: identity rotations relative to SOMA's joint orient)
        tpose = layer.pose(torch.zeros(1, len(model["parents"]) - 1, 3), pose2rot=True, apply_correctives=False)
    ref_tpose = tpose.transforms.double().numpy()
    our_tpose = bake.forward_kinematics(model, bind, np.tile(np.eye(3), (1, len(model["parents"]) - 1, 1, 1)),
                                        np.zeros((1, 3)), absolute=False)
    verts = bake.skin(model, rest, bind, world)
    print(f"rest {np.abs(rest - ref_rest).max() * 1000:.3g} mm, bind rotations {np.abs(bind[:, :3, :3] - ref_bind[:, :3, :3]).max():.3g}, "
          f"bind positions {np.abs(bind[:, :3, 3] - ref_bind[:, :3, 3]).max() * 1000:.3g} mm, "
          f"posed joints {np.abs(world - out.transforms.double().numpy()).max():.3g}, "
          f"posed vertices {np.abs(verts - out.vertices.double().numpy()).max() * 1000:.3g} mm, "
          f"T-pose {np.abs(our_tpose - ref_tpose).max():.3g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soma-x", required=True, help="SOMA-X assets directory")
    ap.add_argument("--commit", required=True, help="the SOMA-X commit the assets come from (recorded)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-shape", help="BONES-SEED soma_base_fit_mhr_params.npz: compare the numpy bake with SOMA-X")
    args = ap.parse_args()
    data, layer = export(args.soma_x, args.commit)
    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
    if args.check_shape:
        check(layer, dict(np.load(args.out)), args.check_shape)


if __name__ == "__main__":
    main()
