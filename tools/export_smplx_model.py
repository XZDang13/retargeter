#!/usr/bin/env python3
"""Export the parts of the SMPL-X body model the retargeter needs, as plain uncompressed arrays.

The official SMPLX_{NEUTRAL,MALE,FEMALE}.npz files are zip-compressed and hold pickled objects, which the Swift npz
reader does not read; this writes, per gender, assets/body_models/smplx/smplx_lbs_<gender>.npz with only the linear
blend skinning model (no pose correctives, no expressions):

  v_template  (V, 3) f4     rest vertices, SMPL-X frame (Y-up, facing +Z, +X the body's left)
  shapedirs   (V, 3, B) f4  first B shape components
  j_template  (J, 3) f4     rest joints = J_regressor @ v_template
  j_shapedirs (J, 3, B) f4  J_regressor @ shapedirs
  parents     (J,) i4       kinematic tree (-1 for the pelvis)
  weight_joints / weight_values (V, 4)  the 4 largest skinning weights per vertex, renormalised
  triangles   (F, 3) i4
  joint_names (J,) U        smplx library joint names (pelvis, left_hip, ...)

numpy only (no torch, no smplx package). The SMPL-X licence forbids redistribution, so the outputs are gitignored
like the models themselves:
  python tools/export_smplx_model.py [--models assets/body_models/smplx] [--betas 16]
"""

import argparse
import os
import sys

import numpy as np

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle",
    "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye", "right_eye",
] + [f"{side}_{finger}{k}" for side in ("left", "right")
     for finger in ("index", "middle", "pinky", "ring", "thumb") for k in (1, 2, 3)]


def export(source, out, betas):
    m = np.load(source, allow_pickle=False)   # the pickled entries (part2num, joint2num) are never touched
    v = m["v_template"].astype(np.float64)
    shapedirs = m["shapedirs"][:, :, :betas].astype(np.float64)
    regressor = m["J_regressor"].astype(np.float64)
    parents = m["kintree_table"][0].astype(np.int64)
    parents[0] = -1
    weights = m["weights"].astype(np.float64)
    assert regressor.shape[0] == len(JOINT_NAMES) == len(parents), regressor.shape
    assert all(parents[j] < j for j in range(1, len(parents))), "joints not in topological order"

    top = np.argsort(-weights, axis=1)[:, :4]
    top_w = np.take_along_axis(weights, top, axis=1)
    top_w /= top_w.sum(axis=1, keepdims=True)
    np.savez(out,
             v_template=v.astype(np.float32),
             shapedirs=shapedirs.astype(np.float32),
             j_template=(regressor @ v).astype(np.float32),
             j_shapedirs=np.einsum("jv,vcb->jcb", regressor, shapedirs).astype(np.float32),
             parents=parents.astype(np.int32),
             weight_joints=top.astype(np.int32),
             weight_values=top_w.astype(np.float32),
             triangles=m["f"].astype(np.int32),
             joint_names=np.array(JOINT_NAMES))
    print(f"{out}: {v.shape[0]} vertices, {len(parents)} joints, {betas} shape components "
          f"(max weight beyond the 4 kept: {np.sort(weights, axis=1)[:, -5].max():.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=os.path.join(os.path.dirname(__file__), "..", "assets", "body_models", "smplx"))
    parser.add_argument("--betas", type=int, default=16, help="shape components to keep (default 16, as AMASS)")
    args = parser.parse_args()
    found = 0
    for gender in ("neutral", "male", "female"):
        source = os.path.join(args.models, f"SMPLX_{gender.upper()}.npz")
        if not os.path.exists(source):
            continue
        export(source, os.path.join(args.models, f"smplx_lbs_{gender}.npz"), args.betas)
        found += 1
    if not found:
        sys.exit(f"no SMPLX_*.npz in {args.models} (download them from smpl-x.is.tue.mpg.de)")


if __name__ == "__main__":
    main()
