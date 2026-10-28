#!/usr/bin/env python3
"""Export a robot's visual meshes (payloads/geometries.usd, binary USD) to one GLB for the viewer, and a
voxel-downsampled point set per mesh for the validator (ground penetration, self-collision capsules).

Each /Geometries/<name> becomes a glTF node named <name> with its mesh in the link-local frame; the
viewer places it with the visual origins from RobotModel (base.usda). Run with the tooling venv:
  build/tools/venv/bin/python -I tools/export_robot_meshes.py assets/robots/unitree_g1/g1_29_dof build/viewer/robots/g1_29dof.glb \
      [--points assets/robots/unitree_g1/g1_29_dof/mesh_points.npz]

mesh_points.npz: names (M,) str, offsets (M+1,) i32, points (P, 3) f32 (mesh-local frame, metres).
"""

import json
import os
import struct
import sys

import numpy as np
from pxr import Usd, UsdGeom


def triangulate(counts, indices):
    tris, i = [], 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[i], indices[i + k], indices[i + k + 1]))
        i += c
    return np.array(tris, dtype=np.uint32)


def voxel_downsample(points, voxel=0.015):
    keys = np.floor(points / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def main():
    args = sys.argv[1:]
    points_out = None
    if "--points" in args:
        i = args.index("--points")
        points_out = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    robot_dir, out = args
    sampled = {}
    stage = Usd.Stage.Open(os.path.join(robot_dir, "payloads", "geometries.usd"))
    root = stage.GetPrimAtPath("/Geometries")
    xforms = UsdGeom.XformCache()

    binary = bytearray()
    gltf = {"asset": {"version": "2.0", "generator": "retargeter export_robot_meshes"},
            "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
            "accessors": [], "bufferViews": [], "buffers": []}

    def add_view(data, target):
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(data)
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        return len(gltf["bufferViews"]) - 1

    for geometry in root.GetChildren():
        primitives = []
        root_inverse = xforms.GetLocalToWorldTransform(geometry).GetInverse()
        for prim in Usd.PrimRange(geometry):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
            local = xforms.GetLocalToWorldTransform(prim) * root_inverse
            m = np.array(local, dtype=np.float64)  # row-vector convention: p' = p @ M
            points = (np.c_[points, np.ones(len(points))] @ m)[:, :3].astype(np.float32)
            tris = triangulate(mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get())
            color = mesh.GetDisplayColorAttr().Get()
            rgb = [float(c) for c in color[0]] if color else [0.7, 0.7, 0.7]

            pos_view = add_view(points.tobytes(), 34962)
            idx_view = add_view(tris.astype(np.uint32).tobytes(), 34963)
            gltf["accessors"].append({"bufferView": pos_view, "componentType": 5126, "count": len(points),
                                      "type": "VEC3", "min": points.min(0).tolist(), "max": points.max(0).tolist()})
            gltf["accessors"].append({"bufferView": idx_view, "componentType": 5125, "count": int(tris.size),
                                      "type": "SCALAR"})
            gltf["materials"].append({"pbrMetallicRoughness": {"baseColorFactor": rgb + [1.0], "metallicFactor": 0.1,
                                                               "roughnessFactor": 0.6}})
            primitives.append({"attributes": {"POSITION": len(gltf["accessors"]) - 2},
                               "indices": len(gltf["accessors"]) - 1, "material": len(gltf["materials"]) - 1})
            sampled.setdefault(geometry.GetName(), []).append(points)
        if not primitives:
            continue
        gltf["meshes"].append({"name": geometry.GetName(), "primitives": primitives})
        gltf["nodes"].append({"name": geometry.GetName(), "mesh": len(gltf["meshes"]) - 1})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)

    while len(binary) % 4:
        binary.append(0)
    gltf["buffers"].append({"byteLength": len(binary)})
    header = json.dumps(gltf, separators=(",", ":")).encode()
    header += b" " * ((4 - len(header) % 4) % 4)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(header) + 8 + len(binary)))
        f.write(struct.pack("<II", len(header), 0x4E4F534A))
        f.write(header)
        f.write(struct.pack("<II", len(binary), 0x004E4942))
        f.write(binary)
    print(f"wrote {out}: {len(gltf['nodes'])} meshes, {len(binary) / 1e6:.1f} MB")

    if points_out:
        names, chunks = [], []
        for name, parts in sampled.items():
            names.append(name)
            chunks.append(voxel_downsample(np.concatenate(parts)).astype(np.float32))
        offsets = np.concatenate([[0], np.cumsum([len(c) for c in chunks])]).astype(np.int32)
        np.savez(points_out, names=np.array(names), offsets=offsets, points=np.concatenate(chunks))
        print(f"wrote {points_out}: {offsets[-1]} points over {len(names)} meshes")


if __name__ == "__main__":
    main()
