import Core

extension BodySurface {
    /// Loads a surface baked by tools/bake_soma_surface.py.
    public static func load(file: String) throws -> BodySurface {
        let a = try NPZ.read(file: file)
        func array(_ key: String) throws -> NPYArray {
            guard let value = a[key] else { throw NPYError(message: "\(file): missing \(key)") }
            return value
        }
        let rest = try array("rest_vertices").doubles()
        let bindRot = try array("bind_rot").doubles()
        let bindPos = try array("bind_pos").doubles()
        let joints = try array("weight_joints")
        let triangles = try array("triangles").ints()
        let jointCount = bindPos.count / 3
        var reference: [Transform]?
        if let r = try a["tpose_rot"]?.doubles(), let p = try a["tpose_pos"]?.doubles() {
            reference = (0..<jointCount).map { j in
                Transform(rotation: Quat(w: r[4 * j], x: r[4 * j + 1], y: r[4 * j + 2], z: r[4 * j + 3]).normalized,
                          translation: Vec3(p[3 * j], p[3 * j + 1], p[3 * j + 2]))
            }
        }
        return BodySurface(
            jointNames: try array("joint_names").strings(),
            restVertices: stride(from: 0, to: rest.count, by: 3).map { Vec3(rest[$0], rest[$0 + 1], rest[$0 + 2]) },
            bind: (0..<jointCount).map { j in
                Transform(rotation: Quat(w: bindRot[4 * j], x: bindRot[4 * j + 1], y: bindRot[4 * j + 2], z: bindRot[4 * j + 3]).normalized,
                          translation: Vec3(bindPos[3 * j], bindPos[3 * j + 1], bindPos[3 * j + 2]))
            },
            weightJoints: try joints.ints().map { Int32($0) },
            weightValues: try array("weight_values").doubles().map { Float($0) },
            influences: joints.shape.last ?? 1,
            triangles: stride(from: 0, to: triangles.count, by: 3).map {
                SIMD3(Int32(triangles[$0]), Int32(triangles[$0 + 1]), Int32(triangles[$0 + 2]))
            },
            excluded: try a["vertex_excluded"]?.ints().map { $0 != 0 },
            referencePose: reference)
    }
}
