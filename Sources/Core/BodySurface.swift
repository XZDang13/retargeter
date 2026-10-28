// Skinned human body surface (linear blend skinning against the human skeleton).

public struct BodySurface: Sendable {
    public var jointNames: [String]
    /// Bind-pose vertices, world frame.
    public var restVertices: [Vec3]
    /// Bind-pose world transform of each joint.
    public var bind: [Transform]
    /// Per vertex: up to `influences` (joint, weight) pairs, weights summing to 1.
    public var weightJoints: [Int32]
    public var weightValues: [Float]
    public var influences: Int
    public var triangles: [SIMD3<Int32>]
    /// Inner geometry (mouth / eye bags) that can never touch anything.
    public var excluded: [Bool]
    /// World joint transforms of the reference (T-) pose, facing +X with the feet on z = 0.
    public var referencePose: [Transform]?

    public init(jointNames: [String], restVertices: [Vec3], bind: [Transform], weightJoints: [Int32],
                weightValues: [Float], influences: Int, triangles: [SIMD3<Int32>], excluded: [Bool]? = nil,
                referencePose: [Transform]? = nil) {
        self.jointNames = jointNames
        self.restVertices = restVertices
        self.bind = bind
        self.weightJoints = weightJoints
        self.weightValues = weightValues
        self.influences = influences
        self.triangles = triangles
        self.excluded = excluded ?? Array(repeating: false, count: restVertices.count)
        self.referencePose = referencePose
    }

    public var vertexCount: Int { restVertices.count }

    /// Posed vertices for one frame of world joint transforms (reference implementation).
    public func skin(world: [Transform]) -> [Vec3] {
        let skinning = zip(world, bind).map { $0.0 * $0.1.inverse }
        return restVertices.indices.map { v in
            var p = Vec3.zero
            for k in 0..<influences {
                let i = v * influences + k
                let w = Double(weightValues[i])
                if w == 0 { continue }
                p += w * skinning[Int(weightJoints[i])].apply(restVertices[v])
            }
            return p
        }
    }

    /// Posed vertices of every frame, flat `[frame * vertexCount]`, in single precision: the batch version of
    /// `skin(world:)`, frames in parallel.
    public func skin(frames world: [[Transform]]) -> [SIMD3<Float>] {
        let n = vertexCount, frames = world.count, influences = self.influences
        let inverseBind = bind.map(\.inverse)
        let rest = restVertices.map { SIMD3<Float>(Float($0.x), Float($0.y), Float($0.z)) }
        let joints = weightJoints, weights = weightValues
        var out = [SIMD3<Float>](repeating: .zero, count: frames * n)
        out.withUnsafeMutableBufferPointer { buffer in
            let target = SendablePointer(base: buffer.baseAddress!)
            let chunk = 32
            _ = parallelMap((frames + chunk - 1) / chunk) { c in
                for f in (c * chunk)..<min(frames, (c + 1) * chunk) {
                    // Skinning transforms as float 3x4 (rotation columns, translation).
                    let skinning = zip(world[f], inverseBind).map { pair -> (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>) in
                        let t = pair.0 * pair.1
                        func f3(_ v: Vec3) -> SIMD3<Float> { SIMD3(Float(v.x), Float(v.y), Float(v.z)) }
                        return (f3(t.rotation.rotate(Vec3(1, 0, 0))), f3(t.rotation.rotate(Vec3(0, 1, 0))),
                                f3(t.rotation.rotate(Vec3(0, 0, 1))), f3(t.translation))
                    }
                    let row = target.base + f * n
                    for v in 0..<n {
                        let r = rest[v]
                        var p = SIMD3<Float>.zero
                        for k in 0..<influences {
                            let w = weights[v * influences + k]
                            if w == 0 { continue }
                            let m = skinning[Int(joints[v * influences + k])]
                            p += w * (m.0 * r.x + m.1 * r.y + m.2 * r.z + m.3)
                        }
                        row[v] = p
                    }
                }
            }
        }
        return out
    }
}

