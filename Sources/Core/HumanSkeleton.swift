// Human skeleton and motion in our canonical frame: metres, Z-up, +X forward.

public struct HumanSkeleton: Sendable {
    public var name: String
    public var jointNames: [String]
    /// Parent index per joint (topological order; root has nil).
    public var parents: [Int?]
    /// Rest translation of each joint in its parent frame.
    public var restOffsets: [Vec3]

    public init(name: String, jointNames: [String], parents: [Int?], restOffsets: [Vec3]) {
        self.name = name
        self.jointNames = jointNames
        self.parents = parents
        self.restOffsets = restOffsets
    }

    public var jointCount: Int { jointNames.count }

    public func jointIndex(named name: String) -> Int? { jointNames.firstIndex(of: name) }
}

public struct HumanMotion: Sendable {
    public var skeleton: HumanSkeleton
    public var fps: Double
    /// Per frame, per joint local transform (rotation + translation in the parent frame).
    /// Joints without animated translation carry their rest offset.
    public var local: [[Transform]]

    public init(skeleton: HumanSkeleton, fps: Double, local: [[Transform]]) {
        self.skeleton = skeleton
        self.fps = fps
        self.local = local
    }

    public var frameCount: Int { local.count }

    /// World transforms of every joint for one frame (reference implementation).
    public func globalTransforms(frame: Int) -> [Transform] {
        let locals = local[frame]
        var world = [Transform](repeating: .identity, count: skeleton.jointCount)
        for j in 0..<skeleton.jointCount {
            world[j] = skeleton.parents[j].map { world[$0] * locals[j] } ?? locals[j]
        }
        return world
    }
}
