// Floating-base robot kinematic tree. Links are stored in topological order (parent index < child index).

public enum JointKind: Sendable, Equatable {
    case fixed
    case revolute(axis: Vec3)  // unit axis in the joint frame
}

public struct JointLimits: Sendable, Equatable {
    public var lower: Double     // rad
    public var upper: Double     // rad
    public var effort: Double?   // N·m
    public var velocity: Double? // rad/s (not in the USD assets; supplied by robot config)
    public var armature: Double?
}

public struct Inertial: Sendable, Equatable {
    public var mass: Double
    public var centerOfMass: Vec3        // link frame
    public var diagonalInertia: Vec3     // principal moments
    public var principalAxes: Quat       // link frame → principal frame
}

public enum CollisionShape: Sendable, Equatable {
    case sphere(radius: Double)
    case cylinder(radius: Double, height: Double, axis: Int)  // axis: 0 = X, 1 = Y, 2 = Z
    case capsule(radius: Double, height: Double, axis: Int)
}

public struct Collision: Sendable, Equatable {
    public var name: String
    public var shape: CollisionShape
    public var origin: Transform  // link frame
}

/// A visual mesh instance: `mesh` names an entry of the robot's exported glTF mesh set.
public struct Visual: Sendable, Equatable {
    public var name: String
    public var mesh: String
    public var origin: Transform
}

public struct Link: Sendable {
    public var name: String
    public var parent: Int?          // index into links
    /// parent link frame → joint frame at q = 0 (for a fixed link: parent → child).
    public var jointOrigin: Transform
    /// joint frame → child link frame.
    public var jointToChild: Transform
    public var joint: JointKind
    public var jointName: String?
    public var dofIndex: Int?        // index into q for actuated joints
    public var limits: JointLimits?
    public var inertial: Inertial?
    public var collisions: [Collision] = []
    public var visuals: [Visual] = []
}

public struct RobotModel: Sendable {
    public var name: String
    public var links: [Link]
    /// Actuated joint names in canonical (Unitree SDK / depth-first) order; q follows this order.
    public var dofNames: [String]
    /// Default root pose (pelvis standing on the ground at q = 0).
    public var defaultRootPose: Transform

    public init(name: String, links: [Link], dofNames: [String], defaultRootPose: Transform) {
        self.name = name
        self.links = links
        self.dofNames = dofNames
        self.defaultRootPose = defaultRootPose
    }

    public var dofCount: Int { dofNames.count }

    public func linkIndex(named name: String) -> Int? { links.firstIndex { $0.name == name } }

    /// World poses of all links (reference implementation).
    public func forwardKinematics(root: Transform, q: [Double]) -> [Transform] {
        precondition(q.count == dofCount, "expected \(dofCount) joint values, got \(q.count)")
        var world = [Transform](repeating: .identity, count: links.count)
        for (i, link) in links.enumerated() {
            guard let parent = link.parent else {
                world[i] = root
                continue
            }
            var motion = Transform.identity
            if case .revolute(let axis) = link.joint, let dof = link.dofIndex {
                motion.rotation = Quat(axis: axis, angle: q[dof])
            }
            world[i] = world[parent] * link.jointOrigin * motion * link.jointToChild
        }
        return world
    }

    public var totalMass: Double { links.reduce(0) { $0 + ($1.inertial?.mass ?? 0) } }
}
