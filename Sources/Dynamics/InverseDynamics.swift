// Floating-base inverse dynamics (recursive Newton-Euler) in the world frame, double precision. This is the
// reference path (validator, tests); the solver kernels use a float32 port (Kernels/dynamics_common.slang).
//
// All vectors are world-frame. Positions inside the recursion are taken relative to the root origin, which
// keeps float32 ports accurate far from the world origin. Conventions:
//   - root velocity / acceleration: of the root link origin (linear) and of the root body (angular);
//   - the "base wrench" is the external wrench (force, moment about the root origin) that would have to act
//     on the root for the motion to happen: zero when gravity, inertia and the given external forces balance.
//     For a free-floating robot it is the part of the motion the contacts do not explain.

import Core

#if canImport(Glibc)
import Glibc
#endif

public let standardGravity = Vec3(0, 0, -9.81)

/// Kinematic state of the floating base and joints at one instant.
public struct RobotState: Sendable {
    public var root: Transform
    public var q: [Double]
    public var rootVelocity: Vec3         // world, of the root origin
    public var rootAngularVelocity: Vec3  // world
    public var qd: [Double]
    public var rootAcceleration: Vec3
    public var rootAngularAcceleration: Vec3
    public var qdd: [Double]

    public init(root: Transform, q: [Double], rootVelocity: Vec3 = .zero, rootAngularVelocity: Vec3 = .zero,
                qd: [Double]? = nil, rootAcceleration: Vec3 = .zero, rootAngularAcceleration: Vec3 = .zero,
                qdd: [Double]? = nil) {
        self.root = root
        self.q = q
        self.rootVelocity = rootVelocity
        self.rootAngularVelocity = rootAngularVelocity
        self.qd = qd ?? Array(repeating: 0, count: q.count)
        self.rootAcceleration = rootAcceleration
        self.rootAngularAcceleration = rootAngularAcceleration
        self.qdd = qdd ?? Array(repeating: 0, count: q.count)
    }
}

/// A force acting on the robot at a world point of a link.
public struct PointForce: Sendable {
    public var link: Int
    public var point: Vec3
    public var force: Vec3

    public init(link: Int, point: Vec3, force: Vec3) {
        self.link = link
        self.point = point
        self.force = force
    }
}

public struct InverseDynamicsResult: Sendable {
    public var baseForce: Vec3
    public var baseMoment: Vec3   // about the root origin
    public var tau: [Double]      // per dof
}

/// Per-link kinematics of one instant (world frame).
public struct LinkMotion: Sendable {
    public var pose: [Transform]
    public var com: [Vec3]
    public var omega: [Vec3]
    public var alpha: [Vec3]
    public var comVelocity: [Vec3]
    public var comAcceleration: [Vec3]
    /// World joint axis and joint origin of actuated links (zero for others).
    public var axis: [Vec3]
    public var jointOrigin: [Vec3]
}

/// Mass properties and the precomputed tree for dynamics.
public struct RobotDynamics: Sendable {
    public let robot: RobotModel
    public let mass: [Double]
    public let comLocal: [Vec3]
    /// Rotational inertia about the link CoM, link frame, row-major 3x3.
    public let inertiaLocal: [[Double]]
    public let totalMass: Double
    public let effort: [Double]   // per dof (N·m), .infinity when unknown
    public let dofLink: [Int]     // per dof: link index

    public init(robot: RobotModel) {
        self.robot = robot
        mass = robot.links.map { $0.inertial?.mass ?? 0 }
        comLocal = robot.links.map { $0.inertial?.centerOfMass ?? .zero }
        inertiaLocal = robot.links.map { link in
            guard let inertial = link.inertial else { return Array(repeating: 0, count: 9) }
            // I = R diag R^T with R the principal axes (principal frame -> link frame).
            let r = inertial.principalAxes.matrix, d = inertial.diagonalInertia
            var out = [Double](repeating: 0, count: 9)
            for i in 0..<3 {
                for j in 0..<3 {
                    out[3 * i + j] = (0..<3).reduce(0) { $0 + r[3 * i + $1] * d[$1] * r[3 * j + $1] }
                }
            }
            return out
        }
        totalMass = mass.reduce(0, +)
        var effort = [Double](repeating: .infinity, count: robot.dofCount)
        var dofLink = [Int](repeating: 0, count: robot.dofCount)
        for (l, link) in robot.links.enumerated() {
            guard let d = link.dofIndex else { continue }
            dofLink[d] = l
            if let e = link.limits?.effort, e > 0 { effort[d] = e }
        }
        self.effort = effort
        self.dofLink = dofLink
    }

    /// Forward pass: link poses, velocities and accelerations.
    public func motion(_ s: RobotState) -> LinkMotion {
        let links = robot.links, L = links.count
        var pose = [Transform](repeating: .identity, count: L)
        var omega = [Vec3](repeating: .zero, count: L), alpha = omega
        var v = omega, a = omega   // link origin velocity / acceleration
        var axis = omega, origin = omega
        for (i, link) in links.enumerated() {
            guard let p = link.parent else {
                pose[i] = s.root
                omega[i] = s.rootAngularVelocity
                alpha[i] = s.rootAngularAcceleration
                v[i] = s.rootVelocity
                a[i] = s.rootAcceleration
                continue
            }
            let jointFrame = pose[p] * link.jointOrigin
            let o = jointFrame.translation
            var w = omega[p], dw = alpha[p]
            var vo = v[p] + cross(omega[p], o - pose[p].translation)
            var ao = a[p] + cross(alpha[p], o - pose[p].translation) + cross(omega[p], cross(omega[p], o - pose[p].translation))
            var motion = Transform.identity
            if case .revolute(let localAxis) = link.joint, let d = link.dofIndex {
                let worldAxis = jointFrame.rotation.rotate(localAxis)
                axis[i] = worldAxis
                origin[i] = o
                motion.rotation = Quat(axis: localAxis, angle: s.q[d])
                dw = dw + worldAxis * s.qdd[d] + cross(w, worldAxis * s.qd[d])
                w = w + worldAxis * s.qd[d]
            }
            pose[i] = jointFrame * motion * link.jointToChild
            let r = pose[i].translation - o
            vo = vo + cross(w, r)
            ao = ao + cross(dw, r) + cross(w, cross(w, r))
            omega[i] = w
            alpha[i] = dw
            v[i] = vo
            a[i] = ao
        }
        var com = omega, vc = omega, ac = omega
        for i in 0..<L {
            com[i] = pose[i].apply(comLocal[i])
            let r = com[i] - pose[i].translation
            vc[i] = v[i] + cross(omega[i], r)
            ac[i] = a[i] + cross(alpha[i], r) + cross(omega[i], cross(omega[i], r))
        }
        return LinkMotion(pose: pose, com: com, omega: omega, alpha: alpha, comVelocity: vc, comAcceleration: ac,
                          axis: axis, jointOrigin: origin)
    }

    /// World-frame inertia of link `i` at pose `t` applied to `x`.
    func inertiaTimes(_ i: Int, _ t: Transform, _ x: Vec3) -> Vec3 {
        let local = t.rotation.conjugate.rotate(x)
        let m = inertiaLocal[i]
        let y = Vec3(m[0] * local.x + m[1] * local.y + m[2] * local.z,
                     m[3] * local.x + m[4] * local.y + m[5] * local.z,
                     m[6] * local.x + m[7] * local.y + m[8] * local.z)
        return t.rotation.rotate(y)
    }

    /// Recursive Newton-Euler: joint torques and the residual base wrench for the given external forces.
    public func inverseDynamics(_ s: RobotState, external: [PointForce] = [], gravity: Vec3 = standardGravity,
                                motion precomputed: LinkMotion? = nil) -> InverseDynamicsResult {
        let m = precomputed ?? motion(s)
        let links = robot.links, L = links.count
        let root = s.root.translation
        // Wrench each link needs from its parent: force f and moment n about the root origin (accumulated
        // over the subtree), so moments transfer between links without lever-arm bookkeeping.
        var f = [Vec3](repeating: .zero, count: L), n = f
        for i in 0..<L {
            let F = mass[i] * (m.comAcceleration[i] - gravity)
            let N = inertiaTimes(i, m.pose[i], m.alpha[i]) + cross(m.omega[i], inertiaTimes(i, m.pose[i], m.omega[i]))
            f[i] = F
            n[i] = N + cross(m.com[i] - root, F)
        }
        for e in external {
            f[e.link] -= e.force
            n[e.link] -= cross(e.point - root, e.force)
        }
        var tau = [Double](repeating: 0, count: robot.dofCount)
        for i in stride(from: L - 1, through: 1, by: -1) {
            if let d = links[i].dofIndex {
                // Moment about the joint origin = n − (o − root) × f.
                tau[d] = dot(m.axis[i], n[i] - cross(m.jointOrigin[i] - root, f[i]))
            }
            if let p = links[i].parent {
                f[p] += f[i]
                n[p] += n[i]
            }
        }
        return InverseDynamicsResult(baseForce: f[0], baseMoment: n[0], tau: tau)
    }

    /// Joint torques produced by unit-free point forces: column of d tau / d force for a force at `point` on
    /// `link` (tau = −Σ_j axis_j · ((point − o_j) × force) over the actuated ancestors j).
    public func torqueFromForce(link: Int, point: Vec3, force: Vec3, motion m: LinkMotion, into tau: inout [Double]) {
        var i: Int? = link
        while let k = i {
            if let d = robot.links[k].dofIndex {
                tau[d] -= dot(m.axis[k], cross(point - m.jointOrigin[k], force))
            }
            i = robot.links[k].parent
        }
    }

    /// Per dof: the acceleration the joint reaches at its effort limit, effort / (inertia of everything beyond
    /// the joint about its axis + rotor armature), in `state`'s pose (default: the default pose, legs straight:
    /// the largest leg inertias) without gravity load; infinity without an effort limit.
    public func effortLimitedAcceleration(_ state: RobotState? = nil) -> [Double] {
        let m = motion(state ?? RobotState(root: robot.defaultRootPose, q: Array(repeating: 0, count: robot.dofCount)))
        return (0..<robot.dofCount).map { j in
            let lj = dofLink[j], axis = m.axis[lj], o = m.jointOrigin[lj]
            var inertia = robot.links[lj].limits?.armature ?? 0
            for i in robot.links.indices {
                var k: Int? = i
                while let x = k, x != lj { k = robot.links[x].parent }
                guard k == lj else { continue }
                let r = cross(axis, m.com[i] - o)
                inertia += mass[i] * dot(r, r) + dot(axis, inertiaTimes(i, m.pose[i], axis))
            }
            return inertia > 0 ? effort[j] / inertia : .infinity
        }
    }

    public func centerOfMass(_ m: LinkMotion) -> Vec3 {
        var c = Vec3.zero
        for i in mass.indices { c += mass[i] * m.com[i] }
        return c / totalMass
    }

    /// Kinetic energy (translational + rotational) of all links.
    public func kineticEnergy(_ m: LinkMotion) -> Double {
        var e = 0.0
        for i in mass.indices {
            e += 0.5 * mass[i] * dot(m.comVelocity[i], m.comVelocity[i])
            e += 0.5 * dot(m.omega[i], inertiaTimes(i, m.pose[i], m.omega[i]))
        }
        return e
    }

    /// Linear momentum and angular momentum about `point`.
    public func momentum(_ m: LinkMotion, about point: Vec3) -> (linear: Vec3, angular: Vec3) {
        var p = Vec3.zero, l = Vec3.zero
        for i in mass.indices {
            let mv = mass[i] * m.comVelocity[i]
            p += mv
            l += cross(m.com[i] - point, mv) + inertiaTimes(i, m.pose[i], m.omega[i])
        }
        return (p, l)
    }
}

extension Quat {
    /// Rotation vector (axis · angle) of the unit quaternion, angle in [0, π].
    public var rotationVector: Vec3 {
        let q = w < 0 ? Quat(w: -w, v: -v) : self
        let s = length(q.v)
        if s < 1e-12 { return 2 * q.v }
        return q.v * (2 * atan2(s, q.w) / s)
    }
}
