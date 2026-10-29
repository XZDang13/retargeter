// Mapping configuration (Configs/mappings/<skeleton>_<robot>.json) and its resolution against a robot
// model and a human skeleton.

import Core

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public struct RetargetMapping: Sendable {
    public struct Target: Sendable {
        public var human: String
        public var robot: String
        public var positionWeight: Double
        public var rotationWeight: Double
        /// Share of the rotation weight about `twistAxis` (link frame): below 1 for a hand, whose captured twist about
        /// the forearm is the least reliable part of its orientation.
        public var twist = 1.0
        public var twistAxis = Vec3(1, 0, 0)
        /// Human joint of another target (e.g. the shin for a foot): this goal's rotation relative to that goal is
        /// clamped to what the robot joints between the two links can reach within their limits.
        public var reach: String? = nil
    }

    public struct Solver: Sendable {
        public var iterations = 8
        public var damping = 1e-3
        public var smoothness = 0.5
        public var posture = 0.02
        public var limitMargin = 0.98
        public var limitWeight = 10.0
        /// Causal centroidal-dynamics weight (GPU solver only; 0 disables), its time step (s), the ground
        /// friction coefficient and the contact-force solve iterations. Off by default: at lag 0 a correction
        /// of frame t only moves the acceleration error to t + K (measured worse at every weight); it is the
        /// building block for the fixed-lag window (milestone 6).
        public var dynamics = 0.0
        public var dynamicsStride = 0.025
        public var friction = 0.7
        public var contactIterations = 30
        /// Arm branch switch (GPU solver; `Kernels/retarget_ik_group.slang`): every `branchInterval` s an arm
        /// tracking worse than `branchError` (m) is solved again from its joints at 0 (`branchIterations`); if
        /// that halves its tracking cost the output arm eases into the new branch over `branchBlend` s (0: off), and
        /// toward whichever branch has tracked better by `branchHysteresis` (cost ratio) for `branchDwell` s; the
        /// other one is dropped after `branchKeep` s.
        public var branchInterval = 0.05
        public var branchError = 0.05
        public var branchIterations = 20
        public var branchBlend = 0.0
        public var branchHysteresis = 0.5
        public var branchDwell = 0.2
        public var branchKeep = 1.0
    }

    public struct Offline: Sendable {
        public var sweeps = 150
        /// Sweeps between refreshes of the stance anchors and contact points (host work).
        public var anchorInterval = 5
        /// Realtime warm start in segments of this many seconds (each starting `warmOverlap` s earlier, discarded),
        /// solved in parallel; 0: whole clips.
        public var warmSegment = 4.0
        public var warmOverlap = 1.0
        public var iterations = 1
        public var velocity = 1.0
        public var acceleration = 60.0
        public var rootVelocity = 2.0
        public var lock = 60.0
        /// Time (s) over which the stance lock's vertical pull ramps in / out at the ends of each stance run.
        public var lockRamp = 0.05
        /// A sole corner is locked during a stance run only if its mean height over the run is within this (m) of the
        /// foot's lowest corner.
        public var lockHeight = 0.04
        public var ground = 200.0
        public var groundMargin = 0.005
        public var contact = 100.0
        /// Time (s) over which a non-foot link's ground-contact pull ramps in / out at the ends of each contact run.
        public var contactRamp = 0.1
        public var collision = 100.0
        public var collisionMargin = 0.0
        /// Smooth onset of the self-collision push over this distance (m) before the margin.
        public var collisionBand = 0.03
        public var rootAcceleration = 60.0
        /// Weight of the hands' position second differences (per frame, metres).
        public var handAcceleration = 300.0
        /// Weight of the centroidal-dynamics residual (unexplained wrench in units of m·g); 0 disables.
        public var dynamics = 3.0
        /// "contact_forces": contact forces solved per frame (friction cones); "force_free" (experimental): CoM
        /// support without forces — ballistic flight, no ground pull, total force inside the friction cone, ZMP
        /// inside the support polygon. Measured unstable on 5 of 8 comparison clips (docs/baseline-comparison.md).
        public var dynamicsModel = "contact_forces"
        public var friction = 0.7
        public var contactIterations = 5   // warm-started from the previous sweep
        /// Time step (s) of the dynamics accelerations (frames t ± K).
        public var dynamicsStride = 0.025
        /// Trust region per frame step (rad for joints and root rotation, m for the root position).
        public var maxJointStep = 0.05
        public var maxRootStep = 0.01
    }

    public enum ScaleMode: String, Sendable { case perChain = "per_chain", uniform, none }

    public var skeleton: String
    public var robot: String
    public var rootHuman: String
    public var rootRobot: String
    public var scaleMode: ScaleMode
    /// Human joints that anchor a limb (shoulders): the limb's goals are placed relative to the anchor, which sits
    /// at the robot's own width (see RetargetSetup.goals). Empty: everything relative to the root.
    public var scaleAnchors: [Anchor]

    public struct Anchor: Sendable {
        public var joint: String
        /// The limb's horizontal placement (anchor parent's reference X and Y) scaled uniformly from the root, only
        /// the height hanging from the anchor: for legs, so a stance foot stays under the body (front-back and
        /// sideways) as the person's was; uniform scaling preserves that balance geometry.
        public var uniformHorizontal: Bool
    }
    public var tposeOffsets: Bool
    /// Robot joint values of the reference pose matching the human reference pose (nil: solve for it).
    public var referencePose: [String: Double]?
    public var targets: [Target]
    public var solver: Solver
    public var offline: Offline
    /// Fixed-lag window of live streams (FixedLagSolver): the offline costs except as set here.
    public var fixedLag = FixedLag()
    /// Joint ranges the solvers use instead of the robot's (narrower, e.g. human-like), by joint name.
    public var solverLimits: [String: (lower: Double, upper: Double)] = [:]

    /// The solvers' range of every dof: `solverLimits` where given, else the robot's.
    public func solverLimits(_ robot: RobotModel) -> [(lower: Double, upper: Double)] {
        (0..<robot.dofCount).map { d in
            let link = robot.links.first { $0.dofIndex == d }!
            let own = (link.limits!.lower, link.limits!.upper)
            guard let narrow = solverLimits[robot.dofNames[d]] else { return own }
            return (max(own.0, narrow.lower), min(own.1, narrow.upper))
        }
    }

    public struct FixedLag: Sendable {
        /// Weight of the prior pulling each frame to its realtime solve: joints and root rotation (1/rad), root
        /// position (1/m). It keeps errors along directions nothing else holds from accumulating frame to frame.
        public var reference = 1.0
        public var referenceRoot = 3.0
        /// Weight of the centroidal-dynamics residual in the window (offline: `offline.dynamics`).
        public var dynamics = 1.0
    }
    /// Human contact group -> robot link held on the ground while that part touches it (offline).
    public var contacts: [String: String]

    public static func load(file: String) throws -> RetargetMapping {
        let json = try JSONValue.load(file: file)
        func need<T>(_ value: T?, _ what: String) throws -> T {
            guard let value else { throw JSONError(message: "\(file): missing \(what)") }
            return value
        }
        var solver = Solver()
        if let s = json["solver"] {
            solver.iterations = Int(s["iterations"]?.double ?? Double(solver.iterations))
            solver.damping = s["damping"]?.double ?? solver.damping
            solver.smoothness = s["smoothness"]?.double ?? solver.smoothness
            solver.posture = s["posture"]?.double ?? solver.posture
            solver.limitMargin = s["limit_margin"]?.double ?? solver.limitMargin
            solver.limitWeight = s["limit_weight"]?.double ?? solver.limitWeight
            solver.dynamics = s["dynamics"]?.double ?? solver.dynamics
            solver.dynamicsStride = s["dynamics_stride"]?.double ?? solver.dynamicsStride
            solver.friction = s["friction"]?.double ?? solver.friction
            solver.contactIterations = Int(s["contact_iterations"]?.double ?? Double(solver.contactIterations))
            solver.branchInterval = s["branch_interval"]?.double ?? solver.branchInterval
            solver.branchError = s["branch_error"]?.double ?? solver.branchError
            solver.branchIterations = Int(s["branch_iterations"]?.double ?? Double(solver.branchIterations))
            solver.branchBlend = s["branch_blend"]?.double ?? solver.branchBlend
            solver.branchHysteresis = s["branch_hysteresis"]?.double ?? solver.branchHysteresis
            solver.branchKeep = s["branch_keep"]?.double ?? solver.branchKeep
            solver.branchDwell = s["branch_dwell"]?.double ?? solver.branchDwell
        }
        var offline = Offline()
        if let o = json["offline"] {
            offline.sweeps = Int(o["sweeps"]?.double ?? Double(offline.sweeps))
            offline.anchorInterval = Int(o["anchor_interval"]?.double ?? Double(offline.anchorInterval))
            offline.warmSegment = o["warm_segment"]?.double ?? offline.warmSegment
            offline.warmOverlap = o["warm_overlap"]?.double ?? offline.warmOverlap
            offline.iterations = Int(o["iterations"]?.double ?? Double(offline.iterations))
            offline.velocity = o["velocity"]?.double ?? offline.velocity
            offline.acceleration = o["acceleration"]?.double ?? offline.acceleration
            offline.rootVelocity = o["root_velocity"]?.double ?? offline.rootVelocity
            offline.lock = o["lock"]?.double ?? offline.lock
            offline.lockRamp = o["lock_ramp"]?.double ?? offline.lockRamp
            offline.lockHeight = o["lock_height"]?.double ?? offline.lockHeight
            offline.ground = o["ground"]?.double ?? offline.ground
            offline.groundMargin = o["ground_margin"]?.double ?? offline.groundMargin
            offline.contact = o["contact"]?.double ?? offline.contact
            offline.contactRamp = o["contact_ramp"]?.double ?? offline.contactRamp
            offline.collision = o["collision"]?.double ?? offline.collision
            offline.collisionMargin = o["collision_margin"]?.double ?? offline.collisionMargin
            offline.collisionBand = o["collision_band"]?.double ?? offline.collisionBand
            offline.rootAcceleration = o["root_acceleration"]?.double ?? offline.rootAcceleration
            offline.handAcceleration = o["hand_acceleration"]?.double ?? offline.handAcceleration
            offline.dynamics = o["dynamics"]?.double ?? offline.dynamics
            offline.dynamicsModel = o["dynamics_model"]?.string ?? offline.dynamicsModel
            guard ["contact_forces", "force_free"].contains(offline.dynamicsModel) else {
                throw JSONError(message: "\(file): offline.dynamics_model must be contact_forces or force_free")
            }
            offline.friction = o["friction"]?.double ?? offline.friction
            offline.contactIterations = Int(o["contact_iterations"]?.double ?? Double(offline.contactIterations))
            offline.maxJointStep = o["max_joint_step"]?.double ?? offline.maxJointStep
            offline.dynamicsStride = o["dynamics_stride"]?.double ?? offline.dynamicsStride
            offline.maxRootStep = o["max_root_step"]?.double ?? offline.maxRootStep
        }
        var fixedLag = FixedLag()
        if let w = json["fixed_lag"] {
            fixedLag.reference = w["reference"]?.double ?? fixedLag.reference
            fixedLag.referenceRoot = w["reference_root"]?.double ?? fixedLag.referenceRoot
            fixedLag.dynamics = w["dynamics"]?.double ?? fixedLag.dynamics
        }
        var solverLimits: [String: (lower: Double, upper: Double)] = [:]
        for (name, range) in json["solver_limits"]?.object ?? [:] {
            guard let r = range.array, r.count == 2, let lo = r[0].double, let hi = r[1].double else {
                throw JSONError(message: "\(file): solver_limits.\(name) must be [lower, upper]")
            }
            solverLimits[name] = (lo, hi)
        }
        let mode = json["scale"]?["mode"]?.string ?? "per_chain"
        return RetargetMapping(
            skeleton: try need(json["skeleton"]?.string, "skeleton"),
            robot: try need(json["robot"]?.string, "robot"),
            rootHuman: try need(json["root"]?["human"]?.string, "root.human"),
            rootRobot: try need(json["root"]?["robot"]?.string, "root.robot"),
            scaleMode: try need(ScaleMode(rawValue: mode), "valid scale.mode"),
            scaleAnchors: (json["scale"]?["anchors"]?.array ?? []).compactMap { a -> Anchor? in
                if let name = a.string { return Anchor(joint: name, uniformHorizontal: false) }
                guard let name = a["joint"]?.string else { return nil }
                return Anchor(joint: name, uniformHorizontal: a["horizontal"]?.string == "uniform")
            },
            tposeOffsets: (json["rotation_offsets"]?.string ?? "tpose") == "tpose",
            referencePose: json["reference_pose"]?.object?.compactMapValues { $0.double },
            targets: try need(json["targets"]?.array, "targets").map { t in
                Target(human: try need(t["human"]?.string, "target.human"), robot: try need(t["robot"]?.string, "target.robot"),
                       positionWeight: t["pos"]?.double ?? 0, rotationWeight: t["rot"]?.double ?? 0,
                       twist: t["twist"]?.double ?? 1,
                       twistAxis: t["twist_axis"]?.array.map { a in Vec3(a[0].double ?? 1, a[1].double ?? 0, a[2].double ?? 0) } ?? Vec3(1, 0, 0),
                       reach: t["reach"]?.string)
            },
            solver: solver, offline: offline, fixedLag: fixedLag, solverLimits: solverLimits,
            contacts: json["contacts"]?.object?.compactMapValues { $0.string } ?? [:])
    }
}

/// A mapping resolved against concrete models: indices, kinematic chains, scales and rotation offsets.
public struct RetargetSetup: Sendable {
    public var mapping: RetargetMapping
    public var robot: RobotModel
    public var skeleton: HumanSkeleton
    public var humanJoint: [Int]        // per target
    public var robotLink: [Int]         // per target
    public var chains: [[Int]]          // per target: actuated links whose joint moves the target link
    public var rootTarget: Int
    public var rootHumanJoint: Int
    public var rootScale: Double
    public var scales: [Double]         // per target, applied to the offset from the root (or from its anchor)
    /// Per target: index of the anchor target its goal hangs from (nil: the root).
    public var anchorOf: [Int?]
    /// Per anchor target: correction (reference frame) that puts the scaled human anchor at the robot's anchor
    /// position, carried along by the parent joint's rotation relative to the reference pose.
    public var anchorCorrection: [Int: (offset: Vec3, parentJoint: Int)]
    let humanReferenceRotations: [Quat]
    /// Anchored targets whose horizontal placement is scaled uniformly from the root (with the joint whose rotation
    /// carries the horizontal axes).
    var uniformHorizontal: [Int: Int]
    /// Per target: goal rotation = human joint world rotation · offset.
    public var rotationOffsets: [Quat]
    /// Targets with `reach`: the other target, the links from it down to this target's link and the limb axis
    /// between the two (the other link's frame).
    var reachPaths: [(target: Int, parent: Int, path: [Int], twist: Vec3)] = []
    let limits: [(lower: Double, upper: Double)]

    public init(mapping: RetargetMapping, robot: RobotModel, skeleton: HumanSkeleton, humanReference: [Transform]) throws {
        self.mapping = mapping
        self.robot = robot
        self.skeleton = skeleton
        limits = mapping.solverLimits(robot)
        func human(_ name: String) throws -> Int {
            guard let j = skeleton.jointIndex(named: name) else { throw JSONError(message: "mapping: no human joint \(name)") }
            return j
        }
        func link(_ name: String) throws -> Int {
            guard let l = robot.linkIndex(named: name) else { throw JSONError(message: "mapping: no robot link \(name)") }
            return l
        }
        humanJoint = try mapping.targets.map { try human($0.human) }
        robotLink = try mapping.targets.map { try link($0.robot) }
        let rootJoint = try human(mapping.rootHuman)
        rootHumanJoint = rootJoint
        let rootLink = try link(mapping.rootRobot)
        guard let root = mapping.targets.firstIndex(where: { $0.robot == mapping.rootRobot }) else {
            throw JSONError(message: "mapping: the root link needs a target")
        }
        rootTarget = root
        chains = robotLink.map { l in
            var chain: [Int] = []
            var i: Int? = l
            while let k = i {
                if robot.links[k].dofIndex != nil { chain.append(k) }
                i = robot.links[k].parent
            }
            return chain
        }

        // Scales from chain lengths (pose independent): sum of link-to-link / joint-to-joint distances.
        let robotZero = robot.forwardKinematics(root: .identity, q: Array(repeating: 0, count: robot.dofCount))
        func robotChain(_ l: Int) -> Double {
            var length = 0.0
            var i = l
            while i != rootLink, let p = robot.links[i].parent {
                length += Core.length(robotZero[i].translation - robotZero[p].translation)
                i = p
            }
            return length
        }
        func humanChain(_ j: Int) -> Double {
            var length = 0.0
            var i = j
            while i != rootJoint, let p = skeleton.parents[i] {
                length += Core.length(skeleton.restOffsets[i])
                i = p
            }
            return length
        }
        let humanRootHeight = humanReference[rootJoint].translation.z
        switch mapping.scaleMode {
        case .none:
            rootScale = 1
            scales = Array(repeating: 1, count: mapping.targets.count)
        case .uniform:
            rootScale = robot.defaultRootPose.translation.z / humanRootHeight
            scales = Array(repeating: rootScale, count: mapping.targets.count)
        case .perChain:
            let uniform = robot.defaultRootPose.translation.z / humanRootHeight
            rootScale = uniform
            let robotLink = self.robotLink, humanJoint = self.humanJoint
            scales = mapping.targets.indices.map { t in
                let h = humanChain(humanJoint[t]), r = robotChain(robotLink[t])
                return h > 1e-6 && r > 1e-6 ? r / h : uniform
            }
        }
        rotationOffsets = Array(repeating: .identity, count: mapping.targets.count)
        humanReferenceRotations = humanReference.map(\.rotation)

        // Limb anchors (shoulders, hips). Everything scaled from the root by its chain ratio puts limbs where the
        // robot's are not: on the G1 the shoulders 4 cm too narrow and 3 cm too low (spine-dominated ratio 0.59), the
        // hip goals 2 cm too wide (ratio 1.36 over a short, differently shaped chain) and the feet 3 cm too narrow and
        // below the reach of a straight leg. An anchor goal is placed at the robot's own anchor position (in the robot
        // reference pose, carried by the parent joint's rotation); the limb's targets hang from it, scaled by the
        // limb's own length ratio.
        var anchorOf = [Int?](repeating: nil, count: mapping.targets.count)
        var anchorCorrection: [Int: (offset: Vec3, parentJoint: Int)] = [:]
        let robotRest = robot.forwardKinematics(root: .identity, q: robot.dofNames.map { mapping.referencePose?[$0] ?? 0 })
        func isDescendant(_ j: Int, of a: Int) -> Bool {
            var k = skeleton.parents[j]
            while let x = k { if x == a { return true }; k = skeleton.parents[x] }
            return false
        }
        var uniformHorizontal: [Int: Int] = [:]   // anchored target -> its anchor parent joint
        for anchor in mapping.scaleAnchors {
            let name = anchor.joint
            guard let a = mapping.targets.indices.first(where: { mapping.targets[$0].human == name && mapping.targets[$0].positionWeight > 0 })
                ?? mapping.targets.indices.first(where: { mapping.targets[$0].human == name }) else {
                throw JSONError(message: "mapping: scale anchor \(name) has no target")
            }
            let joint = humanJoint[a]
            guard let parent = skeleton.parents[joint] else { throw JSONError(message: "mapping: scale anchor \(name) is the root") }
            let robotOffset = robotRest[robotLink[a]].translation - robotRest[rootLink].translation
            let humanOffset = humanReference[joint].translation - humanReference[rootJoint].translation
            anchorCorrection[a] = (robotOffset - scales[a] * humanOffset, parent)
            for t in mapping.targets.indices where t != a && isDescendant(humanJoint[t], of: joint) {
                anchorOf[t] = a
                if anchor.uniformHorizontal { uniformHorizontal[t] = parent }
                if mapping.scaleMode == .perChain {
                    // Straight-line anchor-to-target distance in the two reference poses: summed link lengths
                    // overstate a zig-zag chain (the G1 hip, 1.5 cm on the leg).
                    let r = Core.length(robotRest[robotLink[t]].translation - robotRest[robotLink[a]].translation)
                    let h = Core.length(humanReference[humanJoint[t]].translation - humanReference[joint].translation)
                    if h > 1e-6 && r > 1e-6 { scales[t] = r / h }
                }
            }
        }
        self.anchorOf = anchorOf
        self.anchorCorrection = anchorCorrection
        self.uniformHorizontal = uniformHorizontal

        for t in mapping.targets.indices {
            guard let name = mapping.targets[t].reach else { continue }
            guard let p = mapping.targets.indices.first(where: { mapping.targets[$0].human == name }) else {
                throw JSONError(message: "mapping: reach target \(name) has no target")
            }
            var path: [Int] = []
            var i: Int? = robotLink[t]
            while let k = i, k != robotLink[p] { path.append(k); i = robot.links[k].parent }
            guard i != nil else { throw JSONError(message: "mapping: \(mapping.targets[t].robot) is not below \(mapping.targets[p].robot)") }
            let axis = robotRest[robotLink[p]].rotation.conjugate.rotate(robotRest[robotLink[t]].translation - robotRest[robotLink[p]].translation)
            reachPaths.append((t, p, path.reversed(), normalize(axis)))
        }
    }

    /// Rotation closest to `goal` (relative to the parent link) that the joints on `path` reach within limits, plus a
    /// share `keep` of the twist about the limb axis `twist` (a free extra joint in the fit, which the hip produces):
    /// projected Gauss-Newton from the joints at 0. The rest is dropped. On a kneeling person's foot it is the
    /// dorsiflexion of the tucked toes and the foot's turn against the shin lying on the floor, which the solver
    /// otherwise found by twisting the whole leg (hip yaw and roll) and splayed the shin up to 90° sideways.
    func reachable(_ goal: Quat, path: [Int], twist: Vec3, keep: Double, limits: [(lower: Double, upper: Double)]) -> Quat {
        let joints = path.indices.filter { robot.links[path[$0]].dofIndex != nil }
        var q = [Double](repeating: 0, count: path.count)
        var spin = 0.0
        func chain() -> (rotation: Quat, axes: [Vec3]) {
            var r = Quat(axis: twist, angle: spin)
            var axes: [Vec3] = [twist]
            for (k, l) in path.enumerated() {
                let link = robot.links[l]
                r = r * link.jointOrigin.rotation
                if case .revolute(let axis) = link.joint, link.dofIndex != nil {
                    axes.append(r.rotate(axis))
                    r = r * Quat(axis: axis, angle: q[k])
                }
                r = r * link.jointToChild.rotation
            }
            return (r, axes)
        }
        for _ in 0..<12 {
            let (r, axes) = chain()
            var e = goal * r.conjugate
            if e.w < 0 { e = Quat(w: -e.w, v: -e.v) }
            let s = length(e.v)
            if s < 1e-9 { break }
            let error = 2 * atan2(s, e.w) / s * e.v
            // Normal equations (n ≤ 3 joints), damped.
            let n = axes.count
            var a = [[Double]](repeating: [Double](repeating: 0, count: n + 1), count: n)
            for i in 0..<n {
                for j in 0..<n { a[i][j] = dot(axes[i], axes[j]) + (i == j ? 1e-6 : 0) }
                a[i][n] = dot(axes[i], error)
            }
            for c in 0..<n {
                let pivot = (c..<n).max { abs(a[$0][c]) < abs(a[$1][c]) }!
                a.swapAt(c, pivot)
                for r in 0..<n where r != c {
                    let f = a[r][c] / a[c][c]
                    for k in c...n { a[r][k] -= f * a[c][k] }
                }
            }
            spin += a[0][n] / a[0][0]
            for (i, k) in joints.enumerated() {
                let range = limits[robot.links[path[k]].dofIndex!]
                q[k] = min(max(q[k] + a[i + 1][n] / a[i + 1][i + 1], range.lower), range.upper)
            }
        }
        spin *= keep
        return chain().rotation
    }

    /// Goal poses for one frame of human world joint transforms (already shifted so the ground is z = 0).
    public func goals(human world: [Transform]) -> [Transform] {
        let root = world[rootHumanJoint].translation
        let rootGoal = rootScale * root
        var position = [Vec3](repeating: .zero, count: mapping.targets.count)
        for t in mapping.targets.indices where anchorOf[t] == nil {
            position[t] = t == rootTarget ? rootGoal : rootGoal + scales[t] * (world[humanJoint[t]].translation - root)
            if let c = anchorCorrection[t] {
                // In the anchor parent's (chest's, pelvis's) current frame: the reference correction carried along.
                let turn = world[c.parentJoint].rotation * humanReferenceRotations[c.parentJoint].conjugate
                position[t] += turn.rotate(c.offset)
            }
        }
        for t in mapping.targets.indices {
            guard let a = anchorOf[t] else { continue }
            position[t] = position[a] + scales[t] * (world[humanJoint[t]].translation - world[humanJoint[a]].translation)
            if let parent = uniformHorizontal[t] {
                let turn = world[parent].rotation * humanReferenceRotations[parent].conjugate
                let uniform = rootGoal + rootScale * (world[humanJoint[t]].translation - root)
                for axis in [Vec3(1, 0, 0), Vec3(0, 1, 0)] {
                    let u = turn.rotate(axis)
                    position[t] += dot(uniform - position[t], u) * u
                }
            }
        }
        var goals = mapping.targets.indices.map { t in
            Transform(rotation: (world[humanJoint[t]].rotation * rotationOffsets[t]).normalized, translation: position[t])
        }
        for (t, p, path, twist) in reachPaths {
            let parent = goals[p].rotation
            // Twist about the shin comes from the hip, which turns the whole leg: right for a foot flat under an
            // upright shin (standing, squatting, sitting), but a kneeling person's shin lies on the floor and turning
            // the hip swings it sideways. Kept in proportion to how upright the person's shin is.
            let shin = world[humanJoint[t]].translation - world[humanJoint[p]].translation
            let keep = length(shin) > 1e-6 ? max(0, -normalize(shin).z) : 1
            goals[t].rotation = (parent * reachable(parent.conjugate * goals[t].rotation, path: path, twist: twist, keep: keep,
                                                    limits: limits)).normalized
        }
        return goals
    }
}
