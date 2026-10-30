// Offline whole-trajectory solver driver (`Kernels/offline_ik_group.slang`): coloured block Gauss-Seidel
// sweeps over all frames of all clips, with stance anchors recomputed between sweeps.

import Compute
import Core
import Dynamics
import Kinematics

#if canImport(Glibc)
import Glibc
#endif

/// One clip for the offline solver.
public struct OfflineClip: Sendable {
    public var fps: Double
    public var goals: [[Transform]]               // [frame][target]
    public var initial: RobotTrajectory           // warm start (the realtime solution)
    /// Per frame: [left heel, left toe, right heel, right toe] human contact.
    public var footContact: [[Bool]]
    /// Per frame, per robot link: ground-contact equality weight (0: barrier only).
    public var linkContact: [[Float]]
}

public final class OfflineSolver {
    public let setup: RetargetSetup
    public let geometry: RobotGeometry
    let device: ComputeDevice
    let kernel: Kernel
    /// One group per fixed-lag window, all its sweeps in one dispatch (FixedLagSolver).
    let windowKernel: Kernel
    let model: [String: DeviceBuffer]
    let scratch: [String: ScratchBuffer]
    let soleHeel: [[Bool]]              // per foot, per sole point: true for the heel points
    public let pairCount: Int
    /// The link pairs the solver keeps apart (enclosing capsules, not overlapping at rest).
    public let collisionPairsForDiagnostics: [(Int, Int)]
    let solverCapsules: [Capsule?]
    let handPairs: [(Int, Int)]
    let armTargets: [Int32]
    let linkSegment: [SIMD4<Int32>]   // (first, last target, arm: 1 / body: 0, -)
    let dynamics: RobotDynamics

    public init(device: ComputeDevice, setup: RetargetSetup, geometry: RobotGeometry) throws {
        guard device.backend == .cuda else { throw ComputeError(message: "the offline solver needs a GPU backend (v1: CUDA)") }
        self.device = device
        self.setup = setup
        self.geometry = geometry
        kernel = try device.loadKernel(module: "offline_ik_group", entryPoint: "sweep")
        windowKernel = try device.loadKernel(module: "offline_ik_group", entryPoint: "sweepWindow")
        let robot = setup.robot
        dynamics = RobotDynamics(robot: robot)
        guard geometry.soles.count == 2, geometry.soles.allSatisfy({ $0.points.count == 4 }) else {
            throw ComputeError(message: "offline solver expects two feet with four sole points")
        }
        let links = robot.links
        let limits = links.compactMap { link in link.dofIndex.map { ($0, link.limits!) } }.sorted { $0.0 < $1.0 }
        // Actuated ancestor chains of every link.
        var linkChainStart: [Int32] = [], linkChainCount: [Int32] = [], linkChainLinks: [Int32] = []
        for l in links.indices {
            var chain: [Int32] = []
            var i: Int? = l
            while let k = i {
                if links[k].dofIndex != nil { chain.append(Int32(k)) }
                i = links[k].parent
            }
            linkChainStart.append(Int32(linkChainLinks.count))
            linkChainCount.append(Int32(chain.count))
            linkChainLinks += chain
        }
        var targetChainStart: [Int32] = [], targetChainCount: [Int32] = [], targetChainLinks: [Int32] = []
        for chain in setup.chains {
            targetChainStart.append(Int32(targetChainLinks.count))
            targetChainCount.append(Int32(chain.count))
            targetChainLinks += chain.map { Int32($0) }
        }
        // Solver capsules: enclosing radius; pairs that already overlap at rest (incl. margin) are skipped.
        // Solver capsules: enclosing (every surface point inside), so the surface-based check passes.
        let outer = geometry.outerCapsules
        solverCapsules = outer
        let rest = robot.forwardKinematics(root: robot.defaultRootPose, q: Array(repeating: 0, count: robot.dofCount))
        let margin = setup.mapping.offline.collisionMargin
        // Hands: the wrist-roll link and everything below it, per side. Pairs between the two hands become one
        // hand-to-hand row in the kernel (see there); the remaining pairs get a row each.
        // Hands (wrist-roll link and below): pairs between the two hands become one hand-to-hand row in the kernel.
        func hand(_ side: String) -> [Int] {
            guard let root = links.firstIndex(where: { $0.name == "\(side)_wrist_roll_link" }) else { return [] }
            return links.indices.filter { l in
                var i: Int? = l
                while let k = i { if k == root { return true }; i = links[k].parent }
                return false
            }.filter { outer[$0] != nil }
        }
        let leftHand = hand("left"), rightHand = hand("right")
        let candidates = geometry.collisionPairs.filter { a, b in
            guard let ca = outer[a], let cb = outer[b] else { return false }
            return RobotGeometry.penetration(ca, rest[a], cb, rest[b]) < -margin
        }
        func isHandPair(_ a: Int, _ b: Int) -> Bool {
            (leftHand.contains(a) && rightHand.contains(b)) || (leftHand.contains(b) && rightHand.contains(a))
        }
        let pairs = candidates.filter { !isHandPair($0.0, $0.1) }
        handPairs = candidates.filter { isHandPair($0.0, $0.1) }.map { leftHand.contains($0.0) ? $0 : ($0.1, $0.0) }
        // Shoulder, elbow and hand position targets per side (the kernel uses the hands').
        let targets = setup.mapping.targets
        func target(_ human: String) -> Int32 {
            Int32(targets.indices.first { targets[$0].human == human && targets[$0].positionWeight > 0 } ?? -1)
        }
        armTargets = ["LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand"].map(target)
        // Limb links and the human segment each stands for. Arms: shoulder (pitch / roll links) the shoulder point,
        // upper arm (shoulder-yaw link) shoulder -> elbow, forearm (elbow link) elbow -> hand, hand links the hand
        // point. Legs: hip (pitch / roll links) the hip point, thigh (hip-yaw link) hip -> knee, shin (knee link)
        // knee -> ankle. Pelvis: the hips point.
        let armTargets = self.armTargets
        let legTargets = ["LeftLeg", "LeftShin", "LeftFoot", "RightLeg", "RightShin", "RightFoot"].map(target)
        let hips = target(setup.mapping.rootHuman)
        linkSegment = links.indices.map { l -> SIMD4<Int32> in
            let name = links[l].name
            if name == setup.mapping.rootRobot { return hips >= 0 ? SIMD4(hips, hips, 0, 0) : SIMD4(-1, -1, 0, 0) }
            for (side, base) in [("left_", 0), ("right_", 3)] where name.hasPrefix(side) {
                let (s, e, h) = (armTargets[base], armTargets[base + 1], armTargets[base + 2])
                if s >= 0, e >= 0, h >= 0 {
                    if name.contains("shoulder_yaw") { return SIMD4(s, e, 1, 0) }
                    if name.contains("shoulder") { return SIMD4(s, s, 1, 0) }
                    if name.contains("elbow") { return SIMD4(e, h, 1, 0) }
                    if name.contains("wrist") || name.contains("hand") { return SIMD4(h, h, 1, 0) }
                }
                let (p, k, a) = (legTargets[base], legTargets[base + 1], legTargets[base + 2])
                if p >= 0, k >= 0, a >= 0 {
                    if name.contains("hip_yaw") { return SIMD4(p, k, 0, 0) }
                    if name.contains("hip") { return SIMD4(p, p, 0, 0) }
                    if name.contains("knee") { return SIMD4(k, a, 0, 0) }
                }
            }
            return SIMD4(-1, -1, 0, 0)
        }
        pairCount = pairs.count
        collisionPairsForDiagnostics = pairs
        soleHeel = geometry.soles.map { foot in
            let meanX = foot.points.map(\.x).reduce(0, +) / Double(foot.points.count)
            return foot.points.map { $0.x < meanX }
        }
        func f4(_ v: Vec3, _ w: Double = 0) -> SIMD4<Float> { SIMD4(Float(v.x), Float(v.y), Float(v.z), Float(w)) }
        model = [
            "linkParent": try device.makeBuffer(links.map { Int32($0.parent ?? -1) }),
            "linkDof": try device.makeBuffer(links.map { Int32($0.dofIndex ?? -1) }),
            "jointOriginRot": try device.makeBuffer(links.map { $0.jointOrigin.rotation.device }),
            "jointOriginPos": try device.makeBuffer(links.map { $0.jointOrigin.translation.device }),
            "jointToChildRot": try device.makeBuffer(links.map { $0.jointToChild.rotation.device }),
            "jointToChildPos": try device.makeBuffer(links.map { $0.jointToChild.translation.device }),
            "jointAxis": try device.makeBuffer(links.map { link -> SIMD4<Float> in
                if case .revolute(let a) = link.joint { return a.device }
                return SIMD4(0, 0, 1, 0)
            }),
            "dofLimit": try device.makeBuffer(setup.mapping.solverLimits(robot).map { SIMD2(Float($0.lower), Float($0.upper)) }),
            "targetLink": try device.makeBuffer(setup.robotLink.map { Int32($0) }),
            "targetTwist": try device.makeBuffer(setup.mapping.targets.map { t in
                SIMD4(Float(t.twistAxis.x), Float(t.twistAxis.y), Float(t.twistAxis.z), Float(t.twist))
            }),
            "targetWeight": try device.makeBuffer(setup.mapping.targets.map { SIMD2(Float($0.positionWeight), Float($0.rotationWeight)) }),
            "chainStart": try device.makeBuffer(targetChainStart),
            "chainCount": try device.makeBuffer(targetChainCount),
            "chainLinks": try device.makeBuffer(targetChainLinks.isEmpty ? [0] : targetChainLinks),
            "linkChainStart": try device.makeBuffer(linkChainStart),
            "linkChainCount": try device.makeBuffer(linkChainCount),
            "linkChainLinks": try device.makeBuffer(linkChainLinks.isEmpty ? [0] : linkChainLinks),
            "soleLink": try device.makeBuffer(geometry.soles.map { Int32($0.link) }),
            "solePoint": try device.makeBuffer(geometry.soles.flatMap { $0.points.map { f4($0) } }),
            "capsuleA": try device.makeBuffer(outer.map { c in c.map { f4($0.a, $0.radius) } ?? .zero }),
            "capsuleB": try device.makeBuffer(outer.map { c in c.map { f4($0.b, $0.radius) } ?? .zero }),
            "linkPointRange": try device.makeBuffer(geometry.linkPoints.reduce(into: [SIMD2<Int32>]()) { out, points in
                let first = out.last.map { $0.x + $0.y } ?? 0
                out.append(SIMD2(first, Int32(points.count)))
            }),
            "linkPoints": try device.makeBuffer(geometry.linkPoints.flatMap { $0.map { f4($0) } }),
            "linkBound": try device.makeBuffer(geometry.linkPoints.map { points -> SIMD4<Float> in
                guard !points.isEmpty else { return .zero }
                let lo = points.reduce(points[0]) { Vec3(min($0.x, $1.x), min($0.y, $1.y), min($0.z, $1.z)) }
                let hi = points.reduce(points[0]) { Vec3(max($0.x, $1.x), max($0.y, $1.y), max($0.z, $1.z)) }
                let centre = 0.5 * (lo + hi)
                return f4(centre, points.map { length($0 - centre) }.max()! * 1.0001)
            }),
            "collisionPairs": try device.makeBuffer(pairs.isEmpty ? [SIMD2<Int32>(0, 0)] : pairs.map { SIMD2(Int32($0.0), Int32($0.1)) }),
            "linkCom": try device.makeBuffer(links.map { f4($0.inertial?.centerOfMass ?? .zero, $0.inertial?.mass ?? 0) }),
            "linkAncestors": try device.makeBuffer(links.indices.map { l -> SIMD2<UInt32> in
                var mask = SIMD2<UInt32>(0, 0)
                var i: Int? = l
                while let k = i {
                    if k < 32 { mask.x |= 1 << UInt32(k) } else { mask.y |= 1 << UInt32(k - 32) }
                    i = links[k].parent
                }
                return mask
            }),
        ]
        precondition(links.count <= 40 && robot.dofCount <= 30 && setup.mapping.targets.count <= 18,
                     "offline kernel limits: 40 links, 30 dofs, 18 targets")
        scratch = Dictionary(uniqueKeysWithValues: ((0..<Self.maxColors).map { "frameList\($0)" } + [
            "frameRange", "accelerationLimit", "trajRootPos", "trajRootRot", "trajQ", "outCostOffline",
            "goalPos", "goalRot", "soleAnchor", "linkGroundContact", "trajLinkCom", "contactPoint", "frameFps", "contactLambda",
            "trajSole", "trajPatch",
        ]).map { ($0, ScratchBuffer(device: device)) })
    }

    /// Wall-clock time of the last solve.
    public private(set) var lastDuration: Duration = .zero

    public func solve(_ clips: [OfflineClip]) throws -> [RobotTrajectory] {
        let start = ContinuousClock.now
        let robot = setup.robot, o = setup.mapping.offline, s = setup.mapping.solver
        let T = setup.mapping.targets.count, n = robot.dofCount, L = robot.links.count
        var offsets: [Int] = []
        // Dynamics accelerations over frames t ± K, K ≈ dynamics_stride seconds; colours: 2K + 1.
        let stride = max(1, Int((o.dynamicsStride * (clips.first?.fps ?? 120)).rounded()))
        let colors = min(2 * stride + 1, Self.maxColors - 1)   // odd, ≤ the scratch lists
        var frameRange: [SIMD2<Int32>] = [], lists = [[Int32]](repeating: [], count: colors)
        var rootPos: [SIMD4<Float>] = [], rootRot: [SIMD4<Float>] = [], qs: [Float] = []
        var goalPos: [SIMD4<Float>] = [], goalRot: [SIMD4<Float>] = [], contact: [Float] = []
        var frameFps: [Float] = [], linkCom: [SIMD4<Float>] = []
        // Joint acceleration limits per frame: effort-limited at the warm-start pose (the validator judges each frame
        // at its own pose), times the margin.
        var accelerationLimit: [Float] = []
        for clip in clips {
            let first = rootPos.count
            offsets.append(first)
            let contactWeights = Self.rampedContact(clip.linkContact, ramp: o.contactRamp * clip.fps)
            for f in 0..<clip.goals.count {
                frameRange.append(SIMD2(Int32(first), Int32(first + clip.goals.count)))
                lists[f % colors].append(Int32(first + f))
                rootPos.append(clip.initial.root[f].translation.device)
                rootRot.append(clip.initial.root[f].rotation.device)
                qs += clip.initial.q[f].map(Float.init)
                for goal in clip.goals[f] {
                    goalPos.append(goal.translation.device)
                    goalRot.append(goal.rotation.device)
                }
                contact += contactWeights[f].map { $0 * Float(o.contact) }
                frameFps.append(Float(clip.fps))
                let world = robot.forwardKinematics(root: clip.initial.root[f], q: clip.initial.q[f])
                linkCom += robot.links.indices.map { world[$0].apply(robot.links[$0].inertial?.centerOfMass ?? .zero).device }
                if o.accelerationLimit > 0 {
                    accelerationLimit += dynamics.effortLimitedAcceleration(world: world).map { Float(min($0 * o.accelerationLimitMargin, 1e6)) }
                }
                precondition(clip.goals[f].count == T && clip.linkContact[f].count == L)
            }
        }
        let frames = rootPos.count
        func buf<V: BitwiseCopyable>(_ name: String, _ values: [V]) throws -> DeviceBuffer { try scratch[name]!.upload(values) }
        let rp = try buf("trajRootPos", rootPos), rr = try buf("trajRootRot", rootRot), q = try buf("trajQ", qs)
        let cost = try scratch["outCostOffline"]!.ensure(Float.self, count: frames)
        var arguments = try constantArguments(stride: stride)
        arguments += [
            .buffer("frameRange", try buf("frameRange", frameRange)),
            .buffer("trajRootPos", rp), .buffer("trajRootRot", rr), .buffer("trajQ", q), .buffer("outCostOffline", cost),
            .buffer("goalPos", try buf("goalPos", goalPos)), .buffer("goalRot", try buf("goalRot", goalRot)),
            .buffer("linkGroundContact", try buf("linkGroundContact", contact)),
            .buffer("trajLinkCom", try buf("trajLinkCom", linkCom)),
            .buffer("contactLambda", try buf("contactLambda", [Float](repeating: 0, count: frames * Self.contactPoints * 4))), .buffer("frameFps", try buf("frameFps", frameFps)),
            .value("referenceWeight", Float(0)), .value("referenceRootWeight", Float(0)),
            .buffer("accelerationLimit", try buf("accelerationLimit", accelerationLimit.isEmpty ? [Float(0)] : accelerationLimit)),
        ]
        let listBuffers = try (0..<colors).map { try buf("frameList\($0)", lists[$0].isEmpty ? [Int32(0)] : lists[$0]) }

        // Stance anchors and contact points follow the solution; they settle quickly, so they are refreshed every
        // `anchorInterval` sweeps from the sole points and contact patches the sweep before wrote (the only host
        // round trip; the sweeps in between stay on the device). The first ones come from the warm start.
        let interval = max(1, o.anchorInterval)
        let soleOut = try scratch["trajSole"]!.ensure(SIMD4<Float>.self, count: frames * 8)
        let patchOut = try scratch["trajPatch"]!.ensure(SIMD4<Float>.self, count: frames * Self.contactPoints)
        arguments += [.buffer("trajSole", soleOut), .buffer("trajPatch", patchOut)]
        var anchors = try buf("soleAnchor", [SIMD4<Float>(0, 0, 0, 0)]), contacts = anchors
        for sweep in 0..<o.sweeps {
            // Continuation: constraint weights ramp from 10 % to full over the first half of the sweeps.
            let scale = min(1, 0.1 + 0.9 * Double(sweep) / max(1, Double(o.sweeps) * 0.5))
            if sweep % interval == 0 {
                let features = sweep == 0
                    ? hostFeatures(clips: clips, trajectories: clips.map(\.initial))
                    : Features(sole: try soleOut.download(as: SIMD4<Float>.self, count: frames * 8),
                               patch: try patchOut.download(as: SIMD4<Float>.self, count: frames * Self.contactPoints))
                let (anchorValues, contactValues) = stanceAnchors(clips: clips, offsets: offsets, features: features)
                anchors = try buf("soleAnchor", anchorValues)
                contacts = try buf("contactPoint", contactValues)
            }
            for color in 0..<colors where !lists[color].isEmpty {
                try device.dispatch(kernel, threads: lists[color].count * Self.groupSize, arguments: arguments + [
                    .buffer("frameList", listBuffers[color]), .value("numFrameList", UInt32(lists[color].count)),
                    .buffer("soleAnchor", anchors), .buffer("contactPoint", contacts), .value("constraintScale", Float(scale)),
                    .value("writeFeatures", UInt32((sweep + 1) % interval == 0 ? 1 : 0)),
                ])
            }
        }
        let current = try download(clips: clips, offsets: offsets, rootPos: rp, rootRot: rr, q: q, cost: cost)
        lastDuration = ContinuousClock.now - start
        return current
    }

    /// Kernel arguments that do not depend on the frames: the model, the weights and the per-link / per-column
    /// tables; built once per dynamics stride.
    /// `window`: the dynamics weight of a fixed-lag window (FixedLagSolver), in place of `offline.dynamics`.
    func constantArguments(stride: Int, window: Double? = nil) throws -> [KernelArgument] {
        let key = "\(stride) \(window.map { "\($0)" } ?? "-")"
        if let cached = constants[key] { return cached }
        let robot = setup.robot, o = setup.mapping.offline, s = setup.mapping.solver
        let T = setup.mapping.targets.count, n = robot.dofCount, L = robot.links.count
        let linkDepth = robot.links.map { link -> UInt32 in
            var depth: UInt32 = 0, parent = link.parent
            while let p = parent { depth += 1; parent = robot.links[p].parent }
            return depth
        }
        // Per column, the targets whose position / rotation rows reach it (the kernel's closed-form target rows).
        let dofLink = (0..<n).map { d in Int32(robot.links.firstIndex { $0.dofIndex == d }!) }
        let colTargets = (0..<(6 + n)).map { a -> SIMD2<UInt32> in
            var mask = SIMD2<UInt32>(0, 0)
            for (t, target) in setup.mapping.targets.enumerated() {
                guard a < 6 || setup.chains[t].contains(Int(dofLink[a - 6])) else { continue }
                if target.positionWeight > 0 { mask.x |= 1 << UInt32(t) }
                if target.rotationWeight > 0 && a >= 3 { mask.y |= 1 << UInt32(t) }
            }
            return mask
        }
        var arguments: [KernelArgument] = model.map { .buffer($0.key, $0.value) }
        // Fixed-lag windows leave the acceleration limit off (no per-frame limits); batch solves bind theirs.
        if window != nil { arguments.append(.buffer("accelerationLimit", try device.makeBuffer([Float(0)]))) }
        arguments += [
            .value("numLinks", UInt32(L)), .value("numDofs", UInt32(n)), .value("numTargets", UInt32(T)),
            .buffer("linkDepth", try device.makeBuffer(linkDepth)), .value("maxDepth", UInt32(linkDepth.max() ?? 0)),
            .buffer("colTargets", try device.makeBuffer(colTargets)), .buffer("dofLink", try device.makeBuffer(dofLink)),
            .value("numPairs", UInt32(pairCount)), .value("numHandPairs", UInt32(handPairs.count)),
            .buffer("handPairs", try device.makeBuffer(handPairs.isEmpty ? [SIMD2<Int32>(0, 0)] : handPairs.map { SIMD2(Int32($0.0), Int32($0.1)) })),
            .buffer("armTargets", try device.makeBuffer(armTargets)),
            .buffer("linkSegment", try device.makeBuffer(linkSegment)),
            .buffer("medianRadius", try device.makeBuffer(geometry.capsules.map { Float($0?.radius ?? 0) })),
            .value("sweepIterations", UInt32(o.iterations)),
            .value("damping", Float(s.damping)), .value("posture", Float(s.posture)),
            .value("limitMargin", Float(s.limitMargin)), .value("limitWeight", Float(s.limitWeight)),
            .value("velocityWeight", Float(o.velocity)), .value("accelerationWeight", Float(o.acceleration)),
            .value("rootVelocityWeight", Float(o.rootVelocity)), .value("groundWeight", Float(o.ground)),
            .value("groundMargin", Float(o.groundMargin)), .value("collisionWeight", Float(o.collision)),
            .value("collisionMargin", Float(o.collisionMargin)), .value("collisionBand", Float(o.collisionBand)),
            .value("smoothLinks", SIMD2<Int32>(Int32(robot.linkIndex(named: "left_wrist_yaw_link") ?? robot.linkIndex(named: "left_wrist_roll_link") ?? -1),
                                               Int32(robot.linkIndex(named: "right_wrist_yaw_link") ?? robot.linkIndex(named: "right_wrist_roll_link") ?? -1))),
            .value("linkAccelerationWeight", Float(o.handAcceleration)),
            .value("accelerationLimitWeight", Float(window == nil ? o.accelerationLimit : 0)),
            .value("rootAccelerationWeight", Float(o.rootAcceleration)), .value("dynamicsWeight", Float(window ?? o.dynamics)),
            .value("friction", Float(o.friction)), .value("totalMass", Float(robot.totalMass)),
            .value("contactIterations", UInt32(o.contactIterations)),
            .value("dynamicsStride", UInt32(stride)), .value("forceFree", UInt32(o.dynamicsModel == "force_free" ? 1 : 0)), .value("maxJointStep", Float(o.maxJointStep)), .value("maxRootStep", Float(o.maxRootStep)),
        ]
        constants[key] = arguments
        return arguments
    }
    var constants: [String: [KernelArgument]] = [:]

    func download(clips: [OfflineClip], offsets: [Int], rootPos: DeviceBuffer, rootRot: DeviceBuffer, q: DeviceBuffer,
                  cost: DeviceBuffer) throws -> [RobotTrajectory] {
        let n = setup.robot.dofCount
        let frames = offsets.last.map { $0 + clips.last!.goals.count } ?? 0
        let rp = try rootPos.download(as: SIMD4<Float>.self, count: frames)
        let rr = try rootRot.download(as: SIMD4<Float>.self, count: frames)
        let qs = try q.download(as: Float.self, count: frames * n)
        let cs = try cost.download(as: Float.self, count: frames)
        return clips.indices.map { c in
            let range = offsets[c]..<(offsets[c] + clips[c].goals.count)
            return RobotTrajectory(
                fps: clips[c].fps,
                root: range.map { Transform(rotation: Quat(device: rr[$0]).normalized, translation: Vec3(device: rp[$0])) },
                q: range.map { f in (0..<n).map { Double(qs[f * n + $0]) } },
                cost: range.map { Double(cs[$0]) })
        }
    }

    /// Diagnostics: the dynamics contact points (world, w = link + 1) per frame the solver would use for these
    /// trajectories.
    public func contactPointsForDiagnostics(clips: [OfflineClip], trajectories: [RobotTrajectory]) -> [SIMD4<Float>] {
        var offsets: [Int] = [], first = 0
        for clip in clips { offsets.append(first); first += clip.goals.count }
        return stanceAnchors(clips: clips, offsets: offsets, features: hostFeatures(clips: clips, trajectories: trajectories)).1
    }

    static let contactPoints = 24   // CONTACT_POINTS in the kernel
    static let groupSize = 64       // O_GROUP in the kernel

    /// Link contact weights ramped in and out over `ramp` frames at the ends of each contact run: switched on at full
    /// weight, a hand whose person's fingertips touch the floor while the scaled hand target is higher dropped
    /// ~7 cm in a few frames, and jumped back up at release (230127 body check A155, 20.6 s / 20.9 s).
    static func rampedContact(_ weights: [[Float]], ramp: Double) -> [[Float]] {
        guard ramp > 1, let links = weights.first?.count else { return weights }
        var out = weights
        let F = weights.count
        for l in 0..<links {
            var f = 0
            while f < F {
                guard weights[f][l] > 0 else { f += 1; continue }
                var e = f
                while e < F && weights[e][l] > 0 { e += 1 }
                for g in f..<e { out[g][l] *= Float(min(1, Double(min(g - f + 1, e - g)) / ramp)) }
                f = e
            }
        }
        return out
    }

    /// Contact patch of a link held on the ground: the extreme (±x, ±y) points among its surface points within
    /// 1 cm of its lowest one (as the validator's contact model), projected onto the ground.
    static func contactPatch(_ points: [Vec3]) -> [Vec3] {
        guard let low = points.map(\.z).min() else { return [] }
        let near = points.filter { $0.z < low + 0.01 }
        var out: [Vec3] = []
        for (sx, sy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            let p = near.max { sx * $0.x + sy * $0.y < sx * $1.x + sy * $1.y }!
            if !out.contains(p) { out.append(p) }
        }
        return out
    }
    static let maxColors = 16

    /// Per frame of a batch (clips back to back): the world sole points (8) and the ground contact patches of the
    /// links in mapped contact, (x, y, 0, link + 1) in link order, `contactPoints` slots (w = 0: none). The sweep
    /// kernel writes them for its result; the host version (FK) serves the warm start and diagnostics.
    struct Features {
        var sole: [SIMD4<Float>]
        var patch: [SIMD4<Float>]
    }

    func hostFeatures(clips: [OfflineClip], trajectories: [RobotTrajectory]) -> Features {
        let perClip = zip(clips, trajectories).map { hostFeatures(clip: $0, trajectory: $1) }
        return Features(sole: perClip.flatMap(\.sole), patch: perClip.flatMap(\.patch))
    }

    func hostFeatures(clip: OfflineClip, trajectory traj: RobotTrajectory) -> Features {
        let robot = setup.robot, geometry = self.geometry, chunk = 128, F = clip.goals.count
        let perFrame = parallelMap((F + chunk - 1) / chunk) { c -> [(sole: [SIMD4<Float>], patch: [SIMD4<Float>])] in
            (c * chunk..<min(F, (c + 1) * chunk)).map { f in
                Self.frameFeatures(robot: robot, geometry: geometry, world: robot.forwardKinematics(root: traj.root[f], q: traj.q[f]),
                                   inContact: { clip.linkContact[f][$0] > 0 })
            }
        }.flatMap { $0 }
        return Features(sole: perFrame.flatMap(\.sole), patch: perFrame.flatMap(\.patch))
    }

    /// One frame's sole points (8) and contact patches (`contactPoints` slots) from its link transforms.
    static func frameFeatures(robot: RobotModel, geometry: RobotGeometry, world: [Transform], inContact: (Int) -> Bool)
        -> (sole: [SIMD4<Float>], patch: [SIMD4<Float>]) {
        let sole = geometry.soles.flatMap { foot in foot.points.map { world[foot.link].apply($0).device } }
        var patch: [SIMD4<Float>] = []
        for l in robot.links.indices where inContact(l) {
            patch += OfflineSolver.contactPatch(geometry.linkPoints[l].map { world[l].apply($0) })
                .map { SIMD4(Float($0.x), Float($0.y), 0, Float(l + 1)) }
        }
        patch = Array(patch.prefix(contactPoints))
        return (sole, patch + Array(repeating: .zero, count: contactPoints - patch.count))
    }

    /// Stance anchors for every frame and sole point: during each run of human contact of the point's part
    /// (heel or toe of that foot) the point is pulled to the run's mean ground position at z = 0.
    /// Also returns the dynamics contact points per frame: the locked sole anchors, then the contact patch of
    /// every link in mapped body-part contact. Clips start at `offsets` in `features` and in the results; this runs
    /// between sweeps while the device waits, so clips write straight into the batch arrays.
    func stanceAnchors(clips: [OfflineClip], offsets: [Int], features: Features) -> ([SIMD4<Float>], [SIMD4<Float>]) {
        let o = setup.mapping.offline, soleHeel = self.soleHeel, soleLinks = geometry.soles.map(\.link)
        let P = Self.contactPoints, frames = offsets.last.map { $0 + clips.last!.goals.count } ?? 0
        let sole = features.sole, patch = features.patch
        var anchors = [SIMD4<Float>](repeating: .zero, count: frames * 8)
        var contacts = [SIMD4<Float>](repeating: .zero, count: frames * P)
        anchors.withUnsafeMutableBufferPointer { anchorBuffer in
        contacts.withUnsafeMutableBufferPointer { contactBuffer in
            let anchorOut = SendablePointer(base: anchorBuffer.baseAddress!)
            let contactOut = SendablePointer(base: contactBuffer.baseAddress!)
            _ = parallelMap(clips.count) { c in
                let clip = clips[c], F = clip.goals.count, first = offsets[c]
                let anchors = anchorOut.base + first * 8, contacts = contactOut.base + first * P
                sole.withUnsafeBufferPointer { soleBuffer in
                    let points = soleBuffer.baseAddress! + first * 8
                    for foot in 0..<2 {
                        for i in 0..<4 {
                            let k = foot * 4 + i
                            let contact = foot * 2 + (soleHeel[foot][i] ? 0 : 1)   // footContact index of the point's part
                            var f = 0
                            while f < F {
                                guard clip.footContact[f][contact] else { f += 1; continue }
                                var e = f
                                var mean = Vec3.zero
                                while e < F && clip.footContact[e][contact] { mean += Vec3(device: points[e * 8 + k]); e += 1 }
                                mean /= Double(e - f)
                                // A corner that stays well above the foot's lowest corner during the run (the robot on
                                // its toes while the person's whole foot touches, e.g. crawling) is not locked: pinning
                                // it to the ground dragged the corners that do touch. The lowest corner is always locked.
                                let lowest = (0..<4).map { j in (f..<e).reduce(0.0) { $0 + Double(points[$1 * 8 + foot * 4 + j].z) } / Double(e - f) }.min()!
                                guard mean.z < lowest + o.lockHeight else { f = e; continue }
                                // The vertical pull to the ground ramps in and out over `lockRamp` at the ends of the
                                // run (heel strike, toe-off): at full weight it slammed the foot down while the person's
                                // heel was still above the ground, jolting the whole body. The horizontal lock is full
                                // from the first frame (ramping it let the foot slide).
                                let ramp = max(1.0, o.lockRamp * clip.fps)
                                for g in f..<e {
                                    let edge = Double(min(g - f + 1, e - g))
                                    anchors[g * 8 + k] = SIMD4(Float(mean.x), Float(mean.y), Float(min(1, edge / ramp)), Float(o.lock))
                                }
                                f = e
                            }
                        }
                    }
                }
                patch.withUnsafeBufferPointer { patchBuffer in
                    let patches = patchBuffer.baseAddress! + first * P
                    for f in 0..<F {
                        // w = link index + 1 (0: inactive); the slots left over stay zero.
                        var count = 0
                        for k in 0..<8 where anchors[f * 8 + k].w > 0 {
                            let a = anchors[f * 8 + k]
                            contacts[f * P + count] = SIMD4(a.x, a.y, 0, Float(soleLinks[k / 4] + 1))
                            count += 1
                        }
                        for slot in 0..<P where count < P {
                            let p = patches[f * P + slot]
                            if p.w == 0 { break }
                            contacts[f * P + count] = p
                            count += 1
                        }
                    }
                }
            }
        }}
        return (anchors, contacts)
    }
}
