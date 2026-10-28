// Batched causal IK on a compute device (`Kernels/retarget_ik.slang`).

import Compute
import Core
import Kinematics

/// A solved robot trajectory at the source frame rate.
public struct RobotTrajectory: Sendable {
    public var fps: Double
    public var root: [Transform]
    public var q: [[Double]]
    /// Final weighted tracking cost per frame (sum of squared residuals).
    public var cost: [Double]

    public var frameCount: Int { root.count }

    public init(fps: Double, root: [Transform], q: [[Double]], cost: [Double]) {
        self.fps = fps
        self.root = root
        self.q = q
        self.cost = cost
    }
}

/// Human contact inputs of one clip for the realtime dynamics term: per frame, the 8 sole corners (left foot
/// then right, heel corners first; true while the human heel / toe above them touches) and per robot link a
/// mapped body-part contact (feet excluded).
public struct RealtimeContacts: Sendable {
    public var sole: [[Bool]]
    public var link: [[Bool]]

    public init(sole: [[Bool]], link: [[Bool]]) {
        self.sole = sole
        self.link = link
    }
}

public final class RetargetSolver {
    public let setup: RetargetSetup
    let device: ComputeDevice
    let kernel: Kernel
    let cooperative: Bool
    var model: [String: DeviceBuffer]
    let scratch: [String: ScratchBuffer]
    /// Per foot: link and the bottoms of its four collision spheres (link frame), heel corners first.
    public let soles: [(link: Int, points: [Vec3])]

    public init(device: ComputeDevice, setup: RetargetSetup) throws {
        self.device = device
        self.setup = setup
        // GPU: one cooperating 32-thread group per stream; CPU: one thread per stream (no group barriers).
        cooperative = device.backend == .cuda
        kernel = try device.loadKernel(module: cooperative ? "retarget_ik_group" : "retarget_ik",
                                       entryPoint: cooperative ? "solveGroup" : "solve")
        let robot = setup.robot
        guard robot.links.count <= 40, robot.dofCount <= 32, setup.chains.allSatisfy({ $0.count <= 16 }),
              setup.mapping.targets.count <= 24 else {
            throw ComputeError(message: "exceeds retarget_ik limits (40 links, 32 dofs, chains of 16, 24 targets)")
        }
        let links = robot.links
        let limits = robot.links.compactMap { link in link.dofIndex.map { ($0, link.limits!) } }.sorted { $0.0 < $1.0 }
        var chainStart: [Int32] = [], chainCount: [Int32] = [], chainLinks: [Int32] = []
        for chain in setup.chains {
            chainStart.append(Int32(chainLinks.count))
            chainCount.append(Int32(chain.count))
            chainLinks += chain.map { Int32($0) }
        }
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
            "chainStart": try device.makeBuffer(chainStart),
            "chainCount": try device.makeBuffer(chainCount),
            "chainLinks": try device.makeBuffer(chainLinks.isEmpty ? [0] : chainLinks),
        ]
        var soles: [(link: Int, points: [Vec3])] = []
        for side in ["left", "right"] {
            guard let foot = robot.linkIndex(named: "\(side)_ankle_roll_link") else { continue }
            let points = robot.links[foot].collisions.compactMap { c -> Vec3? in
                guard case .sphere(let r) = c.shape else { return nil }
                return c.origin.translation - Vec3(0, 0, r)
            }.sorted { $0.x < $1.x }
            if points.count == 4 { soles.append((foot, points)) }
        }
        self.soles = soles
        if cooperative {   // dynamics inputs exist only in the cooperative kernel
            model["linkCom"] = try device.makeBuffer(links.map { l -> SIMD4<Float> in
                let c = l.inertial?.centerOfMass ?? .zero
                return SIMD4(Float(c.x), Float(c.y), Float(c.z), Float(l.inertial?.mass ?? 0))
            })
            model["linkAncestors"] = try device.makeBuffer(links.indices.map { l -> SIMD2<UInt32> in
                var mask = SIMD2<UInt32>(0, 0)
                var i: Int? = l
                while let k = i {
                    if k < 32 { mask.x |= 1 << UInt32(k) } else { mask.y |= 1 << UInt32(k - 32) }
                    i = links[k].parent
                }
                return mask
            })
            // Arm membership (branch switch): links of the shoulder / elbow / wrist chains, by side.
            func arm(_ link: Int) -> Int32 {
                let name = links[link].name
                guard ["shoulder", "elbow", "wrist"].contains(where: name.contains) else { return 0 }
                return name.hasPrefix("left_") ? 1 : name.hasPrefix("right_") ? 2 : 0
            }
            model["dofArm"] = try device.makeBuffer((0..<robot.dofCount).map { d in arm(links.firstIndex { $0.dofIndex == d }!) })
            model["targetArm"] = try device.makeBuffer(setup.robotLink.map(arm))
            model["soleLink"] = try device.makeBuffer(soles.count == 2 ? soles.map { Int32($0.link) } : [0, 0])
            model["solePoint"] = try device.makeBuffer(soles.count == 2 ? soles.flatMap { $0.points.map { $0.device } }
                                                                       : [SIMD4<Float>](repeating: .zero, count: 8))
        }
        scratch = Dictionary(uniqueKeysWithValues: [
            "targetWeight", "streamFrameOffset", "streamFrameCount", "goalPos", "goalRot", "stateRootPos", "stateRootRot",
            "stateQ", "stateValid", "outRootPos", "outRootRot", "outQ", "outCost",
            "soleContact", "linkContactRT", "streamFps", "stateLinkCom", "stateHistory", "stateBranchQ", "stateBranch",
        ].map { ($0, ScratchBuffer(device: device)) })
    }

    /// Solves every clip from a cold start. `goals[clip][frame][target]`; `weights` overrides the mapping
    /// weights (pos, rot) per target, e.g. for position-only solves.
    /// `contacts` (per clip) enables the causal dynamics term (GPU variant only).
    public func solve(goals: [[[Transform]]], fps: [Double], weights: [SIMD2<Float>]? = nil,
                      iterations: Int? = nil, contacts: [RealtimeContacts]? = nil) throws -> [RobotTrajectory] {
        let streams = goals.count, n = setup.robot.dofCount, L = setup.robot.links.count
        func zeros<V: BitwiseCopyable>(_ name: String, _ value: V, _ count: Int) throws -> DeviceBuffer {
            try scratch[name]!.upload([V](repeating: value, count: max(1, count)))
        }
        let state = StreamState(
            rootPos: try zeros("stateRootPos", SIMD4<Float>.zero, streams),
            rootRot: try zeros("stateRootRot", SIMD4<Float>(0, 0, 0, 1), streams),
            q: try zeros("stateQ", Float(0), streams * n),
            linkCom: try scratch["stateLinkCom"]!.ensure(SIMD4<Float>.self, count: max(1, streams * 16 * L)),
            history: try zeros("stateHistory", Int32(0), streams),
            branchQ: try scratch["stateBranchQ"]!.ensure(Float.self, count: max(1, streams * n)),
            branch: try zeros("stateBranch", Int32(0), 16 * streams))
        let solved = try run(goals: goals, fps: fps, valid: [Int32](repeating: 0, count: streams), state: state,
                             weights: weights, iterations: iterations, contacts: contacts)
        return goals.indices.map { c in
            RobotTrajectory(fps: fps[c], root: solved[c].map(\.root), q: solved[c].map(\.q), cost: solved[c].map(\.cost))
        }
    }

    /// Device state carried between calls per stream: root, joints, and the CoM history of the dynamics term.
    struct StreamState {
        var rootPos: DeviceBuffer
        var rootRot: DeviceBuffer
        var q: DeviceBuffer
        var linkCom: DeviceBuffer
        var history: DeviceBuffer
        /// Arm branch switch: the incoming branch's joints, and the blend progress per arm.
        var branchQ: DeviceBuffer
        var branch: DeviceBuffer
    }

    /// Device copies of `factorStructure` per (weights, dynamics), built on first use.
    var structures: [[Float]: DeviceBuffer] = [:]

    func structureBuffer(weights w: [SIMD2<Float>], dynamics: Bool) throws -> DeviceBuffer {
        let key: [Float] = w.flatMap { [$0.x, $0.y] } + [dynamics ? 1 : 0]
        if let buffer = structures[key] { return buffer }
        let buffer = try device.makeBuffer(factorStructure(weights: w, dynamics: dynamics))
        structures[key] = buffer
        return buffer
    }

    /// One dispatch: every stream solves its frames in order, starting from `state` where `valid[stream]` is set
    /// (else from a cold start) and leaving its last pose there.
    func run(goals: [[[Transform]]], fps: [Double], valid: [Int32], state: StreamState, weights: [SIMD2<Float>]?,
             iterations: Int?, contacts: [RealtimeContacts]?) throws -> [[(root: Transform, q: [Double], cost: Double)]] {
        let T = setup.mapping.targets.count, n = setup.robot.dofCount
        var offsets: [Int32] = [], counts: [Int32] = []
        var goalPos: [SIMD4<Float>] = [], goalRot: [SIMD4<Float>] = []
        for clip in goals {
            offsets.append(Int32(goalPos.count / T))
            counts.append(Int32(clip.count))
            for frame in clip {
                precondition(frame.count == T)
                for goal in frame {
                    goalPos.append(goal.translation.device)
                    goalRot.append(goal.rotation.device)
                }
            }
        }
        let streams = goals.count, frames = goalPos.count / T
        // Streams without frames (idle teleop slots) start at frame 0 and leave their state as it is.
        offsets = offsets.map { min($0, Int32(max(0, frames - 1))) }
        let w = weights ?? setup.mapping.targets.map { SIMD2(Float($0.positionWeight), Float($0.rotationWeight)) }
        let s = setup.mapping.solver
        func buf<V: BitwiseCopyable>(_ name: String, _ values: [V]) throws -> DeviceBuffer { try scratch[name]!.upload(values) }
        func out<V: BitwiseCopyable>(_ name: String, _ type: V.Type, _ count: Int) throws -> DeviceBuffer {
            try scratch[name]!.ensure(type, count: count)
        }
        let L = setup.robot.links.count
        var soleContact: [Float] = [], linkContact: [Float] = []
        if let contacts {
            for clip in contacts {
                for f in clip.sole.indices {
                    soleContact += clip.sole[f].map { $0 ? 1 : 0 }
                    linkContact += clip.link[f].map { $0 ? 1 : 0 }
                }
            }
            precondition(soleContact.count == frames * 8 && linkContact.count == frames * L, "contacts must cover every frame")
        }
        // Accelerations over frames t, t - K, t - 2K (K ≈ dynamics_stride s; the history ring holds 16 frames).
        let rate = fps.first ?? 120
        let K = max(1, Int((s.dynamicsStride * rate).rounded()))
        let dynamics = contacts != nil && soles.count == 2 && 2 * K < 16 ? s.dynamics : 0
        let rootPosOut = try out("outRootPos", SIMD4<Float>.self, max(1, frames))
        let rootRotOut = try out("outRootRot", SIMD4<Float>.self, max(1, frames))
        let qOut = try out("outQ", Float.self, max(1, frames * n))
        let costOut = try out("outCost", Float.self, max(1, frames))
        var arguments: [KernelArgument] = model.map { .buffer($0.key, $0.value) }
        arguments += [
            .buffer("targetWeight", try buf("targetWeight", w)),
            .buffer("streamFrameOffset", try buf("streamFrameOffset", offsets)),
            .buffer("streamFrameCount", try buf("streamFrameCount", counts)),
            .buffer("goalPos", try buf("goalPos", goalPos.isEmpty ? [.zero] : goalPos)),
            .buffer("goalRot", try buf("goalRot", goalRot.isEmpty ? [SIMD4(0, 0, 0, 1)] : goalRot)),
            .buffer("stateRootPos", state.rootPos), .buffer("stateRootRot", state.rootRot), .buffer("stateQ", state.q),
            .buffer("stateValid", try buf("stateValid", valid)),
            .buffer("outRootPos", rootPosOut), .buffer("outRootRot", rootRotOut), .buffer("outQ", qOut),
            .buffer("outCost", costOut),
            .value("numStreams", UInt32(streams)), .value("numLinks", UInt32(setup.robot.links.count)),
            .value("numDofs", UInt32(n)), .value("numTargets", UInt32(T)), .value("rootTarget", UInt32(setup.rootTarget)),
            .value("iterations", UInt32(iterations ?? s.iterations)), .value("coldIterations", UInt32(50)),
            .value("damping", Float(s.damping)),
            .value("smoothness", Float(s.smoothness)), .value("posture", Float(s.posture)),
            .value("limitMargin", Float(s.limitMargin)), .value("limitWeight", Float(s.limitWeight)),
        ]
        if cooperative {
            arguments += [
                .buffer("soleContact", try buf("soleContact", soleContact.isEmpty ? [Float(0)] : soleContact)),
                .buffer("linkContactRT", try buf("linkContactRT", linkContact.isEmpty ? [Float(0)] : linkContact)),
                .buffer("streamFps", try buf("streamFps", fps.map(Float.init))),
                .buffer("factorStructure", try structureBuffer(weights: w, dynamics: dynamics > 0)),
                .buffer("stateLinkCom", state.linkCom), .buffer("stateHistory", state.history),
                .value("dynamicsWeight", Float(dynamics)), .value("dynamicsStride", UInt32(K)),
                .value("friction", Float(s.friction)), .value("totalMass", Float(setup.robot.totalMass)),
                .value("contactIterations", UInt32(s.contactIterations)),
                .buffer("stateBranchQ", state.branchQ), .buffer("stateBranch", state.branch),
                .value("branchInterval", UInt32(max(1, Int((s.branchInterval * rate).rounded())))),
                .value("branchError", Float(s.branchError)), .value("branchIterations", UInt32(s.branchIterations)),
                .value("branchBlend", UInt32(s.branchBlend > 0 ? max(1, Int((s.branchBlend * rate).rounded())) : 0)),
                .value("branchHysteresis", Float(s.branchHysteresis)),
                .value("branchKeep", UInt32(max(1, Int((s.branchKeep * rate).rounded())))),
                .value("branchDwell", UInt32(max(1, Int((s.branchDwell * rate).rounded())))),
            ]
        }
        try device.dispatch(kernel, threads: cooperative ? streams * 32 : streams, arguments: arguments)
        guard frames > 0 else { return goals.map { _ in [] } }
        let rp = try rootPosOut.download(as: SIMD4<Float>.self, count: frames)
        let rr = try rootRotOut.download(as: SIMD4<Float>.self, count: frames)
        let qs = try qOut.download(as: Float.self, count: frames * n)
        let cost = try costOut.download(as: Float.self, count: frames)
        return goals.indices.map { c in
            (Int(offsets[c])..<(Int(offsets[c]) + Int(counts[c]))).map { f in
                (Transform(rotation: Quat(device: rr[f]).normalized, translation: Vec3(device: rp[f])),
                 (0..<n).map { Double(qs[f * n + $0]) }, Double(cost[f]))
            }
        }
    }

    /// The cooperative kernel's sparsity structure for target weights `w` (and the dynamics rows when `dynamics`),
    /// laid out as `factorStructure` in `Kernels/retarget_ik_group.slang` reads it: a 9-word header (pair count,
    /// offsets of pairs, column masks, elimination positions, factor column starts, columns, row starts, rows; the
    /// factor entry count), then the arrays. The same algorithm the kernel used to run on one lane.
    func factorStructure(weights w: [SIMD2<Float>], dynamics: Bool) -> [UInt32] {
        let n = setup.robot.dofCount, nv = 6 + n, T = setup.mapping.targets.count
        let dofOfLink = setup.robot.links.map { $0.dofIndex ?? -1 }
        var mask = [UInt64](repeating: 0, count: nv)
        for a in 0..<nv {
            for t in 0..<T {
                let touches = a < 6 || setup.chains[t].contains { dofOfLink[$0] == a - 6 }
                guard touches else { continue }
                if w[t].x != 0 { mask[a] |= 1 << UInt64(2 * t) }
                if w[t].y != 0 && a >= 3 { mask[a] |= 1 << UInt64(2 * t + 1) }   // rotation rows ignore the root position
            }
            if dynamics { mask[a] |= 3 << UInt64(2 * T) }
        }
        var pairs: [UInt32] = []
        for a in 0..<nv { for b in a..<nv where mask[a] & mask[b] != 0 { pairs.append(UInt32(a | (b << 8))) } }
        // Heaviest first (shared triples), stable, so the lanes' round-robin share of the pairs is balanced.
        func cost(_ p: UInt32) -> Int { (mask[Int(p & 0xff)] & mask[Int(p >> 8)]).nonzeroBitCount }
        for k in pairs.indices.dropFirst() {
            let pair = pairs[k], c = cost(pair)
            var at = k - 1
            while at >= 0 && cost(pairs[at]) < c { pairs[at + 1] = pairs[at]; at -= 1 }
            pairs[at + 1] = pair
        }
        // Minimum-degree ordering on the pair graph (lowest index on ties), with the fill each elimination adds.
        var adjacency = [UInt64](repeating: 0, count: nv), below = [UInt64](repeating: 0, count: nv)
        var inverse = [UInt32](repeating: 0, count: nv)
        for p in pairs {
            let a = Int(p & 0xff), b = Int(p >> 8)
            if a != b { adjacency[a] |= 1 << UInt64(b); adjacency[b] |= 1 << UInt64(a) }
        }
        var remaining: UInt64 = nv == 64 ? ~0 : (1 << UInt64(nv)) - 1
        for j in 0..<nv {
            var best = 0, bestDegree = Int.max
            for a in 0..<nv where (remaining >> UInt64(a)) & 1 != 0 {
                let degree = (adjacency[a] & remaining).nonzeroBitCount
                if degree < bestDegree { bestDegree = degree; best = a }
            }
            remaining &= ~(1 << UInt64(best))
            let neighbours = adjacency[best] & remaining
            below[j] = neighbours
            inverse[best] = UInt32(j)
            for a in 0..<nv where (neighbours >> UInt64(a)) & 1 != 0 { adjacency[a] |= neighbours & ~(1 << UInt64(a)) }
        }
        // Factor structure in elimination positions: column j has rows i > j where columnRows[j] bit i is set.
        let columnRows = (0..<nv).map { j -> UInt64 in
            (0..<nv).reduce(0) { rows, a in (below[j] >> UInt64(a)) & 1 != 0 ? rows | (1 << UInt64(inverse[a])) : rows }
        }
        var colStart: [UInt32] = [], cols: [UInt32] = [], rowStart: [UInt32] = [], rows: [UInt32] = []
        for j in 0..<nv {
            colStart.append(UInt32(cols.count))
            for i in (j + 1)..<max(j + 1, nv) where (columnRows[j] >> UInt64(i)) & 1 != 0 { cols.append(UInt32(i)) }
        }
        colStart.append(UInt32(cols.count))
        for i in 0..<nv {
            rowStart.append(UInt32(rows.count))
            for j in 0..<i where (columnRows[j] >> UInt64(i)) & 1 != 0 { rows.append(UInt32(j)) }
        }
        rowStart.append(UInt32(rows.count))
        let masks = mask.flatMap { [UInt32(truncatingIfNeeded: $0), UInt32(truncatingIfNeeded: $0 >> 32)] }
        var offset = 9
        var header: [UInt32] = [UInt32(pairs.count)]
        for block in [pairs, masks, inverse, colStart, cols, rowStart, rows] {
            header.append(UInt32(offset))
            offset += block.count
        }
        header.append(UInt32(cols.count))
        return header + pairs + masks + inverse + colStart + cols + rowStart + rows
    }

    /// Robot pose that matches the human reference (T-) pose: position-only solve from q = 0, used to derive
    /// the per-target rotation offsets.
    public func referencePose(human reference: [Transform]) throws -> (root: Transform, q: [Double]) {
        let goals = setup.goals(human: reference)
        let weights = setup.mapping.targets.map { t -> SIMD2<Float> in
            SIMD2(t.positionWeight > 0 ? Float(t.positionWeight) : 1, 0)
        }
        // Hold the goal for a while so the warm-started solve converges.
        let solved = try solve(goals: [Array(repeating: goals, count: 30)], fps: [30], weights: weights, iterations: 20)[0]
        return (solved.root.last!, solved.q.last!)
    }
}

extension RetargetSetup {
    /// Fills `rotationOffsets` so that in the reference pose every goal rotation equals the robot link rotation.
    public mutating func alignRotations(human reference: [Transform], robotRoot: Transform, robotQ: [Double]) {
        guard mapping.tposeOffsets else { return }
        let robotWorld = robot.forwardKinematics(root: robotRoot, q: robotQ)
        rotationOffsets = mapping.targets.indices.map { t in
            (reference[humanJoint[t]].rotation.conjugate * robotWorld[robotLink[t]].rotation).normalized
        }
    }
}
