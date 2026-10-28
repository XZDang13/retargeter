// Fixed-lag smoother for live streams (DESIGN §7, M6 phase B), on the offline kernel. Each tick appends the newest
// frame of every stream, runs `sweeps` symmetric Gauss-Seidel sweeps (one group per stream, one dispatch per tick:
// `sweepWindow`) of the offline costs — tracking, smoothness, stance lock, ground contact / clearance,
// self-collision, centroidal dynamics — over the frames not yet emitted (the last `lag` seconds), and emits the
// frames that leave the window. Emitted frames are never revised: the frames before the window are fixed context
// for the smoothness and dynamics stencils, so the output stays consistent with what a robot has already executed.
// Each measured on the comparison clips (DESIGN §7 "As built (M6 phase B)"):
//
// - A new frame starts from the previous frame's solution moved by the realtime solve's change since its previous
//   frame. Started at the realtime solution itself (which drifts from the window's: it locks no feet), it disagreed
//   with its neighbours by more than a few sweeps undo (jerk above lag 0 on walk and sweep clips).
// - A prior pulls every frame to its realtime solve (`fixed_lag.reference`, `reference_root`). Without it, errors
//   along directions nothing else holds accumulated from frame to frame through the starting point: the self-
//   collision push ratcheted an arm 0.45 rad away along its null space, and the dynamics term let the robot sink.
// - The dynamics term at a third of the offline weight (`fixed_lag.dynamics`, 1): with a 40 ms horizon the offline
//   weight (3) diverged on long clips and 2 already cost tracking. Its contact points are the current solution's
//   locked sole corners and contact patches, as offline.
// - Contacts: flags decided with the offline test a few frames late (LaggedContactEstimator) where they are known,
//   the last decided ones for the newest frames. The causal tracker's later flags left planted feet unlocked (foot
//   skate excess 0.039 against 0.019; 0.30 against 0.009 on a 3-minute dance clip).
// - A stance run's anchor is the mean position of the sole point in the realtime solve over the run's frames received
//   so far (anchors from the window's own poses fed back its drift: 0.64 m on the dance clip), fixed once the run's
//   first frame is emitted; contact and lock ramps only ramp out at run ends that have been seen.

import Compute
import Core
import Kinematics

public struct FixedLagFrame: Sendable {
    public var slot: Int
    /// Index of the frame within its stream (0 = the stream's first frame).
    public var frame: Int
    public var root: Transform
    public var q: [Double]
    /// Weighted tracking cost of the pose before the frame's last update.
    public var cost: Double
}

public final class FixedLagSolver {
    let offline: OfflineSolver
    public let lag: Double
    public let sweeps: Int
    public let slots: Int
    /// Window solves per second of stream time (nil: one per frame received). Frames arriving between solves are
    /// appended and solved by the next one, which emits every frame that is `lag` old by then.
    public let tickRate: Double?
    /// History kept per stream for the stance-run means (seconds); longer runs average over the last part.
    let historySeconds = 10.0
    var streams: [Stream?]
    let scratch: [String: ScratchBuffer]

    struct Stream {
        var fps: Double
        var lagFrames: Int
        var base = 0       // stream frame index of the first kept frame
        var count = 0      // frames received
        var emitted = 0    // frames emitted (the window is emitted..<count)
        // Per kept frame, in device layout.
        var goalPos: [SIMD4<Float>] = [], goalRot: [SIMD4<Float>] = []
        var foot: [[Bool]] = [], link: [[Bool]] = []
        var rootPos: [SIMD4<Float>] = [], rootRot: [SIMD4<Float>] = [], q: [Float] = []
        var linkCom: [SIMD4<Float>] = []   // world link CoMs (numLinks per frame): dynamics and hand acceleration rows
        var contactPoint: [SIMD4<Float>] = [], lambda: [Float] = []   // dynamics contacts and their force coefficients
        var cost: [Float] = []
        /// The realtime solve per frame: the prior's reference pose, and its sole points (8) the stance anchors.
        var realtimeRootPos: [SIMD4<Float>] = [], realtimeRootRot: [SIMD4<Float>] = [], realtimeQ: [Float] = []
        var realtimeSole: [SIMD4<Float>] = []
        /// Per sole point: the anchor of its current stance run, fixed when the run's first frame was emitted
        /// (start frame, ground position, locked or not). Frames already emitted were solved against it.
        var runAnchor: [(start: Int, x: Float, y: Float, locked: Bool)?] = Array(repeating: nil, count: 8)
        /// The realtime solve of the previous frame (its change to the new frame starts the new frame).
        var lastRealtime: (root: Transform, q: [Double])?
    }

    public init(offline: OfflineSolver, slots: Int, lag: Double, sweeps: Int, tickRate: Double? = nil) {
        self.offline = offline
        self.slots = slots
        self.lag = lag
        self.sweeps = sweeps
        self.tickRate = tickRate
        streams = [Stream?](repeating: nil, count: slots)
        scratch = Dictionary(uniqueKeysWithValues: [
            "windowFrames", "frameRange", "trajRootPos", "trajRootRot", "trajQ", "outCostOffline", "goalPos", "goalRot",
            "soleAnchor", "linkGroundContact", "frameFps", "trajLinkCom", "contactPoint", "contactLambda", "features",
            "referenceRootPos", "referenceRootRot", "referenceQ",
        ].map { ($0, ScratchBuffer(device: offline.device)) })
    }

    var setup: RetargetSetup { offline.setup }

    /// Starts (or restarts) `slot`; its frames are emitted `lag` seconds after they arrive.
    public func open(_ slot: Int, fps: Double) {
        streams[slot] = Stream(fps: fps, lagFrames: max(0, Int((lag * fps).rounded())))
    }

    public func close(_ slot: Int) { streams[slot] = nil }

    /// Frames of `slot` received but not yet emitted, at their current values (the end of a clip).
    public func flush(_ slot: Int) -> [FixedLagFrame] {
        guard let s = streams[slot] else { return [] }
        return (s.emitted..<s.count).map { frame(slot, s, $0) }
    }

    func frame(_ slot: Int, _ s: Stream, _ f: Int) -> FixedLagFrame {
        let i = f - s.base, n = setup.robot.dofCount
        return FixedLagFrame(slot: slot, frame: f, root: pose(s, f).root, q: (0..<n).map { Double(s.q[i * n + $0]) },
                             cost: Double(s.cost[i]))
    }

    func pose(_ s: Stream, _ f: Int) -> (root: Transform, world: [Transform]) {
        let i = f - s.base, robot = setup.robot, n = robot.dofCount
        let root = Transform(rotation: Quat(device: s.rootRot[i]).normalized, translation: Vec3(device: s.rootPos[i]))
        return (root, robot.forwardKinematics(root: root, q: (0..<n).map { Double(s.q[i * n + $0]) }))
    }

    /// The newest frame of each listed slot (goals, human heel / toe contacts [left heel, left toe, right heel,
    /// right toe] and mapped link contacts, provisional; its realtime solve) and, in `decided`, the contact flags just
    /// decided for an earlier frame (LaggedContactEstimator), replacing its provisional ones. The windows of the
    /// slots whose stream time crosses a tick are solved; returns the frames that leave them, oldest first per slot.
    public func step(_ inputs: [(slot: Int, goals: [Transform], foot: [Bool], link: [Bool], root: Transform, q: [Double],
                                 decided: (frame: Int, foot: [Bool], link: [Bool])?)]) throws
        -> [FixedLagFrame] {
        let robot = setup.robot, o = setup.mapping.offline, w = setup.mapping.fixedLag
        let n = robot.dofCount, L = robot.links.count, T = setup.mapping.targets.count, P = OfflineSolver.contactPoints
        let limits = setup.mapping.solverLimits(robot)
        for input in inputs {
            guard var s = streams[input.slot] else { throw ComputeError(message: "fixed-lag stream \(input.slot) is not open") }
            precondition(input.goals.count == T && input.foot.count == 4 && input.link.count == L)
            s.goalPos += input.goals.map(\.translation.device)
            s.goalRot += input.goals.map(\.rotation.device)
            s.foot.append(input.foot)
            s.link.append(input.link)
            var root = input.root, q = input.q
            if let last = s.lastRealtime, s.count > s.base {
                let i = s.count - 1 - s.base
                let previous = Transform(rotation: Quat(device: s.rootRot[i]).normalized, translation: Vec3(device: s.rootPos[i]))
                root = Transform(rotation: (input.root.rotation * last.root.rotation.conjugate * previous.rotation).normalized,
                                 translation: previous.translation + input.root.translation - last.root.translation)
                q = (0..<n).map { j in min(max(Double(s.q[i * n + j]) + input.q[j] - last.q[j], limits[j].lower), limits[j].upper) }
            }
            s.lastRealtime = (input.root, input.q)
            s.realtimeRootPos.append(input.root.translation.device)
            s.realtimeRootRot.append(input.root.rotation.device)
            s.realtimeQ += input.q.map(Float.init)
            let realtimeWorld = robot.forwardKinematics(root: input.root, q: input.q)
            s.realtimeSole += offline.geometry.soles.flatMap { foot in foot.points.map { realtimeWorld[foot.link].apply($0).device } }
            s.rootPos.append(root.translation.device)
            s.rootRot.append(root.rotation.device)
            s.q += q.map(Float.init)
            let world = robot.forwardKinematics(root: root, q: q)
            s.linkCom += robot.links.indices.map { world[$0].apply(robot.links[$0].inertial?.centerOfMass ?? .zero).device }
            s.contactPoint += [SIMD4<Float>](repeating: .zero, count: P)
            s.lambda += [Float](repeating: 0, count: P * 4)
            s.cost.append(0)
            s.count += 1
            if let d = input.decided, d.frame >= max(s.base, s.emitted), d.frame < s.count {
                s.foot[d.frame - s.base] = d.foot
                s.link[d.frame - s.base] = d.link
            }
            streams[input.slot] = s
        }

        // The window of every ticking stream: context (emitted frames) then the frames being solved.
        struct Window { var slot: Int; var first: Int; var solved: Int; var end: Int; var offset: Int }
        var windows: [Window] = []
        let ticking = inputs.map(\.slot).filter { slot in
            let s = streams[slot]!, t = Double(s.count - 1)
            guard let rate = tickRate else { return true }
            return ((t + 1) * rate / s.fps).rounded(.down) > (t * rate / s.fps).rounded(.down) || s.count == 1
        }
        var frameRange: [SIMD2<Int32>] = [], rootPos: [SIMD4<Float>] = [], rootRot: [SIMD4<Float>] = [], qs: [Float] = []
        var goalPos: [SIMD4<Float>] = [], goalRot: [SIMD4<Float>] = [], ground: [Float] = [], fps: [Float] = []
        var anchors: [SIMD4<Float>] = [], linkCom: [SIMD4<Float>] = [], contacts: [SIMD4<Float>] = [], lambda: [Float] = []
        var referenceRootPos: [SIMD4<Float>] = [], referenceRootRot: [SIMD4<Float>] = [], referenceQ: [Float] = []
        var stride = 0
        for slot in ticking {
            var s = streams[slot]!
            let K = max(1, Int((o.dynamicsStride * s.fps).rounded()))
            if stride == 0 { stride = K }
            precondition(K == stride, "fixed-lag streams solved together need the same frame rate")
            // Context: the dynamics stencil reaches t ± 2K (the smoothness rows t ± 2).
            let end = s.count, solved = max(s.base, s.emitted), first = max(s.base, solved - (2 * K + 2))
            let solvedAnchors = stanceAnchors(s, frames: solved..<end)
            for (row, f) in (solved..<end).enumerated() {
                // Dynamics contacts, as offline: the current solution's locked sole corners, then its contact patches.
                let i = f - s.base, link = s.link[i]
                let features = OfflineSolver.frameFeatures(robot: robot, geometry: offline.geometry, world: pose(s, f).world,
                                                           inContact: { link[$0] })
                var points: [SIMD4<Float>] = []
                for k in 0..<8 where solvedAnchors[row * 8 + k].w > 0 {
                    points.append(SIMD4(features.sole[k].x, features.sole[k].y, 0, Float(offline.geometry.soles[k / 4].link + 1)))
                }
                for p in features.patch where p.w > 0 && points.count < P { points.append(p) }
                s.contactPoint.replaceSubrange((i * P)..<((i + 1) * P), with: points + [SIMD4<Float>](repeating: .zero, count: P - points.count))
            }
            streams[slot] = s
            let offset = rootPos.count
            windows.append(Window(slot: slot, first: first, solved: solved, end: end, offset: offset))
            let lo = first - s.base, hi = end - s.base
            frameRange += [SIMD2<Int32>](repeating: SIMD2(Int32(offset), Int32(offset + hi - lo)), count: hi - lo)
            rootPos += s.rootPos[lo..<hi]
            rootRot += s.rootRot[lo..<hi]
            qs += s.q[(lo * n)..<(hi * n)]
            linkCom += s.linkCom[(lo * L)..<(hi * L)]
            contacts += s.contactPoint[(lo * P)..<(hi * P)]
            lambda += s.lambda[(lo * P * 4)..<(hi * P * 4)]
            referenceRootPos += s.realtimeRootPos[lo..<hi]
            referenceRootRot += s.realtimeRootRot[lo..<hi]
            referenceQ += s.realtimeQ[(lo * n)..<(hi * n)]
            goalPos += s.goalPos[(lo * T)..<(hi * T)]
            goalRot += s.goalRot[(lo * T)..<(hi * T)]
            ground += groundWeights(s, frames: first..<end).map { $0 * Float(o.contact) }
            fps += [Float](repeating: Float(s.fps), count: hi - lo)
            anchors += [SIMD4<Float>](repeating: .zero, count: (solved - first) * 8) + solvedAnchors
        }
        let frames = rootPos.count
        guard frames > 0 else { return [] }

        // One group per stream: its solved frames, oldest first.
        let windowFrames = windows.map { SIMD2(Int32($0.offset + $0.solved - $0.first), Int32($0.offset + $0.end - $0.first)) }
        func buf<V: BitwiseCopyable>(_ name: String, _ values: [V]) throws -> DeviceBuffer { try scratch[name]!.upload(values) }
        let rp = try buf("trajRootPos", rootPos), rr = try buf("trajRootRot", rootRot), q = try buf("trajQ", qs)
        let cost = try scratch["outCostOffline"]!.ensure(Float.self, count: frames)
        let com = try buf("trajLinkCom", linkCom), forces = try buf("contactLambda", lambda)
        let features = try scratch["features"]!.ensure(SIMD4<Float>.self, count: frames * P)   // not written (writeFeatures 0)
        var arguments = try offline.constantArguments(stride: stride, window: w.dynamics)
        arguments += [
            .buffer("frameRange", try buf("frameRange", frameRange)),
            .buffer("trajRootPos", rp), .buffer("trajRootRot", rr), .buffer("trajQ", q), .buffer("outCostOffline", cost),
            .buffer("goalPos", try buf("goalPos", goalPos)), .buffer("goalRot", try buf("goalRot", goalRot)),
            .buffer("linkGroundContact", try buf("linkGroundContact", ground)), .buffer("frameFps", try buf("frameFps", fps)),
            .buffer("soleAnchor", try buf("soleAnchor", anchors)), .buffer("trajLinkCom", com),
            .buffer("contactLambda", forces), .buffer("contactPoint", try buf("contactPoint", contacts)),
            .buffer("trajSole", features), .buffer("trajPatch", features),
            .value("constraintScale", Float(1)), .value("writeFeatures", UInt32(0)),
            .value("referenceWeight", Float(w.reference)), .value("referenceRootWeight", Float(w.referenceRoot)),
            .buffer("referenceRootPos", try buf("referenceRootPos", referenceRootPos)),
            .buffer("referenceRootRot", try buf("referenceRootRot", referenceRootRot)),
            .buffer("referenceQ", try buf("referenceQ", referenceQ)),
            .buffer("windowFrames", try buf("windowFrames", windowFrames)), .value("numWindows", UInt32(windows.count)),
            .value("windowSweeps", UInt32(sweeps)),
        ]
        try offline.device.dispatch(offline.windowKernel, threads: windows.count * OfflineSolver.groupSize, arguments: arguments)
        let rpOut = try rp.download(as: SIMD4<Float>.self, count: frames)
        let rrOut = try rr.download(as: SIMD4<Float>.self, count: frames)
        let qOut = try q.download(as: Float.self, count: frames * n)
        let costOut = try cost.download(as: Float.self, count: frames)
        let comOut = try com.download(as: SIMD4<Float>.self, count: frames * L)
        let lambdaOut = try forces.download(as: Float.self, count: frames * P * 4)

        var emitted: [FixedLagFrame] = []
        for window in windows {
            var s = streams[window.slot]!
            for f in window.solved..<window.end {
                let i = f - s.base, g = window.offset + f - window.first
                s.rootPos[i] = rpOut[g]
                s.rootRot[i] = rrOut[g]
                s.q.replaceSubrange((i * n)..<((i + 1) * n), with: qOut[(g * n)..<((g + 1) * n)])
                s.linkCom.replaceSubrange((i * L)..<((i + 1) * L), with: comOut[(g * L)..<((g + 1) * L)])
                s.lambda.replaceSubrange((i * P * 4)..<((i + 1) * P * 4), with: lambdaOut[(g * P * 4)..<((g + 1) * P * 4)])
                s.cost[i] = costOut[g]
            }
            // Frames leave the window once `lag` seconds of newer frames exist.
            for out in s.emitted..<max(s.emitted, s.count - s.lagFrames) {
                // A stance run whose first frame leaves now keeps the anchor it was solved with from here on.
                for k in 0..<8 {
                    let flag = soleFlag(k)
                    guard s.foot[out - s.base][flag], out == s.base || !s.foot[out - 1 - s.base][flag] else { continue }
                    s.runAnchor[k] = runAnchor(s, k, run(s.count, base: s.base, out) { s.foot[$0 - s.base][flag] })
                }
                emitted.append(frame(window.slot, s, out))
                s.emitted = out + 1
            }
            trim(&s)
            streams[window.slot] = s
        }
        return emitted
    }

    /// The human contact flag (index into `foot`) of sole point k: its foot's heel or toe.
    func soleFlag(_ k: Int) -> Int { (k / 4) * 2 + (offline.soleHeel[k / 4][k % 4] ? 0 : 1) }

    /// Run around frame f of a contact flag: its first frame and end (exclusive), and whether it is still open
    /// (reaches the newest frame, so its end is not known yet).
    func run(_ count: Int, base: Int, _ f: Int, _ on: (Int) -> Bool) -> (start: Int, end: Int, open: Bool) {
        var start = f, end = f
        while start - 1 >= base && on(start - 1) { start -= 1 }
        while end < count && on(end) { end += 1 }
        return (start, end, end == count)
    }

    /// A stance run's anchor from the frames of it received so far: the mean ground position of sole point k in the
    /// realtime solve, and whether to lock it (its mean height within `lockHeight` of the foot's lowest corner).
    func runAnchor(_ s: Stream, _ k: Int, _ r: (start: Int, end: Int, open: Bool)) -> (start: Int, x: Float, y: Float, locked: Bool) {
        let foot = k / 4
        var mean = Vec3.zero
        for g in r.start..<r.end { mean += Vec3(device: s.realtimeSole[(g - s.base) * 8 + k]) }
        mean /= Double(r.end - r.start)
        let lowest = (0..<4).map { j in
            (r.start..<r.end).reduce(0.0) { $0 + Double(s.realtimeSole[($1 - s.base) * 8 + foot * 4 + j].z) } / Double(r.end - r.start)
        }.min()!
        return (r.start, Float(mean.x), Float(mean.y), mean.z < lowest + setup.mapping.offline.lockHeight)
    }

    /// Stance anchors (8 per frame) for `frames`, as `OfflineSolver.stanceAnchors` with the causal run anchors.
    func stanceAnchors(_ s: Stream, frames: Range<Int>) -> [SIMD4<Float>] {
        let o = setup.mapping.offline
        var anchors = [SIMD4<Float>](repeating: .zero, count: frames.count * 8)
        for k in 0..<8 {
            let flag = soleFlag(k)
            func on(_ f: Int) -> Bool { s.foot[f - s.base][flag] }
            var f = frames.lowerBound
            while f < frames.upperBound {
                guard on(f) else { f += 1; continue }
                let r = run(s.count, base: s.base, f, on)
                let stop = min(r.end, frames.upperBound)
                let anchor = s.runAnchor[k].flatMap { $0.start == r.start ? $0 : nil } ?? runAnchor(s, k, r)
                if anchor.locked {
                    // The vertical pull ramps in and out over `lockRamp` at the run's ends (as offline).
                    let ramp = max(1.0, o.lockRamp * s.fps)
                    for g in f..<stop {
                        let edge = Double(r.open ? g - r.start + 1 : min(g - r.start + 1, r.end - g))
                        anchors[(g - frames.lowerBound) * 8 + k] = SIMD4(anchor.x, anchor.y, Float(min(1, edge / ramp)), Float(o.lock))
                    }
                }
                f = stop
            }
        }
        return anchors
    }

    /// Ground-contact weights (per frame and link) for `frames`: `OfflineSolver.rampedContact` with ramps only at
    /// run ends that have been seen.
    func groundWeights(_ s: Stream, frames: Range<Int>) -> [Float] {
        let L = setup.robot.links.count, ramp = setup.mapping.offline.contactRamp * s.fps
        var out = [Float](repeating: 0, count: frames.count * L)
        for (row, f) in frames.enumerated() {
            for l in 0..<L where s.link[f - s.base][l] {
                guard ramp > 1 else { out[row * L + l] = 1; continue }
                let r = run(s.count, base: s.base, f) { s.link[$0 - s.base][l] }
                let edge = Double(r.open ? f - r.start + 1 : min(f - r.start + 1, r.end - f))
                out[row * L + l] = Float(min(1, edge / ramp))
            }
        }
        return out
    }

    /// Drops frames older than the history (in one-second blocks).
    func trim(_ s: inout Stream) {
        let keep = Int(historySeconds * s.fps) + (s.count - s.emitted) + 16
        let block = Int(s.fps)
        guard s.count - s.base > keep + block else { return }
        let n = setup.robot.dofCount, T = setup.mapping.targets.count, L = setup.robot.links.count, P = OfflineSolver.contactPoints
        s.goalPos.removeFirst(block * T); s.goalRot.removeFirst(block * T)
        s.foot.removeFirst(block); s.link.removeFirst(block)
        s.rootPos.removeFirst(block); s.rootRot.removeFirst(block); s.q.removeFirst(block * n)
        s.linkCom.removeFirst(block * L); s.contactPoint.removeFirst(block * P); s.lambda.removeFirst(block * P * 4)
        s.cost.removeFirst(block)
        s.realtimeRootPos.removeFirst(block); s.realtimeRootRot.removeFirst(block); s.realtimeQ.removeFirst(block * n)
        s.realtimeSole.removeFirst(block * 8)
        s.base += block
    }
}

/// Clips run through the fixed-lag solver as live streams would be (one stream per clip, frames in lockstep), for
/// evaluating the lag offline: same causal CPU stage, lagged contact flags and realtime solve as LiveRetargeter.
public final class LaggedRetargeter {
    public let base: HumanRetargeter
    public let offline: OfflineSolver
    public let lag: Double
    public let sweeps: Int
    public let tickRate: Double?
    /// Wall-clock time of the window solves in the last `solve` and the number of ticks.
    public private(set) var lastWindowTime: Duration = .zero
    public private(set) var lastTicks = 0

    public init(device: ComputeDevice, robot: RobotModel, mapping: RetargetMapping, surface: BodySurface,
                skeleton: HumanSkeleton, geometry: RobotGeometry, lag: Double, sweeps: Int, tickRate: Double? = nil) throws {
        base = try HumanRetargeter(device: device, robot: robot, mapping: mapping, surface: surface, skeleton: skeleton)
        offline = try OfflineSolver(device: device, setup: base.setup, geometry: geometry)
        self.lag = lag
        self.sweeps = sweeps
        self.tickRate = tickRate
    }

    public var setup: RetargetSetup { base.setup }

    public func prepare(_ motions: [HumanMotion]) throws -> [PreparedClip] { try base.prepare(motions, laggedContacts: true) }

    public func solve(_ clips: [PreparedClip]) throws -> [RetargetResult] {
        guard !clips.isEmpty else { return [] }
        let realtime = try base.solve(clips)
        let solver = FixedLagSolver(offline: offline, slots: clips.count, lag: lag, sweeps: sweeps, tickRate: tickRate)
        for (c, clip) in clips.enumerated() { solver.open(c, fps: clip.fps) }
        var out = [[FixedLagFrame]](repeating: [], count: clips.count)
        let ticks = clips.map(\.goals.count).max() ?? 0
        let start = ContinuousClock.now
        for t in 0..<ticks {
            let inputs = clips.indices.filter { t < clips[$0].goals.count }.map { c in
                let clip = clips[c], G = clip.causal.groupNames.count
                // As LiveRetargeter: the flags decided `delay` frames back; the new frame starts with them (the
                // causal ones before the first is decided).
                let delay = clip.goals.count - clip.lagged.count
                let decided = t >= delay ? base.robotContacts(clip.lagged[t - delay]) : nil
                let provisional = decided ?? base.robotContacts((0..<G).map { clip.causal.contact(t, $0) })
                return (slot: c, goals: clip.goals[t], foot: provisional.foot, link: provisional.link,
                        root: realtime[c].trajectory.root[t], q: realtime[c].trajectory.q[t],
                        decided: decided.map { (frame: t - delay, foot: $0.foot, link: $0.link) })
            }
            for frame in try solver.step(inputs) { out[frame.slot].append(frame) }
            for c in clips.indices where t == clips[c].goals.count - 1 { out[c] += solver.flush(c) }
        }
        lastWindowTime = ContinuousClock.now - start
        lastTicks = ticks
        return clips.indices.map { c in
            precondition(out[c].map(\.frame) == Array(0..<clips[c].goals.count), "every frame emitted once, in order")
            let trajectory = RobotTrajectory(fps: clips[c].fps, root: out[c].map(\.root), q: out[c].map(\.q), cost: out[c].map(\.cost))
            return RetargetResult(trajectory: trajectory, contacts: clips[c].causal, prepared: clips[c])
        }
    }
}
