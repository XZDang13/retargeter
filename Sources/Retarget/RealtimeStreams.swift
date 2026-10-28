// Live (teleop) retargeting: the realtime solver stepped one frame at a time per stream (DESIGN §7). Each tick
// takes the newest human frame of every stream that has one, runs the same CPU stage as the batch path (skeleton
// FK, body-surface skinning, causal ground / contact tracking, goals) and one IK dispatch for all of them, with the
// solver state of every stream kept on the device between ticks. At lag 0 a streamed clip reproduces the batch
// realtime solve exactly. On output, a safety layer limits each joint's step to its velocity limit relative to the
// last emitted pose and reports it in `flags`; realtime never rejects.

import Compute
import Contact
import Core

/// Device state of `slots` realtime streams, advanced one frame per `step`.
public final class RealtimeStreams {
    let solver: RetargetSolver
    public let slots: Int
    let state: RetargetSolver.StreamState
    var valid: [Int32]

    init(solver: RetargetSolver, slots: Int) throws {
        self.solver = solver
        self.slots = slots
        let device = solver.device, n = solver.setup.robot.dofCount, L = solver.setup.robot.links.count
        state = .init(rootPos: try device.makeBuffer([SIMD4<Float>](repeating: .zero, count: slots)),
                      rootRot: try device.makeBuffer([SIMD4<Float>](repeating: SIMD4(0, 0, 0, 1), count: slots)),
                      q: try device.makeBuffer([Float](repeating: 0, count: max(1, slots * n))),
                      linkCom: try device.makeBuffer(of: SIMD4<Float>.self, count: max(1, slots * 16 * L)),
                      history: try device.makeBuffer([Int32](repeating: 0, count: slots)),
                      branchQ: try device.makeBuffer([Float](repeating: 0, count: max(1, slots * n))),
                      branch: try device.makeBuffer([Int32](repeating: 0, count: 16 * slots)))
        valid = [Int32](repeating: 0, count: slots)
    }

    /// The next step of `slot` starts cold (reference-free, 50 iterations), as a new clip does.
    public func reset(_ slot: Int) { valid[slot] = 0 }

    /// Solves the listed frames in one dispatch, a slot's frames in the order listed (a slot may appear several
    /// times); the other slots keep their state. Results in the order listed.
    public func step(_ frames: [(slot: Int, goals: [Transform], fps: Double, sole: [Bool], link: [Bool])]) throws
        -> [(root: Transform, q: [Double], cost: Double)] {
        guard !frames.isEmpty else { return [] }
        var goals = [[[Transform]]](repeating: [], count: slots)
        var fps = [Double](repeating: frames[0].fps, count: slots)
        var contacts = [RealtimeContacts](repeating: RealtimeContacts(sole: [], link: []), count: slots)
        var index: [Int] = []   // each frame's position within its slot
        for f in frames {
            index.append(goals[f.slot].count)
            goals[f.slot].append(f.goals)
            fps[f.slot] = f.fps
            contacts[f.slot].sole.append(f.sole)
            contacts[f.slot].link.append(f.link)
        }
        let solved = try solver.run(goals: goals, fps: fps, valid: valid, state: state, weights: nil, iterations: nil,
                                    contacts: contacts)
        for f in frames { valid[f.slot] = 1 }
        return zip(frames, index).map { solved[$0.0.slot][$0.1] }
    }
}

/// One solved, safety-checked frame of a live stream.
public struct LiveRobotFrame: Sendable {
    public var slot: Int
    /// Index of the frame within its stream (0 = the first frame since the stream opened). With a lag, frames are
    /// emitted that much later than they arrive, in order.
    public var frame: Int
    public var root: Transform
    public var q: [Double]
    public var qVelocity: [Double]
    /// Raw solver output before the velocity clamp (what the batch solver returns).
    public var solvedQ: [Double]
    public var cost: Double
    public var contacts: FrameContacts
    public var flags: UInt32
}

public enum LiveFlags {
    public static let velocityClamped: UInt32 = 1
    public static let coldStart: UInt32 = 2
    public static let unsupported: UInt32 = 4
    public static let penetration: UInt32 = 8
    public static let environmentSupport: UInt32 = 16
    public static let flight: UInt32 = 32
}

public final class LiveRetargeter {
    public let retargeter: HumanRetargeter
    let streams: RealtimeStreams
    /// Fixed-lag window on top of the realtime solve (nil: lag 0, the realtime solve is the output).
    public let fixedLag: FixedLagSolver?
    /// Per joint: the largest step per second the output may take (velocity limit; infinite if unknown).
    let velocityLimit: [Double]

    struct Stream {
        var skeleton: HumanSkeleton
        var fps: Double
        var tracker: CausalContactTracker
        /// Contact flags with the offline test, decided a few frames late (fixed lag only).
        var estimator: LaggedContactEstimator?
        var received = 0
        /// Human-side contacts of the frames not yet emitted (fixed lag), oldest first.
        var pending: [FrameContacts] = []
        var last: (q: [Double], root: Transform)?
    }
    var active: [Stream?]

    public var slots: Int { streams.slots }
    public var lag: Double { fixedLag?.lag ?? 0 }

    /// `velocityLimits` per dof (rad/s; the robot config's, as the validator resolves them): the output safety
    /// layer's step bound. Without it, the model's (usually none: unbounded). `fixedLag`: a window solver with the
    /// same number of slots.
    public init(retargeter: HumanRetargeter, slots: Int, velocityLimits: [Double]? = nil, fixedLag: FixedLagSolver? = nil) throws {
        self.retargeter = retargeter
        self.fixedLag = fixedLag
        precondition(fixedLag.map { $0.slots == slots } ?? true, "fixed-lag solver slots differ")
        streams = try RealtimeStreams(solver: retargeter.solver, slots: slots)
        active = [Stream?](repeating: nil, count: slots)
        let robot = retargeter.robot
        velocityLimit = velocityLimits ?? (0..<robot.dofCount).map { d in
            robot.links.first { $0.dofIndex == d }?.limits?.velocity ?? .infinity
        }
        precondition(velocityLimit.count == robot.dofCount)
        // Warm-up: in a fresh server process the first step took ~150 ms (one-time first-use cost on the device;
        // not isolated further), which delayed a stream's first frames and the ~30 queued behind them.
        if let reference = retargeter.surface.referencePose {
            let goals = retargeter.setup.goals(human: reference)
            let solved = try streams.step([(0, goals, 30, [Bool](repeating: false, count: 8),
                                            [Bool](repeating: false, count: robot.links.count))])
            streams.reset(0)
            if let fixedLag {
                fixedLag.open(0, fps: 30)
                _ = try fixedLag.step([(0, goals, [false, false, false, false], [Bool](repeating: false, count: robot.links.count),
                                        solved[0].root, solved[0].q, nil)])
                fixedLag.close(0)
            }
        }
    }

    /// Starts (or restarts) `slot` with a human skeleton whose joints match the retargeter's by name and order.
    public func open(_ slot: Int, skeleton: HumanSkeleton, fps: Double) throws {
        guard slot >= 0 && slot < slots else { throw ComputeError(message: "stream slot \(slot) out of range 0..<\(slots)") }
        guard skeleton.jointNames == retargeter.setup.skeleton.jointNames else {
            throw ComputeError(message: "skeleton joints differ from the server's \(retargeter.setup.skeleton.name) skeleton")
        }
        let surface = retargeter.surface
        active[slot] = Stream(skeleton: skeleton, fps: fps,
                              tracker: CausalContactTracker(groups: retargeter.groups, fps: fps,
                                                            vertexCount: surface.vertexCount, excluded: surface.excluded),
                              estimator: fixedLag == nil ? nil : LaggedContactEstimator(groups: retargeter.groups, fps: fps,
                                                                                        vertexCount: surface.vertexCount,
                                                                                        excluded: surface.excluded))
        streams.reset(slot)
        fixedLag?.open(slot, fps: fps)
    }

    public func close(_ slot: Int) {
        guard slot >= 0 && slot < slots else { return }
        active[slot] = nil
        fixedLag?.close(slot)
    }

    public func isOpen(_ slot: Int) -> Bool { slot >= 0 && slot < slots && active[slot] != nil }

    /// Retargets the listed frames (local joint transforms), a stream's frames in the order listed (a stream may
    /// appear several times: its frames are solved in one realtime dispatch). Returns the frames emitted by this
    /// step: one per listed frame at lag 0; with a fixed lag, those that are `lag` seconds old at the window solves
    /// this step reaches.
    public func step(_ frames: [(slot: Int, local: [Transform])]) throws -> [LiveRobotFrame] {
        let retargeter = self.retargeter
        struct Prepared {
            var goals: [Transform]; var contacts: FrameContacts; var sole: [Bool]; var foot: [Bool]; var link: [Bool]
            var decided: [Bool]?
        }
        var prepared: [Prepared] = []
        for (slot, local) in frames {
            guard let stream = active[slot] else { throw ComputeError(message: "stream \(slot) is not open") }
            let world = HumanMotion(skeleton: stream.skeleton, fps: stream.fps, local: [local]).globalTransforms(frame: 0)
            let skinned = retargeter.surface.skin(frames: [world])
            let contacts = stream.tracker.update(skinned)
            let decided = stream.estimator?.update(skinned, ground: contacts.ground)
            let shift = Transform(translation: Vec3(0, 0, -Double(contacts.ground)))
            let groups = retargeter.groups.names.count
            func touching(_ g: Int) -> Bool { g < groups && contacts.contact[g] }
            let sole = (0..<8).map { k in touching(retargeter.footGroups[(k / 4) * 2 + (k % 4 < 2 ? 0 : 1)]) }
            var link = [Bool](repeating: false, count: retargeter.robot.links.count)
            for (g, l) in retargeter.contactLinks where touching(g) { link[l] = true }
            prepared.append(Prepared(goals: retargeter.setup.goals(human: world.map { shift * $0 }), contacts: contacts,
                                     sole: sole, foot: retargeter.footGroups.map(touching), link: link, decided: decided))
        }
        let solved = try streams.step(zip(frames, prepared).map {
            ($0.0.slot, $0.1.goals, active[$0.0.slot]!.fps, $0.1.sole, $0.1.link)
        })
        guard let fixedLag else {
            return try zip(frames, zip(prepared, solved)).map { input, pair in
                let index = active[input.slot]!.received
                active[input.slot]!.received += 1
                return try emit(input.slot, frame: index, root: pair.1.root, q: pair.1.q, cost: pair.1.cost, contacts: pair.0.contacts)
            }
        }
        // The window takes the frames one per slot at a time, so it solves at the same frames of stream time
        // however the frames were batched.
        var rounds: [[(slot: Int, goals: [Transform], foot: [Bool], link: [Bool], root: Transform, q: [Double],
                       decided: (frame: Int, foot: [Bool], link: [Bool])?)]] = []
        var seen = [Int](repeating: 0, count: slots)
        for (input, (p, s)) in zip(frames, zip(prepared, solved)) {
            let index = active[input.slot]!.received
            active[input.slot]!.received += 1
            active[input.slot]!.pending.append(p.contacts)
            if seen[input.slot] == rounds.count { rounds.append([]) }
            // Flags decided for the frame `delay` back; the new frame starts with them (the causal ones before).
            let decided = p.decided.map { retargeter.robotContacts($0) }
            let delay = active[input.slot]!.estimator?.delay ?? 0
            rounds[seen[input.slot]].append((input.slot, p.goals, decided?.foot ?? p.foot, decided?.link ?? p.link, s.root, s.q,
                                            decided.map { (frame: index - delay, foot: $0.foot, link: $0.link) }))
            seen[input.slot] += 1
        }
        return try rounds.flatMap { try fixedLag.step($0).map(emitLagged) }
    }

    /// The frames of `slot` still in the fixed-lag window, at their current values (when a stream ends).
    public func flush(_ slot: Int) throws -> [LiveRobotFrame] {
        guard let fixedLag, active[slot] != nil else { return [] }
        return try fixedLag.flush(slot).map(emitLagged)
    }

    func emitLagged(_ frame: FixedLagFrame) throws -> LiveRobotFrame {
        let contacts = active[frame.slot]!.pending.removeFirst()
        return try emit(frame.slot, frame: frame.frame, root: frame.root, q: frame.q, cost: frame.cost, contacts: contacts)
    }

    /// The safety layer on one solved frame: each joint's step limited to its velocity limit since the last
    /// emitted frame, the velocity as their difference, and the flags.
    func emit(_ slot: Int, frame: Int, root: Transform, q solved: [Double], cost: Double, contacts: FrameContacts) throws -> LiveRobotFrame {
        guard var stream = active[slot] else { throw ComputeError(message: "stream \(slot) is not open") }
        var flags: UInt32 = 0
        var q = solved, velocity = [Double](repeating: 0, count: q.count)
        if let last = stream.last {
            let dt = 1 / stream.fps
            for j in q.indices {
                let step = velocityLimit[j] * dt
                let d = q[j] - last.q[j]
                if abs(d) > step {
                    q[j] = last.q[j] + (d > 0 ? step : -step)
                    flags |= LiveFlags.velocityClamped
                }
                velocity[j] = (q[j] - last.q[j]) / dt
            }
        } else {
            flags |= LiveFlags.coldStart
        }
        if contacts.unsupported { flags |= LiveFlags.unsupported }
        if contacts.penetration { flags |= LiveFlags.penetration }
        if contacts.envSupport { flags |= LiveFlags.environmentSupport }
        if contacts.flight { flags |= LiveFlags.flight }
        stream.last = (q, root)
        active[slot] = stream
        return LiveRobotFrame(slot: slot, frame: frame, root: root, q: q, qVelocity: velocity, solvedQ: solved, cost: cost,
                              contacts: contacts, flags: flags)
    }
}
