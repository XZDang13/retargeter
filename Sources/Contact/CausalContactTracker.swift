// Causal (realtime) ground and contact tracking: one frame in, one frame of state out, using only the
// past (DESIGN §5a "realtime"). Differences from the offline analysis:
//   - ground: calibrated from the first second of static foot heights, then a slow EMA that only learns
//     from frames with both feet planted near it and may rise by at most `groundMaxRise`;
//   - speeds: backward differences, one-pole filtered;
//   - state changes are reported after they persist for `commitDelay` (debounce);
//   - run-length rules (unsupported, penetration) use trailing counters.

import Core

public struct FrameContacts: Sendable {
    public var ground: Float
    public var height: [Float]       // per group, relative to ground
    public var speed: [Float]
    public var contact: [Bool]
    public var envContact: [Bool]
    public var envSupport: Bool
    public var unsupported: Bool
    public var flight: Bool
    public var penetration: Bool
    public var com: SIMD3<Float>
    public var zmp: SIMD2<Float>
}

public final class CausalContactTracker {
    public let groups: ContactGroups
    public let config: ContactConfig
    let fps: Double
    let vertexCount: Int
    let excluded: [Bool]

    private var history: [[SIMD3<Float>]] = []   // most recent last
    private let speedLag: Int
    private let accelerationLag: Int
    private let commitFrames: Int
    private let holdFrames: Int
    private let unsupportedFrames: Int
    private let penetrationFrames: Int
    private let calibrationFrames: Int

    private var frameIndex = 0
    private var ground: Float?
    private var calibration: [Float] = []
    private var filteredSpeed: [Float]
    private var raw: [Bool], reported: [Bool], pending: [Int], exitCount: [Int]
    private var comHistory: [SIMD3<Float>] = []
    private var filteredCom: SIMD3<Float>?
    /// Filtered body-part centroids (point masses of the multi-body ZMP) over the last 2 * accelerationLag + 1 frames.
    private var partHistory: [[SIMD3<Float>]] = []
    private var filteredParts: [SIMD3<Float>]?
    private var outsideRun = 0, penetrationRun = 0, terrainRun = 0
    private let terrainFrames: Int

    public init(groups: ContactGroups, fps: Double, vertexCount: Int, excluded: [Bool], config: ContactConfig = ContactConfig()) {
        self.groups = groups
        self.config = config
        self.fps = fps
        self.vertexCount = vertexCount
        self.excluded = excluded
        let frames = { (seconds: Double) in max(1, Int((seconds * fps).rounded())) }
        speedLag = frames(0.025)
        accelerationLag = frames(config.accelerationSpan)
        commitFrames = frames(config.commitDelay)
        holdFrames = frames(config.exitHold)
        unsupportedFrames = frames(config.unsupportedMinDuration)
        penetrationFrames = frames(config.penetrationMinDuration)
        calibrationFrames = frames(config.calibrationDuration)
        terrainFrames = frames(0.25)
        let G = groups.count
        filteredSpeed = Array(repeating: 0, count: G)
        raw = Array(repeating: false, count: G)
        reported = raw
        pending = Array(repeating: 0, count: G)
        exitCount = pending
    }

    /// Feeds one frame of posed vertices (`vertexCount` positions).
    public func update(_ vertices: [SIMD3<Float>]) -> FrameContacts {
        precondition(vertices.count == vertexCount)
        let c = config, G = groups.count
        let dt = Float(1 / fps)
        history.append(vertices)
        if history.count > max(speedLag, 1) + 1 { history.removeFirst() }
        let current = PosedSurface(fps: fps, vertexCount: vertexCount, positions: vertices, excluded: excluded)
        let measures = measure(current, frame: 0, groups: groups, k: c.lowestK)

        // Speeds: backward difference of the current lowest vertices, one-pole filtered.
        let alpha = Float(1 - exp(-2 * .pi * c.speedFilterCutoff / fps))
        let past = history.first!
        let lag = history.count - 1
        for g in 0..<G {
            var d = SIMD2<Float>.zero
            for v in measures[g].lowest {
                let delta = vertices[v] - past[v]
                d += SIMD2(delta.x, delta.y)
            }
            let n = Float(max(measures[g].lowest.count, 1))
            let speed = lag > 0 ? (d * d).sum().squareRoot() / n / (Float(lag) * dt) : 0
            filteredSpeed[g] += alpha * (speed - filteredSpeed[g])
        }

        // Ground.
        let feet = (0..<G).filter { groups.classes[$0] == .foot }
        let staticFeet = feet.filter { Double(filteredSpeed[$0]) < c.groundStaticSpeed }
        if frameIndex < calibrationFrames || ground == nil {
            calibration += staticFeet.map { measures[$0].height }
            if !calibration.isEmpty { ground = ContactAnalysis.lowestDenseMode(calibration, band: Float(c.groundBand)) }
        } else if let g = ground {
            let planted = staticFeet.filter { abs(measures[$0].height - g) < Float(c.groundBand) }
            let leftDown = planted.contains { groups.names[$0].hasPrefix("left_") }
            let rightDown = planted.contains { groups.names[$0].hasPrefix("right_") }
            if leftDown && rightDown {
                let target = planted.map { measures[$0].height }.reduce(0, +) / Float(planted.count)
                var step = Float(1 - exp(-1 / (c.groundTimeConstant * fps))) * (target - g)
                step = min(step, Float(c.groundMaxRise) * dt)
                ground = g + step
            }
        }
        let groundHeight = ground ?? measures.map(\.height).min() ?? 0

        // Contacts: hysteresis on the raw state, then a commit delay before reporting changes.
        var height = [Float](repeating: 0, count: G)
        for g in 0..<G {
            height[g] = measures[g].height - groundHeight
            let h = c.hysteresis(groups.classes[g])
            let z = Double(height[g]), v = Double(filteredSpeed[g])
            if !raw[g] {
                if z < h.enterHeight && v < h.enterSpeed { raw[g] = true; exitCount[g] = 0 }
            } else if z > h.exitHeight || v > h.exitSpeed {
                exitCount[g] += 1
                if exitCount[g] >= holdFrames { raw[g] = false }
            } else { exitCount[g] = 0 }
            if raw[g] != reported[g] {
                pending[g] += 1
                if pending[g] >= commitFrames { reported[g] = raw[g]; pending[g] = 0 }
            } else { pending[g] = 0 }
        }

        // CoM, acceleration (backward second difference of the filtered CoM) and the ZMP of the body parts as point
        // masses (as offline): p = Σ m_i [(g + z̈_i) r_i − z_i r̈_i] / Σ m_i (g + z̈_i).
        let comAlpha = Float(1 - exp(-2 * .pi * (1 / max(c.comSmoothing, 1e-3)) / fps))
        let parts = groupCentroids(current, frame: 0, groups: groups)
        filteredParts = filteredParts.map { old in zip(old, parts).map { $0 + comAlpha * ($1 - $0) } } ?? parts
        let mass = groups.massFractions.map(Float.init)
        filteredCom = zip(mass, filteredParts!).reduce(SIMD3<Float>.zero) { $0 + $1.0 * $1.1 }
        comHistory.append(filteredCom!)
        partHistory.append(filteredParts!)
        if comHistory.count > 2 * accelerationLag + 1 { comHistory.removeFirst(); partHistory.removeFirst() }
        var acc = SIMD3<Float>.zero
        if comHistory.count == 2 * accelerationLag + 1 {
            let h = Float(accelerationLag) * dt
            acc = (comHistory[2 * accelerationLag] - 2 * comHistory[accelerationLag] + comHistory[0]) / (h * h)
        }
        let vertical = 9.81 + acc.z
        let flight = vertical < Float(c.flightGravityFraction) * 9.81
        var moment = SIMD2<Float>.zero, load: Float = 0
        for g in 0..<G where mass[g] > 0 {
            let r = filteredParts![g]
            var ag = SIMD3<Float>.zero
            if partHistory.count == 2 * accelerationLag + 1 {
                let h = Float(accelerationLag) * dt
                ag = (partHistory[2 * accelerationLag][g] - 2 * partHistory[accelerationLag][g] + partHistory[0][g]) / (h * h)
            }
            let w = mass[g] * (9.81 + ag.z)
            moment += w * SIMD2(r.x, r.y) - mass[g] * (r.z - groundHeight) * SIMD2(ag.x, ag.y)
            load += w
        }
        let zmp = moment / max(load, 0.5 * mass.reduce(0, +))

        // Support hull: contacts and parts on the ground too fast for a contact (sliding feet), as offline.
        var groundPatch: [SIMD2<Float>] = []
        for g in 0..<G where reported[g] || Double(height[g]) < c.hysteresis(groups.classes[g]).enterHeight {
            groundPatch += patch(current, frame: 0, group: g, groups: groups, measures: [measures], c: c)
        }
        let outside = !flight && Double(distanceToHull(zmp, groundPatch)) > c.unsupportedMargin
        outsideRun = outside ? outsideRun + 1 : 0
        let unsupported = outsideRun >= unsupportedFrames

        var envContact = [Bool](repeating: false, count: G)
        var envSupport = false
        if unsupported {
            let candidates = (0..<G).filter { g in
                let elevated = Double(height[g]) > (groups.classes[g] == .foot ? c.terrainStep : c.envMinHeight)
                return elevated && Double(filteredSpeed[g]) < c.envStaticSpeed
            }
            for g in supportingGroups(current, frame: 0, ground: groundPatch, candidates: candidates, zmp: zmp,
                                      groups: groups, measures: [measures], c: c) {
                if groups.classes[g] == .foot { envSupport = true } else { envContact[g] = true }
            }
        }

        // Terrain: static feet holding off the tracked ground (a step up or down) for a sustained run.
        let staticOff = staticFeet.filter { abs(Double(height[$0])) > c.terrainStep }
        let staticOn = staticFeet.filter { abs(Double(height[$0])) <= c.terrainStep }
        terrainRun = !flight && !staticOff.isEmpty && staticOn.isEmpty ? terrainRun + 1 : 0
        if terrainRun >= terrainFrames { envSupport = true }

        let penetrating = (0..<G).contains { penetrates(height[$0], group: $0, groups: groups, c: c) }
        penetrationRun = penetrating ? penetrationRun + 1 : 0

        frameIndex += 1
        return FrameContacts(ground: groundHeight, height: height, speed: filteredSpeed, contact: reported,
                             envContact: envContact, envSupport: envSupport, unsupported: unsupported, flight: flight,
                             penetration: penetrationRun >= penetrationFrames, com: filteredCom!, zmp: zmp)
    }
}

extension ContactAnalysis {
    /// Runs the causal tracker over a whole clip (for evaluation and display).
    public static func causal(_ s: PosedSurface, groups: ContactGroups, config: ContactConfig = ContactConfig()) -> ContactTimeline {
        let tracker = CausalContactTracker(groups: groups, fps: s.fps, vertexCount: s.vertexCount, excluded: s.excluded,
                                           config: config)
        let F = s.frameCount, V = s.vertexCount
        var t = ContactTimeline(fps: s.fps, frameCount: F, groupNames: groups.names, ground: [], groundTilt: 0,
                                height: [], speed: [], contact: [], envContact: [], envSupport: [], unsupported: [],
                                flight: [], penetration: [], com: [], zmp: [])
        for f in 0..<F {
            let r = tracker.update(Array(s.positions[(f * V)..<((f + 1) * V)]))
            t.ground.append(r.ground)
            t.height += r.height
            t.speed += r.speed
            t.contact += r.contact
            t.envContact += r.envContact
            t.envSupport.append(r.envSupport)
            t.unsupported.append(r.unsupported)
            t.flight.append(r.flight)
            t.penetration.append(r.penetration)
            t.com.append(r.com)
            t.zmp.append(r.zmp)
        }
        return t
    }
}

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif
