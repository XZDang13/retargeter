// Ground estimation and any-body contact detection on the posed human surface (DESIGN §5a).
//
// Inputs are posed vertices (`[frame * vertexCount]`, world frame, Z-up metres). Per frame and contact
// group we measure `h`, the mean height of the group's lowest-k vertices, and `v`, their horizontal
// speed. From those:
//   - ground: lowest dense mode of quasi-static foot heights (one value per clip, offline);
//   - contacts: per-class height/speed hysteresis, gap bridging, minimum run length;
//   - support: ZMP (cart-table model on the surface centroid) against the hull of ground contacts. When
//     ground contacts cannot explain the motion for a sustained run, the cause is classified as
//     environment support (an elevated static foot: stairs, platform) or environment contact (another
//     elevated static body part: chair, table, wall); otherwise the run is just `unsupported`.

import Core

public struct ContactConfig: Sendable {
    public struct Hysteresis: Sendable {
        public var enterHeight: Double, enterSpeed: Double, exitHeight: Double, exitSpeed: Double
    }

    public var lowestK = 8
    public var foot = Hysteresis(enterHeight: 0.03, enterSpeed: 0.15, exitHeight: 0.05, exitSpeed: 0.30)
    public var limb = Hysteresis(enterHeight: 0.04, enterSpeed: 0.25, exitHeight: 0.08, exitSpeed: 0.40)
    public var trunk = Hysteresis(enterHeight: 0.05, enterSpeed: 0.25, exitHeight: 0.08, exitSpeed: 0.40)
    /// Exit conditions must hold this long before a contact ends.
    public var exitHold = 0.017
    public var bridgeGap = 0.025
    public var minRun = 0.05

    // Ground
    public var groundStaticSpeed = 0.2
    public var groundBand = 0.02
    public var terrainStep = 0.04

    // Support / environment
    public var envStaticSpeed = 0.1
    public var envMinHeight = 0.06
    public var unsupportedMargin = 0.06
    /// A part can rest on the environment only where the body does not continue below it: its contact-patch vertices
    /// count as covered when a vertex of another group lies within `coverRadius` horizontally and more than
    /// `coverDrop` lower, and a part with more than half of them covered is no support (a torso above the legs,
    /// a hanging upper arm, the head). Without it a static upper body explained support on step-downs.
    public var coverRadius = 0.04
    public var coverDrop = 0.05
    /// A static, free-below part whose lowest vertices are within this (m) of a supporting part's rests on the same
    /// surface (pelvis and thighs on a seat, elbow and forearm on an armrest).
    public var restingNeighbour = 0.15
    public var restingLevel = 0.08
    public var unsupportedMinDuration = 0.3
    public var comSmoothing = 0.1        // s, centred moving average before differentiating
    public var accelerationSpan = 0.05   // s, finite-difference spacing for CoM acceleration
    public var flightGravityFraction = 0.25  // g + a_z below this fraction of g: ballistic

    /// Vertices of a contacting group within this height of its lowest point form its contact patch.
    public var patchHeight = 0.03

    // Source data. Mocap soft tissue sinks 1–2 cm and LBS has no flesh compression, so limbs and trunk get
    // more slack than feet. Hands are not checked: SEED finger capture bends through the floor under load
    // (fingertips up to ~10 cm below it) and fingers are not retargeted.
    public var penetrationDepth = 0.04          // feet (kneeling on tucked toes: heels 3-4 cm under the floor)
    public var penetrationDepthBody = 0.08      // limbs and trunk (kneeling: knees and thighs 6-7 cm under load)
    public var penetrationIgnored: Set<String> = ["left_hand", "right_hand"]
    public var penetrationMinDuration = 0.025

    // Causal tracker
    public var calibrationDuration = 1.0
    public var groundTimeConstant = 5.0
    public var groundMaxRise = 0.01      // m/s
    public var commitDelay = 0.017       // s a new state must persist before it is reported
    public var speedFilterCutoff = 10.0  // Hz

    public init() {}

    func hysteresis(_ cls: ContactClass) -> Hysteresis {
        switch cls {
        case .foot: foot
        case .limb: limb
        case .trunk: trunk
        }
    }
}

/// Posed surface for a clip: `positions[frame * vertexCount + vertex]`.
public struct PosedSurface: Sendable {
    public var fps: Double
    public var vertexCount: Int
    public var positions: [SIMD3<Float>]
    public var excluded: [Bool]

    public init(fps: Double, vertexCount: Int, positions: [SIMD3<Float>], excluded: [Bool]) {
        self.fps = fps
        self.vertexCount = vertexCount
        self.positions = positions
        self.excluded = excluded
    }

    public var frameCount: Int { positions.count / max(vertexCount, 1) }

    @inline(__always) public func p(_ f: Int, _ v: Int) -> SIMD3<Float> { positions[f * vertexCount + v] }
}

public struct ContactTimeline: Sendable {
    public var fps: Double
    public var frameCount: Int
    public var groupNames: [String]
    /// Ground height per frame (constant offline; tracked in causal mode).
    public var ground: [Float]
    public var groundTilt: Double          // degrees, plane fit over static foot contacts
    /// Per [frame * group]: h − ground and horizontal speed of the lowest-k vertices.
    public var height: [Float]
    public var speed: [Float]
    public var contact: [Bool]             // [frame * group]
    public var envContact: [Bool]          // [frame * group]
    public var envSupport: [Bool]          // [frame]
    public var unsupported: [Bool]         // [frame]
    public var flight: [Bool]              // [frame]
    public var penetration: [Bool]         // [frame]
    public var com: [SIMD3<Float>]
    public var zmp: [SIMD2<Float>]

    public init(fps: Double, frameCount: Int, groupNames: [String], ground: [Float], groundTilt: Double, height: [Float],
                speed: [Float], contact: [Bool], envContact: [Bool], envSupport: [Bool], unsupported: [Bool],
                flight: [Bool], penetration: [Bool], com: [SIMD3<Float>], zmp: [SIMD2<Float>]) {
        self.fps = fps
        self.frameCount = frameCount
        self.groupNames = groupNames
        self.ground = ground
        self.groundTilt = groundTilt
        self.height = height
        self.speed = speed
        self.contact = contact
        self.envContact = envContact
        self.envSupport = envSupport
        self.unsupported = unsupported
        self.flight = flight
        self.penetration = penetration
        self.com = com
        self.zmp = zmp
    }

    public func contact(_ frame: Int, _ group: Int) -> Bool { contact[frame * groupNames.count + group] }

    /// Reasons this clip cannot be retargeted against a flat ground (input to the auto-reject).
    public var flags: [String] {
        var out: [String] = []
        if envSupport.contains(true) { out.append("env_support") }
        if envContact.contains(true) { out.append("env_contact") }
        let explained = zip(envSupport, (0..<frameCount).map { f in (0..<groupNames.count).contains { envContact[f * groupNames.count + $0] } })
            .map { $0 || $1 }
        if zip(unsupported, explained).contains(where: { $0 && !$1 }) { out.append("unsupported") }
        if penetration.contains(true) { out.append("penetration") }
        if abs(groundTilt) > 2 { out.append("ground_tilt") }
        return out
    }
}

// MARK: - Per-frame measurements shared by both modes

struct GroupMeasure {
    var height: Float = 0         // mean z of the lowest-k vertices
    var lowest: [Int] = []        // their vertex ids
}

func measure(_ s: PosedSurface, frame f: Int, groups: ContactGroups, k: Int) -> [GroupMeasure] {
    groups.vertices.map { members in
        guard !members.isEmpty else { return GroupMeasure(height: .infinity, lowest: []) }
        // Partial selection of the k lowest vertices.
        var best: [(Float, Int)] = []
        best.reserveCapacity(k + 1)
        for v in members {
            let z = s.p(f, v).z
            if best.count < k {
                best.append((z, v))
                if best.count == k { best.sort { $0.0 < $1.0 } }
            } else if z < best[k - 1].0 {
                var i = k - 1
                while i > 0 && best[i - 1].0 > z { best[i] = best[i - 1]; i -= 1 }
                best[i] = (z, v)
            }
        }
        let sum = best.reduce(Float(0)) { $0 + $1.0 }
        return GroupMeasure(height: sum / Float(best.count), lowest: best.map(\.1))
    }
}

func horizontalSpeed(_ s: PosedSurface, vertices: [Int], from a: Int, to b: Int) -> Float {
    guard !vertices.isEmpty, a != b else { return 0 }
    var d = SIMD2<Float>.zero
    for v in vertices {
        let delta = s.p(b, v) - s.p(a, v)
        d += SIMD2(delta.x, delta.y)
    }
    d /= Float(vertices.count)
    return (d * d).sum().squareRoot() * Float(s.fps) / Float(b - a)
}

/// Vertex centroid of every group (the point masses of `centerOfMass` and of the multi-body ZMP).
func groupCentroids(_ s: PosedSurface, frame f: Int, groups: ContactGroups) -> [SIMD3<Float>] {
    groups.vertices.map { members in
        guard !members.isEmpty else { return .zero }
        var sum = SIMD3<Float>.zero
        for v in members { sum += s.p(f, v) }
        return sum / Float(members.count)
    }
}

func minimumHeight(_ s: PosedSurface, frame f: Int) -> Float {
    var m = Float.infinity
    for v in 0..<s.vertexCount where !s.excluded[v] { m = min(m, s.p(f, v).z) }
    return m
}

func penetrates(_ height: Float, group g: Int, groups: ContactGroups, c: ContactConfig) -> Bool {
    if c.penetrationIgnored.contains(groups.names[g]) { return false }
    return Double(height) < -(groups.classes[g] == .foot ? c.penetrationDepth : c.penetrationDepthBody)
}

/// Greedy explanation of an unsupported frame: repeatedly add the elevated static part that brings the ZMP
/// closest to the support hull (a chair seat carries the pelvis rather than the shins or the hands on the lap).
/// Returns the parts needed, or [] if even all of them cannot explain the ZMP.
func supportingGroups(_ s: PosedSurface, frame f: Int, ground: [SIMD2<Float>], candidates: [Int], zmp: SIMD2<Float>,
                      groups: ContactGroups, measures: [[GroupMeasure]], c: ContactConfig) -> [Int] {
    var hull = ground
    var supporting: [Int] = []
    var remaining = uncovered(s, frame: f, candidates: candidates, groups: groups, measures: measures, c: c)
    var neighbours = candidates.filter { !remaining.contains($0) }
    var distance = distanceToHull(zmp, hull)
    while Double(distance) > c.unsupportedMargin && !remaining.isEmpty {
        let patches = remaining.map { patch(s, frame: f, group: $0, groups: groups, measures: measures, c: c) }
        let scores = patches.map { distanceToHull(zmp, hull + $0) }
        let best = scores.indices.min { scores[$0] < scores[$1] }!
        guard scores[best] < distance else { break }
        hull += patches[best]
        supporting.append(remaining[best])
        distance = scores[best]
        remaining.remove(at: best)
    }
    guard Double(distance) <= c.unsupportedMargin else { return [] }
    // Parts resting on the same surface as a needed one count too: the greedy picks the fewest parts (the pelvis
    // alone on a chair, while the thighs carry it). A neighbour is any static elevated part whose patch comes within
    // restingNeighbour of a supporting part's, at most 2 cm above it and restingLevel below, free below or not: a
    // seated thigh's patch lies at the knee end, above the hanging shin. Not above, or the torso and then the arms
    // chain on from a seated pelvis.
    neighbours += remaining
    var added = true
    while added {
        added = false
        for (i, r) in neighbours.enumerated().reversed() where supporting.contains(where: {
            let rise = measures[f][r].height - measures[f][$0].height
            return rise <= 0.02 && rise >= -Float(c.restingLevel)
                && touches(s, frame: f, $0, r, groups: groups, measures: measures, c: c)
        }) {
            supporting.append(r)
            neighbours.remove(at: i)
            added = true
        }
    }
    return supporting
}

/// A group's contact patch (vertices within patchHeight of its lowest), up to 32 of them evenly. The lowest-k
/// vertices alone cluster at one end: a seated thigh's lie at the knee, above the hanging shin, 20 cm from the pelvis.
func patchVertices(_ s: PosedSurface, frame f: Int, group g: Int, groups: ContactGroups, measures: [[GroupMeasure]],
                   c: ContactConfig) -> [Int] {
    let limit = measures[f][g].height + Float(c.patchHeight)
    let surface = groups.vertices[g].filter { !s.excluded[$0] && s.p(f, $0).z <= limit }
    let step = max(1, surface.count / 32)
    return stride(from: 0, to: surface.count, by: step).map { surface[$0] }
}

/// True if the contact patches of groups `a` and `b` come within `ContactConfig.restingNeighbour` of each other.
func touches(_ s: PosedSurface, frame f: Int, _ a: Int, _ b: Int, groups: ContactGroups, measures: [[GroupMeasure]],
             c: ContactConfig) -> Bool {
    let r2 = Float(c.restingNeighbour * c.restingNeighbour)
    let other = patchVertices(s, frame: f, group: b, groups: groups, measures: measures, c: c)
    for u in patchVertices(s, frame: f, group: a, groups: groups, measures: measures, c: c) {
        let p = s.p(f, u)
        for v in other {
            let d = s.p(f, v) - p
            if (d * d).sum() <= r2 { return true }
        }
    }
    return false
}

/// The candidates whose contact patch is mostly free below (see `ContactConfig.coverRadius`).
func uncovered(_ s: PosedSurface, frame f: Int, candidates: [Int], groups: ContactGroups, measures: [[GroupMeasure]],
               c: ContactConfig) -> [Int] {
    guard !candidates.isEmpty else { return [] }
    // Lowest vertex height per (grid cell, group) on a coverRadius grid; a probe checks its 3 x 3 cells.
    let cell = Float(c.coverRadius), G = groups.count
    var lowest: [SIMD2<Int32>: [Float]] = [:]
    for (g, members) in groups.vertices.enumerated() {
        for v in members where !s.excluded[v] {
            let p = s.p(f, v)
            let key = SIMD2(Int32((p.x / cell).rounded(.down)), Int32((p.y / cell).rounded(.down)))
            var row = lowest[key] ?? [Float](repeating: .infinity, count: G)
            row[g] = min(row[g], p.z)
            lowest[key] = row
        }
    }
    let r2 = cell * cell
    return candidates.filter { g in
        let probes = patchVertices(s, frame: f, group: g, groups: groups, measures: measures, c: c)
        let covered = probes.filter { v in
            let p = s.p(f, v)
            let cx = Int32((p.x / cell).rounded(.down)), cy = Int32((p.y / cell).rounded(.down))
            for dx in Int32(-1)...1 {
                for dy in Int32(-1)...1 {
                    guard let row = lowest[SIMD2(cx + dx, cy + dy)] else { continue }
                    for h in 0..<G where h != g && row[h] < p.z - Float(c.coverDrop) {
                        // The cell is a coarse filter; confirm against the group's actual vertices.
                        if groups.vertices[h].contains(where: { u in
                            let q = s.p(f, u)
                            let d = SIMD2(q.x - p.x, q.y - p.y)
                            return !s.excluded[u] && q.z < p.z - Float(c.coverDrop) && (d * d).sum() <= r2
                        }) { return true }
                    }
                }
            }
            return false
        }.count
        return 2 * covered <= probes.count
    }
}

/// Ground-plane footprint of a group: its vertices within `patchHeight` of its lowest-k height.
func patch(_ s: PosedSurface, frame f: Int, group g: Int, groups: ContactGroups, measures: [[GroupMeasure]],
           c: ContactConfig) -> [SIMD2<Float>] {
    let limit = measures[f][g].height + Float(c.patchHeight)
    var out: [SIMD2<Float>] = []
    for v in groups.vertices[g] {
        let p = s.p(f, v)
        if p.z <= limit { out.append(SIMD2(p.x, p.y)) }
    }
    return out
}

// MARK: - Geometry helpers

/// Distance from `p` to the convex hull of `points` (0 inside). Empty hull: +∞.
func distanceToHull(_ p: SIMD2<Float>, _ points: [SIMD2<Float>]) -> Float {
    guard !points.isEmpty else { return .infinity }
    let hull = convexHull(points)
    func segmentDistance(_ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        let ab = b - a
        let t = max(0, min(1, ((p - a) * ab).sum() / max((ab * ab).sum(), 1e-12)))
        let d = p - (a + t * ab)
        return (d * d).sum().squareRoot()
    }
    if hull.count == 1 { let d = p - hull[0]; return (d * d).sum().squareRoot() }
    if hull.count == 2 { return segmentDistance(hull[0], hull[1]) }
    var inside = true
    var best = Float.infinity
    for i in 0..<hull.count {
        let a = hull[i], b = hull[(i + 1) % hull.count]
        let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
        if cross < 0 { inside = false }
        best = min(best, segmentDistance(a, b))
    }
    return inside ? 0 : best
}

/// Andrew's monotone chain, counter-clockwise.
func convexHull(_ input: [SIMD2<Float>]) -> [SIMD2<Float>] {
    let pts = input.sorted { $0.x < $1.x || ($0.x == $1.x && $0.y < $1.y) }
    guard pts.count > 2 else { return pts }
    func cross(_ o: SIMD2<Float>, _ a: SIMD2<Float>, _ b: SIMD2<Float>) -> Float {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
    var lower: [SIMD2<Float>] = [], upper: [SIMD2<Float>] = []
    for p in pts {
        while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 { lower.removeLast() }
        lower.append(p)
    }
    for p in pts.reversed() {
        while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 { upper.removeLast() }
        upper.append(p)
    }
    return Array(lower.dropLast() + upper.dropLast())
}

/// Runs of `true` shorter than `minLength` frames are cleared; gaps of `false` up to `maxGap` frames
/// between runs are filled.
func cleanRuns(_ mask: inout [Bool], maxGap: Int, minLength: Int) {
    let n = mask.count
    var i = 0
    while i < n {  // bridge gaps
        if !mask[i], i > 0, mask[i - 1] {
            var j = i
            while j < n && !mask[j] { j += 1 }
            if j < n && j - i <= maxGap { for k in i..<j { mask[k] = true } }
            i = j
        } else { i += 1 }
    }
    i = 0
    while i < n {  // drop short runs
        if mask[i] {
            var j = i
            while j < n && mask[j] { j += 1 }
            if j - i < minLength { for k in i..<j { mask[k] = false } }
            i = j
        } else { i += 1 }
    }
}

/// Keeps only runs of `true` lasting at least `minLength` frames.
func sustained(_ mask: [Bool], minLength: Int) -> [Bool] {
    var out = mask
    cleanRuns(&out, maxGap: 0, minLength: minLength)
    return out
}

// MARK: - Offline analysis

public enum ContactAnalysis {
    public static func offline(_ s: PosedSurface, groups: ContactGroups, config c: ContactConfig = ContactConfig()) -> ContactTimeline {
        let F = s.frameCount, G = groups.count
        let frames = { (seconds: Double) in max(1, Int((seconds * s.fps).rounded())) }
        let measures = (0..<F).map { measure(s, frame: $0, groups: groups, k: c.lowestK) }

        // Horizontal speed of each group's lowest vertices: centred difference over ±20 ms.
        let d = frames(0.02)
        var speed = [Float](repeating: 0, count: F * G)
        for f in 0..<F {
            let a = max(0, f - d), b = min(F - 1, f + d)
            for g in 0..<G { speed[f * G + g] = horizontalSpeed(s, vertices: measures[f][g].lowest, from: a, to: b) }
        }

        // Ground: lowest dense mode of quasi-static foot heights.
        let footGroups = (0..<G).filter { groups.classes[$0] == .foot }
        var samples: [(z: Float, xy: SIMD2<Float>, group: Int)] = []
        for f in 0..<F {
            for g in footGroups where Double(speed[f * G + g]) < c.groundStaticSpeed {
                let lowest = measures[f][g].lowest
                var xy = SIMD2<Float>.zero
                for v in lowest { let p = s.p(f, v); xy += SIMD2(p.x, p.y) }
                samples.append((measures[f][g].height, xy / Float(max(lowest.count, 1)), g))
            }
        }
        let minimum = (0..<F).map { minimumHeight(s, frame: $0) }
        let ground: Float
        if samples.count >= 10 {
            ground = lowestDenseMode(samples.map(\.z), band: Float(c.groundBand))
        } else {
            let sorted = minimum.sorted()
            ground = sorted[min(sorted.count - 1, sorted.count / 100)]
        }
        let tilt = planeFit(samples.filter { abs($0.z - ground) < 0.015 })?.tilt ?? 0
        // Local floor for penetration: a capture floor that is not quite level drifts 3-5 cm over a few metres of
        // travel, which one height per clip reads as the feet sinking in (jog_sideway_right_loop_001__A026). The
        // median static foot height within 0.5 m (samples within 5 cm of the ground, at least 20), else the ground.
        let floorCell: Float = 0.5
        var floorSamples: [SIMD2<Int32>: [Float]] = [:]
        for p in samples where abs(p.z - ground) < 0.05 {
            floorSamples[SIMD2(Int32((p.xy.x / floorCell).rounded(.down)), Int32((p.xy.y / floorCell).rounded(.down))), default: []].append(p.z)
        }
        var floorCache: [SIMD2<Int32>: Float] = [:]
        func localFloor(_ xy: SIMD2<Float>) -> Float {
            let key = SIMD2(Int32((xy.x / floorCell).rounded(.down)), Int32((xy.y / floorCell).rounded(.down)))
            if let z = floorCache[key] { return z }
            var near: [Float] = []
            for dx in Int32(-1)...1 { for dy in Int32(-1)...1 { near += floorSamples[SIMD2(key.x + dx, key.y + dy)] ?? [] } }
            let z = near.count >= 20 ? near.sorted()[near.count / 2] : ground
            floorCache[key] = z
            return z
        }

        var height = [Float](repeating: 0, count: F * G)
        for f in 0..<F { for g in 0..<G { height[f * G + g] = measures[f][g].height - ground } }

        // Contacts per group with hysteresis.
        var contact = [Bool](repeating: false, count: F * G)
        let hold = frames(c.exitHold)
        for g in 0..<G {
            let h = c.hysteresis(groups.classes[g])
            var mask = [Bool](repeating: false, count: F)
            var state = false
            var exitCount = 0
            for f in 0..<F {
                let z = Double(height[f * G + g]), v = Double(speed[f * G + g])
                if !state {
                    if z < h.enterHeight && v < h.enterSpeed { state = true; exitCount = 0 }
                } else if z > h.exitHeight || v > h.exitSpeed {
                    exitCount += 1
                    if exitCount >= hold { state = false }
                } else { exitCount = 0 }
                mask[f] = state
            }
            cleanRuns(&mask, maxGap: frames(c.bridgeGap), minLength: frames(c.minRun))
            for f in 0..<F { contact[f * G + g] = mask[f] }
        }

        // Support: ZMP against the hull of the parts touching the ground: the contacts, and parts on the ground but too
        // fast for a contact (a sliding foot still carries weight: dance slides, sideways jog push-off).
        func touching(_ f: Int, _ g: Int) -> Bool {
            contact[f * G + g] || Double(height[f * G + g]) < c.hysteresis(groups.classes[g]).enterHeight
        }
        // ZMP of the body parts as point masses (group centroids, de Leva fractions):
        //   p = Σ m_i [(g + z̈_i) r_i − z_i r̈_i] / Σ m_i (g + z̈_i).
        // The CoM alone (cart-table) ignores the momentum of swinging limbs and a twisting torso, which put its ZMP
        // 20-40 cm outside the feet through turning jogs (neutral_dancecard_go_turn_jog_001__A536).
        let radius = frames(c.comSmoothing) / 2
        let centroids = (0..<F).map { groupCentroids(s, frame: $0, groups: groups) }
        let parts = (0..<G).map { g in movingAverage(centroids.map { $0[g] }, radius: radius) }
        let mass = groups.massFractions.map(Float.init)
        let com = (0..<F).map { f in (0..<G).reduce(SIMD3<Float>.zero) { $0 + mass[$1] * parts[$1][f] } }
        let span = frames(c.accelerationSpan)
        var zmp = [SIMD2<Float>](repeating: .zero, count: F)
        var flight = [Bool](repeating: false, count: F)
        var outside = [Bool](repeating: false, count: F)
        let gravity: Float = 9.81
        for f in 0..<F {
            let a = max(0, f - span), b = min(F - 1, f + span)
            let h = Float(b - a) / 2 / Float(s.fps)
            let acc = b - a >= 2 ? (com[b] - 2 * com[(a + b) / 2] + com[a]) / (h * h) : .zero
            let vertical = gravity + acc.z
            flight[f] = vertical < Float(c.flightGravityFraction) * gravity
            var moment = SIMD2<Float>.zero, load: Float = 0
            for g in 0..<G where mass[g] > 0 {
                let r = parts[g][f]
                let ag = b - a >= 2 ? (parts[g][b] - 2 * parts[g][(a + b) / 2] + parts[g][a]) / (h * h) : .zero
                let w = mass[g] * (gravity + ag.z)
                moment += w * SIMD2(r.x, r.y) - mass[g] * (r.z - ground) * SIMD2(ag.x, ag.y)
                load += w
            }
            zmp[f] = moment / max(load, 0.5 * Float(mass.reduce(0, +)))
            var points: [SIMD2<Float>] = []
            for g in 0..<G where touching(f, g) { points += patch(s, frame: f, group: g, groups: groups, measures: measures, c: c) }
            outside[f] = !flight[f] && Double(distanceToHull(zmp[f], points)) > c.unsupportedMargin
        }
        let unsupported = sustained(outside, minLength: frames(c.unsupportedMinDuration))

        // Explain unsupported frames: an elevated static part whose contact patch brings the ZMP back inside
        // the support hull is supporting the body (feet: env support such as stairs; others: env contact).
        var envSupport = [Bool](repeating: false, count: F)
        var envContact = [Bool](repeating: false, count: F * G)
        // Elevated and static in 3D: horizontal speed alone let knees and thighs swinging up in a burpee kick-back count.
        func resting(_ f: Int, _ g: Int) -> Bool {
            let elevated = Double(height[f * G + g]) > (groups.classes[g] == .foot ? c.terrainStep : c.envMinHeight)
            guard elevated && Double(speed[f * G + g]) < c.envStaticSpeed else { return false }
            let a = max(0, f - d), b = min(F - 1, f + d)
            let lowest = measures[f][g].lowest
            var dz: Float = 0
            for v in lowest { dz += s.p(b, v).z - s.p(a, v).z }
            let vertical = abs(dz) / Float(max(lowest.count, 1)) * Float(s.fps) / Float(max(b - a, 1))
            return Double(vertical) < c.envStaticSpeed
        }
        for f in 0..<F where unsupported[f] {
            var ground: [SIMD2<Float>] = []
            for g in 0..<G where touching(f, g) { ground += patch(s, frame: f, group: g, groups: groups, measures: measures, c: c) }
            let candidates = (0..<G).filter { resting(f, $0) }
            let supporting = supportingGroups(s, frame: f, ground: ground, candidates: candidates, zmp: zmp[f],
                                              groups: groups, measures: measures, c: c)
            for g in supporting {
                if groups.classes[g] == .foot { envSupport[f] = true } else { envContact[f * G + g] = true }
            }
        }
        // A part found on the environment stays there for its whole rest (elevated and static), also where the feet
        // alone would carry the body (sitting leaning forward): the run of resting frames around each labelled frame.
        for g in 0..<G where groups.classes[g] != .foot && (0..<F).contains(where: { envContact[$0 * G + g] }) {
            let rest = (0..<F).map { resting($0, g) }
            var f = 0
            while f < F {
                guard rest[f] else { f += 1; continue }
                var e = f
                while e < F && rest[e] { e += 1 }
                if (f..<e).contains(where: { envContact[$0 * G + g] }) { for k in f..<e { envContact[k * G + g] = true } }
                f = e
            }
        }

        // Penetration below the local floor (heights elsewhere are against the clip's one ground).
        func belowFloor(_ f: Int, _ g: Int) -> Float {
            let flat = height[f * G + g]
            guard flat < 0 else { return flat }
            let lowest = measures[f][g].lowest
            var xy = SIMD2<Float>.zero
            for v in lowest { let p = s.p(f, v); xy += SIMD2(p.x, p.y) }
            return measures[f][g].height - localFloor(xy / Float(max(lowest.count, 1)))
        }
        var penetration = (0..<F).map { f in (0..<G).contains { g in penetrates(belowFloor(f, g), group: g, groups: groups, c: c) } }
        penetration = sustained(penetration, minLength: frames(c.penetrationMinDuration))
        _ = minimum

        return ContactTimeline(fps: s.fps, frameCount: F, groupNames: groups.names,
                               ground: Array(repeating: ground, count: F), groundTilt: tilt,
                               height: height, speed: speed, contact: contact, envContact: envContact,
                               envSupport: envSupport, unsupported: unsupported, flight: flight,
                               penetration: penetration, com: com, zmp: zmp)
    }

    /// The lowest height whose ±band window holds at least 30 % of the densest window's samples.
    static func lowestDenseMode(_ values: [Float], band: Float) -> Float {
        let sorted = values.sorted()
        var counts = [Int](repeating: 0, count: sorted.count)
        var lo = 0, hi = 0
        for i in sorted.indices {
            while sorted[lo] < sorted[i] - band { lo += 1 }
            while hi + 1 < sorted.count && sorted[hi + 1] <= sorted[i] + band { hi += 1 }
            counts[i] = hi - lo + 1
        }
        let peak = counts.max() ?? 0
        guard let first = counts.indices.first(where: { Double(counts[$0]) >= 0.3 * Double(peak) }) else { return sorted.first ?? 0 }
        // Refine: densest point within the first dense window, then the median of its window.
        var best = first
        var i = first
        while i < sorted.count && sorted[i] <= sorted[first] + 2 * band {
            if counts[i] > counts[best] { best = i }
            i += 1
        }
        let window = sorted.filter { abs($0 - sorted[best]) <= band }
        return window[window.count / 2]
    }

    /// A floor plane z = mean.z + slope · (xy − mean.xy).
    struct Plane {
        var mean: SIMD3<Float>
        var slope: SIMD2<Float>
        var tilt: Double { atan(Double((slope * slope).sum().squareRoot())) * 180 / .pi }
        func height(_ xy: SIMD2<Float>) -> Float { mean.z + (slope * (xy - SIMD2(mean.x, mean.y))).sum() }
    }

    /// Least-squares plane through the points; nil for fewer than 20 points or points spread along a line only
    /// (walking straight constrains one slope).
    static func planeFit(_ points: [(z: Float, xy: SIMD2<Float>, group: Int)]) -> Plane? {
        guard points.count >= 20 else { return nil }
        // Each foot group about its own mean: a heel's lowest vertices sit a little higher or lower than a toe's, and
        // on a straight walk the toe samples lie ahead of the heel samples, which would read as slope.
        var means: [Int: (SIMD3<Double>, Double)] = [:]
        for p in points {
            let m = means[p.group] ?? (.zero, 0)
            means[p.group] = (m.0 + SIMD3(Double(p.xy.x), Double(p.xy.y), Double(p.z)), m.1 + 1)
        }
        let n = Double(points.count)
        var total = SIMD3<Double>.zero
        var cxx = 0.0, cyy = 0.0, cxy = 0.0, cxz = 0.0, cyz = 0.0
        for p in points {
            let m = means[p.group]!, c = m.0 / m.1
            let x = Double(p.xy.x) - c.x, y = Double(p.xy.y) - c.y, z = Double(p.z) - c.z
            cxx += x * x; cyy += y * y; cxy += x * y; cxz += x * z; cyz += y * z
            total += SIMD3(Double(p.xy.x), Double(p.xy.y), Double(p.z))
        }
        let det = cxx * cyy - cxy * cxy
        let mean = SIMD3<Float>(total / n)
        if det > 1e-6 * max(cxx * cyy, 1e-12), cxx > 0.01 * n, cyy > 0.01 * n {
            let a = (cxz * cyy - cyz * cxy) / det, b = (cyz * cxx - cxz * cxy) / det
            return Plane(mean: mean, slope: SIMD2(Float(a), Float(b)))
        }
        return nil
    }

    static func movingAverage(_ values: [SIMD3<Float>], radius: Int) -> [SIMD3<Float>] {
        guard radius > 0 else { return values }
        return values.indices.map { i in
            let a = max(0, i - radius), b = min(values.count - 1, i + radius)
            var sum = SIMD3<Float>.zero
            for j in a...b { sum += values[j] }
            return sum / Float(b - a + 1)
        }
    }
}

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif
