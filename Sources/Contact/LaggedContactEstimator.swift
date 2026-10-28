// Contact flags for fixed-lag streams: the offline analysis' per-frame test (DESIGN §5a) with the little lookahead it
// needs, decided `delay` frames late. Frame t is decided once frame t + delay arrives: horizontal speed of the
// group's lowest vertices as a centred difference over ±delay frames (offline: ±20 ms), the same height / speed
// hysteresis with exit hold, no commit delay, on the causal tracker's ground. The causal tracker reports contact
// changes ~45 ms late (backward differences, a 10 Hz speed filter, a 17 ms commit delay), and a fixed-lag window that
// locks feet on those flags let planted feet slide wherever they disagreed with the offline contacts it is judged
// against (comparison clips: foot skate excess 0.039 against 0.014 with the offline contacts).
// Not done online: the offline run cleaning (gaps bridged, short runs dropped), which needs ~50 ms of lookahead.

public final class LaggedContactEstimator {
    public let groups: ContactGroups
    public let config: ContactConfig
    public let delay: Int
    let fps: Double
    let vertexCount: Int
    let excluded: [Bool]
    let holdFrames: Int
    /// The last 2 * delay + 1 frames: posed vertices, lowest-vertex measures, ground.
    var window: [(vertices: [SIMD3<Float>], measures: [GroupMeasure], ground: Float)] = []
    var state: [Bool]
    var exitCount: [Int]

    public init(groups: ContactGroups, fps: Double, vertexCount: Int, excluded: [Bool], config: ContactConfig = ContactConfig()) {
        self.groups = groups
        self.config = config
        self.fps = fps
        self.vertexCount = vertexCount
        self.excluded = excluded
        delay = max(1, Int((0.02 * fps).rounded()))
        holdFrames = max(1, Int((config.exitHold * fps).rounded()))
        state = Array(repeating: false, count: groups.count)
        exitCount = Array(repeating: 0, count: groups.count)
    }

    /// Feeds one frame (posed vertices, the causal tracker's ground at it). Returns the contact flags per group of
    /// the frame `delay` frames back, once there is one.
    public func update(_ vertices: [SIMD3<Float>], ground: Float) -> [Bool]? {
        precondition(vertices.count == vertexCount)
        let surface = PosedSurface(fps: fps, vertexCount: vertexCount, positions: vertices, excluded: excluded)
        window.append((vertices, measure(surface, frame: 0, groups: groups, k: config.lowestK), ground))
        if window.count > 2 * delay + 1 { window.removeFirst() }
        guard window.count > delay else { return nil }
        // The decided frame and its neighbours (fewer before it at the start of a stream).
        let center = window.count - 1 - delay
        let decided = window[center], first = window[0].vertices, last = window[window.count - 1].vertices
        let span = Float(window.count - 1) / Float(fps)
        for g in 0..<groups.count {
            let lowest = decided.measures[g].lowest
            var d = SIMD2<Float>.zero
            for v in lowest { d += SIMD2(last[v].x - first[v].x, last[v].y - first[v].y) }
            let speed = lowest.isEmpty || span <= 0 ? 0 : (d * d).sum().squareRoot() / Float(lowest.count) / span
            let z = Double(decided.measures[g].height - decided.ground), v = Double(speed)
            let h = config.hysteresis(groups.classes[g])
            if !state[g] {
                if z < h.enterHeight && v < h.enterSpeed { state[g] = true; exitCount[g] = 0 }
            } else if z > h.exitHeight || v > h.exitSpeed {
                exitCount[g] += 1
                if exitCount[g] >= holdFrames { state[g] = false }
            } else { exitCount[g] = 0 }
        }
        return state
    }
}
