// Clip validation and auto-reject (DESIGN §9). Two stages:
//   human — pre-filter on the source motion and its contact analysis (environment support, penetration,
//           jerk, length, data sanity);
//   robot — metrics of a robot trajectory (ours or imported): tracking, limits, smoothness, foot contact,
//           ground penetration, self-collision, boundary stability.
// Every metric is recorded; rules from Configs/reject/*.json decide accept / reject / warn.

import Contact
import Core
import Dynamics
import Retarget

#if canImport(Glibc)
import Glibc
#endif

// MARK: - Rules

public struct ValidationRule: Sendable {
    public enum Severity: String, Sendable { case reject, warn }
    public var metric: String
    public var max: Double?
    public var min: Double?
    public var severity: Severity
    /// Values are rounded to this step before the comparison (and in the message), so a metric a hair over its
    /// bound (1.004 > 1) does not fail. Default: two significant digits of the bound (0.25 → 0.01, 1 → 0.1,
    /// 0.015 → 0.001, 200 → 10), so up to half a step (≤ 5 %) over passes; integer steps for bounds of 0 (frame counts).
    public var precision: Double

    public init(metric: String, max: Double?, min: Double?, severity: Severity, precision: Double? = nil) {
        self.metric = metric
        self.max = max
        self.min = min
        self.severity = severity
        let bound = abs(max ?? min ?? 0)
        self.precision = precision ?? (bound > 0 ? pow(10, (log10(bound)).rounded(.down) - 1) : 1)
    }

    /// `value` rounded to `precision`.
    public func rounded(_ value: Double) -> Double { (value / precision).rounded() * precision }

    /// `value` printed with the digits of `precision`, trailing zeros dropped (0.0163 against 0.015: "0.016").
    public func format(_ value: Double) -> String {
        let digits = Swift.max(0, Int((-log10(precision)).rounded(.up)))
        var scale = 1.0
        for _ in 0..<digits { scale *= 10 }
        let n = Int((value * scale).rounded()), whole = String(abs(n) / Int(scale))
        var fraction = digits == 0 ? "" : String(abs(n) % Int(scale))
        while fraction.count < digits { fraction = "0" + fraction }
        while fraction.hasSuffix("0") { fraction.removeLast() }
        return (n < 0 ? "-" : "") + whole + (fraction.isEmpty ? "" : "." + fraction)
    }
}

public struct RejectConfig: Sendable {
    public var human: [ValidationRule]
    public var robot: [ValidationRule]
    public var parameters: [String: Double]

    public func parameter(_ name: String, _ fallback: Double) -> Double { parameters[name] ?? fallback }

    public static func load(file: String) throws -> RejectConfig {
        let json = try JSONValue.load(file: file)
        func rules(_ key: String) throws -> [ValidationRule] {
            try (json[key]?.array ?? []).map { r in
                guard let metric = r["metric"]?.string else { throw JSONError(message: "\(file): rule without metric") }
                return ValidationRule(metric: metric, max: r["max"]?.double, min: r["min"]?.double,
                                      severity: ValidationRule.Severity(rawValue: r["severity"]?.string ?? "reject") ?? .reject,
                                      precision: r["precision"]?.double)
            }
        }
        return RejectConfig(human: try rules("human"), robot: try rules("robot"),
                            parameters: json["parameters"]?.object?.compactMapValues { $0.double } ?? [:])
    }
}

// MARK: - Report

public struct ValidationReport: Sendable {
    public enum Status: String, Sendable { case accepted, rejected }

    public var status: Status = .accepted
    /// "human" or "robot" for the first stage that rejected; nil when accepted.
    public var stage: String?
    public var reasons: [String] = []
    public var warnings: [String] = []
    public var metrics: [(String, Double)] = []
    /// Offending frame ranges (start, end exclusive) per metric, in the metric's frame rate.
    public var ranges: [String: [(Int, Int)]] = [:]
    /// Diagnostics, not used by rules: "tracking:<link>" mean position error, "self_collision:<a>|<b>"
    /// deepest penetration per link pair, "ground_penetration:<link>" deepest point per link.
    public var details: [String: Double] = [:]
    /// Per-frame torques, contact forces and residuals of the robot stage (not serialised in `json`).
    public var dynamics: DynamicsTrace?

    public init() {}

    public func metric(_ name: String) -> Double? { metrics.first { $0.0 == name }?.1 }

    /// Test hooks: set metrics directly and apply rules.
    public mutating func metricsForTesting(_ values: [(String, Double)]) { metrics = values }
    public mutating func applyForTesting(_ rules: [ValidationRule], stage: String) { apply(rules, stage: stage) }

    mutating func record(_ name: String, _ value: Double, ranges: [(Int, Int)] = []) {
        metrics.append((name, value))
        if !ranges.isEmpty { self.ranges[name] = Array(ranges.prefix(8)) }
    }

    /// Applies `rules` to the metrics recorded so far; the first rejecting stage wins.
    mutating func apply(_ rules: [ValidationRule], stage: String) {
        for rule in rules {
            guard let raw = metric(rule.metric), raw.isFinite else { continue }
            let value = rule.rounded(raw)
            var message: String?
            if let max = rule.max, value > max { message = "\(rule.metric) \(rule.format(value)) > \(rule.format(max))" }
            if let min = rule.min, value < min { message = "\(rule.metric) \(rule.format(value)) < \(rule.format(min))" }
            guard let message else { continue }
            switch rule.severity {
            case .reject:
                reasons.append("\(stage): \(message)")
                if status == .accepted { status = .rejected; self.stage = stage }
            case .warn:
                warnings.append("\(stage): \(message)")
            }
        }
    }

    public var json: JSON {
        .object([
            ("status", .string(status.rawValue)),
            ("stage", stage.map { .string($0) } ?? .null),
            ("reasons", .array(reasons.map { .string($0) })),
            ("warnings", .array(warnings.map { .string($0) })),
            ("metrics", .object(metrics.map { ($0.0, .number($0.1)) })),
            ("ranges", .object(ranges.sorted { $0.key < $1.key }.map { key, value in
                (key, .array(value.map { .array([.number(Double($0.0)), .number(Double($0.1))]) }))
            })),
            ("details", .object(details.sorted { $0.key < $1.key }.map { ($0.key, .number($0.value)) })),
        ])
    }
}

public func format(_ v: Double) -> String {
    let rounded = (v * 1000).rounded() / 1000
    return rounded == rounded.rounded() ? String(Int(rounded)) : String(rounded)
}

/// Runs of `true` as (start, end) pairs.
func runs(_ mask: [Bool]) -> [(Int, Int)] {
    var out: [(Int, Int)] = []
    var start: Int?
    for (i, m) in mask.enumerated() {
        if m, start == nil { start = i }
        if !m, let s = start { out.append((s, i)); start = nil }
    }
    if let s = start { out.append((s, mask.count)) }
    return out
}

func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return .nan }
    let s = values.sorted()
    return s[min(s.count - 1, Int(p * Double(s.count)))]
}

// MARK: - Validator

public struct Validator: Sendable {
    public var config: RejectConfig
    public var robot: RobotModel
    public var geometry: RobotGeometry
    /// Velocity limit per dof (rad/s).
    public var velocityLimits: [Double]
    /// Acceleration limit per dof (rad/s²) in the default pose: the effort-limited acceleration of the model
    /// unless the robot config overrides the joint (`acceleration_limits`). Motions are judged against the limit at
    /// each frame's own pose (`accelerationOverrides` where set).
    public var accelerationLimits: [Double]
    public var accelerationOverrides: [Double?]
    public var dynamics: RobotDynamics
    /// Ground friction coefficient from the robot config (reject config parameter `friction` overrides).
    public var robotFriction: Double

    public init(config: RejectConfig, robot: RobotModel, geometry: RobotGeometry, robotConfig: JSONValue?) {
        self.config = config
        self.robot = robot
        self.geometry = geometry
        let limits = robotConfig?["velocity_limits"]
        let fallback = limits?["default"]?.double ?? 20
        velocityLimits = robot.dofNames.map { limits?[$0]?.double ?? fallback }
        dynamics = RobotDynamics(robot: robot)
        let derived = dynamics.effortLimitedAcceleration()
        let overrides = robotConfig?["acceleration_limits"]
        let set = robot.dofNames.map { overrides?[$0]?.double }
        accelerationOverrides = set
        accelerationLimits = (0..<robot.dofCount).map { set[$0] ?? derived[$0] }
        robotFriction = robotConfig?["friction"]?.double ?? 0.8
    }

    // MARK: Human stage

    /// Pre-filter metrics of the source motion (`world` joint transforms per frame) and its contact analysis.
    public func checkHuman(world: [[Transform]], fps: Double, contacts: ContactTimeline, rootJoint: Int,
                           into report: inout ValidationReport) {
        let F = world.count, G = contacts.groupNames.count
        report.record("frames", Double(F))
        let envContact = (0..<contacts.frameCount).map { f in (0..<G).contains { contacts.envContact[f * G + $0] } }
        report.record("env_contact_frames", Double(envContact.filter { $0 }.count), ranges: runs(envContact))
        report.record("env_support_frames", Double(contacts.envSupport.filter { $0 }.count), ranges: runs(contacts.envSupport))
        let unexplained = (0..<contacts.frameCount).map { contacts.unsupported[$0] && !envContact[$0] && !contacts.envSupport[$0] }
        report.record("unsupported_frames", Double(unexplained.filter { $0 }.count), ranges: runs(unexplained))
        report.record("unsupported_seconds", Double(unexplained.filter { $0 }.count) / fps)
        report.record("penetration_frames", Double(contacts.penetration.filter { $0 }.count), ranges: runs(contacts.penetration))
        report.record("ground_tilt_deg", contacts.groundTilt)
        report.record("flight_frames", Double(contacts.flight.filter { $0 }.count))

        var nonFinite = 0
        for frame in world { for t in frame where !(t.translation.x.isFinite && t.translation.y.isFinite && t.translation.z.isFinite && t.rotation.w.isFinite) { nonFinite += 1 } }
        report.record("non_finite_values", Double(nonFinite))

        let root = world.map { $0[rootJoint].translation }
        let teleportSpeed = config.parameter("root_teleport_speed", 10)
        let teleport = (0..<F).map { f in f > 0 && length(root[f] - root[f - 1]) * fps > teleportSpeed }
        report.record("root_teleport_frames", Double(teleport.filter { $0 }.count), ranges: runs(teleport))
        report.record("root_jerk_window_max", rootJerk(root, fps: fps, window: config.parameter("root_jerk_window", 4)))
        report.apply(config.human, stage: "human")
    }

    /// Largest windowed mean root jerk (m/s³), after ~3 Hz low-pass smoothing (a 0.2 s moving average, as PHUMA
    /// filters the root at 3 Hz) and resampling to 30 Hz.
    func rootJerk(_ positions: [Vec3], fps: Double, window: Double) -> Double {
        let step = max(1, Int((fps / 30).rounded()))
        let radius = max(1, Int((0.1 * fps).rounded()))
        let smoothed = positions.indices.map { i -> Vec3 in
            let a = max(0, i - radius), b = min(positions.count - 1, i + radius)
            return (a...b).reduce(Vec3.zero) { $0 + positions[$1] } / Double(b - a + 1)
        }
        let samples = stride(from: 0, to: smoothed.count, by: step).map { smoothed[$0] }
        guard samples.count >= 5 else { return 0 }
        let h = Double(step) / fps
        let jerk = (1..<(samples.count - 2)).map { k in
            length(samples[k + 2] - 3 * samples[k + 1] + 3 * samples[k] - samples[k - 1]) / (h * h * h)
        }
        let width = max(1, Int(window / h))
        if jerk.count <= width { return jerk.reduce(0, +) / Double(jerk.count) }
        var best = 0.0
        for start in stride(from: 0, through: jerk.count - width, by: max(1, width / 4)) {
            best = max(best, jerk[start..<(start + width)].reduce(0, +) / Double(width))
        }
        return best
    }

    // MARK: Robot stage

    /// Robot metrics. `goals` (per trajectory frame, per mapping target) enables tracking metrics; `contacts`
    /// (human, any frame rate) enables the foot-contact metrics.
    public func checkRobot(_ trajectory: RobotTrajectory, goals: [[Transform]]?, setup: RetargetSetup?,
                           contacts: ContactTimeline?, into report: inout ValidationReport) {
        let T = trajectory.frameCount, n = robot.dofCount, fps = trajectory.fps
        let world = (0..<T).map { robot.forwardKinematics(root: trajectory.root[$0], q: trajectory.q[$0]) }
        let p = config.parameters

        // Tracking.
        if let goals, let setup, goals.count == T {
            var pos: [Double] = [], rot: [Double] = [], ends: [Double] = []
            for f in 0..<T {
                for (t, target) in setup.mapping.targets.enumerated() {
                    let link = world[f][setup.robotLink[t]], goal = goals[f][t]
                    if target.positionWeight > 0 {
                        let e = length(link.translation - goal.translation)
                        pos.append(e)
                        report.details["tracking:" + target.robot, default: 0] += e / Double(T)
                        if target.human.hasSuffix("Foot") || target.human.hasSuffix("Hand") { ends.append(e) }
                    }
                    if target.rotationWeight > 0 { rot.append(Quat.angle(link.rotation, goal.rotation) * 180 / .pi) }
                }
            }
            report.record("tracking_pos_mean", pos.reduce(0, +) / Double(max(pos.count, 1)))
            report.record("tracking_pos_p95", percentile(pos, 0.95))
            report.record("tracking_rot_p95_deg", percentile(rot, 0.95))
            report.record("tracking_end_effector_max", ends.max() ?? .nan)
        }

        // Limits.
        let limits = robot.links.compactMap { l in l.dofIndex.map { ($0, l.limits!) } }.sorted { $0.0 < $1.0 }.map(\.1)
        let limitHit = trajectory.q.map { q in (0..<n).contains { q[$0] < limits[$0].lower - 1e-3 || q[$0] > limits[$0].upper + 1e-3 } }
        report.record("limit_violation_frames", Double(limitHit.filter { $0 }.count), ranges: runs(limitHit))
        var maxRatio = 0.0
        var velocityHit = [Bool](repeating: false, count: T)
        var jumps = [Bool](repeating: false, count: T)
        var acc: [Double] = [], jerk: [Double] = []
        // A jump is a single step far beyond what the joint can do (`joint_jump` x its velocity limit), so the same at
        // any output rate; a fixed step in rad (0.35 rad) was 17.5 rad/s at 50 Hz, below the velocity limits.
        let jump = p["joint_jump"] ?? 1.5
        for f in 1..<max(T, 1) {
            for j in 0..<n {
                let dq = trajectory.q[f][j] - trajectory.q[f - 1][j]
                let ratio = abs(dq) * fps / velocityLimits[j]
                maxRatio = max(maxRatio, ratio)
                if ratio > 1 { velocityHit[f] = true }
                if ratio > jump { jumps[f] = true }
                if f >= 2 {
                    let a = (trajectory.q[f][j] - 2 * trajectory.q[f - 1][j] + trajectory.q[f - 2][j]) * fps * fps
                    acc.append(abs(a))
                    if f >= 3 {
                        let jk = (trajectory.q[f][j] - 3 * trajectory.q[f - 1][j] + 3 * trajectory.q[f - 2][j] - trajectory.q[f - 3][j]) * fps * fps * fps
                        jerk.append(abs(jk))
                    }
                }
            }
        }
        report.record("velocity_violation_frames", Double(velocityHit.filter { $0 }.count), ranges: runs(velocityHit))
        report.record("velocity_ratio_max", maxRatio)
        report.record("joint_jump_frames", Double(jumps.filter { $0 }.count), ranges: runs(jumps))
        report.record("joint_acc_p99", percentile(acc, 0.99))
        report.record("joint_jerk_p99", percentile(jerk, 0.99))

        let linear = p["root_teleport_speed"] ?? 10, angular = p["root_teleport_angular_speed"] ?? 20
        let teleport = (0..<T).map { f -> Bool in
            guard f > 0 else { return false }
            let v = length(trajectory.root[f].translation - trajectory.root[f - 1].translation) * fps
            let w = Quat.angle(trajectory.root[f].rotation, trajectory.root[f - 1].rotation) * fps
            return v > linear || w > angular
        }
        report.record("root_teleport_frames", Double(teleport.filter { $0 }.count), ranges: runs(teleport))

        // Ground penetration over every link surface point, and the lowest sole point per foot.
        var deepest = 0.0
        var penetrating = [Bool](repeating: false, count: T)
        var sole = [[Double]](repeating: [], count: geometry.soles.count)          // lowest sole height
        var solePoints = [[[Vec3]]](repeating: [], count: geometry.soles.count)    // all sole points per frame
        for f in 0..<T {
            for (l, points) in geometry.linkPoints.enumerated() {
                for point in points {
                    let z = world[f][l].apply(point).z
                    if -z > 0.005 { report.details["ground_penetration:" + robot.links[l].name] = max(report.details["ground_penetration:" + robot.links[l].name] ?? 0, -z) }
                    if -z > deepest { deepest = -z }
                    if z < -0.01 { penetrating[f] = true }
                }
            }
            for (s, foot) in geometry.soles.enumerated() {
                let points = foot.points.map { world[f][foot.link].apply($0) }
                sole[s].append(points.map(\.z).min() ?? world[f][foot.link].translation.z)
                solePoints[s].append(points)
            }
        }
        report.record("ground_penetration_max", deepest)
        report.record("ground_penetration_frames", Double(penetrating.filter { $0 }.count), ranges: runs(penetrating))

        // Heel corners of each sole (behind the foot's mid-length), as the solver's stance lock splits them.
        let soleHeel = geometry.soles.map { foot in
            let mid = foot.points.map(\.x).reduce(0, +) / Double(max(foot.points.count, 1))
            return foot.points.map { $0.x < mid }
        }
        // Foot contact quality, against the human's foot contacts on the same side.
        if let contacts {
            var skate: [Double] = [], float: [Double] = []
            var skating = [Bool](repeating: false, count: T)
            let skateSpeed = p["skate_speed"] ?? 0.1
            let skateHeight = p["skate_contact_height"] ?? 0.02
            let plantedSpeed = p["skate_planted_speed"] ?? 0.1
            // A part counts as planted from `skate_settle` after its contact begins: on the first frame of a heel
            // strike the robot foot is still arriving (0.4-1.4 cm per 20 ms frame on jogs), one frame per stance,
            // and a jogging stance is only ~12 frames, so these transients alone set the 95th percentile.
            let settle = max(0, Int(((p["skate_settle"] ?? 0.04) * contacts.fps).rounded()))
            for (s, side) in ["left", "right"].enumerated() where s < geometry.soles.count {
                let groups = ["\(side)_heel", "\(side)_toe"].compactMap { contacts.groupNames.firstIndex(of: $0) }
                for f in 1..<max(T, 1) {
                    let source = min(contacts.frameCount - 1, Int((Double(f) / fps * contacts.fps).rounded()))
                    guard groups.contains(where: { contacts.contact(source, $0) }), !solePoints[s][f].isEmpty else { continue }
                    // Skating is judged where the person's foot is planted: a touching heel or toe slower than
                    // `skate_planted_speed`. A dragged foot (crawling, sliding steps) may drag on the robot too.
                    let plantedGroups = groups.indices.filter { k in
                        contacts.contact(source, groups[k]) && Double(contacts.speed[source * contacts.groupNames.count + groups[k]]) < plantedSpeed
                            && (1...max(1, settle)).allSatisfy { settle == 0 || (source - $0 >= 0 && contacts.contact(source - $0, groups[k])) }
                    }
                    float.append(max(0, sole[s][f]))
                    // Sliding needs the sole on the ground: a robot foot in the air while the person's touches (e.g.
                    // a crawl carried on the knees) is measured by foot_float, not counted as skating.
                    guard sole[s][f] < skateHeight, !plantedGroups.isEmpty else { continue }
                    // Slide of the robot's part that matches a planted part of the person's foot (heel or toe
                    // corners, as the solver locks them): the centroid of its corners on the ground, the slower part
                    // if both are planted. The lowest corner alone, or the whole sole, moved whenever the foot pivoted
                    // on the ball with the heel swinging (turning on a jog or walk) while the person's planted toe
                    // stayed put: skate excess up to 0.12 on clips without any slide.
                    var slide = Double.infinity
                    for k in plantedGroups {
                        let part = solePoints[s][f].indices.filter { soleHeel[s][$0] == (k == 0) && solePoints[s][f][$0].z < sole[s][f] + skateHeight }
                        guard !part.isEmpty else { continue }
                        var d = part.reduce(Vec3.zero) { $0 + solePoints[s][f][$1] - solePoints[s][f - 1][$1] } / Double(part.count)
                        d.z = 0
                        slide = min(slide, length(d))
                    }
                    guard slide.isFinite else { continue }
                    let v = slide * fps
                    skate.append(v)
                    if v > skateSpeed { skating[f] = true }
                }
            }
            report.record("foot_skate_p95", percentile(skate, 0.95))
            // The source's own foot slide while in contact (e.g. dragged toes when crawling): the retargeter is
            // judged on what it adds on top.
            var humanSkate: [Double] = []
            for g in ["left_heel", "left_toe", "right_heel", "right_toe"].compactMap({ contacts.groupNames.firstIndex(of: $0) }) {
                for f in 0..<contacts.frameCount where contacts.contact(f, g) {
                    let v = Double(contacts.speed[f * contacts.groupNames.count + g])
                    if v < plantedSpeed { humanSkate.append(v) }   // same frames as the robot's (planted feet)
                }
            }
            let human = percentile(humanSkate, 0.95)
            report.record("human_foot_skate_p95", human)
            report.record("foot_skate_excess", max(0, percentile(skate, 0.95) - (human.isFinite ? human : 0)))
            report.record("foot_skate_frames", Double(skating.filter { $0 }.count), ranges: runs(skating))
            report.record("foot_float_p95", percentile(float, 0.95))
        }

        // Self-collision between non-adjacent capsules.
        var worst = 0.0
        var colliding = [Bool](repeating: false, count: T)
        for f in 0..<T {
            for (a, b) in geometry.collisionPairs {
                let d = geometry.penetration(a, world[f][a], b, world[f][b])
                if d > 0.005 {
                    let key = "self_collision:" + robot.links[a].name + "|" + robot.links[b].name
                    report.details[key] = max(report.details[key] ?? 0, d)
                }
                if d > worst { worst = d }
                if d > 0.005 { colliding[f] = true }
            }
        }
        report.record("self_collision_max", worst)
        report.record("self_collision_frames", Double(colliding.filter { $0 }.count), ranges: runs(colliding))

        // Boundaries: upright start, statically stable start / end (CoM over the planted soles).
        if T > 0 {
            report.record("start_tilt_deg", acos(min(1, trajectory.root[0].rotation.rotate(Vec3(0, 0, 1)).z)) * 180 / .pi)
            report.record("start_stability_margin", stabilityMargin(world[0]))
            report.record("end_stability_margin", stabilityMargin(world[T - 1]))
        }
        dynamicsMetrics(trajectory, world: world, into: &report)
        report.apply(config.robot, stage: "robot")
    }

    /// Signed distance (m) of the ground projection of the robot CoM inside the support polygon of all robot
    /// surface points within `contact_points_height` of the ground (feet, knees, hands …; positive inside);
    /// NaN when fewer than three points touch.
    func stabilityMargin(_ world: [Transform]) -> Double {
        var mass = 0.0, com = Vec3.zero
        for (l, link) in robot.links.enumerated() {
            guard let inertial = link.inertial else { continue }
            mass += inertial.mass
            com += inertial.mass * world[l].apply(inertial.centerOfMass)
        }
        guard mass > 0 else { return .nan }
        com /= mass
        let height = config.parameter("contact_points_height", 0.03)
        var support: [SIMD2<Double>] = []
        for (l, points) in geometry.linkPoints.enumerated() {
            for point in points {
                let w = world[l].apply(point)
                if w.z < height { support.append(SIMD2(w.x, w.y)) }
            }
        }
        for foot in geometry.soles {
            for point in foot.points {
                let w = world[foot.link].apply(point)
                if w.z < height { support.append(SIMD2(w.x, w.y)) }
            }
        }
        guard support.count >= 3 else { return .nan }
        return signedDistanceInside(SIMD2(com.x, com.y), hull: convexHull2D(support))
    }
}

func convexHull2D(_ input: [SIMD2<Double>]) -> [SIMD2<Double>] {
    let pts = input.sorted { $0.x < $1.x || ($0.x == $1.x && $0.y < $1.y) }
    guard pts.count > 2 else { return pts }
    func cross(_ o: SIMD2<Double>, _ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
    var lower: [SIMD2<Double>] = [], upper: [SIMD2<Double>] = []
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

/// Positive inside a counter-clockwise convex polygon: distance to the nearest edge; negative outside.
func signedDistanceInside(_ p: SIMD2<Double>, hull: [SIMD2<Double>]) -> Double {
    guard hull.count >= 3 else { return .nan }
    var inside = true
    var nearest = Double.infinity
    for i in 0..<hull.count {
        let a = hull[i], b = hull[(i + 1) % hull.count]
        let ab = b - a
        if ab.x * (p.y - a.y) - ab.y * (p.x - a.x) < 0 { inside = false }
        let t = max(0, min(1, ((p - a) * ab).sum() / max((ab * ab).sum(), 1e-12)))
        let d = p - (a + t * ab)
        nearest = min(nearest, (d * d).sum().squareRoot())
    }
    return inside ? nearest : -nearest
}
