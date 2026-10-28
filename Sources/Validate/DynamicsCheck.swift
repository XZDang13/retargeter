// Dynamics stage of the robot validation (DESIGN §6, §9): per frame, the ground contact forces that best
// explain the motion (friction pyramids at the robot's own ground-touching surface points), the joint
// torques they imply (floating-base RNEA) and whatever wrench they cannot explain.
//
// Contacts are geometric — surface points touching the ground (below `dynamics_contact_touch`), or within
// `dynamics_contact_height` of it and slower than `dynamics_contact_speed` — so any trajectory (ours,
// imported) is judged the same way.

import Core
import Dynamics
import Retarget

/// Per-frame dynamics of a validated trajectory (exported next to the motion).
public struct DynamicsTrace: Sendable {
    public var tau: [[Double]]          // [frame][dof] N·m
    public var contactForce: [[Vec3]]   // [frame][link] net ground force on the link, N
    public var residual: [Double]       // [frame] unexplained wrench, dimensionless (DESIGN §9)
    public var zmp: [Vec3?]             // [frame] ZMP of the motion on the ground (nil in flight)
}

extension Validator {
    /// Ground contact points of one frame: per link, the extreme (±x, ±y) surface points near the ground.
    func contactPoints(_ world: [Transform], previous: [Transform]?, fps: Double) -> [ContactPoint] {
        let height = config.parameter("dynamics_contact_height", 0.02)
        let speed = config.parameter("dynamics_contact_speed", 0.5)
        let touch = config.parameter("dynamics_contact_touch", 0.005)
        var out: [ContactPoint] = []
        for (l, points) in geometry.linkPoints.enumerated() {
            var near: [(local: Vec3, world: Vec3)] = []
            for p in points {
                let w = world[l].apply(p)
                if w.z < height { near.append((p, w)) }
            }
            if let foot = geometry.soles.first(where: { $0.link == l }) {
                for p in foot.points {
                    let w = world[l].apply(p)
                    if w.z < height { near.append((p, w)) }
                }
            }
            guard !near.isEmpty else { continue }
            var chosen: [(index: Int, corner: Int)] = []
            for (corner, (sx, sy)) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)].enumerated() {
                let k = near.indices.max { sx * near[$0].world.x + sy * near[$0].world.y < sx * near[$1].world.x + sy * near[$1].world.y }!
                if !chosen.contains(where: { $0.index == k }) { chosen.append((k, corner)) }
            }
            for (k, corner) in chosen {
                // A point skimming above the ground only counts when slow (a swing foot is not a contact); one
                // touching the ground counts even while sliding.
                if let previous, near[k].world.z > touch {
                    var d = near[k].world - previous[l].apply(near[k].local)
                    d.z = 0
                    if length(d) * fps > speed { continue }
                }
                var p = near[k].world
                p.z = max(p.z, 0)
                out.append(ContactPoint(link: l, point: p, id: l * 4 + corner))
            }
        }
        return out
    }

    func dynamicsMetrics(_ trajectory: RobotTrajectory, world: [[Transform]], into report: inout ValidationReport) {
        let T = trajectory.frameCount, n = robot.dofCount, fps = trajectory.fps
        guard T >= 3, dynamics.totalMass > 0 else { return }
        let solver = ContactForceSolver(dynamics: dynamics, friction: friction,
                                        momentScale: config.parameter("dynamics_moment_scale", 0.1))
        let states = TrajectoryDerivatives.states(root: trajectory.root, q: trajectory.q, fps: fps,
                                                  smoothing: config.parameter("dynamics_smoothing", 0.02))
        let mg = dynamics.totalMass * 9.81
        // Velocity smoothing spreads impacts (landing, take-off) and the shift of the centre of pressure at heel
        // strike / toe-off over ±2σ, so contacts are extended by as much: each contact point (link and patch
        // corner) missing at a frame is borrowed from the nearest frame within that radius that has it. (Borrowing
        // only for links without any contact left a heel-only polygon at heel strike while the smoothed ZMP had
        // moved on: 40-100 ms "ZMP outside" on visibly stable walking.)
        let raw = (0..<T).map { contactPoints(world[$0], previous: $0 > 0 ? world[$0 - 1] : nil, fps: fps) }
        let radius = Int((2 * config.parameter("dynamics_smoothing", 0.02) * fps).rounded())
        let contacts = (0..<T).map { f -> [ContactPoint] in
            var contacts = raw[f]
            guard radius > 0 else { return contacts }
            var present = Set(contacts.map(\.id))
            for d in 1...radius {
                for g in [f - d, f + d] where g >= 0 && g < T {
                    let borrowed = raw[g].filter { !present.contains($0.id) }
                    contacts += borrowed
                    present.formUnion(borrowed.map(\.id))
                }
            }
            return reduceContacts(contacts)
        }
        // Contact forces per frame: chunks of frames in parallel, each warm-started along its frames.
        let chunk = 64
        let solutions = parallelMap((T + chunk - 1) / chunk) { c -> [ContactSolution] in
            var warm: [Int] = []
            return (c * chunk..<min(T, (c + 1) * chunk)).map { f in
                let sol = solver.solve(states[f], contacts: contacts[f], warmStart: warm)
                warm = sol.activeEdges
                return sol
            }
        }.flatMap { $0 }
        var trace = DynamicsTrace(tau: [], contactForce: [], residual: [], zmp: [])
        var ratio = [[Double]](repeating: [], count: n)   // [dof][frame] |tau| / effort
        var outside = [Double](repeating: 0, count: T)
        var peakForce = 0.0
        for f in 0..<T {
            let sol = solutions[f]
            trace.tau.append(sol.tau)
            var perLink = [Vec3](repeating: .zero, count: robot.links.count)
            for force in sol.forces { perLink[force.link] += force.force }
            trace.contactForce.append(perLink)
            peakForce = max(peakForce, length(perLink.reduce(.zero, +)) / mg)
            trace.residual.append(sol.residual(mg: mg, momentScale: solver.momentScale))
            for j in 0..<n { ratio[j].append(abs(sol.tau[j]) / dynamics.effort[j]) }
            // ZMP of the gravito-inertial wrench on z = 0 (meaningful when the ground carries the weight).
            let F = sol.requiredForce, N = sol.requiredMoment, c = sol.com
            if F.z > 0.5 * mg {
                let zmp = Vec3(c.x - (N.y + c.z * F.x) / F.z, c.y + (N.x - c.z * F.y) / F.z, 0)
                trace.zmp.append(zmp)
                outside[f] = distanceOutside(SIMD2(zmp.x, zmp.y), contacts[f].map { SIMD2($0.point.x, $0.point.y) })
            } else {
                trace.zmp.append(nil)
            }
        }
        func window(_ name: String, _ seconds: Double) -> Int { max(1, Int((config.parameter(name, seconds) * fps).rounded())) }
        let torqueWindow = window("dynamics_torque_window", 0.05), residualWindow = window("dynamics_residual_window", 0.1)
        let zmpWindow = window("dynamics_zmp_window", 0.3)

        // Supported frames and flight (no ground contact) apart: while supported, an unexplained wrench is force or
        // moment the contacts cannot give; in flight, a CoM that does not fall ballistically. The latter is partly
        // inherent to retargeting — scaled down uniformly (~0.8), a person's flight falls at ~0.8 g at robot scale —
        // and costs a policy only a shifted landing, so it has its own bound (`dyn_flight_residual_max`).
        let residualLimit = config.parameter("dynamics_residual_threshold", 0.3)
        let flight = contacts.map(\.isEmpty)
        let supported = zip(trace.residual, flight).map { $1 ? 0 : $0 }, airborne = zip(trace.residual, flight).map { $1 ? $0 : 0 }
        report.record("dyn_residual_max", sustained(supported, window: residualWindow),
                      ranges: runs(supported.map { $0 > residualLimit }))
        report.record("dyn_flight_residual_max", sustained(airborne, window: residualWindow),
                      ranges: runs(airborne.map { $0 > residualLimit }))
        report.record("dyn_residual_p95", percentile(trace.residual, 0.95))
        report.record("zmp_outside_max", sustained(outside, window: zmpWindow), ranges: runs(outside.map { $0 > 0.05 }))
        var worst = 0.0, peak = 0.0
        var over = [Bool](repeating: false, count: T)
        for j in 0..<n {
            let s = sustained(ratio[j], window: torqueWindow)
            worst = max(worst, s)
            peak = max(peak, ratio[j].max() ?? 0)
            for f in 0..<T where ratio[j][f] > 1 { over[f] = true }
            if s > 0.8 { report.details["torque:" + robot.dofNames[j]] = s }
        }
        report.record("torque_ratio_max", worst, ranges: runs(over))
        // Joint accelerations (rad/s², from the same smoothed derivatives) against the acceleration limit at each
        // frame's pose, held for the torque window. (The default pose's limit, legs straight, is the lowest for the
        // hips: with the knee bent at a jump's take-off the leg turns more easily, and the fixed limit rejected
        // take-offs at 1.1x it.)
        let poseLimits = parallelMap(T) { f in dynamics.effortLimitedAcceleration(states[f]) }
        var accWorst = 0.0, ratioWorst = 0.0
        var accOver = [Bool](repeating: false, count: T)
        for j in 0..<n {
            let acc = states.map { abs($0.qdd[j]) }
            let limit = (0..<T).map { accelerationOverrides[j] ?? poseLimits[$0][j] }
            let held = sustained(acc, window: torqueWindow)
            report.details["joint_acc:" + robot.dofNames[j]] = held
            accWorst = max(accWorst, held)
            let ratio = sustained((0..<T).map { acc[$0] / limit[$0] }, window: torqueWindow)
            ratioWorst = max(ratioWorst, ratio)
            if ratio > 0.8 { report.details["joint_acc_ratio:" + robot.dofNames[j]] = ratio }
            for f in 0..<T where acc[f] > limit[f] { accOver[f] = true }
        }
        report.record("joint_acc_max", accWorst)
        report.record("joint_acc_ratio_max", ratioWorst, ranges: runs(accOver))
        report.record("torque_ratio_peak", peak)
        report.record("torque_over_limit_frames", Double(over.filter { $0 }.count))
        report.record("contact_force_peak", peakForce)
        report.dynamics = trace
    }

    var friction: Double { config.parameter("friction", robotFriction) }
}

/// At most `limit` contact points: beyond it, the corners of the support polygon plus one point per touching
/// link (a robot lying in, or sinking through, the floor touches with dozens of links; the inner points add no
/// support and make the force solve slow).
func reduceContacts(_ contacts: [ContactPoint], limit: Int = 24) -> [ContactPoint] {
    guard contacts.count > limit else { return contacts }
    let hull = Set(convexHull2D(contacts.map { SIMD2($0.point.x, $0.point.y) }).map { [$0.x, $0.y] })
    var out: [ContactPoint] = []
    var links = Set<Int>()
    for c in contacts where hull.contains([c.point.x, c.point.y]) {
        out.append(c)
        links.insert(c.link)
    }
    for c in contacts where !links.contains(c.link) {
        out.append(c)
        links.insert(c.link)
    }
    return out
}

/// Largest value held for at least `window` consecutive samples (max over windows of the window minimum).
func sustained(_ x: [Double], window: Int) -> Double {
    guard !x.isEmpty else { return .nan }
    let w = min(window, x.count)
    var best = -Double.infinity
    for s in 0...(x.count - w) {
        var m = Double.infinity
        for i in s..<(s + w) { m = min(m, x[i]) }
        best = max(best, m)
    }
    return best
}

/// Distance of `p` outside the convex hull of `points` (0 inside; distance to the segment / point for
/// degenerate supports; 0 without points — flight is judged by the residual instead).
func distanceOutside(_ p: SIMD2<Double>, _ points: [SIMD2<Double>]) -> Double {
    guard !points.isEmpty else { return 0 }
    let hull = convexHull2D(points)
    if hull.count >= 3 { return max(0, -signedDistanceInside(p, hull: hull)) }
    func len(_ v: SIMD2<Double>) -> Double { (v * v).sum().squareRoot() }
    if hull.count == 2 {
        let a = hull[0], ab = hull[1] - hull[0]
        let t = max(0, min(1, ((p - a) * ab).sum() / max((ab * ab).sum(), 1e-12)))
        return len(p - (a + t * ab))
    }
    return len(p - hull[0])
}
