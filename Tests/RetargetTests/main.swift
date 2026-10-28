// Realtime (lag = 0) retargeting: solver recovery on robot-generated goals, real SOMA clips, CUDA/CPU
// parity and batching.

import Compute
import Core
import Dynamics
import Kinematics
import MotionIO
import Retarget
import Validate

#if canImport(Glibc)
import Glibc
#endif

let g1 = repositoryRoot + "/assets/robots/unitree_g1/"
let robots = [("g1_29dof", try RobotUSDLoader.load(directory: g1 + "g1_29_dof", name: "g1_29dof")),
              ("g1_23dof", try RobotUSDLoader.load(directory: g1 + "g1_23_dof", name: "g1_23dof"))]
let surface = try BodySurface.load(file: repositoryRoot + "/assets/body_models/soma/surface_lo.npz")
let bvhRoot = "/home/troja/Downloads/soma_uniform/bvh/"
let clips = ["210531/walk_forward_amateur_001__A001.bvh", "230322/sweep_floor_side_walk_R_004__A298.bvh",
             "220713/kneeling_loop_001__A021.bvh", "221010/jump_ff_180_R_001__A045.bvh"]
let motions = try clips.map { try BVH.load(file: bvhRoot + $0) }
let cuda = try ComputeDevice.makeDefault(backend: .cuda)

extension String {
    func padding(_ n: Int) -> String { count >= n ? self : self + String(repeating: " ", count: n - count) }
}

func mapping(_ robot: String) throws -> RetargetMapping {
    try RetargetMapping.load(file: repositoryRoot + "/Configs/mappings/soma77_\(robot).json")
}

/// Unweighted tracking errors of a solution against its goals, for targets with nonzero weights.
func trackingErrors(_ setup: RetargetSetup, _ traj: RobotTrajectory, _ goals: [[Transform]]) -> (pos: [Double], rot: [Double]) {
    var pos: [Double] = [], rot: [Double] = []
    for f in 0..<traj.frameCount {
        let world = setup.robot.forwardKinematics(root: traj.root[f], q: traj.q[f])
        for (t, target) in setup.mapping.targets.enumerated() {
            let link = world[setup.robotLink[t]], goal = goals[f][t]
            if target.positionWeight > 0 { pos.append(length(link.translation - goal.translation)) }
            guard target.rotationWeight > 0 else { continue }
            // As the solver weighs it: the share `twist` of the error about the link's twist axis (hands: 0.25).
            var r = goal.rotation * link.rotation.conjugate
            if r.w < 0 { r = Quat(w: -r.w, v: -r.v) }
            let s = length(r.v), e = s > 1e-12 ? r.v * (2 * atan2(s, r.w) / s) : Vec3(0, 0, 0)
            let a = link.rotation.rotate(target.twistAxis)
            rot.append(length(e - (1 - target.twist) * dot(a, e) * a))
        }
    }
    return (pos, rot)
}

func percentile(_ v: [Double], _ p: Double) -> Double {
    let s = v.sorted()
    return s.isEmpty ? 0 : s[min(s.count - 1, Int(p * Double(s.count)))]
}

func limitViolations(_ robot: RobotModel, _ traj: RobotTrajectory) -> Int {
    let limits = robot.links.compactMap { l in l.dofIndex.map { ($0, l.limits!) } }
    return traj.q.reduce(0) { n, q in n + limits.filter { q[$0.0] < $0.1.lower - 1e-5 || q[$0.0] > $0.1.upper + 1e-5 }.count }
}

if getenv("SUPPORT_DIAG") != nil {
    // The force-free solver's view of CoM support (point-mass wrench over frames t ± K, ZMP against the
    // solver's contact polygon) on the realtime warm start, against the validator's.
    let (_, robot) = robots[0]
    let directory = g1 + "g1_29_dof"
    let geometry = try RobotGeometry(robot: robot, pointsFile: directory + "/mesh_points.npz")
    let validator = try Validator.load(repository: repositoryRoot, robot: robot, robotDirectory: directory)
    let motion = try BVH.load(file: bvhRoot + (getenv("SUPPORT_CLIP").map { String(cString: $0) } ?? "230322/body_check_001__A296.bvh"))
    let offline = try OfflineRetargeter(device: cuda, robot: robot, mapping: try mapping("g1_29dof"), surface: surface,
                                        skeleton: motion.skeleton, geometry: geometry)
    let prepared = try offline.prepare([motion])
    let start = try offline.base.solve(prepared)
    let inputs = offline.inputs(prepared, initial: start)
    let traj = start[0].trajectory
    let points = offline.solver.contactPointsForDiagnostics(clips: inputs, trajectories: [traj])
    let dyn = RobotDynamics(robot: robot)
    let K = 3, rate = traj.fps / Double(K)
    let coms = (0..<traj.frameCount).map { f -> [Vec3] in
        let w = robot.forwardKinematics(root: traj.root[f], q: traj.q[f])
        return robot.links.indices.map { w[$0].apply(robot.links[$0].inertial?.centerOfMass ?? .zero) }
    }
    var outside: [Double] = [], noContact = 0, contactFrames = 0
    for t in K..<(traj.frameCount - K) {
        var c = Vec3.zero
        for i in robot.links.indices { c += dyn.mass[i] * coms[t][i] }
        c /= dyn.totalMass
        var F = Vec3.zero, N = Vec3.zero
        for i in robot.links.indices {
            let A = dyn.mass[i] * ((coms[t + K][i] - 2 * coms[t][i] + coms[t - K][i]) * rate * rate - standardGravity)
            F += A
            N += cross(coms[t][i] - c, A)
        }
        let pts = (0..<24).map { points[t * 24 + $0] }.filter { $0.w > 0 }.map { SIMD2(Double($0.x), Double($0.y)) }
        if pts.isEmpty { noContact += 1; continue }
        contactFrames += 1
        guard F.z > 0.3 * dyn.totalMass * 9.81 else { continue }
        let p = SIMD2(c.x - (N.y + c.z * F.x) / F.z, c.y + (N.x - c.z * F.y) / F.z)
        // Distance outside the convex hull (gift wrapping as in the kernel is not needed here: brute force).
        var inside = true, best = Double.infinity
        let hull = pts   // all points; distance to segments between all pairs, inside via the convex hull
        func cross2(_ o: SIMD2<Double>, _ a: SIMD2<Double>, _ b: SIMD2<Double>) -> Double { (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x) }
        let sorted = hull.sorted { $0.x < $1.x || ($0.x == $1.x && $0.y < $1.y) }
        var lower: [SIMD2<Double>] = [], upper: [SIMD2<Double>] = []
        for q in sorted { while lower.count >= 2 && cross2(lower[lower.count - 2], lower[lower.count - 1], q) <= 0 { lower.removeLast() }; lower.append(q) }
        for q in sorted.reversed() { while upper.count >= 2 && cross2(upper[upper.count - 2], upper[upper.count - 1], q) <= 0 { upper.removeLast() }; upper.append(q) }
        let poly = Array(lower.dropLast() + upper.dropLast())
        for e in 0..<poly.count {
            let a = poly[e], b = poly[(e + 1) % poly.count], ab = b - a
            if poly.count >= 3 && ab.x * (p.y - a.y) - ab.y * (p.x - a.x) < 0 { inside = false }
            let tt = max(0, min(1, ((p - a) * ab).sum() / max((ab * ab).sum(), 1e-12)))
            let d = p - (a + tt * ab)
            best = min(best, (d * d).sum().squareRoot())
        }
        if poly.count < 3 { inside = false }
        outside.append(inside ? 0 : best)
    }
    var report = ValidationReport()
    validator.checkRobot(start[0], setup: offline.setup, fps: 50, into: &report)
    let s = outside.sorted()
    print("  solver view: \(contactFrames) contact frames, \(noContact) without; ZMP outside in \(outside.filter { $0 > 0.01 }.count) of \(outside.count), p50 \(format(s[s.count / 2])) p90 \(format(s[s.count * 9 / 10])) max \(format(s.last!)) m")
    print("  validator: zmp_outside_max \(format(report.metric("zmp_outside_max")!)), residual \(format(report.metric("dyn_residual_max")!))")
    exit(0)
}

if let file = getenv("HANDPAIR_DIAG").map({ String(cString: $0) }) {
    // Active solver collision pairs (enclosing capsules within the margin) per frame of an exported trajectory,
    // with the separation direction, around HANDPAIR_TIME (s).
    let (_, robot) = robots[0]
    let geometry = try RobotGeometry(robot: robot, pointsFile: g1 + "g1_29_dof/mesh_points.npz")
    let setup = try RetargetSetup(mapping: try mapping("g1_29dof"), robot: robot, skeleton: motions[0].skeleton, humanReference: surface.referencePose!)
    let solver = try OfflineSolver(device: cuda, setup: setup, geometry: geometry)
    let outer = geometry.outerCapsules
    let m = try RobotMotionImport.npz(file: file)
    let center = Double(getenv("HANDPAIR_TIME").map { String(cString: $0) } ?? "0") ?? 0
    var maxActive = 0
    for f in 0..<m.root.count {
        let w = robot.forwardKinematics(root: m.root[f], q: m.q[f])
        var lines: [String] = []
        for (a, b) in solver.collisionPairsForDiagnostics {
            // The solver's gap: axis distance minus max(outer_a + median_b, median_a + outer_b); active below the band.
            let ca = outer[a]!, cb = outer[b]!
            let (pa, pb) = RobotGeometry.closestPoints(w[a].apply(ca.a), w[a].apply(ca.b), w[b].apply(cb.a), w[b].apply(cb.b))
            let required = max(ca.radius + (geometry.capsules[b]?.radius ?? 0), (geometry.capsules[a]?.radius ?? 0) + cb.radius)
            let d = required - length(pa - pb)
            guard d > -0.03 else { continue }
            if let only = getenv("HANDPAIR_LINK").map({ String(cString: $0) }),
               !robot.links[a].name.contains(only) && !robot.links[b].name.contains(only) { continue }
            let n = normalize(pa - pb)
            lines.append("\(robot.links[a].name)|\(robot.links[b].name) pen \(format(d)) n (\(format(n.x)), \(format(n.y)), \(format(n.z)))")
        }
        maxActive = max(maxActive, lines.count)
        let t = Double(f) / m.fps
        if abs(t - center) < 0.2 { print("  t=\(format(t)) active \(lines.count): " + lines.joined(separator: "; ")) }
    }
    print("max active pairs in a frame: \(maxActive) (solver rows: 24)")
    exit(0)
}

if getenv("CAPSULE_DIAG") != nil {
    let (_, robot) = robots[0]
    let geometry = try RobotGeometry(robot: robot, pointsFile: g1 + "g1_29_dof/mesh_points.npz")
    for (l, link) in robot.links.enumerated() where link.name.contains("wrist") || link.name.contains("elbow") || link.name.contains("torso") || link.name.contains("hip_roll") {
        let radii = [0.5, 0.75, 0.9, 1.0].map { RobotGeometry.fitCapsuleForDiagnostics(geometry.linkPoints[l], radiusPercentile: $0)?.radius ?? 0 }
        print("  \(link.name): radius p50 \(format(radii[0])) p75 \(format(radii[1])) p90 \(format(radii[2])) max \(format(radii[3]))")
    }
    exit(0)
}

if getenv("PROPORTION_DIAG") != nil {
    // Shoulder vs hip widths: robot (default pose, joint origins and surface) and human (reference pose).
    let (_, robot) = robots[0]
    let geometry = try RobotGeometry(robot: robot, pointsFile: g1 + "g1_29_dof/mesh_points.npz")
    let w = robot.forwardKinematics(root: robot.defaultRootPose, q: Array(repeating: 0, count: robot.dofCount))
    func y(_ name: String) -> Double { w[robot.linkIndex(named: name)!].translation.y }
    func outer(_ names: [String]) -> Double {
        names.flatMap { n -> [Double] in let l = robot.linkIndex(named: n)!; return geometry.linkPoints[l].map { abs(w[l].apply($0).y) } }.max()!
    }
    print("  robot: shoulder joints \(format(y("left_shoulder_pitch_link") - y("right_shoulder_pitch_link"))) m apart, hip joints \(format(y("left_hip_pitch_link") - y("right_hip_pitch_link")))")
    print("  robot: hip surface half-width \(format(outer(["left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "pelvis"]))), torso half-width \(format(outer(["torso_link"]))), upper arm inner surface \(format(geometry.linkPoints[robot.linkIndex(named: "left_shoulder_yaw_link")!].map { w[robot.linkIndex(named: "left_shoulder_yaw_link")!].apply($0).y }.min()!))")
    let skel = motions[0].skeleton
    let ref = surface.referencePose!
    func hy(_ name: String) -> Vec3 { ref[skel.jointIndex(named: name)!].translation }
    print("  human: shoulder (LeftArm) joints \(format(hy("LeftArm").y - hy("RightArm").y)) m apart, hip (LeftLeg) joints \(format(hy("LeftLeg").y - hy("RightLeg").y)), hips height \(format(hy("Hips").z))")
    let setup = try RetargetSetup(mapping: try mapping("g1_29dof"), robot: robot, skeleton: skel, humanReference: ref)
    for (t, target) in setup.mapping.targets.enumerated() where target.human.contains("Arm") || target.human.contains("Hand") || target.human == "LeftLeg" {
        print("  scale \(target.human) -> \(target.robot): \(format(setup.scales[t]))")
    }
    // Lateral goal positions: shoulders and hips in the human reference pose after the goal mapping.
    let goals = setup.goals(human: ref)
    for (t, target) in setup.mapping.targets.enumerated() where ["LeftArm", "LeftLeg", "LeftHand"].contains(target.human) && target.positionWeight > 0 {
        print("  goal \(target.robot) y \(format(goals[t].translation.y - goals[setup.rootTarget].translation.y)) (robot rest \(format(y(target.robot) - y("pelvis"))))")
    }
    exit(0)
}

if getenv("SCALE_DIAG") != nil {
    // Every target's goal in the human reference pose against the robot link in its matching reference pose
    // (mapping reference_pose, default root), both relative to the root; plus segment lengths.
    for (name, robot) in robots {
        let map = try mapping(name)
        let skel = motions[0].skeleton, ref = surface.referencePose!
        let setup = try RetargetSetup(mapping: map, robot: robot, skeleton: skel, humanReference: ref)
        let q = robot.dofNames.map { map.referencePose?[$0] ?? 0 }
        let w = robot.forwardKinematics(root: robot.defaultRootPose, q: q)
        // Ground the human reference: feet on z = 0.
        let minZ = ref.map(\.translation.z).min()!
        let refGrounded = ref.map { Transform(rotation: $0.rotation, translation: $0.translation - Vec3(0, 0, minZ)) }
        let goals = setup.goals(human: refGrounded)
        let rootGoal = goals[setup.rootTarget].translation, rootRobot = w[setup.robotLink[setup.rootTarget]].translation
        print("== \(name): root goal z \(format(rootGoal.z)) robot \(format(rootRobot.z))")
        for (t, target) in map.targets.enumerated() where target.positionWeight > 0 {
            let g = goals[t].translation - rootGoal, r = w[setup.robotLink[t]].translation - rootRobot
            let d = g - r
            print("  \(target.human.padding(13)) -> \(target.robot.padding(26)) goal (\(format(g.x)), \(format(g.y)), \(format(g.z)))  robot (\(format(r.x)), \(format(r.y)), \(format(r.z)))  diff \(format(length(d))) (\(format(d.x)), \(format(d.y)), \(format(d.z)))  scale \(format(setup.scales[t]))")
        }
    }
    exit(0)
}

if getenv("PAIR_DIAG") != nil {
    let (_, robot) = robots[0]
    let geometry = try RobotGeometry(robot: robot, pointsFile: g1 + "g1_29_dof/mesh_points.npz")
    let setup = try RetargetSetup(mapping: try mapping("g1_29dof"), robot: robot, skeleton: motions[0].skeleton, humanReference: surface.referencePose!)
    let solver = try OfflineSolver(device: cuda, setup: setup, geometry: geometry)
    let names = { (p: (Int, Int)) in robot.links[p.0].name + "|" + robot.links[p.1].name }
    print("pairs: validator \(geometry.collisionPairs.count), solver \(solver.pairCount)")
    let outer = geometry.outerCapsules
    let rest = robot.forwardKinematics(root: robot.defaultRootPose, q: Array(repeating: 0, count: robot.dofCount))
    for p in geometry.collisionPairs where names(p).contains("wrist") && names(p).contains("left") && names(p).contains("right") {
        let d = RobotGeometry.penetration(outer[p.0]!, rest[p.0], outer[p.1]!, rest[p.1])
        print("pair \(names(p)) rest outer penetration \(d)")
    }
    let file = repositoryRoot + "/build/compare/t_a60/rejected/230322/body_check_001__A296.npz"
    let m = try RobotMotionImport.npz(file: file)
    var maxActive = 0
    for f in 0..<m.root.count {
        let w = robot.forwardKinematics(root: m.root[f], q: m.q[f])
        var active = 0
        for p in geometry.collisionPairs {
            let d = RobotGeometry.penetration(outer[p.0]!, w[p.0], outer[p.1]!, w[p.1])
            if d > -0.01 { active += 1 }
        }
        maxActive = max(maxActive, active)
    }
    print("max active outer pairs in a frame: \(maxActive)")
}

if getenv("COLLIDE_DIAG") != nil {
    let (_, robot) = robots[0]
    let geometry = try RobotGeometry(robot: robot, pointsFile: g1 + "g1_29_dof/mesh_points.npz")
    for (l, c) in geometry.capsules.enumerated() {
        if let c { print("capsule \(robot.links[l].name): radius \(c.radius) length \(length(c.b - c.a)) points \(geometry.linkPoints[l].count)") }
    }
    print("pairs \(geometry.collisionPairs.count)")
    let r = try HumanRetargeter(device: cuda, robot: robot, mapping: try mapping("g1_29dof"), surface: surface, skeleton: motions[0].skeleton)
    let traj = try r.retarget([motions[0]])[0].trajectory
    var worst: [String: Double] = [:]
    for f in stride(from: 0, to: traj.frameCount, by: 5) {
        let w = robot.forwardKinematics(root: traj.root[f], q: traj.q[f])
        for (a, b) in geometry.collisionPairs {
            let d = RobotGeometry.penetration(geometry.capsules[a]!, w[a], geometry.capsules[b]!, w[b])
            let key = robot.links[a].name + " / " + robot.links[b].name
            if d > 0.005 { worst[key] = max(worst[key] ?? 0, d) }
        }
    }
    for (k, v) in worst.sorted(by: { $0.value > $1.value }).prefix(12) { print("collide \(k): \(v)") }
}

if getenv("ELBOW_DIAG") != nil {
    for (_, robot) in robots {
        var q = [Double](repeating: 0, count: robot.dofCount)
        let elbow = robot.dofNames.firstIndex(of: "left_elbow_joint")!, roll = robot.dofNames.firstIndex(of: "left_shoulder_roll_joint")!
        let sh = robot.linkIndex(named: "left_shoulder_roll_link")!, wr = robot.links.firstIndex { $0.name.hasPrefix("left_wrist_roll") }!
        let el = robot.linkIndex(named: "left_elbow_link")!
        var best = (0.0, 0.0)
        let lim = robot.links[el].limits!
        for i in 0...200 {
            q[elbow] = lim.lower + (lim.upper - lim.lower) * Double(i) / 200
            let w = robot.forwardKinematics(root: .identity, q: q)
            let d = length(w[wr].translation - w[sh].translation)
            if d > best.1 { best = (q[elbow], d) }
        }
        q[elbow] = best.0
        for r in [1.5708, -1.5708] {
            q[roll] = r
            let w = robot.forwardKinematics(root: .identity, q: q)
            print("elbow \(robot.name): straight at \(best.0) (reach \(best.1)); roll \(r): wrist \(w[wr].translation) shoulder \(w[sh].translation)")
        }
    }
}

if getenv("TPOSE_DIAG") != nil {
    let (name, robot) = robots[0]
    let r = try HumanRetargeter(device: cuda, robot: robot, mapping: try mapping(name), surface: surface, skeleton: motions[0].skeleton)
    let world = robot.forwardKinematics(root: r.referenceRobotPose.root, q: r.referenceRobotPose.q)
    let goals = r.setup.goals(human: surface.referencePose!)
    for (t, target) in r.setup.mapping.targets.enumerated() {
        let p = world[r.setup.robotLink[t]].translation, g = goals[t].translation
        print("tpose \(target.robot): robot \(p) goal \(g) err \(length(p - g))")
    }
    print("tpose q \(zip(robot.dofNames, r.referenceRobotPose.q).map { "\($0.0)=\(($0.1 * 100).rounded() / 100)" })")
    print("tpose root \(r.referenceRobotPose.root)")
}

if getenv("IK_DIAG") != nil {
    let (name, robot) = robots[0]
    let setup = try RetargetSetup(mapping: try mapping(name), robot: robot, skeleton: motions[0].skeleton, humanReference: surface.referencePose!)
    let solver = try RetargetSolver(device: cuda, setup: setup)
    var q = [Double](repeating: 0, count: robot.dofCount)
    let probe = Int(getenv("IK_DOF").map { atoi($0) } ?? 3)
    q[probe] = 0.5
    let root = Transform(rotation: .identity, translation: Vec3(0, 0, 0.75))
    let world = robot.forwardKinematics(root: root, q: q)
    let goals = setup.robotLink.map { world[$0] }
    for frames in [1, 2, 5, 20] {
        let traj = try solver.solve(goals: [Array(repeating: goals, count: frames)], fps: [120])[0]
        let (pos, _) = trackingErrors(setup, traj, Array(repeating: goals, count: frames))
        let last = Array(pos.suffix(pos.count / frames))
        print("diag frames \(frames): max pos err \(last.max()!) cost \(traj.cost.last!) q[\(probe)] \(traj.q.last![probe]) root \(traj.root.last!.translation)")
    }
}

for (name, robot) in robots {
    test("[\(name)] mapping resolves") {
        let m = try mapping(name)
        let setup = try RetargetSetup(mapping: m, robot: robot, skeleton: motions[0].skeleton, humanReference: surface.referencePose!)
        print("  root scale \(setup.rootScale); chain scales " +
              zip(m.targets, setup.scales).map { "\($0.0.human)→\($0.0.robot) \(($0.1 * 1000).rounded() / 1000)" }.joined(separator: ", "))
        // Torso (rotation only) and hips (very short chains) aside, scales sit between the arm and leg ratios.
        let checked = zip(m.targets, setup.scales).filter { $0.0.positionWeight > 0 && !$0.0.human.hasSuffix("Leg") }
        expect(checked.allSatisfy { $0.1 > 0.5 && $0.1 < 1.0 }, "plausible scales")
    }

    test("[\(name)] recovers robot-generated motion") {
        let m = try mapping(name)
        let setup = try RetargetSetup(mapping: m, robot: robot, skeleton: motions[0].skeleton, humanReference: surface.referencePose!)
        let solver = try RetargetSolver(device: cuda, setup: setup)
        // Smooth joint trajectory inside 60 % of each range, root walking forward.
        let limits = robot.links.compactMap { l in l.dofIndex.map { ($0, l.limits!) } }.sorted { $0.0 < $1.0 }.map(\.1)
        var rng = SplitMix64(state: 3)
        let phases = limits.map { _ in rng.uniform(0, 6.28) }
        let fps = 120.0
        var goals: [[Transform]] = []
        for f in 0..<600 {
            let t = Double(f) / fps
            // Ramp in from q = 0 over 2 s (a stream starts near the default pose), then oscillate around the
            // middle of each range.
            let ramp = min(1, t / 2)
            let q = limits.enumerated().map { j, l -> Double in
                let mid = 0.5 * (l.lower + l.upper), amp = 0.3 * (l.upper - l.lower)
                return ramp * (mid + amp * sin(1.5 * t + phases[j]))
            }
            let root = Transform(rotation: Quat(axis: Vec3(0, 0, 1), angle: 0.3 * t),
                                 translation: Vec3(0.5 * t, 0, 0.75 + 0.03 * sin(4 * t)))
            let world = robot.forwardKinematics(root: root, q: q)
            goals.append(setup.robotLink.map { world[$0] })
        }
        let traj = try solver.solve(goals: [goals], fps: [fps])[0]
        let (pos, rot) = trackingErrors(setup, traj, goals)
        let settled = Array(pos.dropFirst(pos.count / 10)), settledRot = Array(rot.dropFirst(rot.count / 10))
        print("  after 10 %: position error mean \(settled.reduce(0, +) / Double(settled.count) * 1000) mm, max \(settled.max()! * 1000) mm; rotation max \(settledRot.max()! * 180 / .pi)°")
        expect(settled.max()! < 0.01, "positions recovered within 1 cm")
        expect(settledRot.max()! < 0.05, "rotations recovered within 3°")
        expect(limitViolations(robot, traj) == 0, "joint limits respected")
    }

    test("[\(name)] retarget SOMA clips (batched)") {
        let retargeter = try HumanRetargeter(device: cuda, robot: robot, mapping: try mapping(name), surface: surface,
                                             skeleton: motions[0].skeleton)
        let start = ContinuousClock.now
        let results = try retargeter.retarget(motions)
        let elapsed = ContinuousClock.now - start
        let frames = results.reduce(0) { $0 + $1.trajectory.frameCount }
        print("  \(clips.count) clips, \(frames) frames in \(elapsed): prepare (FK, skinning, contacts, goals) \(retargeter.lastTiming.prepare), IK \(retargeter.lastTiming.solve)")
        for (clip, r) in zip(clips, results) {
            let world = try SkeletonFK(device: cuda, skeleton: retargeter.setup.skeleton).compute(try BVH.load(file: bvhRoot + clip))
            let goals = retargeter.goals(world: world, ground: r.contacts.ground)
            let (pos, rot) = trackingErrors(retargeter.setup, r.trajectory, goals)
            let mean = pos.reduce(0, +) / Double(pos.count)
            print("  \(clip): position error mean \(Int(mean * 1000)) mm, p95 \(Int(percentile(pos, 0.95) * 1000)) mm; rotation p95 \(Int(percentile(rot, 0.95) * 180 / .pi))°; limit violations \(limitViolations(robot, r.trajectory))")
            expect(limitViolations(robot, r.trajectory) == 0, "limits respected")
            expect(mean < 0.06, "mean position error under 6 cm")
            expect(r.trajectory.root.allSatisfy { $0.translation.z > 0.2 && $0.translation.z < 1.2 }, "pelvis height plausible")
        }
    }

    test("[\(name)] batched = individual, CUDA = CPU") {
        let m = try mapping(name)
        let retargeter = try HumanRetargeter(device: cuda, robot: robot, mapping: m, surface: surface, skeleton: motions[0].skeleton)
        let short = motions.map { HumanMotion(skeleton: $0.skeleton, fps: $0.fps, local: Array($0.local.prefix(240))) }
        let batched = try retargeter.retarget(short)
        let single = try retargeter.retarget([short[1]])[0]
        let dq = zip(batched[1].trajectory.q, single.trajectory.q).flatMap { zip($0.0, $0.1).map { abs($0.0 - $0.1) } }.max()!
        print("  batched vs individual max |Δq| \(dq)")
        expect(dq < 1e-6, "batch independence")

        let cpu = try ComputeDevice.makeDefault(backend: .cpu)
        let cpuRetargeter = try HumanRetargeter(device: cpu, robot: robot, mapping: m, surface: surface, skeleton: motions[0].skeleton)
        let cpuResult = try cpuRetargeter.retarget([short[1]])[0]
        let dqBackend = zip(cpuResult.trajectory.q, single.trajectory.q).flatMap { zip($0.0, $0.1).map { abs($0.0 - $0.1) } }.max()!
        print("  CUDA vs CPU max |Δq| over 240 frames: \(dqBackend) rad")
        expect(dqBackend < 1e-2, "backend parity")
    }
}

test("[both] reference-pose goals sit on the robot's limbs") {
    // In the matching reference poses, anchors (shoulders, hips) land on the robot's own links and the limbs reach
    // where the robot's do: no limb starts out too narrow, too wide or out of reach.
    for (name, robot) in robots {
        let map = try mapping(name)
        let ref = surface.referencePose!
        let setup = try RetargetSetup(mapping: map, robot: robot, skeleton: motions[0].skeleton, humanReference: ref)
        let w = robot.forwardKinematics(root: robot.defaultRootPose, q: robot.dofNames.map { map.referencePose?[$0] ?? 0 })
        let goals = setup.goals(human: ref)
        let rootGoal = goals[setup.rootTarget].translation, rootRobot = w[setup.robotLink[setup.rootTarget]].translation
        for (t, target) in map.targets.enumerated() where target.positionWeight > 0 {
            let d = (goals[t].translation - rootGoal) - (w[setup.robotLink[t]].translation - rootRobot)
            let isAnchor = ["LeftArm", "RightArm", "LeftLeg", "RightLeg"].contains(target.human)
            let isLeg = ["Shin", "Foot", "ToeBase"].contains { target.human.hasSuffix($0) }
            if isAnchor { expect(length(d) < 1e-6, "\(name) \(target.robot) anchor at the robot's (\(d))") }
            // Legs: height from the hips (reach), horizontal from the uniformly scaled pelvis (balance).
            if isLeg { expect(abs(d.z) < 0.005, "\(name) \(target.robot) height \(format(d.z))") }
            expect(length(d) < 0.045, "\(name) \(target.robot) goal within 4.5 cm of the robot (\(format(length(d))))")
        }
    }
}

// MARK: - Offline solver

test("[g1_29dof] live streams reproduce the batch realtime solve") {
    // Two streams stepped one frame per tick, the second joining later and on another slot, against one batch
    // solve of both clips: same CPU stage, same kernel, solver state carried on the device between ticks.
    let (_, robot) = robots[0]
    let r = try HumanRetargeter(device: cuda, robot: robot, mapping: try mapping("g1_29dof"), surface: surface,
                                skeleton: motions[0].skeleton)
    let a = HumanMotion(skeleton: motions[0].skeleton, fps: motions[0].fps, local: Array(motions[0].local.prefix(400)))
    let b = HumanMotion(skeleton: motions[1].skeleton, fps: motions[1].fps, local: Array(motions[1].local.prefix(300)))
    let batch = try r.retarget([a, b])
    let validator = try Validator.load(repository: repositoryRoot, robot: robot, robotDirectory: g1 + "g1_29_dof")
    let live = try LiveRetargeter(retargeter: r, slots: 4, velocityLimits: validator.velocityLimits)
    try live.open(1, skeleton: a.skeleton, fps: a.fps)
    let late = 100
    var out: [[LiveRobotFrame]] = [[], []]
    for t in 0..<a.frameCount {
        if t == late { try live.open(3, skeleton: b.skeleton, fps: b.fps) }
        var inputs = [(slot: 1, local: a.local[t])]
        if t >= late && t - late < b.frameCount { inputs.append((slot: 3, local: b.local[t - late])) }
        let frames = try live.step(inputs)
        out[0].append(frames[0])
        if frames.count > 1 { out[1].append(frames[1]) }
    }
    for (k, result) in batch.enumerated() {
        let streamed = out[k]
        expect(streamed.count == result.trajectory.frameCount, "every frame streamed")
        let sameQ = zip(streamed, result.trajectory.q).allSatisfy { $0.0.solvedQ == $0.1 }
        let sameRoot = zip(streamed, result.trajectory.root).allSatisfy { $0.0.root == $0.1 }
        let sameGround = zip(streamed, result.contacts.ground).allSatisfy { $0.0.contacts.ground == $0.1 }
        expect(sameQ && sameRoot && sameGround, "clip \(k): streamed solve equals the batch solve bit for bit")
        let clamped = streamed.filter { $0.flags & LiveFlags.velocityClamped != 0 }.count
        print("  clip \(k): \(streamed.count) frames, velocity-clamped \(clamped)")
        expect(streamed[0].flags & LiveFlags.coldStart != 0 && streamed[1].flags & LiveFlags.coldStart == 0, "cold start flagged once")
    }
    // A restarted stream starts cold again and reproduces the clip's start.
    try live.open(1, skeleton: a.skeleton, fps: a.fps)
    let again = try live.step([(slot: 1, local: a.local[0])])[0]
    expect(again.solvedQ == batch[0].trajectory.q[0] && again.flags & LiveFlags.coldStart != 0, "restart is a cold start")
}

test("[g1_29dof] arm branch switch: eases out of a joint-limit corner, live = batch") {
    // jump_right_003__A030: the right arm, raised sideways through shoulder abduction, sits in a roll / yaw limit
    // corner from 1.4 s and snaps across at 3.6 s without the switch (~55 rad/s on the shoulder).
    let (_, robot) = robots[0]
    let motion = try BVH.load(file: bvhRoot + "220720/jump_right_003__A030.bvh")
    var off = try mapping("g1_29dof")
    off.solver.branchBlend = 0
    func peakArmSpeed(_ q: [[Double]], _ fps: Double) -> Double {
        let arm = robot.dofNames.indices.filter { ["shoulder", "elbow"].contains(where: robot.dofNames[$0].contains) }
        return zip(q, q.dropFirst()).map { a, b in arm.map { abs(b[$0] - a[$0]) * fps }.max()! }.max()!
    }
    let r = try HumanRetargeter(device: cuda, robot: robot, mapping: try mapping("g1_29dof"), surface: surface,
                                skeleton: motion.skeleton)
    let rOff = try HumanRetargeter(device: cuda, robot: robot, mapping: off, surface: surface, skeleton: motion.skeleton)
    let on = try r.retarget([motion])[0].trajectory, snapped = try rOff.retarget([motion])[0].trajectory
    let peakOn = peakArmSpeed(on.q, motion.fps), peakOff = peakArmSpeed(snapped.q, motion.fps)
    print("  peak shoulder / elbow speed \(peakOff) -> \(peakOn) rad/s; mean cost \(snapped.cost.reduce(0, +) / Double(snapped.cost.count)) -> \(on.cost.reduce(0, +) / Double(on.cost.count))")
    expect(peakOff > 30 && peakOn < 20, "switch eases the arm across")
    // The branch state (alternative arm, blend position) is carried on the device between live ticks.
    let validator = try Validator.load(repository: repositoryRoot, robot: robot, robotDirectory: g1 + "g1_29_dof")
    let live = try LiveRetargeter(retargeter: r, slots: 2, velocityLimits: validator.velocityLimits)
    try live.open(0, skeleton: motion.skeleton, fps: motion.fps)
    let streamed = try (0..<motion.frameCount).map { try live.step([(slot: 0, local: motion.local[$0])])[0] }
    expect(zip(streamed, on.q).allSatisfy { $0.0.solvedQ == $0.1 } && zip(streamed, on.root).allSatisfy { $0.0.root == $0.1 },
           "streamed solve equals the batch solve bit for bit")
}

test("[g1_29dof] live fixed lag reproduces the batch lagged solve") {
    // The teleop path (LiveRetargeter with a fixed-lag window solved at 50 Hz of stream time) against
    // LaggedRetargeter on the same clips, with the live frames handed over in batches of varying size (as the server
    // does): same causal CPU stage, realtime solves and window solves at the same frames, so the same poses; every
    // frame emitted once, in order, at least `lag` late.
    let (_, robot) = robots[0]
    let directory = g1 + "g1_29_dof"
    let geometry = try RobotGeometry(robot: robot, pointsFile: directory + "/mesh_points.npz")
    let lagged = try LaggedRetargeter(device: cuda, robot: robot, mapping: try mapping("g1_29dof"), surface: surface,
                                      skeleton: motions[0].skeleton, geometry: geometry, lag: 0.04, sweeps: 2, tickRate: 50)
    let a = HumanMotion(skeleton: motions[0].skeleton, fps: motions[0].fps, local: Array(motions[0].local.prefix(240)))
    let b = HumanMotion(skeleton: motions[3].skeleton, fps: motions[3].fps, local: Array(motions[3].local.prefix(200)))
    let batch = try lagged.solve(try lagged.prepare([a, b]))
    let live = try LiveRetargeter(retargeter: lagged.base, slots: 2,
                                  fixedLag: FixedLagSolver(offline: lagged.offline, slots: 2, lag: 0.04, sweeps: 2, tickRate: 50))
    try live.open(0, skeleton: a.skeleton, fps: a.fps)
    try live.open(1, skeleton: b.skeleton, fps: b.fps)
    var out: [[LiveRobotFrame]] = [[], []]
    let lagFrames = Int((0.04 * a.fps).rounded())
    var early = 0
    var t = 0, chunk = 1
    while t < a.frameCount {
        let end = min(a.frameCount, t + chunk)
        var inputs: [(slot: Int, local: [Transform])] = []
        for f in t..<end {
            inputs.append((slot: 0, local: a.local[f]))
            if f < b.frameCount { inputs.append((slot: 1, local: b.local[f])) }
        }
        for frame in try live.step(inputs) {
            if end - 1 - frame.frame < lagFrames { early += 1 }
            out[frame.slot].append(frame)
        }
        if t < b.frameCount && end >= b.frameCount { out[1] += try live.flush(1) }
        t = end
        chunk = chunk % 4 + 1
    }
    out[0] += try live.flush(0)
    expect(early == 0, "no frame emitted before \(lagFrames) newer frames arrived (\(early))")
    for (k, result) in batch.enumerated() {
        expect(out[k].map(\.frame) == Array(0..<result.trajectory.frameCount), "clip \(k): every frame once, in order")
        let same = zip(out[k], zip(result.trajectory.q, result.trajectory.root)).allSatisfy { $0.0.solvedQ == $0.1.0 && $0.0.root == $0.1.1 }
        expect(same, "clip \(k): live fixed-lag poses equal the batch lagged solve")
    }
}

test("[g1_29dof] offline improves on realtime (walk) and is deterministic") {
    let (_, robot) = robots[0]
    let directory = g1 + "g1_29_dof"
    let geometry = try RobotGeometry(robot: robot, pointsFile: directory + "/mesh_points.npz")
    let validator = try Validator.load(repository: repositoryRoot, robot: robot, robotDirectory: directory)
    let offline = try OfflineRetargeter(device: cuda, robot: robot, mapping: try mapping("g1_29dof"), surface: surface,
                                        skeleton: motions[0].skeleton, geometry: geometry)
    let walk = HumanMotion(skeleton: motions[0].skeleton, fps: motions[0].fps, local: Array(motions[0].local.prefix(1200)))
    let prepared = try offline.prepare([walk])
    let realtime = try offline.base.solve(prepared)[0]
    let solved = try offline.solve(prepared)[0]
    let again = try offline.solve(prepared)[0]
    print("  offline solve: \(offline.solver.lastDuration) for \(walk.frameCount) frames")
    let dq = zip(solved.trajectory.q, again.trajectory.q).flatMap { zip($0.0, $0.1).map { abs($0.0 - $0.1) } }.max()!
    expect(dq == 0, "deterministic (max |Δq| \(dq))")
    func metrics(_ r: RetargetResult) -> ValidationReport {
        var report = ValidationReport()
        validator.checkRobot(r, setup: offline.setup, fps: 50, into: &report)
        return report
    }
    let a = metrics(realtime), b = metrics(solved)
    for key in ["foot_skate_p95", "ground_penetration_max", "self_collision_max", "joint_acc_p99", "tracking_pos_mean",
                "dyn_residual_max", "zmp_outside_max", "torque_ratio_max"] {
        print("  \(key): realtime \(format(a.metric(key)!)) -> offline \(format(b.metric(key)!))")
    }
    // The dynamics term trades a little stance locking for dynamic consistency; skate stays far below the rule.
    expect(b.metric("foot_skate_p95")! < 0.05, "little skate")
    expect(b.metric("dyn_residual_max")! < 0.1, "dynamically consistent")
    expect(b.metric("ground_penetration_max")! < 0.02, "no ground penetration")
    expect(b.metric("self_collision_max")! < 0.01, "no self-collision")
    expect(b.metric("limit_violation_frames") == 0 && b.metric("joint_jump_frames") == 0, "limits, no jumps")
    expect(b.metric("tracking_pos_mean")! < 0.06, "still tracks")
}

test("[g1_29dof] offline dynamics: jump flight") {
    let (_, robot) = robots[0]
    let directory = g1 + "g1_29_dof"
    let geometry = try RobotGeometry(robot: robot, pointsFile: directory + "/mesh_points.npz")
    let validator = try Validator.load(repository: repositoryRoot, robot: robot, robotDirectory: directory)
    let offline = try OfflineRetargeter(device: cuda, robot: robot, mapping: try mapping("g1_29dof"), surface: surface,
                                        skeleton: motions[3].skeleton, geometry: geometry)
    let prepared = try offline.prepare([motions[3]])
    let realtime = try offline.base.solve(prepared)[0]
    let solved = try offline.solve(prepared)[0]
    func metrics(_ r: RetargetResult) -> ValidationReport {
        var report = ValidationReport()
        validator.checkRobot(r, setup: offline.setup, fps: 50, into: &report)
        return report
    }
    let a = metrics(realtime), b = metrics(solved)
    for key in ["dyn_residual_max", "dyn_residual_p95", "zmp_outside_max"] {
        print("  \(key): realtime \(format(a.metric(key)!)) -> offline \(format(b.metric(key)!))")
    }
    // Uniform root scaling makes the human's flight phases non-ballistic for the robot (0.785 g); the
    // centroidal term restores consistent flight and landing.
    // (The stance lock's vertical ramp, which removes whole-body jolts at heel strike, costs some of the landing:
    // 0.14 -> 0.17 m·g here.)
    expect(b.metric("dyn_residual_max")! < a.metric("dyn_residual_max")!, "less unexplained wrench")
    expect(b.metric("zmp_outside_max")! <= a.metric("zmp_outside_max")!, "ZMP no further from the support")
    expect(b.metric("dyn_residual_max")! < 0.25, "passes the residual rule")

}

test("[g1_29dof] offline: clasped hands move smoothly") {
    // The person holds the hands together; at robot scale the hand meshes overlap. Separate pairs between the
    // hand links used to push the hands apart in different directions frame to frame (hand accelerations up to
    // 80 m/s²); the single hand-to-hand row keeps them near the realtime level.
    let (_, robot) = robots[0]
    let directory = g1 + "g1_29_dof"
    let geometry = try RobotGeometry(robot: robot, pointsFile: directory + "/mesh_points.npz")
    let validator = try Validator.load(repository: repositoryRoot, robot: robot, robotDirectory: directory)
    let motion = try BVH.load(file: bvhRoot + "230124/i_got_this_002__A143.bvh")
    let offline = try OfflineRetargeter(device: cuda, robot: robot, mapping: try mapping("g1_29dof"), surface: surface,
                                        skeleton: motion.skeleton, geometry: geometry)
    let prepared = try offline.prepare([motion])
    let realtime = try offline.base.solve(prepared)[0], solved = try offline.solve(prepared)[0]
    // Many active self-collision rows per frame: their order (and which ones fit) must not depend on thread timing.
    let again = try offline.solve(prepared)[0]
    expect(zip(solved.trajectory.q, again.trajectory.q).allSatisfy { $0.0 == $0.1 } && solved.trajectory.root == again.trajectory.root,
           "bit-reproducible with self-collision rows")
    func handAcceleration(_ r: RetargetResult) -> Double {
        let t = r.trajectory.resampled(to: 50)
        let hands = ["left_wrist_yaw_link", "right_wrist_yaw_link"].map { robot.linkIndex(named: $0)! }
        let p = (0..<t.frameCount).map { f in hands.map { robot.forwardKinematics(root: t.root[f], q: t.q[f])[$0].translation } }
        var worst = 0.0
        for f in 1..<(p.count - 1) { for h in 0..<2 { worst = max(worst, length(p[f + 1][h] - 2 * p[f][h] + p[f - 1][h]) * 2500) } }
        return worst
    }
    let a = handAcceleration(realtime), b = handAcceleration(solved)
    var report = ValidationReport()
    validator.checkRobot(solved, setup: offline.setup, fps: 50, into: &report)
    print("  peak hand acceleration: realtime \(format(a)) -> offline \(format(b)) m/s²; self-collision \(format(report.metric("self_collision_max")!)) m")
    expect(b < 35, "offline peak hand acceleration (was 84 m/s²; realtime \(format(a)))")
    expect(report.metric("self_collision_max")! < 0.01, "hands kept apart")
}

test("[g1_29dof] offline keeps the robot above the ground when goals go below it") {
    let (_, robot) = robots[0]
    let directory = g1 + "g1_29_dof"
    let geometry = try RobotGeometry(robot: robot, pointsFile: directory + "/mesh_points.npz")
    let offline = try OfflineRetargeter(device: cuda, robot: robot, mapping: try mapping("g1_29dof"), surface: surface,
                                        skeleton: motions[0].skeleton, geometry: geometry)
    var prepared = try offline.prepare([HumanMotion(skeleton: motions[0].skeleton, fps: motions[0].fps,
                                                    local: Array(motions[0].local.prefix(240)))])[0]
    // Sink every goal by 10 cm.
    prepared.goals = prepared.goals.map { $0.map { Transform(rotation: $0.rotation, translation: $0.translation - Vec3(0, 0, 0.1)) } }
    let result = try offline.solve([prepared])[0]
    var lowest = Double.infinity
    for f in 0..<result.trajectory.frameCount {
        let world = robot.forwardKinematics(root: result.trajectory.root[f], q: result.trajectory.q[f])
        for (l, points) in geometry.linkPoints.enumerated() { for p in points { lowest = min(lowest, world[l].apply(p).z) } }
    }
    print("  lowest robot surface point with goals 10 cm under: \(lowest) m")
    expect(lowest > -0.02, "ground barrier holds")
}

finishTests()
