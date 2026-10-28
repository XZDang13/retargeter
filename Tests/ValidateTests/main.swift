// Validator checks on synthetic robot trajectories with known defects, rule evaluation, and the
// soma-retargeter CSV import convention.

import Contact
import Core
import MotionIO
import Retarget
import Validate

#if canImport(Glibc)
import Glibc
#endif

let g1 = repositoryRoot + "/assets/robots/unitree_g1/g1_29_dof"
let robot = try RobotUSDLoader.load(directory: g1, name: "g1_29dof")
let validator = try Validator.load(repository: repositoryRoot, robot: robot, robotDirectory: g1)
let n = robot.dofCount
let fps = 50.0

func standing(_ frames: Int = 100, _ edit: (Int, inout Transform, inout [Double]) -> Void = { _, _, _ in }) -> RobotTrajectory {
    var root: [Transform] = [], q: [[Double]] = []
    for f in 0..<frames {
        var r = robot.defaultRootPose
        var joints = [Double](repeating: 0, count: n)
        edit(f, &r, &joints)
        root.append(r)
        q.append(joints)
    }
    return RobotTrajectory(fps: fps, root: root, q: q, cost: Array(repeating: 0, count: frames))
}

/// Human contacts with both feet planted for the whole clip.
func feetDown(_ frames: Int) -> ContactTimeline {
    let names = ["left_heel", "left_toe", "right_heel", "right_toe"]
    return ContactTimeline(fps: fps, frameCount: frames, groupNames: names, ground: Array(repeating: 0, count: frames),
                           groundTilt: 0, height: Array(repeating: 0, count: frames * 4), speed: Array(repeating: 0, count: frames * 4),
                           contact: Array(repeating: true, count: frames * 4), envContact: Array(repeating: false, count: frames * 4),
                           envSupport: Array(repeating: false, count: frames), unsupported: Array(repeating: false, count: frames),
                           flight: Array(repeating: false, count: frames), penetration: Array(repeating: false, count: frames),
                           com: Array(repeating: .zero, count: frames), zmp: Array(repeating: .zero, count: frames))
}

func check(_ t: RobotTrajectory, contacts: ContactTimeline? = nil) -> ValidationReport {
    var report = ValidationReport()
    validator.checkRobot(t, goals: nil, setup: nil, contacts: contacts ?? feetDown(t.frameCount), into: &report)
    return report
}

test("geometry") {
    print("  \(validator.geometry.collisionPairs.count) self-collision pairs; soles \(validator.geometry.soles.map { $0.points.count })")
    expect(validator.geometry.soles.count == 2 && validator.geometry.soles.allSatisfy { $0.points.count == 4 }, "two soles, four points each")
    expect(validator.geometry.collisionPairs.count > 50, "collision pairs")
}

test("standing still is clean") {
    let r = check(standing())
    print("  " + r.metrics.map { "\($0.0)=\(format($0.1))" }.joined(separator: " "))
    expect(r.status == .accepted, "accepted: \(r.reasons)")
    expect((r.metric("ground_penetration_max") ?? 1) < 0.005, "no penetration")
    expect(r.metric("self_collision_max")! <= 0.005, "no self-collision")
    expect(r.metric("foot_skate_p95")! < 1e-6, "no skate")
    expect(r.metric("end_stability_margin")! > 0.02, "CoM well inside the feet")
}

test("defects are detected") {
    let elbow = robot.dofNames.firstIndex(of: "left_elbow_joint")!
    let jump = check(standing { f, _, q in if f >= 50 { q[elbow] = 2.0 } })   // 2 rad in one step: 1.5x the 37 rad/s limit at 50 fps is 1.1 rad
    expect(jump.metric("joint_jump_frames") == 1 && jump.reasons.contains { $0.contains("joint_jump_frames") }, "joint jump")
    expect(jump.ranges["joint_jump_frames"]?.first?.0 == 50, "jump frame range")

    let fast = check(standing { f, _, q in q[elbow] = 0.5 * sin(Double(f) * 2.0) })  // ~50 rad/s at 50 fps (limit 37)
    expect(fast.metric("velocity_violation_frames")! > 0, "velocity violation")

    let knee = robot.dofNames.firstIndex(of: "left_knee_joint")!
    let limit = check(standing { _, _, q in q[knee] = 5 })
    expect(limit.metric("limit_violation_frames") == 100, "limit violation")

    let sunk = check(standing { _, r, _ in r.translation.z -= 0.05 })
    expectClose(sunk.metric("ground_penetration_max")!, 0.05, tolerance: 0.005, "penetration depth")

    let sliding = check(standing { f, r, _ in r.translation.x += 0.4 * Double(f) / fps })
    expectClose(sliding.metric("foot_skate_p95")!, 0.4, tolerance: 0.01, "skate speed")

    let floating = check(standing { _, r, _ in r.translation.z += 0.06 })
    expect(floating.metric("foot_float_p95")! > 0.05, "float")

    // Left arm swung across the torso.
    let roll = robot.dofNames.firstIndex(of: "left_shoulder_roll_joint")!, pitch = robot.dofNames.firstIndex(of: "left_shoulder_pitch_joint")!
    let crossed = check(standing { _, _, q in q[roll] = -1.2; q[pitch] = -0.6; q[elbow] = 0.3 })
    print("  crossed arm self-collision \(crossed.metric("self_collision_max")!)")
    expect(crossed.metric("self_collision_max")! > 0.01, "self-collision")

    let leaning = check(standing { _, r, _ in r.rotation = Quat(axis: Vec3(0, 1, 0), angle: 0.6) })
    expect(leaning.metric("start_tilt_deg")! > 30, "start tilt")
}

test("dynamics: standing, levitating, ballistic flight, leaning, heavy arms") {
    // Standing still: the feet carry the weight exactly, half each, torques well inside the limits.
    let still = check(standing())
    expect(still.metric("dyn_residual_max")! < 0.01, "standing residual \(still.metric("dyn_residual_max")!)")
    expect(still.metric("zmp_outside_max")! == 0, "standing ZMP inside")
    expect(still.metric("torque_ratio_max")! < 0.5, "standing torque ratio \(still.metric("torque_ratio_max")!)")
    if let d = still.dynamics {
        let left = robot.linkIndex(named: "left_ankle_roll_link")!, right = robot.linkIndex(named: "right_ankle_roll_link")!
        let mg = robot.totalMass * 9.81
        expectClose(d.contactForce[50][left].z / mg, 0.5, tolerance: 0.1, "left foot share")
        expectClose((d.contactForce[50][left].z + d.contactForce[50][right].z) / mg, 1, tolerance: 0.01, "feet carry the weight")
        expect(d.tau.count == 100 && d.tau[0].count == n, "torque trace")
    } else {
        expect(false, "dynamics trace recorded")
    }

    // Held 10 cm above the ground: no contacts, so flight frames, and nothing supports the weight.
    let levitating = check(standing { _, r, _ in r.translation.z += 0.1 })
    expectClose(levitating.metric("dyn_flight_residual_max")!, 1, tolerance: 0.05, "levitating residual")
    expect(levitating.status == .rejected && levitating.reasons.contains { $0.contains("dyn_flight_residual_max") },
           "levitating rejected: \(levitating.reasons)")

    // Ballistic flight: 0.4 s in the air, the root (and with it the CoM) in free fall; consistent away from
    // the clip ends (one-sided differences there).
    let fly = check(standing(21) { f, r, _ in
        let t = Double(f - 10) / fps
        r.translation.z += 0.3 - 0.5 * 9.81 * t * t
    })
    let inner = fly.dynamics.map { Array($0.residual[5..<16]) } ?? [1]   // beyond the smoothing kernel at the ends
    expect(inner.max()! < 0.02, "ballistic residual \(inner.max()!)")

    // Leaning forward about the ankles with the feet kept flat (ankle pitch compensates): the CoM moves ahead
    // of the toes, which only an ankle moment beyond the foot could hold.
    let lp = robot.dofNames.firstIndex(of: "left_ankle_pitch_joint")!, rp = robot.dofNames.firstIndex(of: "right_ankle_pitch_joint")!
    let rest = robot.forwardKinematics(root: robot.defaultRootPose, q: Array(repeating: 0, count: n))
    let ankle = 0.5 * (rest[robot.linkIndex(named: "left_ankle_pitch_link")!].translation
                       + rest[robot.linkIndex(named: "right_ankle_pitch_link")!].translation)
    func leaning(_ angle: Double, _ sign: Double) -> RobotTrajectory {
        standing { _, r, q in
            let tilt = Quat(axis: Vec3(0, 1, 0), angle: angle)
            r = Transform(rotation: (tilt * r.rotation).normalized, translation: tilt.rotate(r.translation - ankle) + ankle)
            q[lp] = sign * angle
            q[rp] = sign * angle
        }
    }
    func soleLift(_ t: RobotTrajectory) -> Double {
        let world = robot.forwardKinematics(root: t.root[0], q: t.q[0])
        return validator.geometry.soles.flatMap { foot in foot.points.map { abs(world[foot.link].apply($0).z) } }.max()!
    }
    let lean = [leaning(0.3, 1), leaning(0.3, -1)].min { soleLift($0) < soleLift($1) }!
    expect(soleLift(lean) < 0.01, "feet flat while leaning (\(soleLift(lean)))")
    let leaned = check(lean)
    print("  leaning: residual \(format(leaned.metric("dyn_residual_max")!)), zmp outside \(format(leaned.metric("zmp_outside_max")!))")
    expect(leaned.metric("zmp_outside_max")! > 0.05, "ZMP ahead of the toes")
    expect(leaned.status == .rejected, "leaning rejected")

    // Arms held straight out with a fast swing: shoulder torque over the 25 N·m limit.
    let shoulder = robot.dofNames.firstIndex(of: "left_shoulder_pitch_joint")!
    let swing = check(standing { f, _, q in q[shoulder] = -1.5 + 1.2 * sin(Double(f) / fps * 2 * .pi * 3) })
    print("  swing torque ratio \(format(swing.metric("torque_ratio_max")!)), standing \(format(still.metric("torque_ratio_max")!))")
    expect(swing.metric("torque_ratio_max")! > 1, "fast arm swing exceeds the shoulder limit")
    expect(swing.details["torque:left_shoulder_pitch_joint"] != nil, "torque detail names the joint")
}

test("joint acceleration limits") {
    let hip = robot.dofNames.firstIndex(of: "left_hip_pitch_joint")!
    let limit = validator.accelerationLimits[hip]
    expectClose(limit, 95, tolerance: 5, "hip pitch limit derived from the model")
    let still = check(standing())
    expect(still.metric("joint_acc_ratio_max")! < 0.05, "standing: \(still.metric("joint_acc_ratio_max")!)")
    // Hip pitch oscillating ±0.3 rad at 4 Hz: peak acceleration 0.3 (8π)² ≈ 190 rad/s², twice the limit
    // (the 20 ms velocity smoothing and 50 fps differences measure ~70 % of it).
    let fast = check(standing { f, _, q in q[hip] = -0.3 + 0.3 * sin(Double(f) / fps * 2 * .pi * 4) })
    print("  hip swing: joint_acc_ratio_max \(format(fast.metric("joint_acc_ratio_max")!))")
    expect(fast.metric("joint_acc_ratio_max")! > 1.2, "fast hip swing over the limit")
    expect(fast.reasons.contains { $0.contains("joint_acc_ratio_max") }, "rejected on acceleration: \(fast.reasons)")
    expect(fast.details["joint_acc_ratio:left_hip_pitch_joint"] != nil, "detail names the joint")
    // A robot-config override replaces the derived value.
    let config = try JSONValue.parse(#"{ "acceleration_limits": { "left_hip_pitch_joint": 1000 } }"#)
    let relaxed = Validator(config: validator.config, robot: robot, geometry: validator.geometry, robotConfig: config)
    expect(relaxed.accelerationLimits[hip] == 1000, "override")
    var report = ValidationReport()
    relaxed.checkRobot(standing { f, _, q in q[hip] = -0.3 + 0.3 * sin(Double(f) / fps * 2 * .pi * 4) },
                       goals: nil, setup: nil, contacts: feetDown(100), into: &report)
    expect(report.metric("joint_acc_ratio_max")! < 1, "within the overridden limit")
}

test("rules") {
    var report = ValidationReport()
    let rules = [ValidationRule(metric: "a", max: 1, min: nil, severity: .reject),
                 ValidationRule(metric: "b", max: nil, min: 0, severity: .warn),
                 ValidationRule(metric: "missing", max: 0, min: nil, severity: .reject)]
    report = ValidationReport()
    var r1 = report
    r1.metricsForTesting([("a", 0.5), ("b", -1)])
    r1.applyForTesting(rules, stage: "robot")
    expect(r1.status == .accepted && r1.warnings.count == 1, "warn only")
    var r2 = report
    r2.metricsForTesting([("a", 2), ("b", 1)])
    r2.applyForTesting(rules, stage: "robot")
    expect(r2.status == .rejected && r2.stage == "robot" && r2.reasons == ["robot: a 2 > 1"], "reject: \(r2.reasons)")
    // Rounded to two significant digits of the bound: a hair over it passes, and messages show the digits that
    // decided (not "0.015 > 0.015").
    var r3 = report
    r3.metricsForTesting([("a", 1.04), ("b", 0)])
    r3.applyForTesting(rules, stage: "robot")
    expect(r3.status == .accepted, "1.04 rounds to the bound 1")
    var r4 = report
    let fine = [ValidationRule(metric: "c", max: 0.015, min: nil, severity: .reject)]
    r4.metricsForTesting([("c", 0.01634)])
    r4.applyForTesting(fine, stage: "robot")
    expect(r4.reasons == ["robot: c 0.016 > 0.015"], "message digits: \(r4.reasons)")
}

test("soma-retargeter CSV convention") {
    // Root rotation written as scipy as_euler('xyz') (extrinsic): R = Rz(c) Ry(b) Rx(a).
    let file = repositoryRoot + "/build/fixtures/nv_test.csv"
    let header = "Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ," + robot.dofNames.joined(separator: ",")
    let row = "0,10,20,79,30,-20,45," + (0..<n).map { _ in "10" }.joined(separator: ",")
    let handle = fopen(file, "w")!
    fputs(header + "\r\n" + row + "\r\n", handle)
    fclose(handle)
    let m = try RobotMotionImport.nvidiaCSV(file: file)
    let expected = Quat(axis: Vec3(0, 0, 1), angle: .pi / 4) * Quat(axis: Vec3(0, 1, 0), angle: -.pi / 9) * Quat(axis: Vec3(1, 0, 0), angle: .pi / 6)
    expect(m.root.count == 1 && m.jointNames == robot.dofNames, "parsed one frame, joint names")
    expect(Quat.angle(m.root[0].rotation, expected) < 1e-9, "rotation convention")
    expect(length(m.root[0].translation - Vec3(0.1, 0.2, 0.79)) < 1e-12, "centimetres")
    expect(abs(m.q[0][0] - 10 * .pi / 180) < 1e-12, "degrees")
}

finishTests()
