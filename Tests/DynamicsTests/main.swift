// Inverse dynamics and contact forces against independent identities: static torques = ∂V/∂q, base wrench =
// rate of momentum − gravity, power balance τ·q̇ + base wrench · base twist = dE/dt, and the contact solver
// on poses whose support is known.

import Core
import Dynamics

#if canImport(Glibc)
import Glibc
#endif

let g1Dir = repositoryRoot + "/assets/robots/unitree_g1"


/// Smooth analytic trajectory: q(t) = q0 + a sin(ω t + φ) per dof, root translating and rotating about two axes.
struct Analytic {
    var q0: [Double], amp: [Double], freq: [Double], phase: [Double]
    var p0: Vec3, pAmp: Vec3, pFreq: Double
    var u1: Vec3, u2: Vec3, r0: Quat

    init(robot: RobotModel, seed: UInt64) {
        var rng = SplitMix64(state: seed)
        let limits = robot.links.compactMap { l in l.dofIndex.map { ($0, l.limits!) } }.sorted { $0.0 < $1.0 }.map(\.1)
        q0 = limits.map { 0.5 * ($0.lower + $0.upper) }
        amp = limits.map { 0.3 * ($0.upper - $0.lower) * rng.uniform(0.2, 1) }
        freq = q0.map { _ in rng.uniform(1, 4) }
        phase = q0.map { _ in rng.uniform(0, 6.28) }
        p0 = Vec3(0.3, -0.2, 0.8)
        pAmp = Vec3(0.2, 0.1, 0.05)
        pFreq = 2.5
        u1 = normalize(Vec3(0.2, 0.3, 1))
        u2 = normalize(Vec3(1, -0.4, 0.1))
        r0 = Quat(axis: Vec3(0, 1, 0), angle: 0.2)
    }

    func state(_ t: Double) -> RobotState {
        let q = q0.indices.map { q0[$0] + amp[$0] * sin(freq[$0] * t + phase[$0]) }
        let qd = q0.indices.map { amp[$0] * freq[$0] * cos(freq[$0] * t + phase[$0]) }
        let qdd = q0.indices.map { -amp[$0] * freq[$0] * freq[$0] * sin(freq[$0] * t + phase[$0]) }
        let s = sin(pFreq * t), c = cos(pFreq * t)
        let p = p0 + pAmp * s, v = pAmp * pFreq * c, a = -pAmp * pFreq * pFreq * s
        // R = exp(u1 θ1) exp(u2 θ2) r0, θ1 = 0.7 sin(1.3 t), θ2 = 0.4 sin(2.1 t).
        let th1 = 0.7 * sin(1.3 * t), dth1 = 0.7 * 1.3 * cos(1.3 * t), ddth1 = -0.7 * 1.69 * sin(1.3 * t)
        let th2 = 0.4 * sin(2.1 * t), dth2 = 0.4 * 2.1 * cos(2.1 * t), ddth2 = -0.4 * 4.41 * sin(2.1 * t)
        let r1 = Quat(axis: u1, angle: th1)
        let w1 = u1 * dth1, u2w = r1.rotate(u2)
        let w = w1 + u2w * dth2
        let alpha = u1 * ddth1 + u2w * ddth2 + cross(w1, u2w * dth2)
        let rot = (r1 * Quat(axis: u2, angle: th2) * r0).normalized
        return RobotState(root: Transform(rotation: rot, translation: p), q: q, rootVelocity: v, rootAngularVelocity: w,
                          qd: qd, rootAcceleration: a, rootAngularAcceleration: alpha, qdd: qdd)
    }
}

func potential(_ d: RobotDynamics, _ s: RobotState) -> Double {
    let m = d.motion(s)
    return d.mass.indices.reduce(0) { $0 - d.mass[$1] * dot(standardGravity, m.com[$1]) }
}

for dir in ["g1_29_dof", "g1_23_dof"] {
    let robot = try! RobotUSDLoader.load(directory: g1Dir + "/" + dir)
    let dyn = RobotDynamics(robot: robot)

    test("static torques = dV/dq, base wrench = weight (\(dir))") {
        let a = Analytic(robot: robot, seed: 7)
        var s = a.state(0.37)
        s = RobotState(root: s.root, q: s.q)
        let r = dyn.inverseDynamics(s)
        let mg = dyn.totalMass * 9.81
        expectClose(r.baseForce.z, mg, tolerance: 1e-9, "vertical base force")
        expectClose(length(Vec3(r.baseForce.x, r.baseForce.y, 0)), 0, tolerance: 1e-9, "horizontal base force")
        let com = dyn.centerOfMass(dyn.motion(s))
        let expectedMoment = cross(com - s.root.translation, Vec3(0, 0, mg))
        expectClose(length(r.baseMoment - expectedMoment), 0, tolerance: 1e-9, "base moment")
        var worst = 0.0
        for j in 0..<robot.dofCount {
            let eps = 1e-6
            var plus = s, minus = s
            plus.q[j] += eps
            minus.q[j] -= eps
            let dv = (potential(dyn, plus) - potential(dyn, minus)) / (2 * eps)
            worst = max(worst, abs(dv - r.tau[j]))
        }
        expect(worst < 1e-6, "max |tau - dV/dq| = \(worst)")
    }

    test("base wrench = momentum rate - gravity (\(dir))") {
        let a = Analytic(robot: robot, seed: 11)
        let t = 0.81, eps = 1e-5
        let s = a.state(t)
        let r = dyn.inverseDynamics(s)
        let point = s.root.translation
        let mp = dyn.momentum(dyn.motion(a.state(t + eps)), about: point)
        let mm = dyn.momentum(dyn.motion(a.state(t - eps)), about: point)
        let dP = (mp.linear - mm.linear) / (2 * eps), dL = (mp.angular - mm.angular) / (2 * eps)
        let m = dyn.motion(s)
        let expectedForce = dP - dyn.totalMass * standardGravity
        var gravityMoment = Vec3.zero
        for i in dyn.mass.indices { gravityMoment += cross(m.com[i] - point, dyn.mass[i] * standardGravity) }
        let expectedMoment = dL - gravityMoment
        expect(length(r.baseForce - expectedForce) < 1e-4 * max(1, length(expectedForce)), "force \(r.baseForce) vs \(expectedForce)")
        expect(length(r.baseMoment - expectedMoment) < 1e-4 * max(1, length(expectedMoment)), "moment \(r.baseMoment) vs \(expectedMoment)")
    }

    test("power balance (\(dir))") {
        let a = Analytic(robot: robot, seed: 23)
        for t in [0.1, 0.55, 1.4] {
            let eps = 1e-5
            let s = a.state(t)
            let r = dyn.inverseDynamics(s)
            func energy(_ x: RobotState) -> Double { dyn.kineticEnergy(dyn.motion(x)) + potential(dyn, x) }
            let dE = (energy(a.state(t + eps)) - energy(a.state(t - eps))) / (2 * eps)
            // The base moment is about the root origin, so it pairs with the root origin's twist.
            var power = dot(r.baseForce, s.rootVelocity) + dot(r.baseMoment, s.rootAngularVelocity)
            for j in 0..<robot.dofCount { power += r.tau[j] * s.qd[j] }
            expect(abs(power - dE) < 1e-4 * max(1, abs(dE)), "t=\(t): power \(power) vs dE/dt \(dE)")
        }
    }

    test("link velocities match finite-differenced FK (\(dir))") {
        let a = Analytic(robot: robot, seed: 5)
        let t = 0.3, eps = 1e-6
        let m = dyn.motion(a.state(t))
        let mp = dyn.motion(a.state(t + eps)), mm = dyn.motion(a.state(t - eps))
        var worst = 0.0
        for i in robot.links.indices {
            worst = max(worst, length((mp.com[i] - mm.com[i]) / (2 * eps) - m.comVelocity[i]))
            worst = max(worst, length((mp.comVelocity[i] - mm.comVelocity[i]) / (2 * eps) - m.comAcceleration[i]))
            let dr = (mp.pose[i].rotation * mm.pose[i].rotation.conjugate).normalized.rotationVector / (2 * eps)
            worst = max(worst, length(dr - m.omega[i]))
            worst = max(worst, length((mp.omega[i] - mm.omega[i]) / (2 * eps) - m.alpha[i]))
        }
        expect(worst < 1e-4, "max derivative mismatch \(worst)")
    }

    test("external forces enter like the Jacobian transpose (\(dir))") {
        let a = Analytic(robot: robot, seed: 3)
        let s = a.state(0.6)
        let link = robot.linkIndex(named: "left_ankle_roll_link")!
        let m = dyn.motion(s)
        let point = m.pose[link].apply(Vec3(0.05, 0.01, -0.03))
        let force = Vec3(12, -7, 150)
        let free = dyn.inverseDynamics(s, motion: m)
        let loaded = dyn.inverseDynamics(s, external: [PointForce(link: link, point: point, force: force)], motion: m)
        var tau = free.tau
        dyn.torqueFromForce(link: link, point: point, force: force, motion: m, into: &tau)
        let worst = zip(tau, loaded.tau).map { abs($0 - $1) }.max()!
        expect(worst < 1e-9, "tau mismatch \(worst)")
        expect(length(free.baseForce - force - loaded.baseForce) < 1e-9, "base force")
        expect(length(free.baseMoment - cross(point - s.root.translation, force) - loaded.baseMoment) < 1e-9, "base moment")
        // Virtual work: τ·δq for a joint perturbation equals the force's work at the moving point.
        let j = robot.dofCount / 2, eps = 1e-6
        var plus = s, minus = s
        plus.q[j] += eps
        minus.q[j] -= eps
        let local = m.pose[link].inverse.apply(point)
        let dp = (dyn.motion(plus).pose[link].apply(local) - dyn.motion(minus).pose[link].apply(local)) / (2 * eps)
        var only = [Double](repeating: 0, count: robot.dofCount)
        dyn.torqueFromForce(link: link, point: point, force: force, motion: m, into: &only)
        expectClose(only[j], -dot(force, dp), tolerance: 1e-6, "virtual work, dof \(j)")
    }

    test("contact forces: standing still is fully explained (\(dir))") {
        let s = RobotState(root: robot.defaultRootPose, q: Array(repeating: 0, count: robot.dofCount))
        let m = dyn.motion(s)
        var contacts: [ContactPoint] = []
        for side in ["left", "right"] {
            let foot = robot.linkIndex(named: "\(side)_ankle_roll_link")!
            for (x, y) in [(-0.05, -0.03), (-0.05, 0.03), (0.12, -0.03), (0.12, 0.03)] {
                var p = m.pose[foot].translation + Vec3(x, y, 0)
                p.z = 0
                contacts.append(ContactPoint(link: foot, point: p))
            }
        }
        let solver = ContactForceSolver(dynamics: dyn)
        let sol = solver.solve(s, contacts: contacts)
        let mg = dyn.totalMass * 9.81
        expect(sol.residual(mg: mg, momentScale: solver.momentScale) < 2e-3, "residual \(sol.residualForce) \(sol.residualMoment)")
        let total = sol.forces.reduce(Vec3.zero) { $0 + $1.force }
        expectClose(total.z, mg, tolerance: 1e-2 * mg, "total normal force")
        let left = sol.forces.filter { robot.links[$0.link].name.hasPrefix("left") }.reduce(0) { $0 + $1.force.z }
        expectClose(left / mg, 0.5, tolerance: 0.1, "weight share of the left foot")

        // Same pose with the robot's root shifted 0.3 m forward over the same contacts: CoM outside the support,
        // so a moment is left unexplained (≈ m·g · distance beyond the toe edge).
        var leaning = s
        leaning.root.translation.x += 0.3
        let lean = solver.solve(leaning, contacts: contacts)
        expect(lean.residual(mg: mg, momentScale: solver.momentScale) > 0.2, "leaning residual \(lean.residualForce) \(lean.residualMoment)")
        // Pulling: accelerating downward faster than gravity cannot be explained.
        var falling = s
        falling.rootAcceleration = Vec3(0, 0, -15)
        let fall = solver.solve(falling, contacts: contacts)
        expectClose(fall.residualForce.z / mg, (9.81 - 15) / 9.81, tolerance: 1e-3, "pull residual")
        // Ballistic flight without contacts: nothing to explain.
        var flight = s
        flight.rootAcceleration = standardGravity
        let fly = solver.solve(flight, contacts: [])
        expect(length(fly.residualForce) < 1e-9 && length(fly.residualMoment) < 1e-9, "ballistic residual")
    }
}

test("effort-limited joint accelerations") {
    let robot = try RobotUSDLoader.load(directory: g1Dir + "/g1_29_dof")
    let dyn = RobotDynamics(robot: robot)
    let limits = dyn.effortLimitedAcceleration()
    func limit(_ name: String) -> Double { limits[robot.dofNames.firstIndex(of: name)!] }
    // Joints that swing large masses are the tight ones; distal joints reach thousands of rad/s².
    expectClose(limit("left_hip_pitch_joint"), 95, tolerance: 5, "hip pitch (88 N·m, whole leg)")
    expectClose(limit("waist_pitch_joint"), 80, tolerance: 5, "waist pitch (50 N·m, upper body)")
    expectClose(limit("left_shoulder_pitch_joint"), 178, tolerance: 10, "shoulder pitch (25 N·m, arm)")
    expect(limit("left_ankle_roll_joint") > 1000, "ankle roll moves only the foot")
    expect(limits.allSatisfy { $0.isFinite && $0 > 0 }, "every joint has a limit")
    expectClose(limit("left_hip_pitch_joint"), limit("right_hip_pitch_joint"), tolerance: 1e-3, "left = right")
}

test("non-negative least squares") {
    // min |A x - b|, x >= 0, A = I (2x2) rotated: known optimum on the boundary.
    let A = [[1.0, 0.5], [0.5, 1.0], [0.2, -0.3]], b = [1.0, -1.0, 0.5]
    var G = [Double](repeating: 0, count: 4), h = [0.0, 0.0]
    for r in 0..<3 {
        for i in 0..<2 {
            h[i] += A[r][i] * b[r]
            for j in 0..<2 { G[i * 2 + j] += A[r][i] * A[r][j] }
        }
    }
    let x = nonNegativeLeastSquares(G: G, h: h, n: 2)
    // Brute force over a grid.
    var best = (Double.infinity, 0.0, 0.0)
    for i in 0...400 {
        for j in 0...400 {
            let x0 = Double(i) / 200, x1 = Double(j) / 200
            let e = (0..<3).reduce(0.0) { s, r in let d = A[r][0] * x0 + A[r][1] * x1 - b[r]; return s + d * d }
            if e < best.0 { best = (e, x0, x1) }
        }
    }
    expect(x.allSatisfy { $0 >= 0 }, "feasible")
    expectClose(x[0], best.1, tolerance: 0.01, "x0")
    expectClose(x[1], best.2, tolerance: 0.01, "x1")
}

test("trajectory derivatives") {
    let robot = try RobotUSDLoader.load(directory: g1Dir + "/g1_29_dof")
    let a = Analytic(robot: robot, seed: 9)
    let fps = 200.0
    let samples = (0..<400).map { a.state(Double($0) / fps) }
    let states = TrajectoryDerivatives.states(root: samples.map(\.root), q: samples.map(\.q), fps: fps, smoothing: 0)
    var worstV = 0.0, worstA = 0.0, worstW = 0.0, worstAlpha = 0.0
    for t in 10..<390 {
        worstV = max(worstV, zip(states[t].qd, samples[t].qd).map { abs($0 - $1) }.max()!)
        worstA = max(worstA, zip(states[t].qdd, samples[t].qdd).map { abs($0 - $1) }.max()!)
        worstW = max(worstW, length(states[t].rootAngularVelocity - samples[t].rootAngularVelocity))
        worstAlpha = max(worstAlpha, length(states[t].rootAngularAcceleration - samples[t].rootAngularAcceleration))
    }
    expect(worstV < 0.02 && worstW < 0.02, "velocity error \(worstV), angular \(worstW)")
    expect(worstA < 0.2 && worstAlpha < 0.2, "acceleration error \(worstA), angular \(worstAlpha)")
}

/// Point-mass link CoMs and joint frames of one pose (for mirrors of the offline kernel's dynamics rows).
struct PointMassFrame {
    var com: [Vec3], origin: [Vec3], axis: [Vec3], pose: [Transform]
    init(_ robot: RobotModel, root: Transform, q: [Double]) {
        let pose = robot.forwardKinematics(root: root, q: q)
        self.pose = pose
        com = robot.links.indices.map { pose[$0].apply(robot.links[$0].inertial?.centerOfMass ?? .zero) }
        origin = Array(repeating: .zero, count: robot.links.count)
        axis = origin
        for (i, link) in robot.links.enumerated() {
            guard let p = link.parent, case .revolute(let a) = link.joint else { continue }
            let frame = pose[p] * link.jointOrigin
            origin[i] = frame.translation
            axis[i] = frame.rotation.rotate(a)
        }
    }
}

func ancestors(_ robot: RobotModel, _ i: Int) -> Set<Int> {
    var out: Set<Int> = [i]
    var k = i
    while let p = robot.links[k].parent { out.insert(p); k = p }
    return out
}

/// Mirror of the offline kernel's force-free ZMP row (slot t): residual u = n · (F_z p) - h F_z for a fixed
/// support half-plane n · x <= h, i.e. the moment the support cannot provide, against finite differences.
test("solver ZMP support row Jacobian matches finite differences") {
    let robot = try RobotUSDLoader.load(directory: g1Dir + "/g1_29_dof")
    let dyn = RobotDynamics(robot: robot)
    let a = Analytic(robot: robot, seed: 41)
    let rate = 40.0
    let sm = a.state(0.4 - 1 / rate), s0 = a.state(0.4), sp = a.state(0.4 + 1 / rate)
    let fm = PointMassFrame(robot, root: sm.root, q: sm.q), fp = PointMassFrame(robot, root: sp.root, q: sp.q)
    let n2 = SIMD2<Double>(0.6, 0.8), h = -0.3
    func wrench(_ f: PointMassFrame) -> (F: Vec3, N: Vec3, c: Vec3) {
        var c = Vec3.zero
        for i in robot.links.indices { c += dyn.mass[i] * f.com[i] }
        c /= dyn.totalMass
        var F = Vec3.zero, N = Vec3.zero
        for i in robot.links.indices {
            let A = dyn.mass[i] * ((fp.com[i] - 2 * f.com[i] + fm.com[i]) * rate * rate - standardGravity)
            F += A
            N += cross(f.com[i] - c, A)
        }
        return (F, N, c)
    }
    func residual(_ root: Transform, _ q: [Double]) -> Double {
        let (F, N, c) = wrench(PointMassFrame(robot, root: root, q: q))
        let p = SIMD2(c.x - (N.y + c.z * F.x) / F.z, c.y + (N.x - c.z * F.y) / F.z)
        return ((n2 * p).sum() - h) * F.z
    }
    let f0 = PointMassFrame(robot, root: s0.root, q: s0.q)
    let (F, N, c) = wrench(f0)
    let p = SIMD2(c.x - (N.y + c.z * F.x) / F.z, c.y + (N.x - c.z * F.y) / F.z)
    let d = (n2 * p).sum() - h
    let anc = robot.links.indices.map { ancestors(robot, $0) }
    let coef = -2 * rate * rate
    var worst = 0.0
    for col in 0..<(6 + robot.dofCount) {
        let e = Vec3(col % 3 == 0 ? 1 : 0, col % 3 == 1 ? 1 : 0, col % 3 == 2 ? 1 : 0)
        let colLink = col >= 6 ? dyn.dofLink[col - 6] : -1
        var jF = Vec3.zero, jN = Vec3.zero, jC = Vec3.zero
        for i in robot.links.indices {
            let ct = f0.com[i]
            var J: Vec3
            if col < 3 { J = e } else if col < 6 { J = cross(e, ct - s0.root.translation) } else {
                guard anc[i].contains(colLink) else { continue }
                J = cross(f0.axis[colLink], ct - f0.origin[colLink])
            }
            let m = dyn.mass[i]
            let A = m * ((fp.com[i] - 2 * ct + fm.com[i]) * rate * rate - standardGravity)
            jF += m * coef * J
            jN += m * cross(ct - c, coef * J) - cross(A, J)
            jC += m * J
        }
        jC /= dyn.totalMass
        jN -= cross(jC, F)
        // Kernel coefficients (out.y, out.z = n; out.x = d).
        let px = (N.y + c.z * F.x) / F.z, py = -(N.x - c.z * F.y) / F.z
        let cF = n2.x * Vec3(-c.z, 0, px) + n2.y * Vec3(0, -c.z, py) + Vec3(0, 0, d)
        let cN = n2.x * Vec3(0, -1, 0) + n2.y * Vec3(1, 0, 0)
        let cC = n2.x * Vec3(F.z, 0, -F.x) + n2.y * Vec3(0, F.z, -F.y)
        let analytic = dot(cF, jF) + dot(cN, jN) + dot(cC, jC)
        let eps = 1e-6
        var rp = s0.root, rm = s0.root, qp = s0.q, qm = s0.q
        if col < 3 { rp.translation += eps * e; rm.translation -= eps * e } else if col < 6 {
            rp.rotation = (Quat(axis: e, angle: eps) * rp.rotation).normalized
            rm.rotation = (Quat(axis: e, angle: -eps) * rm.rotation).normalized
        } else { qp[col - 6] += eps; qm[col - 6] -= eps }
        let numeric = (residual(rp, qp) - residual(rm, qm)) / (2 * eps)
        let err = abs(numeric - analytic) / max(1, abs(numeric))
        if err > 1e-3 { print("  col \(col): analytic \(analytic) numeric \(numeric)") }
        worst = max(worst, err)
    }
    expect(worst < 1e-3, "worst relative error \(worst)")
}

finishTests()
