// Contact-force estimation: the ground forces at the active contact points that best explain a motion.
//
// Per instant, unknowns are non-negative coefficients on the edges of a linearised friction cone (pyramid)
// at every contact point, so every force is unilateral and inside the cone by construction. They are found
// by non-negative least squares on
//   - the 6 base-wrench rows: inertial + gravity wrench of the whole robot minus the contact wrench
//     (force in units of m·g, moment about the CoM in units of m·g·ℓ),
//   - optionally one row per joint: torque / effort limit, weighted by β (only breaks ties between force
//     distributions, e.g. how much weight each foot carries, so the exported torques are realistic; 0 drops them),
//   - a small ridge.
// What the contacts cannot supply stays in the base-wrench residual: pulling on the ground, horizontal
// force beyond friction, a centre of pressure outside the support, non-ballistic flight.

import Core

public struct ContactPoint: Sendable {
    public var link: Int
    public var point: Vec3   // world
    /// Stable identity across frames (e.g. link and patch corner), for warm starts; -1: none.
    public var id: Int

    public init(link: Int, point: Vec3, id: Int = -1) {
        self.link = link
        self.point = point
        self.id = id
    }
}

public struct ContactSolution: Sendable {
    public var forces: [PointForce]
    /// Base wrench left unexplained: force (N) and moment about the CoM (N·m).
    public var residualForce: Vec3
    public var residualMoment: Vec3
    public var tau: [Double]
    /// Gravito-inertial wrench of the motion (what the environment must supply in total), about the CoM.
    public var requiredForce: Vec3
    public var requiredMoment: Vec3
    public var com: Vec3
    /// Active friction-pyramid edges as contact id · 4 + edge (warm start for the next frame).
    public var activeEdges: [Int] = []

    /// Unexplained wrench as one dimensionless number: |f| / (m·g) and |n| / (m·g·ℓ) combined (the solver
    /// can trade one for the other, e.g. a lean shows up as horizontal force or as moment).
    public func residual(mg: Double, momentScale: Double) -> Double {
        let f = length(residualForce) / mg, n = length(residualMoment) / (mg * momentScale)
        return (f * f + n * n).squareRoot()
    }
}

public struct ContactForceSolver: Sendable {
    public var dynamics: RobotDynamics
    public var friction: Double
    public var momentScale: Double   // ℓ (m)
    public var torqueWeight: Double  // β
    public var ridge: Double
    public var gravity: Vec3

    public init(dynamics: RobotDynamics, friction: Double = 0.7, momentScale: Double = 0.1, torqueWeight: Double = 0.02,
                ridge: Double = 1e-4, gravity: Vec3 = standardGravity) {
        self.dynamics = dynamics
        self.friction = friction
        self.momentScale = momentScale
        self.torqueWeight = torqueWeight
        self.ridge = ridge
        self.gravity = gravity
    }

    /// Unit edge directions of the friction pyramid around +Z.
    public var edges: [Vec3] {
        [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)].map { normalize(Vec3(friction * $0.0, friction * $0.1, 1)) }
    }

    /// `warmStart`: the previous frame's `activeEdges`.
    public func solve(_ state: RobotState, contacts: [ContactPoint], warmStart: [Int] = []) -> ContactSolution {
        let motion = dynamics.motion(state)
        let free = dynamics.inverseDynamics(state, gravity: gravity, motion: motion)
        let com = dynamics.centerOfMass(motion)
        let root = state.root.translation
        let requiredForce = free.baseForce
        let requiredMoment = free.baseMoment + cross(root - com, free.baseForce)
        let n = dynamics.robot.dofCount
        let mg = dynamics.totalMass * length(gravity)
        guard !contacts.isEmpty else {
            return ContactSolution(forces: [], residualForce: requiredForce, residualMoment: requiredMoment, tau: free.tau,
                                   requiredForce: requiredForce, requiredMoment: requiredMoment, com: com)
        }
        let edges = self.edges
        let k = contacts.count * edges.count, rows = torqueWeight > 0 ? 6 + n : 6
        // Rows of A (columns = edge coefficients) and b, weighted.
        var A = [Double](repeating: 0, count: rows * k), b = [Double](repeating: 0, count: rows)
        // Coefficients are in units of m·g so the system is well scaled (and the ridge is relative).
        let wf = 1.0, wm = 1 / momentScale
        let wt = (0..<n).map { dynamics.effort[$0].isFinite ? torqueWeight * mg / dynamics.effort[$0] : 0 }
        for i in 0..<3 {
            b[i] = wf * requiredForce[i] / mg
            b[3 + i] = wm * requiredMoment[i] / mg
        }
        if rows > 6 { for j in 0..<n { b[6 + j] = -wt[j] * free.tau[j] / mg } }
        for (c, contact) in contacts.enumerated() {
            for (e, d) in edges.enumerated() {
                let col = c * edges.count + e
                let moment = cross(contact.point - com, d)
                for i in 0..<3 {
                    A[i * k + col] = wf * d[i]
                    A[(3 + i) * k + col] = wm * moment[i]
                }
                guard rows > 6 else { continue }
                var t = [Double](repeating: 0, count: n)
                dynamics.torqueFromForce(link: contact.link, point: contact.point, force: d, motion: motion, into: &t)
                for j in 0..<n { A[(6 + j) * k + col] = wt[j] * t[j] }
            }
        }
        // tau = tau0 + T λ; the torque rows minimise |W (tau0 + T λ)|, i.e. W T λ ≈ −W tau0.
        var G = [Double](repeating: 0, count: k * k), h = [Double](repeating: 0, count: k)
        for r in 0..<rows {
            for a in 0..<k where A[r * k + a] != 0 {
                h[a] += A[r * k + a] * b[r]
                for c in a..<k { G[a * k + c] += A[r * k + a] * A[r * k + c] }
            }
        }
        for a in 0..<k {
            G[a * k + a] += ridge
            for c in 0..<a { G[a * k + c] = G[c * k + a] }
        }
        var initial: [Int] = []
        if !warmStart.isEmpty {
            let active = Set(warmStart)
            for (c, contact) in contacts.enumerated() where contact.id >= 0 {
                for e in 0..<edges.count where active.contains(contact.id * edges.count + e) { initial.append(c * edges.count + e) }
            }
        }
        let lambda = nonNegativeLeastSquares(G: G, h: h, n: k, initial: initial)
        var activeEdges: [Int] = []
        for (c, contact) in contacts.enumerated() where contact.id >= 0 {
            for e in 0..<edges.count where lambda[c * edges.count + e] > 0 { activeEdges.append(contact.id * edges.count + e) }
        }

        var forces: [PointForce] = []
        var force = Vec3.zero, moment = Vec3.zero
        var tau = free.tau
        for (c, contact) in contacts.enumerated() {
            var f = Vec3.zero
            for (e, d) in edges.enumerated() { f += mg * lambda[c * edges.count + e] * d }
            guard length(f) > 0 else { continue }
            forces.append(PointForce(link: contact.link, point: contact.point, force: f))
            force += f
            moment += cross(contact.point - com, f)
            dynamics.torqueFromForce(link: contact.link, point: contact.point, force: f, motion: motion, into: &tau)
        }
        return ContactSolution(forces: forces, residualForce: requiredForce - force, residualMoment: requiredMoment - moment,
                               tau: tau, requiredForce: requiredForce, requiredMoment: requiredMoment, com: com,
                               activeEdges: activeEdges)
    }
}

/// min ½ xᵀGx − hᵀx subject to x ≥ 0 (G symmetric positive definite, n×n row-major): the normal-equation form
/// of non-negative least squares, solved with the Lawson-Hanson active-set method (Bro & de Jong's variant).
/// `initial` seeds the passive (nonzero) set, e.g. with the previous frame's solution support.
public func nonNegativeLeastSquares(G: [Double], h: [Double], n: Int, initial: [Int] = [],
                                    tolerance: Double = 1e-10) -> [Double] {
    var x = [Double](repeating: 0, count: n), s = x, w = x
    var passive = [Bool](repeating: false, count: n)
    var index = [Int](), L = [Double](repeating: 0, count: n * n), y = [Double](repeating: 0, count: n)
    index.reserveCapacity(n)
    // s = argmin over the passive set (zero elsewhere), by Cholesky of G restricted to it. Unchecked buffers:
    // this is the validator's hot loop (dozens of factorisations per frame).
    func solvePassive() {
        index.removeAll(keepingCapacity: true)
        for i in 0..<n where passive[i] { index.append(i) }
        let m = index.count
        G.withUnsafeBufferPointer { G in
        h.withUnsafeBufferPointer { h in
        index.withUnsafeBufferPointer { index in
        L.withUnsafeMutableBufferPointer { L in
        y.withUnsafeMutableBufferPointer { y in
            for a in 0..<m {
                let row = index[a] * n
                for c in 0...a { L[a * m + c] = G[row + index[c]] }
                y[a] = h[index[a]]
            }
            for j in 0..<m {
                var d = L[j * m + j]
                for k in 0..<j { d -= L[j * m + k] * L[j * m + k] }
                d = max(d, 1e-300).squareRoot()
                L[j * m + j] = d
                var i = j + 1
                while i < m {
                    var v = L[i * m + j]
                    for k in 0..<j { v -= L[i * m + k] * L[j * m + k] }
                    L[i * m + j] = v / d
                    i += 1
                }
            }
            for i in 0..<m {
                var v = y[i]
                for k in 0..<i { v -= L[i * m + k] * y[k] }
                y[i] = v / L[i * m + i]
            }
            var i = m - 1
            while i >= 0 {
                var v = y[i]
                var k = i + 1
                while k < m { v -= L[k * m + i] * y[k]; k += 1 }
                y[i] = v / L[i * m + i]
                i -= 1
            }
        }}}}}
        for i in 0..<n { s[i] = 0 }
        for (a, i) in index.enumerated() { s[i] = y[a] }
    }
    func gradient() {
        G.withUnsafeBufferPointer { G in
            for i in 0..<n {
                var v = h[i]
                for j in 0..<n where x[j] != 0 { v -= G[i * n + j] * x[j] }
                w[i] = v
            }
        }
    }
    // Inner loop: step from x towards s until s is feasible on the passive set.
    func makeFeasible() {
        var inner = 0
        while inner < 3 * n {
            var alpha = Double.infinity
            for i in 0..<n where passive[i] && s[i] <= 0 { alpha = min(alpha, x[i] / (x[i] - s[i])) }
            guard alpha.isFinite else { break }
            inner += 1
            for i in 0..<n { x[i] += alpha * (s[i] - x[i]) }
            for i in 0..<n where passive[i] && x[i] <= tolerance { passive[i] = false; x[i] = 0 }
            solvePassive()
        }
        x = s
    }
    if !initial.isEmpty {
        for i in initial where i >= 0 && i < n { passive[i] = true }
        solvePassive()
        // From x = 0 the step towards s is cut at the first passive variable that would turn negative.
        for i in 0..<n where passive[i] && s[i] <= 0 { passive[i] = false }
        solvePassive()
        makeFeasible()
    }
    gradient()
    for _ in 0..<(3 * n + 10) {
        var best = -1
        var bestValue = tolerance
        for i in 0..<n where !passive[i] && w[i] > bestValue { best = i; bestValue = w[i] }
        guard best >= 0 else { break }
        passive[best] = true
        solvePassive()
        makeFeasible()
        gradient()
    }
    return x
}
