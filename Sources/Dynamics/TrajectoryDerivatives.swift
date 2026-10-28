// Velocities and accelerations of a sampled floating-base trajectory: central differences, with velocities
// Gaussian-smoothed before differentiating again (finite-difference accelerations of mocap-derived motion
// are dominated by sample noise otherwise).

import Core

#if canImport(Glibc)
import Glibc
#endif

public enum TrajectoryDerivatives {
    /// States for every frame. `smoothing` is the Gaussian σ in seconds applied to velocities (0: none).
    public static func states(root: [Transform], q: [[Double]], fps: Double, smoothing: Double) -> [RobotState] {
        let T = root.count
        guard T > 0 else { return [] }
        let n = q[0].count
        let h = 1 / fps
        func neighbours(_ t: Int) -> (Int, Int, Double) {
            let a = max(0, t - 1), b = min(T - 1, t + 1)
            return (a, b, Double(max(b - a, 1)) * h)
        }
        var v = [Vec3](repeating: .zero, count: T), w = v
        var qd = [[Double]](repeating: Array(repeating: 0, count: n), count: T)
        for t in 0..<T {
            let (a, b, dt) = neighbours(t)
            v[t] = (root[b].translation - root[a].translation) / dt
            w[t] = (root[b].rotation * root[a].rotation.conjugate).normalized.rotationVector / dt
            for j in 0..<n { qd[t][j] = (q[b][j] - q[a][j]) / dt }
        }
        let sigma = smoothing * fps
        v = smooth(v, sigma: sigma)
        w = smooth(w, sigma: sigma)
        qd = smoothRows(qd, sigma: sigma)
        var states: [RobotState] = []
        states.reserveCapacity(T)
        for t in 0..<T {
            let (a, b, dt) = neighbours(t)
            states.append(RobotState(
                root: root[t], q: q[t], rootVelocity: v[t], rootAngularVelocity: w[t], qd: qd[t],
                rootAcceleration: (v[b] - v[a]) / dt, rootAngularAcceleration: (w[b] - w[a]) / dt,
                qdd: (0..<n).map { (qd[b][$0] - qd[a][$0]) / dt }))
        }
        return states
    }

    static func kernel(sigma: Double) -> [Double] {
        guard sigma > 0.3 else { return [1] }
        let radius = Int((3 * sigma).rounded(.up))
        return (-radius...radius).map { exp(-0.5 * Double($0 * $0) / (sigma * sigma)) }
    }

    /// Gaussian smoothing with the kernel renormalised at the clip ends.
    static func smooth(_ x: [Vec3], sigma: Double) -> [Vec3] {
        let k = kernel(sigma: sigma), r = k.count / 2
        guard r > 0 else { return x }
        return x.indices.map { t in
            var sum = Vec3.zero, weight = 0.0
            for (i, kv) in k.enumerated() {
                let s = t + i - r
                guard s >= 0 && s < x.count else { continue }
                sum += kv * x[s]
                weight += kv
            }
            return sum / weight
        }
    }

    static func smoothRows(_ x: [[Double]], sigma: Double) -> [[Double]] {
        let k = kernel(sigma: sigma), r = k.count / 2
        guard r > 0, let n = x.first?.count else { return x }
        return x.indices.map { t in
            var sum = [Double](repeating: 0, count: n), weight = 0.0
            for (i, kv) in k.enumerated() {
                let s = t + i - r
                guard s >= 0 && s < x.count else { continue }
                for j in 0..<n { sum[j] += kv * x[s][j] }
                weight += kv
            }
            return sum.map { $0 / weight }
        }
    }
}
