// Geometry the solver and validator need beyond kinematics: per-link surface points (from the exported visual
// meshes), capsules fitted to them for self-collision, and the sole contact points of each foot.

import Core
import MotionIO

public struct Capsule: Sendable {
    public var a: Vec3, b: Vec3   // segment endpoints in the link frame
    public var radius: Double
}

public struct RobotGeometry: Sendable {
    /// Surface points per link, link frame.
    public var linkPoints: [[Vec3]]
    /// Median-radius capsules (mostly inside the surface): the validator tests the other link's real surface
    /// points against these.
    public var capsules: [Capsule?]
    /// Enclosing capsules (every surface point inside) the solver keeps apart, so the surface-based check passes.
    public var outerCapsules: [Capsule?]
    /// Link pairs checked for self-collision: not within two joints of each other and not overlapping
    /// in the default pose (where overlap is a property of the capsule approximation).
    public var collisionPairs: [(Int, Int)]
    /// Sole points (link frame) per foot: (link index, points), left then right.
    public var soles: [(link: Int, points: [Vec3])]

    public init(robot: RobotModel, pointsFile: String) throws {
        let npz = try NPZ.read(file: pointsFile)
        guard let names = try npz["names"]?.strings(), let offsets = try npz["offsets"]?.ints(),
              let flat = try npz["points"]?.doubles() else {
            throw NPYError(message: "\(pointsFile): expected names / offsets / points")
        }
        var meshPoints: [String: [Vec3]] = [:]
        for (i, name) in names.enumerated() {
            meshPoints[name] = (offsets[i]..<offsets[i + 1]).map { Vec3(flat[3 * $0], flat[3 * $0 + 1], flat[3 * $0 + 2]) }
        }
        linkPoints = robot.links.map { link in
            link.visuals.flatMap { visual in (meshPoints[visual.mesh] ?? []).map { visual.origin.apply($0) } }
        }
        capsules = linkPoints.map { RobotGeometry.fitCapsule($0, radiusPercentile: 0.5) }
        outerCapsules = linkPoints.map { RobotGeometry.fitCapsule($0, radiusPercentile: 1.0) }

        // Tree distance between links.
        func ancestors(_ i: Int) -> [Int] {
            var out = [i]
            var k = i
            while let p = robot.links[k].parent { out.append(p); k = p }
            return out
        }
        let chains = robot.links.indices.map(ancestors)
        func treeDistance(_ a: Int, _ b: Int) -> Int {
            for (da, x) in chains[a].enumerated() { if let db = chains[b].firstIndex(of: x) { return da + db } }
            return .max
        }
        let rest = robot.forwardKinematics(root: robot.defaultRootPose, q: Array(repeating: 0, count: robot.dofCount))
        let capsules = self.capsules
        var pairs: [(Int, Int)] = []
        for a in robot.links.indices {
            for b in (a + 1)..<robot.links.count where treeDistance(a, b) > 2 {
                guard let ca = capsules[a], let cb = capsules[b] else { continue }
                if RobotGeometry.penetration(ca, rest[a], cb, rest[b]) > 0.02 { continue }
                pairs.append((a, b))
            }
        }
        collisionPairs = pairs

        // Sole points: the four bottom corners of each foot mesh (front/back x left/right among the points within
        // 5 mm of the lowest one), so locking them to z = 0 puts the actual sole on the ground. Falls back to
        // the bottoms of the foot's collision spheres.
        let linkPoints = self.linkPoints
        soles = ["left", "right"].compactMap { side in
            guard let foot = robot.links.firstIndex(where: { $0.name == "\(side)_ankle_roll_link" }) else { return nil }
            let points = linkPoints[foot]
            if let lowest = points.map(\.z).min() {
                let bottom = points.filter { $0.z < lowest + 0.005 }
                if bottom.count >= 4 {
                    let corners = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)].map { sx, sy in
                        bottom.max { sx * $0.x + sy * $0.y < sx * $1.x + sy * $1.y }!
                    }
                    return (foot, corners.map { Vec3($0.x, $0.y, lowest) })
                }
            }
            let spheres = robot.links[foot].collisions.compactMap { c -> Vec3? in
                guard case .sphere(let r) = c.shape else { return nil }
                return c.origin.translation - Vec3(0, 0, r)
            }
            return (foot, spheres)
        }
    }

    public static func fitCapsuleForDiagnostics(_ points: [Vec3], radiusPercentile: Double) -> Capsule? {
        fitCapsule(points, radiusPercentile: radiusPercentile)
    }

    /// Capsule along the principal axis with the given percentile of radial distances as radius.
    public static func fitCapsule(_ points: [Vec3], radiusPercentile: Double) -> Capsule? {
        guard points.count >= 8 else { return nil }
        let mean = points.reduce(Vec3.zero, +) / Double(points.count)
        var c = [Double](repeating: 0, count: 9)
        for p in points {
            let d = p - mean
            for i in 0..<3 { for j in 0..<3 { c[3 * i + j] += d[i] * d[j] } }
        }
        // Principal axis by power iteration.
        var axis = Vec3(1, 0.3, 0.1)
        for _ in 0..<50 {
            let next = Vec3(c[0] * axis.x + c[1] * axis.y + c[2] * axis.z,
                            c[3] * axis.x + c[4] * axis.y + c[5] * axis.z,
                            c[6] * axis.x + c[7] * axis.y + c[8] * axis.z)
            let n = length(next)
            if n < 1e-12 { break }
            axis = next / n
        }
        let t = points.map { dot($0 - mean, axis) }
        let radial = zip(points, t).map { length($0.0 - (mean + $0.1 * axis)) }.sorted()
        let radius = radial[min(radial.count - 1, Int(radiusPercentile * Double(radial.count)))]
        let lo = (t.min() ?? 0) + radius, hi = (t.max() ?? 0) - radius
        let mid = 0.5 * (lo + hi)
        return Capsule(a: mean + min(lo, mid) * axis, b: mean + max(hi, mid) * axis, radius: radius)
    }

    /// Deepest penetration of either link's surface points into the other link's capsule (≤ 0: apart).
    public func penetration(_ a: Int, _ ta: Transform, _ b: Int, _ tb: Transform) -> Double {
        guard let ca = capsules[a], let cb = capsules[b] else { return 0 }
        // Cheap reject with capsules grown to enclose all points.
        if RobotGeometry.penetration(ca, ta, cb, tb) < -0.05 { return -1 }
        func depth(_ points: [Vec3], _ tp: Transform, into c: Capsule, _ tc: Transform) -> Double {
            let a = tc.apply(c.a), b = tc.apply(c.b), ab = b - a
            let len2 = max(dot(ab, ab), 1e-12)
            var worst = -Double.infinity
            for point in points {
                let p = tp.apply(point)
                let t = max(0, min(1, dot(p - a, ab) / len2))
                worst = max(worst, c.radius - length(p - (a + t * ab)))
            }
            return worst
        }
        return max(depth(linkPoints[a], ta, into: cb, tb), depth(linkPoints[b], tb, into: ca, ta))
    }

    /// Penetration depth of two capsules in world placement (≤ 0: apart).
    public static func penetration(_ a: Capsule, _ ta: Transform, _ b: Capsule, _ tb: Transform) -> Double {
        let d = segmentDistance(ta.apply(a.a), ta.apply(a.b), tb.apply(b.a), tb.apply(b.b))
        return a.radius + b.radius - d
    }

    static func segmentDistance(_ p1: Vec3, _ q1: Vec3, _ p2: Vec3, _ q2: Vec3) -> Double {
        let (a, b) = closestPoints(p1, q1, p2, q2)
        return length(a - b)
    }

    /// Closest points of segments p1q1 and p2q2 (Ericson).
    public static func closestPoints(_ p1: Vec3, _ q1: Vec3, _ p2: Vec3, _ q2: Vec3) -> (Vec3, Vec3) {
        let d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2
        let a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r)
        var s = 0.0, t = 0.0
        if a <= 1e-12 && e <= 1e-12 { return (p1, p2) }
        if a <= 1e-12 {
            t = max(0, min(1, f / e))
        } else {
            let c = dot(d1, r)
            if e <= 1e-12 {
                s = max(0, min(1, -c / a))
            } else {
                let b = dot(d1, d2), denom = a * e - b * b
                s = denom > 1e-12 ? max(0, min(1, (b * f - c * e) / denom)) : 0
                t = (b * s + f) / e
                if t < 0 { t = 0; s = max(0, min(1, -c / a)) } else if t > 1 { t = 1; s = max(0, min(1, (b - c) / a)) }
            }
        }
        return (p1 + s * d1, p2 + t * d2)
    }
}
