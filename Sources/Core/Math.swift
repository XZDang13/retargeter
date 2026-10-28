// Double-precision rigid-body math. This is the reference path; device kernels use float32.

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public typealias Vec3 = SIMD3<Double>

@inlinable public func dot(_ a: Vec3, _ b: Vec3) -> Double { (a * b).sum() }
@inlinable public func cross(_ a: Vec3, _ b: Vec3) -> Vec3 {
    Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}
@inlinable public func length(_ a: Vec3) -> Double { dot(a, a).squareRoot() }
@inlinable public func normalize(_ a: Vec3) -> Vec3 { a / length(a) }

/// Unit quaternion, scalar first (w, x, y, z) — the convention used in all our files and streams.
public struct Quat: Equatable, Sendable {
    public var w: Double
    public var v: Vec3

    @inlinable public init(w: Double, x: Double, y: Double, z: Double) {
        self.w = w
        self.v = Vec3(x, y, z)
    }

    @inlinable public init(w: Double, v: Vec3) {
        self.w = w
        self.v = v
    }

    public static let identity = Quat(w: 1, x: 0, y: 0, z: 0)

    @inlinable public init(axis: Vec3, angle: Double) {
        let half = 0.5 * angle
        self.init(w: cos(half), v: normalize(axis) * sin(half))
    }

    @inlinable public var conjugate: Quat { Quat(w: w, v: -v) }
    @inlinable public var norm: Double { (w * w + dot(v, v)).squareRoot() }
    @inlinable public var normalized: Quat { let n = norm; return Quat(w: w / n, v: v / n) }

    @inlinable public static func * (a: Quat, b: Quat) -> Quat {
        Quat(w: a.w * b.w - dot(a.v, b.v), v: a.w * b.v + b.w * a.v + cross(a.v, b.v))
    }

    @inlinable public func rotate(_ p: Vec3) -> Vec3 {
        let t = 2 * cross(v, p)
        return p + w * t + cross(v, t)
    }

    /// Row-major 3x3 rotation matrix.
    public var matrix: [Double] {
        let (x, y, z) = (v.x, v.y, v.z)
        return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    }

    /// From a row-major 3x3 rotation matrix (Shepperd's method).
    public init(matrix m: [Double]) {
        let trace = m[0] + m[4] + m[8]
        if trace > 0 {
            let s = (trace + 1).squareRoot() * 2
            self.init(w: 0.25 * s, x: (m[7] - m[5]) / s, y: (m[2] - m[6]) / s, z: (m[3] - m[1]) / s)
        } else if m[0] > m[4] && m[0] > m[8] {
            let s = (1 + m[0] - m[4] - m[8]).squareRoot() * 2
            self.init(w: (m[7] - m[5]) / s, x: 0.25 * s, y: (m[1] + m[3]) / s, z: (m[2] + m[6]) / s)
        } else if m[4] > m[8] {
            let s = (1 + m[4] - m[0] - m[8]).squareRoot() * 2
            self.init(w: (m[2] - m[6]) / s, x: (m[1] + m[3]) / s, y: 0.25 * s, z: (m[5] + m[7]) / s)
        } else {
            let s = (1 + m[8] - m[0] - m[4]).squareRoot() * 2
            self.init(w: (m[3] - m[1]) / s, x: (m[2] + m[6]) / s, y: (m[5] + m[7]) / s, z: 0.25 * s)
        }
        self = normalized
    }

    /// Rotation angle (rad) between two orientations, in [0, π].
    /// Uses atan2 on the relative rotation: acos(|a·b|) loses precision for nearby orientations.
    public static func angle(_ a: Quat, _ b: Quat) -> Double {
        let r = a.conjugate * b
        return 2 * atan2(length(r.v), abs(r.w))
    }
}

/// Rigid transform: p ↦ rotation · p + translation.
public struct Transform: Equatable, Sendable {
    public var rotation: Quat
    public var translation: Vec3

    @inlinable public init(rotation: Quat = .identity, translation: Vec3 = .zero) {
        self.rotation = rotation
        self.translation = translation
    }

    public static let identity = Transform()

    @inlinable public static func * (a: Transform, b: Transform) -> Transform {
        Transform(rotation: a.rotation * b.rotation, translation: a.rotation.rotate(b.translation) + a.translation)
    }

    @inlinable public func apply(_ p: Vec3) -> Vec3 { rotation.rotate(p) + translation }

    @inlinable public var inverse: Transform {
        let r = rotation.conjugate
        return Transform(rotation: r, translation: -r.rotate(translation))
    }
}

/// Fixed change of basis from Y-up / +Z-forward (SOMA, BVH, SMPL-X) to our Z-up / +X-forward world.
/// (x, y, z) ↦ (z, x, y); a proper rotation, so left stays +Y.
public enum YUpToZUp {
    public static let rotation = Quat(matrix: [0, 0, 1, 1, 0, 0, 0, 1, 0])

    @inlinable public static func point(_ p: Vec3) -> Vec3 { Vec3(p.z, p.x, p.y) }
    /// Conjugates a rotation expressed in the Y-up basis into the Z-up basis.
    @inlinable public static func rotation(_ q: Quat) -> Quat { Quat(w: q.w, v: point(q.v)) }
}
