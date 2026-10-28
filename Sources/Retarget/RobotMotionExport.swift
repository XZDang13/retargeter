// Resampling and NPZ export of retargeted robot motion (DESIGN §10).

import Contact
import Core
import MotionIO

#if canImport(Glibc)
import Glibc
#endif

extension Quat {
    /// Spherical interpolation along the shortest arc.
    public static func slerp(_ a: Quat, _ b: Quat, _ t: Double) -> Quat {
        var b = b
        var d = a.w * b.w + dot(a.v, b.v)
        if d < 0 { b = Quat(w: -b.w, v: -b.v); d = -d }
        if d > 0.9995 { return Quat(w: a.w + t * (b.w - a.w), v: a.v + t * (b.v - a.v)).normalized }
        let theta = acos(d), s = sin(theta)
        let wa = sin((1 - t) * theta) / s, wb = sin(t * theta) / s
        return Quat(w: wa * a.w + wb * b.w, v: wa * a.v + wb * b.v).normalized
    }

    /// Rotation vector (axis · angle) of the relative rotation from a to b, in world frame.
    public static func deltaVector(from a: Quat, to b: Quat) -> Vec3 {
        var r = b * a.conjugate
        if r.w < 0 { r = Quat(w: -r.w, v: -r.v) }
        let s = length(r.v)
        return s < 1e-12 ? 2 * r.v : r.v * (2 * atan2(s, r.w) / s)
    }
}

extension RobotTrajectory {
    /// Linear (joints, root position) / spherical (root rotation) resampling to `fps`.
    public func resampled(to fps: Double) -> RobotTrajectory {
        guard frameCount > 1, abs(fps - self.fps) > 1e-9 else { return self }
        let duration = Double(frameCount - 1) / self.fps
        let count = Int((duration * fps).rounded(.down)) + 1
        var out = RobotTrajectory(fps: fps, root: [], q: [], cost: [])
        for i in 0..<count {
            let x = Double(i) / fps * self.fps
            let a = min(Int(x), frameCount - 1), b = min(a + 1, frameCount - 1), t = x - Double(a)
            out.root.append(Transform(rotation: Quat.slerp(root[a].rotation, root[b].rotation, t),
                                      translation: root[a].translation + t * (root[b].translation - root[a].translation)))
            out.q.append(zip(q[a], q[b]).map { $0 + t * ($1 - $0) })
            out.cost.append(cost[a] + t * (cost[b] - cost[a]))
        }
        return out
    }
}

public struct ExportOptions: Sendable {
    public var fps: Double = 50
    public var withBodies = false
    public var source = ""
    public var configHash = ""
    public var status = "unvalidated"   // accepted / rejected (validator), or unvalidated
    public var rejectReasons: [String] = []
    public var warnings: [String] = []
    public var metricsJSON = "{}"
    public var mirrorOf = ""
    public var mode = "realtime"
    /// Per exported frame, from the validator's dynamics stage (written when present): joint torques (N·m),
    /// net ground force per link (N), unexplained wrench (units of m·g), ZMP on the ground (nil in flight).
    public var tau: [[Double]]?
    public var contactForce: [[Vec3]]?
    public var dynamicsResidual: [Double]?
    public var zmp: [Vec3?]?
    public init() {}
}

public enum RobotMotionExport {
    public static let version = "0.2.0"

    /// Writes one retargeted clip as .npz. Contacts (human-side, per group) are resampled by nearest frame.
    public static func write(_ result: RetargetResult, robot: RobotModel, to file: String, options: ExportOptions) throws {
        let traj = result.trajectory.resampled(to: options.fps)
        let T = traj.frameCount, n = robot.dofCount, dt = 1 / traj.fps
        var npz = NPZWriter()
        npz.add("fps", scalar: Float(traj.fps))
        npz.add("robot", string: robot.name)
        npz.add("joint_names", strings: robot.dofNames)
        npz.add("root_pos", float32: traj.root.flatMap { [Float($0.translation.x), Float($0.translation.y), Float($0.translation.z)] }, shape: [T, 3])
        npz.add("root_quat", float32: traj.root.flatMap { [Float($0.rotation.w), Float($0.rotation.v.x), Float($0.rotation.v.y), Float($0.rotation.v.z)] }, shape: [T, 4])
        npz.add("dof_pos", float32: traj.q.flatMap { $0.map(Float.init) }, shape: [T, n])

        // Central differences (one-sided at the ends).
        func span(_ i: Int) -> (Int, Int) { (max(0, i - 1), min(T - 1, i + 1)) }
        var dofVel: [Float] = [], linVel: [Float] = [], angVel: [Float] = []
        for i in 0..<T {
            let (a, b) = span(i)
            let h = Double(max(b - a, 1)) * dt
            dofVel += (0..<n).map { Float((traj.q[b][$0] - traj.q[a][$0]) / h) }
            let v = (traj.root[b].translation - traj.root[a].translation) / h
            let w = Quat.deltaVector(from: traj.root[a].rotation, to: traj.root[b].rotation) / h
            linVel += [Float(v.x), Float(v.y), Float(v.z)]
            angVel += [Float(w.x), Float(w.y), Float(w.z)]
        }
        npz.add("dof_vel", float32: dofVel, shape: [T, n])
        npz.add("root_lin_vel", float32: linVel, shape: [T, 3])
        npz.add("root_ang_vel", float32: angVel, shape: [T, 3])

        if options.withBodies {
            var pos: [Float] = [], rot: [Float] = []
            for i in 0..<T {
                for t in robot.forwardKinematics(root: traj.root[i], q: traj.q[i]) {
                    pos += [Float(t.translation.x), Float(t.translation.y), Float(t.translation.z)]
                    rot += [Float(t.rotation.w), Float(t.rotation.v.x), Float(t.rotation.v.y), Float(t.rotation.v.z)]
                }
            }
            npz.add("body_names", strings: robot.links.map(\.name))
            npz.add("body_pos", float32: pos, shape: [T, robot.links.count, 3])
            npz.add("body_quat", float32: rot, shape: [T, robot.links.count, 4])
        }

        let c = result.contacts
        let G = c.groupNames.count
        let sourceFrame = { (i: Int) in min(c.frameCount - 1, Int((Double(i) * dt * c.fps).rounded())) }
        npz.add("contact_groups", strings: c.groupNames)
        npz.add("contact", uint8: (0..<T).flatMap { i in (0..<G).map { c.contact(sourceFrame(i), $0) ? 1 : 0 } }, shape: [T, G])
        npz.add("env_contact", uint8: (0..<T).flatMap { i in (0..<G).map { c.envContact[sourceFrame(i) * G + $0] ? 1 : 0 } }, shape: [T, G])
        npz.add("ground_height", float32: (0..<T).map { c.ground[sourceFrame($0)] }, shape: [T])
        npz.add("tracking_cost", float32: traj.cost.map(Float.init), shape: [T])
        npz.add("status", string: options.status)
        npz.add("reject_reasons", strings: options.rejectReasons)
        npz.add("warnings", strings: options.warnings)
        npz.add("metrics_json", string: options.metricsJSON)
        npz.add("human_flags", strings: c.flags)
        npz.add("source", string: options.source)
        npz.add("source_fps", scalar: Float(result.trajectory.fps))
        npz.add("config_hash", string: options.configHash)
        npz.add("mode", string: options.mode)
        npz.add("version", string: version)
        npz.add("mirror_of", string: options.mirrorOf)
        if let tau = options.tau, tau.count == T {
            npz.add("tau", float32: tau.flatMap { $0.map(Float.init) }, shape: [T, n])
        }
        if let forces = options.contactForce, forces.count == T {
            npz.add("link_names", strings: robot.links.map(\.name))
            npz.add("contact_force", float32: forces.flatMap { $0.flatMap { [Float($0.x), Float($0.y), Float($0.z)] } },
                    shape: [T, robot.links.count, 3])
        }
        if let residual = options.dynamicsResidual, residual.count == T {
            npz.add("dynamics_residual", float32: residual.map(Float.init), shape: [T])
        }
        if let zmp = options.zmp, zmp.count == T {
            npz.add("zmp", float32: zmp.flatMap { p in p.map { [Float($0.x), Float($0.y), Float($0.z)] } ?? [.nan, .nan, .nan] },
                    shape: [T, 3])
        }
        try npz.write(file: file)
    }
}
