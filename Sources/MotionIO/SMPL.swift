// SMPL-family motion files (SMPL / SMPL+H / SMPL-X parameters in .npz or .pkl) and the SMPL-X body model that poses
// them (DESIGN §17). Every layout is read into the SMPL-X joint order (55 joints: body 0-21, jaw and eyes 22-24,
// hands 25-54); the skeleton comes from the model for the clip's shape and gender.

import Core

/// How to read the parameter files of one dataset family: which keys hold what, the up axis and the frame rate.
public struct SMPLOptions: Sendable {
    public enum Up: String, Sendable { case auto, y, z }

    /// Role -> candidate keys, first present wins. Roles: root (global orientation), body (joints 1-21, or SMPL's
    /// 1-23), hands (30 hand joints), poses (all joints, root first: 22, 24 SMPL, 52 SMPL+H or 55 SMPL-X per frame),
    /// trans, betas, gender, fps.
    public var keys: [String: [String]] = [
        "root": ["root_orient", "global_orient", "Rh"],
        "body": ["pose_body", "body_pose"],
        "hands": ["pose_hand", "hand_pose"],
        "poses": ["poses", "fullpose", "full_pose"],
        "trans": ["trans", "transl"],
        "betas": ["betas", "shapes", "beta"],
        "gender": ["gender"],
        "fps": ["mocap_frame_rate", "mocap_framerate", "fps", "frame_rate", "framerate"],
    ]
    /// World up axis of the files; auto: per clip, the axis the pelvis's up direction is closest to on average.
    public var up: Up = .auto
    /// File units to metres.
    public var unitScale: Double = 1
    /// Frame rate when a file has none.
    public var defaultFPS: Double = 30
    /// Gender of every clip (neutral, male, female); nil: the file's own, neutral when it has none.
    public var gender: String?

    public init() {}
}

/// One clip's parameters in the SMPL-X joint order.
public struct SMPLClip: Sendable {
    /// Per frame, 55 axis-angle rotations (radians); joints the file does not have are zero.
    public var poses: [[Vec3]]
    /// Per frame model translation (file world frame, metres).
    public var trans: [Vec3]
    public var betas: [Double]
    public var gender: String
    public var fps: Double
    /// The layout read, e.g. "root+body" or "poses/52 (SMPL+H)".
    public var layout: String

    public var frameCount: Int { poses.count }

    public init(poses: [[Vec3]], trans: [Vec3], betas: [Double], gender: String, fps: Double, layout: String) {
        self.poses = poses
        self.trans = trans
        self.betas = betas
        self.gender = gender
        self.fps = fps
        self.layout = layout
    }
}

public enum SMPLFile {
    public static func arrays(file: String) throws -> [String: NPYArray] {
        file.hasSuffix(".pkl") ? try Pickle.arrays(file: file) : try NPZ.read(file: file)
    }

    public static func read(file: String, options: SMPLOptions = SMPLOptions()) throws -> SMPLClip {
        let a = try arrays(file: file)
        func error(_ m: String) -> NPYError { NPYError(message: "\(file): \(m)") }
        func first(_ role: String) -> (key: String, array: NPYArray)? {
            for key in options.keys[role] ?? [] { if let v = a[key] { return (key, v) } }
            return nil
        }
        let root = first("root"), body = first("body"), hands = first("hands"), poses = first("poses")
        let transArray = first("trans")
        guard let frames = (transArray ?? root ?? poses)?.array.shape.first, frames > 0 else {
            throw error("no frames (keys \(a.keys.sorted().joined(separator: ", ")))")
        }
        /// Rotations per frame of an array, (F, n, 3) or (F, 3n).
        func rotations(_ entry: (key: String, array: NPYArray)) throws -> (n: Int, values: [Double]) {
            let v = try entry.array.doubles()
            guard v.count % (3 * frames) == 0 else { throw error("\(entry.key) has \(v.count) values for \(frames) frames") }
            return (v.count / (3 * frames), v)
        }
        var out = [[Vec3]](repeating: [Vec3](repeating: .zero, count: 55), count: frames)
        /// Copies joints [from, from + count) of `entry` to SMPL-X joints starting at `to`.
        func place(_ r: (n: Int, values: [Double]), from: Int = 0, count: Int, to: Int) {
            for f in 0..<frames {
                for k in 0..<count {
                    let i = 3 * (f * r.n + from + k)
                    out[f][to + k] = Vec3(r.values[i], r.values[i + 1], r.values[i + 2])
                }
            }
        }
        var layout: String
        if let root, let body {
            let r = try rotations(root), b = try rotations(body)
            guard r.n == 1, b.n == 21 || b.n == 23 else { throw error("\(root.key) / \(body.key): \(r.n) / \(b.n) joints") }
            place(r, count: 1, to: 0)
            place(b, count: 21, to: 1)   // SMPL's two extra body joints are its hands: not used
            layout = "\(root.key)+\(body.key)"
        } else if let poses {
            let p = try rotations(poses)
            switch p.n {
            case 55: place(p, count: 55, to: 0); layout = "poses/55 (SMPL-X)"
            case 52: place(p, count: 22, to: 0); place(p, from: 22, count: 30, to: 25); layout = "poses/52 (SMPL+H)"
            case 24: place(p, count: 22, to: 0); layout = "poses/24 (SMPL)"
            case 22: place(p, count: 22, to: 0); layout = "poses/22 (body)"
            default: throw error("\(poses.key) has \(p.n) joints per frame (22, 24, 52 or 55 expected)")
            }
            if let root {
                let r = try rotations(root)
                guard r.n == 1 else { throw error("\(root.key) has \(r.n) rotations per frame") }
                place(r, count: 1, to: 0)
                layout += " + \(root.key)"
            }
        } else {
            throw error("no pose arrays (keys \(a.keys.sorted().joined(separator: ", ")))")
        }
        if let hands, root != nil, body != nil {
            let h = try rotations(hands)
            guard h.n == 30 else { throw error("\(hands.key) has \(h.n) joints per frame (30 expected)") }
            place(h, count: 30, to: 25)
            layout += "+\(hands.key)"
        }

        var trans = [Vec3](repeating: .zero, count: frames)
        if let t = transArray {
            let v = try t.array.doubles()
            guard v.count == 3 * frames else { throw error("\(t.key) has \(v.count) values for \(frames) frames") }
            trans = (0..<frames).map { Vec3(v[3 * $0], v[3 * $0 + 1], v[3 * $0 + 2]) * options.unitScale }
        }
        var betas: [Double] = []
        if let b = first("betas") {
            let v = try b.array.doubles()
            // (B,), (1, B) or per frame (F, B): the first row.
            let width = b.array.shape.count >= 2 ? b.array.shape.last! : v.count
            betas = Array(v.prefix(width))
        }
        let fileGender = try first("gender")?.array.strings().first?.lowercased()
        let gender = options.gender ?? fileGender.flatMap { ["neutral", "male", "female"].contains($0) ? $0 : nil } ?? "neutral"
        let fps = try first("fps")?.array.doubles().first ?? options.defaultFPS
        guard fps > 0 else { throw error("frame rate \(fps)") }
        return SMPLClip(poses: out, trans: trans, betas: betas, gender: gender, fps: fps, layout: layout)
    }
}

/// The SMPL-X linear blend skinning model, as exported by tools/export_smplx_model.py (model frame: Y-up, facing +Z).
public struct SMPLModel: Sendable {
    public var jointNames: [String]
    public var parents: [Int?]
    public var vTemplate: [Vec3]
    /// [vertex][component] offsets.
    public var shapedirs: [[Vec3]]
    public var jTemplate: [Vec3]
    public var jShapedirs: [[Vec3]]
    public var weightJoints: [Int32]
    public var weightValues: [Float]
    public var influences: Int
    public var triangles: [SIMD3<Int32>]

    public var betaCount: Int { jShapedirs.first?.count ?? 0 }

    public static func load(file: String) throws -> SMPLModel {
        let a = try NPZ.read(file: file)
        func array(_ key: String) throws -> NPYArray {
            guard let v = a[key] else { throw NPYError(message: "\(file): missing \(key) (re-run tools/export_smplx_model.py)") }
            return v
        }
        func vec3s(_ v: [Double]) -> [Vec3] { stride(from: 0, to: v.count, by: 3).map { Vec3(v[$0], v[$0 + 1], v[$0 + 2]) } }
        /// (N, 3, B) -> [N][B] Vec3.
        func dirs(_ key: String) throws -> [[Vec3]] {
            let x = try array(key), v = try x.doubles()
            let n = x.shape[0], b = x.shape[2]
            return (0..<n).map { i in (0..<b).map { k in Vec3(v[(i * 3) * b + k], v[(i * 3 + 1) * b + k], v[(i * 3 + 2) * b + k]) } }
        }
        let joints = try array("weight_joints"), tri = try array("triangles").ints()
        return SMPLModel(
            jointNames: try array("joint_names").strings(),
            parents: try array("parents").ints().map { $0 < 0 ? nil : $0 },
            vTemplate: vec3s(try array("v_template").doubles()),
            shapedirs: try dirs("shapedirs"),
            jTemplate: vec3s(try array("j_template").doubles()),
            jShapedirs: try dirs("j_shapedirs"),
            weightJoints: try joints.ints().map { Int32($0) },
            weightValues: try array("weight_values").doubles().map { Float($0) },
            influences: joints.shape.last ?? 1,
            triangles: stride(from: 0, to: tri.count, by: 3).map { SIMD3(Int32(tri[$0]), Int32(tri[$0 + 1]), Int32(tri[$0 + 2])) })
    }

    /// Rest joint positions for a shape (model frame); betas beyond the model's components are ignored.
    public func joints(betas: [Double]) -> [Vec3] {
        shaped(jTemplate, jShapedirs, betas)
    }

    /// Rest vertices for a shape (model frame).
    public func vertices(betas: [Double]) -> [Vec3] {
        shaped(vTemplate, shapedirs, betas)
    }

    func shaped(_ template: [Vec3], _ dirs: [[Vec3]], _ betas: [Double]) -> [Vec3] {
        let n = min(betas.count, betaCount)
        return template.indices.map { i in
            var p = template[i]
            for k in 0..<n { p += betas[k] * dirs[i][k] }
            return p
        }
    }

    /// The clip as a HumanMotion in our frame (metres, Z-up, the body's rest pose facing +X), joints renamed by
    /// `renames` (model name -> motion name). `up`: the file's world up axis (auto resolved by `upAxis`).
    public func motion(_ clip: SMPLClip, up: SMPLOptions.Up, renames: [String: String] = [:], name: String = "smplx") -> HumanMotion {
        let J = joints(betas: clip.betas)
        let offsets = jointNames.indices.map { j in YUpToZUp.point(parents[j].map { J[j] - J[$0] } ?? J[j]) }
        let skeleton = HumanSkeleton(name: name, jointNames: jointNames.map { renames[$0] ?? $0 }, parents: parents,
                                     restOffsets: offsets)
        let zUp = (up == .auto ? Self.upAxis(clip) : up) == .z
        let toModel = YUpToZUp.rotation.conjugate
        let local = clip.poses.indices.map { f -> [Transform] in
            jointNames.indices.map { j in
                let q = Self.rotation(clip.poses[f][j])
                guard j == 0 else { return Transform(rotation: YUpToZUp.rotation(q), translation: offsets[j]) }
                // The root turns about the pelvis joint and the translation moves it: pelvis = J0 + trans (file frame).
                let pelvis = J[0] + clip.trans[f]
                return zUp ? Transform(rotation: (q * toModel).normalized, translation: pelvis)
                           : Transform(rotation: YUpToZUp.rotation(q), translation: YUpToZUp.point(pelvis))
            }
        }
        return HumanMotion(skeleton: skeleton, fps: clip.fps, local: local)
    }

    /// World up axis of a clip: the one the pelvis's up direction (model +Y turned by the root) is closest to on
    /// average (Y for camera-frame fits, Z for AMASS-style captures).
    public static func upAxis(_ clip: SMPLClip) -> SMPLOptions.Up {
        var mean = Vec3.zero
        for pose in clip.poses { mean += rotation(pose[0]).rotate(Vec3(0, 1, 0)) }
        return abs(mean.z) > abs(mean.y) ? .z : .y
    }

    /// Axis-angle vector to a quaternion.
    public static func rotation(_ v: Vec3) -> Quat {
        let angle = length(v)
        return angle < 1e-12 ? .identity : Quat(axis: v / angle, angle: angle)
    }
}
