// Reading robot trajectories from other tools for validation / comparison:
//   - our .npz (root_pos, root_quat wxyz, dof_pos, fps);
//   - NVIDIA soma-retargeter CSV: Frame, root_translate{X,Y,Z} (cm), root_rotate{X,Y,Z} (deg, extrinsic xyz
//     as scipy's 'xyz'), then actuated joints in degrees; 120 fps, no frame rate stored.

import Core

public struct ImportedRobotMotion: Sendable {
    public var fps: Double
    public var jointNames: [String]
    public var root: [Transform]
    public var q: [[Double]]   // in jointNames order
}

public enum RobotMotionImport {
    public static func npz(file: String) throws -> ImportedRobotMotion {
        let a = try NPZ.read(file: file)
        guard let fps = try a["fps"]?.doubles().first, let names = try a["joint_names"]?.strings(),
              let pos = try a["root_pos"]?.doubles(), let quat = try a["root_quat"]?.doubles(),
              let dof = try a["dof_pos"]?.doubles() else { throw NPYError(message: "\(file): not a robot motion npz") }
        let T = pos.count / 3, n = names.count
        return ImportedRobotMotion(
            fps: fps, jointNames: names,
            root: (0..<T).map { f in
                Transform(rotation: Quat(w: quat[4 * f], x: quat[4 * f + 1], y: quat[4 * f + 2], z: quat[4 * f + 3]).normalized,
                          translation: Vec3(pos[3 * f], pos[3 * f + 1], pos[3 * f + 2]))
            },
            q: (0..<T).map { f in Array(dof[(f * n)..<((f + 1) * n)]) })
    }

    public static func nvidiaCSV(file: String, fps: Double = 120) throws -> ImportedRobotMotion {
        guard let bytes = readFile(file) else { throw NPYError(message: "cannot read \(file)") }
        let lines = String(decoding: bytes, as: UTF8.self).split(whereSeparator: \.isNewline)  // files use \\r\\n
        guard let header = lines.first?.split(separator: ",").map({ String($0).filter { !$0.isWhitespace } }),
              header.count > 7, header[0] == "Frame" else { throw NPYError(message: "\(file): not a soma-retargeter CSV") }
        let names = Array(header[7...])
        let degrees = Double.pi / 180
        var root: [Transform] = [], q: [[Double]] = []
        for line in lines.dropFirst() {
            let v = line.split(separator: ",").map { Double($0.filter { !$0.isWhitespace }) ?? .nan }
            guard v.count == header.count else { continue }
            // Extrinsic x-y-z: R = Rz(c) · Ry(b) · Rx(a).
            let rotation = Quat(axis: Vec3(0, 0, 1), angle: v[6] * degrees) * Quat(axis: Vec3(0, 1, 0), angle: v[5] * degrees)
                * Quat(axis: Vec3(1, 0, 0), angle: v[4] * degrees)
            root.append(Transform(rotation: rotation.normalized, translation: Vec3(v[1], v[2], v[3]) * 0.01))
            q.append(v[7...].map { $0 * degrees })
        }
        return ImportedRobotMotion(fps: fps, jointNames: names, root: root, q: q)
    }
}
