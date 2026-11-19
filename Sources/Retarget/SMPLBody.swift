// SMPL-family clips (DESIGN §17): the skeleton and body surface come from the SMPL-X model for the clip's shape and
// gender, so unlike other custom skeletons the body is the real skinned mesh and the reference T-pose is the model's
// own rest pose (straightened like a custom skeleton's).

import Contact
import Core
import MotionIO
import Synchronization

public final class SMPLSource: Sendable {
    public let config: SkeletonConfig
    public let repository: String
    private let models = Mutex<[String: SMPLModel]>([:])

    public init(config: SkeletonConfig, repository: String) {
        self.config = config
        self.repository = repository
    }

    /// The body model of a gender (loaded once).
    public func model(_ gender: String) throws -> SMPLModel {
        if let m = models.withLock({ $0[gender] }) { return m }
        var path = config.model.replacing("{gender}", with: gender)
        if !path.hasPrefix("/") { path = repository + "/" + path }
        let m: SMPLModel
        do { m = try SMPLModel.load(file: path) } catch {
            throw NPYError(message: "\(error) — export the SMPL-X model first: python tools/export_smplx_model.py")
        }
        models.withLock { $0[gender] = m }
        return m
    }

    /// One file as a clip (parameters) and motion (canonical joint names).
    public func load(file: String) throws -> (clip: SMPLClip, motion: HumanMotion) {
        let clip = try SMPLFile.read(file: file, options: config.smpl)
        let model = try model(clip.gender)
        let motion = model.motion(clip, up: config.smpl.up, renames: try config.renames(for: model.jointNames),
                                  name: config.name)
        return (clip, motion)
    }

    /// Body surface of a clip's shape: the SMPL-X mesh (rest pose, our frame) skinned to the clip's skeleton, with
    /// the standard contact groups, and the reference T-pose (feet on z = 0, facing +X). The reference is built like
    /// a custom skeleton's from a standing frame of `motion` (config reference.frame: "auto" or an index), which keeps
    /// the collars as a person holds them with the arms down; "rest" uses the model's rest pose instead.
    public func surface(for clip: SMPLClip, motion: HumanMotion) throws -> (surface: BodySurface, frame: Int?) {
        let model = try model(clip.gender)
        let s = motion.skeleton
        let J = model.joints(betas: clip.betas).map(YUpToZUp.point)
        let bind = J.map { Transform(rotation: .identity, translation: $0) }
        var surface = BodySurface(jointNames: s.jointNames, restVertices: model.vertices(betas: clip.betas).map(YUpToZUp.point),
                                  bind: bind, weightJoints: model.weightJoints, weightValues: model.weightValues,
                                  influences: model.influences, triangles: model.triangles)
        // Eyeballs are inner geometry.
        let eyes = Set(["left_eye", "right_eye"].compactMap(model.jointNames.firstIndex(of:)))
        surface.excluded = (0..<surface.vertexCount).map { eyes.contains(Int(model.weightJoints[$0 * model.influences])) }
        surface.vertexGroups = (ContactGroups.standardNames, Self.groups(model, surface))
        let frame = config.referenceRest ? nil
            : min(max(config.referenceFrame ?? CustomBody.standingFrame(motion), 0), motion.frameCount - 1)
        var pose = CustomBody.tPose(s, world: frame.map { motion.globalTransforms(frame: $0) } ?? bind)
        let floor = surface.skin(world: pose).enumerated().filter { !surface.excluded[$0.offset] }.map(\.element.z).min() ?? 0
        pose = pose.map { Transform(rotation: $0.rotation, translation: $0.translation - Vec3(0, 0, floor)) }
        surface.referencePose = pose
        return (surface, frame)
    }

    /// Contact group of every vertex from its dominant skinning joint, as on the SOMA mesh (ContactGroups.soma):
    /// the lower leg split into knee (upper 35 %) and shin, the forearm into elbow (upper 30 %), forearm and the
    /// wrist end (last 15 %, hand), the collar with the upper arm.
    static func groups(_ model: SMPLModel, _ surface: BodySurface) -> [Int32] {
        let names = ContactGroups.standardNames
        func g(_ name: String) -> Int32 { Int32(names.firstIndex(of: name)!) }
        func j(_ name: String) -> Int { model.jointNames.firstIndex(of: name) ?? -1 }
        var byJoint: [Int: Int32] = [:]
        var split: [Int: (from: Vec3, to: Vec3, near: Int32, nearFraction: Double, far: Int32, distal: Int32?, distalFraction: Double)] = [:]
        let p = surface.bind.map(\.translation)
        for side in ["left", "right"] {
            byJoint[j("\(side)_hip")] = g("\(side)_thigh")
            split[j("\(side)_knee")] = (p[j("\(side)_knee")], p[j("\(side)_ankle")], g("\(side)_knee"), 0.35, g("\(side)_shin"), nil, 1)
            byJoint[j("\(side)_ankle")] = g("\(side)_heel")
            byJoint[j("\(side)_foot")] = g("\(side)_toe")
            byJoint[j("\(side)_collar")] = g("\(side)_upper_arm")
            byJoint[j("\(side)_shoulder")] = g("\(side)_upper_arm")
            split[j("\(side)_elbow")] = (p[j("\(side)_elbow")], p[j("\(side)_wrist")], g("\(side)_elbow"), 0.3,
                                          g("\(side)_forearm"), g("\(side)_hand"), 0.85)
            for (k, name) in model.jointNames.enumerated() where name == "\(side)_wrist"
                || ["index", "middle", "pinky", "ring", "thumb"].contains(where: { name.hasPrefix("\(side)_\($0)") }) {
                byJoint[k] = g("\(side)_hand")
            }
        }
        byJoint[j("pelvis")] = g("pelvis")
        for name in ["spine1", "spine2", "spine3"] { byJoint[j(name)] = g("torso") }
        for name in ["neck", "head", "jaw", "left_eye", "right_eye"] { byJoint[j(name)] = g("head") }

        return (0..<surface.vertexCount).map { v in
            let dominant = Int(model.weightJoints[v * model.influences])
            if let s = split[dominant] {
                let axis = s.to - s.from
                let t = dot(surface.restVertices[v] - s.from, axis) / dot(axis, axis)
                return t < s.nearFraction ? s.near : (s.distal != nil && t > s.distalFraction ? s.distal! : s.far)
            }
            return byJoint[dominant] ?? -1
        }
    }
}
