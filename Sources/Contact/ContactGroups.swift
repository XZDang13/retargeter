// Body-part contact groups on the skinned human surface (DESIGN §5a).

import Core

public enum ContactClass: Sendable {
    case foot   // heel, toe
    case limb   // knee, shin, thigh, hand, elbow, forearm, upper arm
    case trunk  // pelvis, torso, head
}

public struct ContactGroups: Sendable {
    public var names: [String]
    public var classes: [ContactClass]
    /// Vertex indices of each group (excluded inner geometry is never assigned).
    public var vertices: [[Int]]
    /// Fraction of body mass carried by each group (sums to 1), for the CoM estimate.
    public var massFractions: [Double]

    public var count: Int { names.count }

    public func index(of name: String) -> Int? { names.firstIndex(of: name) }

    /// Groups for the SOMA skeleton, from each vertex's dominant skinning joint. Shin and forearm are split
    /// into knee / elbow (the third nearest the joint) and the rest of the segment.
    public static func soma(_ surface: BodySurface) throws -> ContactGroups {
        let joint = { (name: String) throws -> Int in
            guard let j = surface.jointNames.firstIndex(of: name) else { throw ContactError(message: "no joint \(name)") }
            return j
        }
        var names: [String] = []
        var classes: [ContactClass] = []
        var byJoint: [Int: Int] = [:]            // dominant joint -> group
        /// Segments split along their bone: t < near → `near` group, t > distal → `distal` group, else `far`.
        var split: [Int: (near: Int, far: Int, from: Vec3, to: Vec3, fraction: Double, distal: Int?, distalFraction: Double)] = [:]

        func add(_ name: String, _ cls: ContactClass) -> Int {
            names.append(name)
            classes.append(cls)
            return names.count - 1
        }

        for side in ["left", "right"] {
            let s = side == "left" ? "Left" : "Right"
            let heel = add("\(side)_heel", .foot), toe = add("\(side)_toe", .foot)
            byJoint[try joint("\(s)Foot")] = heel
            byJoint[try joint("\(s)ToeBase")] = toe
            byJoint[try joint("\(s)ToeEnd")] = toe
            let knee = add("\(side)_knee", .limb), shin = add("\(side)_shin", .limb)
            split[try joint("\(s)Shin")] = (knee, shin, surface.bind[try joint("\(s)Shin")].translation,
                                            surface.bind[try joint("\(s)Foot")].translation, 0.35, nil, 1)
            byJoint[try joint("\(s)Leg")] = add("\(side)_thigh", .limb)
            let hand = add("\(side)_hand", .limb)
            for (j, name) in surface.jointNames.enumerated() where name.hasPrefix("\(s)Hand") { byJoint[j] = hand }
            let elbow = add("\(side)_elbow", .limb), forearm = add("\(side)_forearm", .limb)
            // The wrist end of the forearm belongs to the hand (it carries the hand's load).
            split[try joint("\(s)ForeArm")] = (elbow, forearm, surface.bind[try joint("\(s)ForeArm")].translation,
                                               surface.bind[try joint("\(s)Hand")].translation, 0.3, hand, 0.85)
            let upperArm = add("\(side)_upper_arm", .limb)
            byJoint[try joint("\(s)Arm")] = upperArm
            byJoint[try joint("\(s)Shoulder")] = upperArm
        }
        byJoint[try joint("Hips")] = add("pelvis", .trunk)
        let torso = add("torso", .trunk)
        for name in ["Spine1", "Spine2", "Chest"] { byJoint[try joint(name)] = torso }
        let head = add("head", .trunk)
        for name in ["Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye"] { byJoint[try joint(name)] = head }

        var vertices = [[Int]](repeating: [], count: names.count)
        for v in 0..<surface.vertexCount where !surface.excluded[v] {
            let dominant = Int(surface.weightJoints[v * surface.influences])
            if let s = split[dominant] {
                let axis = s.to - s.from
                let t = dot(surface.restVertices[v] - s.from, axis) / dot(axis, axis)
                vertices[t < s.fraction ? s.near : (s.distal != nil && t > s.distalFraction ? s.distal! : s.far)].append(v)
            } else if let g = byJoint[dominant] {
                vertices[g].append(v)
            }
        }
        // Segment masses after de Leva (1996), split across our finer groups.
        let mass: [String: Double] = [
            "heel": 0.007, "toe": 0.007, "knee": 0.012, "shin": 0.031, "thigh": 0.142, "hand": 0.006,
            "elbow": 0.004, "forearm": 0.012, "upper_arm": 0.027, "pelvis": 0.13, "torso": 0.30, "head": 0.069,
        ]
        var fractions = names.map { name in
            mass[name.hasPrefix("left_") ? String(name.dropFirst(5)) : name.hasPrefix("right_") ? String(name.dropFirst(6)) : name] ?? 0
        }
        let total = fractions.reduce(0, +)
        fractions = fractions.map { $0 / total }
        return ContactGroups(names: names, classes: classes, vertices: vertices, massFractions: fractions)
    }
}

public struct ContactError: Error, CustomStringConvertible {
    public let message: String
    public init(message: String) { self.message = message }
    public var description: String { message }
}
