// Custom human skeletons (DESIGN §16). A skeleton config (Configs/skeletons/<name>.json) says how to read a
// skeleton's BVH files: units, up axis, and which of its joints are the canonical ones the mappings name (SOMA's
// Hips, Chest, LeftLeg = hip, LeftShin = knee, LeftFoot, LeftToeBase, LeftArm, LeftForeArm, LeftHand, Head). From
// one standing frame of a clip it builds what the SOMA path gets from the baked body model: the reference T-pose
// (rotation offsets, limb scales) and a body surface of capsules carrying the standard contact groups.

import Contact
import Core
import MotionIO

#if canImport(Glibc)
import Glibc
#endif

public struct SkeletonConfig: Sendable {
    public var name: String
    /// File units to metres.
    public var unitScale: Double
    /// File basis Y-up (BVH convention); false: already Z-up.
    public var yUp: Bool
    /// Canonical joint -> candidate file joints (the first one the file has is used).
    public var joints: [String: [String]]
    /// Frame of a clip whose pose (standing, feet flat) becomes the reference; nil: chosen per skeleton (the most
    /// upright frame with both feet planted and the body still).
    public var referenceFrame: Int?
    /// Body capsule radii (m, for a 1.75 m person; scaled by the skeleton's height), keyed like `Body.defaults`.
    public var radii: [String: Double]

    /// Joints every custom skeleton must provide.
    public static let required = ["Hips", "Chest", "Head"] + ["Left", "Right"].flatMap { s in
        ["Leg", "Shin", "Foot", "ToeBase", "Arm", "ForeArm", "Hand"].map { s + $0 }
    }

    public static func load(file: String) throws -> SkeletonConfig {
        let json = try JSONValue.load(file: file)
        var joints: [String: [String]] = [:]
        for (canonical, value) in json["joints"]?.object ?? [:] {
            if let name = value.string { joints[canonical] = [name] }
            else if let names = value.array?.compactMap(\.string), !names.isEmpty { joints[canonical] = names }
            else { throw JSONError(message: "\(file): joints.\(canonical) must be a joint name or a list of names") }
        }
        let frame = json["reference"]?["frame"]
        if let f = frame, f.string != "auto" && f.double == nil {
            throw JSONError(message: "\(file): reference.frame must be a frame index or \"auto\"")
        }
        let up = json["up"]?.string ?? "y"
        guard ["y", "z"].contains(up) else { throw JSONError(message: "\(file): up must be \"y\" or \"z\"") }
        return SkeletonConfig(name: json["name"]?.string ?? "custom", unitScale: json["units"]?.double ?? 0.01,
                              yUp: up == "y", joints: joints, referenceFrame: frame?.double.map { Int($0) },
                              radii: json["body"]?["radii"]?.object?.compactMapValues(\.double) ?? [:])
    }

    /// Reader options for one file skeleton (its joints as named in the file): the canonical joints renamed.
    public func bvhOptions(for fileSkeleton: HumanSkeleton) throws -> BVHOptions {
        var options = BVHOptions()
        options.unitScale = unitScale
        options.yUp = yUp
        options.skeletonName = name
        var missing: [String] = []
        for canonical in Set(Self.required).union(joints.keys).sorted() {
            let candidates = joints[canonical] ?? [canonical]
            if let found = candidates.first(where: { fileSkeleton.jointIndex(named: $0) != nil }) {
                options.jointRenames[found] = canonical
            } else if Self.required.contains(canonical) {
                missing.append("\(canonical) (\(candidates.joined(separator: " / ")))")
            }
        }
        guard missing.isEmpty else {
            throw JSONError(message: "skeleton \(name): no joint for \(missing.joined(separator: ", "))")
        }
        return options
    }

    /// Identity of a skeleton (joint names, hierarchy and bone lengths), as a short key: clips with the same key share
    /// a reference pose and body.
    public static func signature(_ skeleton: HumanSkeleton) -> String {
        var h: UInt64 = 1_469_598_103_934_665_603
        func mix(_ s: String) { for b in s.utf8 { h = (h ^ UInt64(b)) &* 1_099_511_628_211 } }
        for j in 0..<skeleton.jointCount {
            let o = skeleton.restOffsets[j]
            mix("\(skeleton.jointNames[j]) \(skeleton.parents[j] ?? -1) \(Int((o.x * 1e4).rounded())) \(Int((o.y * 1e4).rounded())) \(Int((o.z * 1e4).rounded()));")
        }
        return String(h, radix: 16)
    }

    /// Reader options that only fix units and basis (for reading a file's own joint names).
    public var rawOptions: BVHOptions {
        var options = BVHOptions()
        options.unitScale = unitScale
        options.yUp = yUp
        options.skeletonName = name
        return options
    }
}

/// A custom skeleton's reference pose and body surface, built from one clip.
public enum CustomBody {
    /// Body surface (capsules, one rigid joint per point, the standard contact groups) and reference T-pose of
    /// `motion`'s skeleton, with the reference taken at `frame` (nil: automatic). Feet on z = 0, facing +X.
    public static func make(_ motion: HumanMotion, config: SkeletonConfig) throws -> (surface: BodySurface, frame: Int) {
        let frame = min(max(config.referenceFrame ?? standingFrame(motion), 0), motion.frameCount - 1)
        var pose = tPose(motion.skeleton, world: motion.globalTransforms(frame: frame))
        var surface = capsules(motion.skeleton, reference: pose, radii: config.radii)
        let floor = surface.restVertices.map(\.z).min() ?? 0
        pose = pose.map { Transform(rotation: $0.rotation, translation: $0.translation - Vec3(0, 0, floor)) }
        surface.restVertices = surface.restVertices.map { $0 - Vec3(0, 0, floor) }
        surface.bind = pose
        surface.referencePose = pose
        return (surface, frame)
    }

    static func index(_ skeleton: HumanSkeleton, _ name: String) -> Int {
        skeleton.jointIndex(named: name)!   // checked by SkeletonConfig.bvhOptions
    }

    /// The frame that looks most like standing still with the arms hanging: both ankles near their lowest and slow,
    /// the head high, the hands low, the body slow. Frame 0 when no frame qualifies.
    static func standingFrame(_ motion: HumanMotion) -> Int {
        let s = motion.skeleton, F = motion.frameCount
        guard F > 2 else { return 0 }
        let step = max(1, Int((motion.fps / 30).rounded()))
        let frames = Array(stride(from: 0, to: F, by: step))
        let world = parallelMap(frames.count) { motion.globalTransforms(frame: frames[$0]).map(\.translation) }
        let feet = [index(s, "LeftFoot"), index(s, "RightFoot")], head = index(s, "Head"), hips = index(s, "Hips")
        let hands = [index(s, "LeftHand"), index(s, "RightHand")]
        let lowest = feet.map { j in world.map { $0[j].z }.min()! }
        let dt = Double(step) / motion.fps
        var best = 0, bestScore = -Double.infinity
        for k in world.indices {
            let other = world[k == 0 ? 1 : k - 1]
            func speed(_ j: Int) -> Double { length(world[k][j] - other[j]) / dt }
            let planted = zip(feet, lowest).allSatisfy { j, low in world[k][j].z < low + 0.04 && speed(j) < 0.1 }
            guard planted else { continue }
            let still = (0..<s.jointCount).map(speed).reduce(0, +) / Double(s.jointCount)
            // Arms hanging: the arm twist of the reference is the captured one only for a hanging arm (raised sideways).
            let raised = hands.map { max(0, world[k][$0].z - world[k][hips].z) }.reduce(0, +)
            let score = world[k][head].z - world[k][feet[0]].z - 0.05 * still - raised
            if score > bestScore { bestScore = score; best = frames[k] }
        }
        return best
    }

    /// Smallest rotation taking direction `a` to `b`.
    static func swing(_ a: Vec3, _ b: Vec3) -> Quat {
        let u = normalize(a), v = normalize(b)
        let axis = cross(u, v), c = dot(u, v)
        if length(axis) < 1e-9 {
            guard c < 0 else { return .identity }
            let side = abs(u.x) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 1, 0)
            return Quat(axis: cross(u, side), angle: .pi)
        }
        return Quat(axis: axis, angle: atan2(length(axis), c))
    }

    /// The standing pose of `world` turned into the reference T-pose, by swinging bones (each joint's twist about
    /// its bone stays as captured): the pelvis as it stood but square to +X, the spine and neck
    /// straight up, legs straight down, feet as they stood but pointing +X, arms straight out sideways.
    static func tPose(_ s: HumanSkeleton, world input: [Transform]) -> [Transform] {
        var world = input
        var children = [[Int]](repeating: [], count: s.jointCount)
        for (j, p) in s.parents.enumerated() { if let p { children[p].append(j) } }
        func subtree(_ j: Int) -> [Int] { [j] + children[j].flatMap(subtree) }
        func rotate(_ j: Int, _ q: Quat) {
            let pivot = world[j].translation
            for k in subtree(j) {
                world[k] = Transform(rotation: (q * world[k].rotation).normalized,
                                     translation: pivot + q.rotate(world[k].translation - pivot))
            }
        }
        func position(_ name: String) -> Vec3 { world[index(s, name)].translation }
        func aim(_ joint: String, toward child: Vec3, _ direction: Vec3) {
            let j = index(s, joint)
            rotate(j, swing(child - world[j].translation, direction))
        }
        // The pelvis as it stood (a standing person's pelvis is upright for the robot), turned to face +X. Making the
        // hips-to-chest line vertical instead tilted the robot's pelvis back 5° on a SOMA walk (lumbar curve).
        let hips = index(s, "Hips")
        var left = position("LeftLeg") - position("RightLeg")
        left.z = 0
        rotate(hips, swing(left, Vec3(0, 1, 0)))
        // Spine and neck: every bone on the path from the hips to the head vertical.
        var path: [Int] = []
        var k: Int? = index(s, "Head")
        while let j = k, j != hips { path.append(j); k = s.parents[j] }
        path.reverse()   // first spine joint ... head
        for (a, b) in zip(path.dropLast(), path.dropFirst()) {
            rotate(a, swing(world[b].translation - world[a].translation, Vec3(0, 0, 1)))
        }
        for side in ["Left", "Right"] {
            aim(side + "Leg", toward: position(side + "Shin"), Vec3(0, 0, -1))
            aim(side + "Shin", toward: position(side + "Foot"), Vec3(0, 0, -1))
            // Foot: as it stood (flat on the ground), only turned to point +X. Its pitch and roll are the captured
            // world ones, not what the body and leg swings above left of them.
            let foot = index(s, side + "Foot")
            let toe = input[index(s, side + "ToeBase")].translation - input[foot].translation
            let target = Quat(axis: Vec3(0, 0, 1), angle: -atan2(toe.y, toe.x)) * input[foot].rotation
            rotate(foot, (target * world[foot].rotation.conjugate).normalized)
            // Arm: hanging straight down first, then raised sideways about the forward axis (A- to T-pose), so its
            // twist is the hanging arm's whatever pose it was captured in.
            let out = Vec3(0, side == "Left" ? 1 : -1, 0)
            aim(side + "Arm", toward: position(side + "ForeArm"), Vec3(0, 0, -1))
            aim(side + "Arm", toward: position(side + "ForeArm"), out)
            aim(side + "ForeArm", toward: position(side + "Hand"), out)
            aim(side + "Hand", toward: handTip(s, world, side), out)
        }
        return world
    }

    /// Where the hand ends: its End Site, else the mean of its children (fingers), else 9 cm along the forearm.
    static func handTip(_ s: HumanSkeleton, _ world: [Transform], _ side: String) -> Vec3 {
        let hand = index(s, side + "Hand"), elbow = index(s, side + "ForeArm")
        if let tip = s.endSites[hand], length(tip) > 1e-4 { return world[hand].apply(tip) }
        let fingers = s.parents.indices.filter { s.parents[$0] == hand }
        if !fingers.isEmpty {
            return fingers.map { world[$0].translation }.reduce(.zero, +) / Double(fingers.count)
        }
        return world[hand].translation + 0.09 * normalize(world[hand].translation - world[elbow].translation)
    }

    /// Capsule radii (m) of a 1.75 m person.
    public static let defaultRadii: [String: Double] = [
        "thigh": 0.075, "knee": 0.055, "ankle": 0.04, "upper_arm": 0.045, "elbow": 0.04, "wrist": 0.03,
        "hand": 0.028, "pelvis": 0.11, "torso": 0.12, "neck": 0.055, "head": 0.1,
        "foot_width": 0.095, "heel_width": 0.07, "heel_back": 0.06, "sole_below_toe": 0.025,
    ]

    /// Point surface of capsules (rings ~1.5 cm apart) around the bones and boxes for the feet, in the reference
    /// pose; every point moves rigidly with one joint and belongs to one of the standard contact groups.
    static func capsules(_ s: HumanSkeleton, reference w: [Transform], radii: [String: Double]) -> BodySurface {
        func p(_ name: String) -> Vec3 { w[index(s, name)].translation }
        let head = index(s, "Head")
        let top = s.endSites[head].map { w[head].apply($0) } ?? p("Head") + Vec3(0, 0, 0.1)
        let soleZ = min(p("LeftToeBase").z, p("RightToeBase").z) - (radii["sole_below_toe"] ?? 0.025)
        let scale = max(0.5, (top.z + 0.02 - soleZ) / 1.75)
        func r(_ key: String) -> Double { scale * (radii[key] ?? defaultRadii[key]!) }
        let names = ContactGroups.standardNames
        func g(_ name: String) -> Int32 { Int32(names.firstIndex(of: name)!) }

        var points: [Vec3] = [], joints: [Int32] = [], groups: [Int32] = []
        func add(_ point: Vec3, _ joint: Int, _ group: Int32) {
            points.append(point); joints.append(Int32(joint)); groups.append(group)
        }
        let spacing = 0.015
        /// Capsule from a to b, radius r0 to r1, group by the fraction t along it.
        func capsule(_ a: Vec3, _ b: Vec3, _ r0: Double, _ r1: Double, _ joint: Int, _ group: (Double) -> Int32) {
            let axis = b - a, l = length(axis)
            guard l > 1e-6 else { return }
            let x = axis / l
            let side = abs(x.z) < 0.9 ? Vec3(0, 0, 1) : Vec3(1, 0, 0)
            let u = normalize(cross(x, side)), v = cross(x, u)
            func ring(_ c: Vec3, _ radius: Double, _ t: Double) {
                let n = max(8, Int((2 * .pi * radius / spacing).rounded()))
                for i in 0..<n {
                    let a = 2 * .pi * Double(i) / Double(n)
                    add(c + radius * (cos(a) * u + sin(a) * v), joint, group(t))
                }
            }
            let rings = max(2, Int((l / spacing).rounded()))
            for i in 0...rings {
                let t = Double(i) / Double(rings)
                ring(a + t * axis, r0 + t * (r1 - r0), t)
            }
            // Hemispherical caps.
            for (end, radius, sign, t) in [(a, r0, -1.0, 0.0), (b, r1, 1.0, 1.0)] {
                let steps = max(2, Int((0.5 * .pi * radius / spacing).rounded()))
                for i in 1...steps {
                    let phi = 0.5 * .pi * Double(i) / Double(steps)
                    ring(end + sign * radius * sin(phi) * x, radius * cos(phi), t)
                }
            }
        }

        for side in ["left", "right"] {
            let S = side == "left" ? "Left" : "Right"
            let leg = index(s, S + "Leg"), shin = index(s, S + "Shin"), foot = index(s, S + "Foot")
            let toe = index(s, S + "ToeBase"), arm = index(s, S + "Arm"), forearm = index(s, S + "ForeArm")
            let hand = index(s, S + "Hand")
            capsule(p(S + "Leg"), p(S + "Shin"), r("thigh"), r("knee"), leg) { _ in g("\(side)_thigh") }
            capsule(p(S + "Shin"), p(S + "Foot"), r("knee"), r("ankle"), shin) { $0 < 0.35 ? g("\(side)_knee") : g("\(side)_shin") }
            capsule(p(S + "Arm"), p(S + "ForeArm"), r("upper_arm"), r("elbow"), arm) { _ in g("\(side)_upper_arm") }
            capsule(p(S + "ForeArm"), p(S + "Hand"), r("elbow"), r("wrist"), forearm) {
                $0 < 0.3 ? g("\(side)_elbow") : $0 > 0.85 ? g("\(side)_hand") : g("\(side)_forearm")
            }
            capsule(p(S + "Hand"), handTip(s, w, S), r("hand"), r("hand"), hand) { _ in g("\(side)_hand") }

            // Foot: a box from behind the ankle to the toe tip, sole flat at `soleZ`; behind the toe joint it moves
            // with the foot (heel group), ahead of it with the toe.
            let ankle = p(S + "Foot"), ball = p(S + "ToeBase")
            var forward = ball - ankle
            forward.z = 0
            forward = normalize(forward)
            let lateral = cross(Vec3(0, 0, 1), forward)
            let tipPoint = s.endSites[toe].map { w[toe].apply($0) } ?? ball + 0.07 * scale * forward
            let heelU = -r("heel_back"), ballU = dot(ball - ankle, forward), tipU = max(ballU + 0.02, dot(tipPoint - ankle, forward))
            func width(_ u: Double) -> Double {
                u < ballU ? r("heel_width") + (r("foot_width") - r("heel_width")) * max(0, u - heelU) / (ballU - heelU)
                          : r("foot_width") * (1 - 0.3 * (u - ballU) / max(tipU - ballU, 1e-3))
            }
            func topZ(_ u: Double) -> Double {
                u < ballU ? ankle.z - 0.5 * (ankle.z - ball.z) * max(0, u) / max(ballU, 1e-3) : ball.z + 0.015 * scale
            }
            func point(_ u: Double, _ v: Double, _ z: Double) {
                let q = ankle + u * forward + v * lateral
                if u < ballU { add(Vec3(q.x, q.y, z), foot, g("\(side)_heel")) } else { add(Vec3(q.x, q.y, z), toe, g("\(side)_toe")) }
            }
            var u = heelU
            while u <= tipU + 1e-9 {
                let half = 0.5 * width(u), tz = max(topZ(u), soleZ + 0.02)
                var v = -half
                while v <= half + 1e-9 { point(u, v, soleZ); point(u, v, tz); v += 0.01 }
                var z = soleZ + spacing
                while z < tz { point(u, -half, z); point(u, half, z); z += spacing }
                u += 0.01
            }
            for end in [heelU, tipU] {
                let half = 0.5 * width(end)
                var v = -half
                while v <= half + 1e-9 {
                    var z = soleZ + spacing
                    while z < max(topZ(end), soleZ + 0.02) { point(end, v, z); z += spacing }
                    v += 0.01
                }
            }
        }

        // Trunk: the spine path from the hips to the head as capsules (pelvis around the hips, torso up to the
        // neck, then neck and head), the hip joints joined, the shoulders joined across the chest.
        let hips = index(s, "Hips"), chest = index(s, "Chest")
        var path: [Int] = []
        var k: Int? = head
        while let j = k { path.append(j); if j == hips { break }; k = s.parents[j] }
        path.reverse()   // hips ... head
        let chestAt = path.firstIndex(of: chest) ?? path.count - 2
        let pelvisTop = p("Hips") + Vec3(0, 0, 0.1 * scale)
        capsule(p("LeftLeg"), p("RightLeg"), r("pelvis") * 0.8, r("pelvis") * 0.8, hips) { _ in g("pelvis") }
        capsule(p("Hips"), pelvisTop, r("pelvis"), r("pelvis"), hips) { _ in g("pelvis") }
        for (i, (a, b)) in zip(path.dropLast(), path.dropFirst()).enumerated() {
            let from = i == 0 ? pelvisTop : w[a].translation
            guard dot(w[b].translation - from, Vec3(0, 0, 1)) > 0.005 else { continue }
            if i < chestAt {
                capsule(from, w[b].translation, r("torso"), r("torso"), a) { _ in g("torso") }
            } else if i == chestAt {
                capsule(from, w[b].translation, r("torso"), r("neck") * 1.6, a) { _ in g("torso") }
            } else {
                capsule(from, w[b].translation, r("neck"), r("neck"), a) { _ in g("head") }
            }
        }
        capsule(p("LeftArm"), p("RightArm"), r("upper_arm") * 1.2, r("upper_arm") * 1.2, chest) { _ in g("torso") }
        let center = p("Head") + 0.45 * (top - p("Head"))
        capsule(center - Vec3(0, 0, 0.3 * r("head")), center + Vec3(0, 0, 0.3 * r("head")), r("head"), r("head"), head) { _ in g("head") }

        var surface = BodySurface(jointNames: s.jointNames, restVertices: points, bind: w, weightJoints: joints,
                                  weightValues: Array(repeating: 1, count: points.count), influences: 1, triangles: [],
                                  referencePose: w)
        surface.vertexGroups = (names, groups)
        return surface
    }
}
