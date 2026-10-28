import Core
import MotionIO

let g1Dir = repositoryRoot + "/assets/robots/unitree_g1"

// Unitree SDK motor order.
let g1_29Order = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
let g1_23Order = Array(g1_29Order[0..<13]) + [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint",
]

for (dir, expected) in [("g1_29_dof", g1_29Order), ("g1_23_dof", g1_23Order)] {
    test("load \(dir)") {
        let robot = try RobotUSDLoader.load(directory: g1Dir + "/" + dir)
        print("  \(robot.name): \(robot.links.count) links, \(robot.dofCount) dof, mass \(robot.totalMass) kg")
        expect(robot.dofNames == expected, "dof order \(robot.dofNames)")
        for (i, link) in robot.links.enumerated() {
            if let p = link.parent { expect(p < i, "topological order at \(link.name)") }
            if link.dofIndex != nil {
                expect(link.limits.map { $0.lower < $0.upper && $0.effort != nil } == true, "limits of \(link.name)")
                expect(link.inertial != nil, "inertial of \(link.name)")
            }
        }
        expectClose(robot.defaultRootPose.translation.z, 0.793, tolerance: 1e-9, "pelvis height")
        expect(robot.links.allSatisfy { !$0.visuals.isEmpty || $0.name.contains("toe") || $0.name.contains("rubber") },
               "every link has a visual: \(robot.links.filter { $0.visuals.isEmpty }.map(\.name))")
        let feet = robot.links.filter { $0.name.hasSuffix("ankle_roll_link") }
        expect(feet.count == 2 && feet.allSatisfy { $0.collisions.count == 4 }, "4 sole spheres per foot")
    }

    test("joint frames match link xforms at q = 0 (\(dir))") {
        let robot = try RobotUSDLoader.load(directory: g1Dir + "/" + dir)
        let world = robot.forwardKinematics(root: robot.defaultRootPose, q: Array(repeating: 0, count: robot.dofCount))
        // Soles: lowest sole-sphere point should touch z = 0 in the default pose.
        let soleZ = robot.links.indices.filter { robot.links[$0].name.hasSuffix("ankle_roll_link") }.flatMap { i in
            robot.links[i].collisions.map { c -> Double in
                guard case .sphere(let r) = c.shape else { return .infinity }
                return world[i].apply(c.origin.translation).z - r
            }
        }
        print("  sole sphere bottoms: \(soleZ.map { ($0 * 1e4).rounded() / 1e4 })")

        // Compose the authored xformOps of base.usda independently and compare every link pose.
        let base = try USDLayer(file: g1Dir + "/" + dir + "/payloads/base.usda")
        var authored: [String: Transform] = [:]
        func walk(_ prim: USDPrim, _ parent: Transform) {
            let pose = parent * prim.localTransform
            if prim.typeName == "Xform" { authored[prim.name] = authored[prim.name] ?? pose }
            for child in prim.children { walk(child, pose) }
        }
        walk(base.prims[0], .identity)
        var worst = 0.0
        for (i, link) in robot.links.enumerated() {
            guard let a = authored[link.name] else { expect(false, "no authored prim for \(link.name)"); continue }
            worst = max(worst, length(a.translation - world[i].translation), Quat.angle(a.rotation, world[i].rotation))
        }
        print("  max deviation from authored link poses: \(worst)")
        expect(worst < 1e-6, "joint-frame FK vs authored xforms")
        expect(soleZ.allSatisfy { abs($0) < 0.01 }, "soles within 1 cm of ground")
    }
}

let sampleBVH = "/home/troja/Downloads/soma_uniform/bvh/230322/sweep_floor_side_walk_R_004__A298.bvh"

test("load SOMA BVH") {
    let start = ContinuousClock.now
    let motion = try BVH.load(file: sampleBVH)
    print("  \(motion.frameCount) frames @ \(motion.fps) fps, \(motion.skeleton.jointCount) joints, parsed in \(ContinuousClock.now - start)")
    expect(motion.skeleton.jointCount == 78, "78 SOMA nodes")
    expect(motion.frameCount == 734, "frame count")
    expectClose(motion.fps, 120, tolerance: 0.01, "fps")
    let names = motion.skeleton.jointNames
    expect(names.first == "Root" && names[1] == "Hips" && names.last == "RightToeEnd", "joint names")

    let world = motion.globalTransforms(frame: 0)
    func z(_ name: String) -> Double { world[motion.skeleton.jointIndex(named: name)!].translation.z }
    func p(_ name: String) -> Vec3 { world[motion.skeleton.jointIndex(named: name)!].translation }
    print("  frame 0: hips z \(z("Hips")), head z \(z("Head")), toes z \(z("LeftToeBase")), \(z("RightToeBase"))")
    expect(z("Head") > z("Hips") && z("Hips") > z("LeftFoot"), "Z-up ordering head > hips > foot")
    expect(z("Hips") > 0.8 && z("Hips") < 1.1, "hip height in metres")
    // Handedness: with forward taken from heel→toe, the left shoulder must lie on the left (up × forward).
    var forward = (p("LeftToeBase") - p("LeftFoot")) + (p("RightToeBase") - p("RightFoot"))
    forward.z = 0
    let left = cross(Vec3(0, 0, 1), normalize(forward))
    let side = dot(p("LeftArm") - p("RightArm"), left)
    print("  shoulder separation along left axis: \(side) m")
    expect(side > 0.2, "left side is on the left (right-handed Z-up)")

    // Rest skeleton must equal the BONES-SEED base rig.
    let rig = try BVH.load(file: repositoryRoot + "/assets/body_models/soma/seed_soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh")
    expect(rig.skeleton.jointNames == names, "base rig joint names")
    let offsetError = zip(rig.skeleton.restOffsets, motion.skeleton.restOffsets).dropFirst(2).map { length($0 - $1) }.max() ?? 0
    print("  max rest-offset difference vs base rig: \(offsetError) m")
    expect(offsetError < 1e-6, "uniform skeleton matches base rig")
}


test("teleop wire messages: round trip, malformed input, Python bytes") {
    func hex(_ s: String) -> [UInt8] {
        stride(from: 0, to: s.count, by: 2).map { UInt8(s.dropFirst($0).prefix(2), radix: 16)! }
    }
    let start = StreamStart(streamId: 3, fps: 120, skeleton: "soma77", jointNames: ["Root", "Hips"], parents: [-1, 0],
                            restOffsets: [0, 0, 0, 0, 0, 1])
    let robot = RobotFrame(streamId: 3, seq: 1 << 40, tSource: 1.25, tReceive: 1.5, tEmit: 1.75, rootPos: [1, 2, 3],
                           rootQuat: [1, 0, 0, 0], dofPos: [0.5, -0.5], dofVel: [0, 1], contact: [1, 0], groundHeight: -0.01,
                           cost: 0.25, flags: 6)
    for message in [StreamMessage.streamStart(start), .robotFrame(robot), .streamEnd(StreamEnd(streamId: 9))] {
        let bytes = message.encode()
        expect((try? StreamMessage.decode(bytes)) == message, "round trip \(message)")
        expect((try? StreamMessage.decode(Array(bytes.dropLast()))) == nil, "truncated message rejected")
        var corrupt = bytes
        corrupt[0] ^= 0xff
        expect((try? StreamMessage.decode(corrupt)) == nil, "bad magic rejected")
    }
    // A forged element count must not make the decoder allocate or read past the message.
    var forged = StreamStart(streamId: 1, fps: 30, skeleton: "", jointNames: [], parents: [], restOffsets: []).encode()
    forged[24] = 0xff; forged[25] = 0xff; forged[26] = 0xff; forged[27] = 0x7f   // skeleton name length 2^31 - 1
    expect((try? StreamMessage.decode(forged)) == nil, "forged element count rejected")
    var trailing = StreamEnd(streamId: 1).encode() + [0]
    trailing[8] += 1
    expect((try? StreamMessage.decode(trailing)) == nil, "trailing bytes rejected")
    // Bytes from python/retargeter_client/messages.py (the generated Python encoder).
    let python = try StreamMessage.decode(hex("52544753010002003800000007000000141a99be1c000000000000000000294003000000cdcccc3dcdcc4cbe9a99993e040000000000803f000000000000000000000000"))
    expect(python == .humanFrame(HumanFrame(streamId: 7, seq: 123456789012, tSource: 12.5, jointPos: [0.1, -0.2, 0.3],
                                            jointQuat: [1, 0, 0, 0])), "Python HumanFrame")
    let info = try StreamMessage.decode(hex("52544753010004003000000002000000010600000068c3a96c6c6f020000006731020000000100000061020000006263000000007b14ae47e17aa43f"))
    expect(info == .streamInfo(StreamInfo(streamId: 2, status: 1, message: "héllo", robot: "g1", dofNames: ["a", "bc"],
                                          contactGroups: [], lag: 0.04)), "Python StreamInfo")
}

finishTests()
