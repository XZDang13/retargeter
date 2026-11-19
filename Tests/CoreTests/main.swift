import Core
import MotionIO
#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

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

let sampleBVH = bvhRoot + "230322/sweep_floor_side_walk_R_004__A298.bvh"

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

func hexBytes(_ hex: String) -> [UInt8] {
    var out: [UInt8] = [], digits = Array(hex.utf8)
    for i in stride(from: 0, to: digits.count - 1, by: 2) { out.append(UInt8(String(decoding: digits[i...i + 1], as: UTF8.self), radix: 16)!) }
    return out
}

func writeTemp(_ bytes: [UInt8], _ suffix: String) -> String {
    let path = "/tmp/retargeter_test_\(getpid())_\(suffix)"
    let f = fopen(path, "wb")!
    _ = bytes.withUnsafeBytes { fwrite($0.baseAddress, 1, bytes.count, f) }
    fclose(f)
    return path
}

test("pickle: numpy arrays, strings and numbers (protocols 2 and 4, numpy 1 and 2); code is refused") {
    // pickle.dumps({"betas": f4 (1, 3), "gender": np.array("female"), "trans": f8 (2, 3) / 4, "mocap_frame_rate": 30,
    // "x": 1.5, "name": "clip", "flags": [True, False], "n": np.int64(7)}): protocol 4 from numpy 2 (numpy._core),
    // protocol 2 with the module renamed to numpy 1's numpy.core (bytes as _codecs.encode).
    let protocol4 = hexBytes(
        "80049535020000000000007d94288c056265746173948c166e756d70792e5f636f72652e6d756c74696172726179948c0c5f" +
        "7265636f6e7374727563749493948c056e756d7079948c076e6461727261799493944b0085944301629487945294284b014b" +
        "014b03869468058c0564747970659493948c02663494898887945294284b038c013c944e4e4e4affffffff4affffffff4b00" +
        "74946289430c0000003f0000a0bf00000040947494628c0667656e64657294680468074b008594680987945294284b012968" +
        "0e8c02553694898887945294284b0368124e4e4e4b184b044b0874946289431866000000650000006d000000610000006c00" +
        "000065000000947494628c057472616e7394680468074b008594680987945294284b014b024b038694680e8c026638948988" +
        "87945294284b0368124e4e4e4affffffff4affffffff4b007494628943300000000000000000000000000000d03f00000000" +
        "0000e03f000000000000e83f000000000000f03f000000000000f43f947494628c106d6f6361705f6672616d655f72617465" +
        "944b1e8c017894473ff80000000000008c046e616d65948c04636c6970948c05666c61677394680468074b00859468098794" +
        "5294284b014b028594680e8c02623194898887945294284b038c017c944e4e4e4affffffff4affffffff4b00749462894302" +
        "0100947494628c016e9468028c067363616c6172949394680e8c02693894898887945294284b0368124e4e4e4affffffff4a" +
        "ffffffff4b00749462430807000000000000009486945294752e")
    let protocol2 = hexBytes(
        "80027d710028580500000062657461737101636e756d70792e636f72652e6d756c746961727261790a5f7265636f6e737472" +
        "7563740a7102636e756d70790a6e6461727261790a71034b00857104635f636f646563730a656e636f64650a710558010000" +
        "0062710658060000006c6174696e31710786710852710987710a52710b284b014b014b0386710c636e756d70790a64747970" +
        "650a710d58020000006634710e898887710f527110284b0358010000003c71114e4e4e4affffffff4affffffff4b00747112" +
        "62896805580e0000000000003f0000c2a0c2bf000000407113680786711452711574711662580600000067656e6465727117" +
        "680268034b00857118680987711952711a284b0129680d58020000005536711b898887711c52711d284b0368114e4e4e4b18" +
        "4b044b0874711e62896805581800000066000000650000006d000000610000006c00000065000000711f6807867120527121" +
        "7471226258050000007472616e737123680268034b008571246809877125527126284b014b024b03867127680d5802000000" +
        "66387128898887712952712a284b0368114e4e4e4affffffff4affffffff4b0074712b628968055835000000000000000000" +
        "0000000000000000c3903f000000000000c3a03f000000000000c3a83f000000000000c3b03f000000000000c3b43f712c68" +
        "0786712d52712e74712f6258100000006d6f6361705f6672616d655f7261746571304b1e5801000000787131473ff8000000" +
        "00000058040000006e616d6571325804000000636c697071335805000000666c6167737134680268034b0085713568098771" +
        "36527137284b014b02857138680d580200000062317139898887713a52713b284b0358010000007c713c4e4e4e4affffffff" +
        "4affffffff4b0074713d6289680558020000000100713e680786713f5271407471416258010000006e7142636e756d70792e" +
        "636f72652e6d756c746961727261790a7363616c61720a7143680d5802000000693871448988877145527146284b0368114e" +
        "4e4e4affffffff4affffffff4b00747147626805580800000007000000000000007148680786714952714a86714b52714c75" +
        "2e")
    for (name, bytes) in [("protocol 4", protocol4), ("protocol 2", protocol2)] {
        let path = writeTemp(bytes, "p.pkl")
        defer { unlink(path) }
        let a = try Pickle.arrays(file: path)
        let betas = try a["betas"]?.doubles()
        expect(a["betas"]?.shape == [1, 3] && betas == [0.5, -1.25, 2.0], "\(name): betas")
        expect((try a["trans"]?.doubles()) == [0, 0.25, 0.5, 0.75, 1, 1.25], "\(name): trans")
        expect((try a["gender"]?.strings()) == ["female"], "\(name): gender")
        expect((try a["name"]?.strings()) == ["clip"], "\(name): str")
        let fps = try a["mocap_frame_rate"]?.ints(), n = try a["n"]?.ints()
        expect(fps == [30] && n == [7], "\(name): ints")
        expect((try a["x"]?.doubles()) == [1.5], "\(name): float")
        expect((try a["flags"]?.ints()) == [1, 0], "\(name): bool array")
    }
    // {"a": object whose __reduce__ is (os.system, ("echo pwned",))}
    let evil = writeTemp(hexBytes("8004952c000000000000007d948c0161948c05706f736978948c0673797374656d9493948c0a6563686f2070776e65649485945294732e"), "evil.pkl")
    defer { unlink(evil) }
    do {
        _ = try Pickle.arrays(file: evil)
        expect(false, "a pickle calling os.system loads")
    } catch {
        expect("\(error)".contains("refusing to load"), "refused: \(error)")
    }
}

test("[smpl] SMPL-X forward kinematics and skinning match a numpy reference") {
    let file = repositoryRoot + "/assets/body_models/smplx/smplx_lbs_neutral.npz"
    guard access(file, F_OK) == 0 else {
        print("  skipped: no exported SMPL-X model (python tools/export_smplx_model.py)")
        return
    }
    let model = try SMPLModel.load(file: file)
    // numpy SMPL-X (original model, all skinning weights, no pose correctives) for these parameters.
    var pose = [Vec3](repeating: .zero, count: 55)
    pose[0] = Vec3(-1.4, 0.3, 0.2); pose[1] = Vec3(-0.6, 0.1, 0.2); pose[4] = Vec3(1.1, 0, 0); pose[7] = Vec3(0.2, 0.1, -0.1)
    pose[3] = Vec3(0.15, 0.05, 0); pose[16] = Vec3(0.1, 0.2, -0.9); pose[18] = Vec3(0, -1.2, 0); pose[20] = Vec3(0.3, 0.2, 0.1)
    pose[28] = Vec3(0, 0, 0.8)
    let clip = SMPLClip(poses: [pose], trans: [Vec3(0.3, -0.2, 0.95)], betas: [0.8, -0.5, 0.3, 0.1], gender: "neutral", fps: 30,
                        layout: "test")
    let joints = [0, 7, 10, 15, 20, 21, 28]
    let expectedJoints = [Vec3(0.303006, -0.571064, 0.962362), Vec3(0.710883, -0.753794, 1.625312), Vec3(0.825603, -0.738472, 1.715190),
                          Vec3(0.116861, -0.402885, 0.362937), Vec3(0.634159, -0.499507, 0.755298), Vec3(-0.491795, -0.472061, 0.803200),
                          Vec3(0.721813, -0.449057, 0.810443)]
    let vertices = [0, 3500, 8000]
    let expectedVertices = [Vec3(0.160549, -0.410259, 0.337908), Vec3(0.378826, -0.672227, 1.116729), Vec3(-0.618999, -0.531812, 0.841336)]
    let rest = model.vertices(betas: clip.betas).map(YUpToZUp.point)
    let bind = model.joints(betas: clip.betas).map { Transform(rotation: .identity, translation: YUpToZUp.point($0)) }
    let surface = BodySurface(jointNames: model.jointNames, restVertices: rest, bind: bind, weightJoints: model.weightJoints,
                              weightValues: model.weightValues, influences: model.influences, triangles: [])
    // A Z-up file's world is ours as is; a Y-up file's is turned into Z-up.
    for (up, convert) in [(SMPLOptions.Up.z, { (p: Vec3) in p }), (.y, YUpToZUp.point)] {
        let world = model.motion(clip, up: up).globalTransforms(frame: 0)
        for (j, e) in zip(joints, expectedJoints) {
            expect(length(world[j].translation - convert(e)) < 2e-5, "\(up) joint \(model.jointNames[j]): \(world[j].translation) vs \(convert(e))")
        }
        let skinned = surface.skin(world: world)
        // 4 of the up to 55 weights kept: within 2 mm.
        for (v, e) in zip(vertices, expectedVertices) {
            expect(length(skinned[v] - convert(e)) < 0.002, "\(up) vertex \(v): \(skinned[v]) vs \(convert(e))")
        }
    }
    expect(SMPLModel.upAxis(clip) == .z, "pelvis up direction (root turned about -X by ~1.4 rad) reads as Z-up")
}

finishTests()
