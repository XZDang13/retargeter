// Ground estimation and contact detection on real SOMA clips (one per behaviour) plus synthetic checks.

import Compute
import Contact
import Core
import Kinematics
import MotionIO

let bvhRoot = "/home/troja/Downloads/soma_uniform/bvh/"
let surface = try BodySurface.load(file: repositoryRoot + "/assets/body_models/soma/surface_lo.npz")
let groups = try ContactGroups.soma(surface)
let device = try ComputeDevice.makeDefault(backend: .cuda)
let skinning = try SurfaceSkinning(device: device, surface: surface)

@MainActor func pose(_ file: String) throws -> PosedSurface {
    let motion = try BVH.load(file: bvhRoot + file)
    let fk = try SkeletonFK(device: device, skeleton: motion.skeleton)
    let world = try fk.compute(motion)
    return PosedSurface(fps: motion.fps, vertexCount: surface.vertexCount, positions: try skinning.skin(world: world),
                        excluded: surface.excluded)
}

func fraction(_ t: ContactTimeline, _ names: [String]) -> Double {
    let ids = names.compactMap { t.groupNames.firstIndex(of: $0) }
    let hits = (0..<t.frameCount).filter { f in ids.contains { t.contact(f, $0) } }.count
    return Double(hits) / Double(t.frameCount)
}

func summary(_ t: ContactTimeline) -> String {
    let G = t.groupNames.count
    let used = (0..<G).filter { g in (0..<t.frameCount).contains { t.contact($0, g) } }
        .map { g in "\(t.groupNames[g]) \(Int(100 * Double((0..<t.frameCount).filter { t.contact($0, g) }.count) / Double(t.frameCount)))%" }
    let env = (0..<G).filter { g in (0..<t.frameCount).contains { t.envContact[$0 * G + g] } }.map { t.groupNames[$0] }
    return "ground \(t.ground[0]) m, tilt \(t.groundTilt)°, contacts [\(used.joined(separator: ", "))], "
        + "flight \(t.flight.filter { $0 }.count) f, unsupported \(t.unsupported.filter { $0 }.count) f, env contact \(env), "
        + "env support \(t.envSupport.filter { $0 }.count) f, flags \(t.flags)"
}

test("contact groups cover the body") {
    print("  " + zip(groups.names, groups.vertices).map { "\($0.0)=\($0.1.count)" }.joined(separator: " "))
    expect(groups.vertices.allSatisfy { $0.count >= 20 }, "every group has vertices")
    let assigned = groups.vertices.reduce(0) { $0 + $1.count }
    expect(assigned == surface.excluded.filter { !$0 }.count, "every surface vertex assigned once (\(assigned))")
}

struct Case {
    var name: String, file: String
    var check: (ContactTimeline) -> Void
}

let cases: [Case] = [
    Case(name: "walk", file: "210531/walk_forward_amateur_001__A001.bvh") { t in
        expect(abs(t.ground[0]) < 0.03, "ground near 0")
        expect(fraction(t, ["left_heel", "left_toe", "right_heel", "right_toe"]) > 0.95, "a foot is down almost always")
        expect(fraction(t, ["left_heel", "left_toe"]) < 0.9 && fraction(t, ["right_heel", "right_toe"]) < 0.9, "feet alternate")
        expect(fraction(t, ["left_knee", "right_knee", "left_hand", "right_hand", "pelvis", "torso"]) == 0, "no other contacts")
        expect(t.flags.isEmpty, "clean clip: \(t.flags)")
    },
    Case(name: "kneel", file: "220713/kneeling_loop_001__A021.bvh") { t in
        expect(fraction(t, ["left_knee", "right_knee", "left_shin", "right_shin"]) > 0.5, "knees/shins down")
        expect(!t.flags.contains("env_contact"), "no environment contact")
        expect(t.flags.isEmpty, "clean clip: \(t.flags)")
    },
    Case(name: "crawl", file: "230119/crawl_ff_start_180_R_001__A125.bvh") { t in
        expect(fraction(t, ["left_hand", "right_hand"]) > 0.3, "hands down")
        expect(fraction(t, ["left_knee", "right_knee", "left_shin", "right_shin"]) > 0.3, "knees down")
    },
    Case(name: "push-up", file: "230509/push_up_loop_001__A359.bvh") { t in
        expect(fraction(t, ["left_hand", "right_hand"]) > 0.8, "hands down")
        expect(fraction(t, ["left_toe", "right_toe"]) > 0.8, "toes down")
        expect(t.flags.isEmpty, "finger penetration is not flagged: \(t.flags)")
    },
    Case(name: "lying", file: "231010/stand_up_lying_R_002__A472.bvh") { t in
        expect(fraction(t, ["torso", "pelvis"]) > 0.1, "torso/pelvis on the ground")
    },
    Case(name: "jump", file: "221010/jump_ff_180_R_001__A045.bvh") { t in
        expect(t.flight.contains(true), "flight phase detected")
        let airborne = (0..<t.frameCount).filter { f in !(0..<t.groupNames.count).contains { t.contact(f, $0) } }.count
        expect(airborne > 5, "frames without contact: \(airborne)")
    },
    Case(name: "sit on chair", file: "221010/sit_on_chair_loop_R_001__A045.bvh") { t in
        expect(t.flags.contains("env_contact"), "chair detected")
        let pelvis = t.groupNames.firstIndex(of: "pelvis")!
        expect((0..<t.frameCount).contains { t.envContact[$0 * t.groupNames.count + pelvis] }, "pelvis on the chair")
        // The thighs carry most of a seated body: labelled with the pelvis, not only the one part the support needs.
        let G = t.groupNames.count, thighs = ["left_thigh", "right_thigh"].map { t.groupNames.firstIndex(of: $0)! }
        let seated = (0..<t.frameCount).filter { t.envContact[$0 * G + pelvis] }
        let both = seated.filter { f in thighs.allSatisfy { t.envContact[f * G + $0] } }.count
        expect(Double(both) > 0.8 * Double(seated.count), "thighs on the seat with the pelvis: \(both) of \(seated.count)")
    },
    Case(name: "stairs", file: "230323/stairs_climbing_down_loop_R_001__A300.bvh") { t in
        expect(t.flags.contains("env_support"), "stairs detected")
    },
    // Swinging limbs and a twisting torso move the ZMP 20-40 cm from the CoM-only (cart-table) estimate here.
    Case(name: "turning jog", file: "240527/neutral_dancecard_go_turn_jog_001__A536.bvh") { t in
        expect(t.flags.isEmpty, "supported throughout: \(t.flags)")
    },
    // Stepping down from a box: only the feet on the box explain the support, not the static upper body above them.
    Case(name: "box step-down", file: "221010/come_down_50cm_box_R_001__A045.bvh") { t in
        expect(t.flags.contains("env_support"), "box detected")
        expect(!t.flags.contains("env_contact"), "no body part on the environment: \(t.flags)")
    },
]

/// Expected causal outcome per clip: which flags must (and must not) appear.
let causalFlags: [String: (must: [String], mustNot: [String])] = [
    "walk": ([], ["env_contact", "env_support", "unsupported"]),
    "kneel": ([], ["env_contact"]),
    "sit on chair": (["env_contact"], []),
    "stairs": (["env_support"], []),
]

for c in cases {
    let posed: PosedSurface
    do { posed = try pose(c.file) } catch { test("load \(c.name)") { throw error }; continue }
    var offline: ContactTimeline?
    test("offline: \(c.name)") {
        let start = ContinuousClock.now
        let t = ContactAnalysis.offline(posed, groups: groups)
        print("  \(c.file): \(t.frameCount) f, analysed in \(ContinuousClock.now - start)")
        print("  " + summary(t))
        c.check(t)
        offline = t
    }
    test("causal: \(c.name)") {
        guard let offline else { return }
        let start = ContinuousClock.now
        let t = ContactAnalysis.causal(posed, groups: groups)
        let perFrame = (ContinuousClock.now - start) / t.frameCount
        let G = t.groupNames.count
        let skip = Int(t.fps)  // calibration second
        var agree = 0, total = 0
        for f in skip..<t.frameCount { for g in 0..<G { total += 1; if t.contact(f, g) == offline.contact(f, g) { agree += 1 } } }
        let agreement = Double(agree) / Double(max(total, 1))
        let groundError = (skip..<t.frameCount).map { abs(t.ground[$0] - offline.ground[$0]) }.max() ?? 0
        print("  causal: \(perFrame) per frame, contact agreement \(agreement), max ground deviation \(groundError) m, flags \(t.flags)")
        // On terrain the causal ground legitimately differs (it cannot see the floor yet to come).
        if c.name != "stairs" && c.name != "box step-down" {
            expect(agreement > (c.name == "crawl" ? 0.9 : 0.95), "causal contacts agree with offline")
            expect(groundError < 0.02, "causal ground tracks offline ground")
        }
        expect(perFrame < .milliseconds(2), "per-frame update cost")
        if let rule = causalFlags[c.name] {
            for flag in rule.must { expect(t.flags.contains(flag), "causal flags contain \(flag)") }
            for flag in rule.mustNot { expect(!t.flags.contains(flag), "causal flags must not contain \(flag)") }
        }
    }
}

#if canImport(Glibc)
import Glibc
#endif
if let file = getenv("CONTACT_DIAG").map({ String(cString: $0) }) {
    let posed = try pose(file)
    let t = ContactAnalysis.offline(posed, groups: groups)
    let G = t.groupNames.count
    var minGroupHeight = Float.infinity
    for f in 0..<t.frameCount { for g in 0..<G { minGroupHeight = min(minGroupHeight, t.height[f * G + g]) } }
    var minVertex = Float.infinity
    for p in posed.positions { minVertex = min(minVertex, p.z) }
    print("diag: lowest group height \(minGroupHeight), lowest vertex \(minVertex - t.ground[0])")
    print("diag: " + summary(t))
    for g in 0..<G {
        let lows = (0..<t.frameCount).map { t.height[$0 * G + g] }.sorted()
        if lows[0] < -0.015 { print("diag:   \(t.groupNames[g]) min \(lows[0]) p5 \(lows[lows.count / 20])") }
    }
    if let list = getenv("CONTACT_DIAG_PARTS").map({ String(cString: $0) }) {
        let names = list.split(separator: ",").map(String.init)
        for f in stride(from: 0, to: t.frameCount, by: Int(t.fps / 2)) {
            let row = names.map { n -> String in
                let g = t.groupNames.firstIndex(of: n)!
                return "\(n) h \((t.height[f * G + g] * 100).rounded() / 100) v \((t.speed[f * G + g] * 100).rounded() / 100)\(t.envContact[f * G + g] ? " E" : "")"
            }
            print("diag: part f \(f) \(t.unsupported[f] ? "U" : " ") " + row.joined(separator: " | "))
        }
    }
    if getenv("CONTACT_DIAG_ENV") != nil {
        // Runs of the same set of environment-contact parts.
        var start = 0
        func parts(_ f: Int) -> [String] { (0..<G).filter { t.envContact[f * G + $0] }.map { t.groupNames[$0] } }
        for f in 1...t.frameCount where f == t.frameCount || parts(f) != parts(start) {
            if t.unsupported[start] { print("diag: env \(start)..<\(f) \(parts(start))") }
            start = f
        }
    }
    var f = 0
    while f < t.frameCount {
        if t.unsupported[f] {
            var e = f
            while e < t.frameCount && t.unsupported[e] { e += 1 }
            let mid = (f + e) / 2
            let on = (0..<G).filter { t.contact(mid, $0) }.map { t.groupNames[$0] }
            print("diag: unsupported \(f)..<\(e) (\(Double(e - f) / t.fps) s) mid contacts \(on) com \(t.com[mid]) zmp \(t.zmp[mid])")
            if getenv("CONTACT_DIAG_TRACE") != nil {
                for k in stride(from: max(0, f - 24), to: min(t.frameCount, e + 24), by: 4) {
                    let lf = groups.names.firstIndex(of: "left_toe")!, rf = groups.names.firstIndex(of: "right_toe")!
                    var foot = SIMD2<Float>.zero
                    for v in groups.vertices[lf] { let p = posed.p(k, v); foot += SIMD2(p.x, p.y) }
                    foot /= Float(groups.vertices[lf].count)
                    var rfoot = SIMD2<Float>.zero
                    for v in groups.vertices[rf] { let p = posed.p(k, v); rfoot += SIMD2(p.x, p.y) }
                    rfoot /= Float(groups.vertices[rf].count)
                    let on = (0..<G).filter { t.contact(k, $0) }.map { t.groupNames[$0] }
                    print("diag:   f \(k) \(t.unsupported[k] ? "U" : " ")\(t.flight[k] ? "F" : " ") com (\(t.com[k].x), \(t.com[k].y), \(t.com[k].z)) zmp \(t.zmp[k]) ltoe \(foot) rtoe \(rfoot) \(on)")
                }
            }
            for g in 0..<G where t.groupNames[g].contains("heel") || t.groupNames[g].contains("toe") {
                print("diag:    \(t.groupNames[g]) h \(t.height[mid * G + g]) v \(t.speed[mid * G + g])")
            }
            f = e
        } else { f += 1 }
    }
}

finishTests()
