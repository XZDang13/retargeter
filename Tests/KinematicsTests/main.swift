// Swift (double) vs Slang (float32) forward-kinematics parity on CUDA and CPU.

import Compute
import Core
import Kinematics
import MotionIO

func maxError(_ a: [[Transform]], _ b: [[Transform]]) -> (position: Double, rotation: Double) {
    var p = 0.0, r = 0.0
    for (x, y) in zip(a, b) {
        for (s, t) in zip(x, y) {
            p = max(p, length(s.translation - t.translation))
            r = max(r, Quat.angle(s.rotation, t.rotation))
        }
    }
    return (p, r)
}

let g1Dir = repositoryRoot + "/assets/robots/unitree_g1"
let robots = try ["g1_29_dof", "g1_23_dof"].map { try RobotUSDLoader.load(directory: g1Dir + "/" + $0) }
let motion = try BVH.load(file: "/home/troja/Downloads/soma_uniform/bvh/230322/sweep_floor_side_walk_R_004__A298.bvh")
let humanReference = (0..<motion.frameCount).map { motion.globalTransforms(frame: $0) }

// MARK: - Body surface (CPU reference)

let surfaceFile = repositoryRoot + "/assets/body_models/soma/surface_lo.npz"
let fixtureFile = repositoryRoot + "/build/fixtures/soma_lbs_reference.npz"
nonisolated(unsafe) var surface: BodySurface?

test("SOMA body surface vs SOMA-X reference") {
    let s = try BodySurface.load(file: surfaceFile)
    surface = s
    print("  \(s.vertexCount) vertices, \(s.triangles.count) triangles, \(s.influences) influences")
    expect(s.jointNames == motion.skeleton.jointNames, "surface joints follow the BVH joint order")
    let fixture = try NPZ.read(file: fixtureFile)
    let frames = try fixture["frames"]!.ints()
    let rot = try fixture["transform_rot"]!.doubles(), pos = try fixture["transform_pos"]!.doubles()
    let verts = try fixture["vertices"]!.doubles()
    let J = s.bind.count, V = s.vertexCount
    var exact = 0.0, pipeline = 0.0, feet = 0.0
    let footVertices = Set(s.restVertices.indices.filter { s.restVertices[$0].z < 0.12 })
    for (b, frame) in frames.enumerated() {
        let somaWorld = (0..<J).map { j -> Transform in
            let r = (b * J + j) * 4, p = (b * J + j) * 3
            return Transform(rotation: Quat(w: rot[r], x: rot[r + 1], y: rot[r + 2], z: rot[r + 3]),
                             translation: Vec3(pos[p], pos[p + 1], pos[p + 2]))
        }
        let reference = (0..<V).map { v in Vec3(verts[(b * V + v) * 3], verts[(b * V + v) * 3 + 1], verts[(b * V + v) * 3 + 2]) }
        let ours = s.skin(world: somaWorld)
        let fromBVH = s.skin(world: humanReference[frame])
        for v in 0..<V {
            exact = max(exact, length(ours[v] - reference[v]))
            let e = length(fromBVH[v] - reference[v])
            pipeline = max(pipeline, e)
            if footVertices.contains(v) { feet = max(feet, e) }
        }
    }
    print("  LBS with SOMA-X transforms: max error \(exact * 1000) mm")
    print("  LBS driven by our BVH FK:   max error \(pipeline * 100) cm overall, \(feet * 100) cm on the feet")
    expect(exact < 1e-4, "LBS reproduces SOMA-X")
    expect(feet < 0.01 && pipeline < 0.02, "BVH-driven surface within skeleton-fit tolerance")
    let z0 = s.skin(world: humanReference[0]).map(\.z).min()!
    print("  frame 0 lowest vertex z: \(z0) m")
}

for backend in [Backend.cuda, .cpu] {
    let device: ComputeDevice
    do { device = try ComputeDevice.makeDefault(backend: backend) } catch {
        test("[\(backend)] device") { throw error }
        continue
    }
    print("[\(backend)] \(device.name)")

    for robot in robots {
        test("[\(backend)] robot FK parity \(robot.name)") {
            var rng = SplitMix64(state: 7)
            let batch = backend == .cuda ? 4096 : 512
            var roots: [Transform] = [], qs: [[Double]] = []
            for _ in 0..<batch {
                let axis = normalize(Vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
                roots.append(Transform(rotation: Quat(axis: axis, angle: rng.uniform(-3, 3)),
                                       translation: Vec3(rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0, 1.5))))
                qs.append(robot.links.compactMap { link in link.limits.map { rng.uniform($0.lower, $0.upper) } })
            }
            let fk = try RobotFK(device: device, model: robot)
            let start = ContinuousClock.now
            let gpu = try fk.compute(roots: roots, jointPositions: qs)
            let elapsed = ContinuousClock.now - start
            let reference = zip(roots, qs).map { robot.forwardKinematics(root: $0.0, q: $0.1) }
            let (ep, er) = maxError(gpu, reference)
            print("  \(robot.name) batch \(batch): max |Δp| \(ep) m, max |Δθ| \(er) rad (incl. transfer: \(elapsed))")
            expect(ep < 1e-5 && er < 1e-5, "robot FK parity")
        }
    }

    test("[\(backend)] skeleton FK parity (\(motion.frameCount) frames)") {
        let fk = try SkeletonFK(device: device, skeleton: motion.skeleton)
        let start = ContinuousClock.now
        let gpu = try fk.compute(motion)
        let elapsed = ContinuousClock.now - start
        let (ep, er) = maxError(gpu, humanReference)
        print("  max |Δp| \(ep) m, max |Δθ| \(er) rad (incl. transfer: \(elapsed))")
        expect(ep < 1e-5 && er < 2e-5, "skeleton FK parity")
    }

    test("[\(backend)] surface skinning parity") {
        guard let surface else { throw NPYError(message: "surface not loaded") }
        let skinning = try SurfaceSkinning(device: device, surface: surface)
        let frames = backend == .cuda ? Array(0..<motion.frameCount) : [0, 100, 400]
        let start = ContinuousClock.now
        let posed = try skinning.skin(world: frames.map { humanReference[$0] })
        let elapsed = ContinuousClock.now - start
        var worst = 0.0
        for (b, frame) in frames.enumerated() where b % max(1, frames.count / 8) == 0 {
            let reference = surface.skin(world: humanReference[frame])
            for v in 0..<surface.vertexCount {
                let p = posed[b * surface.vertexCount + v]
                worst = max(worst, length(Vec3(Double(p.x), Double(p.y), Double(p.z)) - reference[v]))
            }
        }
        print("  \(frames.count) frames x \(surface.vertexCount) vertices in \(elapsed); max |Δ| \(worst) m")
        expect(worst < 1e-5, "GPU LBS parity")
    }
}

test("CPU batch skinning parity") {
    guard let surface else { throw NPYError(message: "surface not loaded") }
    let frames = Array(stride(from: 0, to: motion.frameCount, by: 7))
    let start = ContinuousClock.now
    let posed = surface.skin(frames: frames.map { humanReference[$0] })
    let elapsed = ContinuousClock.now - start
    var worst = 0.0
    for (b, frame) in frames.enumerated() where b % max(1, frames.count / 8) == 0 {
        let reference = surface.skin(world: humanReference[frame])
        for v in 0..<surface.vertexCount {
            let p = posed[b * surface.vertexCount + v]
            worst = max(worst, length(Vec3(Double(p.x), Double(p.y), Double(p.z)) - reference[v]))
        }
    }
    print("  \(frames.count) frames x \(surface.vertexCount) vertices in \(elapsed); max |Δ| \(worst) m")
    expect(worst < 1e-5, "CPU batch LBS parity")
}

finishTests()
