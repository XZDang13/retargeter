// Milestone-0 parity test: the differentiable chain-FK spike kernel must match a double-precision
// Swift reference (positions, analytic Jacobian, gradient) on every backend.

import Compute
import Dispatch

#if canImport(Glibc)
import Glibc
#endif

// MARK: - Minimal test harness

var failures = 0

@MainActor func expect(_ condition: Bool, _ message: @autoclosure () -> String) {
    if !condition {
        failures += 1
        print("  FAIL: \(message())")
    }
}

func maxAbsError(_ a: [Float], _ b: [Double]) -> Double {
    zip(a, b).reduce(0) { max($0, abs(Double($1.0) - $1.1)) }
}

/// SplitMix64, so test inputs are identical on every run.
struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform(_ lo: Double, _ hi: Double) -> Double {
        lo + (hi - lo) * Double(next() >> 11) / Double(1 << 53)
    }
}

// MARK: - Double-precision reference

typealias Vec3 = (Double, Double, Double)
typealias Mat3 = [Double]  // row-major 3x3

func mul(_ m: Mat3, _ v: Vec3) -> Vec3 {
    (m[0] * v.0 + m[1] * v.1 + m[2] * v.2, m[3] * v.0 + m[4] * v.1 + m[5] * v.2, m[6] * v.0 + m[7] * v.1 + m[8] * v.2)
}

func mul(_ a: Mat3, _ b: Mat3) -> Mat3 {
    var r = Mat3(repeating: 0, count: 9)
    for i in 0..<3 { for j in 0..<3 { for k in 0..<3 { r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j] } } }
    return r
}

func axisAngle(_ a: Vec3, _ t: Double) -> Mat3 {
    let c = cos(t), s = sin(t), k = 1 - c
    return [c + a.0 * a.0 * k, a.0 * a.1 * k - a.2 * s, a.0 * a.2 * k + a.1 * s,
            a.1 * a.0 * k + a.2 * s, c + a.1 * a.1 * k, a.1 * a.2 * k - a.0 * s,
            a.2 * a.0 * k - a.1 * s, a.2 * a.1 * k + a.0 * s, c + a.2 * a.2 * k]
}

func add(_ a: Vec3, _ b: Vec3) -> Vec3 { (a.0 + b.0, a.1 + b.1, a.2 + b.2) }
func sub(_ a: Vec3, _ b: Vec3) -> Vec3 { (a.0 - b.0, a.1 - b.1, a.2 - b.2) }
func cross(_ a: Vec3, _ b: Vec3) -> Vec3 { (a.1 * b.2 - a.2 * b.1, a.2 * b.0 - a.0 * b.2, a.0 * b.1 - a.1 * b.0) }

struct Chain {
    var axes: [Vec3]     // n
    var offsets: [Vec3]  // n + 1
    var count: Int { axes.count }

    /// End-effector position and analytic Jacobian (column j = world axis_j × (p_ee − p_j)).
    func forward(_ q: ArraySlice<Double>) -> (position: Vec3, jacobian: [Double]) {
        var R: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        var p: Vec3 = (0, 0, 0)
        var jointPositions: [Vec3] = []
        var worldAxes: [Vec3] = []
        for (i, angle) in q.enumerated() {
            p = add(p, mul(R, offsets[i]))
            jointPositions.append(p)
            worldAxes.append(mul(R, axes[i]))
            R = mul(R, axisAngle(axes[i], angle))
        }
        let ee = add(p, mul(R, offsets[count]))
        var jacobian = [Double](repeating: 0, count: 3 * count)
        for j in 0..<count {
            let column = cross(worldAxes[j], sub(ee, jointPositions[j]))
            jacobian[0 * count + j] = column.0
            jacobian[1 * count + j] = column.1
            jacobian[2 * count + j] = column.2
        }
        return (ee, jacobian)
    }
}

// MARK: - Test

let repoRoot: String = {
    var parts = #filePath.split(separator: "/", omittingEmptySubsequences: false)
    parts.removeLast(3)  // Tests/ComputeTests/main.swift
    return parts.joined(separator: "/")
}()
let kernelDir = repoRoot + "/Kernels/spike"

@MainActor func runChainFK(backend: Backend, chain: Chain, q: [Double], goals: [Vec3], batch: Int) throws {
    let device = try ComputeDevice(backend: backend, searchPaths: [kernelDir], cacheDirectory: repoRoot + "/build/kernel-cache")
    print("[\(backend)] \(device.name)")
    let n = chain.count

    let compileStart = ContinuousClock.now
    let kernel = try device.loadKernel(module: "chain_fk", entryPoint: "chainFK")
    print("  kernel load: \(ContinuousClock.now - compileStart)")

    func float4(_ v: Vec3) -> SIMD4<Float> { SIMD4(Float(v.0), Float(v.1), Float(v.2), 0) }
    let axisBuffer = try device.makeBuffer(chain.axes.map(float4))
    let offsetBuffer = try device.makeBuffer(chain.offsets.map(float4))
    let qBuffer = try device.makeBuffer(q.map(Float.init))
    let targetBuffer = try device.makeBuffer(goals.map(float4))
    let posBuffer = try device.makeBuffer(of: Float.self, count: batch * 3)
    let jacBuffer = try device.makeBuffer(of: Float.self, count: batch * 3 * n)
    let gradBuffer = try device.makeBuffer(of: Float.self, count: batch * n)

    let arguments: [KernelArgument] = [
        .buffer("jointAxis", axisBuffer), .buffer("jointOffset", offsetBuffer), .buffer("q", qBuffer),
        .buffer("target", targetBuffer), .buffer("outPos", posBuffer), .buffer("outJac", jacBuffer),
        .buffer("outGrad", gradBuffer), .value("batch", UInt32(batch)), .value("numJoints", UInt32(n)),
    ]
    try device.dispatch(kernel, threads: batch, arguments: arguments)
    try device.wait()

    // Timing: steady-state dispatch + wait.
    let iterations = backend == .cuda ? 50 : 3
    let start = ContinuousClock.now
    for _ in 0..<iterations { try device.dispatch(kernel, threads: batch, arguments: arguments) }
    try device.wait()
    print("  dispatch (batch \(batch), FK + \(n) fwd_diff + 1 bwd_diff): \((ContinuousClock.now - start) / iterations)")

    let pos = try posBuffer.download(as: Float.self)
    let jac = try jacBuffer.download(as: Float.self)
    let grad = try gradBuffer.download(as: Float.self)

    var refPos: [Double] = [], refJac: [Double] = [], refGrad: [Double] = []
    for b in 0..<batch {
        let (p, J) = chain.forward(q[(b * n)..<((b + 1) * n)])
        refPos += [p.0, p.1, p.2]
        refJac += J
        let r = sub(p, goals[b])
        for j in 0..<n { refGrad.append(J[0 * n + j] * r.0 + J[1 * n + j] * r.1 + J[2 * n + j] * r.2) }
    }
    let ePos = maxAbsError(pos, refPos), eJac = maxAbsError(jac, refJac), eGrad = maxAbsError(grad, refGrad)
    print("  max |err|: position \(ePos), jacobian \(eJac), gradient \(eGrad)")
    let tolerance = 1e-4  // float32 vs double on a ~1 m chain
    expect(ePos < tolerance, "[\(backend)] position error \(ePos)")
    expect(eJac < tolerance, "[\(backend)] jacobian error \(eJac)")
    expect(eGrad < tolerance, "[\(backend)] gradient error \(eGrad)")
}

/// The reference Jacobian itself is checked against central differences.
@MainActor func checkReferenceJacobian(chain: Chain, q: [Double]) {
    let n = chain.count, h = 1e-6
    let (_, J) = chain.forward(q[0..<n])
    var worst = 0.0
    for j in 0..<n {
        var qp = Array(q[0..<n]), qm = qp
        qp[j] += h
        qm[j] -= h
        let d = sub(chain.forward(qp[...]).position, chain.forward(qm[...]).position)
        let column = (d.0 / (2 * h), d.1 / (2 * h), d.2 / (2 * h))
        worst = max(worst, abs(column.0 - J[j]), abs(column.1 - J[n + j]), abs(column.2 - J[2 * n + j]))
    }
    print("[reference] analytic vs finite-difference Jacobian: \(worst)")
    expect(worst < 1e-6, "reference Jacobian mismatch \(worst)")
}

var rng = SplitMix64(state: 42)
let n = 7
@MainActor func randomUnit() -> Vec3 {
    let v = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
    let norm = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).squareRoot()
    return (v.0 / norm, v.1 / norm, v.2 / norm)
}
let chain = Chain(axes: (0..<n).map { _ in randomUnit() },
                  offsets: (0...n).map { _ in (rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(0.05, 0.3)) })
let batch = 4096
let q = (0..<(batch * n)).map { _ in rng.uniform(-Double.pi, Double.pi) }
let goals = (0..<batch).map { _ in (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1.5)) }

checkReferenceJacobian(chain: chain, q: q)
for backend in [Backend.cuda, .cpu] {
    do {
        try runChainFK(backend: backend, chain: chain, q: q, goals: goals, batch: batch)
    } catch {
        failures += 1
        print("  FAIL: [\(backend)] \(error)")
    }
}

// A device created on one thread must work from another (the viewer serves requests on worker threads).
do {
    // No cache: the worker thread must really compile (Slang recursion on a small worker stack).
    let device = try ComputeDevice(backend: .cuda, searchPaths: [kernelDir, repoRoot + "/Kernels"], cacheDirectory: nil)
    nonisolated(unsafe) var loaded: Result<Kernel, Error>?
    DispatchQueue.global().sync {
        loaded = Result {
            _ = try device.loadKernel(module: "retarget_ik_group", entryPoint: "solveGroup")  // the heavy one
            return try device.loadKernel(module: "chain_fk", entryPoint: "chainFK")
        }
    }
    let kernel = try loaded!.get()
    nonisolated(unsafe) let shared = (device, kernel)
    nonisolated(unsafe) var result: Result<[Float], Error> = .success([])
    DispatchQueue.global().sync {
        result = Result {
            let (device, kernel) = shared
            let axis = try device.makeBuffer([SIMD4<Float>(0, 0, 1, 0)])
            let offset = try device.makeBuffer([SIMD4<Float>(0, 0, 0, 0), SIMD4<Float>(1, 0, 0, 0)])
            let q = try device.makeBuffer([Float.pi / 2])
            let target = try device.makeBuffer([SIMD4<Float>(0, 0, 0, 0)])
            let pos = try device.makeBuffer(of: Float.self, count: 3)
            let jac = try device.makeBuffer(of: Float.self, count: 3)
            let grad = try device.makeBuffer(of: Float.self, count: 1)
            try device.dispatch(kernel, threads: 1, arguments: [
                .buffer("jointAxis", axis), .buffer("jointOffset", offset), .buffer("q", q), .buffer("target", target),
                .buffer("outPos", pos), .buffer("outJac", jac), .buffer("outGrad", grad),
                .value("batch", UInt32(1)), .value("numJoints", UInt32(1)),
            ])
            return try pos.download(as: Float.self)
        }
    }
    let p = try result.get()
    print("[cuda] cross-thread compile + dispatch: end effector \(p)")
    expect(abs(p[0]) < 1e-5 && abs(p[1] - 1) < 1e-5, "cross-thread result \(p)")
} catch {
    failures += 1
    print("  FAIL: [cuda] cross-thread dispatch: \(error)")
}

if failures > 0 {
    print("\(failures) check(s) failed")
    exit(1)
}
print("all checks passed")
