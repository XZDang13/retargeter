import Compute
import Core

/// Batched body-surface skinning on a compute device (`Kernels/lbs.slang`).
public final class SurfaceSkinning {
    public let surface: BodySurface
    let device: ComputeDevice
    let kernel: Kernel
    let rest, joints, weights: DeviceBuffer
    let skinRot, skinPos, output: ScratchBuffer
    let inverseBind: [Transform]

    public init(device: ComputeDevice, surface: BodySurface) throws {
        self.device = device
        self.surface = surface
        kernel = try device.loadKernel(module: "lbs", entryPoint: "skin")
        rest = try device.makeBuffer(surface.restVertices.map(\.device))
        joints = try device.makeBuffer(surface.weightJoints)
        weights = try device.makeBuffer(surface.weightValues)
        inverseBind = surface.bind.map(\.inverse)
        (skinRot, skinPos, output) = (ScratchBuffer(device: device), ScratchBuffer(device: device), ScratchBuffer(device: device))
    }

    /// Posed vertices per frame from world joint transforms `[frame][joint]`. Returns a flat
    /// `[frame * vertexCount]` array of float positions.
    public func skin(world: [[Transform]]) throws -> [SIMD3<Float>] {
        let frames = world.count
        let skinning = world.flatMap { zip($0, inverseBind).map { $0.0 * $0.1 } }
        let sr = try skinRot.upload(skinning.map { $0.rotation.device })
        let sp = try skinPos.upload(skinning.map { $0.translation.device })
        let n = surface.vertexCount
        let out = try output.ensure(SIMD4<Float>.self, count: frames * n)
        try device.dispatch(kernel, threads: frames * n, arguments: [
            .buffer("restVertex", rest), .buffer("weightJoint", joints), .buffer("weightValue", weights),
            .buffer("skinRot", sr), .buffer("skinPos", sp), .buffer("outVertex", out),
            .value("frames", UInt32(frames)), .value("numVertices", UInt32(n)),
            .value("numJoints", UInt32(surface.bind.count)), .value("influences", UInt32(surface.influences)),
        ])
        // SIMD3<Float> has the stride of the device's float4, so the vertices download as they are.
        return try out.download(as: SIMD3<Float>.self, count: frames * n)
    }
}
