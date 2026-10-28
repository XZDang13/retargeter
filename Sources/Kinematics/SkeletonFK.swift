import Compute
import Core

/// Batched human skeleton forward kinematics on a compute device (`Kernels/skeleton_fk.slang`).
public final class SkeletonFK {
    let device: ComputeDevice
    let kernel: Kernel
    let parents: DeviceBuffer
    public let skeleton: HumanSkeleton
    let localRot, localPos, worldRot, worldPos: ScratchBuffer

    public init(device: ComputeDevice, skeleton: HumanSkeleton) throws {
        self.device = device
        self.skeleton = skeleton
        kernel = try device.loadKernel(module: "skeleton_fk", entryPoint: "skeletonFK")
        parents = try device.makeBuffer(skeleton.parents.map { Int32($0 ?? -1) })
        (localRot, localPos, worldRot, worldPos) = (ScratchBuffer(device: device), ScratchBuffer(device: device),
                                                    ScratchBuffer(device: device), ScratchBuffer(device: device))
    }

    /// World joint transforms for every frame, `[frame][joint]`.
    public func compute(_ motion: HumanMotion) throws -> [[Transform]] {
        let frames = motion.frameCount
        let n = skeleton.jointCount
        let flat = motion.local.flatMap { $0 }
        let lr = try localRot.upload(flat.map { $0.rotation.device })
        let lp = try localPos.upload(flat.map { $0.translation.device })
        let wr = try worldRot.ensure(SIMD4<Float>.self, count: frames * n)
        let wp = try worldPos.ensure(SIMD4<Float>.self, count: frames * n)
        try device.dispatch(kernel, threads: frames, arguments: [
            .buffer("jointParent", parents), .buffer("localRot", lr), .buffer("localPos", lp),
            .buffer("worldRot", wr), .buffer("worldPos", wp),
            .value("frames", UInt32(frames)), .value("numJoints", UInt32(n)),
        ])
        let all = transforms(rotations: try wr.download(as: SIMD4<Float>.self, count: frames * n),
                             positions: try wp.download(as: SIMD4<Float>.self, count: frames * n))
        return (0..<frames).map { Array(all[($0 * n)..<(($0 + 1) * n)]) }
    }
}
