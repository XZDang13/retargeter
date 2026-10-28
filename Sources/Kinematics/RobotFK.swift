import Compute
import Core

/// Batched robot forward kinematics on a compute device (`Kernels/robot_fk.slang`).
public final class RobotFK {
    public let model: RobotModel
    let device: ComputeDevice
    let kernel: Kernel
    let parent, dof, originRot, originPos, childRot, childPos, axis: DeviceBuffer
    let q, rootRot, rootPos, linkRot, linkPos: ScratchBuffer

    public init(device: ComputeDevice, model: RobotModel) throws {
        self.model = model
        self.device = device
        kernel = try device.loadKernel(module: "robot_fk", entryPoint: "robotFK")
        let links = model.links
        parent = try device.makeBuffer(links.map { Int32($0.parent ?? -1) })
        dof = try device.makeBuffer(links.map { Int32($0.dofIndex ?? -1) })
        originRot = try device.makeBuffer(links.map { $0.jointOrigin.rotation.device })
        originPos = try device.makeBuffer(links.map { $0.jointOrigin.translation.device })
        childRot = try device.makeBuffer(links.map { $0.jointToChild.rotation.device })
        childPos = try device.makeBuffer(links.map { $0.jointToChild.translation.device })
        axis = try device.makeBuffer(links.map { link -> SIMD4<Float> in
            if case .revolute(let a) = link.joint { return a.device }
            return SIMD4(0, 0, 1, 0)
        })
        (q, rootRot, rootPos, linkRot, linkPos) = (ScratchBuffer(device: device), ScratchBuffer(device: device),
                                                   ScratchBuffer(device: device), ScratchBuffer(device: device),
                                                   ScratchBuffer(device: device))
    }

    /// World link poses, `[sample][link]`. `jointPositions` is `[sample][dof]`.
    public func compute(roots: [Transform], jointPositions: [[Double]]) throws -> [[Transform]] {
        precondition(roots.count == jointPositions.count)
        let batch = roots.count
        let n = model.links.count
        let qBuffer = try q.upload(jointPositions.flatMap { $0.map(Float.init) })
        let rr = try rootRot.upload(roots.map { $0.rotation.device })
        let rp = try rootPos.upload(roots.map { $0.translation.device })
        let lr = try linkRot.ensure(SIMD4<Float>.self, count: batch * n)
        let lp = try linkPos.ensure(SIMD4<Float>.self, count: batch * n)
        try device.dispatch(kernel, threads: batch, arguments: [
            .buffer("linkParent", parent), .buffer("linkDof", dof),
            .buffer("jointOriginRot", originRot), .buffer("jointOriginPos", originPos),
            .buffer("jointToChildRot", childRot), .buffer("jointToChildPos", childPos), .buffer("jointAxis", axis),
            .buffer("q", qBuffer), .buffer("rootRot", rr), .buffer("rootPos", rp),
            .buffer("linkRot", lr), .buffer("linkPos", lp),
            .value("batch", UInt32(batch)), .value("numLinks", UInt32(n)), .value("numDofs", UInt32(model.dofCount)),
        ])
        let all = transforms(rotations: try lr.download(as: SIMD4<Float>.self, count: batch * n),
                             positions: try lp.download(as: SIMD4<Float>.self, count: batch * n))
        return (0..<batch).map { Array(all[($0 * n)..<(($0 + 1) * n)]) }
    }
}
