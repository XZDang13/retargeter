import Compute
import Core

/// Device layout: rotation float4(x, y, z, w), position float4(x, y, z, 0).
extension Quat {
    @inlinable public var device: SIMD4<Float> { SIMD4(Float(v.x), Float(v.y), Float(v.z), Float(w)) }
    @inlinable public init(device d: SIMD4<Float>) { self.init(w: Double(d.w), x: Double(d.x), y: Double(d.y), z: Double(d.z)) }
}

extension Vec3 {
    @inlinable public var device: SIMD4<Float> { SIMD4(Float(x), Float(y), Float(z), 0) }
    @inlinable public init(device d: SIMD4<Float>) { self.init(Double(d.x), Double(d.y), Double(d.z)) }
}

func transforms(rotations: [SIMD4<Float>], positions: [SIMD4<Float>]) -> [Transform] {
    zip(rotations, positions).map { Transform(rotation: Quat(device: $0.0), translation: Vec3(device: $0.1)) }
}

/// Grows-on-demand device buffer for per-call data.
public final class ScratchBuffer {
    public private(set) var buffer: DeviceBuffer?
    let device: ComputeDevice

    public init(device: ComputeDevice) { self.device = device }

    public func ensure<T: BitwiseCopyable>(_ type: T.Type, count: Int) throws -> DeviceBuffer {
        let bytes = max(count, 1) * MemoryLayout<T>.stride
        if let buffer, buffer.byteCount >= bytes { return buffer }
        let fresh = try device.makeBuffer(of: T.self, count: max(count, 1))
        buffer = fresh
        return fresh
    }

    public func upload<T: BitwiseCopyable>(_ values: [T]) throws -> DeviceBuffer {
        let buffer = try ensure(T.self, count: values.count)
        if !values.isEmpty { try buffer.upload(values) }
        return buffer
    }
}
