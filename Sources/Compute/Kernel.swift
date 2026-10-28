import CRTCompute

/// A compiled compute entry point.
public final class Kernel: CustomStringConvertible {
    let handle: rtc_kernel
    public let device: ComputeDevice

    init(device: ComputeDevice, handle: rtc_kernel) {
        self.device = device
        self.handle = handle
    }

    deinit {
        rtc_kernel_destroy(handle)
    }

    public var threadGroupSize: (Int, Int, Int) {
        var size: (UInt32, UInt32, UInt32) = (1, 1, 1)
        withUnsafeMutableBytes(of: &size) { rtc_kernel_thread_group_size(handle, $0.baseAddress!.assumingMemoryBound(to: UInt32.self)) }
        return (Int(size.0), Int(size.1), Int(size.2))
    }

    public var description: String { "Kernel(\(threadGroupSize))" }
}

/// A shader parameter bound by name (global `uniform` or entry-point parameter).
public struct KernelArgument {
    enum Storage {
        case buffer(DeviceBuffer)
        case bytes([UInt8])
    }

    let name: String
    let storage: Storage

    public static func buffer(_ name: String, _ buffer: DeviceBuffer) -> KernelArgument {
        KernelArgument(name: name, storage: .buffer(buffer))
    }

    /// A plain value (scalar, SIMD vector, or POD struct matching the Slang layout).
    public static func value<T: BitwiseCopyable>(_ name: String, _ value: T) -> KernelArgument {
        let bytes = withUnsafeBytes(of: value) { [UInt8]($0) }
        return KernelArgument(name: name, storage: .bytes(bytes))
    }
}
