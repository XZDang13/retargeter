#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif
import CRTCompute

/// Compute backend. v1 targets CUDA; the CPU backend is used for parity tests and as a fallback.
public enum Backend: Sendable, CustomStringConvertible {
    case cpu
    case cuda

    var raw: rtc_backend {
        switch self {
        case .cpu: RTC_BACKEND_CPU
        case .cuda: RTC_BACKEND_CUDA
        }
    }

    public var description: String {
        switch self {
        case .cpu: "cpu"
        case .cuda: "cuda"
        }
    }
}

public struct ComputeError: Error, CustomStringConvertible {
    public let message: String
    public init(message: String) { self.message = message }
    public var description: String { message }
}

@inline(__always)
func check(_ status: rtc_status, _ context: @autoclosure () -> String) throws {
    guard status == RTC_OK else {
        throw ComputeError(message: "\(context()): \(String(cString: rtc_last_error()))")
    }
}

/// A Slang compute device (one slang-rhi device plus its queue).
public final class ComputeDevice {
    let handle: rtc_device
    public let backend: Backend
    public let name: String

    /// - Parameters:
    ///   - searchPaths: directories searched for Slang modules.
    ///   - cacheDirectory: persistent compiled-kernel cache; `nil` disables it.
    public init(backend: Backend, searchPaths: [String] = [], cacheDirectory: String? = nil, verbose: Bool = false) throws {
        var handle: rtc_device?
        let paths = searchPaths.map { strdup($0) }
        defer { paths.forEach { free($0) } }
        let status = paths.map { UnsafePointer($0) }.withUnsafeBufferPointer { pathPointers in
            withOptionalCString(cacheDirectory) { cacheDir in
                var desc = rtc_device_desc(
                    backend: backend.raw,
                    search_paths: pathPointers.baseAddress,
                    search_path_count: UInt32(pathPointers.count),
                    cache_dir: cacheDir,
                    verbose: verbose ? 1 : 0)
                return rtc_device_create(&desc, &handle)
            }
        }
        try check(status, "creating \(backend) device")
        self.handle = handle!
        self.backend = backend
        self.name = String(cString: rtc_device_name(self.handle))
    }

    deinit {
        rtc_device_destroy(handle)
    }

    public func makeBuffer<T: BitwiseCopyable>(_ values: [T]) throws -> DeviceBuffer {
        try values.withUnsafeBytes { bytes in
            try DeviceBuffer(device: self, byteCount: bytes.count, elementStride: MemoryLayout<T>.stride,
                             initialData: bytes.baseAddress)
        }
    }

    public func makeBuffer<T: BitwiseCopyable>(of type: T.Type, count: Int) throws -> DeviceBuffer {
        try DeviceBuffer(device: self, byteCount: count * MemoryLayout<T>.stride, elementStride: MemoryLayout<T>.stride,
                         initialData: nil)
    }

    /// Loads entry point `entryPoint` of Slang module `module` from the search paths.
    public func loadKernel(module: String, entryPoint: String) throws -> Kernel {
        var kernel: rtc_kernel?
        try check(rtc_kernel_load(handle, module, entryPoint, &kernel), "loading \(module).\(entryPoint)")
        return Kernel(device: self, handle: kernel!)
    }

    /// Launches `kernel` over `threads` threads (x, y, z). Asynchronous.
    public func dispatch(_ kernel: Kernel, threads: (Int, Int, Int), arguments: [KernelArgument]) throws {
        try withArgumentArray(arguments) { args in
            try check(
                rtc_dispatch(handle, kernel.handle, args.baseAddress, UInt32(args.count),
                             UInt32(threads.0), UInt32(threads.1), UInt32(threads.2)),
                "dispatching \(kernel)")
        }
    }

    public func dispatch(_ kernel: Kernel, threads: Int, arguments: [KernelArgument]) throws {
        try dispatch(kernel, threads: (threads, 1, 1), arguments: arguments)
    }

    /// Blocks until all submitted work is complete.
    public func wait() throws {
        try check(rtc_device_wait(handle), "waiting for device")
    }
}

private func withOptionalCString<R>(_ string: String?, _ body: (UnsafePointer<CChar>?) -> R) -> R {
    guard let string else { return body(nil) }
    return string.withCString(body)
}

/// Builds the C argument array; names and value bytes stay alive for the duration of `body`.
private func withArgumentArray<R>(_ arguments: [KernelArgument], _ body: (UnsafeBufferPointer<rtc_arg>) throws -> R) rethrows -> R {
    let names = arguments.map { strdup($0.name)! }
    defer { names.forEach { free($0) } }
    var values: [UnsafeMutableRawPointer?] = []
    defer { values.forEach { free($0) } }
    var args: [rtc_arg] = []
    args.reserveCapacity(arguments.count)
    for (index, argument) in arguments.enumerated() {
        switch argument.storage {
        case .buffer(let buffer):
            args.append(rtc_arg(name: names[index], kind: RTC_ARG_BUFFER, buffer: buffer.handle, data: nil, size: 0))
        case .bytes(let bytes):
            let copy = malloc(max(bytes.count, 1))!
            bytes.withUnsafeBytes { copy.copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
            values.append(copy)
            args.append(rtc_arg(name: names[index], kind: RTC_ARG_DATA, buffer: nil, data: copy, size: UInt64(bytes.count)))
        }
    }
    return try withExtendedLifetime(arguments) {
        try args.withUnsafeBufferPointer(body)
    }
}
