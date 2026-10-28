#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

/// Finds the repository's `Kernels/` directory: `$RETARGETER_KERNELS` if set, otherwise the first
/// ancestor of the running executable that contains `Kernels/quat.slang`.
public enum KernelLocation {
    public static func directory() -> String? {
        if let env = getenv("RETARGETER_KERNELS") { return String(cString: env) }
        var buffer = [CChar](repeating: 0, count: 4096)
        let n = readlink("/proc/self/exe", &buffer, buffer.count - 1)
        guard n > 0 else { return nil }
        var dir = String(decoding: buffer[0..<n].map { UInt8(bitPattern: $0) }, as: UTF8.self)
        while let slash = dir.lastIndex(of: "/"), slash != dir.startIndex {
            dir = String(dir[..<slash])
            if access(dir + "/Kernels/quat.slang", F_OK) == 0 { return dir + "/Kernels" }
        }
        return nil
    }

    /// Persistent kernel cache next to the build products (`<repo>/build/kernel-cache`).
    public static func cacheDirectory() -> String? {
        directory().map { String($0.dropLast("/Kernels".count)) + "/build/kernel-cache" }
    }
}

extension ComputeDevice {
    /// A device with the repository kernel directory on its search path and the shared kernel cache.
    public static func makeDefault(backend: Backend, verbose: Bool = false) throws -> ComputeDevice {
        guard let kernels = KernelLocation.directory() else {
            throw ComputeError(message: "Kernels/ not found; set RETARGETER_KERNELS")
        }
        return try ComputeDevice(backend: backend, searchPaths: [kernels], cacheDirectory: KernelLocation.cacheDirectory(),
                                 verbose: verbose)
    }
}
