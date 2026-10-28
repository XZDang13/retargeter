#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

/// Locates the repository root: `$RETARGETER_ROOT`, or the first ancestor of the running executable
/// that contains `xmake.lua` and `assets/`.
public enum RepositoryLocation {
    public static func root() -> String? {
        if let env = getenv("RETARGETER_ROOT") { return String(cString: env) }
        var buffer = [CChar](repeating: 0, count: 4096)
        let n = readlink("/proc/self/exe", &buffer, buffer.count - 1)
        guard n > 0 else { return nil }
        var dir = String(decoding: buffer[0..<n].map { UInt8(bitPattern: $0) }, as: UTF8.self)
        while let slash = dir.lastIndex(of: "/"), slash != dir.startIndex {
            dir = String(dir[..<slash])
            if access(dir + "/xmake.lua", F_OK) == 0 && access(dir + "/assets", F_OK) == 0 { return dir }
        }
        return nil
    }
}

/// Reads a whole file; nil if it cannot be read.
public func readFile(_ path: String) -> [UInt8]? { readFileBytes(path) }

/// Directory entries (names only, excluding . and ..), sorted.
public func listDirectory(_ path: String) -> [(name: String, isDirectory: Bool)] {
    guard let dir = opendir(path) else { return [] }
    defer { closedir(dir) }
    var out: [(String, Bool)] = []
    while let entry = readdir(dir) {
        let name = withUnsafePointer(to: entry.pointee.d_name) {
            $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) }
        }
        if name == "." || name == ".." { continue }
        out.append((name, entry.pointee.d_type == UInt8(DT_DIR)))
    }
    return out.sorted { $0.0 < $1.0 }
}
