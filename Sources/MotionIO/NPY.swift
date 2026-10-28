// NumPy .npy / .npz reading (uncompressed archives, as written by np.savez).

import Core

public struct NPYError: Error, CustomStringConvertible {
    public let message: String
    public init(message: String) { self.message = message }
    public var description: String { message }
}

public struct NPYArray: Sendable {
    public enum DType: Sendable, Equatable {
        case float32, float64, int32, int64, uint8, bool
        case unicode(Int)  // <U<n>: n UTF-32 code units per element
    }

    public var dtype: DType
    public var shape: [Int]
    public var bytes: [UInt8]

    public var count: Int { shape.reduce(1, *) }

    private func load<T: BitwiseCopyable>(_ type: T.Type, count: Int? = nil) -> [T] {
        let n = count ?? self.count
        return bytes.withUnsafeBytes { raw in (0..<n).map { raw.loadUnaligned(fromByteOffset: $0 * MemoryLayout<T>.size, as: T.self) } }
    }

    public func doubles() throws -> [Double] {
        switch dtype {
        case .float32: load(Float.self).map(Double.init)
        case .float64: load(Double.self)
        case .int32: load(Int32.self).map(Double.init)
        case .int64: load(Int64.self).map(Double.init)
        case .uint8, .bool: bytes.map(Double.init)
        case .unicode: throw NPYError(message: "string array is not numeric")
        }
    }

    public func ints() throws -> [Int] {
        switch dtype {
        case .int32: load(Int32.self).map(Int.init)
        case .int64: load(Int64.self).map(Int.init)
        case .uint8, .bool: bytes.map(Int.init)
        default: throw NPYError(message: "array is not integer (\(dtype))")
        }
    }

    public func strings() throws -> [String] {
        guard case .unicode(let n) = dtype else { throw NPYError(message: "array is not a string array") }
        let scalars = load(UInt32.self, count: count * n)
        return (0..<count).map { i in
            var s = String.UnicodeScalarView()
            for c in scalars[(i * n)..<((i + 1) * n)] where c != 0 { if let u = Unicode.Scalar(c) { s.append(u) } }
            return String(s)
        }
    }

    public static func parse(_ data: ArraySlice<UInt8>) throws -> NPYArray {
        let b = Array(data)
        guard b.count >= 10, b[0] == 0x93, Array(b[1...5]) == Array("NUMPY".utf8) else {
            throw NPYError(message: "not an .npy array")
        }
        let major = b[6]
        let headerLength: Int
        let headerStart: Int
        if major == 1 {
            headerLength = Int(b[8]) | Int(b[9]) << 8
            headerStart = 10
        } else {
            headerLength = Int(b[8]) | Int(b[9]) << 8 | Int(b[10]) << 16 | Int(b[11]) << 24
            headerStart = 12
        }
        let header = String(decoding: b[headerStart..<(headerStart + headerLength)], as: UTF8.self)
        func field(_ key: String) -> Substring? {
            guard let r = header.firstRange(of: "'\(key)':") else { return nil }
            return header[r.upperBound...].drop { $0 == " " }
        }
        guard let descrField = field("descr"), let shapeField = field("shape") else {
            throw NPYError(message: "bad .npy header: \(header)")
        }
        let descr = descrField.dropFirst().prefix { $0 != "'" }
        if field("fortran_order")?.hasPrefix("True") == true { throw NPYError(message: "fortran order unsupported") }
        let shapeText = shapeField.dropFirst().prefix { $0 != ")" }
        let shape = shapeText.split(separator: ",").compactMap { Int($0.filter { $0 != " " }) }

        let dtype: DType
        switch descr {
        case "<f4": dtype = .float32
        case "<f8": dtype = .float64
        case "<i4": dtype = .int32
        case "<i8": dtype = .int64
        case "|u1": dtype = .uint8
        case "|b1": dtype = .bool
        default:
            if descr.hasPrefix("<U"), let n = Int(descr.dropFirst(2)) { dtype = .unicode(n) }
            else { throw NPYError(message: "unsupported dtype \(descr)") }
        }
        let start = headerStart + headerLength
        return NPYArray(dtype: dtype, shape: shape, bytes: Array(b[start...]))
    }
}

public enum NPZ {
    /// Reads every array of an uncompressed .npz archive. Keys drop the ".npy" suffix.
    public static func read(file: String) throws -> [String: NPYArray] {
        guard let bytes = readBytes(file) else { throw NPYError(message: "cannot read \(file)") }
        func u16(_ o: Int) -> Int { Int(bytes[o]) | Int(bytes[o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        func u64(_ o: Int) -> Int { u32(o) | u32(o + 4) << 32 }

        // End of central directory (no archive comment expected from numpy, but scan back anyway).
        var eocd = bytes.count - 22
        while eocd >= 0 && u32(eocd) != 0x0605_4b50 { eocd -= 1 }
        guard eocd >= 0 else { throw NPYError(message: "\(file): not a zip archive") }
        var entries = u16(eocd + 10)
        var directory = u32(eocd + 16)
        if entries == 0xFFFF || directory == 0xFFFF_FFFF {
            // Zip64 end of central directory locator precedes the EOCD.
            let locator = eocd - 20
            guard locator >= 0, u32(locator) == 0x0706_4b50 else { throw NPYError(message: "\(file): bad zip64 locator") }
            let record = u64(locator + 8)
            entries = u64(record + 32)
            directory = u64(record + 48)
        }

        var arrays: [String: NPYArray] = [:]
        var p = directory
        for _ in 0..<entries {
            guard u32(p) == 0x0201_4b50 else { throw NPYError(message: "\(file): bad central directory") }
            let method = u16(p + 10)
            var compressed = u32(p + 20)
            let nameLength = u16(p + 28), extraLength = u16(p + 30), commentLength = u16(p + 32)
            var localOffset = u32(p + 42)
            let name = String(decoding: bytes[(p + 46)..<(p + 46 + nameLength)], as: UTF8.self)
            // Zip64 extended information in the central directory extra field.
            var e = p + 46 + nameLength
            let extraEnd = e + extraLength
            while e + 4 <= extraEnd {
                let id = u16(e), size = u16(e + 2)
                if id == 0x0001 {
                    var f = e + 4
                    if u32(p + 24) == 0xFFFF_FFFF { f += 8 }  // uncompressed size
                    if compressed == 0xFFFF_FFFF { compressed = u64(f); f += 8 }
                    if localOffset == 0xFFFF_FFFF { localOffset = u64(f) }
                }
                e += 4 + size
            }
            guard method == 0 else { throw NPYError(message: "\(file): \(name) is compressed (use np.savez)") }
            let dataStart = localOffset + 30 + u16(localOffset + 26) + u16(localOffset + 28)
            let key = name.hasSuffix(".npy") ? String(name.dropLast(4)) : name
            arrays[key] = try NPYArray.parse(bytes[dataStart..<(dataStart + compressed)])
            p += 46 + nameLength + extraLength + commentLength
        }
        return arrays
    }

    private static func readBytes(_ path: String) -> [UInt8]? {
        guard let file = fopen(path, "rb") else { return nil }
        defer { fclose(file) }
        fseek(file, 0, SEEK_END)
        let size = ftell(file)
        fseek(file, 0, SEEK_SET)
        var bytes = [UInt8](repeating: 0, count: max(size, 0))
        let n = bytes.withUnsafeMutableBytes { fread($0.baseAddress, 1, size, file) }
        return n == size ? bytes : nil
    }
}

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif
