// NumPy .npz writing: an uncompressed (stored) zip of .npy members, readable with np.load.

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public struct NPZWriter {
    private var members: [(name: String, data: [UInt8])] = []

    public init() {}

    public mutating func add(_ name: String, float32 values: [Float], shape: [Int]) {
        precondition(shape.reduce(1, *) == values.count, "\(name): shape \(shape) does not match \(values.count) values")
        members.append((name, NPZWriter.npy(descr: "<f4", shape: shape, payload: values.withUnsafeBytes { Array($0) })))
    }

    public mutating func add(_ name: String, float64 values: [Double], shape: [Int]) {
        precondition(shape.reduce(1, *) == values.count)
        members.append((name, NPZWriter.npy(descr: "<f8", shape: shape, payload: values.withUnsafeBytes { Array($0) })))
    }

    public mutating func add(_ name: String, int32 values: [Int32], shape: [Int]) {
        precondition(shape.reduce(1, *) == values.count)
        members.append((name, NPZWriter.npy(descr: "<i4", shape: shape, payload: values.withUnsafeBytes { Array($0) })))
    }

    public mutating func add(_ name: String, uint8 values: [UInt8], shape: [Int]) {
        precondition(shape.reduce(1, *) == values.count)
        members.append((name, NPZWriter.npy(descr: "|u1", shape: shape, payload: values)))
    }

    /// Scalar float32 (shape ()).
    public mutating func add(_ name: String, scalar value: Float) {
        members.append((name, NPZWriter.npy(descr: "<f4", shape: [], payload: withUnsafeBytes(of: value) { Array($0) })))
    }

    /// Unicode string array (`<U<n>`); shape [] for a single string.
    public mutating func add(_ name: String, strings: [String], shape: [Int]? = nil) {
        let width = max(1, strings.map { $0.unicodeScalars.count }.max() ?? 1)
        var payload: [UInt8] = []
        for s in strings {
            var scalars = s.unicodeScalars.map(\.value)
            scalars += Array(repeating: 0, count: width - scalars.count)
            payload += scalars.withUnsafeBytes { Array($0) }
        }
        members.append((name, NPZWriter.npy(descr: "<U\(width)", shape: shape ?? [strings.count], payload: payload)))
    }

    public mutating func add(_ name: String, string: String) { add(name, strings: [string], shape: []) }

    static func npy(descr: String, shape: [Int], payload: [UInt8]) -> [UInt8] {
        let shapeText = shape.isEmpty ? "()" : shape.count == 1 ? "(\(shape[0]),)" : "(" + shape.map(String.init).joined(separator: ", ") + ")"
        var header = "{'descr': '\(descr)', 'fortran_order': False, 'shape': \(shapeText), }"
        // Pad so that magic (6) + version (2) + length (2) + header is a multiple of 64, ending in '\n'.
        let unpadded = 10 + header.utf8.count + 1
        header += String(repeating: " ", count: (64 - unpadded % 64) % 64) + "\n"
        let length = header.utf8.count
        return [0x93] + Array("NUMPY".utf8) + [1, 0, UInt8(length & 0xFF), UInt8(length >> 8)] + Array(header.utf8) + payload
    }

    /// Writes the archive atomically (temporary file + rename).
    public func write(file: String) throws {
        var out: [UInt8] = []
        var central: [UInt8] = []
        func le16(_ v: Int) -> [UInt8] { [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF)] }
        func le32(_ v: Int) -> [UInt8] { le16(v & 0xFFFF) + le16((v >> 16) & 0xFFFF) }
        for (name, data) in members {
            let file = Array((name + ".npy").utf8)
            let crc = Int(CRC32.checksum(data))
            precondition(data.count < 0xFFFF_FFFF && out.count < 0xFFFF_FFFF, "npz member too large (zip64 unsupported)")
            let offset = out.count
            // local file header: version 20, no flags, stored, time/date 0
            out += le32(0x0403_4b50) + le16(20) + le16(0) + le16(0) + le16(0) + le16(0x21)
            out += le32(crc) + le32(data.count) + le32(data.count) + le16(file.count) + le16(0) + file + data
            central += le32(0x0201_4b50) + le16(20) + le16(20) + le16(0) + le16(0) + le16(0) + le16(0x21)
            central += le32(crc) + le32(data.count) + le32(data.count) + le16(file.count) + le16(0) + le16(0)
            central += le16(0) + le16(0) + le32(0) + le32(offset) + file
        }
        let directoryOffset = out.count
        out += central
        out += le32(0x0605_4b50) + le16(0) + le16(0) + le16(members.count) + le16(members.count)
        out += le32(central.count) + le32(directoryOffset) + le16(0)

        let temporary = file + ".tmp"
        guard let handle = fopen(temporary, "wb") else { throw NPYError(message: "cannot write \(temporary)") }
        let written = out.withUnsafeBytes { fwrite($0.baseAddress, 1, $0.count, handle) }
        fclose(handle)
        guard written == out.count, rename(temporary, file) == 0 else { throw NPYError(message: "failed writing \(file)") }
    }
}

enum CRC32 {
    static let table: [UInt32] = (0..<256).map { i in
        var c = UInt32(i)
        for _ in 0..<8 { c = c & 1 != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    static func checksum(_ bytes: [UInt8]) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for b in bytes { c = table[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }
}
