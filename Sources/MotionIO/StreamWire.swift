// Little-endian reader / writer behind the generated teleop messages (StreamMessages.swift): fixed-size
// scalars, u32-count-prefixed arrays and UTF-8 strings, after a 12-byte header (magic, version, type, length).

import Core

public struct WireError: Error, CustomStringConvertible {
    public let message: String
    public init(message: String) { self.message = message }
    public var description: String { message }
}

/// A value with a wire encoding.
public protocol WireValue {
    static func read(_ r: inout WireReader) throws -> Self
    func write(_ w: inout WireWriter)
}

public struct WireWriter {
    var bytes: [UInt8]

    public init(type: UInt16) {
        bytes = []
        bytes.reserveCapacity(256)
        appendRaw(StreamWire.magic)
        appendRaw(StreamWire.version)
        appendRaw(type)
        appendRaw(UInt32(0))   // payload length, filled in by finish()
    }

    mutating func appendRaw<T: FixedWidthInteger>(_ value: T) {
        withUnsafeBytes(of: value.littleEndian) { bytes.append(contentsOf: $0) }
    }

    public mutating func put<T: WireValue>(_ value: T) { value.write(&self) }

    public mutating func finish() -> [UInt8] {
        let length = UInt32(bytes.count - StreamWire.headerSize).littleEndian
        withUnsafeBytes(of: length) { raw in for i in 0..<4 { bytes[8 + i] = raw[i] } }
        return bytes
    }
}

public struct WireReader {
    let bytes: [UInt8]
    var at: Int
    public private(set) var type: UInt16

    /// Checks the header of one whole message and positions the reader at its payload.
    public init(message bytes: [UInt8]) throws {
        self.bytes = bytes
        at = 0
        type = 0
        guard bytes.count >= StreamWire.headerSize else { throw WireError(message: "message shorter than its header") }
        let magic: UInt32 = try raw(), version: UInt16 = try raw(), type: UInt16 = try raw(), length: UInt32 = try raw()
        guard magic == StreamWire.magic, version == StreamWire.version else {
            throw WireError(message: "bad magic / version \(magic) / \(version)")
        }
        guard Int(length) + StreamWire.headerSize == bytes.count else { throw WireError(message: "length mismatch") }
        self.type = type
    }

    mutating func raw<T: FixedWidthInteger>() throws -> T {
        let size = MemoryLayout<T>.size
        guard at + size <= bytes.count else { throw WireError(message: "truncated message") }
        var value: T = 0
        withUnsafeMutableBytes(of: &value) { dst in
            bytes.withUnsafeBytes { src in dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[at..<(at + size)])) }
        }
        at += size
        return T(littleEndian: value)
    }

    mutating func count(elementSize: Int) throws -> Int {
        let n = Int(try raw() as UInt32)
        guard n * elementSize <= bytes.count - at else { throw WireError(message: "truncated message") }
        return n
    }

    public mutating func get<T: WireValue>() throws -> T { try T.read(&self) }

    /// Checks that the whole payload was read.
    public func finish() throws {
        guard at == bytes.count else { throw WireError(message: "trailing bytes after the message") }
    }
}

extension UInt8: WireValue {
    public static func read(_ r: inout WireReader) throws -> UInt8 { try r.raw() }
    public func write(_ w: inout WireWriter) { w.appendRaw(self) }
}
extension UInt16: WireValue {
    public static func read(_ r: inout WireReader) throws -> UInt16 { try r.raw() }
    public func write(_ w: inout WireWriter) { w.appendRaw(self) }
}
extension UInt32: WireValue {
    public static func read(_ r: inout WireReader) throws -> UInt32 { try r.raw() }
    public func write(_ w: inout WireWriter) { w.appendRaw(self) }
}
extension UInt64: WireValue {
    public static func read(_ r: inout WireReader) throws -> UInt64 { try r.raw() }
    public func write(_ w: inout WireWriter) { w.appendRaw(self) }
}
extension Int32: WireValue {
    public static func read(_ r: inout WireReader) throws -> Int32 { try r.raw() }
    public func write(_ w: inout WireWriter) { w.appendRaw(self) }
}
extension Float: WireValue {
    public static func read(_ r: inout WireReader) throws -> Float { Float(bitPattern: try r.raw()) }
    public func write(_ w: inout WireWriter) { w.appendRaw(bitPattern) }
}
extension Double: WireValue {
    public static func read(_ r: inout WireReader) throws -> Double { Double(bitPattern: try r.raw()) }
    public func write(_ w: inout WireWriter) { w.appendRaw(bitPattern) }
}
extension String: WireValue {
    public static func read(_ r: inout WireReader) throws -> String {
        let n = try r.count(elementSize: 1)
        defer { r.at += n }
        return String(decoding: r.bytes[r.at..<(r.at + n)], as: UTF8.self)
    }
    public func write(_ w: inout WireWriter) {
        let utf8 = Array(self.utf8)
        w.appendRaw(UInt32(utf8.count))
        w.bytes += utf8
    }
}
extension Array: WireValue where Element: WireValue {
    public static func read(_ r: inout WireReader) throws -> [Element] {
        // Strings are at least their 4-byte length; fixed-size elements their own size.
        let n = try r.count(elementSize: Element.self == String.self ? 4 : MemoryLayout<Element>.size)
        var out: [Element] = []
        out.reserveCapacity(n)
        for _ in 0..<n { out.append(try Element.read(&r)) }
        return out
    }
    public func write(_ w: inout WireWriter) {
        w.appendRaw(UInt32(count))
        for e in self { e.write(&w) }
    }
}

extension HumanFrame {
    /// Local joint transforms; nil if the arrays do not match the skeleton.
    public func localTransforms(skeleton: HumanSkeleton) -> [Transform]? {
        guard jointQuat.count == 4 * skeleton.jointCount, jointPos.count == 3 * skeleton.jointCount else { return nil }
        return (0..<skeleton.jointCount).map { j in
            Transform(rotation: Quat(w: Double(jointQuat[4 * j]), x: Double(jointQuat[4 * j + 1]),
                                     y: Double(jointQuat[4 * j + 2]), z: Double(jointQuat[4 * j + 3])).normalized,
                      translation: Vec3(Double(jointPos[3 * j]), Double(jointPos[3 * j + 1]), Double(jointPos[3 * j + 2])))
        }
    }
}

extension StreamStart {
    /// The skeleton this stream declares; nil if its arrays disagree in length.
    public var humanSkeleton: HumanSkeleton? {
        let J = jointNames.count
        guard parents.count == J, restOffsets.count == 3 * J else { return nil }
        return HumanSkeleton(name: skeleton, jointNames: jointNames, parents: parents.map { $0 < 0 ? nil : Int($0) },
                             restOffsets: (0..<J).map { Vec3(Double(restOffsets[3 * $0]), Double(restOffsets[3 * $0 + 1]),
                                                             Double(restOffsets[3 * $0 + 2])) })
    }
}
