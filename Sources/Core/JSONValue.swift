// Small JSON reader for configuration files (the project avoids Foundation).

public indirect enum JSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public subscript(key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }

    public var double: Double? { if case .number(let d) = self { d } else { nil } }
    public var string: String? { if case .string(let s) = self { s } else { nil } }
    public var bool: Bool? { if case .bool(let b) = self { b } else { nil } }
    public var array: [JSONValue]? { if case .array(let a) = self { a } else { nil } }
    public var object: [String: JSONValue]? { if case .object(let o) = self { o } else { nil } }

    public static func parse(_ text: String) throws -> JSONValue {
        var parser = JSONParser(bytes: Array(text.utf8))
        let value = try parser.value()
        parser.skipSpace()
        guard parser.pos == parser.bytes.count else { throw parser.error("trailing characters") }
        return value
    }

    public static func load(file: String) throws -> JSONValue {
        guard let bytes = readFileBytes(file) else { throw JSONError(message: "cannot read \(file)") }
        do { return try parse(String(decoding: bytes, as: UTF8.self)) } catch let e as JSONError {
            throw JSONError(message: "\(file): \(e.message)")
        }
    }
}

public struct JSONError: Error, CustomStringConvertible {
    public let message: String
    public init(message: String) { self.message = message }
    public var description: String { message }
}

private struct JSONParser {
    let bytes: [UInt8]
    var pos = 0

    func error(_ what: String) -> JSONError {
        let line = bytes[..<min(pos, bytes.count)].filter { $0 == 10 }.count + 1
        return JSONError(message: "\(what) at line \(line)")
    }

    mutating func skipSpace() {
        while pos < bytes.count {
            let c = bytes[pos]
            if c == 32 || c == 9 || c == 10 || c == 13 { pos += 1; continue }
            // Allow // line comments in config files.
            if c == UInt8(ascii: "/") && pos + 1 < bytes.count && bytes[pos + 1] == UInt8(ascii: "/") {
                while pos < bytes.count && bytes[pos] != 10 { pos += 1 }
                continue
            }
            break
        }
    }

    mutating func value() throws -> JSONValue {
        skipSpace()
        guard pos < bytes.count else { throw error("unexpected end") }
        switch bytes[pos] {
        case UInt8(ascii: "{"):
            pos += 1
            var object: [String: JSONValue] = [:]
            skipSpace()
            if pos < bytes.count && bytes[pos] == UInt8(ascii: "}") { pos += 1; return .object(object) }
            while true {
                skipSpace()
                guard case .string(let key) = try value() else { throw error("expected key") }
                skipSpace()
                guard pos < bytes.count && bytes[pos] == UInt8(ascii: ":") else { throw error("expected ':'") }
                pos += 1
                object[key] = try value()
                skipSpace()
                guard pos < bytes.count else { throw error("unterminated object") }
                if bytes[pos] == UInt8(ascii: ",") { pos += 1; continue }
                if bytes[pos] == UInt8(ascii: "}") { pos += 1; return .object(object) }
                throw error("expected ',' or '}'")
            }
        case UInt8(ascii: "["):
            pos += 1
            var array: [JSONValue] = []
            skipSpace()
            if pos < bytes.count && bytes[pos] == UInt8(ascii: "]") { pos += 1; return .array(array) }
            while true {
                array.append(try value())
                skipSpace()
                guard pos < bytes.count else { throw error("unterminated array") }
                if bytes[pos] == UInt8(ascii: ",") { pos += 1; continue }
                if bytes[pos] == UInt8(ascii: "]") { pos += 1; return .array(array) }
                throw error("expected ',' or ']'")
            }
        case UInt8(ascii: "\""):
            pos += 1
            var out: [UInt8] = []
            while pos < bytes.count && bytes[pos] != UInt8(ascii: "\"") {
                if bytes[pos] == UInt8(ascii: "\\") && pos + 1 < bytes.count {
                    pos += 1
                    switch bytes[pos] {
                    case UInt8(ascii: "n"): out.append(10)
                    case UInt8(ascii: "t"): out.append(9)
                    case UInt8(ascii: "r"): out.append(13)
                    case UInt8(ascii: "u"):
                        let hex = String(decoding: bytes[(pos + 1)..<min(pos + 5, bytes.count)], as: UTF8.self)
                        if let code = UInt32(hex, radix: 16), let scalar = Unicode.Scalar(code) {
                            out += Array(String(Character(scalar)).utf8)
                        }
                        pos += 4
                    default: out.append(bytes[pos])
                    }
                } else {
                    out.append(bytes[pos])
                }
                pos += 1
            }
            pos += 1
            return .string(String(decoding: out, as: UTF8.self))
        case UInt8(ascii: "t"):
            pos += 4
            return .bool(true)
        case UInt8(ascii: "f"):
            pos += 5
            return .bool(false)
        case UInt8(ascii: "n"):
            pos += 4
            return .null
        default:
            let start = pos
            while pos < bytes.count, "+-0123456789.eE".utf8.contains(bytes[pos]) { pos += 1 }
            guard let d = Double(String(decoding: bytes[start..<pos], as: UTF8.self)) else { throw error("bad value") }
            return .number(d)
        }
    }
}
