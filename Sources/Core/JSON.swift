// Tiny JSON writer (the project avoids Foundation).

public indirect enum JSON {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSON])
    case object([(String, JSON)])

    public var text: String {
        var out = ""
        write(&out)
        return out
    }

    public func write(_ out: inout String) {
        switch self {
        case .null: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .number(let d):
            if d.isFinite { out += d == d.rounded() && abs(d) < 1e15 ? String(Int(d)) : String(d) } else { out += "null" }
        case .string(let s): JSON.escape(s, into: &out)
        case .array(let items):
            out += "["
            for (i, item) in items.enumerated() {
                if i > 0 { out += "," }
                item.write(&out)
            }
            out += "]"
        case .object(let fields):
            out += "{"
            for (i, (key, value)) in fields.enumerated() {
                if i > 0 { out += "," }
                JSON.escape(key, into: &out)
                out += ":"
                value.write(&out)
            }
            out += "}"
        }
    }

    public static func escape(_ s: String, into out: inout String) {
        out += "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if scalar.value < 0x20 {
                    let hex = String(scalar.value, radix: 16)
                    out += "\\u" + String(repeating: "0", count: 4 - hex.count) + hex
                } else { out.unicodeScalars.append(scalar) }
            }
        }
        out += "\""
    }
}

extension JSON {
    public static func numbers(_ values: [Double]) -> JSON { .array(values.map { .number($0) }) }
}
