// Minimal USDA (text USD) reader: enough of the grammar to read robot assets exported by the
// MuJoCo USD converter. Prims, attributes, relationships and list-op metadata are kept; variant
// sets, dictionaries and time samples are skipped.

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public indirect enum USDValue: Sendable, Equatable {
    case number(Double)
    case string(String)
    case token(String)        // bare identifier, e.g. None, true
    case path(String)         // </a/b>
    case asset(String)        // @./file.usda@
    case tuple([USDValue])
    case array([USDValue])

    public var double: Double? {
        switch self {
        case .number(let d): d
        case .token("true"): 1
        case .token("false"): 0
        default: nil
        }
    }

    public var stringValue: String? {
        switch self {
        case .string(let s), .token(let s), .path(let s), .asset(let s): s
        default: nil
        }
    }

    public var vec3: Vec3? {
        guard case .tuple(let items) = self, items.count == 3,
              let x = items[0].double, let y = items[1].double, let z = items[2].double else { return nil }
        return Vec3(x, y, z)
    }

    /// USD quaternions are written (real, i, j, k).
    public var quat: Quat? {
        guard case .tuple(let items) = self, items.count == 4 else { return nil }
        let d = items.compactMap(\.double)
        guard d.count == 4 else { return nil }
        return Quat(w: d[0], x: d[1], y: d[2], z: d[3]).normalized
    }

    public var items: [USDValue] {
        switch self {
        case .array(let a), .tuple(let a): a
        default: [self]
        }
    }
}

public struct USDProperty: Sendable {
    public var typeName: String?   // nil for relationships
    public var value: USDValue?
}

public final class USDPrim: @unchecked Sendable {
    public let name: String
    public internal(set) var typeName: String?
    public let path: String
    public internal(set) var specifier: String   // def | over | class
    public internal(set) var properties: [String: USDProperty] = [:]
    public internal(set) var apiSchemas: [String] = []
    public internal(set) var references: [String] = []  // raw asset+path, e.g. "@./instances.usda@</Instances/pelvis>"
    public internal(set) var children: [USDPrim] = []

    init(name: String, typeName: String?, path: String, specifier: String) {
        self.name = name
        self.typeName = typeName
        self.path = path
        self.specifier = specifier
    }

    public subscript(attribute name: String) -> USDValue? { properties[name]?.value }

    public func child(named name: String) -> USDPrim? { children.first { $0.name == name } }

    /// Local transform from xformOp:translate / xformOp:orient (scale must be 1 in our assets).
    public var localTransform: Transform {
        Transform(rotation: self[attribute: "xformOp:orient"]?.quat ?? .identity,
                  translation: self[attribute: "xformOp:translate"]?.vec3 ?? .zero)
    }

    /// Depth-first traversal including self.
    public func walk(_ visit: (USDPrim) -> Void) {
        visit(self)
        for child in children { child.walk(visit) }
    }
}

public struct USDParseError: Error, CustomStringConvertible {
    public let message: String
    public let line: Int
    public var description: String { "USDA parse error at line \(line): \(message)" }
}

/// A parsed .usda layer. `prims` are the root prims; `index` maps absolute paths to prims.
public final class USDLayer: @unchecked Sendable {
    public let file: String
    public private(set) var prims: [USDPrim] = []
    public private(set) var index: [String: USDPrim] = [:]

    public init(file: String) throws {
        self.file = file
        guard let bytes = readFileBytes(file) else {
            throw USDParseError(message: "cannot read \(file)", line: 0)
        }
        var parser = Parser(bytes: bytes)
        prims = try parser.parseLayer()
        for prim in prims { prim.walk { index[$0.path] = $0 } }
    }

    public func prim(at path: String) -> USDPrim? { index[path] }
}

/// Several layers composed by path: later layers' opinions ("over" or "def") merge into earlier prims.
/// This is the small subset of USD composition our assets need (sublayers + payload overs).
public final class USDStage: @unchecked Sendable {
    public private(set) var index: [String: USDPrim] = [:]
    public private(set) var roots: [USDPrim] = []

    public init(layers: [USDLayer]) {
        for layer in layers {
            for prim in layer.prims { merge(prim, parent: nil) }
        }
    }

    private func merge(_ prim: USDPrim, parent: USDPrim?) {
        let target: USDPrim
        if let existing = index[prim.path] {
            target = existing
            if prim.specifier == "def", let typeName = prim.typeName { existing.typeName = typeName }
            for (key, value) in prim.properties { existing.properties[key] = value }
            existing.apiSchemas = Array(Set(existing.apiSchemas + prim.apiSchemas)).sorted()
            existing.references += prim.references
        } else {
            target = USDPrim(name: prim.name, typeName: prim.typeName, path: prim.path, specifier: prim.specifier)
            target.properties = prim.properties
            target.apiSchemas = prim.apiSchemas
            target.references = prim.references
            index[prim.path] = target
            if let parent { parent.children.append(target) } else { roots.append(target) }
        }
        for child in prim.children { merge(child, parent: target) }
    }

    public func prim(at path: String) -> USDPrim? { index[path] }
}

func readFileBytes(_ path: String) -> [UInt8]? {
    guard let file = fopen(path, "rb") else { return nil }
    defer { fclose(file) }
    fseek(file, 0, SEEK_END)
    let size = ftell(file)
    fseek(file, 0, SEEK_SET)
    guard size >= 0 else { return nil }
    var bytes = [UInt8](repeating: 0, count: size)
    let read = bytes.withUnsafeMutableBytes { fread($0.baseAddress, 1, size, file) }
    return read == size ? bytes : nil
}

// MARK: - Tokenizer + parser

private enum Token: Equatable {
    case identifier(String)  // includes namespaced names (a:b.c) and type names (float3[])
    case string(String)
    case number(Double)
    case path(String)
    case asset(String)
    case punct(UInt8)
    case end
}

private struct Parser {
    let bytes: [UInt8]
    var pos = 0
    var line = 1
    var lookahead: Token?
    var lookaheadLine = 0
    /// Line on which the most recently returned token started.
    var tokenLine = 0

    init(bytes: [UInt8]) {
        self.bytes = bytes
        // Skip the "#usda 1.0" header line.
        while pos < bytes.count && bytes[pos] != UInt8(ascii: "\n") { pos += 1 }
    }

    // MARK: Tokens

    mutating func skipTrivia() {
        while pos < bytes.count {
            let c = bytes[pos]
            if c == UInt8(ascii: "\n") { line += 1; pos += 1 }
            else if c == UInt8(ascii: " ") || c == UInt8(ascii: "\t") || c == UInt8(ascii: "\r") { pos += 1 }
            else if c == UInt8(ascii: "#") {
                while pos < bytes.count && bytes[pos] != UInt8(ascii: "\n") { pos += 1 }
            } else { break }
        }
    }

    static func isIdentStart(_ c: UInt8) -> Bool {
        (c >= 65 && c <= 90) || (c >= 97 && c <= 122) || c == UInt8(ascii: "_")
    }

    static func isIdentBody(_ c: UInt8) -> Bool {
        isIdentStart(c) || (c >= 48 && c <= 57) || c == UInt8(ascii: ":") || c == UInt8(ascii: ".")
    }

    mutating func lex() throws -> Token {
        skipTrivia()
        guard pos < bytes.count else { return .end }
        let c = bytes[pos]
        if c == UInt8(ascii: "\"") || c == UInt8(ascii: "'") {
            // Triple-quoted or single-line string.
            if pos + 2 < bytes.count && bytes[pos + 1] == c && bytes[pos + 2] == c {
                pos += 3
                let start = pos
                while pos + 2 < bytes.count && !(bytes[pos] == c && bytes[pos + 1] == c && bytes[pos + 2] == c) {
                    if bytes[pos] == UInt8(ascii: "\n") { line += 1 }
                    pos += 1
                }
                let s = String(decoding: bytes[start..<pos], as: UTF8.self)
                pos += 3
                return .string(s)
            }
            pos += 1
            var out: [UInt8] = []
            while pos < bytes.count && bytes[pos] != c {
                if bytes[pos] == UInt8(ascii: "\\") && pos + 1 < bytes.count { pos += 1 }
                out.append(bytes[pos])
                pos += 1
            }
            pos += 1
            return .string(String(decoding: out, as: UTF8.self))
        }
        if c == UInt8(ascii: "<") {
            let start = pos + 1
            while pos < bytes.count && bytes[pos] != UInt8(ascii: ">") { pos += 1 }
            let s = String(decoding: bytes[start..<pos], as: UTF8.self)
            pos += 1
            return .path(s)
        }
        if c == UInt8(ascii: "@") {
            let start = pos + 1
            pos += 1
            while pos < bytes.count && bytes[pos] != UInt8(ascii: "@") { pos += 1 }
            let s = String(decoding: bytes[start..<pos], as: UTF8.self)
            pos += 1
            return .asset(s)
        }
        if (c >= 48 && c <= 57) || c == UInt8(ascii: "-") || c == UInt8(ascii: "+") || c == UInt8(ascii: ".") {
            var end = pos
            while end < bytes.count {
                let d = bytes[end]
                if (d >= 48 && d <= 57) || d == UInt8(ascii: "-") || d == UInt8(ascii: "+") || d == UInt8(ascii: ".")
                    || d == UInt8(ascii: "e") || d == UInt8(ascii: "E") { end += 1 } else { break }
            }
            let text = String(decoding: bytes[pos..<end], as: UTF8.self)
            pos = end
            guard let value = Double(text) else { throw USDParseError(message: "bad number \(text)", line: line) }
            return .number(value)
        }
        if Parser.isIdentStart(c) {
            let start = pos
            while pos < bytes.count && Parser.isIdentBody(bytes[pos]) { pos += 1 }
            // Array type suffix: float3[]
            if pos + 1 < bytes.count && bytes[pos] == UInt8(ascii: "[") && bytes[pos + 1] == UInt8(ascii: "]") { pos += 2 }
            return .identifier(String(decoding: bytes[start..<pos], as: UTF8.self))
        }
        pos += 1
        return .punct(c)
    }

    mutating func peek() throws -> Token {
        if let lookahead { return lookahead }
        skipTrivia()
        lookaheadLine = line
        let token = try lex()
        lookahead = token
        return token
    }

    /// Line on which the next token starts.
    mutating func peekLine() throws -> Int {
        _ = try peek()
        return lookaheadLine
    }

    mutating func next() throws -> Token {
        if let token = lookahead {
            lookahead = nil
            tokenLine = lookaheadLine
            return token
        }
        skipTrivia()
        tokenLine = line
        return try lex()
    }

    mutating func expect(_ c: Character) throws {
        let token = try next()
        guard token == .punct(UInt8(ascii: c.unicodeScalars.first!)) else {
            throw USDParseError(message: "expected '\(c)', got \(token)", line: line)
        }
    }

    mutating func isPunct(_ c: Character) throws -> Bool {
        try peek() == .punct(UInt8(ascii: c.unicodeScalars.first!))
    }

    /// Skips a balanced (...), [...] or {...} group whose opening token was already consumed.
    mutating func skipGroup(open: UInt8, close: UInt8) throws {
        var depth = 1
        while depth > 0 {
            let token = try next()
            if token == .end { throw USDParseError(message: "unterminated group", line: line) }
            if token == .punct(open) { depth += 1 }
            if token == .punct(close) { depth -= 1 }
        }
    }

    // MARK: Grammar

    mutating func parseLayer() throws -> [USDPrim] {
        if try isPunct("(") {
            _ = try next()
            try skipGroup(open: UInt8(ascii: "("), close: UInt8(ascii: ")"))
        }
        var prims: [USDPrim] = []
        while try peek() != .end {
            prims.append(try parsePrim(parentPath: ""))
        }
        return prims
    }

    mutating func parsePrim(parentPath: String) throws -> USDPrim {
        guard case .identifier(let specifier) = try next(), ["def", "over", "class"].contains(specifier) else {
            throw USDParseError(message: "expected prim specifier", line: line)
        }
        var typeName: String?
        if case .identifier(let t) = try peek() {
            typeName = t
            _ = try next()
        }
        guard case .string(let name) = try next() else { throw USDParseError(message: "expected prim name", line: line) }
        let prim = USDPrim(name: name, typeName: typeName, path: parentPath + "/" + name, specifier: specifier)
        if try isPunct("(") {
            _ = try next()
            try parsePrimMetadata(into: prim)
        }
        try expect("{")
        while !(try isPunct("}")) {
            try parseBodyItem(into: prim)
        }
        try expect("}")
        return prim
    }

    /// Reads `apiSchemas` and `references` out of prim metadata; everything else is skipped.
    mutating func parsePrimMetadata(into prim: USDPrim) throws {
        while true {
            let token = try next()
            switch token {
            case .punct(UInt8(ascii: ")")):
                return
            case .identifier(let key) where key == "apiSchemas" || key == "references":
                try expect("=")
                var value = try parseValue()
                if case .asset(let a) = value, case .path(let p) = try peek() {
                    _ = try next()
                    value = .array([.asset(a), .path(p)])
                }
                for item in value.items {
                    if key == "apiSchemas", let s = item.stringValue { prim.apiSchemas.append(s) }
                }
                if key == "references" { prim.references += referenceStrings(value) }
            case .punct(let c) where c == UInt8(ascii: "(") || c == UInt8(ascii: "[") || c == UInt8(ascii: "{"):
                try skipGroup(open: c, close: c == UInt8(ascii: "(") ? UInt8(ascii: ")") : c == UInt8(ascii: "[") ? UInt8(ascii: "]") : UInt8(ascii: "}"))
            case .end:
                throw USDParseError(message: "unterminated prim metadata", line: line)
            default:
                continue
            }
        }
    }

    func referenceStrings(_ value: USDValue) -> [String] {
        // A reference is an asset optionally followed by a path; parseValue yields them as array items.
        var out: [String] = []
        var pending: String?
        for item in value.items {
            switch item {
            case .asset(let a):
                if let p = pending { out.append(p) }
                pending = "@\(a)@"
            case .path(let p):
                out.append((pending ?? "") + "<\(p)>")
                pending = nil
            default: break
            }
        }
        if let p = pending { out.append(p) }
        return out
    }

    mutating func parseBodyItem(into prim: USDPrim) throws {
        let token = try peek()
        guard case .identifier(let word) = token else {
            throw USDParseError(message: "unexpected \(token) in prim body", line: line)
        }
        if ["def", "over", "class"].contains(word) {
            prim.children.append(try parsePrim(parentPath: prim.path))
            return
        }
        if word == "variantSet" {
            _ = try next()
            _ = try next()  // name
            try expect("=")
            try expect("{")
            try skipGroup(open: UInt8(ascii: "{"), close: UInt8(ascii: "}"))
            return
        }
        try parseProperty(into: prim)
    }

    mutating func parseProperty(into prim: USDPrim) throws {
        var words: [String] = []
        // Qualifiers, type and name are identifiers on one line; the last one is the property name.
        let startLine = try peekLine()
        while case .identifier(let w) = try peek(), try peekLine() == startLine {
            _ = try next()
            words.append(w)
        }
        guard let name = words.last else { throw USDParseError(message: "expected property", line: line) }
        let qualifiers: Set<String> = ["uniform", "custom", "prepend", "append", "delete", "add", "reorder", "varying", "config"]
        let rest = words.dropLast().filter { !qualifiers.contains($0) }
        let typeName = rest.last
        let isRelationship = typeName == "rel"
        let listOp = words.first { ["prepend", "append", "delete", "add"].contains($0) }

        var value: USDValue?
        if try isPunct("=") {
            _ = try next()
            value = try parseValue()
        }
        if try isPunct("(") {
            _ = try next()
            try skipGroup(open: UInt8(ascii: "("), close: UInt8(ascii: ")"))
        }
        if name.hasSuffix(".timeSamples") || name.hasSuffix(".connect") { return }
        // A bare declaration (e.g. "custom rel physics:body0") must not erase a value set by a list op.
        if value == nil, prim.properties[name] != nil { return }
        if listOp == "delete" { return }
        prim.properties[name] = USDProperty(typeName: isRelationship ? nil : typeName, value: value)
    }

    mutating func parseValue() throws -> USDValue {
        let token = try next()
        switch token {
        case .number(let d): return .number(d)
        case .string(let s): return .string(s)
        case .path(let p): return .path(p)
        case .asset(let a):
            return .asset(a)
        case .identifier(let w): return .token(w)
        case .punct(UInt8(ascii: "(")):
            return .tuple(try parseList(close: UInt8(ascii: ")")))
        case .punct(UInt8(ascii: "[")):
            return .array(try parseList(close: UInt8(ascii: "]")))
        case .punct(UInt8(ascii: "{")):
            try skipGroup(open: UInt8(ascii: "{"), close: UInt8(ascii: "}"))
            return .tuple([])
        default:
            throw USDParseError(message: "unexpected \(token) in value", line: line)
        }
    }

    mutating func parseList(close: UInt8) throws -> [USDValue] {
        var items: [USDValue] = []
        while true {
            if try peek() == .punct(close) {
                _ = try next()
                return items
            }
            if try isPunct(",") {
                _ = try next()
                continue
            }
            items.append(try parseValue())
        }
    }
}
