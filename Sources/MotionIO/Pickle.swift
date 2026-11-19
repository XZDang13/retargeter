// Restricted pickle reader for motion files saved with pickle (e.g. OMOMO's SMPL-X .pkl): a dict of numpy arrays,
// strings and numbers. It interprets the pickle opcodes itself and only knows how to build numpy arrays, so a file
// can never run code; any other class or function in the file is an error.

import Core

public enum Pickle {
    indirect enum Value {
        case none, bool(Bool), int(Int), float(Double), string(String), bytes([UInt8])
        case list([Value]), tuple([Value]), dict([(Value, Value)])
        case global(String)            // "module.name" of an allowed callable
        case dtype(String, String)     // numpy dtype: type code ("f8", "U6", ...) and byte order
        case array(NPYArray)
        case mark
    }

    /// The numpy callables a pickled array uses (numpy 1 and 2 module paths).
    static let allowed: Set<String> = [
        "numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct", "numpy.ndarray", "numpy.dtype",
        "numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar",
        "_codecs.encode",   // protocol 2 stores bytes as encode(latin-1 text)
    ]

    /// Every entry of a pickled dict that is an array, a number or a string, as arrays (numbers and strings become
    /// 0-d arrays, lists of numbers 1-d float64 arrays). Other entries are skipped.
    public static func arrays(file: String) throws -> [String: NPYArray] {
        guard let bytes = readFile(file) else { throw NPYError(message: "cannot read \(file)") }
        let value: Value
        do { value = try load(bytes) } catch let e as NPYError { throw NPYError(message: "\(file): \(e.message)") }
        guard case .dict(let items) = value else { throw NPYError(message: "\(file): pickle is not a dict") }
        var out: [String: NPYArray] = [:]
        for (key, item) in items {
            guard case .string(let name) = key, let array = asArray(item) else { continue }
            out[name] = array
        }
        return out
    }

    static func asArray(_ value: Value) -> NPYArray? {
        func scalar<T: BitwiseCopyable>(_ v: T, _ dtype: NPYArray.DType) -> NPYArray {
            NPYArray(dtype: dtype, shape: [], bytes: withUnsafeBytes(of: v) { Array($0) })
        }
        switch value {
        case .array(let a): return a
        case .int(let i): return scalar(Int64(i), .int64)
        case .float(let d): return scalar(d, .float64)
        case .bool(let b): return NPYArray(dtype: .bool, shape: [], bytes: [b ? 1 : 0])
        case .string(let s):
            let units = s.unicodeScalars.map(\.value)
            return NPYArray(dtype: .unicode(max(units.count, 1)), shape: [],
                            bytes: (units.isEmpty ? [0] : units).withUnsafeBytes { Array($0) })
        case .list(let items), .tuple(let items):
            var numbers: [Double] = []
            for item in items {
                switch item {
                case .int(let i): numbers.append(Double(i))
                case .float(let d): numbers.append(d)
                default: return nil
                }
            }
            return NPYArray(dtype: .float64, shape: [numbers.count], bytes: numbers.withUnsafeBytes { Array($0) })
        default: return nil
        }
    }

    static func load(_ b: [UInt8]) throws -> Value {
        var stack: [Value] = [], memo: [Int: Value] = [:], p = 0
        func fail(_ m: String) -> NPYError { NPYError(message: "pickle: \(m) at byte \(p)") }
        func take(_ n: Int) throws -> ArraySlice<UInt8> {
            guard n >= 0, p + n <= b.count else { throw fail("truncated") }
            defer { p += n }
            return b[p..<(p + n)]
        }
        func uint(_ n: Int) throws -> Int { try take(n).reversed().reduce(0) { $0 << 8 | Int($1) } }
        func line() throws -> String {
            guard let end = b[p...].firstIndex(of: 0x0A) else { throw fail("truncated") }
            defer { p = end + 1 }
            return String(decoding: b[p..<end], as: UTF8.self)
        }
        func pop() throws -> Value {
            guard let v = stack.popLast() else { throw fail("stack underflow") }
            return v
        }
        func popMark() throws -> [Value] {
            guard let m = stack.lastIndex(where: { if case .mark = $0 { return true }; return false }) else { throw fail("no mark") }
            let items = Array(stack[(m + 1)...])
            stack.removeSubrange(m...)
            return items
        }
        func global(_ module: String, _ name: String) throws -> Value {
            let full = module + "." + name
            guard allowed.contains(full) else { throw fail("refusing to load \(full) (only numpy arrays are read)") }
            return .global(full)
        }
        func reduce(_ callable: Value, _ args: Value) throws -> Value {
            guard case .global(let f) = callable, case .tuple(let a) = args else { throw fail("unsupported call") }
            if f.hasSuffix("._reconstruct") { return .array(NPYArray(dtype: .uint8, shape: [0], bytes: [])) }  // filled by BUILD
            if f == "numpy.dtype", case .string(let code)? = a.first { return .dtype(code, "|") }
            if f == "_codecs.encode", a.count == 2, case .string(let text) = a[0], case .string(let encoding) = a[1],
               ["latin1", "latin-1"].contains(encoding) {
                return .bytes(text.unicodeScalars.map { UInt8(truncatingIfNeeded: $0.value) })
            }
            if f.hasSuffix(".scalar"), a.count == 2, case .dtype(let code, let order) = a[0], case .bytes(let raw) = a[1] {
                return .array(NPYArray(dtype: try dtype(code, order), shape: [], bytes: raw))
            }
            throw fail("unsupported call \(f)")
        }
        func build(_ object: Value, _ state: Value) throws -> Value {
            switch object {
            case .dtype(let code, _):
                // State (version, byte order, ...).
                guard case .tuple(let s) = state, s.count > 1, case .string(let order) = s[1] else { return object }
                return .dtype(code, order)
            case .array:
                // State (version, shape, dtype, fortran order, raw data).
                guard case .tuple(let s) = state, s.count == 5, case .tuple(let shape) = s[1], case .dtype(let code, let order) = s[2]
                else { throw fail("unsupported array state") }
                if case .bool(true) = s[3] { throw fail("fortran-order arrays are not supported") }
                let dims = try shape.map { v -> Int in guard case .int(let n) = v else { throw fail("bad shape") }; return n }
                let raw: [UInt8]
                switch s[4] {
                case .bytes(let r): raw = r
                case .string(let latin1): raw = latin1.unicodeScalars.map { UInt8(truncatingIfNeeded: $0.value) }  // protocol 2
                default: throw fail("object arrays are not supported")
                }
                let array = NPYArray(dtype: try dtype(code, order), shape: dims, bytes: raw)
                guard raw.count == array.count * array.dtype.size else { throw fail("array data size mismatch") }
                return .array(array)
            default:
                throw fail("unsupported BUILD")
            }
        }
        func dtype(_ code: String, _ order: String) throws -> NPYArray.DType {
            guard order != ">" else { throw fail("big-endian arrays are not supported") }
            guard let t = NPYArray.DType(descr: (code.hasPrefix("U") ? "<" : "|") + code)
                    ?? NPYArray.DType(descr: "<" + code) else { throw fail("unsupported dtype \(code)") }
            return t
        }

        while true {
            let op = try take(1).first!
            switch op {
            case 0x80: _ = try take(1)                                   // PROTO
            case 0x95: _ = try take(8)                                   // FRAME
            case 0x2E: return try pop()                                  // STOP
            case 0x28: stack.append(.mark)                               // MARK
            case 0x4E: stack.append(.none)                               // NONE
            case 0x88: stack.append(.bool(true))                         // NEWTRUE
            case 0x89: stack.append(.bool(false))                        // NEWFALSE
            case 0x4A: stack.append(.int(Int(Int32(truncatingIfNeeded: try uint(4)))))  // BININT
            case 0x4B: stack.append(.int(try uint(1)))                   // BININT1
            case 0x4D: stack.append(.int(try uint(2)))                   // BININT2
            case 0x8A:                                                   // LONG1
                let n = try uint(1), digits = try take(n)
                guard n <= 8 else { throw fail("integer too large") }
                var v = digits.reversed().reduce(0) { $0 << 8 | Int($1) }
                if n > 0, n < 8, digits.last! & 0x80 != 0 { v -= 1 << (8 * n) }
                stack.append(.int(v))
            case 0x47:                                                   // BINFLOAT (big-endian double)
                stack.append(.float(Double(bitPattern: try take(8).reduce(UInt64(0)) { $0 << 8 | UInt64($1) })))
            case 0x8C: stack.append(.string(String(decoding: try take(try uint(1)), as: UTF8.self)))  // SHORT_BINUNICODE
            case 0x58: stack.append(.string(String(decoding: try take(try uint(4)), as: UTF8.self)))  // BINUNICODE
            case 0x8D: stack.append(.string(String(decoding: try take(try uint(8)), as: UTF8.self)))  // BINUNICODE8
            case 0x55, 0x54:                                             // SHORT_BINSTRING, BINSTRING (latin-1)
                let raw = try take(op == 0x55 ? try uint(1) : try uint(4))
                stack.append(.string(String(raw.map { Character(Unicode.Scalar($0)) })))
            case 0x43: stack.append(.bytes(Array(try take(try uint(1)))))  // SHORT_BINBYTES
            case 0x42: stack.append(.bytes(Array(try take(try uint(4)))))  // BINBYTES
            case 0x8E: stack.append(.bytes(Array(try take(try uint(8)))))  // BINBYTES8
            case 0x96: stack.append(.bytes(Array(try take(try uint(8)))))  // BYTEARRAY8
            case 0x29: stack.append(.tuple([]))                          // EMPTY_TUPLE
            case 0x85: stack.append(.tuple([try pop()]))                 // TUPLE1
            case 0x86: let b2 = try pop(), a = try pop(); stack.append(.tuple([a, b2]))   // TUPLE2
            case 0x87: let c = try pop(), b2 = try pop(), a = try pop(); stack.append(.tuple([a, b2, c]))  // TUPLE3
            case 0x74: stack.append(.tuple(try popMark()))               // TUPLE
            case 0x5D: stack.append(.list([]))                           // EMPTY_LIST
            case 0x6C: stack.append(.list(try popMark()))                // LIST
            case 0x7D: stack.append(.dict([]))                           // EMPTY_DICT
            case 0x64:                                                   // DICT
                let items = try popMark()
                stack.append(.dict(stride(from: 0, to: items.count - 1, by: 2).map { (items[$0], items[$0 + 1]) }))
            case 0x61, 0x65:                                             // APPEND, APPENDS
                let items = op == 0x61 ? [try pop()] : try popMark()
                guard case .list(let l) = try pop() else { throw fail("APPEND to a non-list") }
                stack.append(.list(l + items))
            case 0x73, 0x75:                                             // SETITEM, SETITEMS
                let items: [Value]
                if op == 0x73 { let v = try pop(), k = try pop(); items = [k, v] } else { items = try popMark() }
                guard case .dict(var d) = try pop() else { throw fail("SETITEM on a non-dict") }
                for i in stride(from: 0, to: items.count - 1, by: 2) { d.append((items[i], items[i + 1])) }
                stack.append(.dict(d))
            case 0x94: memo[memo.count] = stack.last                     // MEMOIZE
            case 0x71: memo[try uint(1)] = stack.last                    // BINPUT
            case 0x72: memo[try uint(4)] = stack.last                    // LONG_BINPUT
            case 0x70: memo[Int(try line()) ?? -1] = stack.last          // PUT
            case 0x68, 0x6A, 0x67:                                       // BINGET, LONG_BINGET, GET
                let k = op == 0x68 ? try uint(1) : op == 0x6A ? try uint(4) : Int(try line()) ?? -1
                guard let v = memo[k] else { throw fail("bad memo reference") }
                stack.append(v)
            case 0x63:                                                   // GLOBAL
                let module = try line(), name = try line()
                stack.append(try global(module, name))
            case 0x93:                                                   // STACK_GLOBAL
                guard case .string(let name) = try pop(), case .string(let module) = try pop() else { throw fail("bad STACK_GLOBAL") }
                stack.append(try global(module, name))
            case 0x52:                                                   // REDUCE
                let args = try pop(), callable = try pop()
                stack.append(try reduce(callable, args))
            case 0x62:                                                   // BUILD
                let state = try pop(), object = try pop()
                stack.append(try build(object, state))
            case 0x30: _ = try pop()                                     // POP
            case 0x32: stack.append(stack.last ?? .none)                 // DUP
            default:
                throw fail("unsupported opcode 0x\(String(op, radix: 16))")
            }
        }
    }
}
