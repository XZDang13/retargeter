// Server -> browser WebSocket connections (RFC 6455) for the live viewer: the handshake (SHA-1 + base64 of the
// client key) and unmasked, unfragmented server frames. Messages from the browser are not read.

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

enum WebSocket {
    static let guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    /// The 101 response that accepts `key` (the request's Sec-WebSocket-Key).
    static func handshake(key: String) -> [UInt8] {
        let accept = base64(sha1(Array((key + guid).utf8)))
        return Array(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      + "Sec-WebSocket-Accept: \(accept)\r\n\r\n").utf8)
    }

    /// One frame: opcode 1 text, 2 binary, 8 close.
    static func frame(_ payload: [UInt8], opcode: UInt8) -> [UInt8] {
        var out: [UInt8] = [0x80 | opcode]
        let n = payload.count
        if n < 126 {
            out.append(UInt8(n))
        } else if n < 65536 {
            out += [126, UInt8(n >> 8), UInt8(n & 0xff)]
        } else {
            out.append(127)
            for shift in stride(from: 56, through: 0, by: -8) { out.append(UInt8((UInt64(n) >> UInt64(shift)) & 0xff)) }
        }
        return out + payload
    }

    static func sha1(_ message: [UInt8]) -> [UInt8] {
        var h: [UInt32] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
        var data = message + [0x80]
        while data.count % 64 != 56 { data.append(0) }
        let bits = UInt64(message.count) * 8
        for shift in stride(from: 56, through: 0, by: -8) { data.append(UInt8((bits >> UInt64(shift)) & 0xff)) }
        func rotl(_ x: UInt32, _ n: UInt32) -> UInt32 { (x << n) | (x >> (32 - n)) }
        for chunk in stride(from: 0, to: data.count, by: 64) {
            var w = [UInt32](repeating: 0, count: 80)
            for i in 0..<16 {
                let b = chunk + 4 * i
                w[i] = UInt32(data[b]) << 24 | UInt32(data[b + 1]) << 16 | UInt32(data[b + 2]) << 8 | UInt32(data[b + 3])
            }
            for i in 16..<80 { w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1) }
            var (a, b, c, d, e) = (h[0], h[1], h[2], h[3], h[4])
            for i in 0..<80 {
                let (f, k): (UInt32, UInt32) =
                    i < 20 ? ((b & c) | (~b & d), 0x5A827999)
                    : i < 40 ? (b ^ c ^ d, 0x6ED9EBA1)
                    : i < 60 ? ((b & c) | (b & d) | (c & d), 0x8F1BBCDC)
                    : (b ^ c ^ d, 0xCA62C1D6)
                let t = rotl(a, 5) &+ f &+ e &+ k &+ w[i]
                (e, d, c, b, a) = (d, c, rotl(b, 30), a, t)
            }
            h[0] &+= a; h[1] &+= b; h[2] &+= c; h[3] &+= d; h[4] &+= e
        }
        return h.flatMap { v in [24, 16, 8, 0].map { UInt8((v >> UInt32($0)) & 0xff) } }
    }

    static func base64(_ bytes: [UInt8]) -> String {
        let table = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".utf8)
        var out: [UInt8] = []
        for i in stride(from: 0, to: bytes.count, by: 3) {
            let n = min(3, bytes.count - i)
            let v = UInt32(bytes[i]) << 16 | (n > 1 ? UInt32(bytes[i + 1]) << 8 : 0) | (n > 2 ? UInt32(bytes[i + 2]) : 0)
            for k in 0..<4 { out.append(k <= n ? table[Int((v >> UInt32(18 - 6 * k)) & 63)] : UInt8(ascii: "=")) }
        }
        return String(decoding: out, as: UTF8.self)
    }
}
