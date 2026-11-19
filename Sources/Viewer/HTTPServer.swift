// Minimal HTTP/1.1 server (GET, and POST with a Content-Length body; one request per connection) on POSIX sockets.
// It serves the browser viewer on localhost; it is not meant to face a network.

import Dispatch

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public struct HTTPRequest: Sendable {
    public var method: String
    public var path: String
    public var query: [String: String]
    /// Header fields, names lowercased.
    public var headers: [String: String] = [:]
    /// POST body.
    public var body: [UInt8] = []
}

public struct HTTPResponse: Sendable {
    public var status: Int
    public var contentType: String
    public var body: [UInt8]
    /// Extra header lines (name, value), e.g. Content-Disposition.
    public var headers: [(String, String)] = []

    public init(status: Int = 200, contentType: String, body: [UInt8], headers: [(String, String)] = []) {
        self.status = status
        self.contentType = contentType
        self.body = body
        self.headers = headers
    }

    public static func text(_ s: String, status: Int = 200) -> HTTPResponse {
        HTTPResponse(status: status, contentType: "text/plain; charset=utf-8", body: Array(s.utf8))
    }

    public static func json(_ s: String) -> HTTPResponse {
        HTTPResponse(contentType: "application/json", body: Array(s.utf8))
    }

    public static let notFound = HTTPResponse.text("not found", status: 404)
}

public final class HTTPServer: @unchecked Sendable {
    public typealias Handler = @Sendable (HTTPRequest) -> HTTPResponse
    /// Takes over an accepted WebSocket connection (after the handshake was sent); it owns the descriptor.
    public typealias Upgrade = @Sendable (HTTPRequest, Int32) -> Void
    let port: UInt16
    let handler: Handler
    let upgrade: Upgrade?
    /// Largest POST body accepted (bytes); larger requests get 413.
    let maxBody: Int
    private var listener: Int32 = -1

    public init(port: UInt16, maxBody: Int = 256 << 20, handler: @escaping Handler, upgrade: Upgrade? = nil) {
        self.port = port
        self.maxBody = maxBody
        self.handler = handler
        self.upgrade = upgrade
    }

    /// Binds 127.0.0.1:port and serves forever.
    public func run() throws -> Never {
        #if canImport(Glibc)
        listener = socket(AF_INET, Int32(SOCK_STREAM.rawValue), 0)
        #else
        listener = socket(AF_INET, SOCK_STREAM, 0)
        #endif
        guard listener >= 0 else { throw HTTPServerError(message: "socket() failed") }
        var yes: Int32 = 1
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = port.bigEndian
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bound == 0 else { throw HTTPServerError(message: "cannot bind 127.0.0.1:\(port) (in use?)") }
        guard listen(listener, 64) == 0 else { throw HTTPServerError(message: "listen() failed") }
        signal(SIGPIPE, SIG_IGN)
        while true {
            let client = accept(listener, nil, nil)
            if client < 0 { continue }
            let handler = self.handler, upgrade = self.upgrade, maxBody = self.maxBody
            DispatchQueue.global().async { HTTPServer.serve(client, handler, upgrade, maxBody: maxBody) }
        }
    }

    private static func serve(_ fd: Int32, _ handler: Handler, _ upgrade: Upgrade?, maxBody: Int) {
        var keep = false
        defer { if !keep { close(fd) } }
        var buffer = [UInt8](repeating: 0, count: 16384)
        var received: [UInt8] = []
        while received.count < 65536 {
            let n = buffer.withUnsafeMutableBytes { recv(fd, $0.baseAddress, $0.count, 0) }
            if n <= 0 { return }
            received += buffer[0..<n]
            if received.count >= 4 && received.suffix(4) == [13, 10, 13, 10] { break }
            if let _ = headerEnd(received) { break }
        }
        guard let end = headerEnd(received) else { return }
        let headText = String(decoding: received[..<end], as: UTF8.self)
        guard let line = headText.split(separator: "\r\n").first else { return }
        let parts = line.split(separator: " ")
        guard parts.count >= 2 else { return }
        let target = String(parts[1])
        let (path, query) = splitTarget(target)
        var request = HTTPRequest(method: String(parts[0]), path: percentDecode(path), query: query)
        for field in headText.split(separator: "\r\n").dropFirst() {
            guard let colon = field.firstIndex(of: ":") else { continue }
            request.headers[field[..<colon].lowercased()] = field[field.index(after: colon)...].trimmingCharacters()
        }
        // WebSocket upgrade: hand the connection over after the handshake.
        if let upgrade, request.method == "GET", let key = request.headers["sec-websocket-key"],
           request.headers["upgrade"]?.lowercased().contains("websocket") == true {
            sendAll(fd, WebSocket.handshake(key: key))
            keep = true
            upgrade(request, fd)
            return
        }
        var response: HTTPResponse
        switch request.method {
        case "GET":
            response = handler(request)
        case "POST":
            guard let length = request.headers["content-length"].flatMap({ Int($0) }), length >= 0 else {
                response = .text("Content-Length required", status: 411)
                break
            }
            guard length <= maxBody else {
                response = .text("too large (limit \(maxBody >> 20) MB)", status: 413)
                break
            }
            var body = Array(received[(end + 4)...].prefix(length))
            body.reserveCapacity(length)
            var chunk = [UInt8](repeating: 0, count: 1 << 20)
            while body.count < length {
                let n = chunk.withUnsafeMutableBytes { recv(fd, $0.baseAddress, min($0.count, length - body.count), 0) }
                if n <= 0 { return }
                body += chunk[0..<n]
            }
            request.body = body
            response = handler(request)
        default:
            response = .text("method not allowed", status: 405)
        }
        let reason = [200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed",
                      411: "Length Required", 413: "Payload Too Large", 500: "Internal Server Error"][response.status] ?? "OK"
        let head = "HTTP/1.1 \(response.status) \(reason)\r\nContent-Type: \(response.contentType)\r\n"
            + response.headers.map { "\($0.0): \($0.1)\r\n" }.joined()
            + "Content-Length: \(response.body.count)\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        sendAll(fd, Array(head.utf8))
        sendAll(fd, response.body)
    }

    private static func headerEnd(_ bytes: [UInt8]) -> Int? {
        guard bytes.count >= 4 else { return nil }
        for i in 0...(bytes.count - 4) where bytes[i] == 13 && bytes[i + 1] == 10 && bytes[i + 2] == 13 && bytes[i + 3] == 10 {
            return i
        }
        return nil
    }

    static func sendAll(_ fd: Int32, _ bytes: [UInt8]) {
        var offset = 0
        while offset < bytes.count {
            let n = bytes[offset...].withUnsafeBytes { send(fd, $0.baseAddress, $0.count, 0) }
            if n <= 0 { return }
            offset += n
        }
    }

    static func splitTarget(_ target: String) -> (String, [String: String]) {
        guard let q = target.firstIndex(of: "?") else { return (target, [:]) }
        var query: [String: String] = [:]
        for pair in target[target.index(after: q)...].split(separator: "&") {
            let kv = pair.split(separator: "=", maxSplits: 1)
            query[percentDecode(String(kv[0]))] = kv.count > 1 ? percentDecode(String(kv[1])) : ""
        }
        return (String(target[..<q]), query)
    }

    static func percentDecode(_ s: String) -> String {
        var out: [UInt8] = []
        var bytes = Array(s.utf8)[...]
        while let c = bytes.popFirst() {
            if c == UInt8(ascii: "+") { out.append(UInt8(ascii: " ")) }
            else if c == UInt8(ascii: "%"), bytes.count >= 2,
                    let v = UInt8(String(decoding: bytes.prefix(2), as: UTF8.self), radix: 16) {
                out.append(v)
                bytes = bytes.dropFirst(2)
            } else { out.append(c) }
        }
        return String(decoding: out, as: UTF8.self)
    }
}

public struct HTTPServerError: Error, CustomStringConvertible {
    public let message: String
    public var description: String { message }
}

extension Substring {
    func trimmingCharacters() -> String {
        String(drop { $0 == " " || $0 == "\t" }.reversed().drop { $0 == " " || $0 == "\t" }.reversed())
    }
}
