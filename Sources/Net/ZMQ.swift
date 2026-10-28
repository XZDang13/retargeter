// ZeroMQ sockets over the CRTNet bridge (native/net): the teleop server's transport.

import CRTNet

public struct NetError: Error, CustomStringConvertible {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var description: String { message }
}

public final class ZMQSocket {
    public enum Kind: Int32 {
        case pair = 0, pub, sub, req, rep, dealer, router, pull, push
    }

    let handle: rtnet_socket

    public init(_ kind: Kind) throws {
        guard let h = rtnet_open(kind.rawValue) else { throw NetError("zmq socket: \(String(cString: rtnet_error()))") }
        handle = h
    }

    deinit { rtnet_close(handle) }

    public func bind(_ endpoint: String) throws {
        guard rtnet_bind(handle, endpoint) == 0 else { throw NetError("zmq bind \(endpoint): \(String(cString: rtnet_error()))") }
    }

    public func connect(_ endpoint: String) throws {
        guard rtnet_connect(handle, endpoint) == 0 else {
            throw NetError("zmq connect \(endpoint): \(String(cString: rtnet_error()))")
        }
    }

    /// Integer option by ZeroMQ number (ZMQ_SNDHWM 23, ZMQ_RCVHWM 24, ZMQ_LINGER 17, ZMQ_ROUTER_MANDATORY 33).
    public func set(_ option: Int32, _ value: Int32) throws {
        guard rtnet_set_int(handle, option, value) == 0 else { throw NetError("zmq option \(option): \(String(cString: rtnet_error()))") }
    }

    /// Sends a multipart message; false if it would block (non-blocking sends only).
    @discardableResult
    public func send(_ parts: [[UInt8]], nonblocking: Bool = false) throws -> Bool {
        for (i, part) in parts.enumerated() {
            let r = part.withUnsafeBytes { rtnet_send(handle, $0.baseAddress, $0.count, i + 1 < parts.count ? 1 : 0, nonblocking ? 1 : 0) }
            if r == -2 { return false }
            guard r == 0 else { throw NetError("zmq send: \(String(cString: rtnet_error()))") }
        }
        return true
    }

    /// Receives one multipart message, waiting up to `timeout` ms (-1: forever, 0: poll); nil on timeout.
    public func receive(timeout: Int32 = -1, capacity: Int = 1 << 16) throws -> [[UInt8]]? {
        var parts: [[UInt8]] = []
        var wait = timeout
        while true {
            var more: Int32 = 0
            var buffer = [UInt8](repeating: 0, count: capacity)
            let n = buffer.withUnsafeMutableBytes { rtnet_recv(handle, $0.baseAddress, $0.count, wait, &more) }
            if n == -2 {
                guard parts.isEmpty else { throw NetError("zmq receive: message parts stopped arriving") }
                return nil
            }
            guard n >= 0 else { throw NetError("zmq receive: \(String(cString: rtnet_error()))") }
            guard Int(n) <= capacity else { throw NetError("zmq receive: message part of \(n) bytes exceeds \(capacity)") }
            parts.append(Array(buffer.prefix(Int(n))))
            if more == 0 { return parts }
            wait = -1   // the rest of a multipart message is already there
        }
    }

    public static var version: (Int, Int, Int) {
        let v = Int(rtnet_version())
        return (v / 10000, v / 100 % 100, v % 100)
    }
}
