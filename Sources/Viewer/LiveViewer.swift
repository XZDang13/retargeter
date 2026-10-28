// Live teleop viewer: `retarget teleop --viewer-port N` serves Web/live.html and pushes every stream's robot
// pose (link transforms) and human skeleton over a WebSocket (/ws), at most `maxRate` frames per second per stream.
// Messages: JSON text {"type": "streams", ...} when streams open or close; binary frames of float32:
//   [slot, seq, receive->emit ms, cost, flags, ground, links L, joints J, L x (pos 3, quat wxyz 4), J x pos 3].

import Core
import Dispatch
import MotionIO

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public final class LiveViewerHub: @unchecked Sendable {
    let robot: RobotModel
    let maxRate: Double
    let lock = Lock()
    var sockets: [Int32] = []
    var skeletons: [Int: HumanSkeleton] = [:]
    var lastSent: [Int: Double] = [:]

    public init(repository: String, port: UInt16, robot: RobotModel, maxRate: Double = 60) throws {
        self.robot = robot
        self.maxRate = maxRate
        let description = robotJSON(robot).text
        let server = HTTPServer(port: port, handler: { request in
            switch request.path {
            case "/", "/index.html": return Self.file(repository + "/Web/live.html")
            case "/api/robot": return .json(description)
            default:
                if request.path.hasPrefix("/web/") { return Self.file(repository + "/Web/" + request.path.dropFirst(5)) }
                if request.path.hasPrefix("/robots/") { return Self.file(repository + "/build/viewer/robots/" + request.path.dropFirst(8)) }
                return .notFound
            }
        }, upgrade: { [unowned self] request, fd in
            guard request.path == "/ws" else { Glibc.close(fd); return }
            var timeout = timeval(tv_sec: 0, tv_usec: 20_000)   // a stalled browser cannot hold up the server
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            self.lock.withLock {
                self.sockets.append(fd)
                self.send(fd, WebSocket.frame(Array(self.streamsMessage().utf8), opcode: 1))
            }
        })
        DispatchQueue.global().async {
            do { try server.run() } catch { print("live viewer: \(error)") }
        }
    }

    static func file(_ path: String) -> HTTPResponse {
        guard let fd = fopen(path, "rb") else { return .notFound }
        defer { fclose(fd) }
        var bytes: [UInt8] = []
        var chunk = [UInt8](repeating: 0, count: 65536)
        while true {
            let n = fread(&chunk, 1, chunk.count, fd)
            if n <= 0 { break }
            bytes += chunk[0..<n]
        }
        let type = path.hasSuffix(".html") ? "text/html; charset=utf-8" : path.hasSuffix(".js") ? "text/javascript"
            : path.hasSuffix(".css") ? "text/css" : path.hasSuffix(".glb") ? "model/gltf-binary" : "application/octet-stream"
        return HTTPResponse(contentType: type, body: bytes)
    }

    func streamsMessage() -> String {
        let slots = skeletons.keys.sorted().map { slot -> JSON in
            .object([("slot", .number(Double(slot))),
                     ("parents", .array(skeletons[slot]!.parents.map { .number(Double($0 ?? -1)) }))])
        }
        return JSON.object([("type", .string("streams")), ("robot", .string(robot.name)), ("streams", .array(slots))]).text
    }

    /// Sends one frame, dropping the connection if it fails (callers hold the lock).
    func send(_ fd: Int32, _ bytes: [UInt8]) {
        var offset = 0
        while offset < bytes.count {
            let n = bytes[offset...].withUnsafeBytes { Glibc.send(fd, $0.baseAddress, $0.count, Int32(MSG_NOSIGNAL)) }
            if n <= 0 {
                Glibc.close(fd)
                sockets.removeAll { $0 == fd }
                return
            }
            offset += n
        }
    }

    func broadcast(_ bytes: [UInt8]) {
        for fd in sockets { send(fd, bytes) }
    }

    public func open(slot: Int, skeleton: HumanSkeleton) {
        lock.withLock {
            skeletons[slot] = skeleton
            lastSent[slot] = nil
            broadcast(WebSocket.frame(Array(streamsMessage().utf8), opcode: 1))
        }
    }

    public func close(slot: Int) {
        lock.withLock {
            skeletons[slot] = nil
            broadcast(WebSocket.frame(Array(streamsMessage().utf8), opcode: 1))
        }
    }

    public func publish(slot: Int, human: HumanFrame, robot frame: RobotFrame) {
        lock.withLock {
            guard !sockets.isEmpty, let skeleton = skeletons[slot] else { return }
            // 10 % slack: at 120 Hz input every second frame lands just either side of 1 / 60 s.
            if let last = lastSent[slot], frame.tEmit - last < 0.9 / maxRate { return }
            lastSent[slot] = frame.tEmit
            let root = Transform(rotation: Quat(w: Double(frame.rootQuat[0]), x: Double(frame.rootQuat[1]),
                                                y: Double(frame.rootQuat[2]), z: Double(frame.rootQuat[3])).normalized,
                                 translation: Vec3(Double(frame.rootPos[0]), Double(frame.rootPos[1]), Double(frame.rootPos[2])))
            let links = robot.forwardKinematics(root: root, q: frame.dofPos.map(Double.init))
            let joints = human.localTransforms(skeleton: skeleton)
                .map { HumanMotion(skeleton: skeleton, fps: 1, local: [$0]).globalTransforms(frame: 0) } ?? []
            var values: [Float] = [Float(slot), Float(frame.seq), Float(1000 * (frame.tEmit - frame.tReceive)), frame.cost,
                                   Float(frame.flags), frame.groundHeight, Float(links.count), Float(joints.count)]
            values.reserveCapacity(values.count + 7 * links.count + 3 * joints.count)
            for t in links {
                values += [Float(t.translation.x), Float(t.translation.y), Float(t.translation.z),
                           Float(t.rotation.w), Float(t.rotation.v.x), Float(t.rotation.v.y), Float(t.rotation.v.z)]
            }
            for t in joints { values += [Float(t.translation.x), Float(t.translation.y), Float(t.translation.z)] }
            let bytes = values.withUnsafeBytes { Array($0) }
            broadcast(WebSocket.frame(bytes, opcode: 2))
        }
    }
}
