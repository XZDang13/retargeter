// `retarget teleop`: live retargeting server (DESIGN §7, §10). Clients connect a ZeroMQ DEALER socket to the
// server's ROUTER, open streams (StreamStart -> StreamInfo), send HumanFrames and receive one RobotFrame per
// HumanFrame on the same socket. Frames of all streams that are waiting are solved together, one frame per stream
// per tick, in arrival order; a stream that falls further behind than --max-backlog drops its oldest frames
// (flag 64). Lag 0 (causal): each frame is solved as soon as it arrives.

import Compute
import Contact
import Core
import MotionIO
import Net
import Retarget
import Validate
import Viewer

#if canImport(Glibc)
import Glibc
#endif

/// Monotonic clock (s); CLOCK_MONOTONIC is shared by the processes of one machine (Python: time.monotonic()).
/// A server's log is read while it runs: every line is flushed.
func log(_ line: String) {
    print(line)
    fflush(nil)
}

func monotonicSeconds() -> Double {
    var t = timespec()
    clock_gettime(CLOCK_MONOTONIC, &t)
    return Double(t.tv_sec) + Double(t.tv_nsec) * 1e-9
}

struct TeleopCommand {
    var robot = "g1_29dof"
    var endpoint = "tcp://127.0.0.1:5560"
    var slots = 8
    var maxBacklog = 0.25
    var viewerPort: UInt16? = nil
    var quiet = false
    var lag = 0.0
    var lagSweeps = 5
    var lagRate = 50.0

    static let usage = """
    usage: retarget teleop [options]
      Live retargeting server. Clients (python/retargeter_client) connect a ZeroMQ DEALER socket, open streams
      with StreamStart and send HumanFrames; each one is answered with a RobotFrame (with --lag, that much later).
      --robot NAME        g1_29dof (default) | g1_23dof
      --bind ENDPOINT     ZeroMQ ROUTER endpoint (default tcp://127.0.0.1:5560; tcp://*:5560 for other hosts)
      --slots N           concurrent streams (default 8)
      --max-backlog S     drop a stream's oldest waiting frames beyond S seconds of input (default 0.25)
      --lag S             fixed lag: each frame is refined with the offline costs (stance lock, ground, collisions,
                          dynamics) for S seconds of newer input before it is sent (default 0: causal)
      --lag-sweeps N      Gauss-Seidel sweeps over the window per window solve (default 5)
      --lag-rate HZ       window solves per second of input (default 50, the policy rate; 0: every frame); frames
                          received in between are answered together at the next solve
      --viewer-port N     also serve the live viewer on http://127.0.0.1:N
      --quiet             no periodic statistics
    """

    static let framesDropped: UInt32 = 64

    init(arguments: [String]) throws {
        var i = 0
        func value() throws -> String {
            i += 1
            guard i < arguments.count else { throw CLIError("missing value for \(arguments[i - 1])") }
            return arguments[i]
        }
        while i < arguments.count {
            switch arguments[i] {
            case "--robot": robot = try value()
            case "--bind": endpoint = try value()
            case "--slots": slots = max(1, Int(try value()) ?? slots)
            case "--max-backlog": maxBacklog = Double(try value()) ?? maxBacklog
            case "--viewer-port": viewerPort = UInt16(try value())
            case "--quiet": quiet = true
            case "--lag": lag = max(0, Double(try value()) ?? lag)
            case "--lag-sweeps": lagSweeps = max(1, Int(try value()) ?? lagSweeps)
            case "--lag-rate": lagRate = max(0, Double(try value()) ?? lagRate)
            default: throw CLIError("unknown option \(arguments[i])")
            }
            i += 1
        }
    }

    /// An open stream: the client that owns it, its id there, and its waiting frames.
    struct Stream {
        var client: [UInt8]
        var id: UInt32
        var fps: Double
        var pending: [(frame: HumanFrame, received: Double)] = []
        /// Frames handed to the solver and not yet answered (fixed lag), oldest first.
        var inflight: [(frame: HumanFrame, received: Double)] = []
        var dropped = false
        var seqLast: UInt64? = nil
    }

    func run(repository: String) throws {
        let robotDirectory = [("g1_29dof", "g1_29_dof"), ("g1_23dof", "g1_23_dof")].first { $0.0 == robot }?.1
        guard let robotDirectory else { throw CLIError("unknown robot \(robot)") }
        let directory = repository + "/assets/robots/unitree_g1/" + robotDirectory
        let model = try RobotUSDLoader.load(directory: directory, name: robot)
        let validator = try Validator.load(repository: repository, robot: model, robotDirectory: directory, rejectConfig: nil)
        let mapping = try RetargetMapping.load(file: repository + "/Configs/mappings/soma77_\(robot).json")
        let surface = try BodySurface.load(file: repository + "/assets/body_models/soma/surface_lo.npz")
        let device = try ComputeDevice.makeDefault(backend: .cuda)
        let groups = try ContactGroups.soma(surface).names

        let socket = try ZMQSocket(.router)
        try socket.set(33, 1)        // ZMQ_ROUTER_MANDATORY: report unroutable replies instead of dropping silently
        try socket.set(24, 100_000)  // ZMQ_RCVHWM
        try socket.set(23, 100_000)  // ZMQ_SNDHWM
        try socket.bind(endpoint)
        let viewer = try viewerPort.map { try LiveViewerHub(repository: repository, port: $0, robot: model) }
        let (major, minor, patch) = ZMQSocket.version
        log("teleop: robot \(robot), \(slots) slots, lag \(format(lag)) s\(lag > 0 ? " (\(lagSweeps) sweeps per window solve, \(lagRate > 0 ? format(lagRate) + " solves/s" : "every frame"))" : ""), \(device.name), ZeroMQ \(major).\(minor).\(patch), listening on \(endpoint)"
              + (viewerPort.map { ", live viewer http://127.0.0.1:\($0)" } ?? ""))

        // The retargeter is built with the first stream's skeleton (all streams must share its joints).
        var live: LiveRetargeter?
        var streams = [Stream?](repeating: nil, count: slots)
        var statistics = (frames: 0, ticks: 0, solve: 0.0, latency: 0.0, since: monotonicSeconds())

        func reply(_ client: [UInt8], _ message: [UInt8]) {
            do { try socket.send([client, message], nonblocking: true) } catch {
                log("teleop: cannot reach a client (\(error)); closing its streams")
                for s in streams.indices where streams[s]?.client == client {
                    streams[s] = nil
                    live?.close(s)
                }
            }
        }
        func info(_ client: [UInt8], _ id: UInt32, error: String? = nil) {
            reply(client, StreamInfo(streamId: id, status: error == nil ? 0 : 1, message: error ?? "", robot: robot,
                                     dofNames: model.dofNames, contactGroups: groups, lag: lag).encode())
        }
        func slot(of client: [UInt8], _ id: UInt32) -> Int? {
            streams.firstIndex { $0?.client == client && $0?.id == id }
        }

        // Solves until no stream has waiting frames: one frame per stream per tick.
        func flush() throws {
            guard let live else { return }
            while true {
                var inputs: [(slot: Int, local: [Transform])] = []
                for s in streams.indices {
                    guard var stream = streams[s], !stream.pending.isEmpty else { continue }
                    let backlog = Int((maxBacklog * stream.fps).rounded(.up))
                    if stream.pending.count > max(1, backlog) {
                        stream.pending.removeFirst(stream.pending.count - max(1, backlog))
                        stream.dropped = true
                    }
                    // Lag 0: one frame per stream per step (each answered at once). Fixed lag: every waiting frame
                    // (answers leave at window solves anyway), so their realtime solves share one dispatch.
                    let take = live.lag > 0 ? stream.pending.count : 1
                    let next = stream.pending.prefix(take)
                    stream.pending.removeFirst(take)
                    streams[s] = stream
                    for item in next {
                        guard let local = item.frame.localTransforms(skeleton: live.retargeter.setup.skeleton) else {
                            info(stream.client, stream.id, error: "frame \(item.frame.seq): joint arrays do not match the stream skeleton")
                            continue
                        }
                        inputs.append((s, local))
                        streams[s]!.inflight.append((item.frame, item.received))
                    }
                }
                guard !inputs.isEmpty else { return }
                let t0 = monotonicSeconds()
                let solved = try live.step(inputs)
                let t1 = monotonicSeconds()
                statistics.ticks += 1
                statistics.solve += t1 - t0
                send(solved)
            }
        }

        // Answers solved frames: each is the oldest in-flight frame of its stream.
        func send(_ solved: [LiveRobotFrame]) {
            for out in solved {
                guard var stream = streams[out.slot], !stream.inflight.isEmpty else { continue }
                let m = stream.inflight.removeFirst()
                let flags = out.flags | (stream.dropped ? Self.framesDropped : 0)
                stream.dropped = false
                streams[out.slot] = stream
                let r = out.root
                let frame = RobotFrame(
                    streamId: stream.id, seq: m.frame.seq, tSource: m.frame.tSource, tReceive: m.received,
                    tEmit: monotonicSeconds(),
                    rootPos: [Float(r.translation.x), Float(r.translation.y), Float(r.translation.z)],
                    rootQuat: [Float(r.rotation.w), Float(r.rotation.v.x), Float(r.rotation.v.y), Float(r.rotation.v.z)],
                    dofPos: out.q.map(Float.init), dofVel: out.qVelocity.map(Float.init),
                    contact: out.contacts.contact.map { $0 ? 1 : 0 }, groundHeight: out.contacts.ground,
                    cost: Float(out.cost), flags: flags)
                reply(stream.client, frame.encode())
                viewer?.publish(slot: out.slot, human: m.frame, robot: frame)
                statistics.frames += 1
                statistics.latency += frame.tEmit - m.received
            }
        }

        // Ends a stream: the frames still in its fixed-lag window are answered first.
        func finishStream(_ s: Int) throws {
            if let live { send(try live.flush(s)) }
            streams[s] = nil
            live?.close(s)
            viewer?.close(slot: s)
        }

        while true {
            if !quiet && monotonicSeconds() - statistics.since >= 5 {
                let dt = monotonicSeconds() - statistics.since
                if statistics.frames > 0 {
                    log("teleop: \(streams.compactMap { $0 }.count) streams, \(format(Double(statistics.frames) / dt)) frames/s, "
                          + "solve \(format(1000 * statistics.solve / Double(max(1, statistics.ticks)))) ms/tick, "
                          + "receive -> emit \(format(1000 * statistics.latency / Double(statistics.frames))) ms")
                }
                statistics = (0, 0, 0, 0, monotonicSeconds())
            }
            guard var parts = try socket.receive(timeout: 1000) else { continue }
            // Drain everything that is waiting, then solve; control messages flush the frames before them.
            while true {
                let received = monotonicSeconds()
                if parts.count == 2 {
                    let client = parts[0]
                    do {
                        switch try StreamMessage.decode(parts[1]) {
                        case .streamStart(let start):
                            try flush()
                            guard let skeleton = start.humanSkeleton else {
                                info(client, start.streamId, error: "StreamStart: joint, parent and offset counts differ")
                                break
                            }
                            if live == nil {
                                let t0 = monotonicSeconds()
                                let retargeter = try HumanRetargeter(device: device, robot: model, mapping: mapping,
                                                                     surface: surface, skeleton: skeleton)
                                let fixedLag = lag > 0 ? FixedLagSolver(
                                    offline: try OfflineSolver(device: device, setup: retargeter.setup,
                                                               geometry: try RobotGeometry(robot: model, pointsFile: directory + "/mesh_points.npz")),
                                    slots: slots, lag: lag, sweeps: lagSweeps, tickRate: lagRate > 0 ? lagRate : nil) : nil
                                live = try LiveRetargeter(retargeter: retargeter, slots: slots,
                                                          velocityLimits: validator.velocityLimits, fixedLag: fixedLag)
                                log("teleop: retargeter ready for skeleton \(skeleton.name) in \(format(monotonicSeconds() - t0)) s")
                            }
                            guard let s = slot(of: client, start.streamId) ?? streams.firstIndex(where: { $0 == nil }) else {
                                info(client, start.streamId, error: "all \(slots) stream slots are in use")
                                break
                            }
                            if streams[s] != nil { try finishStream(s) }   // a restart answers the old stream's last frames
                            do {
                                try live!.open(s, skeleton: skeleton, fps: start.fps)
                                streams[s] = Stream(client: client, id: start.streamId, fps: start.fps)
                                viewer?.open(slot: s, skeleton: skeleton)
                                info(client, start.streamId)
                                log("teleop: stream \(start.streamId) opened on slot \(s) (\(format(start.fps)) Hz)")
                            } catch {
                                info(client, start.streamId, error: "\(error)")
                            }
                        case .humanFrame(let frame):
                            if let s = slot(of: client, frame.streamId) {
                                streams[s]!.pending.append((frame, received))
                            } else {
                                info(client, frame.streamId, error: "stream \(frame.streamId) is not open")
                            }
                        case .streamEnd(let end):
                            try flush()
                            if let s = slot(of: client, end.streamId) {
                                try finishStream(s)
                                log("teleop: stream \(end.streamId) closed")
                            }
                        default:
                            info(client, 0, error: "unexpected message from a client")
                        }
                    } catch {
                        info(client, 0, error: "\(error)")
                    }
                }
                guard let next = try socket.receive(timeout: 0) else { break }
                parts = next
            }
            try flush()
        }
    }
}
