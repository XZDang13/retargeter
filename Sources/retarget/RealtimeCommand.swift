// `retarget realtime`: batch-retarget BVH files with the causal (lag = 0) solver and write .npz files.

import Compute
import Core
import MotionIO
import Retarget
import Validate

#if canImport(Glibc)
import Glibc
#endif

struct RealtimeCommand {
    /// "realtime" (causal, lag 0) or "offline" (whole-trajectory).
    var mode = "realtime"
    var inputs: [String] = []
    var robot = "g1_29dof"
    var output = "out"
    var fps = 50.0
    var batch = 64
    var backend = Backend.cuda
    var withBodies = false
    var overwrite = false
    var autoReject = true
    var dropRejected = false
    var rejectConfig: String?
    var include: [String] = []
    var exclude: [String] = []
    var limit: Int?
    var lag: Double?
    var lagSweeps = 5
    var lagRate = 50.0

    static let usage = """
    usage: retarget realtime|offline <bvh files or directories>... [options]
      realtime: causal solver (lag 0, what teleop sees); offline: whole-trajectory solver with stance-foot
      locking, ground contact / non-penetration and self-collision terms, warm-started from realtime.
      --robot NAME        g1_29dof (default) | g1_23dof
      --out DIR           output directory (default ./out); mirrors the input layout
      --fps N             output frame rate (default 50)
      --batch N           clips solved per device dispatch (default 64)
      --backend B         cuda (default) | cpu
      --with-bodies       also store link poses (body_pos / body_quat)
      --include GLOB      only files whose name matches (repeatable)
      --exclude GLOB      skip files whose name matches, e.g. '*_M.bvh' (repeatable)
      --limit N           process at most N clips
      --overwrite         redo clips whose output already exists (default: skip, so runs can resume)
      --no-auto-reject    validate and record status, but write every clip to DIR (no accepted/ rejected/ split)
      --drop-rejected     do not write rejected clips (human-stage rejects are then not solved at all)
      --reject-config F   reject rules (default Configs/reject/g1.json)
      --lag S             realtime only: run the clips through the teleop fixed-lag solver as live streams
                          (frames emitted S seconds late, refined meanwhile with the offline costs)
      --lag-sweeps N      sweeps of the fixed-lag window per window solve (default 5)
      --lag-rate HZ       window solves per second of input (default 50, the policy rate; 0: every frame)
    """

    init(mode: String, arguments: [String]) throws {
        self.mode = mode
        var i = 0
        func value() throws -> String {
            i += 1
            guard i < arguments.count else { throw CLIError("missing value for \(arguments[i - 1])") }
            return arguments[i]
        }
        while i < arguments.count {
            switch arguments[i] {
            case "--robot": robot = try value()
            case "--out": output = try value()
            case "--fps": fps = Double(try value()) ?? fps
            case "--batch": batch = max(1, Int(try value()) ?? batch)
            case "--backend": backend = try value() == "cpu" ? .cpu : .cuda
            case "--with-bodies": withBodies = true
            case "--overwrite": overwrite = true
            case "--include": include.append(try value())
            case "--exclude": exclude.append(try value())
            case "--limit": limit = Int(try value())
            case "--no-auto-reject": autoReject = false
            case "--drop-rejected": dropRejected = true
            case "--reject-config": rejectConfig = try value()
            case "--lag": lag = Double(try value())
            case "--lag-sweeps": lagSweeps = max(1, Int(try value()) ?? lagSweeps)
            case "--lag-rate": lagRate = max(0, Double(try value()) ?? lagRate)
            case let flag where flag.hasPrefix("--"): throw CLIError("unknown option \(flag)")
            default: inputs.append(arguments[i])
            }
            i += 1
        }
        guard !inputs.isEmpty else { throw CLIError("no input files") }
    }

    /// (absolute path, path relative to its input root) for every selected BVH file.
    func collectFiles() -> [(path: String, relative: String)] {
        var files: [(String, String)] = []
        func matches(_ name: String, _ patterns: [String]) -> Bool { patterns.contains { fnmatch($0, name, 0) == 0 } }
        func visit(_ path: String, relative: String) {
            var info = stat()
            guard stat(path, &info) == 0 else { return }
            if (info.st_mode & S_IFMT) == S_IFDIR {
                for entry in listDirectory(path) {
                    visit(path + "/" + entry.name, relative: relative.isEmpty ? entry.name : relative + "/" + entry.name)
                }
            } else if path.hasSuffix(".bvh") {
                let name = String(path.split(separator: "/").last!)
                if (!include.isEmpty && !matches(name, include)) || matches(name, exclude) { return }
                files.append((path, relative.isEmpty ? name : relative))
            }
        }
        for input in inputs { visit(input, relative: "") }
        return limit.map { Array(files.prefix($0)) } ?? files
    }

    func run(repository: String) throws {
        let robotDirectory = [("g1_29dof", "g1_29_dof"), ("g1_23dof", "g1_23_dof")].first { $0.0 == robot }?.1
        guard let robotDirectory else { throw CLIError("unknown robot \(robot)") }
        let directory = repository + "/assets/robots/unitree_g1/" + robotDirectory
        let model = try RobotUSDLoader.load(directory: directory, name: robot)
        let validator = try Validator.load(repository: repository, robot: model, robotDirectory: directory,
                                           rejectConfig: rejectConfig)
        let mappingFile = repository + "/Configs/mappings/soma77_\(robot).json"
        let mapping = try RetargetMapping.load(file: mappingFile)
        let configHash = hash(readFile(mappingFile) ?? []) + "-" + RobotMotionExport.version
        let surface = try BodySurface.load(file: repository + "/assets/body_models/soma/surface_lo.npz")
        let device = try ComputeDevice.makeDefault(backend: backend)

        let files = collectFiles()
        let todo = files.filter { file in
            overwrite || !["", "accepted/", "rejected/"].contains { access(output + "/" + $0 + stem(file.relative) + ".npz", F_OK) == 0 }
        }
        print("\(mode): \(files.count) clips selected, \(files.count - todo.count) already done, robot \(robot), \(device.name)")
        guard !todo.isEmpty else { return }
        mkdirs(output)
        let manifest = fopen(output + "/manifest.jsonl", "a")
        defer { if let manifest { fclose(manifest) } }

        let geometry = try RobotGeometry(robot: model, pointsFile: directory + "/mesh_points.npz")
        let start = ContinuousClock.now
        var done = 0, failed = 0, frames = 0, rejected = 0, ticks = 0
        var timing: (prepare: Duration, solve: Duration, sweeps: Duration, validate: Duration) = (.zero, .zero, .zero, .zero)
        func seconds(_ d: Duration) -> String { format(Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18) }
        let chunks = stride(from: 0, to: todo.count, by: batch).map { Array(todo[$0..<min($0 + batch, todo.count)]) }

        // Load each clip on its own so one bad file never takes down the batch.
        typealias Loaded = [(file: (path: String, relative: String), motion: HumanMotion)]
        func load(_ chunk: [(path: String, relative: String)]) -> (loaded: Loaded, failed: Int) {
            var loaded: Loaded = [], failures = 0
            for file in chunk {
                do { loaded.append((file, try BVH.load(file: file.path))) } catch {
                    failures += 1
                    log(manifest, file.path, status: "error", detail: "\(error)")
                }
            }
            return (loaded, failures)
        }
        // The retargeter is built from the first clip that loads (its skeleton).
        var next = 0
        var firstLoad: (loaded: Loaded, failed: Int)?
        while next < chunks.count {
            let attempt = load(chunks[next])
            next += 1
            if attempt.loaded.isEmpty { failed += attempt.failed; continue }
            firstLoad = attempt
            break
        }
        guard let firstLoad else {
            print("\(mode): 0/\(todo.count) clips (\(failed) failed)")
            return
        }
        let skeleton = firstLoad.loaded[0].motion.skeleton
        let offlineRetargeter = mode == "offline"
            ? try OfflineRetargeter(device: device, robot: model, mapping: mapping, surface: surface, skeleton: skeleton, geometry: geometry)
            : nil
        let laggedRetargeter = mode == "realtime" ? try lag.map {
            try LaggedRetargeter(device: device, robot: model, mapping: mapping, surface: surface, skeleton: skeleton,
                                 geometry: geometry, lag: $0, sweeps: lagSweeps, tickRate: lagRate > 0 ? lagRate : nil)
        } : nil
        let retargeter = try offlineRetargeter?.base ?? laggedRetargeter?.base
            ?? HumanRetargeter(device: device, robot: model, mapping: mapping, surface: surface, skeleton: skeleton)
        let setup = retargeter.setup

        // A batch flows through three stages: load + prepare + human checks (CPU), solve (device), robot checks +
        // export (CPU). The CPU stages of the neighbouring batches run on their own threads while the device solves.
        struct Batch {
            var loaded: Loaded
            var failed: Int
            var prepared: [PreparedClip]
            var reports: [ValidationReport]
            var solveIndices: [Int]
            var duration: Duration
        }
        let dropRejected = self.dropRejected
        func prepare(_ load: (loaded: Loaded, failed: Int)) throws -> Batch {
            let t0 = ContinuousClock.now
            let motions = load.loaded.map(\.motion)
            let prepared = try offlineRetargeter?.prepare(motions) ?? laggedRetargeter?.prepare(motions) ?? retargeter.prepare(motions)
            let reports = prepared.map { validator.checkHuman($0, setup: setup) }
            // Human-stage rejects that will not be written need no solve.
            let solveIndices = prepared.indices.filter { !(dropRejected && reports[$0].status == .rejected) }
            return Batch(loaded: load.loaded, failed: load.failed, prepared: prepared, reports: reports,
                         solveIndices: solveIndices, duration: ContinuousClock.now - t0)
        }
        struct Outcome { var done = 0, failed = 0, frames = 0, rejected = 0; var duration = Duration.zero }
        func finish(_ batch: Batch, _ results: [RetargetResult]) -> Outcome {
            let t0 = ContinuousClock.now
            var outcome = Outcome()
            var reports = batch.reports
            // Robot stage of every solved clip in parallel.
            let robotReports = parallelMap(batch.solveIndices.count) { [reports] k -> ValidationReport in
                var report = reports[batch.solveIndices[k]]
                validator.checkRobot(results[k], setup: setup, fps: fps, into: &report)
                return report
            }
            for (k, i) in batch.solveIndices.enumerated() {
                let item = batch.loaded[i], result = results[k]
                reports[i] = robotReports[k]
                let report = reports[i]
                if report.status == .rejected { outcome.rejected += 1 }
                if dropRejected && report.status == .rejected {
                    log(manifest, item.file.path, report: report, frames: item.motion.frameCount)
                    continue
                }
                let folder = autoReject ? (report.status == .accepted ? "accepted/" : "rejected/") : ""
                let out = output + "/" + folder + stem(item.file.relative) + ".npz"
                mkdirs(String(out[..<out.lastIndex(of: "/")!]))
                var options = ExportOptions()
                options.fps = fps
                options.withBodies = withBodies
                options.source = item.file.path
                options.configHash = configHash
                options.status = report.status.rawValue
                options.rejectReasons = report.reasons
                options.warnings = report.warnings
                options.metricsJSON = report.json.text
                options.mode = lag.map { "realtime_lag_\(format($0))" } ?? mode
                if let d = report.dynamics {
                    options.tau = d.tau
                    options.contactForce = d.contactForce
                    options.dynamicsResidual = d.residual
                    options.zmp = d.zmp
                }
                do {
                    try RobotMotionExport.write(result, robot: model, to: out, options: options)
                    outcome.done += 1
                    outcome.frames += item.motion.frameCount
                    log(manifest, item.file.path, report: report, output: out, frames: item.motion.frameCount)
                } catch {
                    outcome.failed += 1
                    log(manifest, item.file.path, status: "error", detail: "\(error)")
                }
            }
            for i in batch.prepared.indices where !batch.solveIndices.contains(i) {
                outcome.rejected += 1
                log(manifest, batch.loaded[i].file.path, report: reports[i], frames: batch.loaded[i].motion.frameCount)
            }
            outcome.duration = ContinuousClock.now - t0
            return outcome
        }
        func account(_ outcome: Outcome) {
            done += outcome.done
            failed += outcome.failed
            frames += outcome.frames
            rejected += outcome.rejected
            timing.validate += outcome.duration
            let elapsed = ContinuousClock.now - start
            print("\(mode): \(done + failed)/\(todo.count) clips (\(rejected) rejected, \(failed) failed), \(frames) frames in \(elapsed)")
            print("\(mode): prepare \(seconds(timing.prepare)) s, solve \(seconds(timing.solve)) s"
                  + (mode == "offline" ? " (offline sweeps \(seconds(timing.sweeps)) s)" : "")
                  + (lag != nil ? " (fixed-lag windows \(seconds(timing.sweeps)) s, \(format(1000 * (Double(timing.sweeps.components.seconds) + Double(timing.sweeps.components.attoseconds) * 1e-18) / Double(max(1, ticks)))) ms per tick)" : "")
                  + ", validate + export \(seconds(timing.validate)) s (stages overlap across batches)")
        }

        var preparing = Background { try prepare(firstLoad) }
        var finishing: Background<Outcome>?
        while true {
            let batch = try preparing.wait()
            failed += batch.failed
            timing.prepare += batch.duration
            let more = next < chunks.count
            if more {
                let chunk = chunks[next]
                next += 1
                preparing = Background { try prepare(load(chunk)) }
            }
            let t0 = ContinuousClock.now
            let toSolve = batch.solveIndices.map { batch.prepared[$0] }
            let results = try offlineRetargeter?.solve(toSolve) ?? laggedRetargeter?.solve(toSolve) ?? retargeter.solve(toSolve)
            timing.solve += ContinuousClock.now - t0
            if let offline = offlineRetargeter { timing.sweeps += offline.solver.lastDuration }
            if let lagged = laggedRetargeter {
                timing.sweeps += lagged.lastWindowTime
                ticks += lagged.lastTicks
            }
            if let previous = finishing { account(try previous.wait()) }
            finishing = Background { finish(batch, results) }
            if !more { break }
        }
        if let last = finishing { account(try last.wait()) }
    }

    func stem(_ relative: String) -> String { String(relative.dropLast(4)) }

    /// Manifest row for a validated clip.
    func log(_ manifest: UnsafeMutablePointer<FILE>?, _ source: String, report: ValidationReport, output: String? = nil,
             frames: Int) {
        guard let manifest else { return }
        let row = JSON.object([
            ("source", .string(source)), ("status", .string(report.status.rawValue)),
            ("stage", report.stage.map { .string($0) } ?? .null), ("robot", .string(robot)), ("mode", .string(mode)),
            ("output", output.map { .string($0) } ?? .null), ("frames", .number(Double(frames))),
            ("reasons", .array(report.reasons.map { .string($0) })), ("warnings", .array(report.warnings.map { .string($0) })),
            ("metrics", .object(report.metrics.map { ($0.0, .number($0.1)) })),
        ])
        fputs(row.text + "\n", manifest)
        fflush(manifest)
    }

    func log(_ manifest: UnsafeMutablePointer<FILE>?, _ source: String, status: String, detail: String?, output: String? = nil,
             frames: Int? = nil, flags: [String] = [], meanCost: Double? = nil) {
        guard let manifest else { return }
        func quoted(_ s: String) -> String {
            "\"" + s.unicodeScalars.map { $0 == "\"" ? "\\\"" : $0 == "\\" ? "\\\\" : $0 == "\n" ? "\\n" : String($0) }.joined() + "\""
        }
        var fields = ["\"source\":\(quoted(source))", "\"status\":\(quoted(status))", "\"robot\":\(quoted(robot))",
                      "\"mode\":\"realtime\""]
        if let output { fields.append("\"output\":\(quoted(output))") }
        if let frames { fields.append("\"frames\":\(frames)") }
        fields.append("\"human_flags\":[" + flags.map(quoted).joined(separator: ",") + "]")
        if let meanCost { fields.append("\"mean_tracking_cost\":\(meanCost)") }
        if let detail { fields.append("\"error\":\(quoted(detail))") }
        fputs("{" + fields.joined(separator: ",") + "}\n", manifest)
        fflush(manifest)
    }
}

struct CLIError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

func mkdirs(_ path: String) {
    var current = path.hasPrefix("/") ? "" : "."
    for part in path.split(separator: "/") {
        current += "/" + part
        mkdir(current, 0o755)
    }
}

func hash(_ bytes: [UInt8]) -> String {
    var h: UInt64 = 1_469_598_103_934_665_603
    for b in bytes { h = (h ^ UInt64(b)) &* 1_099_511_628_211 }
    return String(h, radix: 16)
}
