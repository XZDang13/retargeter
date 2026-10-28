// `retarget validate`: validate robot trajectories (ours or another tool's) against their source BVH clips,
// with the same metrics and rules as the retarget pipeline. Writes one JSON report per line.

import Compute
import Core
import MotionIO
import Retarget
import Validate

#if canImport(Glibc)
import Glibc
#endif

struct ValidateCommand {
    var inputs: [String] = []
    var robot = "g1_29dof"
    var humanRoot = ""
    var report = "validation.jsonl"
    var fps = 50.0
    var tracking = false
    var label = ""
    var rejectConfig: String?
    var verbose = false

    static let usage = """
    usage: retarget validate <robot motion files or directories (.npz / .csv)>... --human DIR [options]
      --human DIR        root of the source BVH clips; files are matched by relative path and stem
      --robot NAME       g1_29dof (default) | g1_23dof
      --report FILE      JSON lines output (default validation.jsonl)
      --fps N            resample before checking (default 50, as exported), so rates match across tools
      --tracking         also compare against our mapping's goals (only meaningful for our own outputs)
      --label NAME       label stored in each report row (e.g. "ours", "nvidia")
      --reject-config F  reject rules (default Configs/reject/g1.json)
      --verbose          print per-target tracking, per-link penetration, per-pair self-collision and per-joint torque details
    """

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
            case "--human": humanRoot = try value()
            case "--report": report = try value()
            case "--fps": fps = Double(try value()) ?? fps
            case "--tracking": tracking = true
            case "--label": label = try value()
            case "--reject-config": rejectConfig = try value()
            case "--verbose": verbose = true
            case let flag where flag.hasPrefix("--"): throw CLIError("unknown option \(flag)")
            default: inputs.append(arguments[i])
            }
            i += 1
        }
        guard !inputs.isEmpty, !humanRoot.isEmpty else { throw CLIError("need robot motion inputs and --human DIR") }
    }

    func run(repository: String) throws {
        guard let robotDirectory = [("g1_29dof", "g1_29_dof"), ("g1_23dof", "g1_23_dof")].first(where: { $0.0 == robot })?.1 else {
            throw CLIError("unknown robot \(robot)")
        }
        let directory = repository + "/assets/robots/unitree_g1/" + robotDirectory
        let model = try RobotUSDLoader.load(directory: directory, name: robot)
        let validator = try Validator.load(repository: repository, robot: model, robotDirectory: directory, rejectConfig: rejectConfig)
        let mapping = try RetargetMapping.load(file: repository + "/Configs/mappings/soma77_\(robot).json")
        let surface = try BodySurface.load(file: repository + "/assets/body_models/soma/surface_lo.npz")
        let device = try ComputeDevice.makeDefault(backend: .cuda)
        var retargeter: HumanRetargeter?

        var files: [(String, String)] = []
        func visit(_ path: String, _ relative: String) {
            var info = stat()
            guard stat(path, &info) == 0 else { return }
            if (info.st_mode & S_IFMT) == S_IFDIR {
                for e in listDirectory(path) { visit(path + "/" + e.name, relative.isEmpty ? e.name : relative + "/" + e.name) }
            } else if path.hasSuffix(".npz") || path.hasSuffix(".csv") {
                files.append((path, relative.isEmpty ? String(path.split(separator: "/").last!) : relative))
            }
        }
        for input in inputs { visit(input, "") }
        guard let out = fopen(report, "w") else { throw CLIError("cannot write \(report)") }
        defer { fclose(out) }

        var accepted = 0, rejected = 0, missing = 0
        // Human stage (device work) per clip in order; the CPU-heavy robot stage of each batch in parallel.
        struct Pending: @unchecked Sendable {
            var path: String, stem: String, trajectory: RobotTrajectory, goals: [[Transform]]?, clip: PreparedClip
            var report: ValidationReport
        }
        var pending: [Pending] = []
        func flush() {
            guard let setup = retargeter?.setup else { return }
            let batch = pending
            pending.removeAll()
            let results = parallelMap(batch.count) { i -> ValidationReport in
                var result = batch[i].report
                validator.checkRobot(batch[i].trajectory, goals: batch[i].goals, setup: setup,
                                     contacts: batch[i].clip.offline, into: &result)
                return result
            }
            for (item, result) in zip(batch, results) {
                if result.status == .accepted { accepted += 1 } else { rejected += 1 }
                let row = JSON.object([("label", .string(label)), ("file", .string(item.path)), ("clip", .string(item.stem)),
                                       ("report", result.json)])
                fputs(row.text + "\n", out)
                print("validate: \(item.stem): \(result.status.rawValue)\(result.reasons.isEmpty ? "" : " — " + result.reasons.joined(separator: "; "))")
                if verbose {
                    for category in ["tracking:", "ground_penetration:", "self_collision:", "torque:", "joint_acc_ratio:"] {
                        let top = result.details.filter { $0.key.hasPrefix(category) }.sorted { $0.value > $1.value }.prefix(4)
                        if !top.isEmpty { print("    " + top.map { "\($0.key) \(format($0.value))" }.joined(separator: ", ")) }
                    }
                }
            }
        }
        for (path, relative) in files {
            // Our outputs live under accepted/ or rejected/; the human clip path is what follows.
            var stem = String(relative.dropLast(4))
            for prefix in ["accepted/", "rejected/"] where stem.hasPrefix(prefix) { stem = String(stem.dropFirst(prefix.count)) }
            let bvh = humanRoot + "/" + stem + ".bvh"
            guard access(bvh, F_OK) == 0 else {
                missing += 1
                print("validate: no source clip for \(relative) (\(bvh))")
                continue
            }
            let imported = path.hasSuffix(".csv") ? try RobotMotionImport.nvidiaCSV(file: path) : try RobotMotionImport.npz(file: path)
            // Reorder joints into the model's order.
            let order = try model.dofNames.map { name -> Int in
                guard let i = imported.jointNames.firstIndex(of: name) else { throw CLIError("\(path): missing joint \(name)") }
                return i
            }
            let trajectory = RobotTrajectory(fps: imported.fps, root: imported.root,
                                             q: imported.q.map { q in order.map { q[$0] } },
                                             cost: Array(repeating: 0, count: imported.root.count)).resampled(to: fps)
            let motion = try BVH.load(file: bvh)
            if retargeter == nil {
                retargeter = try HumanRetargeter(device: device, robot: model, mapping: mapping, surface: surface,
                                                 skeleton: motion.skeleton)
            }
            let clip = try retargeter!.prepare([motion])[0]
            let result = validator.checkHuman(clip, setup: retargeter!.setup)
            let goals = tracking ? (0..<trajectory.frameCount).map { i in
                clip.goals[min(clip.goals.count - 1, Int((Double(i) / trajectory.fps * clip.fps).rounded()))]
            } : nil
            pending.append(Pending(path: path, stem: stem, trajectory: trajectory, goals: goals, clip: clip, report: result))
            if pending.count >= 16 { flush() }
        }
        flush()
        print("validate: \(accepted) accepted, \(rejected) rejected, \(missing) without source clip -> \(report)")
    }
}
