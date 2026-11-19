// Routes for the browser viewer: static files, robot models + meshes, human motions and demo robot motions.

import Compute
import Contact
import Core
import Kinematics
import MotionIO
import Retarget
import Synchronization
import Validate

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public struct ViewerConfiguration: Sendable {
    public var repositoryRoot: String
    /// Root directory of human motion files (e.g. ~/Downloads/soma_uniform/bvh); requests cannot leave it.
    public var motionRoot: String
    public var port: UInt16
    /// Custom skeleton config (Configs/skeletons, DESIGN §16); nil: SOMA clips and the baked body surface.
    public var skeleton: SkeletonConfig?

    public init(repositoryRoot: String, motionRoot: String, port: UInt16, skeleton: SkeletonConfig? = nil) {
        self.repositoryRoot = repositoryRoot
        self.motionRoot = motionRoot
        self.port = port
        self.skeleton = skeleton
    }
}

/// One human skeleton: how its files are read, its body surface and contact groups, and (with a GPU) its skinning
/// and retargeters, built on first use. SOMA: one, from the baked surface; custom skeletons: one per distinct
/// skeleton, its reference pose and body built from the first clip of it that is opened.
final class SkeletonContext: @unchecked Sendable {
    let key: String
    /// Reads a clip of this skeleton.
    let read: @Sendable (String) throws -> HumanMotion
    /// nil: SOMA without its rig file (the body shows, retargeting is unavailable).
    let skeleton: HumanSkeleton?
    let surface: BodySurface
    let groups: ContactGroups
    // Under `ViewerApp.gpu`'s lock:
    var skinning: SurfaceSkinning?
    var retargeters: [String: HumanRetargeter] = [:]  // per robot
    var offline: [String: OfflineRetargeter] = [:]

    init(key: String, skeleton: HumanSkeleton?, surface: BodySurface, read: @escaping @Sendable (String) throws -> HumanMotion) throws {
        self.key = key
        self.read = read
        self.skeleton = skeleton
        self.surface = surface
        groups = try ContactGroups.of(surface)
    }
}

public final class ViewerApp: Sendable {
    let config: ViewerConfiguration
    let robots: [(name: String, model: RobotModel)]
    /// The SOMA body with the base rig (nil with a custom BVH skeleton config, or without the baked surface): the
    /// default body and the kernel warm-up; clips get a context per own skeleton (`context`).
    let soma: SkeletonContext?
    /// Custom BVH skeleton config: .bvh files are read with it instead of as SOMA.
    let bvhConfig: SkeletonConfig?
    /// Custom skeletons seen so far, by `SkeletonConfig.signature`.
    let contexts: Mutex<[String: SkeletonContext]>
    /// SMPL-family clips (.npz / .pkl): the --skeleton config when it is an SMPL one, else Configs/skeletons/smplx.json.
    let smpl: SMPLSource?
    /// Serialises uploads (choosing a free name, writing).
    let uploads = Mutex(())
    /// The last retargeted clip (see `solve`).
    let lastSolve = Mutex<Solved?>(nil)
    /// GPU skinning and retargeting; slang-rhi devices are not thread-safe, so requests take turns.
    let gpu: Mutex<GPUState?>

    final class GPUState {
        let device: ComputeDevice
        var validators: [String: Validator] = [:]
        init(device: ComputeDevice) { self.device = device }
    }

    public init(config: ViewerConfiguration) throws {
        self.config = config
        let g1 = config.repositoryRoot + "/assets/robots/unitree_g1"
        robots = try [("g1_29dof", "g1_29_dof"), ("g1_23dof", "g1_23_dof")].map {
            ($0.0, try RobotUSDLoader.load(directory: g1 + "/" + $0.1, name: $0.0))
        }
        contexts = Mutex([:])
        bvhConfig = config.skeleton?.format == "bvh" ? config.skeleton : nil
        let smplConfig = config.skeleton?.format == "smpl" ? config.skeleton
            : try? SkeletonConfig.load(file: config.repositoryRoot + "/Configs/skeletons/smplx.json")
        smpl = smplConfig.map { SMPLSource(config: $0, repository: config.repositoryRoot) }
        var soma: SkeletonContext?
        if bvhConfig == nil, let surface = try? BodySurface.load(file: config.repositoryRoot + "/assets/body_models/soma/surface_lo.npz") {
            let rig = config.repositoryRoot + "/assets/body_models/soma/seed_soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh"
            soma = try? SkeletonContext(key: "soma", skeleton: try? BVH.load(file: rig).skeleton, surface: surface) {
                try BVH.load(file: $0)
            }
        }
        self.soma = soma
        var state: GPUState?
        if let device = try? ComputeDevice.makeDefault(backend: .gpu) {
            state = GPUState(device: device)
            for (name, model) in robots {
                let directory = config.repositoryRoot + "/assets/robots/unitree_g1/" + (name == "g1_29dof" ? "g1_29_dof" : "g1_23_dof")
                state?.validators[name] = try? Validator.load(repository: config.repositoryRoot, robot: model, robotDirectory: directory)
            }
            // Retargeters for the SOMA base rig up front: their kernels compile once, and the per-clip skeletons
            // (`context`) reuse them.
            if let soma, soma.skeleton != nil {
                for (name, model) in robots {
                    do { try Self.prepareGPU(soma, robot: name, model: model, device: device, repository: config.repositoryRoot) }
                    catch { print("viewer: no retargeting for \(name): \(error)") }
                }
            }
        }
        gpu = Mutex(state)
    }

    /// Skinning and the retargeters of `context` for one robot (GPU lock held).
    static func prepareGPU(_ context: SkeletonContext, robot name: String, model: RobotModel, device: ComputeDevice,
                           repository: String) throws {
        if context.skinning == nil { context.skinning = try SurfaceSkinning(device: device, surface: context.surface) }
        guard context.retargeters[name] == nil else { return }
        guard let skeleton = context.skeleton else {
            throw ComputeError(message: "retargeting needs the SOMA rig (assets/body_models/soma/seed_soma_shapes)")
        }
        let mapping = try RetargetMapping.load(file: repository + "/Configs/mappings/soma77_\(name).json")
        let directory = repository + "/assets/robots/unitree_g1/" + (name == "g1_29dof" ? "g1_29_dof" : "g1_23_dof")
        context.retargeters[name] = try HumanRetargeter(device: device, robot: model, mapping: mapping,
                                                        surface: context.surface, skeleton: skeleton)
        context.offline[name] = try OfflineRetargeter(
            device: device, robot: model, mapping: mapping, surface: context.surface, skeleton: skeleton,
            geometry: try RobotGeometry(robot: model, pointsFile: directory + "/mesh_points.npz"))
    }

    /// SMPL-family parameter file (by extension).
    static func isSMPL(_ path: String) -> Bool { path.hasSuffix(".npz") || path.hasSuffix(".pkl") }

    /// The skeleton a clip is read with, by its extension: SMPL files with the SMPL config, BVH files with the custom
    /// BVH config or as SOMA. Custom ones are built the first time a clip of that skeleton (or SMPL shape) is opened.
    func context(_ path: String) throws -> SkeletonContext {
        let made: SkeletonContext
        if Self.isSMPL(path) {
            guard let smpl else { throw ComputeError(message: "SMPL files need Configs/skeletons/smplx.json") }
            // One body per gender and shape.
            let (clip, motion) = try smpl.load(file: path)
            let key = SkeletonConfig.signature(motion.skeleton)
            if let known = contexts.withLock({ $0[key] }) { return known }
            let body = try smpl.surface(for: clip, motion: motion)
            made = try SkeletonContext(key: key, skeleton: motion.skeleton, surface: body.surface) { try smpl.load(file: $0).motion }
            print("viewer: SMPL body \(key) (\(clip.gender), \(clip.layout)) from \(path), reference "
                  + (body.frame.map { "frame \($0)" } ?? "rest pose"))
        } else if let config = bvhConfig {
            let raw = try BVH.loadSkeleton(file: path, options: config.rawOptions)
            let key = SkeletonConfig.signature(raw)
            if let known = contexts.withLock({ $0[key] }) { return known }
            let options = try config.bvhOptions(for: raw)
            let motion = try BVH.load(file: path, options: options)
            let body = try CustomBody.make(motion, config: config)
            made = try SkeletonContext(key: key, skeleton: motion.skeleton, surface: body.surface) { try BVH.load(file: $0, options: options) }
            print("viewer: skeleton \(key) (\(motion.skeleton.jointCount) joints) from \(path), reference frame \(body.frame)")
        } else {
            // SOMA: the baked body with the clip's own skeleton (its bone lengths set the mapping's scales), as the
            // batch commands use it, so a download equals `retarget offline` on the clip.
            guard let soma else { throw ComputeError(message: "no body surface (run ./setup.sh)") }
            let skeleton = try BVH.loadSkeleton(file: path)
            let key = "soma-" + SkeletonConfig.signature(skeleton)
            if let known = contexts.withLock({ $0[key] }) { return known }
            made = try SkeletonContext(key: key, skeleton: skeleton, surface: soma.surface) { try BVH.load(file: $0) }
            print("viewer: SOMA skeleton \(key) (\(skeleton.jointCount) joints) from \(path)")
        }
        let key = made.key
        return contexts.withLock { known in
            if let existing = known[key] { return existing }   // built meanwhile by another request
            known[key] = made
            return made
        }
    }

    /// A clip read with its skeleton's options.
    func load(_ path: String) throws -> (SkeletonContext, HumanMotion) {
        let context = try context(path)
        return (context, try context.read(path))
    }

    public func run() throws -> Never {
        print("viewer: http://127.0.0.1:\(config.port)/  (motions from \(config.motionRoot))")
        if let skeleton = bvhConfig { print("viewer: .bvh as custom skeleton \(skeleton.name): bodies generated per skeleton") }
        else if soma == nil { print("viewer: no body surface (run ./setup.sh); SOMA skeleton only") }
        if let smpl { print("viewer: .npz / .pkl as SMPL-family clips (\(smpl.config.name)): SMPL-X body per gender and shape") }
        print("viewer: drop clip files on the page to upload them into \(config.motionRoot)/uploads")
        print("viewer: contact analysis on \(gpu.withLock { $0 == nil } ? "CPU (no GPU device)" : "GPU (\(Backend.gpu))")")
        let server = HTTPServer(port: config.port) { [self] request in self.handle(request) }
        try server.run()
    }

    func handle(_ request: HTTPRequest) -> HTTPResponse {
        switch request.path {
        case "/", "/index.html":
            return file(config.repositoryRoot + "/Web/index.html")
        case "/api/robots":
            return .json(JSON.array(robots.map { .object([("name", .string($0.name)), ("dof", .number(Double($0.model.dofCount)))]) }).text)
        case "/api/robot":
            guard let robot = robot(request.query["name"]) else { return .notFound }
            return .json(robotJSON(robot).text)
        case "/api/robot_demo":
            guard let robot = robot(request.query["name"]) else { return .notFound }
            return robotDemo(robot, kind: request.query["demo"] ?? "wave")
        case "/api/retarget":
            guard let robot = robot(request.query["name"]), let file = request.query["file"], let path = safePath(file) else {
                return .text("bad request", status: 400)
            }
            return retarget(path, robot: robot, offline: request.query["mode"] == "offline")
        case "/api/download":
            guard let robot = robot(request.query["name"]), let file = request.query["file"], let path = safePath(file) else {
                return .text("bad request", status: 400)
            }
            return download(path, file: file, robot: robot, offline: request.query["mode"] != "realtime")
        case "/api/dirs":
            // "." when the root itself holds clips.
            let entries = listDirectory(config.motionRoot)
            let dirs = (entries.contains { !$0.isDirectory && isClip($0.name) } ? ["."] : [])
                + entries.filter(\.isDirectory).map(\.name)
            return .json(JSON.array(dirs.map { .string($0) }).text)
        case "/api/files":
            guard let dir = request.query["dir"], let path = safePath(dir) else { return .text("bad dir", status: 400) }
            let files = listDirectory(path).filter { !$0.isDirectory && isClip($0.name) }.map { JSON.string($0.name) }
            return .json(JSON.array(files).text)
        case "/api/upload":
            return upload(request)
        case "/api/human":
            guard let file = request.query["file"], let path = safePath(file) else { return .text("bad file", status: 400) }
            return human(path)
        case "/api/contacts":
            guard let file = request.query["file"], let path = safePath(file) else { return .text("bad file", status: 400) }
            return contacts(path, causal: request.query["mode"] == "causal")
        case "/api/surface":
            // The body of a clip's skeleton (`file`), or SOMA's.
            if let file = request.query["file"] {
                guard let path = safePath(file) else { return .text("bad file", status: 400) }
                do { return surfacePayload(try context(path)) } catch { return .text("\(error)", status: 400) }
            }
            guard let soma else { return .notFound }
            return surfacePayload(soma)
        default:
            if request.path.hasPrefix("/web/") { return file(config.repositoryRoot + "/Web/" + request.path.dropFirst(5)) }
            if request.path.hasPrefix("/robots/") {
                return file(config.repositoryRoot + "/build/viewer/robots/" + request.path.dropFirst(8))
            }
            return .notFound
        }
    }

    // MARK: - Helpers

    func robot(_ name: String?) -> RobotModel? { robots.first { $0.name == name }?.model }

    /// A clip file the viewer reads (.bvh; .npz / .pkl SMPL parameter files).
    func isClip(_ name: String) -> Bool { name.hasSuffix(".bvh") || (smpl != nil && Self.isSMPL(name)) }

    /// Resolves a path relative to the motion root, refusing anything that could escape it.
    func safePath(_ relative: String) -> String? {
        if relative.isEmpty || relative.hasPrefix("/") || relative.split(separator: "/").contains("..") { return nil }
        return config.motionRoot + "/" + relative
    }

    func file(_ path: String) -> HTTPResponse {
        if path.contains("/../") { return .notFound }
        guard let bytes = readFile(path) else { return .notFound }
        let type: String = if path.hasSuffix(".html") { "text/html; charset=utf-8" }
            else if path.hasSuffix(".js") { "text/javascript; charset=utf-8" }
            else if path.hasSuffix(".css") { "text/css; charset=utf-8" }
            else if path.hasSuffix(".glb") { "model/gltf-binary" }
            else { "application/octet-stream" }
        return HTTPResponse(contentType: type, body: bytes)
    }

    /// Hand-made joint trajectories for checking the robot display:
    /// - "zero": default pose; "wave": every joint oscillates inside its limits;
    /// - "sweep": joints move one at a time 0 → upper → lower → 0 (checks axes and limits).
    func robotDemo(_ robot: RobotModel, kind: String) -> HTTPResponse {
        let fps = 50.0
        let joints = robot.links.compactMap { link in link.limits.map { (dof: link.dofIndex!, limits: $0) } }
            .sorted { $0.dof < $1.dof }
        let perJoint = 2.0  // seconds per joint for "sweep"
        let duration = kind == "sweep" ? perJoint * Double(joints.count) : kind == "zero" ? 1 : 10
        let frames = Int(duration * fps)
        var pos: [Float] = [], rot: [Float] = [], qs: [Float] = []
        for f in 0..<frames {
            let t = Double(f) / fps
            var q = [Double](repeating: 0, count: robot.dofCount)
            for (i, joint) in joints.enumerated() {
                let lo = joint.limits.lower, hi = joint.limits.upper
                switch kind {
                case "wave":
                    let amplitude = 0.35 * min(abs(lo), abs(hi), 1.5)
                    q[joint.dof] = amplitude * sin(2 * .pi * (0.25 + 0.03 * Double(i)) * t + Double(i))
                case "sweep":
                    let local = t - Double(i) * perJoint
                    guard local >= 0 && local < perJoint else { continue }
                    let s = local / perJoint  // 0 → hi (0..0.25) → lo (0.25..0.75) → 0 (0.75..1)
                    q[joint.dof] = s < 0.25 ? hi * s / 0.25 : s < 0.75 ? hi + (lo - hi) * (s - 0.25) / 0.5 : lo * (1 - (s - 0.75) / 0.25)
                default: break
                }
                q[joint.dof] = min(max(q[joint.dof], lo), hi)
            }
            qs += q.map(Float.init)
            for t in robot.forwardKinematics(root: robot.defaultRootPose, q: q) {
                pos += [Float(t.translation.x), Float(t.translation.y), Float(t.translation.z)]
                rot += [Float(t.rotation.w), Float(t.rotation.v.x), Float(t.rotation.v.y), Float(t.rotation.v.z)]
            }
        }
        var payload = PackedPayload()
        payload.fields = [("kind", .string("robot")), ("robot", .string(robot.name)), ("fps", .number(fps)),
                          ("frames", .number(Double(frames))), ("links", .number(Double(robot.links.count))),
                          ("dofNames", .array(robot.dofNames.map { .string($0) }))]
        payload.add("pos", floats: pos)
        payload.add("rot", floats: rot)
        payload.add("q", floats: qs)
        return payload.response
    }

    /// POST /api/upload?name=FILE (body: the file): saves a dropped clip under <motion root>/uploads and checks that
    /// it reads. Only the viewer's own page may upload: it sends X-Retargeter-Upload, which a cross-site request cannot
    /// without a CORS preflight that this server never grants, and a browser's Origin must be this server.
    func upload(_ request: HTTPRequest) -> HTTPResponse {
        guard request.method == "POST" else { return .text("POST the file", status: 405) }
        guard request.headers["x-retargeter-upload"] == "1" else { return .text("missing X-Retargeter-Upload", status: 403) }
        if let origin = request.headers["origin"],
           !["http://127.0.0.1:\(config.port)", "http://localhost:\(config.port)"].contains(origin) {
            return .text("cross-origin upload refused", status: 403)
        }
        // Base name only, from a safe character set; no hidden files.
        let raw = String((request.query["name"] ?? "").split(separator: "/").last ?? "")
        let name = String(raw.unicodeScalars.map { s -> Character in
            s.properties.isAlphabetic || ("0"..."9").contains(s) || "._-()".unicodeScalars.contains(s) ? Character(s) : "_"
        })
        guard !name.isEmpty, !name.hasPrefix("."), isClip(name) else {
            return .text("\(raw): not a clip this viewer reads (.bvh\(smpl != nil ? ", .npz, .pkl" : ""))", status: 400)
        }
        guard !request.body.isEmpty else { return .text("\(name): empty file", status: 400) }
        let directory = config.motionRoot + "/uploads"
        let saved: String
        do {
            saved = try uploads.withLock { _ -> String in
                mkdir(directory, 0o755)
                // An identical file already there is reused; a different one with the same name gets a suffix.
                let dot = name.lastIndex(of: ".")!
                var candidate = name, k = 1
                while let existing = readFile(directory + "/" + candidate) {
                    if existing == request.body { return candidate }
                    k += 1
                    candidate = name[..<dot] + "_\(k)" + name[dot...]
                }
                let temporary = directory + "/.upload_\(getpid())_" + candidate
                guard let f = fopen(temporary, "wb") else { throw ComputeError(message: "cannot write \(directory)") }
                let written = request.body.withUnsafeBytes { fwrite($0.baseAddress, 1, $0.count, f) }
                fclose(f)
                guard written == request.body.count, rename(temporary, directory + "/" + candidate) == 0 else {
                    unlink(temporary)
                    throw ComputeError(message: "cannot write \(directory)/\(candidate)")
                }
                return candidate
            }
        } catch { return .text("\(error)", status: 500) }
        // A file that does not read is not kept.
        let path = directory + "/" + saved
        do { _ = try load(path) } catch {
            unlink(path)
            return .text("\(name): " + "\(error)".replacing(path + ": ", with: "").replacing(path, with: "uploads/" + saved), status: 400)
        }
        print("viewer: uploaded \(path) (\(request.body.count) bytes)")
        return .json(JSON.object([("dir", .string("uploads")), ("file", .string(saved))]).text)
    }

    func human(_ path: String) -> HTTPResponse {
        // Without a body (no SOMA surface) the skeleton still shows: read the clip plainly.
        let skeleton: SkeletonContext?, motion: HumanMotion
        do {
            skeleton = bvhConfig == nil && soma == nil && !Self.isSMPL(path) ? nil : try context(path)
            motion = try skeleton?.read(path) ?? BVH.load(file: path)
        } catch { return .text("\(error)", status: 400) }
        var pos: [Float] = [], rot: [Float] = []
        pos.reserveCapacity(motion.frameCount * motion.skeleton.jointCount * 3)
        rot.reserveCapacity(motion.frameCount * motion.skeleton.jointCount * 4)
        for f in 0..<motion.frameCount {
            for t in motion.globalTransforms(frame: f) {
                pos += [Float(t.translation.x), Float(t.translation.y), Float(t.translation.z)]
                rot += [Float(t.rotation.w), Float(t.rotation.v.x), Float(t.rotation.v.y), Float(t.rotation.v.z)]
            }
        }
        var payload = PackedPayload()
        payload.fields = [("kind", .string("human")), ("fps", .number(motion.fps)), ("frames", .number(Double(motion.frameCount))),
                          ("joints", .array(motion.skeleton.jointNames.map { .string($0) })),
                          ("parents", .array(motion.skeleton.parents.map { $0.map { .number(Double($0)) } ?? .number(-1) })),
                          ("body", skeleton.map { .string($0.key) } ?? .null)]
        payload.add("pos", floats: pos)
        payload.add("rot", floats: rot)
        return payload.response
    }

    /// One retargeted clip: the solver result and its validation.
    struct Solved: Sendable {
        let key: String
        let result: RetargetResult
        let report: ValidationReport
    }

    /// Retargets a clip onto `robot` (realtime: lag 0; offline: whole-trajectory) and validates it. The last solve is
    /// kept, so a download right after viewing a clip returns exactly the motion shown without solving again.
    func solve(_ path: String, robot: RobotModel, offline: Bool) throws -> Solved {
        let key = "\(robot.name)|\(offline ? "offline" : "realtime")|\(path)"
        if let last = lastSolve.withLock({ $0 }), last.key == key { return last }
        let (context, motion) = try load(path)
        let solved = try gpu.withLock { state -> Solved in
            guard let state else { throw ComputeError(message: "retargeting unavailable (needs a GPU)") }
            try Self.prepareGPU(context, robot: robot.name, model: robot, device: state.device,
                                repository: config.repositoryRoot)
            guard let retargeter = context.retargeters[robot.name] else { throw ComputeError(message: "no retargeter") }
            let result: RetargetResult
            if offline, let solver = context.offline[robot.name] {
                result = try solver.solve(try solver.prepare([motion]))[0]
            } else {
                result = try retargeter.retarget([motion])[0]
            }
            var report = ValidationReport()
            if let validator = state.validators[robot.name] {
                report = validator.checkHuman(result.prepared, setup: retargeter.setup)
                validator.checkRobot(result, setup: retargeter.setup, fps: Self.exportFPS, into: &report)
            }
            return Solved(key: key, result: result, report: report)
        }
        lastSolve.withLock { $0 = solved }
        return solved
    }

    /// Frame rate of downloaded motions (the batch commands' default, the policy rate).
    static let exportFPS = 50.0

    /// Retargeting of a clip onto `robot`; link poses at the source frame rate.
    func retarget(_ path: String, robot: RobotModel, offline: Bool) -> HTTPResponse {
        let solved: Solved
        do { solved = try solve(path, robot: robot, offline: offline) } catch { return .text("\(error)", status: 500) }
        let trajectory = solved.result.trajectory, report = solved.report
        var pos: [Float] = [], rot: [Float] = [], qs: [Float] = []
        for f in 0..<trajectory.frameCount {
            qs += trajectory.q[f].map(Float.init)
            for t in robot.forwardKinematics(root: trajectory.root[f], q: trajectory.q[f]) {
                pos += [Float(t.translation.x), Float(t.translation.y), Float(t.translation.z)]
                rot += [Float(t.rotation.w), Float(t.rotation.v.x), Float(t.rotation.v.y), Float(t.rotation.v.z)]
            }
        }
        var payload = PackedPayload()
        payload.fields = [("kind", .string("robot")), ("robot", .string(robot.name)), ("fps", .number(trajectory.fps)),
                          ("frames", .number(Double(trajectory.frameCount))), ("links", .number(Double(robot.links.count))),
                          ("dofNames", .array(robot.dofNames.map { .string($0) })), ("validation", report.json)]
        payload.add("pos", floats: pos)
        payload.add("rot", floats: rot)
        payload.add("q", floats: qs)
        payload.add("cost", floats: trajectory.cost.map(Float.init))
        return payload.response
    }

    /// GET /api/download?name=ROBOT&file=CLIP&mode=offline|realtime: the retargeted clip as the .npz the batch
    /// commands write (50 fps, validation status and metrics included), as a file download.
    func download(_ path: String, file: String, robot: RobotModel, offline: Bool) -> HTTPResponse {
        let solved: Solved
        do { solved = try solve(path, robot: robot, offline: offline) } catch { return .text("\(error)", status: 500) }
        let report = solved.report
        var options = ExportOptions()
        options.fps = Self.exportFPS
        options.source = path
        options.configHash = Self.configHash(repository: config.repositoryRoot, robot: robot.name)
        options.status = report.status.rawValue
        options.rejectReasons = report.reasons
        options.warnings = report.warnings
        options.metricsJSON = report.json.text
        options.mode = offline ? "offline" : "realtime"
        if let d = report.dynamics {
            options.tau = d.tau
            options.contactForce = d.contactForce
            options.dynamicsResidual = d.residual
            options.zmp = d.zmp
        }
        let bytes = RobotMotionExport.archive(solved.result, robot: robot, options: options).bytes()
        // <clip>_<robot>[_realtime].npz, in a safe character set (it goes into a header).
        var stem = String(file.split(separator: "/").last ?? "motion")
        if let dot = stem.lastIndex(of: ".") { stem = String(stem[..<dot]) }
        let name = String((stem + "_" + robot.name + (offline ? "" : "_realtime")).map {
            $0.isASCII && ($0.isLetter || $0.isNumber || "-_.".contains($0)) ? $0 : "_"
        }) + ".npz"
        return HTTPResponse(contentType: "application/octet-stream", body: bytes,
                            headers: [("Content-Disposition", "attachment; filename=\"\(name)\"")])
    }

    /// The batch commands' config hash: FNV-1a of the mapping file + the export format version.
    static func configHash(repository: String, robot: String) -> String {
        let bytes = readFile(repository + "/Configs/mappings/soma77_\(robot).json") ?? []
        var h: UInt64 = 1_469_598_103_934_665_603
        for b in bytes { h = (h ^ UInt64(b)) &* 1_099_511_628_211 }
        return String(h, radix: 16) + "-" + RobotMotionExport.version
    }

    /// Ground + contact analysis of a clip (offline or causal) for display.
    func contacts(_ path: String, causal: Bool) -> HTTPResponse {
        let context: SkeletonContext, motion: HumanMotion
        do { (context, motion) = try load(path) } catch { return .text("\(error)", status: 400) }
        let surface = context.surface, groups = context.groups
        let world = (0..<motion.frameCount).map { motion.globalTransforms(frame: $0) }
        let positions: [SIMD3<Float>]
        do {
            positions = try gpu.withLock { state -> [SIMD3<Float>] in
                if let state {
                    if context.skinning == nil { context.skinning = try SurfaceSkinning(device: state.device, surface: surface) }
                    return try context.skinning!.skin(world: world)
                }
                return world.flatMap { surface.skin(world: $0).map { SIMD3(Float($0.x), Float($0.y), Float($0.z)) } }
            }
        } catch { return .text("\(error)", status: 500) }
        let posed = PosedSurface(fps: motion.fps, vertexCount: surface.vertexCount, positions: positions, excluded: surface.excluded)
        let t = causal ? ContactAnalysis.causal(posed, groups: groups) : ContactAnalysis.offline(posed, groups: groups)
        let bools = { (mask: [Bool]) in mask.map { $0 ? Int32(1) : 0 } }
        var payload = PackedPayload()
        payload.fields = [("fps", .number(t.fps)), ("frames", .number(Double(t.frameCount))),
                          ("mode", .string(causal ? "causal" : "offline")),
                          ("groups", .array(t.groupNames.map { .string($0) })),
                          ("groundTilt", .number(t.groundTilt)), ("flags", .array(t.flags.map { .string($0) }))]
        payload.add("ground", floats: t.ground)
        payload.add("contact", ints: bools(t.contact))
        payload.add("envContact", ints: bools(t.envContact))
        payload.add("envSupport", ints: bools(t.envSupport))
        payload.add("unsupported", ints: bools(t.unsupported))
        payload.add("flight", ints: bools(t.flight))
        payload.add("penetration", ints: bools(t.penetration))
        payload.add("zmp", floats: t.zmp.flatMap { [$0.x, $0.y] })
        payload.add("com", floats: t.com.flatMap { [$0.x, $0.y, $0.z] })
        payload.add("height", floats: t.height)
        return payload.response
    }

    func surfacePayload(_ context: SkeletonContext) -> HTTPResponse {
        let s = context.surface
        var payload = PackedPayload()
        payload.fields = [("vertices", .number(Double(s.vertexCount))), ("influences", .number(Double(s.influences))),
                          ("joints", .array(s.jointNames.map { .string($0) })), ("body", .string(context.key))]
        payload.add("rest", floats: s.restVertices.flatMap { [Float($0.x), Float($0.y), Float($0.z)] })
        let inverseBind = s.bind.map(\.inverse)
        payload.add("invBindPos", floats: inverseBind.flatMap { [Float($0.translation.x), Float($0.translation.y), Float($0.translation.z)] })
        payload.add("invBindRot", floats: inverseBind.flatMap { [Float($0.rotation.w), Float($0.rotation.v.x), Float($0.rotation.v.y), Float($0.rotation.v.z)] })
        payload.add("weightJoints", ints: s.weightJoints)
        payload.add("weightValues", floats: s.weightValues)
        payload.add("triangles", ints: s.triangles.flatMap { [$0.x, $0.y, $0.z] })
        do {
            let groups = context.groups
            var vertexGroup = [Int32](repeating: -1, count: s.vertexCount)
            for (g, members) in groups.vertices.enumerated() { for v in members { vertexGroup[v] = Int32(g) } }
            payload.fields.append(("groups", .array(groups.names.map { .string($0) })))
            payload.add("vertexGroup", ints: vertexGroup)
        }
        return payload.response
    }
}

/// Robot description for the browser (links, joints, limits, visual meshes); shared with the live viewer.
func transformJSON(_ t: Transform) -> [(String, JSON)] {
    [("pos", .numbers([t.translation.x, t.translation.y, t.translation.z])),
     ("quat", .numbers([t.rotation.w, t.rotation.v.x, t.rotation.v.y, t.rotation.v.z]))]
}

func robotJSON(_ robot: RobotModel) -> JSON {
    .object([
        ("name", .string(robot.name)),
        ("dofNames", .array(robot.dofNames.map { .string($0) })),
        ("mesh", .string("/robots/\(robot.name).glb")),
        ("links", .array(robot.links.map { link in
            .object([
                ("name", .string(link.name)),
                ("parent", link.parent.map { .number(Double($0)) } ?? .null),
                ("joint", link.jointName.map { .string($0) } ?? .null),
                ("lower", link.limits.map { .number($0.lower) } ?? .null),
                ("upper", link.limits.map { .number($0.upper) } ?? .null),
                ("visuals", .array(link.visuals.map { .object([("mesh", .string($0.mesh))] + transformJSON($0.origin)) })),
            ])
        })),
    ])
}
