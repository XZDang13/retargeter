// Human motion -> robot trajectory with the causal (realtime, lag = 0) solver, batched over clips:
// skeleton FK and body-surface skinning (CPU), causal ground/contact tracking, goal construction,
// then one IK dispatch for all clips.

import Compute
import Contact
import Core
import Kinematics

public struct RetargetResult: Sendable {
    public var trajectory: RobotTrajectory
    /// Causal contact analysis of the human clip (ground heights, contact masks, flags) — what the solver used.
    public var contacts: ContactTimeline
    /// The prepared inputs (human world joints, offline contact analysis, goals).
    public var prepared: PreparedClip
}

/// Everything derived from a human clip before solving.
public struct PreparedClip: Sendable {
    public var fps: Double
    /// Human world joint transforms per frame (unshifted).
    public var world: [[Transform]]
    /// Causal contacts (realtime ground) and offline contacts (non-causal, for validation).
    public var causal: ContactTimeline
    public var offline: ContactTimeline
    /// Contact flags per group decided `LaggedContactEstimator.delay` frames late (fixed-lag streams), for the
    /// frames that have one (all but the last `delay`); empty unless requested.
    public var lagged: [[Bool]] = []
    /// Goals per frame and target, after the causal ground shift.
    public var goals: [[Transform]]
}

public final class HumanRetargeter {
    public let robot: RobotModel
    public let setup: RetargetSetup
    public let surface: BodySurface
    public let groups: ContactGroups
    let device: ComputeDevice
    let solver: RetargetSolver
    /// Robot pose matched to the human reference pose (diagnostics).
    public let referenceRobotPose: (root: Transform, q: [Double])
    /// Mapped body-part contacts (human group -> robot link) and the heel / toe groups of each foot.
    public let contactLinks: [(group: Int, link: Int)]
    public let footGroups: [Int]

    public init(device: ComputeDevice, robot: RobotModel, mapping: RetargetMapping, surface: BodySurface,
                skeleton: HumanSkeleton) throws {
        guard let reference = surface.referencePose else {
            throw ComputeError(message: "body surface has no reference pose (re-run tools/bake_soma_surface.py)")
        }
        self.device = device
        self.robot = robot
        self.surface = surface
        groups = try ContactGroups.soma(surface)
        var setup = try RetargetSetup(mapping: mapping, robot: robot, skeleton: skeleton, humanReference: reference)
        solver = try RetargetSolver(device: device, setup: setup)
        if let pose = mapping.referencePose {
            for name in pose.keys where !robot.dofNames.contains(name) {
                throw ComputeError(message: "reference_pose: unknown joint \(name)")
            }
            referenceRobotPose = (robot.defaultRootPose, robot.dofNames.map { pose[$0] ?? 0 })
        } else {
            referenceRobotPose = try solver.referencePose(human: reference)
        }
        setup.alignRotations(human: reference, robotRoot: referenceRobotPose.root, robotQ: referenceRobotPose.q)
        self.setup = setup
        let groups = self.groups
        contactLinks = try mapping.contacts.map { group, link in
            guard let g = groups.index(of: group) else { throw ComputeError(message: "mapping contacts: no human group \(group)") }
            guard let l = robot.linkIndex(named: link) else { throw ComputeError(message: "mapping contacts: no robot link \(link)") }
            return (g, l)
        }
        footGroups = try ["left_heel", "left_toe", "right_heel", "right_toe"].map { name in
            guard let g = groups.index(of: name) else { throw ComputeError(message: "no human group \(name)") }
            return g
        }
    }

    /// Goals for every frame: world joints shifted so that the (causally tracked) ground is at z = 0.
    public func goals(world: [[Transform]], ground: [Float]) -> [[Transform]] {
        zip(world, ground).map { frame, g in
            let shift = Transform(translation: Vec3(0, 0, -Double(g)))
            return setup.goals(human: frame.map { shift * $0 })
        }
    }

    /// Wall-clock split of the last `retarget` call.
    public private(set) var lastTiming: (prepare: Duration, solve: Duration) = (.zero, .zero)

    /// FK, skinning, causal + offline contact analysis and goals for each clip. Goals use the causal
    /// (realtime) ground unless `offlineGround`, which uses the non-causal per-clip ground. Clips are prepared in
    /// parallel and on the CPU only, so a batch can be prepared while the device solves another.
    public func prepare(_ motions: [HumanMotion], offlineGround: Bool = false, laggedContacts: Bool = false) throws -> [PreparedClip] {
        let shared = Shared(retargeter: self)
        // Each worker holds one clip's skinned surface (frames x vertices); a few workers bound the memory.
        return parallelMap(motions.count, threads: 8) { i in
            shared.retargeter.prepare(motions[i], offlineGround: offlineGround, laggedContacts: laggedContacts)
        }
    }

    private struct Shared: @unchecked Sendable { let retargeter: HumanRetargeter }

    private func prepare(_ motion: HumanMotion, offlineGround: Bool, laggedContacts: Bool) -> PreparedClip {
        let chunk = 256
        let world = parallelMap((motion.frameCount + chunk - 1) / chunk, threads: 4) { c in
            ((c * chunk)..<min(motion.frameCount, (c + 1) * chunk)).map { motion.globalTransforms(frame: $0) }
        }.flatMap { $0 }
        let posed = PosedSurface(fps: motion.fps, vertexCount: surface.vertexCount, positions: surface.skin(frames: world),
                                 excluded: surface.excluded)
        let groups = self.groups
        let analyses = parallelMap(2) { k in
            k == 0 ? ContactAnalysis.causal(posed, groups: groups) : ContactAnalysis.offline(posed, groups: groups)
        }
        let (causal, offline) = (analyses[0], analyses[1])
        var clip = PreparedClip(fps: motion.fps, world: world, causal: causal, offline: offline,
                                goals: goals(world: world, ground: offlineGround ? offline.ground : causal.ground))
        if laggedContacts {
            // As a live stream decides them: frame by frame, on the causal ground.
            let estimator = LaggedContactEstimator(groups: groups, fps: motion.fps, vertexCount: posed.vertexCount,
                                                   excluded: posed.excluded)
            let V = posed.vertexCount
            for f in 0..<posed.frameCount {
                if let flags = estimator.update(Array(posed.positions[(f * V)..<((f + 1) * V)]), ground: causal.ground[f]) {
                    clip.lagged.append(flags)
                }
            }
        }
        return clip
    }

    /// One batched solve over prepared clips.
    public func solve(_ clips: [PreparedClip]) throws -> [RetargetResult] {
        guard !clips.isEmpty else { return [] }
        let trajectories = try solver.solve(goals: clips.map(\.goals), fps: clips.map(\.fps),
                                            contacts: clips.map { realtimeContacts($0.causal) })
        return zip(trajectories, clips).map { RetargetResult(trajectory: $0.0, contacts: $0.1.causal, prepared: $0.1) }
    }

    /// Contact flags per group -> heel / toe contacts [left heel, left toe, right heel, right toe] and the robot links
    /// in mapped body-part contact.
    public func robotContacts(_ flags: [Bool]) -> (foot: [Bool], link: [Bool]) {
        var link = [Bool](repeating: false, count: robot.links.count)
        for (g, l) in contactLinks where flags[g] { link[l] = true }
        return (footGroups.map { flags[$0] }, link)
    }

    /// Dynamics contacts from a (causal) contact timeline: sole corners under touching heels / toes (two
    /// heel corners, then two toe corners per foot) and mapped body parts.
    public func realtimeContacts(_ c: ContactTimeline) -> RealtimeContacts {
        let L = robot.links.count
        return RealtimeContacts(
            sole: (0..<c.frameCount).map { f in
                (0..<8).map { k in c.contact(f, footGroups[(k / 4) * 2 + (k % 4 < 2 ? 0 : 1)]) }
            },
            link: (0..<c.frameCount).map { f in
                var on = [Bool](repeating: false, count: L)
                for (g, l) in contactLinks where c.contact(f, g) { on[l] = true }
                return on
            })
    }

    public func retarget(_ motions: [HumanMotion]) throws -> [RetargetResult] {
        let start = ContinuousClock.now
        let clips = try prepare(motions)
        let prepared = ContinuousClock.now
        let results = try solve(clips)
        lastTiming = (prepared - start, ContinuousClock.now - prepared)
        return results
    }
}
