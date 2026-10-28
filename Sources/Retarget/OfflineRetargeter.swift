// Offline (whole-trajectory, non-causal) retargeting: offline ground, realtime solve as the warm start,
// then the offline solver with stance locking, ground contact / non-penetration and self-collision terms.

import Compute
import Contact
import Core

public final class OfflineRetargeter {
    public let base: HumanRetargeter
    public let solver: OfflineSolver

    public init(device: ComputeDevice, robot: RobotModel, mapping: RetargetMapping, surface: BodySurface,
                skeleton: HumanSkeleton, geometry: RobotGeometry) throws {
        base = try HumanRetargeter(device: device, robot: robot, mapping: mapping, surface: surface, skeleton: skeleton)
        solver = try OfflineSolver(device: device, setup: base.setup, geometry: geometry)
    }

    public var setup: RetargetSetup { base.setup }

    public func prepare(_ motions: [HumanMotion]) throws -> [PreparedClip] {
        try base.prepare(motions, offlineGround: true)
    }

    /// Realtime warm start, then the offline sweeps. Results carry the offline contact analysis.
    public func solve(_ clips: [PreparedClip]) throws -> [RetargetResult] {
        guard !clips.isEmpty else { return [] }
        let initial = try warmStart(clips)
        let trajectories = try solver.solve(inputs(clips, initial: initial))
        return zip(trajectories, clips).map { RetargetResult(trajectory: $0.0, contacts: $0.1.offline, prepared: $0.1) }
    }

    /// The realtime (causal) solve as the warm start, with long clips cut into segments solved side by side: the
    /// device runs one sequential stream per segment, and a whole long clip kept one stream busy while the rest of
    /// the device idled. Each segment starts `warmOverlap` earlier from a cold start; those frames are dropped.
    func warmStart(_ clips: [PreparedClip]) throws -> [RetargetResult] {
        let o = setup.mapping.offline
        var pieces: [(clip: Int, start: Int, warm: Int, end: Int)] = []
        for (c, clip) in clips.enumerated() {
            let F = clip.goals.count
            let length = o.warmSegment > 0 ? max(1, Int((o.warmSegment * clip.fps).rounded())) : F
            for start in stride(from: 0, to: F, by: length) {
                pieces.append((c, start, max(0, start - Int((o.warmOverlap * clip.fps).rounded())), min(F, start + length)))
            }
        }
        let contacts = clips.map { base.realtimeContacts($0.causal) }
        let solved = try base.solver.solve(
            goals: pieces.map { Array(clips[$0.clip].goals[$0.warm..<$0.end]) }, fps: pieces.map { clips[$0.clip].fps },
            contacts: pieces.map { p in
                RealtimeContacts(sole: Array(contacts[p.clip].sole[p.warm..<p.end]), link: Array(contacts[p.clip].link[p.warm..<p.end]))
            })
        return clips.indices.map { c in
            var root: [Transform] = [], q: [[Double]] = [], cost: [Double] = []
            for (p, piece) in pieces.enumerated() where piece.clip == c {
                let keep = (piece.start - piece.warm)...
                root += solved[p].root[keep]
                q += solved[p].q[keep]
                cost += solved[p].cost[keep]
            }
            let trajectory = RobotTrajectory(fps: clips[c].fps, root: root, q: q, cost: cost)
            return RetargetResult(trajectory: trajectory, contacts: clips[c].causal, prepared: clips[c])
        }
    }

    /// Solver inputs: goals, warm start and the offline contact schedule.
    public func inputs(_ clips: [PreparedClip], initial: [RetargetResult]) -> [OfflineClip] {
        let L = setup.robot.links.count
        return zip(clips, initial).map { clip, start in
            let c = clip.offline
            return OfflineClip(
                fps: clip.fps, goals: clip.goals, initial: start.trajectory,
                footContact: (0..<c.frameCount).map { f in base.footGroups.map { c.contact(f, $0) } },
                linkContact: (0..<c.frameCount).map { f in
                    var weights = [Float](repeating: 0, count: L)
                    for (g, l) in base.contactLinks where c.contact(f, g) { weights[l] = 1 }
                    return weights
                })
        }
    }
}
