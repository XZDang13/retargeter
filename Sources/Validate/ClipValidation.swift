// Validation of our own retargeting results (both stages) and loading of the validator from the repository.

import Contact
import Core
import MotionIO
import Retarget

extension Validator {
    /// Loads reject rules, robot properties and robot geometry for `robot` (g1_29dof / g1_23dof).
    public static func load(repository: String, robot: RobotModel, robotDirectory: String,
                            rejectConfig: String? = nil) throws -> Validator {
        let config = try RejectConfig.load(file: rejectConfig ?? repository + "/Configs/reject/g1.json")
        let geometry = try RobotGeometry(robot: robot, pointsFile: robotDirectory + "/mesh_points.npz")
        let robotConfig = try? JSONValue.load(file: repository + "/Configs/robots/g1.json")
        return Validator(config: config, robot: robot, geometry: geometry, robotConfig: robotConfig)
    }

    /// Human pre-filter only (before solving).
    public func checkHuman(_ clip: PreparedClip, setup: RetargetSetup) -> ValidationReport {
        var report = ValidationReport()
        checkHuman(world: clip.world, fps: clip.fps, contacts: clip.offline, rootJoint: setup.rootHumanJoint, into: &report)
        return report
    }

    /// Adds the robot stage to `report`, on the trajectory as exported (resampled to `fps`).
    public func checkRobot(_ result: RetargetResult, setup: RetargetSetup, fps: Double, into report: inout ValidationReport) {
        let trajectory = result.trajectory.resampled(to: fps)
        let clip = result.prepared
        let goals = (0..<trajectory.frameCount).map { i in
            clip.goals[min(clip.goals.count - 1, Int((Double(i) / trajectory.fps * clip.fps).rounded()))]
        }
        checkRobot(trajectory, goals: goals, setup: setup, contacts: clip.offline, into: &report)
    }
}
