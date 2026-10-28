// Builds a RobotModel from a robot asset directory produced by the MuJoCo USD converter:
//   <dir>/<root>.usda, payloads/{base,robot,instances}.usda, payloads/Physics/{physics,mujoco}.usda

public struct RobotLoadError: Error, CustomStringConvertible {
    public let message: String
    public init(message: String) { self.message = message }
    public var description: String { message }
}

public enum RobotUSDLoader {
    /// - Parameter directory: e.g. `assets/robots/unitree_g1/g1_29_dof`.
    public static func load(directory: String, name: String? = nil) throws -> RobotModel {
        let payloads = directory + "/payloads"
        let base = try USDLayer(file: payloads + "/base.usda")
        let layers = [base] + (try ["robot.usda", "Physics/physics.usda", "Physics/mujoco.usda"].map {
            try USDLayer(file: payloads + "/" + $0)
        })
        let stage = USDStage(layers: layers)
        let instances = try USDLayer(file: payloads + "/instances.usda")

        guard let rootPrim = base.prims.first,
              let geometry = stage.prim(at: rootPrim.path + "/Geometry"),
              let pelvisPrim = geometry.children.first(where: { $0.typeName == "Xform" }) else {
            throw RobotLoadError(message: "\(directory): no Geometry/<root link> prim")
        }

        // Joints, keyed by the child body path.
        var jointByChild: [String: USDPrim] = [:]
        for prim in stage.index.values where prim.typeName?.hasSuffix("Joint") == true && prim.typeName != "PhysicsJoint" {
            if let child = prim[attribute: "physics:body1"]?.items.first?.stringValue { jointByChild[child] = prim }
        }

        // Links: rigid bodies plus joint children, visited depth-first in authored order.
        func isLink(_ prim: USDPrim) -> Bool {
            prim.typeName == "Xform" && (prim.apiSchemas.contains("PhysicsRigidBodyAPI") || jointByChild[prim.path] != nil
                                          || prim === pelvisPrim)
        }

        var links: [Link] = []
        var dofNames: [String] = []

        func visit(_ prim: USDPrim, parentLink: Int?, fromParent: Transform) throws {
            guard isLink(prim) else { return }
            let index = links.count
            var link = Link(name: prim.name, parent: parentLink, jointOrigin: fromParent, jointToChild: .identity,
                            joint: .fixed)
            if let joint = jointByChild[prim.path], parentLink != nil {
                link.jointName = joint.name
                let origin0 = Transform(rotation: joint[attribute: "physics:localRot0"]?.quat ?? .identity,
                                        translation: joint[attribute: "physics:localPos0"]?.vec3 ?? .zero)
                let origin1 = Transform(rotation: joint[attribute: "physics:localRot1"]?.quat ?? .identity,
                                        translation: joint[attribute: "physics:localPos1"]?.vec3 ?? .zero)
                link.jointOrigin = origin0
                link.jointToChild = origin1.inverse
                if joint.typeName == "PhysicsRevoluteJoint" {
                    let axis: Vec3 = switch joint[attribute: "physics:axis"]?.stringValue {
                    case "X": Vec3(1, 0, 0)
                    case "Y": Vec3(0, 1, 0)
                    default: Vec3(0, 0, 1)
                    }
                    link.joint = .revolute(axis: axis)
                    link.dofIndex = dofNames.count
                    dofNames.append(joint.name)
                    let degrees = Double.pi / 180
                    link.limits = JointLimits(
                        lower: (joint[attribute: "physics:lowerLimit"]?.double ?? -.infinity) * degrees,
                        upper: (joint[attribute: "physics:upperLimit"]?.double ?? .infinity) * degrees,
                        effort: joint[attribute: "mjc:actuatorfrcrange:max"]?.double,
                        velocity: nil,
                        armature: joint[attribute: "mjc:armature"]?.double)
                }
            }
            if let mass = prim[attribute: "physics:mass"]?.double {
                link.inertial = Inertial(mass: mass,
                                         centerOfMass: prim[attribute: "physics:centerOfMass"]?.vec3 ?? .zero,
                                         diagonalInertia: prim[attribute: "physics:diagonalInertia"]?.vec3 ?? .zero,
                                         principalAxes: prim[attribute: "physics:principalAxes"]?.quat ?? .identity)
            }
            links.append(link)
            collectShapes(of: prim, linkIndex: index, instances: instances, links: &links)
            for child in prim.children {
                try visit(child, parentLink: index, fromParent: child.localTransform)
            }
        }

        try visit(pelvisPrim, parentLink: nil, fromParent: .identity)

        return RobotModel(name: name ?? rootPrim.name, links: links, dofNames: dofNames,
                          defaultRootPose: pelvisPrim.localTransform)
    }

    /// Collision primitives and visual mesh instances below a link (stopping at child links).
    private static func collectShapes(of linkPrim: USDPrim, linkIndex: Int, instances: USDLayer, links: inout [Link]) {
        func walk(_ prim: USDPrim, origin: Transform) {
            for child in prim.children {
                if child.typeName == "Xform" && (child.apiSchemas.contains("PhysicsRigidBodyAPI")) { continue }
                let childOrigin = origin * child.localTransform
                let isGuide = child[attribute: "purpose"]?.stringValue == "guide"
                if child.apiSchemas.contains("PhysicsCollisionAPI"), !isGuide,
                   let shape = collisionShape(child) {
                    links[linkIndex].collisions.append(Collision(name: child.name, shape: shape, origin: childOrigin))
                }
                for reference in child.references {
                    // @./instances.usda@</Instances/<name>> → visual unless the instance is a collision mesh.
                    guard reference.contains("instances.usda"), let open = reference.lastIndex(of: "<") else { continue }
                    let path = String(reference[reference.index(after: open)...].dropLast())
                    guard let instance = instances.prim(at: path) else { continue }
                    let isCollision = instance.children.contains { $0.apiSchemas.contains("PhysicsCollisionAPI") }
                    guard !isCollision else { continue }
                    let mesh = instance.references.compactMap { ref -> String? in
                        guard let slash = ref.lastIndex(of: "/") else { return nil }
                        return String(ref[ref.index(after: slash)...].dropLast())
                    }.first ?? instance.name
                    links[linkIndex].visuals.append(Visual(name: child.name, mesh: mesh, origin: childOrigin))
                }
                if child.typeName == "Xform" && child.references.isEmpty { walk(child, origin: childOrigin) }
            }
        }
        walk(linkPrim, origin: .identity)
    }

    private static func collisionShape(_ prim: USDPrim) -> CollisionShape? {
        let radius = prim[attribute: "radius"]?.double ?? 0
        let height = prim[attribute: "height"]?.double ?? 0
        let axis = switch prim[attribute: "axis"]?.stringValue {
        case "X": 0
        case "Y": 1
        default: 2
        }
        switch prim.typeName {
        case "Sphere": return .sphere(radius: radius)
        case "Cylinder": return .cylinder(radius: radius, height: height, axis: axis)
        case "Capsule": return .capsule(radius: radius, height: height, axis: axis)
        default: return nil
        }
    }
}
