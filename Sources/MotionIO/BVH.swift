// BVH reader. Produces a HumanMotion in our canonical frame (metres, Z-up, +X forward); BVH files
// from SOMA / BONES-SEED are centimetres, Y-up, +Z forward.

import Core

#if canImport(Glibc)
import Glibc
#elseif canImport(Darwin)
import Darwin
#endif

public struct BVHError: Error, CustomStringConvertible {
    public let message: String
    public init(message: String) { self.message = message }
    public var description: String { message }
}

public struct BVHOptions: Sendable {
    /// Multiplier from file units to metres (SOMA BVH: centimetres).
    public var unitScale: Double = 0.01
    /// Convert from the file's Y-up basis to our Z-up basis.
    public var yUp: Bool = true
    public var skeletonName: String = "soma77"

    public init() {}
}

public enum BVH {
    private enum Channel {
        case xPosition, yPosition, zPosition, xRotation, yRotation, zRotation
    }

    private struct JointSpec {
        var name: String
        var parent: Int?
        var offset: Vec3
        var channels: [Channel]
    }

    public static func load(file: String, options: BVHOptions = BVHOptions()) throws -> HumanMotion {
        guard let file_ = fopen(file, "rb") else { throw BVHError(message: "cannot open \(file)") }
        fseek(file_, 0, SEEK_END)
        let size = ftell(file_)
        fseek(file_, 0, SEEK_SET)
        // NUL-terminated buffer so strtod can scan in place.
        let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: size + 1)
        defer { buffer.deallocate() }
        let read = fread(buffer, 1, size, file_)
        fclose(file_)
        guard read == size else { throw BVHError(message: "short read on \(file)") }
        buffer[size] = 0
        return try parse(buffer, options: options, source: file)
    }

    public static func parse(text: String, options: BVHOptions = BVHOptions()) throws -> HumanMotion {
        var copy = Array(text.utf8CString)
        return try copy.withUnsafeMutableBufferPointer { try parse($0.baseAddress!, options: options, source: "<text>") }
    }

    private static func parse(_ start: UnsafeMutablePointer<CChar>, options: BVHOptions, source: String) throws -> HumanMotion {
        var cursor = UnsafeMutablePointer(start)

        func skipSpace() {
            while cursor.pointee != 0 && isspace(Int32(UInt8(bitPattern: cursor.pointee))) != 0 { cursor += 1 }
        }
        func word() -> String {
            skipSpace()
            let begin = cursor
            while cursor.pointee != 0 && isspace(Int32(UInt8(bitPattern: cursor.pointee))) == 0 { cursor += 1 }
            return String(decoding: UnsafeRawBufferPointer(start: begin, count: cursor - begin), as: UTF8.self)
        }
        func number() throws -> Double {
            var end: UnsafeMutablePointer<CChar>?
            let value = strtod(cursor, &end)
            guard let end, end != cursor else { throw BVHError(message: "\(source): expected a number") }
            cursor = end
            return value
        }

        guard word() == "HIERARCHY" else { throw BVHError(message: "\(source): missing HIERARCHY") }

        var joints: [JointSpec] = []
        var stack: [Int?] = []  // nil marks an End Site block
        var pendingParent: Int? = nil
        var pendingEndSite = false
        loop: while true {
            let token = word()
            switch token {
            case "ROOT", "JOINT":
                let name = word()
                joints.append(JointSpec(name: name, parent: stack.last.flatMap { $0 }, offset: .zero, channels: []))
                pendingParent = joints.count - 1
                pendingEndSite = false
            case "End":
                _ = word()  // "Site"
                pendingEndSite = true
            case "{":
                stack.append(pendingEndSite ? nil : pendingParent)
            case "}":
                _ = stack.popLast()
                pendingEndSite = false
            case "OFFSET":
                let offset = Vec3(try number(), try number(), try number())
                if !pendingEndSite, let current = stack.last, let index = current { joints[index].offset = offset }
            case "CHANNELS":
                let count = Int(try number())
                var channels: [Channel] = []
                for _ in 0..<count {
                    switch word() {
                    case "Xposition": channels.append(.xPosition)
                    case "Yposition": channels.append(.yPosition)
                    case "Zposition": channels.append(.zPosition)
                    case "Xrotation": channels.append(.xRotation)
                    case "Yrotation": channels.append(.yRotation)
                    case "Zrotation": channels.append(.zRotation)
                    case let other: throw BVHError(message: "\(source): unknown channel \(other)")
                    }
                }
                if let current = stack.last, let index = current { joints[index].channels = channels }
            case "MOTION":
                break loop
            case "":
                throw BVHError(message: "\(source): unexpected end of hierarchy")
            default:
                throw BVHError(message: "\(source): unexpected token \(token)")
            }
        }

        guard word() == "Frames:" else { throw BVHError(message: "\(source): missing Frames:") }
        let frameCount = Int(try number())
        guard word() == "Frame", word() == "Time:" else { throw BVHError(message: "\(source): missing Frame Time:") }
        let frameTime = try number()

        let scale = options.unitScale
        func basis(_ p: Vec3) -> Vec3 { options.yUp ? YUpToZUp.point(p) : p }
        func basis(_ q: Quat) -> Quat { options.yUp ? YUpToZUp.rotation(q) : q }

        let skeleton = HumanSkeleton(name: options.skeletonName, jointNames: joints.map(\.name),
                                     parents: joints.map(\.parent),
                                     restOffsets: joints.map { basis($0.offset * scale) })

        let degrees = Double.pi / 180
        let axisX = Vec3(1, 0, 0), axisY = Vec3(0, 1, 0), axisZ = Vec3(0, 0, 1)
        var frames: [[Transform]] = []
        frames.reserveCapacity(frameCount)
        for _ in 0..<frameCount {
            var locals: [Transform] = []
            locals.reserveCapacity(joints.count)
            for joint in joints {
                var translation = joint.offset
                var rotation = Quat.identity
                var hasPosition = false
                for channel in joint.channels {
                    let value = try number()
                    switch channel {
                    case .xPosition, .yPosition, .zPosition:
                        if !hasPosition { translation = .zero; hasPosition = true }
                        if channel == .xPosition { translation.x = value }
                        else if channel == .yPosition { translation.y = value }
                        else { translation.z = value }
                    // Channels listed left to right compose as R = R1 · R2 · R3.
                    case .xRotation: rotation = rotation * Quat(axis: axisX, angle: value * degrees)
                    case .yRotation: rotation = rotation * Quat(axis: axisY, angle: value * degrees)
                    case .zRotation: rotation = rotation * Quat(axis: axisZ, angle: value * degrees)
                    }
                }
                locals.append(Transform(rotation: basis(rotation), translation: basis(translation * scale)))
            }
            frames.append(locals)
        }
        return HumanMotion(skeleton: skeleton, fps: 1 / frameTime, local: frames)
    }
}
