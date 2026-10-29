// retarget — command-line entry point. Milestone 1: `retarget view`.

import Core
import Viewer

#if canImport(Glibc)
import Glibc
#endif

let usage = """
usage: retarget <command> [options]
  view [--port N] [--motions DIR]   serve the browser viewer on http://127.0.0.1:N (default 8080);
                                    --motions: human BVH root (default $SOMA_BVH_ROOT or ~/Downloads/soma_uniform/bvh)
  realtime <inputs>... [options]    batch-retarget BVH files with the causal solver (see retarget realtime --help)
  offline <inputs>... [options]     batch-retarget with the whole-trajectory solver (same options)
  validate <files>... --human DIR   validate robot motions (.npz / soma-retargeter .csv) against their source clips
  teleop [options]                  live retargeting server over ZeroMQ (see retarget teleop --help)
"""

var arguments = Array(CommandLine.arguments.dropFirst())
guard let command = arguments.first else {
    print(usage)
    exit(2)
}
arguments.removeFirst()

@MainActor func option(_ name: String) -> String? {
    guard let i = arguments.firstIndex(of: name), i + 1 < arguments.count else { return nil }
    return arguments[i + 1]
}

switch command {
case "view":
    guard let root = RepositoryLocation.root() else {
        print("cannot locate the repository; set RETARGETER_ROOT")
        exit(1)
    }
    let home = getenv("HOME").map { String(cString: $0) } ?? ""
    let config = ViewerConfiguration(repositoryRoot: root,
                                     motionRoot: option("--motions") ?? getenv("SOMA_BVH_ROOT").map { String(cString: $0) }
                                         ?? home + "/Downloads/soma_uniform/bvh",
                                     port: UInt16(option("--port") ?? "8080") ?? 8080)
    do {
        try ViewerApp(config: config).run()
    } catch {
        print("error: \(error)")
        exit(1)
    }
case "realtime", "offline":
    if arguments.contains("--help") {
        print(RealtimeCommand.usage)
        exit(0)
    }
    guard let root = RepositoryLocation.root() else {
        print("cannot locate the repository; set RETARGETER_ROOT")
        exit(1)
    }
    do {
        try RealtimeCommand(mode: command, arguments: arguments).run(repository: root)
    } catch {
        print("error: \(error)")
        exit(1)
    }
case "validate":
    if arguments.contains("--help") {
        print(ValidateCommand.usage)
        exit(0)
    }
    guard let root = RepositoryLocation.root() else {
        print("cannot locate the repository; set RETARGETER_ROOT")
        exit(1)
    }
    do {
        try ValidateCommand(arguments: arguments).run(repository: root)
    } catch {
        print("error: \(error)")
        exit(1)
    }
case "teleop":
    if arguments.contains("--help") {
        print(TeleopCommand.usage)
        exit(0)
    }
    guard let root = RepositoryLocation.root() else {
        print("cannot locate the repository; set RETARGETER_ROOT")
        exit(1)
    }
    do {
        try TeleopCommand(arguments: arguments).run(repository: root)
    } catch {
        print("error: \(error)")
        exit(1)
    }
default:
    print(usage)
    exit(2)
}
