// Minimal test harness shared by the test executables: named cases, soft expectations, exit code.

#if canImport(Glibc)
import Glibc
#endif

nonisolated(unsafe) var testFailures = 0
nonisolated(unsafe) var currentTest = ""

func test(_ name: String, _ body: () throws -> Void) {
    currentTest = name
    let before = testFailures
    do {
        try body()
    } catch {
        testFailures += 1
        print("  FAIL [\(name)]: threw \(error)")
    }
    print(testFailures == before ? "ok   \(name)" : "FAIL \(name)")
}

func expect(_ condition: Bool, _ message: @autoclosure () -> String, file: StaticString = #file, line: UInt = #line) {
    if !condition {
        testFailures += 1
        print("  FAIL [\(currentTest)] \(file):\(line): \(message())")
    }
}

func expectClose(_ a: Double, _ b: Double, tolerance: Double, _ what: @autoclosure () -> String,
                 file: StaticString = #file, line: UInt = #line) {
    expect(abs(a - b) <= tolerance, "\(what()): \(a) vs \(b) (tol \(tolerance))", file: file, line: line)
}

func finishTests() -> Never {
    if testFailures > 0 {
        print("\(testFailures) check(s) failed")
        exit(1)
    }
    print("all checks passed")
    exit(0)
}

/// Repository root, derived from this file's location (Tests/Support/TestSupport.swift).
let repositoryRoot: String = {
    var parts = #filePath.split(separator: "/", omittingEmptySubsequences: false)
    parts.removeLast(3)
    return parts.joined(separator: "/")
}()

/// SplitMix64, so test inputs are identical on every run.
struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform(_ lo: Double, _ hi: Double) -> Double {
        lo + (hi - lo) * Double(next() >> 11) / Double(1 << 53)
    }
}
