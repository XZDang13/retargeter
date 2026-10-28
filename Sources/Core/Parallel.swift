// Minimal data-parallel map on POSIX threads (no Foundation / Dispatch in this project).

#if canImport(Glibc)
import Glibc
#endif

private final class ParallelJob: @unchecked Sendable {
    let body: @Sendable (Int) -> Void
    let count: Int
    var next = 0
    var lock = pthread_mutex_t()

    init(count: Int, body: @escaping @Sendable (Int) -> Void) {
        self.count = count
        self.body = body
        pthread_mutex_init(&lock, nil)
    }

    deinit { pthread_mutex_destroy(&lock) }

    /// Next item index, or nil when all are taken.
    func take() -> Int? {
        pthread_mutex_lock(&lock)
        defer { pthread_mutex_unlock(&lock) }
        guard next < count else { return nil }
        next += 1
        return next - 1
    }
}

private final class ResultBox<T>: @unchecked Sendable {
    var values: [T?]
    var lock = pthread_mutex_t()
    init(count: Int) {
        values = Array(repeating: nil, count: count)
        pthread_mutex_init(&lock, nil)
    }
    deinit { pthread_mutex_destroy(&lock) }
    func set(_ i: Int, _ v: T) {
        pthread_mutex_lock(&lock)
        values[i] = v
        pthread_mutex_unlock(&lock)
    }
}

/// `body(i)` for i in 0..<count on up to `threads` threads (default: online CPUs), results in index order.
public func parallelMap<T>(_ count: Int, threads: Int? = nil, _ body: @escaping @Sendable (Int) -> T) -> [T] {
    let workers = max(1, min(count, threads ?? Int(sysconf(Int32(_SC_NPROCESSORS_ONLN)))))
    if workers <= 1 { return (0..<count).map(body) }
    let results = ResultBox<T>(count: count)
    let job = ParallelJob(count: count) { i in results.set(i, body(i)) }
    var handles: [pthread_t] = []
    for _ in 0..<workers {
        var thread = pthread_t()
        let context = Unmanaged.passRetained(job).toOpaque()
        let status = pthread_create(&thread, nil, { raw in
            let job = Unmanaged<ParallelJob>.fromOpaque(raw!).takeRetainedValue()
            while let i = job.take() { job.body(i) }
            return nil
        }, context)
        if status == 0 {
            handles.append(thread)
        } else {
            Unmanaged<ParallelJob>.fromOpaque(context).release()
        }
    }
    // Work on this thread too (also covers a failed thread start).
    while let i = job.take() { job.body(i) }
    for thread in handles { pthread_join(thread, nil) }
    return results.values.map { $0! }
}

/// A raw pointer handed to `parallelMap` workers that write disjoint ranges.
public struct SendablePointer<T>: @unchecked Sendable {
    public let base: UnsafeMutablePointer<T>
    public init(base: UnsafeMutablePointer<T>) { self.base = base }
}

/// A mutex, e.g. to serialise the use of one compute device from `parallelMap` workers.
public final class Lock: @unchecked Sendable {
    private var mutex = pthread_mutex_t()

    public init() { pthread_mutex_init(&mutex, nil) }

    deinit { pthread_mutex_destroy(&mutex) }

    public func withLock<R>(_ body: () throws -> R) rethrows -> R {
        pthread_mutex_lock(&mutex)
        defer { pthread_mutex_unlock(&mutex) }
        return try body()
    }
}

private final class ThreadJob {
    let run: () -> Void
    init(_ run: @escaping () -> Void) { self.run = run }
}

/// Runs `body` on a thread of its own; `wait()` blocks until it is done and returns its result (once).
public final class Background<T>: @unchecked Sendable {
    private var result: Result<T, Error>?
    private var thread = pthread_t()
    private var started = false

    public init(_ body: @escaping () throws -> T) {
        let job = ThreadJob { self.result = Result { try body() } }   // released when the thread ends
        let context = Unmanaged.passRetained(job).toOpaque()
        started = pthread_create(&thread, nil, { raw in
            Unmanaged<ThreadJob>.fromOpaque(raw!).takeRetainedValue().run()
            return nil
        }, context) == 0
        if !started {
            Unmanaged<ThreadJob>.fromOpaque(context).release()
            result = Result { try body() }   // no thread: run here
        }
    }

    public func wait() throws -> T {
        if started {
            pthread_join(thread, nil)
            started = false
        }
        return try result!.get()
    }
}
