import CRTCompute

/// A structured buffer in device memory.
public final class DeviceBuffer {
    let handle: rtc_buffer
    public let device: ComputeDevice
    public let byteCount: Int
    public let elementStride: Int

    init(device: ComputeDevice, byteCount: Int, elementStride: Int, initialData: UnsafeRawPointer?) throws {
        var handle: rtc_buffer?
        try check(rtc_buffer_create(device.handle, UInt64(byteCount), UInt32(elementStride), initialData, &handle),
                  "creating buffer of \(byteCount) bytes")
        self.handle = handle!
        self.device = device
        self.byteCount = byteCount
        self.elementStride = elementStride
    }

    deinit {
        rtc_buffer_destroy(handle)
    }

    public func upload<T: BitwiseCopyable>(_ values: [T], byteOffset: Int = 0) throws {
        try values.withUnsafeBytes { bytes in
            try check(rtc_buffer_upload(handle, UInt64(byteOffset), UInt64(bytes.count), bytes.baseAddress),
                      "uploading \(bytes.count) bytes")
        }
    }

    /// Waits for pending work, then reads `count` elements (default: the whole buffer).
    public func download<T: BitwiseCopyable>(as type: T.Type, count: Int? = nil, byteOffset: Int = 0) throws -> [T] {
        let count = count ?? (byteCount - byteOffset) / MemoryLayout<T>.stride
        return try [T](unsafeUninitializedCapacity: count) { storage, initialized in
            try check(rtc_buffer_download(handle, UInt64(byteOffset), UInt64(count * MemoryLayout<T>.stride),
                                          UnsafeMutableRawPointer(storage.baseAddress!)),
                      "downloading \(count) elements")
            initialized = count
        }
    }
}
