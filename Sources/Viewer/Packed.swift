import Core

// "Packed" binary payload for the browser: [u32 header length][JSON header][pad to 4][blocks].
// The header lists blocks as {name, type: "f32"|"i32", count, offset} with offsets relative to the
// start of the block area, so the client can wrap each in a typed array without copying.

struct PackedPayload {
    private var blocks: [(name: String, type: String, bytes: [UInt8], count: Int)] = []
    var fields: [(String, JSON)] = []

    mutating func add(_ name: String, floats: [Float]) {
        blocks.append((name, "f32", floats.withUnsafeBytes { Array($0) }, floats.count))
    }

    mutating func add(_ name: String, ints: [Int32]) {
        blocks.append((name, "i32", ints.withUnsafeBytes { Array($0) }, ints.count))
    }

    func encode() -> [UInt8] {
        var offset = 0
        var descriptors: [JSON] = []
        for block in blocks {
            descriptors.append(.object([("name", .string(block.name)), ("type", .string(block.type)),
                                        ("count", .number(Double(block.count))), ("offset", .number(Double(offset)))]))
            offset += block.bytes.count
        }
        var header = Array(JSON.object(fields + [("blocks", .array(descriptors))]).text.utf8)
        while (4 + header.count) % 4 != 0 { header.append(UInt8(ascii: " ")) }
        var out: [UInt8] = []
        out.reserveCapacity(4 + header.count + offset)
        let length = UInt32(header.count)
        out += [UInt8(length & 0xFF), UInt8((length >> 8) & 0xFF), UInt8((length >> 16) & 0xFF), UInt8(length >> 24)]
        out += header
        for block in blocks { out += block.bytes }
        return out
    }

    var response: HTTPResponse { HTTPResponse(contentType: "application/octet-stream", body: encode()) }
}
