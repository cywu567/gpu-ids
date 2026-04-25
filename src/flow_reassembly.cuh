/**
 * @file flow_reassembly.cuh
 * @brief TCP flow reassembly interface (class project Week 3 -- stub only).
 *
 * A TCP flow is identified by its 5-tuple: (src IP, dst IP, src port, dst
 * port, protocol). The reassembler tracks per-flow state, buffers
 * out-of-order segments, and stitches them into the in-order application
 * stream before the stream is handed to the GPU pattern matcher.
 *
 * This is important for IDS correctness: an attacker can split a malicious
 * payload across multiple TCP segments, each of which is individually
 * innocent. Without reassembly the matcher misses it; with reassembly it
 * sees the full stream and fires.
 *
 * == Implementation plan (Week 3) ==
 *
 * CPU path (Week 3a):
 *   - std::unordered_map<FlowKey, FlowBuffer> as the flow table.
 *   - FlowBuffer holds a gap-list of received segments and the next expected
 *     sequence number.
 *   - On FIN/RST the completed stream is returned to the caller as a flat
 *     byte buffer suitable for passing to run_naive_match_gpu or run_pfac_match_gpu.
 *
 * GPU path (Week 3b):
 *   - Lock-free open-addressing hash table on device for parallel insertions.
 *   - One warp per flow insert/lookup.
 *   - Parallel prefix-sum compaction to gather completed flow buffers into a
 *     contiguous batch for the GPU matcher.
 *
 * NOT IMPLEMENTED. Stubs declared here so the rest of the build compiles and
 * the interface is settled before implementation begins.
 */

#pragma once
#include <cstdint>
#include <vector>

/** Five-tuple identifying a TCP/UDP flow. */
struct FlowKey {
    uint32_t src_ip,  dst_ip;
    uint16_t src_port, dst_port;
    uint8_t  proto;

    bool operator==(const FlowKey& o) const;
};

/** Hash support for FlowKey so it can be used in std::unordered_map. */
struct FlowKeyHash {
    std::size_t operator()(const FlowKey& k) const noexcept;
};

/**
 * Per-flow reassembly state.
 *
 * data holds the in-order stream bytes assembled so far. Gaps (missing
 * segments) are represented as zero-filled holes; the gap-list tracks which
 * ranges have actually been received. next_seq is the sequence number of the
 * next byte we expect to append without a gap.
 */
struct FlowBuffer {
    FlowKey              key;
    std::vector<uint8_t> data;      // reassembled in-order stream bytes
    uint32_t             next_seq;  // next expected TCP sequence number
};

/**
 * Insert a TCP segment into the per-flow reassembly table.
 *
 * Handles out-of-order delivery: if seq > next_seq the segment is buffered.
 * If seq <= next_seq it is appended (or partially appended if it overlaps).
 *
 * @param key      Five-tuple for the flow.
 * @param payload  Raw TCP payload bytes for this segment.
 * @param len      Payload byte count.
 * @param seq      TCP sequence number of payload[0].
 * @param fin      True if FIN or RST was seen -- flush and remove this flow.
 * @return         Pointer to the completed FlowBuffer if fin==true and the
 *                 flow is now fully reassembled, else nullptr.
 *                 The pointer is invalidated by the next call to this function.
 */
FlowBuffer* reassemble_segment(
    const FlowKey& key,
    const uint8_t* payload,
    int            len,
    uint32_t       seq,
    bool           fin
);

/**
 * Flush all open flows and return their (possibly incomplete) buffers.
 *
 * Used at end-of-capture to process trailing flows that never received FIN.
 * Clears the internal flow table.
 */
std::vector<FlowBuffer> flush_all_flows();
