/**
 * @file flow_reassembly.cuh
 * @brief TCP flow reassembly interface.
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
 * == Implementation ==
 *
 * CPU path:
 *   std::unordered_map<FlowKey, FlowBuffer> as the flow table.
 *   FlowBuffer holds a gap-map (std::map<uint32_t, bytes>) for out-of-order
 *   segments and next_seq for the next expected byte.
 *   On FIN/RST the completed stream is returned as a flat byte buffer.
 *
 * GPU path:
 *   Open-addressing hash table (GpuFlowSlot[]) on device.
 *   One warp per segment: lane 0 drives atomicCAS linear probe;
 *   all 32 lanes cooperate to copy payload into the per-flow buffer slot
 *   indexed by (seq - init_seq) — works for both in-order and out-of-order.
 *   Prefix-sum compaction (mark_complete_kernel -> thrust::exclusive_scan ->
 *   compact_flows_kernel) gathers finished flows into a contiguous buffer
 *   that is fed directly into run_pfac_match_gpu.
 *
 *   Known limitation: each flow's payload buffer is pre-allocated at a fixed
 *   capacity (MAX_FLOW_BYTES). Flows exceeding this are truncated.
 */

#pragma once
#include <cstdint>
#include <map>
#include <vector>

// ---------------------------------------------------------------------------
// Five-tuple + per-flow state
// ---------------------------------------------------------------------------

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
 * data holds in-order stream bytes assembled so far.
 * pending buffers out-of-order segments keyed by their sequence number.
 * next_seq is the sequence number of the next expected byte.
 * seq_initialized is false until the first segment arrives.
 */
struct FlowBuffer {
    FlowKey              key;
    std::vector<uint8_t> data;
    uint32_t             next_seq        = 0;
    bool                 seq_initialized = false;
    std::map<uint32_t, std::vector<uint8_t>> pending;  // seq -> payload bytes
};

// ---------------------------------------------------------------------------
// Parsed TCP segment (CPU helper)
// ---------------------------------------------------------------------------

/**
 * A single TCP segment parsed from a raw Ethernet/IP/TCP frame.
 * valid == false means the frame was not IPv4/TCP or was malformed.
 */
struct TcpSegment {
    FlowKey        key;
    const uint8_t* payload;  // points into the original raw frame buffer
    int            len;
    uint32_t       seq;
    bool           fin;      // true if FIN or RST flag is set
    bool           valid;
};

/**
 * Parse a raw Ethernet/IPv4/TCP frame into a TcpSegment.
 *
 * Handles 802.1Q VLAN tags (EtherType 0x8100). Sets valid=false for any
 * non-IPv4, non-TCP, or truncated packet. The payload pointer points into
 * raw_pkt so the caller must not free raw_pkt while using the TcpSegment.
 */
TcpSegment parse_tcp_segment(const uint8_t* raw_pkt, int pkt_len);

// ---------------------------------------------------------------------------
// CPU-path reassembly API
// ---------------------------------------------------------------------------

/**
 * Insert a TCP segment into the per-flow reassembly table.
 *
 * Handles out-of-order delivery: if seq > next_seq the segment is buffered.
 * Handles retransmits: if seq < next_seq, the already-seen prefix is skipped.
 *
 * @param key      Five-tuple for the flow.
 * @param payload  Raw TCP payload bytes for this segment.
 * @param len      Payload byte count.
 * @param seq      TCP sequence number of payload[0].
 * @param fin      True if FIN or RST was seen — flush and remove this flow.
 * @return         Pointer to the completed FlowBuffer if fin==true,
 *                 else nullptr.
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
 * Used at end-of-capture for flows that never received FIN/RST.
 * Clears the internal flow table.
 */
std::vector<FlowBuffer> flush_all_flows();

// ---------------------------------------------------------------------------
// GPU-path structs and API
// ---------------------------------------------------------------------------

/** Maximum payload bytes stored per flow in GPU memory. */
static constexpr int MAX_FLOW_BYTES = 128 * 1024;  // 128 KB per flow

/**
 * One slot in the GPU open-addressing hash table.
 * Layout is packed to minimize wasted space in the device hash table.
 */
struct GpuFlowSlot {
    // Flow key fields (duplicated from FlowKey for direct device access)
    uint32_t src_ip,  dst_ip;
    uint16_t src_port, dst_port;
    uint8_t  proto;
    uint8_t  _pad[3];
    // Slot state (0 = empty, 1 = active; claimed via atomicCAS on lane 0)
    int      state;
    // Reassembly bookkeeping
    uint32_t init_seq;      // seq number of the first byte we stored
    int      buf_offset;    // byte offset into d_payload_buf
    int      buf_capacity;  // MAX_FLOW_BYTES
    int      fin_seen;      // set to 1 atomically when FIN/RST arrives
};

/**
 * Host-side handle for the GPU flow table allocation.
 * All d_* fields are device pointers.
 */
struct GpuFlowTable {
    GpuFlowSlot* d_slots;
    uint8_t*     d_payload_buf;
    int*         d_payload_buf_top;  // atomically-allocated byte cursor
    int          table_size;         // must be a power of two
    int          payload_buf_size;
};

/** Allocate device memory for a GPU flow table. */
GpuFlowTable alloc_gpu_flow_table(int max_flows, int max_total_payload_bytes);

/** Free all device memory for a GPU flow table. */
void free_gpu_flow_table(GpuFlowTable& t);

/**
 * GPU-accelerated TCP flow reassembly.
 *
 * Parses TCP segments from the raw input buffer on the CPU, uploads segment
 * descriptors to the GPU, runs flow_insert_kernel (warp-level CAS probing +
 * cooperative payload copy), then compact_flows_kernel (prefix-sum
 * compaction) to collect finished flows.
 *
 * Output vectors are formatted identically to PcapData: out_bytes is a flat
 * concatenated buffer of reassembled streams, out_offsets[i] is the start
 * of stream i, and out_offsets[num_flows] is out_bytes.size() (sentinel).
 *
 * @param h_input    Raw packet bytes (all packets concatenated).
 * @param h_offsets  h_offsets[i] = start of packet i in h_input.
 *                   h_offsets[num_packets] = total bytes (sentinel).
 * @param num_packets
 * @param out_bytes     Reassembled stream bytes (host output).
 * @param out_offsets   Stream start offsets + sentinel (host output).
 */
void run_gpu_reassembly(
    const uint8_t*         h_input,
    const int*             h_offsets,
    int                    num_packets,
    std::vector<uint8_t>&  out_bytes,
    std::vector<int>&      out_offsets
);
