/**
 * @file flow_reassembly.cu
 * @brief TCP flow reassembly implementation (class project Week 3 -- stubs only).
 *
 * CPU path (Week 3a):
 *   Maintain an std::unordered_map<FlowKey, FlowBuffer> as the flow table.
 *   reassemble_segment inserts segments in sequence-number order, buffers
 *   out-of-order arrivals, and returns a completed stream on FIN/RST.
 *
 * GPU path (Week 3b):
 *   Port the hot-path hash lookups and buffer compaction to CUDA.
 *   Parallel open-addressing table with warp-level CAS for insertion.
 *   Parallel prefix-sum to compact complete flow buffers before batching
 *   them into run_pfac_match_gpu.
 *
 * Both paths share the FlowKey / FlowBuffer types defined in the header.
 * Start with the CPU path; once it's correct, profile to find the bottleneck
 * before deciding what to port to GPU.
 */

#include "flow_reassembly.cuh"
#include <functional>
#include <unordered_map>

// ---------------------------------------------------------------------------
// FlowKey equality and hash
// ---------------------------------------------------------------------------

bool FlowKey::operator==(const FlowKey& o) const {
    return src_ip   == o.src_ip   &&
           dst_ip   == o.dst_ip   &&
           src_port == o.src_port &&
           dst_port == o.dst_port &&
           proto    == o.proto;
}

std::size_t FlowKeyHash::operator()(const FlowKey& k) const noexcept {
    // Simple FNV-1a-inspired mix; replace with a proper hash in Week 3a.
    std::size_t h = 2166136261u;
    auto mix = [&](std::size_t v) { h ^= v; h *= 16777619u; };
    mix(k.src_ip);
    mix(k.dst_ip);
    mix((uint32_t)k.src_port << 16 | k.dst_port);
    mix(k.proto);
    return h;
}

// ---------------------------------------------------------------------------
// Flow table (CPU path stub)
// ---------------------------------------------------------------------------

static std::unordered_map<FlowKey, FlowBuffer, FlowKeyHash> g_flows;
static FlowBuffer g_last_completed; // scratch buffer returned to caller

FlowBuffer* reassemble_segment(
    const FlowKey& key,
    const uint8_t* /*payload*/,
    int            /*len*/,
    uint32_t       /*seq*/,
    bool           fin
) {
    // TODO (Week 3a): implement in-order assembly and out-of-order buffering.
    // Skeleton:
    //   auto& buf = g_flows[key];
    //   buf.key = key;
    //   if (seq == buf.next_seq) { append payload; buf.next_seq += len; }
    //   else { stash in gap buffer; }
    //   if (fin) { g_last_completed = std::move(buf); g_flows.erase(key);
    //              return &g_last_completed; }
    (void)fin;
    return nullptr;
}

std::vector<FlowBuffer> flush_all_flows() {
    // TODO (Week 3a): collect all open flows and clear the table.
    std::vector<FlowBuffer> out;
    for (auto& [key, buf] : g_flows)
        out.push_back(std::move(buf));
    g_flows.clear();
    return out;
}

// ---------------------------------------------------------------------------
// TODO (Week 3b): GPU hash table kernel declarations
// ---------------------------------------------------------------------------
// __global__ void flow_insert_kernel(...) { ... }
// __global__ void flow_compact_kernel(...) { ... }
