/**
 * @file flow_stats.cuh
 * @brief GPU-parallel per-flow statistics for beacon detection.
 *
 * After packets are loaded, group them by TCP/UDP 5-tuple and compute
 * inter-arrival time (IAT) and payload-size statistics per flow.  Flows
 * with low IAT coefficient-of-variation (stddev/mean) are beaconing C2 —
 * the kind of traffic that produces zero PFAC hits because it is TLS-
 * encrypted but that is structurally unmistakable.
 *
 * Pipeline:
 *   PcapData → group_flows_by_5tuple() → FlowGrouping
 *                                              ↓
 *                                   run_flow_stats_gpu()
 *                                              ↓
 *                                  vector<FlowStatResult>
 */

#pragma once
#include "load_pcap.h"
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tunable thresholds
// ---------------------------------------------------------------------------

static constexpr float BEACON_CV_THRESH = 0.15f;  // IAT CV below this → beacon
static constexpr int   MIN_BEACON_PKTS  = 5;       // ignore tiny flows

// ---------------------------------------------------------------------------
// Five-tuple identifying a TCP/UDP flow
// ---------------------------------------------------------------------------

struct FSTuple {
    uint32_t src_ip, dst_ip;
    uint16_t sport, dport;
    uint8_t  proto;

    bool operator==(const FSTuple& o) const {
        return src_ip == o.src_ip && dst_ip == o.dst_ip &&
               sport  == o.sport  && dport  == o.dport  &&
               proto  == o.proto;
    }
};

struct FSTupleHash {
    std::size_t operator()(const FSTuple& t) const noexcept {
        // FNV-1a-inspired mix (same style as FlowKeyHash)
        uint64_t h = 14695981039346656037ULL;
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
        mix(t.src_ip);  mix(t.dst_ip);
        mix((uint32_t)t.sport << 16 | t.dport);
        mix(t.proto);
        return static_cast<std::size_t>(h);
    }
};

// ---------------------------------------------------------------------------
// CPU-side grouping output
// ---------------------------------------------------------------------------

struct FlowGrouping {
    std::vector<int>     packet_idx;   // packet indices, sorted by timestamp within each flow
    std::vector<int>     flow_offsets; // flow_offsets[f] = start in packet_idx; sentinel at end
    std::vector<FSTuple> tuples;       // 5-tuple for flow f
    int                  num_flows;
};

// ---------------------------------------------------------------------------
// Per-flow result
// ---------------------------------------------------------------------------

struct FlowStatResult {
    FSTuple  tuple;
    int      packet_count;
    float    iat_mean_ms;   // mean inter-arrival time in milliseconds
    float    iat_cv;        // coefficient of variation = stddev/mean (0 = perfectly regular)
    float    size_mean;     // mean packet size in bytes
    float    size_stddev;
    bool     is_beacon;     // iat_cv < BEACON_CV_THRESH && packet_count >= MIN_BEACON_PKTS
};

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/**
 * Group packets by 5-tuple.  Parses Ethernet/IP/TCP|UDP headers to extract
 * the flow key; non-IP or non-TCP/UDP packets are silently skipped.
 * Within each flow, packets are sorted by timestamp.
 */
FlowGrouping group_flows_by_5tuple(const PcapData& pcap);

/**
 * Compute per-flow IAT and size statistics on the GPU.
 * Uses one threadblock per flow and warp-shuffle parallel reductions.
 */
void run_flow_stats_gpu(
    const PcapData&              pcap,
    const FlowGrouping&          groups,
    std::vector<FlowStatResult>& out
);

/** Dot-decimal string for an IPv4 address stored in host byte order. */
std::string ipv4_str(uint32_t ip_host);
