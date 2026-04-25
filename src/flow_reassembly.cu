/**
 * @file flow_reassembly.cu
 * @brief TCP flow reassembly — CPU path (Week 3a) + GPU kernels (Week 3b).
 *
 * CPU path:
 *   parse_tcp_segment() cracks Ethernet/IP/TCP headers.
 *   reassemble_segment() maintains a per-flow unordered_map, handles
 *   out-of-order delivery via a pending segment map, and returns a completed
 *   FlowBuffer on FIN/RST.
 *
 * GPU path:
 *   Segment descriptors are built on CPU (reusing parse_tcp_segment) then
 *   uploaded to device.
 *   flow_insert_kernel: one warp per segment, lane 0 drives atomicCAS linear
 *   probe to claim/find a GpuFlowSlot, all 32 lanes cooperate to write the
 *   payload at buf_offset + (seq - init_seq).
 *   mark_complete_kernel: flags slots where fin_seen == 1.
 *   compact_flows_kernel: scatter complete flows into contiguous output buffer
 *   using prefix-sum offsets from thrust::exclusive_scan.
 */

#include "flow_reassembly.cuh"
#include <arpa/inet.h>  // ntohs, ntohl
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <stdexcept>
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
    std::size_t h = 2166136261u;
    auto mix = [&](std::size_t v) { h ^= v; h *= 16777619u; };
    mix(k.src_ip);
    mix(k.dst_ip);
    mix((uint32_t)k.src_port << 16 | k.dst_port);
    mix(k.proto);
    return h;
}

// ---------------------------------------------------------------------------
// Ethernet/IP/TCP header layout (host-side parsing only)
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct EthHdr {
    uint8_t  dst[6], src[6];
    uint16_t ethertype;
};
struct VlanHdr {
    uint16_t tci;
    uint16_t ethertype;
};
struct IpHdr {
    uint8_t  ver_ihl, tos;
    uint16_t total_len, id, frag_off;
    uint8_t  ttl, proto;
    uint16_t checksum;
    uint32_t src, dst;
};
struct TcpHdr {
    uint16_t sport, dport;
    uint32_t seq, ack;
    uint8_t  data_off, flags;
    uint16_t window, checksum, urg;
};
#pragma pack(pop)

static constexpr uint8_t  PROTO_TCP   = 6;
static constexpr uint16_t ET_IPV4     = 0x0800;
static constexpr uint16_t ET_VLAN     = 0x8100;
static constexpr uint8_t  TCP_FIN     = 0x01;
static constexpr uint8_t  TCP_RST     = 0x04;

// ---------------------------------------------------------------------------
// parse_tcp_segment
// ---------------------------------------------------------------------------

TcpSegment parse_tcp_segment(const uint8_t* raw, int pkt_len) {
    TcpSegment seg{};
    seg.valid = false;

    if (pkt_len < (int)sizeof(EthHdr)) return seg;

    const auto* eth = reinterpret_cast<const EthHdr*>(raw);
    uint16_t    et  = ntohs(eth->ethertype);
    int         off = sizeof(EthHdr);

    // Strip one level of 802.1Q VLAN tag
    if (et == ET_VLAN) {
        if (pkt_len < off + (int)sizeof(VlanHdr)) return seg;
        const auto* vlan = reinterpret_cast<const VlanHdr*>(raw + off);
        et  = ntohs(vlan->ethertype);
        off += sizeof(VlanHdr);
    }

    if (et != ET_IPV4) return seg;
    if (pkt_len < off + (int)sizeof(IpHdr)) return seg;

    const auto* ip     = reinterpret_cast<const IpHdr*>(raw + off);
    int         ip_hdr = (ip->ver_ihl & 0x0F) * 4;
    if (ip_hdr < (int)sizeof(IpHdr)) return seg;
    if (ip->proto != PROTO_TCP) return seg;

    off += ip_hdr;
    if (pkt_len < off + (int)sizeof(TcpHdr)) return seg;

    const auto* tcp     = reinterpret_cast<const TcpHdr*>(raw + off);
    int         tcp_hdr = (tcp->data_off >> 4) * 4;
    if (tcp_hdr < (int)sizeof(TcpHdr)) return seg;

    off += tcp_hdr;
    int payload_len = pkt_len - off;
    if (payload_len < 0) return seg;

    seg.key.src_ip   = ntohl(ip->src);
    seg.key.dst_ip   = ntohl(ip->dst);
    seg.key.src_port = ntohs(tcp->sport);
    seg.key.dst_port = ntohs(tcp->dport);
    seg.key.proto    = PROTO_TCP;
    seg.seq          = ntohl(tcp->seq);
    seg.fin          = (tcp->flags & (TCP_FIN | TCP_RST)) != 0;
    seg.payload      = raw + off;
    seg.len          = payload_len;
    seg.valid        = true;
    return seg;
}

// ---------------------------------------------------------------------------
// CPU path: flow table
// ---------------------------------------------------------------------------

static std::unordered_map<FlowKey, FlowBuffer, FlowKeyHash> g_flows;
static FlowBuffer g_last_completed;

FlowBuffer* reassemble_segment(
    const FlowKey& key,
    const uint8_t* payload,
    int            len,
    uint32_t       seq,
    bool           fin
) {
    auto& buf = g_flows[key];
    buf.key = key;

    if (!buf.seq_initialized) {
        buf.seq_initialized = true;
        // SYN/ACK packets have len==0 but consume one sequence number.
        // Set next_seq past it so the first data segment is seen as in-order.
        buf.next_seq = (len == 0) ? seq + 1 : seq;
    }

    if (len > 0) {
        if (seq == buf.next_seq) {
            // In-order: append immediately
            buf.data.insert(buf.data.end(), payload, payload + len);
            buf.next_seq += (uint32_t)len;

            // Drain any now-consecutive pending segments
            for (;;) {
                auto it = buf.pending.find(buf.next_seq);
                if (it == buf.pending.end()) break;
                buf.next_seq += (uint32_t)it->second.size();
                buf.data.insert(buf.data.end(),
                                it->second.begin(), it->second.end());
                buf.pending.erase(it);
            }
        } else if (seq > buf.next_seq) {
            // Out-of-order: stash for later
            buf.pending[seq] = std::vector<uint8_t>(payload, payload + len);
        } else {
            // seq < next_seq: retransmit or overlap; skip seen bytes
            uint32_t seen = buf.next_seq - seq;
            if (seen < (uint32_t)len) {
                uint32_t new_len = (uint32_t)len - seen;
                buf.data.insert(buf.data.end(),
                                payload + seen, payload + len);
                buf.next_seq += new_len;
            }
        }
    }

    if (fin) {
        g_last_completed = std::move(buf);
        g_flows.erase(key);
        return &g_last_completed;
    }
    return nullptr;
}

std::vector<FlowBuffer> flush_all_flows() {
    std::vector<FlowBuffer> out;
    out.reserve(g_flows.size());
    for (auto& [key, buf] : g_flows)
        out.push_back(std::move(buf));
    g_flows.clear();
    return out;
}

// ---------------------------------------------------------------------------
// GPU path: flow table allocation
// ---------------------------------------------------------------------------

GpuFlowTable alloc_gpu_flow_table(int max_flows, int max_total_payload_bytes) {
    GpuFlowTable t{};
    t.table_size        = max_flows;        // caller must pass power-of-two
    t.payload_buf_size  = max_total_payload_bytes;

    cudaMalloc(&t.d_slots,           (size_t)max_flows * sizeof(GpuFlowSlot));
    cudaMalloc(&t.d_payload_buf,     (size_t)max_total_payload_bytes);
    cudaMalloc(&t.d_payload_buf_top, sizeof(int));

    cudaMemset(t.d_slots,           0, (size_t)max_flows * sizeof(GpuFlowSlot));
    cudaMemset(t.d_payload_buf,     0, (size_t)max_total_payload_bytes);
    cudaMemset(t.d_payload_buf_top, 0, sizeof(int));
    return t;
}

void free_gpu_flow_table(GpuFlowTable& t) {
    cudaFree(t.d_slots);
    cudaFree(t.d_payload_buf);
    cudaFree(t.d_payload_buf_top);
    t = {};
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

/**
 * Compute a 32-bit hash of a FlowKey for the open-addressing table.
 * FNV-1a-inspired; same mix as FlowKeyHash on the CPU side.
 */
__device__ static unsigned int flow_hash_gpu(
    uint32_t src_ip, uint32_t dst_ip,
    uint16_t sport,  uint16_t dport,
    uint8_t  proto
) {
    unsigned int h = 2166136261u;
    auto mix = [&](unsigned int v) { h ^= v; h *= 16777619u; };
    mix(src_ip);
    mix(dst_ip);
    mix((unsigned int)sport << 16 | dport);
    mix(proto);
    return h;
}

/**
 * flow_insert_kernel: one warp (32 threads) per TCP segment.
 *
 * Lane 0 drives atomicCAS linear probing to find/claim the slot for this
 * flow, then broadcasts the slot index. All 32 lanes then cooperate to copy
 * the payload into payload_buf at buf_offset + (seq - init_seq).
 *
 * Segment descriptors are passed as Structure-of-Arrays for coalesced access.
 */
__global__ void flow_insert_kernel(
    const uint8_t* __restrict__ d_input,
    const int*     __restrict__ d_pay_off,   // payload start in d_input per segment
    const int*     __restrict__ d_pay_len,   // payload byte count per segment
    const uint32_t* __restrict__ d_seq,
    const uint32_t* __restrict__ d_src_ip,
    const uint32_t* __restrict__ d_dst_ip,
    const uint16_t* __restrict__ d_sport,
    const uint16_t* __restrict__ d_dport,
    const uint8_t*  __restrict__ d_proto,
    const int*      __restrict__ d_fin,
    int             num_segments,
    GpuFlowSlot*    hash_table,
    int             table_mask,              // table_size - 1
    uint8_t*        payload_buf,
    int*            payload_buf_top,
    int             payload_buf_size
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x & 31;

    if (warp_id >= num_segments) return;

    // Load this segment's metadata (all lanes see the same values via broadcast)
    uint32_t seg_src_ip = d_src_ip[warp_id];
    uint32_t seg_dst_ip = d_dst_ip[warp_id];
    uint16_t seg_sport  = d_sport[warp_id];
    uint16_t seg_dport  = d_dport[warp_id];
    uint8_t  seg_proto  = d_proto[warp_id];
    uint32_t seg_seq    = d_seq[warp_id];
    int      seg_poff   = d_pay_off[warp_id];
    int      seg_plen   = d_pay_len[warp_id];
    int      seg_fin    = d_fin[warp_id];

    // ----- Phase 1: lane 0 claims a slot, broadcasts slot index -----
    int slot_idx = -1;
    if (lane == 0) {
        unsigned int h = flow_hash_gpu(seg_src_ip, seg_dst_ip,
                                       seg_sport, seg_dport, seg_proto)
                         & (unsigned int)table_mask;
        for (int probe = 0; probe <= table_mask; probe++) {
            int idx = (int)((h + (unsigned int)probe) & (unsigned int)table_mask);
            GpuFlowSlot* slot = &hash_table[idx];

            // Try to claim an empty slot
            int prev = atomicCAS(&slot->state, 0, 1);
            if (prev == 0) {
                // We claimed it — initialize
                slot->src_ip      = seg_src_ip;
                slot->dst_ip      = seg_dst_ip;
                slot->src_port    = seg_sport;
                slot->dst_port    = seg_dport;
                slot->proto       = seg_proto;
                slot->init_seq    = seg_seq;
                slot->buf_capacity = MAX_FLOW_BYTES;
                int top = atomicAdd(payload_buf_top, MAX_FLOW_BYTES);
                slot->buf_offset  = (top + MAX_FLOW_BYTES <= payload_buf_size)
                                    ? top : -1;  // -1 = overflow, drop
                slot->fin_seen    = 0;
                slot_idx = idx;
                break;
            }
            // Slot occupied — check if it's our flow
            if (prev == 1 &&
                slot->src_ip   == seg_src_ip &&
                slot->dst_ip   == seg_dst_ip &&
                slot->src_port == seg_sport  &&
                slot->dst_port == seg_dport  &&
                slot->proto    == seg_proto)
            {
                slot_idx = idx;
                break;
            }
        }
    }

    // Broadcast slot index to all lanes
    slot_idx = __shfl_sync(0xFFFFFFFF, slot_idx, 0);
    if (slot_idx < 0) return;  // table full or payload buffer overflow

    GpuFlowSlot* slot = &hash_table[slot_idx];
    int buf_off = slot->buf_offset;
    if (buf_off < 0) return;  // payload buffer overflow for this flow

    // ----- Phase 2: all 32 lanes cooperate to copy payload -----
    if (seg_plen > 0) {
        uint32_t write_base = seg_seq - slot->init_seq;
        // Guard against writing past the allocated capacity
        int can_write = (int)slot->buf_capacity - (int)write_base;
        int to_copy   = min(seg_plen, can_write);

        for (int i = lane; i < to_copy; i += 32) {
            payload_buf[buf_off + write_base + i] = d_input[seg_poff + i];
        }
    }

    // ----- Phase 3: mark FIN -----
    if (seg_fin && lane == 0)
        atomicOr(&slot->fin_seen, 1);
}

/**
 * mark_complete_kernel: for each table slot, write 1 to d_flags[i] if the
 * slot is active and fin_seen, and write the flow's byte count to d_sizes[i].
 */
__global__ void mark_complete_kernel(
    const GpuFlowSlot* __restrict__ slots,
    int    table_size,
    int*   d_flags,
    int*   d_sizes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= table_size) return;

    if (slots[i].state == 1 && slots[i].fin_seen) {
        d_flags[i] = 1;
        // Bytes actually written = min(buf_capacity, seq_span).
        // We store the full capacity as a conservative upper bound; the CPU
        // will trim trailing zeros when it copies back if needed.
        d_sizes[i] = slots[i].buf_capacity;
    } else {
        d_flags[i] = 0;
        d_sizes[i] = 0;
    }
}

/**
 * compact_flows_kernel: scatter completed flows into the contiguous output
 * buffer using exclusive prefix-sum offsets computed by thrust on the host.
 *
 * d_scan[i] = sum of d_sizes[0..i-1] for complete slots (i.e., exclusive scan
 * of d_sizes masked by d_flags).
 */
__global__ void compact_flows_kernel(
    const GpuFlowSlot* __restrict__ slots,
    int             table_size,
    const int*  __restrict__ d_flags,
    const int*  __restrict__ d_scan,          // exclusive prefix-sum of complete sizes
    const uint8_t* __restrict__ payload_buf,
    uint8_t*    out_buf,
    int*        out_offsets,    // out_offsets[j] = start of flow j in out_buf
    int*        out_num_flows   // total completed flows written
) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    if (i >= table_size) return;
    if (!d_flags[i]) return;

    // One warp handles one complete flow
    int flow_j    = d_scan[i];       // output index (prefix sum over flags, not sizes)
    int out_start = d_scan[i];       // reuse — but we need byte offset separately

    // We actually need two separate prefix sums: one over flags (flow index)
    // and one over sizes (byte offset). compact_flows_kernel receives the size
    // prefix sum (d_scan). The flow index is implicit (not needed here because
    // we only fill out_buf bytes; out_offsets is written by the host after copyback).
    const GpuFlowSlot* slot   = &slots[i];
    int                src    = slot->buf_offset;
    int                nbytes = slot->buf_capacity;

    // Write payload bytes — all threads in the warp cooperate
    for (int b = lane; b < nbytes; b += 32)
        out_buf[out_start + b] = payload_buf[src + b];

    // Lane 0 records the byte offset for this flow
    if (lane == 0)
        atomicMax(out_num_flows, flow_j + 1);  // track highest flow index seen
}

// ---------------------------------------------------------------------------
// Host-side GPU reassembly launcher
// ---------------------------------------------------------------------------

void run_gpu_reassembly(
    const uint8_t*        h_input,
    const int*            h_offsets,
    int                   num_packets,
    std::vector<uint8_t>& out_bytes,
    std::vector<int>&     out_offsets
) {
    // ---- Step 1: parse TCP segments on CPU ----
    struct SegDesc {
        uint32_t src_ip, dst_ip;
        uint16_t sport, dport;
        uint8_t  proto;
        uint32_t seq;
        int      payload_off;  // byte offset in h_input
        int      payload_len;
        int      fin;
    };

    std::vector<SegDesc> segs;
    segs.reserve(num_packets);

    for (int i = 0; i < num_packets; i++) {
        int pkt_start = h_offsets[i];
        int pkt_len   = h_offsets[i + 1] - pkt_start;
        TcpSegment s  = parse_tcp_segment(h_input + pkt_start, pkt_len);
        if (!s.valid) continue;

        SegDesc d;
        d.src_ip      = s.key.src_ip;
        d.dst_ip      = s.key.dst_ip;
        d.sport       = s.key.src_port;
        d.dport       = s.key.dst_port;
        d.proto       = s.key.proto;
        d.seq         = s.seq;
        d.payload_off = pkt_start + (int)(s.payload - (h_input + pkt_start));
        d.payload_len = s.len;
        d.fin         = s.fin ? 1 : 0;
        segs.push_back(d);
    }

    int num_segs = (int)segs.size();
    if (num_segs == 0) {
        out_bytes.clear();
        out_offsets = {0};
        return;
    }

    // ---- Step 2: flatten SoA and upload to device ----
    std::vector<uint32_t> h_src_ip(num_segs), h_dst_ip(num_segs), h_seq(num_segs);
    std::vector<uint16_t> h_sport(num_segs), h_dport(num_segs);
    std::vector<uint8_t>  h_proto(num_segs);
    std::vector<int>      h_pay_off(num_segs), h_pay_len(num_segs), h_fin(num_segs);

    for (int i = 0; i < num_segs; i++) {
        h_src_ip[i]  = segs[i].src_ip;
        h_dst_ip[i]  = segs[i].dst_ip;
        h_seq[i]     = segs[i].seq;
        h_sport[i]   = segs[i].sport;
        h_dport[i]   = segs[i].dport;
        h_proto[i]   = segs[i].proto;
        h_pay_off[i] = segs[i].payload_off;
        h_pay_len[i] = segs[i].payload_len;
        h_fin[i]     = segs[i].fin;
    }

    // Upload full input buffer
    size_t   input_len = (size_t)h_offsets[num_packets];
    uint8_t* d_input;
    cudaMalloc(&d_input, input_len);
    cudaMemcpy(d_input, h_input, input_len, cudaMemcpyHostToDevice);

    // Upload SoA segment descriptors
    uint32_t *d_src_ip, *d_dst_ip, *d_seq;
    uint16_t *d_sport, *d_dport;
    uint8_t  *d_proto;
    int      *d_pay_off, *d_pay_len, *d_fin;

    cudaMalloc(&d_src_ip,  num_segs * sizeof(uint32_t));
    cudaMalloc(&d_dst_ip,  num_segs * sizeof(uint32_t));
    cudaMalloc(&d_seq,     num_segs * sizeof(uint32_t));
    cudaMalloc(&d_sport,   num_segs * sizeof(uint16_t));
    cudaMalloc(&d_dport,   num_segs * sizeof(uint16_t));
    cudaMalloc(&d_proto,   num_segs * sizeof(uint8_t));
    cudaMalloc(&d_pay_off, num_segs * sizeof(int));
    cudaMalloc(&d_pay_len, num_segs * sizeof(int));
    cudaMalloc(&d_fin,     num_segs * sizeof(int));

    cudaMemcpy(d_src_ip,  h_src_ip.data(),  num_segs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_ip,  h_dst_ip.data(),  num_segs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq,     h_seq.data(),     num_segs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sport,   h_sport.data(),   num_segs * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dport,   h_dport.data(),   num_segs * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proto,   h_proto.data(),   num_segs * sizeof(uint8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_pay_off, h_pay_off.data(), num_segs * sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_pay_len, h_pay_len.data(), num_segs * sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_fin,     h_fin.data(),     num_segs * sizeof(int),      cudaMemcpyHostToDevice);

    // ---- Step 3: allocate GPU flow table ----
    // Heuristic: at most num_segs distinct flows; cap at 64K flows.
    int max_flows      = 1;
    while (max_flows < num_segs && max_flows < 65536) max_flows <<= 1;
    max_flows <<= 1;  // load factor ~0.5

    int max_payload    = max_flows * MAX_FLOW_BYTES;
    GpuFlowTable ft    = alloc_gpu_flow_table(max_flows, max_payload);

    // ---- Step 4: flow_insert_kernel (one warp per segment) ----
    int warps_per_block = 8;  // 8 warps = 256 threads/block
    int blocks          = (num_segs + warps_per_block - 1) / warps_per_block;

    flow_insert_kernel<<<blocks, warps_per_block * 32>>>(
        d_input,
        d_pay_off, d_pay_len,
        d_seq,
        d_src_ip, d_dst_ip,
        d_sport, d_dport, d_proto,
        d_fin,
        num_segs,
        ft.d_slots, ft.table_size - 1,
        ft.d_payload_buf, ft.d_payload_buf_top,
        ft.payload_buf_size
    );
    cudaDeviceSynchronize();

    // ---- Step 5: mark complete flows ----
    int *d_flags, *d_sizes;
    cudaMalloc(&d_flags, max_flows * sizeof(int));
    cudaMalloc(&d_sizes, max_flows * sizeof(int));

    {
        int tb = 256;
        int gb = (max_flows + tb - 1) / tb;
        mark_complete_kernel<<<gb, tb>>>(ft.d_slots, max_flows, d_flags, d_sizes);
        cudaDeviceSynchronize();
    }

    // ---- Step 6: prefix-sum scan over sizes (for byte offsets in out_buf) ----
    // We run two scans: one over d_flags (to get per-flow output index) and
    // one over d_sizes masked by d_flags (to get byte offsets).
    // Here we compute the masked-size scan: d_scan_bytes[i] = sum of sizes for
    // complete flows before i.
    //
    // Simple approach: multiply flags * sizes, then exclusive_scan.
    int *d_masked_sizes, *d_scan_bytes, *d_scan_flags;
    cudaMalloc(&d_masked_sizes, max_flows * sizeof(int));
    cudaMalloc(&d_scan_bytes,   max_flows * sizeof(int));
    cudaMalloc(&d_scan_flags,   max_flows * sizeof(int));

    // Compute masked sizes (flags[i] * sizes[i]) — tiny kernel inline
    {
        // Use thrust transform for brevity
        thrust::device_ptr<int> tp_flags(d_flags);
        thrust::device_ptr<int> tp_sizes(d_sizes);
        thrust::device_ptr<int> tp_masked(d_masked_sizes);
        thrust::transform(tp_flags, tp_flags + max_flows, tp_sizes,
                          tp_masked,
                          [] __device__ (int f, int s) { return f * s; });

        thrust::exclusive_scan(tp_masked, tp_masked + max_flows,
                               thrust::device_ptr<int>(d_scan_bytes));
        thrust::exclusive_scan(tp_flags, tp_flags + max_flows,
                               thrust::device_ptr<int>(d_scan_flags));
    }

    // Read back total bytes and flow count
    int total_complete_bytes = 0, total_complete_flows = 0;
    {
        int last_flag, last_size, last_scan_b, last_scan_f;
        cudaMemcpy(&last_flag,   d_flags       + max_flows - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_size,   d_sizes       + max_flows - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_scan_b, d_scan_bytes  + max_flows - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_scan_f, d_scan_flags  + max_flows - 1, sizeof(int), cudaMemcpyDeviceToHost);
        total_complete_bytes = last_scan_b + last_flag * last_size;
        total_complete_flows = last_scan_f + last_flag;
    }

    if (total_complete_flows == 0 || total_complete_bytes == 0) {
        // No complete flows
        out_bytes.clear();
        out_offsets = {0};
        goto cleanup;
    }

    // ---- Step 7: compact_flows_kernel ----
    {
        uint8_t* d_out_buf;
        int*     d_out_num;
        cudaMalloc(&d_out_buf, (size_t)total_complete_bytes);
        cudaMalloc(&d_out_num, sizeof(int));
        cudaMemset(d_out_num, 0, sizeof(int));

        int tb = 256;
        int gb = (max_flows + tb - 1) / tb;
        compact_flows_kernel<<<gb, tb>>>(
            ft.d_slots, max_flows,
            d_flags, d_scan_bytes,
            ft.d_payload_buf,
            d_out_buf,
            nullptr,   // out_offsets built on CPU below
            d_out_num
        );
        cudaDeviceSynchronize();

        // Copy reassembled bytes back to host
        out_bytes.resize((size_t)total_complete_bytes);
        cudaMemcpy(out_bytes.data(), d_out_buf,
                   (size_t)total_complete_bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_out_buf);
        cudaFree(d_out_num);
    }

    // Build out_offsets on CPU from d_scan_bytes + d_flags (already on device;
    // copy both arrays to host and iterate).
    {
        std::vector<int> h_scan_b(max_flows), h_scan_f(max_flows),
                         h_flags(max_flows),  h_sizes(max_flows);
        cudaMemcpy(h_scan_b.data(), d_scan_bytes, max_flows * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_scan_f.data(), d_scan_flags, max_flows * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_flags.data(),  d_flags,      max_flows * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sizes.data(),  d_sizes,      max_flows * sizeof(int), cudaMemcpyDeviceToHost);

        out_offsets.resize((size_t)total_complete_flows + 1, 0);
        for (int i = 0; i < max_flows; i++) {
            if (!h_flags[i]) continue;
            int flow_j = h_scan_f[i];
            out_offsets[flow_j]     = h_scan_b[i];
            out_offsets[flow_j + 1] = h_scan_b[i] + h_sizes[i];
        }
        // Ensure sentinel is correct
        out_offsets[total_complete_flows] = total_complete_bytes;
    }

cleanup:
    // Free device resources
    free_gpu_flow_table(ft);
    cudaFree(d_flags);  cudaFree(d_sizes);
    cudaFree(d_masked_sizes); cudaFree(d_scan_bytes); cudaFree(d_scan_flags);
    cudaFree(d_input);
    cudaFree(d_src_ip); cudaFree(d_dst_ip); cudaFree(d_seq);
    cudaFree(d_sport);  cudaFree(d_dport);  cudaFree(d_proto);
    cudaFree(d_pay_off); cudaFree(d_pay_len); cudaFree(d_fin);
}
