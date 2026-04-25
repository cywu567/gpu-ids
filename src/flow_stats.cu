/**
 * @file flow_stats.cu
 * @brief GPU-parallel flow statistics kernel for beacon detection.
 *
 * GPU kernel: one threadblock per flow.  Each thread loads one packet's IAT
 * and size, then two tree reductions (sum, sum-of-squares) via __shfl_down_sync
 * compute mean and variance across the warp and across the block using shared
 * memory.  Handles flows with up to BLOCK_SIZE packets; larger flows are
 * processed in a strided loop before the reduction.
 */

#include "flow_stats.cuh"

#include <arpa/inet.h>   // ntohs / ntohl
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

// ---------------------------------------------------------------------------
// Header layout (same as flow_reassembly.cu — replicated to stay standalone)
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct EthHdr  { uint8_t  dst[6], src[6]; uint16_t ethertype; };
struct IpHdr   { uint8_t  ver_ihl, tos; uint16_t total_len, id, frag;
                 uint8_t  ttl, proto; uint16_t checksum;
                 uint32_t src, dst; };
struct TcpHdr  { uint16_t sport, dport; uint32_t seq, ack;
                 uint8_t  data_off, flags; uint16_t window, checksum, urg; };
struct UdpHdr  { uint16_t sport, dport, length, checksum; };
#pragma pack(pop)

static constexpr uint16_t ETH_IPV4 = 0x0800;

// ---------------------------------------------------------------------------
// group_flows_by_5tuple  (CPU)
// ---------------------------------------------------------------------------

FlowGrouping group_flows_by_5tuple(const PcapData& pcap) {
    std::unordered_map<FSTuple, std::vector<int>, FSTupleHash> table;

    for (int i = 0; i < pcap.num_packets; ++i) {
        const uint8_t* p   = pcap.bytes.data() + pcap.offsets[i];
        int            len = pcap.offsets[i + 1] - pcap.offsets[i];

        if (len < (int)(sizeof(EthHdr) + sizeof(IpHdr))) continue;

        const auto* eth = reinterpret_cast<const EthHdr*>(p);
        if (ntohs(eth->ethertype) != ETH_IPV4)           continue;

        const auto* ip = reinterpret_cast<const IpHdr*>(p + sizeof(EthHdr));
        int ip_hdr_len = (ip->ver_ihl & 0x0F) * 4;
        if ((ip->ver_ihl >> 4) != 4)                     continue;

        FSTuple key{};
        key.src_ip = ntohl(ip->src);
        key.dst_ip = ntohl(ip->dst);
        key.proto  = ip->proto;

        int transport_off = (int)sizeof(EthHdr) + ip_hdr_len;
        if (ip->proto == 6 && len >= transport_off + (int)sizeof(TcpHdr)) {
            const auto* tcp = reinterpret_cast<const TcpHdr*>(p + transport_off);
            key.sport = ntohs(tcp->sport);
            key.dport = ntohs(tcp->dport);
        } else if (ip->proto == 17 && len >= transport_off + (int)sizeof(UdpHdr)) {
            const auto* udp = reinterpret_cast<const UdpHdr*>(p + transport_off);
            key.sport = ntohs(udp->sport);
            key.dport = ntohs(udp->dport);
        } else {
            continue;
        }

        table[key].push_back(i);
    }

    FlowGrouping g;
    g.num_flows = 0;
    for (auto& [key, indices] : table) {
        // Sort by arrival time
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return pcap.timestamps_us[a] < pcap.timestamps_us[b];
        });
        g.tuples.push_back(key);
        g.flow_offsets.push_back((int)g.packet_idx.size());
        for (int idx : indices) g.packet_idx.push_back(idx);
        ++g.num_flows;
    }
    g.flow_offsets.push_back((int)g.packet_idx.size()); // sentinel
    return g;
}

// ---------------------------------------------------------------------------
// GPU kernel internals
// ---------------------------------------------------------------------------

static constexpr int BLOCK_SIZE = 256;

// Device-side per-flow result (plain POD for cudaMemcpy)
struct FlowStatGpu {
    float iat_mean_us;
    float iat_var_us;
    float size_mean;
    float size_var;
    int   pkt_count;
};

/**
 * Reduce a value across a full warp using shuffle-down.
 * All 32 lanes must participate.
 */
__device__ static float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    return v;
}

/**
 * flow_stats_kernel — one block per flow.
 *
 * Each block computes mean and variance of IATs (in µs) and packet sizes
 * for its assigned flow using shared-memory + warp-shuffle two-pass reduction.
 *
 * Flows with < 2 packets produce iat_mean=0, iat_var=0 (no IAT defined).
 */
__global__ void flow_stats_kernel(
    const uint64_t* __restrict__ timestamps_us,  // [total_packets]
    const int*      __restrict__ sizes,           // [total_packets]  caplen per packet
    const int*      __restrict__ flat_idx,        // [sum of per-flow packet counts]
    const int*      __restrict__ flow_offsets,    // [num_flows+1]
    int                          num_flows,
    FlowStatGpu*                 out              // [num_flows]
) {
    int flow_id = blockIdx.x;
    if (flow_id >= num_flows) return;

    int f_start = flow_offsets[flow_id];
    int f_end   = flow_offsets[flow_id + 1];
    int n       = f_end - f_start;

    // --- Shared memory reductions ---
    __shared__ float sh_iat_sum[BLOCK_SIZE];
    __shared__ float sh_iat_sq [BLOCK_SIZE];
    __shared__ float sh_sz_sum [BLOCK_SIZE];
    __shared__ float sh_sz_sq  [BLOCK_SIZE];

    float l_iat_sum = 0.f, l_iat_sq = 0.f;
    float l_sz_sum  = 0.f, l_sz_sq  = 0.f;

    // Each thread accumulates a stripe; handles flows > BLOCK_SIZE via loop
    for (int t = threadIdx.x; t < n; t += BLOCK_SIZE) {
        int pkt_i = flat_idx[f_start + t];
        float sz  = (float)sizes[pkt_i];
        l_sz_sum += sz;
        l_sz_sq  += sz * sz;

        if (t > 0) {
            int pkt_prev = flat_idx[f_start + t - 1];
            // Guard: prev may have been loaded by a different iteration of the
            // outer loop on a different thread — we only compute the IAT for
            // positions where (t-1) belongs to *this* thread's stripe OR we
            // re-read the previous packet's timestamp (cheap global read).
            float iat = (float)((int64_t)timestamps_us[pkt_i]
                              - (int64_t)timestamps_us[pkt_prev]);
            if (iat > 0.f) {           // skip out-of-order edge cases
                l_iat_sum += iat;
                l_iat_sq  += iat * iat;
            }
        }
    }

    sh_iat_sum[threadIdx.x] = l_iat_sum;
    sh_iat_sq [threadIdx.x] = l_iat_sq;
    sh_sz_sum [threadIdx.x] = l_sz_sum;
    sh_sz_sq  [threadIdx.x] = l_sz_sq;
    __syncthreads();

    // Tree reduction over shared memory
    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sh_iat_sum[threadIdx.x] += sh_iat_sum[threadIdx.x + s];
            sh_iat_sq [threadIdx.x] += sh_iat_sq [threadIdx.x + s];
            sh_sz_sum [threadIdx.x] += sh_sz_sum [threadIdx.x + s];
            sh_sz_sq  [threadIdx.x] += sh_sz_sq  [threadIdx.x + s];
        }
        __syncthreads();
    }

    // Final warp reduction (warp 0 only)
    if (threadIdx.x < 32) {
        float iat_s  = warp_reduce_sum(sh_iat_sum[threadIdx.x]);
        float iat_sq = warp_reduce_sum(sh_iat_sq [threadIdx.x]);
        float sz_s   = warp_reduce_sum(sh_sz_sum [threadIdx.x]);
        float sz_sq  = warp_reduce_sum(sh_sz_sq  [threadIdx.x]);

        if (threadIdx.x == 0) {
            int n_iat = (n > 1) ? (n - 1) : 1; // number of IAT samples
            float iat_mean = iat_s  / (float)n_iat;
            float sz_mean  = sz_s   / (float)n;

            // Variance = E[x²] - E[x]²  (biased, fine for flagging)
            float iat_var = (iat_sq / (float)n_iat) - iat_mean * iat_mean;
            float sz_var  = (sz_sq  / (float)n)     - sz_mean  * sz_mean;

            out[flow_id].iat_mean_us = iat_mean;
            out[flow_id].iat_var_us  = (iat_var > 0.f) ? iat_var : 0.f;
            out[flow_id].size_mean   = sz_mean;
            out[flow_id].size_var    = (sz_var  > 0.f) ? sz_var  : 0.f;
            out[flow_id].pkt_count   = n;
        }
    }
}

// ---------------------------------------------------------------------------
// run_flow_stats_gpu  (host launcher)
// ---------------------------------------------------------------------------

void run_flow_stats_gpu(
    const PcapData&              pcap,
    const FlowGrouping&          groups,
    std::vector<FlowStatResult>& out
) {
    out.clear();
    if (groups.num_flows == 0) return;

    int N = pcap.num_packets;

    // Build per-packet sizes array from offsets
    std::vector<int> sizes(N);
    for (int i = 0; i < N; ++i)
        sizes[i] = pcap.offsets[i + 1] - pcap.offsets[i];

    // Upload inputs
    uint64_t* d_ts      = nullptr;
    int*      d_sizes   = nullptr;
    int*      d_flat    = nullptr;
    int*      d_foff    = nullptr;
    FlowStatGpu* d_out  = nullptr;

    int F = groups.num_flows;
    int P = (int)groups.packet_idx.size();

    cudaMalloc(&d_ts,    N * sizeof(uint64_t));
    cudaMalloc(&d_sizes, N * sizeof(int));
    cudaMalloc(&d_flat,  P * sizeof(int));
    cudaMalloc(&d_foff, (F + 1) * sizeof(int));
    cudaMalloc(&d_out,   F * sizeof(FlowStatGpu));
    cudaMemset(d_out, 0,  F * sizeof(FlowStatGpu));

    cudaMemcpy(d_ts,    pcap.timestamps_us.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, sizes.data(),              N * sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_flat,  groups.packet_idx.data(),  P * sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_foff,  groups.flow_offsets.data(),(F + 1) * sizeof(int),cudaMemcpyHostToDevice);

    flow_stats_kernel<<<F, BLOCK_SIZE>>>(d_ts, d_sizes, d_flat, d_foff, F, d_out);
    cudaDeviceSynchronize();

    std::vector<FlowStatGpu> h_out(F);
    cudaMemcpy(h_out.data(), d_out, F * sizeof(FlowStatGpu), cudaMemcpyDeviceToHost);

    cudaFree(d_ts); cudaFree(d_sizes); cudaFree(d_flat);
    cudaFree(d_foff); cudaFree(d_out);

    out.resize(F);
    for (int f = 0; f < F; ++f) {
        const FlowStatGpu& g = h_out[f];
        FlowStatResult& r    = out[f];
        r.tuple        = groups.tuples[f];
        r.packet_count = g.pkt_count;
        r.iat_mean_ms  = g.iat_mean_us / 1000.f;
        r.size_mean    = g.size_mean;
        r.size_stddev  = std::sqrt(g.size_var);

        float iat_stddev = std::sqrt(g.iat_var_us);
        r.iat_cv = (g.iat_mean_us > 0.f)
                 ? (iat_stddev / g.iat_mean_us)
                 : 1.f;    // undefined → treat as non-beacon

        r.is_beacon = (r.iat_cv < BEACON_CV_THRESH) &&
                      (r.packet_count >= MIN_BEACON_PKTS);
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

std::string ipv4_str(uint32_t ip_host) {
    char buf[18];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u",
        (ip_host >> 24) & 0xFF, (ip_host >> 16) & 0xFF,
        (ip_host >>  8) & 0xFF,  ip_host        & 0xFF);
    return buf;
}
