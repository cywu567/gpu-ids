/**
 * @file gpu_matcher.cu
 * @brief GPU pattern-matching kernels and host-side launch code.
 *
 * == Hackathon weekend: naive kernel ==
 *
 * Grid:  blockIdx.x  = packet index (one block per packet)
 * Block: threadIdx.x = pattern index (one thread per pattern)
 *
 * Each thread independently performs a brute-force sliding-window search for
 * its pattern inside its packet, then sets hits[packet * num_patterns + pattern]
 * and returns. No shared memory, no warp-level coordination. Lots of warp
 * divergence when packets have different lengths and patterns match at
 * different positions. This is fine -- it's the punching bag we'll improve.
 *
 * Constraint: num_patterns <= 1024 (CUDA max threads per block). Fine for
 * the weekend's ~50-pattern rule set. The PFAC kernel removes this limit.
 *
 * == Class project Week 2: PFAC kernel (TODO) ==
 *
 * Replace the naive kernel with Parallel Failureless Aho-Corasick:
 *   - Pre-build a DFA (state * 256 transition table) from all patterns.
 *   - One thread per (packet, start_offset): thread t starts at byte t of
 *     the flattened input and walks the DFA forward until a dead state.
 *   - No failure links needed at runtime -- hence "failureless."
 * Reference: Lin, Liu, Chang (2013) "Accelerating Pattern Matching Using a
 * Novel Parallel Algorithm on GPUs." IEEE Transactions on Computers.
 *
 * == Class project Week 3: memory layout optimization (TODO) ==
 *
 * The DFA table is (num_states * 256 * 4) bytes -- potentially megabytes.
 * It doesn't fit in shared memory. Candidates:
 *   a) Global memory (baseline, high latency)
 *   b) Texture cache (hardware caching, 2D spatial locality)
 *   c) __ldg / read-only cache (simpler, often equivalent on Maxwell+)
 * Profile with Nsight Compute to pick the winner for your GPU.
 */

#include "gpu_matcher.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Naive kernel
// ---------------------------------------------------------------------------

/**
 * Brute-force multi-pattern search kernel.
 *
 * Each thread owns exactly one (packet, pattern) pair. It scans the packet
 * bytes with a sliding window and writes 1 to hits on the first match found.
 * Early return avoids redundant work after the first hit, but threads in the
 * same warp will still diverge if they match at different positions.
 */
__global__ void naive_match_kernel(
    const char* input,
    const int*  packet_offsets,
    int         num_packets,
    const char* patterns,
    const int*  pattern_offsets,
    int         num_patterns,
    int*        hits
) {
    int packet_idx  = blockIdx.x;
    int pattern_idx = threadIdx.x;
    if (packet_idx >= num_packets || pattern_idx >= num_patterns) return;

    const char* pkt     = input    + packet_offsets[packet_idx];
    int         pkt_len = packet_offsets[packet_idx + 1] - packet_offsets[packet_idx];
    const char* pat     = patterns + pattern_offsets[pattern_idx];
    int         pat_len = pattern_offsets[pattern_idx + 1] - pattern_offsets[pattern_idx];

    if (pat_len == 0 || pat_len > pkt_len) return;

    for (int i = 0; i + pat_len <= pkt_len; i++) {
        bool match = true;
        for (int j = 0; j < pat_len; j++) {
            if (pkt[i + j] != pat[j]) { match = false; break; }
        }
        if (match) {
            hits[packet_idx * num_patterns + pattern_idx] = 1;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher: naive kernel
// ---------------------------------------------------------------------------

void run_naive_match_gpu(
    const uint8_t* h_input,
    const int*     h_offsets,
    int            num_packets,
    const uint8_t* h_patterns,
    const int*     h_pat_offsets,
    int            num_patterns,
    int*           h_hits,
    size_t         input_len,
    size_t         patterns_len
) {
    char *d_input, *d_patterns;
    int  *d_offsets, *d_pat_offsets, *d_hits;

    cudaMalloc(&d_input,       input_len);
    cudaMalloc(&d_offsets,     (num_packets  + 1) * sizeof(int));
    cudaMalloc(&d_patterns,    patterns_len);
    cudaMalloc(&d_pat_offsets, (num_patterns + 1) * sizeof(int));
    cudaMalloc(&d_hits,        (size_t)num_packets * num_patterns * sizeof(int));

    cudaMemcpy(d_input,       h_input,       input_len,                        cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets,     h_offsets,     (num_packets  + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns,    h_patterns,    patterns_len,                     cudaMemcpyHostToDevice);
    cudaMemcpy(d_pat_offsets, h_pat_offsets, (num_patterns + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hits, 0, (size_t)num_packets * num_patterns * sizeof(int));

    naive_match_kernel<<<num_packets, num_patterns>>>(
        d_input, d_offsets, num_packets,
        d_patterns, d_pat_offsets, num_patterns,
        d_hits
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_hits, d_hits,
               (size_t)num_packets * num_patterns * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_offsets);
    cudaFree(d_patterns);
    cudaFree(d_pat_offsets);
    cudaFree(d_hits);
}

// ---------------------------------------------------------------------------
// TODO (Week 2): PFAC DFA builder (host-side)
// ---------------------------------------------------------------------------

PfacDfa build_pfac_dfa(const PatternSet& ps) {
    const int DEAD = 0;
    const int ROOT = 1;

    // Build using int internally for ease, convert to uint16_t at the end.
    std::vector<int> tmp_table(2 * 256, DEAD);
    std::vector<int> tmp_accepting(2, 0);
    int num_states = 2;
    int next_state = 2;

    for (int p = 0; p < ps.num_patterns; p++) {
        int cur = ROOT;
        const uint8_t* pat     = ps.bytes.data() + ps.offsets[p];
        int            pat_len = ps.offsets[p + 1] - ps.offsets[p];

        for (int i = 0; i < pat_len; i++) {
            unsigned char c   = pat[i];
            int           nxt = tmp_table[cur * 256 + c];

            if (nxt == DEAD) {
                tmp_table.resize((size_t)(next_state + 1) * 256, DEAD);
                tmp_accepting.resize(next_state + 1, 0);
                tmp_table[cur * 256 + c] = next_state;
                nxt = next_state++;
                num_states = next_state;
            }
            cur = nxt;
        }

        if (tmp_accepting[cur] == 0)
            tmp_accepting[cur] = p + 1;
    }

    PfacDfa dfa;
    dfa.num_states = num_states;
    dfa.accepting  = tmp_accepting;
    // Convert transition table to uint16_t (max 65535 states, well within range).
    dfa.table.resize(tmp_table.size());
    for (size_t i = 0; i < tmp_table.size(); i++)
        dfa.table[i] = static_cast<uint16_t>(tmp_table[i]);
    return dfa;
}

// ---------------------------------------------------------------------------
// PFAC kernel
// ---------------------------------------------------------------------------

/**
 * One block per packet, 256 threads per block.
 * Each thread owns start positions threadIdx.x, threadIdx.x+256, ... within the packet.
 * Walks the DFA forward from each start until hitting dead state 0.
 *
 * Optimization: DFA table is uint16_t (144KB vs 288KB for int), fits better in L2.
 * __ldg() routes reads through the read-only cache.
 */
__global__ void pfac_kernel(
    const uint8_t*  __restrict__ input,
    const int*      __restrict__ packet_offsets,
    int                          num_packets,
    const uint16_t* __restrict__ dfa_table,
    const int*      __restrict__ accepting,
    uint8_t*                     hits,
    int                          num_patterns
) {
    int packet_idx = blockIdx.x;
    if (packet_idx >= num_packets) return;

    int pkt_start = packet_offsets[packet_idx];
    int pkt_len   = packet_offsets[packet_idx + 1] - pkt_start;

    for (int start = (int)threadIdx.x; start < pkt_len; start += (int)blockDim.x) {
        int state = 1;   // root
        for (int pos = start; pos < pkt_len; pos++) {
            unsigned char c = input[pkt_start + pos];
            state = (int)__ldg(&dfa_table[state * 256 + c]);
            if (state == 0) break;

            int pat = __ldg(&accepting[state]);
            if (pat != 0)
                hits[packet_idx * num_patterns + (pat - 1)] = 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher: PFAC kernel
// ---------------------------------------------------------------------------

void run_pfac_match_gpu(
    const uint8_t* h_input,
    const int*     h_offsets,
    int            num_packets,
    const PfacDfa& dfa,
    uint8_t*       h_hits,
    int            num_patterns,
    size_t         input_len
) {
    uint8_t*  d_input;
    int       *d_offsets, *d_accepting;
    uint16_t* d_dfa_table;
    uint8_t*  d_hits;

    cudaMalloc(&d_input,     input_len);
    cudaMalloc(&d_offsets,   (num_packets + 1) * sizeof(int));
    cudaMalloc(&d_dfa_table, (size_t)dfa.num_states * 256 * sizeof(uint16_t));
    cudaMalloc(&d_accepting, (size_t)dfa.num_states * sizeof(int));
    cudaMalloc(&d_hits,      (size_t)num_packets * num_patterns * sizeof(uint8_t));

    cudaMemcpy(d_input,     h_input,              input_len,                                       cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets,   h_offsets,            (num_packets + 1) * sizeof(int),                 cudaMemcpyHostToDevice);
    cudaMemcpy(d_dfa_table, dfa.table.data(),     (size_t)dfa.num_states * 256 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accepting, dfa.accepting.data(), (size_t)dfa.num_states * sizeof(int),            cudaMemcpyHostToDevice);
    cudaMemset(d_hits, 0,   (size_t)num_packets * num_patterns * sizeof(uint8_t));

    const int THREADS = 256;
    pfac_kernel<<<num_packets, THREADS>>>(
        d_input, d_offsets, num_packets,
        d_dfa_table, d_accepting,
        d_hits, num_patterns
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_hits, d_hits,
               (size_t)num_packets * num_patterns * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_offsets);
    cudaFree(d_dfa_table);
    cudaFree(d_accepting);
    cudaFree(d_hits);
}
