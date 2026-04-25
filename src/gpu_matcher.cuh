/**
 * @file gpu_matcher.cuh
 * @brief GPU pattern matcher interface -- shared by host (.cpp) and device (.cu) code.
 *
 * Defines the Hit struct and host-callable launcher functions. Including this
 * header in a plain .cpp file is safe because no CUDA-specific types appear
 * here (the launchers take only standard C++ types).
 *
 * Kernel inventory:
 *
 *   run_naive_match_gpu  (hackathon weekend)
 *     One CUDA block per packet, one thread per pattern. Each thread does a
 *     brute-force sliding-window search. Grid config: <<<num_packets, num_patterns>>>.
 *     num_patterns must be <= 1024 (CUDA max threads per block).
 *     Intentionally unoptimized -- this is the punching bag.
 *
 *   run_pfac_match_gpu   (class project Week 2 + Week 3 -- implemented)
 *     Parallel Failureless Aho-Corasick: one thread per (packet, start_offset)
 *     pair. Each thread walks the DFA forward from its assigned byte position
 *     until it reaches a dead state. Requires a pre-built DFA table.
 *     Week 3: hot DFA states cached in shared memory (up to 48 KB = 96 states),
 *     falling back to __ldg for deep states. Entire DFA fits for typical rule sets.
 *     Reference: Lin, Liu, Chang (2013) IEEE Trans. Computers.
 */

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "cpu_matcher.h"  // for PatternSet

/** A single pattern match: which packet, which pattern. */
struct Hit {
    int packet_idx;
    int pattern_idx;
};

/**
 * Naive GPU matcher.
 *
 * Allocates device memory, copies inputs, launches the kernel, copies the
 * hits array back, and frees device memory before returning. All GPU work
 * is synchronous from the caller's perspective.
 *
 * @param h_input          Flat packet bytes on host (PcapData::bytes.data()).
 * @param h_offsets        Packet offset array on host (length num_packets+1).
 * @param num_packets      Number of packets.
 * @param h_patterns       Flat pattern bytes on host (PatternSet::bytes.data()).
 * @param h_pat_offsets    Pattern offset array on host (length num_patterns+1).
 * @param num_patterns     Number of patterns. Must be <= 1024 this weekend.
 * @param h_hits           Output: caller-allocated int array of size
 *                         (num_packets * num_patterns), zero-initialized on device.
 *                         h_hits[p * num_patterns + r] = 1 on match.
 * @param input_len        Byte size of h_input.
 * @param patterns_len     Byte size of h_patterns.
 */
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
);

// ---------------------------------------------------------------------------
// PFAC Aho-Corasick kernel (class project Week 2)
// ---------------------------------------------------------------------------

/**
 * Trie-DFA built from a PatternSet.
 *
 * State 0 is the dead state -- a thread that transitions here stops.
 * State 1 is the root. All other states are trie nodes.
 *
 * table[s * 256 + c] = next state after consuming byte c from state s.
 *                      0 = dead.
 * accepting[s]       = pattern_idx + 1 if state s is terminal for that pattern,
 *                      0 means not accepting. (1-indexed so 0 = "no match".)
 */
struct PfacDfa {
    std::vector<uint16_t> table;    // [num_states * 256] -- uint16_t halves table size vs int
    std::vector<int>      accepting; // [num_states]
    int                   num_states;
};

/** Build the trie-DFA from a PatternSet. Call once before scanning. */
PfacDfa build_pfac_dfa(const PatternSet& ps);

/**
 * PFAC shared-memory variant: loads the first smem_states DFA rows into
 * shared memory at block startup for ~4-cycle access vs ~200-cycle global.
 * Use run_pfac_match_gpu for the default __ldg path.
 */
void run_pfac_match_gpu_smem(
    const uint8_t* h_input,
    const int*     h_offsets,
    int            num_packets,
    const PfacDfa& dfa,
    uint8_t*       h_hits,
    int            num_patterns,
    size_t         input_len
);

/**
 * PFAC GPU matcher.
 *
 * One CUDA block per packet. Each thread handles one or more starting byte
 * positions within the packet (stride = blockDim.x = 256). Each thread walks
 * the DFA forward until it hits the dead state, recording any matches.
 *
 * No limit on num_patterns (unlike the naive kernel's 1024-thread-per-block cap).
 */
void run_pfac_match_gpu(
    const uint8_t* h_input,
    const int*     h_offsets,
    int            num_packets,
    const PfacDfa& dfa,
    uint8_t*       h_hits,       // uint8_t: 1/4 the PCIe bandwidth of int
    int            num_patterns,
    size_t         input_len
);
// ---------------------------------------------------------------------------
