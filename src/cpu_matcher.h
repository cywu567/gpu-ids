/**
 * @file cpu_matcher.h
 * @brief Reference single-threaded CPU pattern matcher.
 *
 * Provides the correctness baseline that the GPU kernel must agree with.
 * Every hit the GPU reports should appear in the CPU hit array and vice
 * versa. Any disagreement is a bug in the GPU kernel, not the CPU code.
 *
 * The hackathon weekend implementation uses memmem (POSIX) for simplicity:
 * for each packet, for each pattern, search once and record a hit.
 * Runtime is O(num_packets * num_patterns * avg_packet_len) -- intentionally
 * naive so the GPU speedup story is honest.
 *
 * Class project Week 1: replace with a proper Aho-Corasick CPU implementation
 * (or wrap the ac-automata library) so the CPU baseline is competitive and
 * the comparison against Hyperscan is fair.
 */

#pragma once
#include <cstdint>
#include <string>
#include <vector>

/**
 * Flat pattern database loaded from a rules file.
 *
 * All pattern bytes are stored in a single contiguous buffer (bytes)
 * with an offset array for slicing, mirroring the PcapData layout so
 * the same pointer+offset idiom works for both the CPU and GPU matchers.
 */
struct PatternSet {
    std::vector<uint8_t>     bytes;       // flat concatenated pattern bytes
    std::vector<int>         offsets;     // offsets[i] = start of pattern i; offsets[n] = total
    std::vector<std::string> labels;      // human-readable string for each pattern (for alert output)
    int                      num_patterns;
};

/**
 * Load patterns from a plain-text rules file.
 *
 * File format: one pattern string per line. Lines that are empty or start
 * with '#' are skipped. Leading/trailing whitespace is preserved -- a
 * pattern that starts with a space matches a leading space in packets.
 *
 * @param path  Path to the rules file (e.g. patterns/rules.txt).
 * @return      Populated PatternSet ready to pass to run_cpu_matcher or
 *              run_naive_match_gpu.
 * @throws      std::runtime_error if the file cannot be opened.
 */
PatternSet load_patterns(const std::string& path);

/**
 * Run the CPU matcher on a batch of packets.
 *
 * Writes into a caller-allocated hits array of size
 * (num_packets * ps.num_patterns). Entry hits[p * num_patterns + r] is set
 * to 1 if pattern r was found anywhere inside packet p, else 0.
 *
 * @param input           Flat packet bytes (PcapData::bytes.data()).
 * @param packet_offsets  Offset array (PcapData::offsets.data()), length num_packets+1.
 * @param num_packets     Number of packets in the batch.
 * @param ps              Pattern database to search for.
 * @param hits            Output array; must be zero-initialized by the caller.
 */
void run_cpu_matcher(
    const uint8_t* input,
    const int*     packet_offsets,
    int            num_packets,
    const PatternSet& ps,
    int*           hits
);
