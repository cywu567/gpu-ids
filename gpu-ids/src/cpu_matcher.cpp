/**
 * @file cpu_matcher.cpp
 * @brief Implementation of the reference single-threaded CPU pattern matcher.
 *
 * load_patterns: reads the rules file line-by-line, skipping blank lines and
 * comments, and packs each pattern string into a flat byte buffer.
 *
 * run_cpu_matcher: outer loop over packets, inner loop over patterns. Uses
 * memmem(3) for each (packet, pattern) pair. memmem is POSIX and available
 * on Linux and macOS; on Windows, replace with a manual two-pointer search.
 *
 * This is the punching bag. Its job is to be correct and easy to verify, not
 * fast. The throughput number it produces is the baseline we need to beat.
 */

#include "cpu_matcher.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

PatternSet load_patterns(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open rules file: " + path);

    PatternSet ps;
    ps.num_patterns = 0;

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        ps.offsets.push_back(static_cast<int>(ps.bytes.size()));
        ps.bytes.insert(ps.bytes.end(), line.begin(), line.end());
        ps.labels.push_back(line);
        ps.num_patterns++;
    }
    ps.offsets.push_back(static_cast<int>(ps.bytes.size())); // sentinel

    return ps;
}

void run_cpu_matcher(
    const uint8_t* input,
    const int*     packet_offsets,
    int            num_packets,
    const PatternSet& ps,
    int*           hits
) {
    for (int p = 0; p < num_packets; p++) {
        const uint8_t* pkt     = input + packet_offsets[p];
        int            pkt_len = packet_offsets[p + 1] - packet_offsets[p];

        for (int r = 0; r < ps.num_patterns; r++) {
            const uint8_t* pat     = ps.bytes.data() + ps.offsets[r];
            int            pat_len = ps.offsets[r + 1] - ps.offsets[r];

            hits[p * ps.num_patterns + r] = 0;
            if (pat_len == 0 || pat_len > pkt_len) continue;

            if (memmem(pkt, pkt_len, pat, pat_len))
                hits[p * ps.num_patterns + r] = 1;
        }
    }
}
