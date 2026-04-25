/**
 * @file benchmark.cpp
 * @brief Standalone throughput benchmarking harness for CPU vs. GPU matching.
 *
 * Runs both matchers for a configurable number of iterations and writes
 * timing results to stdout and optionally to a CSV file for gnuplot charting.
 * The benchmark binary is separate from the demo binary so the demo CLI stays
 * clean and the benchmark can be scripted without the alert printing overhead.
 *
 * Usage:
 *   ./benchmark --pcap PATH --rules PATH [--iters N] [--csv out.csv]
 *
 * CSV columns: mode, iteration, elapsed_ms, throughput_mbps
 *
 * To generate a chart from the CSV (requires gnuplot):
 *   gnuplot -e "
 *     set terminal png; set output 'results/speedup_chart.png';
 *     set title 'CPU vs GPU throughput (MB/s)';
 *     plot 'results/benchmark.csv' using 1:4 with linespoints
 *   "
 *
 * == Class project extensions ==
 * Week 2: add a --pfac mode that times the Aho-Corasick kernel, including
 *         the one-time DFA build cost reported separately.
 * Week 4: add --hyperscan mode (conditional on Hyperscan being available at
 *         link time) for the head-to-head comparison in the report.
 */

#include "cpu_matcher.h"
#include "gpu_matcher.cuh"
#include "load_pcap.h"
#ifdef HAVE_HYPERSCAN
#  include <hs/hs.h>
#endif
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static double elapsed_ms(
    std::chrono::high_resolution_clock::time_point t0,
    std::chrono::high_resolution_clock::time_point t1
) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static void print_row(std::ostream& csv, const std::string& mode, int iter,
                      double ms, double mbps) {
    std::cout << std::left  << std::setw(10) << mode
              << " iter=" << iter
              << "  " << std::fixed << std::setprecision(2) << ms << " ms"
              << "  " << mbps << " MB/s\n";
    if (csv)
        csv << mode << "," << iter << "," << ms << "," << mbps << "\n";
}

int main(int argc, char** argv) {
    std::string pcap_path, rules_path, csv_path;
    int  iters    = 5;
    bool run_pfac     = false;
    bool run_hyperscan = false;

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--pcap")  && i + 1 < argc) pcap_path  = argv[++i];
        if (!std::strcmp(argv[i], "--rules") && i + 1 < argc) rules_path = argv[++i];
        if (!std::strcmp(argv[i], "--csv")   && i + 1 < argc) csv_path   = argv[++i];
        if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters       = std::stoi(argv[++i]);
        if (!std::strcmp(argv[i], "--pfac"))       run_pfac      = true;
        if (!std::strcmp(argv[i], "--hyperscan"))  run_hyperscan = true;
    }

    if (pcap_path.empty() || rules_path.empty()) {
        std::cerr << "Usage: benchmark --pcap PATH --rules PATH [--iters N] [--csv out.csv] [--pfac] [--hyperscan]\n";
        return 1;
    }

    PcapData   pcap = load_pcap(pcap_path);
    PatternSet ps   = load_patterns(rules_path);
    double     mb   = static_cast<double>(pcap.bytes.size()) / 1e6;

    std::cout << "Loaded " << pcap.num_packets << " packets ("
              << mb << " MB), " << ps.num_patterns << " patterns\n"
              << "Running " << iters << " iteration(s) per mode.\n\n";

    std::vector<int>     hits(pcap.num_packets * ps.num_patterns, 0);
    std::vector<uint8_t> pfac_hits(pcap.num_packets * ps.num_patterns, 0);

    std::ofstream csv_file;
    if (!csv_path.empty()) {
        csv_file.open(csv_path);
        csv_file << "mode,iteration,elapsed_ms,throughput_mbps\n";
    }

    // -- CPU --
    std::cout << "--- CPU ---\n";
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_cpu_matcher(pcap.bytes.data(), pcap.offsets.data(),
                        pcap.num_packets, ps, hits.data());
        auto t1  = std::chrono::high_resolution_clock::now();
        print_row(csv_file, "cpu", it, elapsed_ms(t0, t1),
                  mb / (elapsed_ms(t0, t1) / 1e3));
    }

    // -- GPU naive --
    std::cout << "\n--- GPU naive ---\n";
    for (int it = 0; it < iters; it++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_naive_match_gpu(
            pcap.bytes.data(),  pcap.offsets.data(),     pcap.num_packets,
            ps.bytes.data(),    ps.offsets.data(),        ps.num_patterns,
            hits.data(),
            pcap.bytes.size(),  ps.bytes.size()
        );
        auto t1  = std::chrono::high_resolution_clock::now();
        print_row(csv_file, "gpu_naive", it, elapsed_ms(t0, t1),
                  mb / (elapsed_ms(t0, t1) / 1e3));
    }

    // -- GPU PFAC --
    if (run_pfac) {
        std::cout << "\n--- GPU PFAC (Aho-Corasick) ---\n";

        auto tb0 = std::chrono::high_resolution_clock::now();
        PfacDfa dfa = build_pfac_dfa(ps);
        auto tb1 = std::chrono::high_resolution_clock::now();
        std::cout << "DFA build: " << elapsed_ms(tb0, tb1) << " ms, "
                  << dfa.num_states << " states, "
                  << (dfa.num_states * 256 * 2 / 1024) << " KB table\n";

        for (int it = 0; it < iters; it++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            run_pfac_match_gpu(
                pcap.bytes.data(), pcap.offsets.data(), pcap.num_packets,
                dfa, pfac_hits.data(), ps.num_patterns, pcap.bytes.size()
            );
            auto t1 = std::chrono::high_resolution_clock::now();
            print_row(csv_file, "gpu_pfac", it, elapsed_ms(t0, t1),
                      mb / (elapsed_ms(t0, t1) / 1e3));
        }
    }

    // -- Hyperscan / Vectorscan --
#ifdef HAVE_HYPERSCAN
    if (run_hyperscan) {
        std::cout << "\n--- Hyperscan/Vectorscan (SIMD CPU) ---\n";

        // Build the Hyperscan database from our pattern set (one-time cost).
        // Use hs_compile_lit_multi so patterns are treated as raw byte strings,
        // not regexes -- no need to escape special characters.
        auto tb0 = std::chrono::high_resolution_clock::now();
        std::vector<const char*> pats;
        std::vector<size_t>      pat_lens;
        std::vector<unsigned>    ids, flags;
        for (int r = 0; r < ps.num_patterns; r++) {
            pats.push_back(ps.labels[r].c_str());
            pat_lens.push_back(ps.labels[r].size());
            ids.push_back(static_cast<unsigned>(r));
            flags.push_back(HS_FLAG_CASELESS | HS_FLAG_SINGLEMATCH);
        }
        hs_database_t* db   = nullptr;
        hs_compile_error_t* err = nullptr;
        if (hs_compile_lit_multi(pats.data(), flags.data(), ids.data(),
                                 pat_lens.data(),
                                 static_cast<unsigned>(pats.size()),
                                 HS_MODE_BLOCK, nullptr, &db, &err)
            != HS_SUCCESS) {
            std::cerr << "Hyperscan compile error: " << err->message << "\n";
            hs_free_compile_error(err);
        } else {
            auto   tb1    = std::chrono::high_resolution_clock::now();
            std::cout << "DB compile: " << elapsed_ms(tb0, tb1) << " ms\n";

            hs_scratch_t* scratch = nullptr;
            hs_alloc_scratch(db, &scratch);

            // Callback just counts hits; we don't need per-hit detail for throughput.
            struct HsCtx { int count; };
            auto on_match = [](unsigned /*id*/, unsigned long long /*from*/,
                               unsigned long long /*to*/, unsigned /*flags*/,
                               void* ctx) -> int {
                static_cast<HsCtx*>(ctx)->count++;
                return 0;
            };

            for (int it = 0; it < iters; it++) {
                HsCtx ctx{0};
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int p = 0; p < pcap.num_packets; p++) {
                    const char* pkt = reinterpret_cast<const char*>(
                        pcap.bytes.data() + pcap.offsets[p]);
                    unsigned pkt_len = static_cast<unsigned>(
                        pcap.offsets[p + 1] - pcap.offsets[p]);
                    hs_scan(db, pkt, pkt_len, 0, scratch, on_match, &ctx);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                print_row(csv_file, "hyperscan", it, elapsed_ms(t0, t1),
                          mb / (elapsed_ms(t0, t1) / 1e3));
            }
            hs_free_scratch(scratch);
            hs_free_database(db);
        }
    }
#else
    if (run_hyperscan)
        std::cerr << "Hyperscan/Vectorscan not compiled in (rebuild with vectorscan in conda env).\n";
#endif

    // TODO (Week 4): add Hyperscan mode

    return 0;
}
