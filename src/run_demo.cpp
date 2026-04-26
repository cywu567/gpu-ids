/**
 * @file run_demo.cpp
 * @brief End-to-end CLI demo: load a pcap, match patterns, print alerts.
 *
 * Ties together load_pcap, cpu_matcher, and gpu_matcher into a single binary
 * that reads a capture file, scans it for known-bad patterns, and reports
 * throughput alongside human-readable IDS alerts.
 *
 * Usage:
 *   ./ids --pcap PATH --rules PATH [--cpu | --gpu-naive | --pfac | --both]
 *
 * Output:
 *   Loaded N packets (X MB), M patterns
 *   === CPU results ===
 *   [ALERT] packet #42 | pattern "GET /cgi-bin/"
 *   ...
 *   3 alert(s) total
 *   CPU throughput: 210.4 MB/s (47.3 ms)
 *
 * Exit code 0 on success, 1 on usage or I/O error.
 */

#include "cpu_matcher.h"
#include "flow_stats.cuh"
#include "gpu_matcher.cuh"
#include "load_pcap.h"
#include "web_server.h"
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// ANSI escape codes for terminal color. Gracefully ignored on Windows.
static constexpr const char* RED    = "\033[31m";
static constexpr const char* GREEN  = "\033[32m";
static constexpr const char* YELLOW = "\033[33m";
static constexpr const char* RESET  = "\033[0m";

static double elapsed_ms(
    std::chrono::high_resolution_clock::time_point t0,
    std::chrono::high_resolution_clock::time_point t1
) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/** Walk the hits array and print one line per match. Returns total alert count. */
template<typename T>
static int print_alerts(const T* hits, int num_packets, const PatternSet& ps) {
    int total = 0;
    for (int p = 0; p < num_packets; p++) {
        for (int r = 0; r < ps.num_patterns; r++) {
            if (hits[p * ps.num_patterns + r]) {
                std::cout << RED << "[ALERT]" << RESET
                          << " packet #" << p
                          << " | pattern \"" << ps.labels[r] << "\"\n";
                total++;
            }
        }
    }
    std::cout << YELLOW << total << " alert(s) total" << RESET << "\n";
    return total;
}

int main(int argc, char** argv) {
    std::string pcap_path, rules_path;
    bool do_cpu = false, do_gpu_naive = false, do_pfac = false;
    bool do_flow_stats = false;
    bool do_web = false;
    int  web_port = 8080;

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--pcap")        && i + 1 < argc) pcap_path  = argv[++i];
        if (!std::strcmp(argv[i], "--rules")       && i + 1 < argc) rules_path = argv[++i];
        if (!std::strcmp(argv[i], "--cpu"))         do_cpu = true;
        if (!std::strcmp(argv[i], "--gpu-naive"))   do_gpu_naive = true;
        if (!std::strcmp(argv[i], "--pfac"))        do_pfac = true;
        if (!std::strcmp(argv[i], "--both"))        { do_cpu = true; do_gpu_naive = true; }
        if (!std::strcmp(argv[i], "--flow-stats"))  do_flow_stats = true;
        if (!std::strcmp(argv[i], "--web")) {
            do_web = true;
            if (i + 1 < argc && std::isdigit((unsigned char)argv[i + 1][0]))
                web_port = std::stoi(argv[++i]);
        }
    }

    // Web dashboard mode: serve the interactive UI, ignore --cpu/--gpu flags.
    if (do_web) {
        std::cout << "CudaShield dashboard at http://localhost:" << web_port << "\n"
                  << "Public URL: https://cudashield.tech\n"
                  << "Press Ctrl+C to stop.\n";
        run_web_server(web_port, pcap_path, rules_path);
        return 0;
    }

    if (pcap_path.empty() || rules_path.empty() || (!do_cpu && !do_gpu_naive && !do_pfac && !do_flow_stats)) {
        std::cerr << "Usage:\n"
                  << "  ids --pcap PATH --rules PATH [--cpu | --gpu-naive | --pfac | --both] [--flow-stats]\n"
                  << "  ids --web [PORT] [--pcap PATH] [--rules PATH]\n";
        return 1;
    }

    PcapData   pcap = load_pcap(pcap_path);
    PatternSet ps   = load_patterns(rules_path);

    double mb = static_cast<double>(pcap.bytes.size()) / 1e6;
    std::cout << "Scanning " << pcap.num_packets << " packets ("
              << mb << " MB), " << ps.num_patterns << " patterns\n\n";

    if (ps.num_patterns > 1024) {
        std::cerr << "Warning: " << ps.num_patterns
                  << " patterns exceeds 1024 (GPU naive kernel limit).\n"
                  << "Use --pfac or --cpu instead.\n";
        if (do_gpu_naive) return 1;
    }

    std::vector<int>     hits(pcap.num_packets * ps.num_patterns, 0);
    std::vector<uint8_t> pfac_hits(pcap.num_packets * ps.num_patterns, 0);

    // -- CPU --
    if (do_cpu) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_cpu_matcher(pcap.bytes.data(), pcap.offsets.data(),
                        pcap.num_packets, ps, hits.data());
        auto t1  = std::chrono::high_resolution_clock::now();
        double ms   = elapsed_ms(t0, t1);
        double mbps = mb / (ms / 1e3);

        std::cout << "=== CPU results ===\n";
        print_alerts(hits.data(), pcap.num_packets, ps);
        std::cout << GREEN << "CPU throughput: " << mbps << " MB/s"
                  << RESET << " (" << ms << " ms)\n\n";
    }

    // -- GPU naive --
    if (do_gpu_naive) {
        std::fill(hits.begin(), hits.end(), 0);

        auto t0 = std::chrono::high_resolution_clock::now();
        run_naive_match_gpu(
            pcap.bytes.data(),  pcap.offsets.data(),  pcap.num_packets,
            ps.bytes.data(),    ps.offsets.data(),    ps.num_patterns,
            hits.data(),
            pcap.bytes.size(),  ps.bytes.size()
        );
        auto t1  = std::chrono::high_resolution_clock::now();
        double ms   = elapsed_ms(t0, t1);
        double mbps = mb / (ms / 1e3);

        std::cout << "=== GPU (naive) results ===\n";
        print_alerts(hits.data(), pcap.num_packets, ps);
        std::cout << GREEN << "GPU naive throughput: " << mbps << " MB/s"
                  << RESET << " (" << ms << " ms)\n\n";
    }

    // -- Flow statistics / beacon detection --
    if (do_flow_stats) {
        auto t0 = std::chrono::high_resolution_clock::now();
        FlowGrouping groups = group_flows_by_5tuple(pcap);
        std::vector<FlowStatResult> fstats;
        run_flow_stats_gpu(pcap, groups, fstats);

        std::sort(fstats.begin(), fstats.end(), [](const auto& a, const auto& b) {
            return a.packet_count > b.packet_count;
        });
        for (int i = 0; i < std::min(15, (int)fstats.size()); i++) {
            const auto& f = fstats[i];
            printf("  %u.%u.%u.%u -> %u.%u.%u.%u:%u  pkts=%d  iat_cv=%.4f  iat_mean=%.1fms\n",
                (f.tuple.src_ip>>24)&0xFF, (f.tuple.src_ip>>16)&0xFF,
                (f.tuple.src_ip>>8)&0xFF,  f.tuple.src_ip&0xFF,
                (f.tuple.dst_ip>>24)&0xFF, (f.tuple.dst_ip>>16)&0xFF,
                (f.tuple.dst_ip>>8)&0xFF,  f.tuple.dst_ip&0xFF,
                f.tuple.dport, f.packet_count, f.iat_cv, f.iat_mean_ms);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        int beacon_count = 0;
        for (const auto& f : fstats) if (f.is_beacon) ++beacon_count;

        std::cout << "=== Flow statistics (beacon detection) ===\n";
        std::cout << groups.num_flows << " flows analyzed, "
                  << beacon_count << " beacon(s) flagged ("
                  << elapsed_ms(t0, t1) << " ms)\n\n";

        if (beacon_count > 0) {
            std::printf("  %-18s %-18s %5s %7s %9s %8s %9s\n",
                "Src", "Dst", "Port", "Pkts", "IAT mean", "IAT CV", "Size avg");
            std::printf("  %s\n", std::string(76, '-').c_str());
            for (const auto& f : fstats) {
                if (!f.is_beacon) continue;
                char src[18], dst[18];
                std::snprintf(src, sizeof(src), "%u.%u.%u.%u",
                    (f.tuple.src_ip>>24)&0xFF, (f.tuple.src_ip>>16)&0xFF,
                    (f.tuple.src_ip>> 8)&0xFF,  f.tuple.src_ip     &0xFF);
                std::snprintf(dst, sizeof(dst), "%u.%u.%u.%u",
                    (f.tuple.dst_ip>>24)&0xFF, (f.tuple.dst_ip>>16)&0xFF,
                    (f.tuple.dst_ip>> 8)&0xFF,  f.tuple.dst_ip     &0xFF);
                char iat_buf[16];
                if (f.iat_mean_ms >= 1000.f)
                    std::snprintf(iat_buf, sizeof(iat_buf), "%.1f s", f.iat_mean_ms/1000.f);
                else
                    std::snprintf(iat_buf, sizeof(iat_buf), "%.0f ms", f.iat_mean_ms);
                char row[128];
                std::snprintf(row, sizeof(row),
                    "  %-18s %-18s %5u %7d %9s %8.3f %9.0f B",
                    src, dst, f.tuple.dport, f.packet_count,
                    iat_buf, f.iat_cv, f.size_mean);
                std::cout << RED << row << RESET << "\n";
            }
            std::cout << "\n";
        }
    }

    // -- GPU PFAC (Aho-Corasick) --
    if (do_pfac) {
        PfacDfa dfa = build_pfac_dfa(ps);

        auto t0 = std::chrono::high_resolution_clock::now();
        run_pfac_match_gpu(
            pcap.bytes.data(), pcap.offsets.data(), pcap.num_packets,
            dfa, pfac_hits.data(), ps.num_patterns, pcap.bytes.size()
        );
        auto t1  = std::chrono::high_resolution_clock::now();
        double ms   = elapsed_ms(t0, t1);
        double mbps = mb / (ms / 1e3);

        std::cout << "=== GPU (PFAC) results ===\n";
        print_alerts(pfac_hits.data(), pcap.num_packets, ps);
        std::cout << GREEN << "GPU PFAC throughput: " << mbps << " MB/s"
                  << RESET << " (" << ms << " ms)\n\n";
    }

    return 0;
}
