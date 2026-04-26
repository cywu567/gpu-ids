/**
 * @file load_pcap.h
 * @brief libpcap-based packet loader.
 *
 * Reads a pcap file and produces a flat byte buffer of raw packet data
 * plus an offset array so callers can slice out individual packets by index.
 *
 * The output format is intentionally simple: the whole pcap becomes one
 * contiguous allocation, which is what both the CPU matcher (random access)
 * and the GPU kernel launcher (single cudaMemcpy) need. No protocol parsing
 * is done here -- all packet bytes including Ethernet/IP/TCP headers are
 * kept as-is and searched verbatim. For TCP stream-level matching,
 * flow_reassembly.cu groups packets by 5-tuple before they reach the matcher.
 */

#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct PcapData {
    std::vector<uint8_t>  bytes;         // flat concatenated packet bytes (all packets back-to-back)
    std::vector<int>      offsets;       // offsets[i] = start of packet i; offsets[num_packets] = total size
    std::vector<uint64_t> timestamps_us; // epoch microseconds per packet; same indexing as offsets
    int                   num_packets = 0;
};

/**
 * Load all packets from a pcap file into a flat buffer.
 *
 * Iterates the file with pcap_loop and appends each captured packet to
 * a growing byte vector. The sentinel offset at index num_packets allows
 * computing any packet's length as offsets[i+1] - offsets[i].
 *
 * @param path  Path to a .pcap or .pcapng file.
 * @return      Populated PcapData; bytes and offsets are valid until the
 *              struct is destroyed.
 * @throws      std::runtime_error if the file cannot be opened or parsed.
 */
PcapData load_pcap(const std::string& path);
