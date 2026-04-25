/**
 * @file gen_synthetic.cpp
 * @brief Generates a synthetic pcap file with controllable packet content.
 *
 * Writes a mix of benign HTTP-like traffic and packets with injected malicious
 * payloads to a pcap file. Serves two roles in the project:
 *
 *   1. Guaranteed-hit demo: the output is a pcap where you know exactly which
 *      packets should fire alerts. Useful when the real malware capture's
 *      traffic doesn't behave as expected on stage.
 *
 *   2. Benchmark control: fixed packet size distribution and hit rate allow
 *      clean, reproducible throughput charts without variance from capture
 *      content or packet length distribution.
 *
 * Builds raw Ethernet/IPv4/TCP frames by hand (no sockets, no system
 * networking required). IP/TCP checksums are not computed -- pcap replay
 * tools (tcpreplay) don't validate them, and we're only reading the file
 * offline, not replaying it live.
 *
 * Usage:
 *   ./gen_synthetic [output.pcap] [num_benign] [num_malicious] [num_beacon] [beacon_interval_ms]
 * Defaults:
 *   output:              data/synthetic.pcap
 *   num_benign:          10000
 *   num_malicious:       50
 *   num_beacon:          20
 *   beacon_interval_ms:  5000
 *
 * After editing the malicious payload strings to match patterns/rules.txt,
 * run once and commit the output. Or add a make/cmake target that regenerates
 * it automatically before the demo.
 */

#include <arpa/inet.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pcap.h>

// Raw frame headers. #pragma pack ensures no padding bytes are inserted.
#pragma pack(push, 1)
struct EthHdr {
    uint8_t  dst[6], src[6];
    uint16_t ethertype;
};
struct IpHdr {
    uint8_t  ver_ihl, tos;
    uint16_t total_len, id, frag;
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

static void emit_packet(pcap_dumper_t* dumper,
                        const char*    payload,
                        size_t         payload_len,
                        uint32_t       src_ip,
                        uint32_t       dst_ip,
                        uint16_t       sport,
                        uint16_t       dport,
                        uint64_t       ts_us) {
    uint8_t buf[4096] = {};
    auto* eth  = reinterpret_cast<EthHdr*>(buf);
    auto* ip   = reinterpret_cast<IpHdr*>(buf + sizeof(EthHdr));
    auto* tcp  = reinterpret_cast<TcpHdr*>(buf + sizeof(EthHdr) + sizeof(IpHdr));
    char* data = reinterpret_cast<char*>(tcp + 1);

    eth->ethertype = htons(0x0800);         // IPv4
    ip->ver_ihl    = 0x45;                  // version=4, IHL=20 bytes
    ip->ttl        = 64;
    ip->proto      = 6;                     // TCP
    ip->src        = src_ip;
    ip->dst        = dst_ip;
    ip->total_len  = htons(sizeof(IpHdr) + sizeof(TcpHdr) + payload_len);
    tcp->sport     = htons(sport);
    tcp->dport     = htons(dport);
    tcp->data_off  = 0x50;                  // 20-byte TCP header
    tcp->flags     = 0x18;                  // PSH + ACK
    std::memcpy(data, payload, payload_len);

    size_t total = sizeof(EthHdr) + sizeof(IpHdr) + sizeof(TcpHdr) + payload_len;
    pcap_pkthdr hdr{};
    hdr.ts.tv_sec  = static_cast<decltype(hdr.ts.tv_sec)>(ts_us / 1000000ULL);
    hdr.ts.tv_usec = static_cast<decltype(hdr.ts.tv_usec)>(ts_us % 1000000ULL);
    hdr.caplen = hdr.len = static_cast<bpf_u_int32>(total);
    pcap_dump(reinterpret_cast<u_char*>(dumper), &hdr, buf);
}

int main(int argc, char** argv) {
    const char* out_path      = argc > 1 ? argv[1] : "data/synthetic.pcap";
    int         num_benign    = argc > 2 ? std::atoi(argv[2]) : 10000;
    int         num_malicious = argc > 3 ? std::atoi(argv[3]) : 50;
    int         num_beacon    = argc > 4 ? std::atoi(argv[4]) : 20;
    int         beacon_ms     = argc > 5 ? std::atoi(argv[5]) : 5000;

    pcap_t*        p = pcap_open_dead(DLT_EN10MB, 65535);
    pcap_dumper_t* d = pcap_dump_open(p, out_path);
    if (!d) {
        std::fprintf(stderr, "Cannot open %s for writing\n", out_path);
        return 1;
    }

    // Timestamp base (microseconds). We keep timestamps monotonic so IATs are well-defined.
    uint64_t ts_us = 1700000000ULL * 1000000ULL;

    // Benign: normal HTTP GET requests from 10.0.0.1
    char buf[512];
    for (int i = 0; i < num_benign; i++) {
        int n = std::snprintf(buf, sizeof(buf),
            "GET /page%d HTTP/1.1\r\nHost: example.com\r\nUser-Agent: curl/7.68\r\n\r\n",
            i);
        emit_packet(d, buf, static_cast<size_t>(n),
            htonl(0x0A000001), htonl(0x0A000002), 12345, 80, ts_us); // 10.0.0.1 -> 10.0.0.2
        ts_us += 1000; // 1 ms spacing
    }

    // Malicious: payloads containing patterns from rules.txt.
    // Update these strings after finalizing patterns/rules.txt so that the
    // synthetic pcap reliably fires the expected alert set.
    const char* payloads[] = {
        "GET /cgi-bin/../../etc/passwd HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        "POST /wp-login.php HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 0\r\n\r\n",
        "GET /phpmyadmin/ HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // Add more entries that match strings in rules.txt.
    };
    int npayloads = static_cast<int>(sizeof(payloads) / sizeof(payloads[0]));

    for (int i = 0; i < num_malicious; i++) {
        const char* pl = payloads[i % npayloads];
        emit_packet(d, pl, std::strlen(pl),
            htonl(0x0A000063), htonl(0x0A000002), 23456, 80, ts_us); // 10.0.0.99 -> 10.0.0.2
        ts_us += 2000; // 2 ms spacing
    }

    // Beacon: highly periodic flow to trigger low IAT CV.
    const char* beacon_payload = "PING /beacon HTTP/1.1\r\nHost: c2.example\r\n\r\n";
    for (int i = 0; i < num_beacon; i++) {
        emit_packet(d, beacon_payload, std::strlen(beacon_payload),
            htonl(0x0A000050), htonl(0x0A000002), 34567, 443, ts_us); // 10.0.0.80 -> 10.0.0.2
        ts_us += static_cast<uint64_t>(beacon_ms) * 1000ULL;
    }

    pcap_dump_close(d);
    pcap_close(p);
    std::printf("Wrote %d benign + %d malicious + %d beacon packets to %s\n",
                num_benign, num_malicious, num_beacon, out_path);
    return 0;
}
