/**
 * @file load_pcap.cpp
 * @brief Implementation of the libpcap-based packet loader.
 *
 * Uses pcap_open_offline + pcap_loop to iterate every packet in the capture
 * file, copying each packet's raw bytes (caplen bytes, not the original wire
 * length) into a growing flat buffer. The full frame -- Ethernet header,
 * IP header, transport header, payload -- is preserved without any parsing.
 *
 * A sentinel entry appended to the offset array after the loop means callers
 * always compute packet length as offsets[i+1] - offsets[i], with no special
 * case for the last packet.
 *
 * Limitations:
 *   - Loads the entire capture into memory before returning.
 *   - No streaming / incremental processing.
 *   - Truncated packets (caplen < len) are stored at their captured size.
 */

#include "load_pcap.h"
#include <pcap.h>
#include <stdexcept>
#include <string>

namespace {

struct LoadCtx {
    PcapData* out;
};

void packet_handler(u_char* user, const pcap_pkthdr* hdr, const u_char* data) {
    auto* ctx = reinterpret_cast<LoadCtx*>(user);
    PcapData* d = ctx->out;

    d->offsets.push_back(static_cast<int>(d->bytes.size()));
    d->timestamps_us.push_back(
        (uint64_t)hdr->ts.tv_sec * 1000000ULL + (uint64_t)hdr->ts.tv_usec);
    d->bytes.insert(d->bytes.end(), data, data + hdr->caplen);
    d->num_packets++;
}

} // namespace

PcapData load_pcap(const std::string& path) {
    char errbuf[PCAP_ERRBUF_SIZE];
    pcap_t* handle = pcap_open_offline(path.c_str(), errbuf);
    if (!handle)
        throw std::runtime_error(std::string("pcap_open_offline failed: ") + errbuf);

    PcapData result;
    result.num_packets = 0;

    LoadCtx ctx{&result};
    pcap_loop(handle, 0, packet_handler, reinterpret_cast<u_char*>(&ctx));
    pcap_close(handle);

    result.offsets.push_back(static_cast<int>(result.bytes.size())); // sentinel
    return result;
}
