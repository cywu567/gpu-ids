# GPU-Accelerated Intrusion Detection System

A weekend hackathon prototype that grows into a 4-week CUDA class project (CS179).

---

## The one-paragraph pitch

Build a GPU-accelerated multi-pattern matcher that scans network traffic for malicious byte signatures, in the style of Snort/Suricata but with the matching engine running on a GPU instead of a CPU. The hackathon version is a naive but working parallel matcher that beats a CPU baseline on a real malware traffic capture. The 4-week version replaces the naive matcher with a proper Aho-Corasick (PFAC) implementation, adds TCP flow reassembly, and benchmarks against modern CPU baselines.

---

## The problem (the security context)

An Intrusion Detection System (IDS) sits on a network and inspects every packet that passes through. Its job is to spot known-bad traffic: exploit attempts, malware command-and-control, scanning, data exfiltration. Security researchers write *rules* that describe what bad traffic looks like, and the IDS checks every packet against every rule.

A real IDS has tens of thousands of rules running against every packet. The Emerging Threats open ruleset alone is around 40,000 rules. At a 40 Gbps network link (5 GB/s of packet data), CPU-based matching engines struggle to keep up without dropping packets — which means missed attacks. Enterprise-grade appliances solve this with FPGAs or ASICs costing tens of thousands of dollars.

The motivating question for our project: **can a $500 consumer GPU do the same job in software?**

---

## The core algorithm (Aho-Corasick)

Naively scanning a packet 40,000 times (once per rule) is absurd. Aho-Corasick is a 1975 algorithm that lets you scan once and check for all patterns simultaneously, regardless of how many patterns you have.

**The intuition:** arrange all patterns as a trie (prefix tree). Walk the input byte-by-byte through the trie. When you can't extend a match, "fail" to a precomputed link that points to the longest proper suffix of what you've seen so far that's also a prefix of some pattern, and keep reading. You never re-read a single byte.

In practice this is converted to a deterministic finite automaton (DFA) — a giant 2D table of size `(states) × 256` where each cell says "from state S, on byte B, go to state T." Matching is then unbelievably simple:

```c
state = 0;
for (i = 0; i < input_length; i++) {
    state = transition_table[state][input[i]];
    if (is_accepting(state)) emit_match(state, i);
}
```

That inner loop is what we run on a GPU.

---

## Why GPUs (the parallelization story)

The matching loop is sequential per byte (state at step `i` depends on state at step `i-1`), so a single thread walking through one packet gets no parallelism. The wins come from parallelizing across other axes:

1. **Across packets** — one thread per packet (easy).
2. **Across positions** — each thread starts at a different offset (PFAC-style).
3. **Across chunks** — split a large input into overlapping chunks, one thread per chunk.

The interesting CUDA engineering questions:

- **Memory layout:** the transition table is megabytes — too big for shared memory. Where does it live? Global? Texture? Read-only cache?
- **Warp divergence:** GPU threads run in groups of 32 that must execute the same instruction; threads landing in different DFA states diverge.
- **Output collection:** different threads find different numbers of matches; how do you aggregate without expensive atomics?

These three problems are the actual CUDA content. Solving them is what turns this from "I typed an algorithm" into "I optimized a real kernel."

---

## Honest novelty assessment

**The algorithm is not novel.** GPU Aho-Corasick has been studied since the early 2010s. The foundational paper is Lin, Liu, and Chang (2013), "Accelerating Pattern Matching Using a Novel Parallel Algorithm on GPUs" (PFAC). There are many follow-ups.

**What we can legitimately claim:**

1. We did the implementation ourselves (the actual point of CS179).
2. We did it on modern hardware (the original PFAC paper used a 2010-era GTX 480; modern GPUs have very different memory hierarchies).
3. We compared honestly against Hyperscan, the modern fast CPU baseline (most papers compare against textbook AC and "win" — comparing against Hyperscan is harder and more truthful).
4. We built an end-to-end open-source pipeline (most papers benchmark just the kernel in isolation).

**Proposal framing for the TA:** "Nothing about the algorithm is new. PFAC has been around since 2013. Our contribution is engineering and honest measurement, not algorithmic novelty." This is a normal, acceptable framing for a class project.

---

## Current state of the art (what already exists)

- **Snort / Suricata** (open-source IDSes): use Aho-Corasick on CPU.
- **Hyperscan** (Intel library): the modern fast CPU implementation; uses SIMD instructions. State-of-the-art for software multi-pattern matching today.
- **High-end firewall appliances** (Cisco, Palo Alto, Fortinet): use FPGAs or ASICs.
- **NVIDIA Morpheus** (2022): NVIDIA's GPU cybersecurity framework; includes some pattern matching but focuses on ML-based detection.
- **GPU-based open-source IDS as a complete tool**: basically doesn't exist. Research prototypes only.

---

## Hackathon weekend MVP

**Goal:** by Sunday afternoon, have a working naive multi-pattern matcher running on GPU, demoed against a CPU baseline on a real malware pcap.

### What to cut for the weekend

- Skip Aho-Corasick entirely; use naive multi-pattern matching (each thread checks one pattern with a sliding window).
- Skip TCP flow reassembly; treat each packet independently.
- Skip Snort rule parsing; hand-curate ~20-50 patterns.
- Skip anomaly detection.
- Skip live network capture; use pcap files only.

### Tech stack

- **CUDA C++** for the GPU kernels.
- **C++17** for everything else: pcap I/O, CPU baseline, host-side launch code, benchmarking harness, demo UI.
- **libpcap** for reading and writing pcap files (`#include <pcap.h>`).
- **CMake** (or a plain Makefile) as the build system, with `nvcc` for `.cu` files and `g++`/`clang++` for `.cpp` files.
- **Demo UI:** CLI by default. If a web UI is wanted, [cpp-httplib](https://github.com/yhirose/cpp-httplib) is a single-header C++ HTTP server that fits in one file.
- **Public pcaps** from [malware-traffic-analysis.net](https://malware-traffic-analysis.net/).
- **Patterns:** hand-picked HTTP indicators (suspicious URIs, malware user-agents, planted test strings).

### The naive kernel

```cuda
__global__ void naive_match(
    const char* input,
    const int* packet_offsets,
    int num_packets,
    const char* patterns,
    const int* pattern_offsets,
    const int* pattern_lengths,
    int num_patterns,
    int* hits
) {
    int packet_idx = blockIdx.x;
    int pattern_idx = threadIdx.x;
    if (packet_idx >= num_packets || pattern_idx >= num_patterns) return;

    const char* pkt = input + packet_offsets[packet_idx];
    int pkt_len = packet_offsets[packet_idx + 1] - packet_offsets[packet_idx];
    const char* pat = patterns + pattern_offsets[pattern_idx];
    int pat_len = pattern_lengths[pattern_idx];

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
```

Launch with `<<<num_packets, num_patterns>>>`. Bad code on purpose — this is the punching bag we'll improve over the next 3 weeks.

### Day-by-day plan

**Friday night (3-5 hours):**
- Both: dev environment working (CUDA toolkit installed, `nvcc` and a host C++ compiler both functional, libpcap installed, hello-world kernel runs).
- Both: get a minimal CMake / Makefile building a `.cu` and a `.cpp` file together into one binary.
- Person A: C++ pcap loader using libpcap (`pcap_open_offline` + `pcap_loop`), outputting a flat byte buffer + offset array in host memory.
- Person B: hand-curate pattern list (just a text file). Pick a malware pcap. Use Wireshark to find patterns that occur in it.

**Saturday:**
- Person A: write the naive GPU kernel + the host-side `cudaMalloc` / `cudaMemcpy` / kernel-launch code in C++. Get correct matching on test pcap. Write the C++ that turns the `hits[]` array into human-readable alerts on stdout.
- Person B: write the CPU baseline matcher in C++ (`std::string::find` or `memmem` in a loop). Build the C++ benchmarking harness using `std::chrono::high_resolution_clock` to time both implementations.
- Together (evening): wire into a CLI demo. Side-by-side "CPU: X MB/s | GPU: Y MB/s" output. If you want a web UI, drop in `cpp-httplib` and serve a single HTML page that polls a `/stats` endpoint.

**Sunday:**
- Run on multiple pcaps, tune pattern list to reliably fire visible alerts.
- Add a rate-control knob; show CPU dropping behind while GPU keeps up.
- Make a throughput chart.
- Write the proposal draft.

---

## 4-week class project breakdown

### Week 1: infrastructure and baseline

Cleanup of hackathon code. Pick a real pattern source (subset of Snort or Emerging Threats rules). Write a CPU reference Aho-Corasick (using an existing library to verify correctness). Generate / collect a proper test dataset. Submit proposal (May 6).

**Deliverable:** clean CPU IDS that fires alerts on a test pcap.

### Week 2: real GPU kernel

Implement PFAC properly. Two new components in `gpu_matcher.cu`:
- A function that takes the pattern list and builds the DFA.
- A new kernel that uses the DFA instead of dumb comparison.

**Deliverable:** correct PFAC kernel running on GPU.

### Week 3: optimization and flow reassembly (parallel work)

Person A: profile with `nsight compute`, optimize memory layout, tune for warp behavior, push throughput up.

Person B: implement TCP flow reassembly (hash table of flows keyed by 5-tuple, reordering buffers, sequence number handling). First on CPU, then port hot paths to GPU.

**Deliverable:** fast smart kernel + working flow reassembly.

### Week 4: integration, benchmarks, writeup

Wire everything end-to-end. Benchmark against Snort and (if possible) Hyperscan on a standardized pcap dataset. Write the report with proper charts. Build whatever demo UI is wanted.

**Deliverable:** final submission (June 5 or 12).

---

## File structure (what the folder looks like at the end)

```
gpu-ids/
├── README.md              ← how to build and run everything
├── CMakeLists.txt         ← build configuration
├── data/
│   ├── test.pcap          ← real malware traffic capture
│   └── synthetic.pcap     ← generated pcap with known patterns
├── patterns/
│   └── rules.txt          ← list of suspicious byte strings
├── src/
│   ├── load_pcap.cpp      ← libpcap-based packet loader
│   ├── load_pcap.h
│   ├── cpu_matcher.cpp    ← reference CPU matcher
│   ├── cpu_matcher.h
│   ├── gpu_matcher.cu     ← naive + PFAC GPU kernels + host launch code
│   ├── gpu_matcher.cuh
│   ├── flow_reassembly.cu ← TCP stream reconstruction (CPU + GPU paths)
│   ├── flow_reassembly.cuh
│   ├── benchmark.cpp      ← times CPU vs GPU, writes results CSV
│   ├── gen_synthetic.cpp  ← generates synthetic pcaps for benchmarks
│   └── run_demo.cpp       ← end-to-end CLI binary
├── results/
│   ├── benchmark.csv      ← raw throughput numbers
│   ├── speedup_chart.png  ← plotted from benchmark.csv (gnuplot/external tool)
│   └── notes.md
└── report.pdf
```

Plotting can be done by piping `benchmark.csv` through `gnuplot`, or by running a tiny separate Python script just to make the chart image — but no Python is in the runtime path.

---

## Partner split

### Hackathon weekend (CUDA work is concentrated)

- **Partner A (GPU side):** GPU kernel (`gpu_matcher.cu`) + host-side `cudaMalloc`/`cudaMemcpy`/launch code + CPU baseline (`cpu_matcher.cpp`). Owns "the matching engine" — input: bytes + patterns; output: list of hits.
- **Partner B (everything else):** libpcap loader (`load_pcap.cpp`) + pattern curation + CLI demo (`run_demo.cpp`) + benchmarking harness (`benchmark.cpp`). Owns "the wrapper that makes the engine look like an IDS."
- **Interface (a C++ header, e.g., `gpu_matcher.cuh`):** Partner B passes Partner A a `const uint8_t* bytes`, a `const size_t* packet_offsets`, a count, and a pattern list. Partner A returns a `std::vector<Hit>` (or fills a caller-provided `Hit*` buffer).

### 4-week class project (CUDA work splits across both)

- **Partner A:** Aho-Corasick / PFAC kernel implementation, memory layout optimization, performance tuning, Hyperscan/Snort comparison.
- **Partner B:** TCP flow reassembly (parallel hash table on GPU is its own CUDA topic), per-flow statistics kernels (parallel reductions across many flows), anomaly rules.

Partner B's 4-week work is a natural extension of the CPU plumbing they wrote during the hackathon — they port their own code to GPU.

---

## Hardware requirements

- **VRAM:** 4 GB is plenty (DFA + chunk of packets fits easily).
- **Compute capability:** 6.0+ (any GPU from GTX 1060 / 1080 onward).
- **Cloud fallbacks:** Google Colab free tier (T4 16GB), Kaggle (P100), Lambda Labs.
- **Caltech:** check the CS179 cluster availability with the TA on day 1 — that's the expected dev/benchmark hardware.

Develop on whatever's convenient with small inputs; benchmark on one consistent piece of hardware for the report.

---

## Data sources

### Real pcaps (for realism / case studies)

- **[malware-traffic-analysis.net](https://malware-traffic-analysis.net/)** — hundreds of real malware infection captures with writeups. Primary source.
- **Wireshark sample captures wiki** — small grab-bag, useful for parser testing.
- **CIC datasets** (Canadian Institute for Cybersecurity) — academic IDS datasets.
- **DARPA / MIT Lincoln Labs** — classic but old (1999).

### Synthetic pcaps (for benchmarking and demo control)

Generate with C++ and libpcap. Build the Ethernet/IP/TCP headers manually, then dump packets through `pcap_dump`:

```cpp
// gen_synthetic.cpp
#include <pcap.h>
#include <cstring>
#include <cstdio>
#include <arpa/inet.h>

#pragma pack(push, 1)
struct EthHdr { uint8_t dst[6], src[6]; uint16_t ethertype; };
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

static void emit_packet(pcap_dumper_t* dumper, const char* payload,
                        size_t payload_len, uint32_t src_ip) {
    uint8_t buf[2048] = {};
    auto* eth = reinterpret_cast<EthHdr*>(buf);
    auto* ip  = reinterpret_cast<IpHdr*>(buf + sizeof(EthHdr));
    auto* tcp = reinterpret_cast<TcpHdr*>(buf + sizeof(EthHdr) + sizeof(IpHdr));
    char* data = reinterpret_cast<char*>(tcp) + sizeof(TcpHdr);

    eth->ethertype = htons(0x0800);                 // IPv4
    ip->ver_ihl = 0x45;                              // IPv4, 20-byte header
    ip->ttl = 64;
    ip->proto = 6;                                   // TCP
    ip->src = src_ip;
    ip->dst = htonl(0x0A000002);                     // 10.0.0.2
    ip->total_len = htons(sizeof(IpHdr) + sizeof(TcpHdr) + payload_len);
    tcp->sport = htons(12345);
    tcp->dport = htons(80);
    tcp->data_off = 0x50;                            // 20-byte TCP header
    tcp->flags = 0x18;                               // PSH+ACK
    std::memcpy(data, payload, payload_len);

    size_t total = sizeof(EthHdr) + sizeof(IpHdr) + sizeof(TcpHdr) + payload_len;
    pcap_pkthdr h{};
    h.caplen = h.len = total;
    pcap_dump(reinterpret_cast<u_char*>(dumper), &h, buf);
}

int main() {
    pcap_t* p = pcap_open_dead(DLT_EN10MB, 65535);
    pcap_dumper_t* d = pcap_dump_open(p, "synthetic.pcap");

    char benign[256];
    for (int i = 0; i < 10000; i++) {
        int n = std::snprintf(benign, sizeof(benign),
            "GET /page%d HTTP/1.1\r\nHost: example.com\r\n\r\n", i);
        emit_packet(d, benign, n, htonl(0x0A000001));   // 10.0.0.1
    }

    const char* malicious = "GET /cgi-bin/../../etc/passwd HTTP/1.1\r\n\r\n";
    for (int i = 0; i < 50; i++) {
        emit_packet(d, malicious, std::strlen(malicious), htonl(0x0A000063)); // 10.0.0.99
    }

    pcap_dump_close(d);
    pcap_close(p);
    return 0;
}
```

Build: `g++ gen_synthetic.cpp -lpcap -o gen_synthetic`. Run once and check `synthetic.pcap` into the repo (or keep regenerating it as a make target).

**Strategy:** real pcaps for the demo and the case-study chapter; synthetic pcaps for clean benchmark charts (controlled sizes, controlled pattern density).

---

## CS179 deliverables

| Deadline | Deliverable | What it is |
|----------|-------------|------------|
| May 6 | Proposal (PDF) | 1-2 pages: summary, background, technical challenges, deliverables, week-by-week timeline. Email to sfoxman@caltech.edu, subject "CS179 Project Proposal". |
| May 27 | CPU demo | Working CPU baseline (essentially the hackathon weekend code). Proves the concept works pre-GPU. |
| June 5 (seniors/grads) or June 12 (other undergrads) | Final submission | Full code (CPU + GPU) + comprehensive README + performance analysis. |

---

## Key references

- Aho, A. V., & Corasick, M. J. (1975). *Efficient string matching: an aid to bibliographic search.* Communications of the ACM.
- Lin, C., Liu, C., Chang, S. (2013). *Accelerating Pattern Matching Using a Novel Parallel Algorithm on GPUs.* IEEE Transactions on Computers. **(The PFAC paper — primary reference for the kernel.)**
- Tumeo et al., various papers on GPU multi-pattern matching.
- [Snort documentation](https://www.snort.org/documents) — for understanding rule format and CPU baseline behavior.
- [Hyperscan](https://www.hyperscan.io/) — the modern fast CPU multi-pattern library, what we benchmark against.
- [PFAC on GitHub](https://github.com/pfac-lib/PFAC) — reference open-source implementation.

---

## What we're explicitly *not* doing

- Inventing a new pattern-matching algorithm.
- Building a production-grade IDS.
- Hitting 40 Gbps in our final benchmarks (realistic target: 5-15 Gbps on a consumer GPU, which is still a large multiple over CPU).
- Handling all of Snort's rule features (PCRE regexes, protocol filters, offset/depth constraints). We stick to plain `content:` byte-string matches and document this simplification in the writeup.

---

## Open questions to resolve in week 1

- Which exact pattern source? (Snort community rules subset vs. Emerging Threats subset vs. hand-curated list.)
- Which pcaps for the standardized benchmark? (Pick 2-3 specific ones and stick with them.)
- Hyperscan comparison: yes or no? (It's harder but more credible. Decide based on time.)
- GPU target: develop on Colab T4, benchmark on Caltech cluster, or both?
- Build system: CMake or hand-rolled Makefile? (CMake is more standard but Makefile is simpler for a small project.)
