# CudaShield — Presentation Script

---

## 1. Hook (30 sec)

Every packet that hits your network — every login, every file transfer, every DNS lookup —
needs to be scanned against thousands of known-bad signatures before it's allowed through.
Snort. Suricata. These are the industry-standard tools. They run on CPUs.

At 10 Gbps, you have about **120 nanoseconds** per packet to do all of that matching.
CPUs can't keep up. Enterprises spend **$50,000 on FPGA appliances** just to hit line rate.

We asked: what if you just used a consumer GPU?

---

## 2. What We Built (1 min)

**CudaShield** is a GPU-accelerated Intrusion Detection System.

You give it a pcap capture and a rules file — the same format Snort uses.
It scans every packet against every pattern simultaneously on the GPU, flags alerts,
and shows throughput metrics in a live web dashboard.

Three engines, all running side by side:
- **CPU baseline** — reference `memmem` sliding-window search
- **GPU naive kernel** — brute force, one thread per (packet, pattern) pair
- **GPU PFAC kernel** — Parallel Failureless Aho-Corasick, one thread per start byte

Plus a fourth component: **beacon detection** — a separate GPU kernel that catches
C2 (command-and-control) malware that hides inside encrypted traffic where pattern matching
is blind.

---

## 3. The Problem in Detail (1.5 min)

The core challenge is **multi-pattern string matching at network speed**.

A naive approach: for each packet, loop over every rule, scan for a match.
That's O(packet_len × pattern_len × num_patterns) per packet.
At 26 rules and 1500-byte packets, that's already ~100K comparisons per packet.
A CPU does this serially — one packet at a time, one pattern at a time.

The CPU baseline we measured: **~84 MB/s**.

At 10 Gbps line rate you need **1,250 MB/s** just to keep up.
CPUs fall 15× short.

The industry answer is **Hyperscan** — Intel's hand-written SIMD engine that vectorizes
across multiple patterns using AVX-512. That gets to about **~1,490 MB/s**. Better, but
still under 10 Gbps, and it's a proprietary Intel CPU product.

Our GPU answer: **~4,960 MB/s** — a 3.3× speedup over Hyperscan, and **59× over the CPU baseline**.

---

## 4. Kernel 1 — Naive GPU (45 sec)

The naive kernel is exactly what you'd think.

Grid: one block per packet. Block: one thread per pattern.
Each thread independently does a brute-force sliding-window scan for its one pattern
in its one packet and writes a hit flag.

```
blockIdx.x  = packet index
threadIdx.x = pattern index
```

This already gets us **~1,490 MB/s** — matching Hyperscan — just from parallelism across
packets. But it has two problems:

1. **Hard cap of 1,024 patterns** — that's CUDA's max threads per block.
2. **Warp divergence** — threads in the same warp take different branch paths when
   packets have different lengths or patterns match at different positions.

It's a useful baseline, but not production-ready.

---

## 5. Kernel 2 — PFAC Aho-Corasick (2 min)

This is the real engine. Based on a 2013 IEEE paper by Lin, Liu, and Chang.

**The idea:** instead of running one thread per pattern, run one thread per *byte position*
in the packet. Every thread starts at a different byte and walks a precompiled DFA forward
until it hits a dead state. If it reaches an accepting state, that's a match.

```
blockIdx.x  = packet index
threadIdx.x = starting byte offset (stride 256 across the packet)
```

The DFA is the key. Offline, before any scanning, we compile all patterns into a single
trie-based automaton — a `state × 256` transition table. Every state is a node in the
trie. You look up `table[state * 256 + byte]` to get the next state.

Classic Aho-Corasick adds failure links so that when a partial match fails, you jump
back to the longest possible suffix that's still a valid prefix of some pattern.
**PFAC removes failure links entirely** — because every byte position gets its own fresh
thread starting from root, there's nothing to fall back to. The "failureless" property
is the key insight.

This gives us:
- **O(packet_len) per thread** — independent of pattern count
- **No 1,024-pattern limit** — the DFA absorbs any number of patterns
- Measured throughput: **~4,960 MB/s**

The DFA table is stored as `uint16_t` (2 bytes per entry instead of 4), cutting the
table size roughly in half — about **100 KB** for 26 patterns.

**Shared memory variant:** That 100 KB table is larger than the GPU's 32 KB read-only L1
cache, so on large pcaps we get cache misses costing ~200 cycles each. The `pfac_kernel_smem`
variant has all 256 threads in a block cooperatively load up to 48 KB of DFA rows into
shared memory before scanning — shared memory hits in ~4 cycles and can't be evicted.
For our 26-pattern rule set, the entire DFA fits.

---

## 6. Beacon Detection Kernel (1 min)

Pattern matching is blind to **encrypted traffic**. A C2 beacon inside a TLS session
produces zero PFAC hits — the payload is ciphertext. But beacons have a structural tell:
**regular inter-arrival times**.

Malware that phones home every 60 seconds is perfectly clockwork. Legitimate user traffic
is bursty and irregular.

Our **flow statistics kernel** catches this:

1. CPU groups packets by TCP/UDP 5-tuple (src IP, dst IP, src port, dst port, protocol)
2. GPU kernel — one block per flow — computes mean and variance of inter-arrival times
   using **warp-shuffle parallel reductions** (`__shfl_down_sync`)
3. Any flow with IAT coefficient of variation below 0.15 and at least 5 packets
   is flagged as a beacon

This catches C2 traffic that looks completely clean to a signature-based IDS.

---

## 7. Live Demo (90 sec)

*Open the browser to cudashield.tech*

"This is the live dashboard running on a GPU in the Caltech CS cluster."

**Step 1:** Load the pcap — show the packet count and file size.

**Step 2:** Run the scan. Watch the three bars fill in simultaneously:
- Yellow bar — CPU at ~84 MB/s
- Green bar — GPU PFAC at ~4,960 MB/s
- Purple bar — Hyperscan at ~1,490 MB/s

"Same alerts across all three engines — they all find the same packets.
The GPU is just **59× faster than the CPU** and **3.3× faster than Hyperscan**."

**Step 3:** Show the alerts table — packet index and matched pattern string.

**Step 4:** Enable beacon detection. Point out any flagged flows.
"These flows had less than 15% IAT coefficient of variation — statistical fingerprint
of automated, scheduled communication. Pattern matching found nothing. The stats did."

---

## 8. Architecture Summary (30 sec)

```
pcap file → load_pcap.cpp  (flat byte buffer + offsets)
                ↓
        [CPU matcher]    → memmem baseline
        [GPU naive]      → one thread per (packet, pattern)
        [GPU PFAC]       → one thread per start byte, DFA walk
        [Flow stats]     → one block per flow, warp-shuffle IAT reduction
                ↓
        web_server.cpp   → JSON API + single-page dashboard
                ↓
        Cloudflare Tunnel → cudashield.tech
```

All three matching engines produce identical alert sets.
Benchmarking is isolated per-engine with wall-clock timing.

---

## 9. Results

| Engine | Throughput | vs CPU |
|---|---|---|
| CPU (memmem) | ~84 MB/s | 1× |
| GPU Naive | ~1,490 MB/s | ~18× |
| Hyperscan (SIMD CPU) | ~1,490 MB/s | ~18× |
| **GPU PFAC** | **~4,960 MB/s** | **~59×** |

GPU PFAC runs at **3.3× Hyperscan** on a consumer GPU that costs a fraction of an
enterprise FPGA appliance — and it scales with pattern count, not against it.

---

## 10. What We'd Do With More Time

- **Streaming input** — currently scans complete pcap files; a real deployment would
  need a live packet capture loop feeding the GPU continuously
- **Larger rule sets** — Snort ships with ~30,000 rules; we'd need to profile DFA
  memory layout for that scale
- **Alert correlation** — right now each engine's alerts are independent; a real IDS
  would correlate across flows and time windows
- **PCAP benchmarks at line rate** — synthetic traffic generator at 10 Gbps to stress-test
  the pipeline end-to-end

---

## Talking Points for Q&A

**"Why not just buy more CPUs?"**
You can scale CPUs horizontally, but you pay per core and per watt. A single RTX 3090
has 10,496 CUDA cores. The parallelism density per dollar and per watt strongly favors GPU
for embarrassingly parallel workloads like packet scanning.

**"Is PFAC better than Aho-Corasick in all cases?"**
PFAC trades memory bandwidth for no failure-link overhead. For short patterns on modern
GPUs with high memory bandwidth, PFAC wins. For very long patterns or very deep trie
nodes, standard AC with failure links can be more cache-friendly. PFAC is the right
tradeoff for IDS rule sets.

**"Why is the shared-memory variant sometimes slower than __ldg?"**
For short packets (~1500 bytes), the overhead of the cooperative shared memory load at
block startup dominates. The `__ldg` path benefits from L2 cache warmup across blocks.
For longer payloads (e.g., reassembled TCP streams), shared memory would win — we exposed
both variants in the benchmark for exactly this comparison.

**"What's the beacon detection false positive rate?"**
The CV threshold of 0.15 and minimum 5-packet requirement are tunable. In practice, most
real beacons are far more regular than 0.15 CV, and legitimate periodic traffic (NTP, DHCP
renewals) is usually filtered out by excluding well-known ports. For a production system
you'd tune these per-environment.
