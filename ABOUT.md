# CudaShield — Project About

---

## Inspiration

Our inspiration came directly from **CS179, Caltech's GPU Programming course**. Spending weeks diving deep into CUDA kernels and parallel memory hierarchies made us want to apply those techniques to a problem with real-world stakes — not just matrix multiplication. Network intrusion detection immediately stood out: it's embarrassingly parallel, computationally brutal, and the gap between what CPUs can do and what networks demand is already a genuine unsolved problem. Enterprises spend **$50,000 on FPGA appliances** just to hit line rate at 10 Gbps. We walked out of CS179 lectures wondering whether a consumer GPU — something costing a few hundred dollars — could close that gap in software. CudaShield is our answer to that question.

---

## What it does

CudaShield is a **GPU-accelerated network intrusion detection system** that scans real network traffic captures (pcap files) against a set of known-bad byte signatures in the style of Snort and Suricata, but with the matching engine running on a GPU instead of a CPU.

You point it at a pcap file and a rules file, and it runs three engines in parallel:

- **CPU baseline** — reference `memmem` sliding-window search (~84 MB/s)
- **GPU naive kernel** — one thread per (packet, pattern) pair (~1,490 MB/s)
- **GPU PFAC Aho-Corasick kernel** — one thread per start byte, single DFA walk, no pattern-count limit (~4,960 MB/s)
- **Hyperscan (Vectorscan)** — Intel's production-grade SIMD CPU matcher as an industry baseline (~1,490 MB/s)

All engines produce identical alert sets. Results stream to a live web dashboard at **cudashield.tech**, showing throughput bars and matched packet alerts side by side. The GPU PFAC engine runs at **59× the CPU baseline** and **3.3× faster than Hyperscan** — on a consumer GPU that costs a fraction of an enterprise FPGA.

---

## How we built it

The entire runtime is **CUDA C++ and C++17** — no Python, no scripting languages in the critical path.

- **Packet I/O:** `libpcap` reads pcap files into a flat byte buffer with a parallel offset array, so all packet data is contiguous in memory for efficient GPU transfer.
- **CPU matcher:** a `memmem`-based sliding window loop, used as the correctness reference and throughput baseline.
- **GPU naive kernel:** grid of `(num_packets × num_patterns)` threads; each thread brute-forces one pattern through one packet. Simple, correct, and serves as the punching bag that PFAC improves on.
- **GPU PFAC kernel:** all rules are compiled offline into a single `uint16_t` DFA (state × 256 transition table). At scan time, one block per packet, 256 threads per block — each thread starts at a different byte offset and walks the DFA forward from root until it hits a dead state. No failure links needed because every byte gets a fresh start. For our 26-pattern rule set, the entire ~100 KB DFA fits in shared memory (loaded cooperatively at block startup), giving ~4-cycle lookups instead of the ~200-cycle global-memory penalty.
- **Web dashboard:** `cpp-httplib` (single-header C++ HTTP server) serves a live JSON stats endpoint; the frontend polls it to update throughput bars and alerts in real time.
- **Public access:** Cloudflare Tunnel routes `cudashield.tech` to the IDS server running on the Caltech GPU cluster, bypassing firewall restrictions without needing inbound ports or root access.
- **Real malware data:** traffic captures from `malware-traffic-analysis.net` — actual TrickBot, Cobalt Strike, and other infection captures with documented indicators.
- **Build system:** CMake with conda for dependency management (libpcap, Vectorscan) so the project builds without sudo on a shared cluster.

---

## Challenges we ran into

The hardest technical challenge was **TCP flow reassembly**. Many real attacks split their malicious payload across multiple TCP segments — each packet individually looks clean, but reassembled into the full stream, the signature appears. We designed and partially implemented both a CPU path (hash table keyed by 5-tuple, sequence-number gap buffering) and a GPU path (open-addressing hash table with `atomicCAS` probing, Thrust prefix-sum scatter into a contiguous buffer). The CPU path worked correctly. The GPU path, however, could not be made efficient enough to beat Hyperscan on our test data — the overhead of the per-flow atomics, the `thrust::exclusive_scan`, and the extra memory copy into the reassembled buffer ate the GPU's bandwidth advantage on typical pcap sizes. For a deployment that sees long-lived, high-volume TCP streams this tradeoff would flip, but we couldn't demonstrate it cleanly within the project scope.

Other challenges:
- **Warp divergence** — threads in the same warp take different DFA paths for different packets, causing instruction-level serialization that's hard to eliminate without sacrificing the simplicity of the PFAC mapping.
- **Shared memory vs. `__ldg`** — for short packets (~1500 bytes), the cooperative shared-memory load at block startup costs more than it saves; the `__ldg` path benefits from L2 warmup across blocks. We exposed both variants in the benchmark rather than hiding the tradeoff.
- **GPU transfer overhead on small inputs** — `cudaMemcpy` has a fixed overhead that dominates when the pcap is small, making the GPU appear slower than the CPU on tiny captures even though the kernel itself is faster.

---

## Accomplishments that we're proud of

- **~4,960 MB/s throughput** on a consumer GPU — 59× the CPU baseline and 3.3× Intel Hyperscan, the state-of-the-art software SIMD matcher, on identical inputs producing identical alerts.
- A **complete end-to-end open-source pipeline**: pcap in, alerts out, benchmarks, a live public dashboard — not just an isolated kernel experiment.
- **Honest benchmarking** — we included Hyperscan as a comparison target rather than only comparing against a textbook CPU loop, which is the common paper shortcut. Beating Hyperscan is the credible result.
- A **publicly accessible live demo** at cudashield.tech, running on real malware traffic on Caltech hardware.
- Building the entire thing in C++ and CUDA with no runtime Python dependency, which made deployment on the cluster straightforward.

---

## What we learned

Working on CudaShield taught us far more about computer networks than we expected going in. To implement TCP reassembly — even partially — we had to understand 5-tuples, sequence numbers, the SYN/FIN/RST handshake, out-of-order segment buffering, and how a real flow table works. Studying actual malware pcaps from malware-traffic-analysis.net gave us a window into how modern C2 frameworks (TrickBot, Cobalt Strike) structure their traffic: the characteristic URI patterns, the user-agent strings, the beacon cadence, the DNS requests. It made the security problem feel concrete rather than academic.

On the GPU side, the biggest lesson was how sharp the setup overhead cliff is. **For small inputs, GPUs lose.** The `cudaMemcpy` round-trip to device and back is a fixed cost that can easily exceed the time you save parallelizing the computation. We saw this clearly: on small pcap files the GPU gave little or no gain over CPU, but on large captures it pulled ahead dramatically. That taught us that GPU acceleration is not a free win — it requires feeding the device enough work to amortize the transfer cost, which has direct implications for how a real deployment would need to be designed (batching, streaming, pipeline overlap).

We also deepened our understanding of memory hierarchy in CUDA — the difference between global memory, `__ldg` read-only cache, and shared memory is not just theoretical when a 100 KB DFA table is sitting right at the edge of what fits, and whether it fits determines whether you get 4-cycle or 200-cycle lookups on every single DFA transition.

---

## What's next for CudaShield

**Beacon detection calibration on real traffic.** The current beacon kernel is validated on synthetic periodic flows, but real malware traffic is noisier. In several case-study pcaps, an IAT coefficient-of-variation threshold of 0.15 is too strict. Next we need to evaluate a larger real-world pcap corpus, tune the threshold empirically, and add more features beyond IAT alone (for example burstiness, flow duration stability, and packet-size statistics) for robust beacon flagging.

**Live streaming input.** Right now CudaShield processes static pcap files. A real deployment would sit in a network — between the firewall and the rest of the infrastructure — receiving a continuous stream of live packets. The natural next step is a live capture loop using `pcap_loop` or `AF_PACKET` sockets, with a producer-consumer pipeline that keeps the GPU fed continuously while results are read out asynchronously.

**TLS/JA3 fingerprinting.** Most modern C2 traffic is encrypted, so pattern matching on the payload is blind to it. But TLS handshakes are not encrypted — the ClientHello message exposes cipher suite lists, extension types, and elliptic curve preferences in plaintext. JA3 fingerprinting hashes these fields into a short string that reliably identifies malware clients even though the payload is opaque. Parsing TLS handshake metadata and computing JA3 hashes is a natural GPU workload — one thread per connection, zero decryption required.

**ML anomaly detection as a second stage.** The signature engine (PFAC) is fast, low-latency, and high-confidence for known threats. A neural network behavioral scorer would complement it for unknown threats: run PFAC first to flag obvious matches, then feed suspicious flows into a small model that scores behavioral features (packet timing, size distribution, flow duration, connection graph). NVIDIA Morpheus does something similar in production. Both stages are GPU-parallelizable and the handoff between them is a natural pipeline boundary.

**Larger rule sets.** We tested with just some hand-curated patterns. The Emerging Threats open ruleset has ~40,000 rules. Scaling the DFA to that size would require profiling memory layout carefully — the table would no longer fit in shared memory and the `__ldg` + L2 cache hit rate would become the critical variable.

**Alert correlation across flows and time windows.** Today each packet's alerts are independent. A real IDS correlates: the same source IP hitting many patterns across many flows over a time window is a more confident detection than a single hit. Parallel reductions over a sliding time window are a natural extension of the flow statistics kernel already in the codebase.
