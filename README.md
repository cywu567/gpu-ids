# gpu-ids: GPU-Accelerated Intrusion Detection System

A multi-pattern network packet scanner running on a consumer GPU. Scans pcap captures for known-bad byte signatures, in the style of Snort/Suricata but with the matching engine on a GPU instead of a CPU.

Hackathon weekend: naive brute-force kernel, real throughput numbers vs. a CPU baseline.
Class project (CS179): PFAC Aho-Corasick kernel **(implemented)** + TCP flow reassembly **(CPU path implemented, GPU path implemented)** + Hyperscan comparison **(implemented)**.

---

## Quick start

### Dependencies

No sudo needed. Uses a conda environment for libpcap and cmake.

```bash
# Install miniconda if you don't have it (one-time, no sudo)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b    # -b = non-interactive
source ~/miniconda3/etc/profile.d/conda.sh

# Create the project environment (reads environment.yml)
conda env create -f environment.yml
conda activate gpu-ids
```

CUDA Toolkit must be installed separately (needs admin on a bare machine, but
is already present on shared GPU clusters):
```bash
nvcc --version   # verify
```

macOS has no NVIDIA GPU -- use [Google Colab](https://colab.research.google.com/) (T4 free tier) or the Caltech CS179 cluster.

### Build

**CMake (with conda env active)**
```bash
conda activate gpu-ids
cmake -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX   # lets CMake find conda's libpcap + vectorscan
cmake --build build -j$(nproc)
```

Binaries land in `build/`: `ids`, `benchmark`, `gen_synthetic`.

### Generate a synthetic test pcap

```bash
./build/gen_synthetic data/synthetic.pcap 10000 50
# Writes 10 000 benign + 50 malicious packets.
```

Then update `src/gen_synthetic.cpp` malicious payload strings to match your final `patterns/rules.txt`.

### Run the demo

```bash
# CPU only
./build/ids --pcap data/demo.pcap --rules patterns/rules.txt --cpu

# GPU naive kernel (one thread per packet×pattern)
./build/ids --pcap data/demo.pcap --rules patterns/rules.txt --gpu

# GPU PFAC Aho-Corasick kernel (one thread per start byte, single DFA)
./build/ids --pcap data/demo.pcap --rules patterns/rules.txt --pfac

# CPU + GPU naive side by side
./build/ids --pcap data/demo.pcap --rules patterns/rules.txt --both

# TCP reassembly then PFAC match (defeats split-payload evasion)
./build/ids --pcap data/demo.pcap --rules patterns/rules.txt --pfac --reassemble
```

`--gpu` uses the **naive** brute-force kernel (limited to 1 024 patterns).
`--pfac` uses the **PFAC Aho-Corasick** kernel — no pattern limit, ~4 960 MB/s.
`--reassemble` parses Ethernet/IP/TCP headers and stitches TCP streams before matching.

### Run benchmarks

```bash
# CPU vs GPU naive
./build/benchmark \
    --pcap  data/demo.pcap \
    --rules patterns/rules.txt \
    --iters 5

# Add --pfac to also time the PFAC Aho-Corasick kernel
./build/benchmark \
    --pcap  data/demo.pcap \
    --rules patterns/rules.txt \
    --iters 5 --pfac

# Add --hyperscan to also time Vectorscan (Hyperscan fork)
./build/benchmark \
    --pcap  data/demo.pcap \
    --rules patterns/rules.txt \
    --iters 5 --pfac --hyperscan
```

All flags for `benchmark`:

| Flag | Description |
|------|-------------|
| `--pcap PATH` | Input pcap file |
| `--rules PATH` | Pattern file |
| `--iters N` | Timing iterations (default 1) |
| `--csv PATH` | Write results to CSV |
| `--pfac` | Also time the PFAC Aho-Corasick GPU kernel |
| `--hyperscan` | Also time Vectorscan (requires `HAVE_HYPERSCAN`) |

### Run the web UI

Start the server on the remote machine (default port 8080):

```bash
./build/ids --web 8080 --pcap data/demo.pcap --rules patterns/rules.txt
```

The dashboard at **https://cudashield.tech** shows three bars: CPU (yellow), GPU PFAC (green),
Hyperscan (purple), and a live speedup ratio.

**Option A — Local access via SSH port forwarding** (no tunnel needed):

In a separate terminal on your laptop:
```bash
ssh -L 8080:localhost:8080 user@remote-machine
```
Then open [http://localhost:8080](http://localhost:8080).

**Option B — Public access via Cloudflare Tunnel** (see [Cloudflare Tunnel setup](#cloudflare-tunnel-setup) below):

In a second terminal on the server, start the tunnel:
```bash
~/cloudflared tunnel run cudashield
```
Then open [https://cudashield.tech](https://cudashield.tech) from any browser.

> **Chrome users:** if you see `ERR_QUIC_PROTOCOL_ERROR`, go to Cloudflare dashboard →
> cudashield.tech → **Speed → Optimization → Protocol Optimization** and turn off
> **HTTP/3 (with QUIC)**. Alternatively, go to `chrome://flags/#enable-quic`, set to
> Disabled, and fully relaunch Chrome (Cmd+Q).

---

## Cloudflare Tunnel setup

The server runs on a shared GPU cluster where the firewall blocks inbound
connections on non-standard ports. Cloudflare Tunnel creates an outbound-only connection
so the public domain `cudashield.tech` routes through Cloudflare to the server without
needing firewall changes or root access.

### One-time setup (already done — for reference)

**1. Cloudflare account and domain**

- Register `cudashield.tech` at [get.tech](https://get.tech)
- Create a free Cloudflare account and add `cudashield.tech` as a site
- In Cloudflare, get the two assigned nameservers and set them at get.tech under **Nameservers**
- Wait for Cloudflare to confirm the domain is active (email notification)

**2. Install cloudflared on the server** (no root needed)

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O ~/cloudflared
chmod +x ~/cloudflared
~/cloudflared --version   # verify
```

**3. Authenticate**

```bash
~/cloudflared tunnel login
```

Open the printed URL in a browser and select `cudashield.tech`. A certificate is saved to
`~/.cloudflared/` automatically.

**4. Create the tunnel**

```bash
~/cloudflared tunnel create cudashield
```

Note the tunnel ID printed (e.g. `66667b15-9f4d-4baa-a00e-77bfba54d11f`).

**5. Write the config file**

```bash
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: <your-tunnel-id>
credentials-file: /home/<username>/.cloudflared/<your-tunnel-id>.json
ingress:
  - hostname: cudashield.tech
    service: http://localhost:8080
  - service: http_status:404
EOF
```

**6. Route the domain to the tunnel**

```bash
~/cloudflared tunnel route dns cudashield cudashield.tech
```

This creates a CNAME record in Cloudflare DNS pointing `cudashield.tech` to the tunnel.
If you get an error about a conflicting A record, delete the existing A record in the
Cloudflare DNS dashboard first, then re-run.

### Starting the tunnel

Every time you want the public URL to work, run both of these (in separate terminals):

```bash
# Terminal 1 — IDS server
cd ~/gpu-ids
./build/ids --web 8080 --pcap data/2025-06-13-traffic-analysis-exercise.pcap --rules patterns/rules.txt

# Terminal 2 — Cloudflare tunnel
~/cloudflared tunnel run cudashield
```

The tunnel log should show `Registered tunnel connection` for 4 connections and then stay
idle. The site is live at **https://cudashield.tech**.

---

## File structure

```
gpu-ids/
├── CMakeLists.txt              build config
├── environment.yml             conda env (libpcap + vectorscan, no sudo needed)
├── data/
│   └── demo.pcap               pcap for benchmarking
├── patterns/
│   └── rules.txt               one pattern per line; '#' = comment
└── src/
    ├── load_pcap.cpp/h         libpcap loader → flat byte buffer + packet offsets
    ├── cpu_matcher.cpp/h       reference CPU matcher (memmem baseline ~84 MB/s)
    ├── gpu_matcher.cu/cuh      naive kernel + PFAC Aho-Corasick kernel
    │                             naive: one thread per (packet, pattern)
    │                             PFAC:  uint16_t DFA, uint8_t hits, ~4 960 MB/s
    ├── flow_reassembly.cu/cuh  TCP stream reassembly
    │                             CPU:  unordered_map flow table, gap buffering
    │                             GPU:  atomicCAS hash table + thrust prefix-sum
    ├── benchmark.cpp           timed runs; writes CSV
    ├── gen_synthetic.cpp       builds synthetic pcap with known hits
    ├── run_demo.cpp            CLI entry point (ids binary)
    ├── web_server.cpp/h        HTTP dashboard on port 8080
    └── httplib.h               single-header HTTP server library
```

---

## Pattern file format

`patterns/rules.txt` -- one plaintext byte string per line:

```
# comment
GET /cgi-bin/
../../etc/passwd
cmd.exe
```

The naive GPU kernel is limited to 1024 patterns (CUDA max-threads-per-block). Use `--pfac` in the benchmark, or the PFAC path in code, to remove this constraint.

---

## How the kernels work

### Naive kernel

Grid: one block per packet, one thread per pattern (`blockIdx.x` = packet, `threadIdx.x` = pattern).
Each thread does a brute-force sliding-window search for its single pattern in its packet.
Complexity: O(pkt_len × pat_len) per thread. Hard cap of 1 024 patterns (CUDA max threads/block).

### PFAC kernel (Parallel Failureless Aho-Corasick)

All patterns are compiled offline into a single DFA (trie with no failure links):
- State 0 = dead, state 1 = root.
- `table[state * 256 + byte]` gives the next state.
- Accepting states record which pattern matched.
- Stored as `uint16_t` to cut table size in half (~140 KB for 26 patterns).

At scan time: one block per packet, 256 threads per block.
Each thread owns start positions `threadIdx.x, threadIdx.x+256, ...` within the packet.
Every thread walks the DFA forward from root at its start byte until hitting dead state 0.

Because every byte position gets a fresh start, no failure links are needed at runtime — there
is nothing to "fall back" to. This is the "failureless" property.

Complexity: O(pkt_len) per thread, O(1) in pattern count. Measured throughput ~4 960 MB/s
vs ~84 MB/s CPU baseline and ~1 490 MB/s Hyperscan (vectorized SIMD CPU matcher).

---

## TCP flow reassembly

An attacker can split a malicious payload across multiple TCP segments, each individually
innocent. Without reassembly the matcher misses the pattern; with reassembly it sees the
complete application stream.

### CPU path (`reassemble_segment`)

- Flow table: `std::unordered_map<FlowKey, FlowBuffer>` keyed by 5-tuple.
- In-order segments are appended immediately; out-of-order segments are stashed in a
  `std::map<uint32_t, bytes>` gap buffer keyed by sequence number.
- When stashed segments become consecutive they are drained into the main buffer automatically.
- On FIN or RST the completed stream is returned as a flat `std::vector<uint8_t>` and the
  flow entry is removed.

### GPU path (`flow_insert_kernel` + `mark_complete_kernel` + `compact_flows_kernel`)

- Open-addressing hash table (`GpuFlowSlot[]`) lives on device.
- One warp per segment: lane 0 drives `atomicCAS` linear probing to claim/find the flow's
  slot; all 32 lanes cooperate to copy the payload at `buf_offset + (seq - init_seq)`,
  which handles both in-order and out-of-order segments without CPU-side synchronization.
- `mark_complete_kernel` flags slots where `fin_seen == 1`.
- `compact_flows_kernel` uses `thrust::exclusive_scan` prefix-sum offsets to scatter
  finished flows into a contiguous buffer fed directly into `run_pfac_match_gpu`.
- Each flow's payload buffer is pre-allocated at a fixed `MAX_FLOW_BYTES`; flows exceeding
  this are truncated.

---

## Demo script (90 seconds)

1. "IDS systems check every packet against thousands of patterns. CPUs struggle at high speed. Enterprises pay $50k for FPGA appliances. We did it on a consumer GPU."
2. Show the pcap: `ls -lh data/demo.pcap`
3. CPU: `./build/ids --pcap data/demo.pcap --rules patterns/rules.txt --cpu`
4. GPU naive: `./build/ids --pcap data/demo.pcap --rules patterns/rules.txt --gpu`
5. "Same alerts, Nx faster. That's the naive kernel -- one thread per (packet, pattern) pair."
6. Benchmark all three: `./build/benchmark --pcap data/demo.pcap --rules patterns/rules.txt --iters 3 --pfac --hyperscan`
7. "PFAC Aho-Corasick compiles all patterns into a single DFA -- one thread per start byte, O(packet_length) regardless of pattern count. ~5 GB/s vs Hyperscan's ~1.5 GB/s."
8. "We also implemented TCP flow reassembly so split-payload evasion attacks don't fool us."

---

## References

- Aho & Corasick (1975). *Efficient string matching.* CACM.
- Lin, Liu, Chang (2013). *Accelerating Pattern Matching Using a Novel Parallel Algorithm on GPUs.* IEEE Trans. Computers. **(primary kernel reference)**
- [PFAC reference implementation](https://github.com/pfac-lib/PFAC)
- [Hyperscan](https://www.hyperscan.io/) / [Vectorscan](https://github.com/VectorCamp/vectorscan) -- SIMD multi-pattern CPU matcher; our GPU PFAC runs ~3.3× faster
- [malware-traffic-analysis.net](https://malware-traffic-analysis.net/) -- real malware pcap source
