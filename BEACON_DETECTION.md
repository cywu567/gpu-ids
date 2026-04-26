# Motivation and Beacon Detection — Ground-Up Overview

## 1. The Problem: What Is a Network Intrusion?

Your computer doesn't talk directly to other computers — it sends data in small chunks called **packets**. Every packet has a **header** (metadata: who it's from, who it's going to, what port, what protocol) and a **payload** (the actual bytes of data being sent).

A **pcap file** is a recording of all packets that crossed a network interface during some time window — like a security camera recording, but for network traffic.

An **Intrusion Detection System (IDS)** watches those packets live (or replays them from a pcap) and looks for signatures of known attacks. For example: if a packet payload contains `../../etc/passwd`, someone is almost certainly trying a **path traversal attack** — attempting to read the Unix password file. The IDS fires an alert.

### Why is this hard?

A fast corporate network might push **10 gigabits/second**. That's roughly 1.25 billion bytes every second passing through, and the IDS has to check every packet against **thousands of patterns** before the next packet arrives. CPUs are fast but not this fast — the project's CPU baseline runs at ~84 MB/s. Enter the GPU.

`patterns/rules2.txt` shows what signatures look like: 240+ lines covering path traversal (`../../etc/passwd`), web shells (`<?php`, `eval(base64_decode`), command execution (`cmd.exe`, `powershell.exe`), SQL injection (`UNION%20SELECT`), and real malware strings from actual captures (`NetSupport Manager`, `/fakeurl.htm`).

---

## 2. The Gap Pattern Matching Cannot Fill

Here is the core problem beacon detection solves: **pattern matching is blind to encrypted traffic**.

The real-world NetSupport RAT analysis (`pcap-tshark-analysis/pcap_2026-02-28_netsupport.md`) shows a malware that sent 266 beacons to `45.131.214.85` over 4.3 hours. Each beacon's payload was RC4-encrypted:

```
CMD=ENCD\nES=1\nDATA=<rc4-encrypted-binary>
```

If the payload is encrypted, patterns like `NetSupport Manager` and `/fakeurl.htm` no longer appear as plaintext — the PFAC kernel produces zero hits. Yet the malware is actively running and exfiltrating data.

What gives it away when you can't read its contents? **Timing.** The malware phoned home every exactly **60 seconds**. That rigid regularity is structurally unmistakable: no human generates traffic that perfectly periodic. This is called **beaconing**, and it is the universal fingerprint of C2 (command-and-control) malware.

---

## 3. What Is a "Flow" and a "5-tuple"?

Before detecting beaconing, you need to group packets by **conversation**. One flow = one logical connection between two hosts. It is uniquely identified by a **5-tuple**:

```
(source IP, destination IP, source port, destination port, protocol)
```

For example: `10.2.28.88:50123 → 45.131.214.85:443 / TCP` is one flow. This is the exact structure at `flow_stats.cuh:36`:

```cpp
struct FSTuple {
    uint32_t src_ip, dst_ip;
    uint16_t sport, dport;
    uint8_t  proto;
};
```

All 266 NetSupport beacons share the same 5-tuple (same source/dest IP and port), so they all land in the same flow — and the timing pattern becomes visible.

---

## 4. Phase 1: CPU-Side Flow Grouping (`flow_stats.cu:45`)

`group_flows_by_5tuple` runs on the CPU:

1. Iterates every packet in the pcap.
2. Manually parses the binary Ethernet → IP → TCP/UDP headers to extract the 5-tuple.
3. Inserts each packet index into a hash map keyed on its 5-tuple.
4. Sorts each flow's packets by timestamp.

The header parsing at lines 54–79 walks the fixed-size structs overlaid directly on the raw bytes. The IP header's `ver_ihl` byte encodes the header length in its lower 4 bits:

```cpp
int ip_hdr_len = (ip->ver_ihl & 0x0F) * 4;
```

`ntohs` / `ntohl` convert from big-endian network byte order to the CPU's native byte order.

The output is a `FlowGrouping`: a flat array of packet indices sorted by flow, plus an offset array so any flow's packets can be sliced out in O(1). This mirrors the `PcapData` / `PatternSet` layout — a single `cudaMemcpy` is sufficient to move it to the GPU.

---

## 5. The Beacon Signal: Inter-Arrival Time CV

The key metric is **IAT — Inter-Arrival Time**: the time gap between consecutive packets in the same flow. If a flow has packets at t=0s, t=60s, t=120s, t=180s..., all IATs are exactly 60s.

To quantify regularity, the code uses the **Coefficient of Variation (CV)**:

```
CV = standard_deviation / mean
```

- CV near 0 → extremely regular → likely beacon
- CV near 1 or above → irregular → normal human/application traffic

The threshold is at `flow_stats.cuh:29`:

```cpp
static constexpr float BEACON_CV_THRESH = 0.15f;  // IAT CV below this → beacon
static constexpr int   MIN_BEACON_PKTS  = 5;       // ignore tiny flows
```

A flow is flagged `is_beacon` if CV < 0.15 and it has at least 5 packets. The NetSupport RAT at exactly 60s intervals would have CV ≈ 0.

---

## 6. Phase 2: GPU Kernel — One Block Per Flow (`flow_stats.cu:131`)

The kernel launches with one CUDA threadblock per flow and 256 threads per block (`BLOCK_SIZE = 256`).

### What each thread does

Each thread handles a **stripe** of packets within its flow. Thread 0 handles packets 0, 256, 512...; thread 1 handles packets 1, 257, 513... (line 159):

```cpp
for (int t = threadIdx.x; t < n; t += BLOCK_SIZE) {
    int pkt_i = flat_idx[f_start + t];
    float sz  = (float)sizes[pkt_i];
    l_sz_sum += sz;
    l_sz_sq  += sz * sz;

    if (t > 0 && t % BLOCK_SIZE != 0) {
        // compute IAT from previous packet
        float iat = (float)(timestamps_us[pkt_i] - timestamps_us[pkt_prev]);
        l_iat_sum += iat;
        l_iat_sq  += iat * iat;
    }
}
```

Each thread accumulates local sums: packet sizes, squared sizes, IATs, and squared IATs.

### Why sum-of-squares?

Variance can be computed in one pass using the identity:

```
Var(x) = E[x²] - E[x]²
```

That is exactly line 206:

```cpp
float iat_var = (iat_sq / (float)n_iat) - iat_mean * iat_mean;
```

No second pass over the data needed.

### Reduction: collapsing 256 values into 1

After the strided loop, 256 threads each hold a partial sum. These need to be summed together.

**Stage 1 — shared memory tree reduction** (lines 183–190): threads write to `sh_iat_sum[threadIdx.x]`, then repeatedly, threads in the lower half add from threads in the upper half, halving the active count each iteration. After log₂(256) = 8 steps, 32 values remain.

**Stage 2 — warp shuffle** (`warp_reduce_sum`, line 117): the final 32 threads are one **warp** — a group of 32 threads that execute in lockstep on the GPU. `__shfl_down_sync` lets thread i read thread i+offset's register directly (no shared memory write needed), collapsing 32 → 1 in about 4 cycles each step:

```cpp
for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xFFFFFFFF, v, offset);
```

Thread 0 ends up with the grand total and writes to `out[flow_id]`.

### Back on the CPU (lines 271–287)

After `cudaMemcpy` brings results back, the host:
- Converts IAT mean from microseconds to milliseconds
- Computes `iat_cv = stddev / mean`  (stddev = sqrt of variance)
- Sets the `is_beacon` flag

---

## 7. End-to-End Pipeline

```
pcap file
    │
    ▼
load_pcap()
    → flat packet bytes + per-packet timestamps
    │
    ▼
group_flows_by_5tuple()             [CPU]
    → parse Ethernet/IP/TCP/UDP headers
    → hash-map by 5-tuple
    → sort each flow by timestamp
    → FlowGrouping (flat indices + offsets)
    │
    ▼
run_flow_stats_gpu()                [GPU]
    → cudaMemcpy to device
    → flow_stats_kernel<<<num_flows, 256>>>
        each thread: accumulate IAT sums + size sums (strided loop)
        tree reduce over shared memory (256 → 32)
        warp shuffle (32 → 1)
        thread 0 writes FlowStatGpu result
    → cudaMemcpy back
    │
    ▼
FlowStatResult[]
    → iat_cv < 0.15 && packet_count ≥ 5  →  is_beacon = true
```

**Example output for the NetSupport capture:**
```
BEACON: 10.2.28.88 → 45.131.214.85:443  (266 pkts, IAT mean 60000 ms, CV 0.01)
```

---

## 8. Why Both Systems Together?

Pattern matching and beacon detection are **complementary**:

| | Pattern Matching (PFAC) | Beacon Detection |
|---|---|---|
| Catches | Plaintext attack signatures | Periodic / automated C2 traffic |
| Blind to | Encrypted payloads | Non-periodic attacks (port scans, one-shot exploits) |
| Signal | Byte content | Packet timing |

A sophisticated C2 using TLS would defeat pattern matching with zero hits. But the fixed heartbeat interval — a structural property of how the malware is programmed — survives encryption entirely. Combining both gives coverage that neither provides alone.
