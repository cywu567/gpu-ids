# Hackathon Plan — GPU Pattern Matcher

48-72 hour build plan. Goal: by Sunday afternoon, a working CLI demo that scans a pcap on GPU faster than a CPU baseline.

---

## The 60-second pitch (for judges)

"Network intrusion detection systems like Snort scan every packet on a network for thousands of malicious byte patterns. They run on CPUs and struggle to keep up at high speeds — enterprises buy specialized $50k FPGA hardware to handle it. We built the same thing on a consumer GPU. Our prototype scans a real malware traffic capture and matches patterns Nx faster than the CPU baseline. The 4-week class extension turns the naive matcher into a proper Aho-Corasick implementation and adds TCP flow reassembly."

---

## What we're building this weekend

A C++ command-line program that:

1. Loads packets from a pcap file using libpcap.
2. Runs a CUDA kernel that checks every packet for every pattern in a hand-picked rule set.
3. Prints alerts when matches are found.
4. Reports throughput in MB/s, side-by-side with a single-threaded CPU baseline doing the same work.

That's the whole demo.

---

## What we are explicitly NOT doing this weekend

Cut these ruthlessly. They belong in the 4-week project.

- No real Aho-Corasick. Each thread does a dumb sliding-window comparison.
- No TCP flow reassembly. Each packet is independent.
- No Snort rule parsing. Hand-curate ~20-50 patterns.
- No anomaly detection.
- No live network capture. Pcap files only.
- No fancy UI. CLI is fine.
- No multi-GPU.
- No optimization beyond "make it correct."

---

## Tech stack

- **CUDA C++** — GPU kernel.
- **C++17** — everything else.
- **libpcap** — pcap reading and (optionally) writing.
- **CMake or Makefile** — build system. CMake is more standard; Makefile is faster to write.
- **gnuplot** (optional) — for chart generation Sunday.

No Python anywhere in the runtime path.

---

## Setup (Friday night, both partners)

### Install dependencies

On Ubuntu / Debian / WSL:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libpcap-dev
# Install CUDA toolkit per your GPU/distro (https://developer.nvidia.com/cuda-downloads)
nvcc --version    # verify
```

On macOS (Apple Silicon has no NVIDIA GPU; use Colab or a remote box):
- Use [Google Colab](https://colab.research.google.com/) with a T4. Upload your code or sync via GitHub.

### Verify GPU works

```cpp
// hello.cu
#include <cstdio>
__global__ void hi() { printf("hi from thread %d\n", threadIdx.x); }
int main() { hi<<<1, 4>>>(); cudaDeviceSynchronize(); return 0; }
```

```bash
nvcc hello.cu -o hello && ./hello
# Should print "hi from thread 0/1/2/3" in some order.
```

If that works, you have a usable dev environment. Stop here on Friday night.

---

## Minimal file structure (weekend version)

```
gpu-ids-hack/
├── Makefile
├── data/
│   └── test.pcap          ← downloaded malware capture
├── patterns/
│   └── rules.txt          ← one pattern per line
├── src/
│   ├── load_pcap.cpp
│   ├── load_pcap.h
│   ├── cpu_matcher.cpp
│   ├── cpu_matcher.h
│   ├── gpu_matcher.cu
│   ├── gpu_matcher.cuh
│   └── main.cpp           ← CLI entry point + benchmark + alerts
└── README.md
```

Seven source files. Don't add more this weekend.

---

## Starter Makefile

```makefile
NVCC      := nvcc
CXX       := g++
CXXFLAGS  := -O2 -std=c++17
NVCCFLAGS := -O2 -std=c++17 -arch=sm_60
LDFLAGS   := -lpcap -lcudart

OBJS := build/load_pcap.o build/cpu_matcher.o build/gpu_matcher.o build/main.o

all: build/ids

build/:
	mkdir -p build

build/%.o: src/%.cpp | build/
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/%.o: src/%.cu | build/
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

build/ids: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf build/
```

Adjust `-arch=sm_60` to your GPU's compute capability if needed (sm_75 for Turing, sm_86 for Ampere consumer cards, sm_89 for Ada).

---

## The naive kernel (copy-pasteable starting point)

```cpp
// gpu_matcher.cu
#include "gpu_matcher.cuh"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void naive_match_kernel(
    const char* input,
    const int*  packet_offsets,    // length num_packets + 1
    int         num_packets,
    const char* patterns,
    const int*  pattern_offsets,   // length num_patterns + 1
    int         num_patterns,
    int*        hits               // size num_packets * num_patterns, zero-initialized
) {
    int packet_idx  = blockIdx.x;
    int pattern_idx = threadIdx.x;
    if (packet_idx >= num_packets || pattern_idx >= num_patterns) return;

    const char* pkt     = input    + packet_offsets[packet_idx];
    int         pkt_len = packet_offsets[packet_idx + 1] - packet_offsets[packet_idx];
    const char* pat     = patterns + pattern_offsets[pattern_idx];
    int         pat_len = pattern_offsets[pattern_idx + 1] - pattern_offsets[pattern_idx];

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

void run_naive_match_gpu(
    const char* h_input, const int* h_offsets, int num_packets,
    const char* h_patterns, const int* h_pat_offsets, int num_patterns,
    int* h_hits, size_t input_len, size_t patterns_len
) {
    char *d_input, *d_patterns;
    int  *d_offsets, *d_pat_offsets, *d_hits;

    cudaMalloc(&d_input,       input_len);
    cudaMalloc(&d_offsets,     (num_packets + 1) * sizeof(int));
    cudaMalloc(&d_patterns,    patterns_len);
    cudaMalloc(&d_pat_offsets, (num_patterns + 1) * sizeof(int));
    cudaMalloc(&d_hits,        num_packets * num_patterns * sizeof(int));

    cudaMemcpy(d_input,       h_input,       input_len,                          cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets,     h_offsets,     (num_packets + 1) * sizeof(int),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_patterns,    h_patterns,    patterns_len,                       cudaMemcpyHostToDevice);
    cudaMemcpy(d_pat_offsets, h_pat_offsets, (num_patterns + 1) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemset(d_hits, 0, num_packets * num_patterns * sizeof(int));

    naive_match_kernel<<<num_packets, num_patterns>>>(
        d_input, d_offsets, num_packets,
        d_patterns, d_pat_offsets, num_patterns,
        d_hits
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_hits, d_hits, num_packets * num_patterns * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input); cudaFree(d_offsets);
    cudaFree(d_patterns); cudaFree(d_pat_offsets);
    cudaFree(d_hits);
}
```

Bad code on purpose. Make it correct first; we'll improve it for the class project.

**Note:** `<<<num_packets, num_patterns>>>` caps `num_patterns` at 1024 (max threads per block). Fine for the weekend with ~50 patterns. Generalize later.

---

## Hour-by-hour plan

### Friday night (target: 4 hours, 6pm-10pm)

| Time | Both | Person A | Person B |
|------|------|----------|----------|
| 6-7 | Install CUDA, libpcap, get hello.cu running | | |
| 7-8 | Set up shared git repo, get Makefile compiling an empty `main.cpp` + dummy `.cu` | | |
| 8-10 | | Stub out `load_pcap.cpp` — open a pcap, iterate packets, print sizes | Download 2-3 pcaps from malware-traffic-analysis.net. Open them in Wireshark. Identify clean indicators. Write `patterns/rules.txt`. |

**End of Friday checkpoint:** `make` succeeds. `./build/ids` runs and prints something. You can read packets out of a real pcap. You have a list of patterns.

### Saturday (target: 10-12 hours)

**Morning (9am-1pm):**

| Person A | Person B |
|----------|----------|
| Finish `load_pcap.cpp` — produce flat byte buffer + offset array | Write `cpu_matcher.cpp` — for each packet, for each pattern, `memmem` |
| Write `gpu_matcher.cu` from the template above | Wire `main.cpp` to call `cpu_matcher` and print alerts |
| Verify CPU and GPU produce identical hit arrays on a small synthetic input | |

**Mid-Saturday checkpoint (1pm):** CPU matcher works correctly. GPU matcher works correctly. They agree on a small test.

**Afternoon (2pm-6pm):**

| Person A | Person B |
|----------|----------|
| Add `std::chrono` timing around both matcher calls | Build the alert printer: walk the `hits[]` array, print `[ALERT] pattern "X" found in packet #N` |
| Run on the real pcap. Verify alerts fire on known malicious patterns | Add CLI args: `--pcap PATH --rules PATH --gpu/--cpu/--both` |

**Evening (7pm-10pm):**
- Together: integrate everything into a single demo command.
- Together: run on multiple pcaps, hand-tune patterns until you reliably get a few visible alerts.
- Together: capture some throughput numbers in a notebook for the morning.

**End of Saturday checkpoint:** `./build/ids --pcap data/test.pcap --rules patterns/rules.txt --both` prints alerts and shows side-by-side throughput.

### Sunday (target: 6-8 hours, leave time for breakage)

| Time | What |
|------|------|
| 9-12 | Polish. Better alert formatting. Color output (`\033[31m` for red alerts). Throughput summary at end. README. |
| 12-2 | If brave: write a simple gnuplot script that turns benchmark CSV into a chart for the demo. |
| 2-4 | Buffer time for things breaking. They will. |
| 4-5 | Practice the demo. Time it. |
| 5+ | Write the proposal draft for CS179 (separate doc). |

**End of Sunday:** working demo + a notebook of throughput numbers + a proposal draft for the class.

---

## Partner split summary

**Partner A — the engine:**
- `gpu_matcher.cu` (the kernel + host-side launcher)
- `cpu_matcher.cpp` (the baseline)
- Timing logic
- Owns "input bytes + patterns → hits array"

**Partner B — the wrapper:**
- `load_pcap.cpp` (libpcap → byte buffer + offsets)
- `main.cpp` (CLI, orchestration, alert printing)
- Pattern curation (hand-picking strings from real malware pcaps)
- Owns "files on disk → human-readable alerts"

**Interface (`gpu_matcher.cuh`):**
```cpp
struct Hit { int packet_idx; int pattern_idx; };

void run_naive_match_gpu(
    const char* input, const int* packet_offsets, int num_packets,
    const char* patterns, const int* pattern_offsets, int num_patterns,
    int* hits, size_t input_len, size_t patterns_len
);
```

Once that header is locked Friday night, both partners can work without blocking each other Saturday.

---

## Pattern starter pack

Drop these in `patterns/rules.txt`, one per line, then add specific indicators from whatever pcap you pick:

```
GET /cgi-bin/
../../etc/passwd
/wp-login.php
/wp-admin/
/phpmyadmin/
SELECT * FROM
UNION SELECT
<script>
javascript:
cmd.exe
powershell
/bin/sh
nc -e
wget http://
curl http://
User-Agent: Mozilla/4.0 (compatible; MSIE
PASSWORD=
api_key=
authorization: Bearer
```

These are common HTTP/exploit indicators. Mix in 5-10 strings you actually see in your chosen malware pcap (find them in Wireshark by browsing the HTTP streams).

---

## Pcap suggestions

Browse [malware-traffic-analysis.net](https://malware-traffic-analysis.net/). Pick captures with:

- Clear HTTP traffic (easier to find readable patterns)
- Documented indicators in the writeup (so you know what to look for)
- Reasonable size (5-50 MB; multi-GB pcaps are overkill for a weekend)

Good families to look for: Emotet, TrickBot, Qakbot, IcedID, Cobalt Strike. Their writeups list URLs and User-Agents you can grep for.

Also generate a synthetic pcap with planted strings as a backup demo file — guarantees alerts fire even if the malware capture's traffic doesn't go where you expect on stage.

---

## Demo script (for the judges)

90 seconds, max:

1. **Hook (15s):** "Network intrusion detection means scanning every packet for thousands of bad patterns. CPUs can't keep up at high speed. Enterprises spend $50k on FPGAs. We did it on a consumer GPU."
2. **Show the input (15s):** "This is real malware traffic from a TrickBot infection — about 30 MB, 50,000 packets."
3. **Run CPU (15s):** `./build/ids --pcap data/trickbot.pcap --rules patterns/rules.txt --cpu` → "12 alerts, 220 MB/s."
4. **Run GPU (15s):** `./build/ids ... --gpu` → "Same 12 alerts. 8.4 GB/s. ~38× faster."
5. **The arc (15s):** "This is the naive version — every thread brute-forces. The 4-week class extension swaps in Aho-Corasick and adds TCP flow reassembly. We expect another order of magnitude."
6. **Close (15s):** Q&A.

Practice this Sunday afternoon. Do not improvise on stage.

---

## Common time-wasters to avoid

- **Trying to parse Snort rule files.** Just put plain strings in `rules.txt`. One per line. Read with `std::getline`.
- **Building a web UI.** CLI is enough. If you really want one Sunday afternoon, drop in cpp-httplib, but only if everything else is done.
- **Optimizing the kernel.** It will be slow. That's the *point* — naive vs. PFAC is the 4-week story arc. Resist optimizing this weekend.
- **Handling malformed pcaps.** If a pcap doesn't parse, pick a different one. Don't write defensive code.
- **Supporting more than HTTP / TCP.** Just process the whole packet payload as bytes. Don't parse protocols.
- **Overlapping pattern detection.** If a packet matches the same pattern twice, recording it once is fine.
- **Free'ing memory perfectly.** It's a 48-hour demo. Leak a little.

---

## Stretch goals (only if everything works by Saturday night)

In rough order of "easy and impressive":

1. **Color the alerts** with ANSI escape codes. Red `[ALERT]`, green throughput numbers. Looks great in a terminal demo.
2. **A throughput chart** generated by gnuplot from a CSV your benchmark writes.
3. **Replay-rate knob** — process the pcap at a configurable bytes-per-second to demo CPU dropping behind while GPU keeps up.
4. **A second kernel variant** — implement chunked matching where each thread takes a chunk instead of a packet. Compare. This is a great segue into the class proposal because you've shown you understand the parallelization-axis question.
5. **cpp-httplib web UI** — single page that polls `/stats` and displays alerts streaming. Looks much fancier than CLI but only worth it if everything else is solid.

Don't attempt any of these until the core is locked.

---

## What you walk away with Sunday night

- A working `./build/ids` binary with a clean CLI
- A real malware pcap that fires real alerts during the demo
- Throughput numbers showing GPU > CPU on identical work
- Seven C++/CUDA source files, version-controlled
- A draft of the CS179 proposal that says "we already have a CPU-and-GPU baseline working; the project is to replace the naive kernel with PFAC and add flow reassembly"

That's the entire weekend. Anything beyond it is bonus.
