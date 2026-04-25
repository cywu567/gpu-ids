# gpu-ids: GPU-Accelerated Intrusion Detection System

A multi-pattern network packet scanner running on a consumer GPU. Scans pcap captures for known-bad byte signatures, in the style of Snort/Suricata but with the matching engine on a GPU instead of a CPU.

Hackathon weekend: naive brute-force kernel, real throughput numbers vs. a CPU baseline.
Class project (CS179): PFAC Aho-Corasick kernel **(implemented)** + TCP flow reassembly + Hyperscan comparison.

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
cd gpu-ids
cmake -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX   # lets CMake find conda's libpcap
cmake --build build -j$(nproc)
```

Binaries land in `build/` either way.

### Generate a synthetic test pcap

```bash
./build/gen_synthetic data/synthetic.pcap 10000 50
# Writes 10 000 benign + 50 malicious packets.
```

Then update `src/gen_synthetic.cpp` malicious payload strings to match your final `patterns/rules.txt`.

### Run the demo

```bash
./build/ids --pcap data/synthetic.pcap --rules patterns/rules.txt --both
```

### Run benchmarks

```bash
# CPU vs GPU naive
./build/benchmark \
    --pcap  data/demo.pcap \
    --rules patterns/rules.txt \
    --iters 5 \
    --csv   results/benchmark.csv

# Add --pfac to also time the Aho-Corasick kernel
./build/benchmark \
    --pcap  data/demo.pcap \
    --rules patterns/rules.txt \
    --iters 5 --pfac \
    --csv   results/benchmark_pfac.csv
```

### Run the web UI

Start the server on the remote machine (default port 8080):

```bash
./build/ids --web 8080 --pcap data/demo.pcap --rules patterns/rules.txt
```

Then, in a **separate terminal on your laptop**, forward the port:

```bash
ssh -L 8080:localhost:8080 user@remote-machine
```

Open [http://localhost:8080](http://localhost:8080) in your browser. The page shows live match results and throughput stats without needing a public IP.

---

## File structure

```
gpu-ids/
├── CMakeLists.txt          build config
├── environment.yml         conda env (libpcap + cmake, no sudo needed)
├── data/
│   ├── demo.pcap           large synthetic pcap for benchmarking
│   └── synthetic.pcap      small synthetic pcap (quick smoke test)
├── patterns/
│   └── rules.txt           one pattern per line; '#' = comment
├── src/
│   ├── load_pcap.cpp/h     libpcap loader → flat buffer + offsets
│   ├── cpu_matcher.cpp/h   reference CPU matcher (memmem baseline)
│   ├── gpu_matcher.cu/cuh  naive kernel + PFAC Aho-Corasick kernel
│   ├── flow_reassembly.cu/cuh  TCP stream reassembly (Week 3 stub)
│   ├── benchmark.cpp       timed CPU vs GPU naive vs GPU PFAC, writes CSV
│   ├── gen_synthetic.cpp   builds synthetic pcap with known hits
│   └── run_demo.cpp        CLI + web server entry point
└── results/
    ├── benchmark.csv       raw throughput numbers
    ├── speedup_chart.png   plotted from CSV
    └── notes.md            hardware specs + run conditions
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

## Demo script (90 seconds)

1. "IDS systems check every packet against thousands of patterns. CPUs struggle at high speed. Enterprises pay $50k for FPGA appliances. We did it on a consumer GPU."
2. Show the pcap: `ls -lh data/demo.pcap`
3. CPU: `./build/ids --pcap data/demo.pcap --rules patterns/rules.txt --cpu`
4. GPU naive: `./build/ids --pcap data/demo.pcap --rules patterns/rules.txt --gpu`
5. "Same alerts, Nx faster. That's the naive kernel -- one thread per (packet, pattern) pair."
6. Benchmark all three: `./build/benchmark --pcap data/demo.pcap --rules patterns/rules.txt --iters 3 --pfac`
7. "PFAC Aho-Corasick compiles all patterns into a single DFA -- one thread per start byte, O(packet_length) regardless of pattern count. The class project adds TCP flow reassembly on top."

---

## References

- Aho & Corasick (1975). *Efficient string matching.* CACM.
- Lin, Liu, Chang (2013). *Accelerating Pattern Matching Using a Novel Parallel Algorithm on GPUs.* IEEE Trans. Computers. **(primary kernel reference)**
- [PFAC reference implementation](https://github.com/pfac-lib/PFAC)
- [Hyperscan](https://www.hyperscan.io/) -- the CPU baseline we benchmark against in Week 4
- [malware-traffic-analysis.net](https://malware-traffic-analysis.net/) -- real malware pcap source
