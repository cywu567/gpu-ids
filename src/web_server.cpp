/**
 * @file web_server.cpp
 * @brief Live web dashboard for GPU-IDS.
 *
 * Serves a single-page dashboard via cpp-httplib (header-only, no extra deps).
 * The page lets a judge enter a pcap+rules path, trigger a scan, and see:
 *   - Packet count, pattern count, alert count
 *   - CPU vs GPU throughput bars with speedup ratio
 *   - Per-alert table (packet #, pattern string)
 *
 * API:
 *   GET  /          — HTML dashboard
 *   GET  /api/status — current scan state as JSON
 *   POST /api/scan   — trigger a scan; blocks until done, returns JSON results
 *
 * The POST blocks for the duration of the scan (typically < 5s for demo
 * pcaps). The front-end shows a spinner while waiting.
 */

#define CPPHTTPLIB_NO_SSL  // no OpenSSL needed
#include "httplib.h"

#include "cpu_matcher.h"
#include "gpu_matcher.cuh"
#include "load_pcap.h"
#include "web_server.h"
#ifdef HAVE_HYPERSCAN
#  include <hs/hs.h>
#endif

#include <chrono>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

namespace {

enum class ScanState { Idle, Scanning, Done, Error };

struct Alert {
    int         packet_idx;
    std::string pattern;
};

struct ScanStatus {
    ScanState   state       = ScanState::Idle;
    std::string error;
    int         num_packets = 0;
    int         num_patterns= 0;
    bool        cpu_done    = false;
    double      cpu_ms      = 0;
    double      cpu_mbps    = 0;
    bool        gpu_done    = false;
    double      gpu_ms      = 0;
    double      gpu_mbps    = 0;
    bool        hs_done     = false;
    double      hs_ms       = 0;
    double      hs_mbps     = 0;
    std::vector<Alert> alerts;
};

static std::mutex  g_mu;
static ScanStatus  g_status;

static double ms_between(
    std::chrono::high_resolution_clock::time_point t0,
    std::chrono::high_resolution_clock::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Tiny JSON helpers (no external library)
// ---------------------------------------------------------------------------

static std::string json_esc(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

static std::string status_to_json() {
    std::lock_guard<std::mutex> lk(g_mu);

    std::ostringstream o;
    const char* state_str = "idle";
    if      (g_status.state == ScanState::Scanning) state_str = "scanning";
    else if (g_status.state == ScanState::Done)     state_str = "done";
    else if (g_status.state == ScanState::Error)    state_str = "error";

    o << "{\n"
      << "  \"state\":" << '"' << state_str << '"' << ",\n"
      << "  \"error\":" << '"' << json_esc(g_status.error) << '"' << ",\n"
      << "  \"num_packets\":" << g_status.num_packets << ",\n"
      << "  \"num_patterns\":" << g_status.num_patterns << ",\n"
      << "  \"cpu\":{"
          << "\"done\":" << (g_status.cpu_done ? "true" : "false") << ","
          << "\"ms\":"   << g_status.cpu_ms   << ","
          << "\"mbps\":" << g_status.cpu_mbps
      << "},\n"
      << "  \"gpu\":{"
          << "\"done\":" << (g_status.gpu_done ? "true" : "false") << ","
          << "\"ms\":"   << g_status.gpu_ms   << ","
          << "\"mbps\":" << g_status.gpu_mbps
      << "},\n"
      << "  \"hs\":{"
          << "\"done\":" << (g_status.hs_done ? "true" : "false") << ","
          << "\"ms\":"   << g_status.hs_ms   << ","
          << "\"mbps\":" << g_status.hs_mbps
      << "},\n"
      << "  \"alerts\":[";

    bool first = true;
    for (const auto& a : g_status.alerts) {
        if (!first) o << ",";
        o << "{\"packet\":" << a.packet_idx
          << ",\"pattern\":\"" << json_esc(a.pattern) << "\"}";
        first = false;
    }
    o << "]\n}";
    return o.str();
}

// Minimal JSON string-field extractor: finds "key":"value" and returns value.
static std::string json_get(const std::string& body, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = body.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    pos = body.find('"', pos);   // skip : whitespace, find opening "
    if (pos == std::string::npos) return "";
    ++pos;
    std::string val;
    while (pos < body.size() && body[pos] != '"') {
        if (body[pos] == '\\' && pos + 1 < body.size()) {
            ++pos;
            if (body[pos] == '"')       val += '"';
            else if (body[pos] == '\\') val += '\\';
            else if (body[pos] == 'n')  val += '\n';
            else                        val += body[pos];
        } else {
            val += body[pos];
        }
        ++pos;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Core scan logic
// ---------------------------------------------------------------------------

static std::string do_scan(const std::string& pcap_path,
                            const std::string& rules_path,
                            bool do_cpu, bool do_gpu, bool do_hs)
{
    // Mark scanning
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_status = ScanStatus{};
        g_status.state = ScanState::Scanning;
    }

    try {
        PcapData   pcap = load_pcap(pcap_path);
        PatternSet ps   = load_patterns(rules_path);
        double     mb   = static_cast<double>(pcap.bytes.size()) / 1e6;

        {
            std::lock_guard<std::mutex> lk(g_mu);
            g_status.num_packets  = pcap.num_packets;
            g_status.num_patterns = ps.num_patterns;
        }

        std::vector<int>     hits(pcap.num_packets * ps.num_patterns, 0);
        std::vector<uint8_t> pfac_hits(pcap.num_packets * ps.num_patterns, 0);

        // Touch all packet bytes to ensure they're in the OS page cache before
        // any timed scan runs. Without this, the first engine (CPU) pays the
        // page-fault cost and the later engines (GPU, Hyperscan) get an unfair
        // warm-cache advantage, making results non-deterministic on small pcaps.
        volatile uint8_t sink = 0;
        for (const auto& b : pcap.bytes) sink ^= b;
        (void)sink;

        auto collect_alerts = [&](const std::string& mode) {
            // For GPU PFAC path, read from pfac_hits (uint8_t); else from hits (int).
            bool use_pfac = (mode == "gpu");
            for (int p = 0; p < pcap.num_packets; p++) {
                for (int r = 0; r < ps.num_patterns; r++) {
                    int idx = p * ps.num_patterns + r;
                    int hit = use_pfac ? (int)pfac_hits[idx] : hits[idx];
                    if (hit) {
                        std::lock_guard<std::mutex> lk(g_mu);
                        bool dup = false;
                        for (const auto& a : g_status.alerts)
                            if (a.packet_idx == p && a.pattern == ps.labels[r])
                                { dup = true; break; }
                        if (!dup)
                            g_status.alerts.push_back({p, ps.labels[r]});
                    }
                }
            }
        };

        if (do_cpu) {
            auto t0 = std::chrono::high_resolution_clock::now();
            run_cpu_matcher(pcap.bytes.data(), pcap.offsets.data(),
                            pcap.num_packets, ps, hits.data());
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = ms_between(t0, t1);
            {
                std::lock_guard<std::mutex> lk(g_mu);
                g_status.cpu_done = true;
                g_status.cpu_ms   = ms;
                g_status.cpu_mbps = mb / (ms / 1e3);
            }
            collect_alerts("cpu");
        }

        if (do_gpu) {
            std::fill(pfac_hits.begin(), pfac_hits.end(), 0);
            PfacDfa dfa = build_pfac_dfa(ps);
            auto t0 = std::chrono::high_resolution_clock::now();
            run_pfac_match_gpu(
                pcap.bytes.data(), pcap.offsets.data(), pcap.num_packets,
                dfa, pfac_hits.data(), ps.num_patterns, pcap.bytes.size());
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = ms_between(t0, t1);
            {
                std::lock_guard<std::mutex> lk(g_mu);
                g_status.gpu_done = true;
                g_status.gpu_ms   = ms;
                g_status.gpu_mbps = mb / (ms / 1e3);
            }
            collect_alerts("gpu");
        }

#ifdef HAVE_HYPERSCAN
        if (do_hs) {
            std::vector<const char*> pats;
            std::vector<size_t>      pat_lens;
            std::vector<unsigned>    ids, flags;
            for (int r = 0; r < ps.num_patterns; r++) {
                pats.push_back(ps.labels[r].c_str());
                pat_lens.push_back(ps.labels[r].size());
                ids.push_back(static_cast<unsigned>(r));
                flags.push_back(HS_FLAG_CASELESS | HS_FLAG_SINGLEMATCH);
            }
            hs_database_t*      db  = nullptr;
            hs_compile_error_t* err = nullptr;
            if (hs_compile_lit_multi(pats.data(), flags.data(), ids.data(),
                                     pat_lens.data(),
                                     static_cast<unsigned>(pats.size()),
                                     HS_MODE_BLOCK, nullptr, &db, &err) == HS_SUCCESS) {
                hs_scratch_t* scratch = nullptr;
                hs_alloc_scratch(db, &scratch);
                std::fill(hits.begin(), hits.end(), 0);
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int p = 0; p < pcap.num_packets; p++) {
                    const char* pkt = reinterpret_cast<const char*>(
                        pcap.bytes.data() + pcap.offsets[p]);
                    unsigned pkt_len = static_cast<unsigned>(
                        pcap.offsets[p + 1] - pcap.offsets[p]);
                    struct PktCtx { int p; int np; std::vector<int>* hits; };
                    PktCtx pctx{p, ps.num_patterns, &hits};
                    hs_scan(db, pkt, pkt_len, 0, scratch,
                        [](unsigned id, unsigned long long, unsigned long long,
                           unsigned, void* vc) -> int {
                            auto* c = static_cast<PktCtx*>(vc);
                            c->hits->at(c->p * c->np + static_cast<int>(id)) = 1;
                            return 0;
                        }, &pctx);
                }
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = ms_between(t0, t1);
                {
                    std::lock_guard<std::mutex> lk(g_mu);
                    g_status.hs_done = true;
                    g_status.hs_ms   = ms;
                    g_status.hs_mbps = mb / (ms / 1e3);
                }
                collect_alerts("hs");
                hs_free_scratch(scratch);
                hs_free_database(db);
            } else {
                hs_free_compile_error(err);
            }
        }
#else
        (void)do_hs;
#endif

        {
            std::lock_guard<std::mutex> lk(g_mu);
            g_status.state = ScanState::Done;
        }

    } catch (const std::exception& ex) {
        std::lock_guard<std::mutex> lk(g_mu);
        g_status.state = ScanState::Error;
        g_status.error = ex.what();
    }

    return status_to_json();
}

// ---------------------------------------------------------------------------
// Dashboard HTML (embedded as a raw string)
// ---------------------------------------------------------------------------

static const char* DASHBOARD_HTML = R"html(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CudaShield</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Consolas', 'Menlo', monospace;
    background: #0d1117;
    color: #c9d1d9;
    padding: 24px;
    min-height: 100vh;
  }
  h1 { color: #58a6ff; font-size: 1.6rem; margin-bottom: 4px; }
  .subtitle { color: #8b949e; font-size: 0.85rem; margin-bottom: 24px; }
  .card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 20px;
    margin-bottom: 16px;
  }
  .card h2 { font-size: 1rem; color: #8b949e; margin-bottom: 14px; text-transform: uppercase; letter-spacing: 0.05em; }
  .form-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
  .form-row label { color: #8b949e; font-size: 0.85rem; width: 80px; flex-shrink: 0; }
  input[type=text] {
    background: #0d1117;
    border: 1px solid #30363d;
    color: #c9d1d9;
    padding: 6px 10px;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.9rem;
    width: 360px;
    outline: none;
  }
  input[type=text]:focus { border-color: #58a6ff; }
  .radio-group { display: flex; gap: 16px; }
  .radio-group label { color: #c9d1d9; cursor: pointer; display: flex; align-items: center; gap: 6px; width: auto; }
  input[type=radio] { accent-color: #58a6ff; cursor: pointer; }
  .btn {
    background: #238636;
    color: #fff;
    border: none;
    padding: 8px 20px;
    border-radius: 4px;
    font-family: inherit;
    font-size: 0.9rem;
    cursor: pointer;
    transition: background 0.15s;
  }
  .btn:hover { background: #2ea043; }
  .btn:disabled { background: #21262d; color: #8b949e; cursor: not-allowed; }
  .status-msg { color: #8b949e; font-size: 0.85rem; margin-left: 12px; font-style: italic; }
  .status-msg.scanning { color: #e3b341; }
  .status-msg.done     { color: #3fb950; }
  .status-msg.error    { color: #f85149; }

  /* Stats row */
  .stats-grid { display: flex; gap: 32px; flex-wrap: wrap; margin-bottom: 20px; }
  .stat-box { display: flex; flex-direction: column; gap: 4px; }
  .stat-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.06em; }
  .stat-value { font-size: 2rem; font-weight: bold; line-height: 1; }
  .stat-packets  { color: #58a6ff; }
  .stat-patterns { color: #a5d6ff; }
  .stat-alerts   { color: #f85149; }
  .stat-speedup  { color: #e3b341; }

  /* Throughput bars */
  .throughput-section { margin-top: 8px; }
  .bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
  .bar-label { width: 120px; font-size: 0.85rem; color: #8b949e; flex-shrink: 0; }
  .bar-label span { font-size: 1rem; font-weight: bold; }
  .bar-label.cpu span { color: #e3b341; }
  .bar-label.gpu span { color: #3fb950; }
  .bar-track { flex: 1; background: #21262d; border-radius: 3px; height: 22px; overflow: hidden; }
  .bar-fill { height: 22px; border-radius: 3px; width: 0%; transition: width 0.6s ease; }
  .bar-fill.cpu { background: #e3b341; }
  .bar-fill.gpu { background: #3fb950; }
  .bar-val { width: 110px; font-size: 0.85rem; color: #c9d1d9; text-align: right; flex-shrink: 0; }

  /* Alerts table */
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th { color: #8b949e; font-weight: normal; text-align: left; padding: 6px 10px; border-bottom: 1px solid #21262d; }
  td { padding: 7px 10px; border-bottom: 1px solid #21262d; }
  tr.alert-row td { color: #f85149; }
  tr.alert-row:hover td { background: #1f2937; }

  #results-section { display: none; }
  #alerts-section  { display: none; }
</style>
</head>
<body>

<h1>&#x26a1; CudaShield</h1>
<p class="subtitle">GPU PFAC (Aho-Corasick) vs Hyperscan (SIMD CPU) vs naive CPU</p>

<div class="card">
  <h2>Configure Scan</h2>
  <div class="form-row">
    <label>PCAP</label>
    <input type="text" id="pcap-input" placeholder="data/test.pcap" />
  </div>
  <div class="form-row">
    <label>Rules</label>
    <input type="text" id="rules-input" placeholder="patterns/rules.txt" />
  </div>
  <div class="form-row">
    <label>Mode</label>
    <div class="radio-group">
      <label><input type="radio" name="mode" value="both" checked> CPU + GPU</label>
      <label><input type="radio" name="mode" value="cpu">  CPU only</label>
      <label><input type="radio" name="mode" value="gpu">  GPU only</label>
      <label><input type="radio" name="mode" value="all">  All (+ Hyperscan)</label>
    </div>
  </div>
  <div class="form-row" style="margin-top:6px">
    <button class="btn" id="scan-btn" onclick="startScan()">&#x25b6; Run Scan</button>
    <span class="status-msg" id="status-msg"></span>
  </div>
</div>

<div class="card" id="results-section">
  <h2>Results</h2>
  <div class="stats-grid">
    <div class="stat-box">
      <span class="stat-label">Packets</span>
      <span class="stat-value stat-packets" id="s-packets">—</span>
    </div>
    <div class="stat-box">
      <span class="stat-label">Patterns</span>
      <span class="stat-value stat-patterns" id="s-patterns">—</span>
    </div>
    <div class="stat-box">
      <span class="stat-label">Alerts</span>
      <span class="stat-value stat-alerts" id="s-alerts">—</span>
    </div>
    <div class="stat-box" id="speedup-box" style="display:none">
      <span class="stat-label">GPU vs CPU</span>
      <span class="stat-value stat-speedup" id="s-speedup">—</span>
    </div>
  </div>

  <div class="throughput-section">
    <div class="bar-row" id="cpu-row" style="display:none">
      <div class="bar-label cpu">CPU <span id="cpu-mbps">—</span> MB/s</div>
      <div class="bar-track"><div class="bar-fill cpu" id="cpu-bar"></div></div>
      <div class="bar-val" id="cpu-ms">—</div>
    </div>
    <div class="bar-row" id="gpu-row" style="display:none">
      <div class="bar-label gpu">GPU PFAC <span id="gpu-mbps">—</span> MB/s</div>
      <div class="bar-track"><div class="bar-fill gpu" id="gpu-bar"></div></div>
      <div class="bar-val" id="gpu-ms">—</div>
    </div>
    <div class="bar-row" id="hs-row" style="display:none">
      <div class="bar-label" style="color:#a371f7">Hyperscan <span id="hs-mbps">—</span> MB/s</div>
      <div class="bar-track"><div class="bar-fill" id="hs-bar" style="background:#a371f7"></div></div>
      <div class="bar-val" id="hs-ms">—</div>
    </div>
  </div>
</div>

<div class="card" id="alerts-section">
  <h2>Alerts &mdash; <span id="alert-count" style="color:#f85149">0</span> match(es)</h2>
  <table>
    <thead><tr><th>#</th><th>Packet</th><th>Pattern</th></tr></thead>
    <tbody id="alerts-tbody"></tbody>
  </table>
</div>

<script>
const defaults = {pcap: '', rules: ''};

window.addEventListener('DOMContentLoaded', () => {
  if (defaults.pcap)  document.getElementById('pcap-input').value  = defaults.pcap;
  if (defaults.rules) document.getElementById('rules-input').value = defaults.rules;
});

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function setStatus(msg, cls) {
  const el = document.getElementById('status-msg');
  el.textContent = msg;
  el.className = 'status-msg ' + (cls || '');
}

async function startScan() {
  const pcap  = document.getElementById('pcap-input').value.trim()  || 'data/test.pcap';
  const rules = document.getElementById('rules-input').value.trim() || 'patterns/rules.txt';
  const mode  = document.querySelector('input[name=mode]:checked').value;

  document.getElementById('scan-btn').disabled = true;
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('alerts-section').style.display  = 'none';
  setStatus('Scanning…', 'scanning');

  try {
    const resp = await fetch('/api/scan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({pcap, rules, mode})
    });
    const d = await resp.json();
    renderResults(d);
  } catch(e) {
    setStatus('Request failed: ' + e.message, 'error');
  } finally {
    document.getElementById('scan-btn').disabled = false;
  }
}

function renderResults(d) {
  if (d.state === 'error') {
    setStatus('Error: ' + d.error, 'error');
    return;
  }

  setStatus('Done.', 'done');
  document.getElementById('results-section').style.display = 'block';

  document.getElementById('s-packets').textContent  = (d.num_packets  || 0).toLocaleString();
  document.getElementById('s-patterns').textContent = (d.num_patterns || 0).toLocaleString();

  const alerts = d.alerts || [];
  document.getElementById('s-alerts').textContent = alerts.length;

  const cpu = d.cpu || {};
  const gpu = d.gpu || {};
  const hs  = d.hs  || {};

  if (cpu.done) {
    document.getElementById('cpu-row').style.display = 'flex';
    document.getElementById('cpu-mbps').textContent  = cpu.mbps.toFixed(1);
    document.getElementById('cpu-ms').textContent    = cpu.ms.toFixed(1) + ' ms';
  }
  if (gpu.done) {
    document.getElementById('gpu-row').style.display = 'flex';
    document.getElementById('gpu-mbps').textContent  = gpu.mbps.toFixed(1);
    document.getElementById('gpu-ms').textContent    = gpu.ms.toFixed(1) + ' ms';
  }
  if (hs.done) {
    document.getElementById('hs-row').style.display = 'flex';
    document.getElementById('hs-mbps').textContent  = hs.mbps.toFixed(1);
    document.getElementById('hs-ms').textContent    = hs.ms.toFixed(1) + ' ms';
  }

  if (cpu.done || gpu.done || hs.done) {
    const maxMbps = Math.max(cpu.mbps || 0, gpu.mbps || 0, hs.mbps || 0, 1);
    if (cpu.done) document.getElementById('cpu-bar').style.width = (cpu.mbps / maxMbps * 100).toFixed(1) + '%';
    if (gpu.done) document.getElementById('gpu-bar').style.width = (gpu.mbps / maxMbps * 100).toFixed(1) + '%';
    if (hs.done)  document.getElementById('hs-bar').style.width  = (hs.mbps  / maxMbps * 100).toFixed(1) + '%';
  }

  if (cpu.done && gpu.done) {
    const speedup = gpu.mbps / Math.max(cpu.mbps, 0.001);
    let speedupText;
    if (speedup >= 1) {
      speedupText = speedup.toFixed(1) + 'x faster';
    } else {
      speedupText = (1 / speedup).toFixed(1) + 'x slower';
    }
    document.getElementById('s-speedup').textContent = speedupText;
    document.getElementById('speedup-box').style.display = 'flex';
  }

  if (alerts.length > 0) {
    document.getElementById('alerts-section').style.display = 'block';
    document.getElementById('alert-count').textContent = alerts.length;
    const tbody = document.getElementById('alerts-tbody');
    tbody.innerHTML = '';
    alerts.forEach((a, i) => {
      const tr = document.createElement('tr');
      tr.className = 'alert-row';
      tr.innerHTML = `<td>${i+1}</td><td>#${a.packet}</td><td>${esc(a.pattern)}</td>`;
      tbody.appendChild(tr);
    });
  }
}
</script>
</body>
</html>
)html";

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void run_web_server(int port,
                    const std::string& default_pcap,
                    const std::string& default_rules)
{
    // Warm up the CUDA context once at startup so the first timed scan
    // doesn't include the ~200ms CUDA runtime initialization cost.
    {
        const uint8_t dummy_input[1] = {0};
        const int     dummy_offsets[2] = {0, 1};
        int           dummy_hit[1] = {0};
        run_naive_match_gpu(dummy_input, dummy_offsets, 1,
                            dummy_input, dummy_offsets, 1,
                            dummy_hit, 1, 1);
    }

    // Patch the default values into the HTML at startup.
    // Replace the JS object literal placeholders with actual values.
    std::string html = DASHBOARD_HTML;
    auto replace_one = [&](const std::string& key, const std::string& val) {
        std::string needle = "'" + key + "': ''";
        std::string replacement = "'" + key + "': '" + json_esc(val) + "'";
        auto pos = html.find(needle);
        if (pos != std::string::npos)
            html.replace(pos, needle.size(), replacement);
    };
    replace_one("pcap",  default_pcap);
    replace_one("rules", default_rules);

    httplib::Server svr;

    svr.Get("/", [&html](const httplib::Request&, httplib::Response& res) {
        res.set_content(html, "text/html; charset=utf-8");
    });

    svr.Get("/api/status", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(status_to_json(), "application/json");
    });

    // POST /api/scan — runs synchronously, returns full results when done.
    svr.Post("/api/scan", [](const httplib::Request& req, httplib::Response& res) {
        std::string pcap  = json_get(req.body, "pcap");
        std::string rules = json_get(req.body, "rules");
        std::string mode  = json_get(req.body, "mode");
        if (pcap.empty())  pcap  = "data/test.pcap";
        if (rules.empty()) rules = "patterns/rules.txt";
        bool do_cpu = (mode == "cpu"  || mode == "both" || mode == "all" || mode.empty());
        bool do_gpu = (mode == "gpu"  || mode == "both" || mode == "all" || mode.empty());
        bool do_hs  = (mode == "hyperscan" || mode == "all");

        std::string result = do_scan(pcap, rules, do_cpu, do_gpu, do_hs);
        res.set_content(result, "application/json");
    });

    svr.listen("0.0.0.0", port);
}
