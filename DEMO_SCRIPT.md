# CudaShield — Demo Video Script
# Target runtime: ~2 min 30 sec
# Format: voiceover over screen recording
#
# [ACTION] = what to show on screen
# Plain text = what to say out loud

---

[ACTION: open cudashield.tech in browser]

Modern networks move fast — at 10 gigabits per second, you have about 120 nanoseconds
to inspect each packet before the next one arrives. Intrusion detection systems like Snort
have to scan every packet against thousands of known attack signatures in that window.
CPUs can't keep up. We built CudaShield — a GPU-accelerated IDS that runs the entire
matching engine on a consumer GPU.

---

[ACTION: show the dashboard with the pcap and rules fields]

This is the live dashboard running on a GPU in the Caltech cluster, accessible through
a Cloudflare tunnel at cudashield.tech. You give it a pcap capture file and a rules file
— the same plain-text pattern format Snort uses — and it runs four engines simultaneously.

---

[ACTION: click Scan, let results animate in]

Three pattern-matching engines run back to back. The yellow bar is our CPU baseline — a
standard sliding-window memmem search — at ___. The purple bar
is Hyperscan — Intel's production SIMD engine used in real enterprise firewalls — at ____.
And the green bar is our GPU PFAC kernel — Parallel Failureless
Aho-Corasick — at ____.

Same alerts, same packet hits across all three. The GPU is nearly ___ times faster than Hyperscan
and ___ times faster than the CPU baseline.

---

[ACTION: scroll down to the alerts table]

Every matched packet shows up in the alert table — packet index, matched pattern string.
You can see it picking up the exact signatures we'd expect.

---

[ACTION: check the Beacon Detection checkbox, click Scan again, scroll to beacon results]

But, pattern matching is blind to encrypted traffic. A C2 beacon inside a TLS session can produce
zero signature hits because the payload is ciphertext. So we added a second GPU kernel for
flow-level beacon analysis.

Right now this beacon detector is validated on synthetic periodic traffic. On some real malware
case-study pcaps, a coefficient-of-variation threshold of 15 percent is too strict and misses
beacon-like behavior. So at this stage we present beacon detection as an early prototype, not a
production claim.

---

[ACTION: show terminal with benchmark CSV or benchmark output, or stay on dashboard]

The key insight behind the PFAC kernel is that instead of one thread per pattern,
you launch one thread per byte position. Every thread walks a precompiled DFA forward
from its start byte until it hits a dead state. No failure links needed — because every
byte gets a fresh start from root. That's O(packet length) per thread, independent of
how many patterns you have.

---

[ACTION: end on dashboard with all results visible]

CudaShield shows that a consumer GPU — hardware you can buy off the shelf — can out-perform
purpose-built enterprise SIMD engines on network intrusion detection, and catch threats that
signature matching alone will never see.

---
