#pragma once
#include <string>

/**
 * Start the web dashboard HTTP server.
 *
 * Serves a live dashboard at http://localhost:PORT that lets users:
 *   - Enter a pcap path and rules path
 *   - Choose CPU / GPU / both
 *   - Trigger a scan and see alerts + throughput comparison
 *
 * default_pcap / default_rules pre-fill the UI form but are not required.
 * Blocks until the process is killed (Ctrl+C).
 */
void run_web_server(int port,
                    const std::string& default_pcap,
                    const std::string& default_rules);
