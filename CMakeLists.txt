cmake_minimum_required(VERSION 3.24)
project(gpu-ids LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect the GPU's compute capability on the build machine (same as CS179 lab3).
set(CMAKE_CUDA_ARCHITECTURES native)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

find_package(Threads REQUIRED)   # httplib needs pthreads on Linux

find_library(PCAP_LIB  pcap  REQUIRED)
find_path   (PCAP_INCLUDE pcap.h REQUIRED)
include_directories(${PCAP_INCLUDE} src)

# Optional: Vectorscan / Hyperscan for CPU baseline comparison
find_path   (HS_INCLUDE hs/hs.h)
find_library(HS_LIB     hs)
if(HS_INCLUDE AND HS_LIB)
    message(STATUS "Vectorscan/Hyperscan found: ${HS_LIB}")
    add_compile_definitions(HAVE_HYPERSCAN)
    include_directories(${HS_INCLUDE})
else()
    message(STATUS "Vectorscan/Hyperscan not found -- --hyperscan benchmark mode disabled")
endif()

# ---------------------------------------------------------------------------
# Common source sets
# ---------------------------------------------------------------------------

set(MATCHING_SOURCES
    src/load_pcap.cpp
    src/cpu_matcher.cpp
    src/gpu_matcher.cu
)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

# Main demo binary: CLI + web dashboard
add_executable(ids
    src/run_demo.cpp
    src/web_server.cpp
    ${MATCHING_SOURCES}
)
target_link_libraries(ids ${PCAP_LIB} Threads::Threads $<$<BOOL:${HS_LIB}>:${HS_LIB}>)

# Benchmarking binary: times CPU vs GPU, writes CSV
add_executable(benchmark
    src/benchmark.cpp
    ${MATCHING_SOURCES}
)
target_link_libraries(benchmark ${PCAP_LIB} $<$<BOOL:${HS_LIB}>:${HS_LIB}>)

# Synthetic pcap generator (no GPU code)
add_executable(gen_synthetic src/gen_synthetic.cpp)
target_link_libraries(gen_synthetic ${PCAP_LIB})

# ---------------------------------------------------------------------------
# Week 3: flow reassembly
# ---------------------------------------------------------------------------
target_sources(ids       PRIVATE src/flow_reassembly.cu)
target_sources(benchmark PRIVATE src/flow_reassembly.cu)
