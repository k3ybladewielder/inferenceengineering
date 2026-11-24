# Stage 2: GPU Architecture & Modern Silicon

[Back to Curriculum Overview](../../README.md) | [Complete Syllabus](../../syllabus.md)

## Overview

Modern Large Language Model (LLM) inference performance is governed entirely by the physical constraints of GPU silicon: how thousands of concurrent threads are scheduled across Streaming Multiprocessors (SMs), how bytes travel from HBM3/HBM3e through caches and SRAM to register files, how Tensor Cores amortize memory bandwidth via systolic matrix arithmetic, and how architectural innovations in NVIDIA Hopper (H100/H200) and Blackwell (B200/GB200) eliminate data-movement and quantization bottlenecks.

This stage builds a first-principles mental model of NVIDIA GPU hardware architecture, execution semantics, memory physics, and production hardware telemetry—establishing the foundation required for Roofline modeling (Stage 3), serving engine optimization (Phase 2), and custom Triton/CUDA kernel engineering (Phase 3).

---

## Topics in This Stage

1. **[01 — CUDA Execution Hierarchy](./01-cuda-execution-hierarchy.md)**
   * Grids, Thread Blocks (CTAs), Warps (32 threads), and hardware mapping onto Streaming Multiprocessors (SMs).
   * SIMT execution, Independent Thread Scheduling (Volta+), active masks, predication vs. divergence serialization penalties.
   * Zero-overhead warp context switching, warp schedulers, and hardware stall reasons (memory dependency, execution dependency, barriers).

2. **[02 — GPU Memory Hierarchy](./02-gpu-memory-hierarchy.md)**
   * Per-thread register files (256 KB/SM), register pressure, and local memory spilling.
   * Unified L1/Shared Memory (SRAM), 32-bank architecture, bank conflicts, broadcasts, and swizzling.
   * Shared L2 cache capacity, persistence policies, and HBM3/HBM3e stack/channel physics.
   * 32-byte sector and 128-byte cache-line transactions, coalesced vs. strided access math, and vectorized loads.

3. **[03 — Compute Units: CUDA Cores vs. Tensor Cores](./03-compute-units-cuda-vs-tensor-cores.md)**
   * Scalar FMA (`FFMA`) pipelines on CUDA Cores vs. matrix-multiply-accumulate (`mma.sync`, `wgmma`, `tcgen05.mma`) on Tensor Cores.
   * Operand layout, systolic reuse, and why Tensor Cores achieve $15\times$–$30\times+$ higher FLOP/s than FP32 CUDA Cores.
   * Accumulation precision (FP32 accumulators vs. FP16/BF16/FP8 quirks) and exact bit-level formats: FP16, BF16, FP8 (E4M3 vs. E5M2), and FP4 / MXFP4 / NVFP4.

4. **[04 — Occupancy & Modern Architectures (Hopper & Blackwell)](./04-occupancy-and-modern-architectures.md)**
   * Theoretical vs. achieved SM occupancy, hardware resource limiters (registers, shared memory, warp/block slots), and Little's Law for latency hiding.
   * NVIDIA Hopper: Tensor Memory Accelerator (TMA), asynchronous barriers (`mbarrier`), Thread Block Clusters, Distributed Shared Memory (DSMEM), Transformer Engine, and NVLink 4.0.
   * NVIDIA Blackwell: dual-die chiplet architecture over 10 TB/s NV-HBI, NVLink 5.0 / NVLink-C2C, 5th-gen Tensor Cores with TMEM and native MXFP4/NVFP4 microscaling, and the hardware Decompression Engine.

5. **[05 — Hardware Telemetry (`nvidia-smi` & DCGM)](./05-hardware-telemetry.md)**
   * `nvidia-smi dmon` vs. Data Center GPU Manager (`dcgmi dmon`) profiling counters.
   * Exact hardware definitions of `sm_active`, `sm_occupancy`, `tensor_active`, and `dram_active`, plus NVLink/PCIe throughput, clock throttle bitmasks, and ECC counters.
   * Telemetry signatures for diagnosing compute-bound, memory-bandwidth-bound, launch/latency-bound, interconnect-bound, and throttled inference workloads—and why `GPU-Util` misleads LLM engineers.

---

## Navigation

* **Up:** [Phase 1 / Docs Overview](../../README.md) | [Complete Syllabus](../../syllabus.md)
* **Previous Stage:** [← Stage 1: Systems Foundations](../stage-01/README.md)
* **Start Reading:** [01 — CUDA Execution Hierarchy](./01-cuda-execution-hierarchy.md)
* **Next Stage:** [Stage 3: Transformer & MoE Inference Anatomy →](../stage-03/README.md)
