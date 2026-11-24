# LLM Inference Engineering

This project is a structured 24-week engineering curriculum and portfolio monorepo focused on **LLM Inference Engineering**—covering everything from bare-metal hardware physics and custom GPU kernels to distributed serving, FinOps, and edge fallback architectures.

Its goal is to bridge low-level systems theory with 12 concrete, benchmarked engineering projects, validating every optimization empirically (latency, throughput, memory bandwidth, MFU, and unit economics).

## Documentation Structure

* [**Complete Syllabus (`syllabus.md`)**](./syllabus.md) — Full curriculum covering all 4 Phases, 12 Stages, core technical topics, study deliverables, and the 12 practical portfolio projects.
* [**24-Week Execution Schedule (`schedule.md`)**](./schedule.md) — Week-by-week execution plan (Monday–Saturday cadence), deliverables, validation criteria, and GPU cost-management recommendations.

### Phase & Stage Study Modules

Exhaustive technical study guides (broken down by stage and individual syllabus topic, complete with Mermaid architecture flowcharts and execution sequence diagrams) are organized under each phase directory:

* **`phase-1/` — Hardware Foundations, Silicon & Model Anatomy** *(Weeks 1–4)*
  * [Stage 1: Systems Foundations](./phase-1/stage-01/README.md)
  * [Stage 2: GPU Architecture & Modern Silicon](./phase-1/stage-02/README.md)
  * [Stage 3: Transformer & MoE Inference Anatomy](./phase-1/stage-03/README.md)
* **`phase-2/` — Serving Engines, Memory & Compression** *(Weeks 5–10)*
  * [Stage 4: Serving Engines & Constrained Decoding](./phase-2/stage-04/README.md)
  * [Stage 5: KV Cache & Memory Hierarchies](./phase-2/stage-05/README.md)
  * [Stage 6: Quantization & Compression at Scale](./phase-2/stage-06/README.md)
* **`phase-3/` — Kernels, Advanced Parallelism & Distributed Systems** *(Weeks 11–18)*
  * [Stage 7: Speculative Decoding & Advanced Parallelism](./phase-3/stage-07/README.md)
  * [Stage 8: Kernel-Level Optimization](./phase-3/stage-08/README.md)
  * [Stage 9: Distributed & Disaggregated Inference](./phase-3/stage-09/README.md)
* **`phase-4/` — FinOps, Intelligent Routing, Edge & Portfolio** *(Weeks 19–24)*
  * [Stage 10: Autoscaling, FinOps & Economics](./phase-4/stage-10/README.md)
  * [Stage 11: AI Gateway, Routing & Reliability](./phase-4/stage-11/README.md)
  * [Stage 12: Edge Inference & Portfolio](./phase-4/stage-12/README.md)

## Core Pillars

1. **Silicon & Model Physics:** Memory hierarchy, CUDA/Tensor Cores, Roofline modeling, Prefill vs. Decode bottlenecks, and KV Cache math.
2. **Production Serving & Compression:** Continuous Batching, PagedAttention, Grammar-Constrained Decoding, Hierarchical KV Offloading, and FP4/FP8 Quantization.
3. **Kernels & Distributed Scale:** Speculative Decoding, Triton kernels, FlashAttention, Nsight Compute profiling, Disaggregated Prefill/Decode clusters, and MoE Expert Parallelism.
4. **Resilience & Economics:** Queue-based GPU autoscaling (KEDA), Semantic AI Gateways, Complexity Routing, WebGPU/MLX Edge Fallback, and Pareto-frontier benchmarking.
