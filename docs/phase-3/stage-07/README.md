# Stage 7: Speculative Decoding & Advanced Parallelism

[Project Overview](../../README.md) | [Complete Syllabus](../../syllabus.md) | [24-Week Schedule](../../schedule.md)

---

## Overview

Stage 7 bridges single-GPU memory-bandwidth physics with multi-GPU and multi-node distributed execution. Autoregressive LLM decoding is fundamentally bottlenecked by **HBM memory bandwidth** (low arithmetic intensity GEMV operations reading multi-gigabyte model weights for a single token per step), while frontier dense and Mixture-of-Experts (MoE) models exceed the VRAM capacity, compute budget, or context-length limits of a single accelerator.

This stage covers the two primary architectural levers for breaking single-GPU decode latency and capacity walls without sacrificing mathematical exactness:

1. **Speculative Decoding & Drafting Architectures (Topics 1–2):** Trading underutilized Tensor Core FLOPs for lower Inter-Token Latency (ITL) by verifying $\gamma$ candidate tokens in a single batched forward pass while mathematically preserving the target model's exact probability distribution.
2. **Parallelism Topologies & Collective Communication (Topics 3–5):** Partitioning weights, layers, experts, and ultra-long sequences across multiple GPUs and nodes—and quantifying the exact `NCCL` communication cost of every partitioning scheme on PCIe, NVLink/NVSwitch, and InfiniBand fabrics.

---

## Topics in Stage 7

| # | Topic | Core Focus |
| :---: | :--- | :--- |
| **01** | [**Speculative Decoding Foundations**](./01-speculative-decoding-foundations.md) | Memory-bound decode physics, parallel verification, exact rejection sampling proof, speedup & optimal $\gamma^*$ derivation |
| **02** | [**Drafting Architectures**](./02-drafting-architectures.md) | Draft-Target, Medusa (parallel MLP heads + tree attention), EAGLE / EAGLE-2 / EAGLE-3 (feature extrapolation + dynamic trees), N-gram / Prompt Lookup / Lookahead Jacobi decoding |
| **03** | [**Intra-Layer & Inter-Layer Parallelism**](./03-intra-and-inter-layer-parallelism.md) | Megatron-LM Column/Row Tensor Parallelism (2 `AllReduce`s per layer), Sequence Parallelism (`ReduceScatter` + `AllGather`), Pipeline Parallelism (GPipe, 1F1B, bubble formula, throughput vs. latency) |
| **04** | [**Specialized Parallelism (EP & SP)**](./04-specialized-parallelism-ep-and-sp.md) | MoE Expert Parallelism (`AllToAll` dispatch/combine, load imbalance, hybrid TP+EP), DeepSpeed-Ulysses vs. Ring Attention overlapped P2P KV rotation |
| **05** | [**Collective Communication (NCCL)**](./05-collective-communication-nccl.md) | `AllReduce`, `AllGather`, `ReduceScatter`, `AllToAll`, `Broadcast` byte formulas; Ring vs. Tree vs. NVLS (SHARP); PCIe vs. NVLink vs. InfiniBand impact on per-token ITL |

---

## Navigation

* **Up:** [Project Overview](../../README.md) | [Complete Syllabus](../../syllabus.md)
* **Previous Stage:** [← Stage 6: Quantization & Compression at Scale](../../phase-2/stage-06/README.md)
* **Start Reading:** [01 — Speculative Decoding Foundations](./01-speculative-decoding-foundations.md)
* **Next Stage:** [Stage 8: Kernel-Level Optimization →](../stage-08/README.md)
