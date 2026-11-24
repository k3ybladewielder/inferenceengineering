# 24-Week Execution Schedule

[Back to Project Overview](./README.md) | [Complete Syllabus](./syllabus.md)

This schedule is structured for **24 weeks (6 months)**, maintaining a sustainable and rigorous pace. It balances theoretical study of low-level architecture with hands-on implementation and empirical validation of all 12 projects.

Each week follows a **Monday-to-Saturday** cadence (assuming focused blocks of ~1.5h to 2h per day):

* **Days 1–2:** Theoretical foundations, paper reading, and architecture documentation.
* **Days 3–5:** Hands-on implementation of code, kernels, or infrastructure and debugging.
* **Day 6:** Profiling, rigorous benchmarking (metric collection, stress tests), and technical documentation in the repository.

---

## Phase 1: Hardware Foundations, Silicon & Model Anatomy (Weeks 1–4)

*Phase Goal: Master hardware physics and build analytical memory and FLOPs prediction tools before touching serving frameworks.*

| Week | Stage / Focus | Execution Plan | Deliverable & Validation |
| --- | --- | --- | --- |
| **Week 1** | [**Stage 1:** Systems Foundations (Part 1)](./syllabus.md#stage-1-systems-foundations) | • **Theory:** Memory hierarchy (L1/L2/L3, RAM, NVMe), CPU vs. GPU vs. NPU execution models, PCIe bus bottlenecks.<br>• **Practice:** Profile Python scripts with `py-spy` and `cProfile`; identify GIL overhead and host-to-device memory copies. | Profiling report demonstrating elimination of CPU/memory bottlenecks in a simple inference pipeline. |
| **Week 2** | [**Stage 1:** Systems Foundations (Part 2)](./syllabus.md#stage-1-systems-foundations) | • **Theory:** Linux internals (page cache, `mmap`, zero-copy I/O, NUMA nodes) and C++/Rust fundamentals for high-performance bindings.<br>• **Practice:** Implement a fast tensor loader (`safetensors`) using `mmap` and measure disk-to-RAM read time. | Benchmark comparing standard loading vs. `mmap` on multi-gigabyte weight files. |
| **Week 3** | [**Stage 2:** GPU Architecture & Modern Silicon](./syllabus.md#stage-2-gpu-architecture--modern-silicon) | • **Theory:** CUDA architecture (Warps, Threads, Blocks), Tensor Cores, HBM3e bandwidth, SM occupancy, and differences between Hopper (H100) and Blackwell (B200).<br>• **Practice:** Real-time telemetry monitoring using `nvidia-smi` and DCGM (SM clock, memory bandwidth utilization, PCIe throughput). | Technical spec map (FP16/FP8 TFLOPs and GB/s bandwidth) of the development GPU for *Roofline* analysis. |
| **Week 4** | [**Stage 3:** Transformer & MoE Anatomy](./syllabus.md#stage-3-transformer--moe-inference-anatomy) | • **Theory:** Physics of **Prefill** (*compute-bound*) vs. **Decode** (*memory-bandwidth-bound*), exact KV Cache math, arithmetic intensity, and MoE routing.<br>• **Practice:** Build a Python library/CLI that computes KV Cache VRAM usage and theoretical MFU for dense and MoE architectures. | **Analytical Inference Calculator** validated against actual allocated VRAM measurements in PyTorch. |

---

## Phase 2: Serving Engines, Memory & Compression (Weeks 5–10)

*Phase Goal: Understand the state of the art in serving engines and build the first three major engineering components.*

| Week | Stage / Project | Execution Plan | Deliverable & Validation |
| --- | --- | --- | --- |
| **Week 5** | [**Stage 4:** Serving Engines Internals](./syllabus.md#stage-4-serving-engines--constrained-decoding) | • **Theory:** Architecture of vLLM, SGLang, TGI, and TensorRT-LLM; *Continuous Batching* mechanics and block management in *PagedAttention*.<br>• **Practice:** Spin up vLLM and SGLang locally, inspect scheduler logs under concurrent load, and analyze memory fragmentation. | Technical comparison documenting how each engine manages request queues and KV block allocation. |
| **Week 6** | [**Project 4:** Grammar-Constrained Decoding](./syllabus.md#stage-4-serving-engines--constrained-decoding) | • **Theory:** Structured decoding, finite state machines (FSM), and sampling-time *logit masking*.<br>• **Practice:** Build a custom sampling loop using *Context-Free Grammars* (CFG) to guarantee 100% adherence to JSON/Pydantic schemas. | **Project 4 Completed:** Test suite with 1,000 complex nested generations with 0% syntax errors and per-token overhead measurement. |
| **Week 7** | [**Stage 5:** KV Cache & Memory Hierarchies](./syllabus.md#stage-5-kv-cache--memory-hierarchies) | • **Theory:** *Prefix Caching* (RadixAttention), *Chunked Prefill*, KV Cache quantization (FP8), and eviction policies (LRU, LFU).<br>• **Practice:** Simulate multi-turn workloads and measure TTFT (*Time to First Token*) gains when enabling prefix caching on long system prompts. | Empirical analysis of *cache hit rate* and latency reduction in long multi-turn conversations. |
| **Week 8** | [**Project 3:** Hierarchical KV-Cache Offloader](./syllabus.md#stage-5-kv-cache--memory-hierarchies) | • **Theory:** Asynchronous transfers via CUDA Streams (`cudaMemcpyAsync`), *pinned* (page-locked) memory, and direct NVMe I/O.<br>• **Practice:** Implement the hierarchical manager (`VRAM → CPU RAM → NVMe SSD`) for inactive KV blocks without blocking the *decode* thread. | **Project 3 Completed:** Long-context session benchmark (e.g., 64k+ tokens) proving VRAM stability without *Out-Of-Memory* errors. |
| **Week 9** | [**Stage 6:** Quantization & Compression](./syllabus.md#stage-6-quantization--compression-at-scale) | • **Theory:** FP4, FP8, INT8 formats; post-training calibration algorithms (AWQ, GPTQ), sparsity, and *mixed-precision serving*.<br>• **Practice:** Run existing quantization pipelines and analyze the distribution of activation *outliers* in linear layers. | Per-layer sensitivity study showing where aggressive quantization degrades representation the most. |
| **Week 10** | [**Project 6:** FP4 Quantization Pipeline](./syllabus.md#stage-6-quantization--compression-at-scale) | • **Theory:** Degradation evaluation metrics: perplexity (WikiText-2), KL divergence on output distributions, and reasoning benchmarks.<br>• **Practice:** Calibrate an open model to FP4/INT4, run the quality evaluation suite, and measure real *tokens/s* gains and VRAM reduction. | **Project 6 Completed:** Mathematical *trade-off* table (Perplexity vs. VRAM vs. Throughput) published in the repository. |

---

## Phase 3: Kernels, Advanced Parallelism & Distributed Systems (Weeks 11–18)

*Phase Goal: Drop down to the silicon level by writing custom kernels and scaling inference across multiple GPUs and nodes.*

| Week | Stage / Project | Execution Plan | Deliverable & Validation |
| --- | --- | --- | --- |
| **Week 11** | [**Stage 7 & Project 7:** Speculative Decoding](./syllabus.md#stage-7-speculative-decoding--advanced-parallelism) | • **Theory:** *Draft-Target* pipelines, Medusa/EAGLE heads, and *rejection sampling* math guaranteeing identical output distribution.<br>• **Practice:** Implement the *harness* connecting a *draft* model (e.g., 1B–3B) to a *target* model (e.g., 32B–70B) in a single verification step. | **Project 7 Completed:** Plot of *acceptance rate* vs. net *decode* speedup across different domains (code vs. text). |
| **Week 12** | [**Stage 7:** Advanced Parallelism](./syllabus.md#stage-7-speculative-decoding--advanced-parallelism) | • **Theory:** Tensor (TP) vs. Pipeline (PP) vs. Expert (EP) vs. Sequence Parallelism (Ring Attention); NCCL primitives (`AllReduce`, `AllGather`, `AllToAll`).<br>• **Practice:** Profile NCCL communication cost across GPUs and calculate the breakeven point where network overhead outweighs compute gains. | Architectural decision diagram for choosing parallelism strategies based on model size and network topology. |
| **Week 13** | [**Stage 8:** Kernel-Level Optimization](./syllabus.md#stage-8-kernel-level-optimization) | • **Theory:** **Triton** block-level programming model, shared memory (SRAM), global memory access coalescing, CUDA Graphs, and *operator fusion*.<br>• **Practice:** Write basic Triton kernels (vector addition, fused softmax) and inspect generated PTX. | Functional introductory kernels numerically validated against PyTorch with `atol=1e-5` tolerance. |
| **Week 14** | [**Project 1:** Custom Triton Kernel Optimizer](./syllabus.md#stage-8-kernel-level-optimization) | • **Theory:** Online **FlashAttention-3** algorithm, *Roofline* model analysis (arithmetic vs. memory bandwidth), and NVIDIA Nsight Compute metrics.<br>• **Practice:** Implement a fused *RMSNorm* or *Attention* kernel in Triton, profile with Nsight Compute, and optimize occupancy and memory access. | **Project 1 Completed:** *Roofline* chart and benchmark showing proximity to theoretical hardware limits vs. native PyTorch. |
| **Week 15** | [**Stage 9:** Distributed & Disaggregated Inference](./syllabus.md#stage-9-distributed--disaggregated-inference) | • **Theory:** Disaggregated *Prefill* and *Decode* pool architecture, NVLink/InfiniBand interconnects, and direct KV transfer via RDMA.<br>• **Practice:** Configure communication between two isolated processes/nodes simulating role separation (one node generating KV, another consuming it). | Prototype of KV state transfer between separate instances with network latency measurements. |
| **Week 16** | [**Project 2:** Disaggregated Prefill/Decode Cluster](./syllabus.md#stage-9-distributed--disaggregated-inference) | • **Theory:** Dynamic balancing of Prefill/Decode node ratio (*P/D ratio tuning*) based on average input vs. output length.<br>• **Practice:** Build the orchestrator routing compute-heavy requests to FLOP-optimized instances and *decode* to bandwidth/cost-optimized instances. | **Project 2 Completed:** Simulation/deployment proving increased global MFU and stabilized ITL (*Inter-Token Latency*) under long-prompt bursts. |
| **Week 17** | [**Stage 9 (MoE):** Sparse Routing Internals](./syllabus.md#stage-9-distributed--disaggregated-inference) | • **Theory:** Physics of Mixture-of-Experts models (DeepSeek/Mixtral), *expert load imbalance*, and *capacity factor*.<br>• **Practice:** Instrument an MoE model to log activation frequency of each expert across different datasets. | Statistical analysis of routing distribution and identification of *straggler* bottlenecks (overloaded experts). |
| **Week 18** | [**Project 5:** MoE Expert-Parallel Router](./syllabus.md#stage-9-distributed--disaggregated-inference) | • **Theory:** Optimizing *All-to-All* communication on multi-GPU networks for routing tokens across distributed experts.<br>• **Practice:** Implement expert weight *sharding* across GPUs with dynamic load balancing. | **Project 5 Completed:** Distributed MoE layer latency benchmark showing reduced *All-to-All* synchronization time. |

---

## Phase 4: FinOps, Intelligent Routing, Edge & Portfolio (Weeks 19–24)

*Phase Goal: Consolidate engines into a resilient platform optimized for unit economics and publish results with scientific rigor.*

| Week | Stage / Project | Execution Plan | Deliverable & Validation |
| --- | --- | --- | --- |
| **Week 19** | [**Stage 10 & Project 8:** Autoscaling & FinOps](./syllabus.md#stage-10-autoscaling-finops--economics) | • **Theory:** Event-driven autoscaling (KEDA), *cold start* mitigation (weight streaming), Spot GPUs, and cost-per-million-tokens modeling.<br>• **Practice:** Build an autoscaler based on pending token queue depth with graceful *drain* so in-flight generations are not dropped. | **Project 8 Completed:** Synthetic load test (sine-wave traffic) demonstrating automatic scaling and zero dropped requests during *scale-down*. |
| **Week 20** | [**Stage 11 & Project 9:** Semantic AI Gateway](./syllabus.md#stage-11-ai-gateway-routing--reliability) | • **Theory:** Production SLOs (TTFT, ITL, TPOT), token-based *rate limiting*, observability (OpenTelemetry/Prometheus), and vector semantic caching.<br>• **Practice:** Implement the gateway in Python/Rust generating fast prompt embeddings and returning cached responses when similarity exceeds 98%. | **Project 9 Completed:** Working gateway with telemetry dashboard showing *cache hit* rate, estimated dollar savings, and sub-10ms hit latency. |
| **Week 21** | [**Project 10:** Multi-Model Complexity Router](./syllabus.md#stage-11-ai-gateway-routing--reliability) | • **Theory:** Prompt complexity estimation (entropy, lightweight classifiers, structural heuristics) and resilient *fallback* chains.<br>• **Practice:** Integrate a decision router into the gateway sending simple queries to a quantized 8B model and complex reasoning tasks to a frontier model. | **Project 10 Completed:** Cross-evaluation proving ~60% average cost reduction while maintaining >95% quality on mixed benchmarks. |
| **Week 22** | [**Stage 12 & Project 11:** Edge Inference Fallback](./syllabus.md#stage-12-edge-inference--portfolio) | • **Theory:** *On-device* execution (WebGPU in the browser, MLX on Apple Silicon, `llama.cpp` with GGUF) and *hybrid-edge* architectures.<br>• **Practice:** Configure a ~3B model running via WebGPU/MLX that automatically takes over inference when the cloud gateway injects failures (HTTP 503 / *chaos testing*). | **Project 11 Completed:** Video demo and code showing seamless cloud-to-edge transition during a simulated network outage. |
| **Week 23** | [**Stage 12 (Part 1):** Benchmark Experimental Design](./syllabus.md#stage-12-edge-inference--portfolio) | • **Theory:** Scientific inference benchmarking methodology (GPU warmup, Poisson arrival distributions, P50/P95/P99 percentiles).<br>• **Practice:** Automate a reproducible suite comparing **vLLM vs. SGLang vs. TensorRT-LLM** under multiple load profiles (short chat, long RAG, high concurrency). | Complete dataset with raw metrics for TTFT, ITL, total throughput, and cost per million tokens for each engine. |
| **Week 24** | [**Project 12:** Public Inference Benchmark Teardown](./syllabus.md#stage-12-edge-inference--portfolio) | • **Theory:** Visualizing efficiency frontiers (Pareto curves) and systems engineering technical writing.<br>• **Practice:** Generate Pareto curves (**Cost vs. TTFT vs. Throughput**), polish all 12 repositories with thorough documentation, and publish the technical *teardown*. | **Project 12 Completed:** Published technical article accompanied by a reproducible repository closing the entire roadmap cycle. |

---

## Execution & Infrastructure Recommendations

1. **GPU Cost Management:** **Weeks 1–10** and **19–22** can be executed almost entirely on a single local GPU or an affordable Spot instance (e.g., RTX 4090 or L4). Reserve cloud credits for multi-GPU instances (e.g., 2x or 4x H100/A100 Spot) specifically during **Weeks 12, 14, 15–18, and 23–24**.
2. **Single Repository (Study Monorepo):** Maintain a repository structured by phases (`/phase-1`, `/phase-2`, etc.), where each project includes a `README.md` containing: *chosen architecture, hypothesis, benchmark methodology, and result plots*.
