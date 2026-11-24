# LLM Inference Engineering — Complete Syllabus

[Back to Project Overview](./README.md) | [24-Week Execution Schedule](./schedule.md)

This syllabus unifies **12 theoretical and technical stages** of LLM inference engineering with **12 hands-on portfolio projects**, organizing progression so that each theoretical module leads directly into building and benchmarking the corresponding system.

The curriculum is divided into **4 Macro Phases**, spanning from bare metal (hardware architecture and GPU kernels) up to edge deployment and production unit economics.

---

## Phase 1: Hardware Foundations, Silicon & Model Anatomy

*Focus: Understand the physics behind GPU computation, memory vs. compute bottlenecks, and the mathematics of modern Transformers.*

### Stage 1: Systems Foundations

* **Topics:** C/C++ or Rust applied to inference, Python performance profiling (`py-spy`, `cProfile`), Linux internals (memory management, I/O), memory hierarchy (L1/L2, RAM, NVMe), and CPU vs. GPU vs. NPU execution models.
  * *Memory Hierarchy & Buses:* Registers, L1/L2/L3 caches, DDR5 RAM, PCIe Gen4/Gen5 bandwidth, NVMe, and access latency vs. throughput.
  * *Hardware Execution Models:* MIMD/SIMD architecture (CPU), SIMT/Massively Parallel (GPU), and Systolic Arrays & dedicated dataflow (NPU).
  * *Linux Internals for I/O & Memory:* Virtual memory, Page Cache, `mmap` (memory-mapped files), *zero-copy I/O*, *huge pages*, pinned memory (`mlock`), and NUMA topology.
  * *C/C++ and Rust on the Hot Path:* Manual memory management, RAII, *ownership/borrowing*, SIMD vector instructions (AVX-512/NEON), thread parallelism (OpenMP / Rayon), and zero-copy FFI (`PyO3`/`maturin`, `nanobind`, DLPack protocol).
  * *Python Profiling:* GIL bottlenecks, flamegraphs (`py-spy`), deterministic profiling (`cProfile`), allocation tracking (`tracemalloc`), and `torch.profiler`.
* **Study Deliverable:** Create profiling scripts identifying CPU/GIL bottlenecks and PCIe transfer overhead in simple inference pipelines.
  * *Step 1 — Baseline & Flamegraphs:* Instrument a tokenization + forward pass + sampling loop with `py-spy` and `torch.profiler` to isolate CPU time vs. GPU idle time.
  * *Step 2 — Bus Bottlenecks (Host-to-Device):* Measure tensor copy latency (`pin_memory=True` vs. pageable memory) and implicit synchronizations (`tensor.item()`, `.cpu()`).
  * *Step 3 — Loading & Hot Path Acceleration:* Implement weight reading via `mmap` (Rust/C++) and rewrite the sampler or pre-processing in native code via FFI, validating the latency reduction.

### Stage 2: GPU Architecture & Modern Silicon

* **Topics:** CUDA architecture and Tensor Cores, HBM3e memory bandwidth, Streaming Multiprocessor (SM) occupancy, specifics of Hopper (H100/H200) and Blackwell (B200) architectures, chiplet designs, and hardware telemetry with `nvidia-smi` and DCGM.
  * *CUDA Execution Hierarchy:* Grids, Thread Blocks, Warps (32 threads), SIMT execution, warp divergence, and warp scheduling.
  * *GPU Memory Hierarchy:* Per-thread registers, Shared Memory (SRAM), L1/L2 Cache, Global Memory (HBM3 / HBM3e), and memory access coalescing.
  * *Compute Units:* CUDA Cores (FP32/INT32) vs. Tensor Cores (MMA — Matrix Multiply-Accumulate in FP16, BF16, FP8, FP4).
  * *Occupancy & Modern Architectures:* SM Occupancy (registers and shared memory as limiters), Hopper (TMA — Tensor Memory Accelerator, Transformer Engine, NVLink 4.0), and Blackwell (dual-die chiplet, NVLink-C2C, native FP4 microscaling support).
  * *Hardware Telemetry:* `nvidia-smi dmon`, DCGM (Data Center GPU Manager), SM Active, SM Occupancy, Tensor Active, and DRAM Bandwidth Utilization metrics.
* **Study Deliverable:** Map the theoretical FLOP/s and Memory Bandwidth limits of the target GPU for future *Roofline* calculations.
  * *Step 1 — Specification Gathering:* Extract boost clocks, SM count, bus width, and HBM/GDDR speed to compute theoretical peak compute ($\text{FLOP/s}$) per precision and memory bandwidth ($\text{GB/s}$).
  * *Step 2 — Ridge Point Calculation:* Determine the critical arithmetic intensity ($\text{FLOPs/Byte} = \frac{\text{Peak FLOP/s}}{\text{Peak Bandwidth}}$) that separates *memory-bound* from *compute-bound* operations.
  * *Step 3 — Empirical Peak Validation:* Run microbenchmarks (large dense GEMM and STREAM vector copy) monitored via DCGM to record the practical peak achievable on real silicon.

### Stage 3: Transformer & MoE Inference Anatomy

* **Topics:** Physical differences between **Prefill** (*compute-bound*) and **Decode** (*memory-bandwidth-bound*), KV Cache mathematics, attention memory, arithmetic intensity, Mixture-of-Experts (MoE) routing, and long-context scalability.
  * *Inference Phase Anatomy:* Prefill (matrix-matrix multiplication GEMM, high arithmetic intensity, *compute-bound*) vs. Decode (matrix-vector multiplication GEMV, low arithmetic intensity, *memory-bandwidth-bound*).
  * *KV Cache Mathematics:* Exact calculation of bytes per token ($2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_param}$), and the impact of MHA, MQA, GQA, and MLA (Multi-Head Latent Attention).
  * *Attention Memory & Long Contexts:* Quadratic $O(N^2)$ attention complexity vs. memory linearization via tiling, RoPE scaling, and linear growth of the KV Cache with batch size.
  * *Mixture-of-Experts (MoE) Architectures:* Total parameters vs. active parameters per token, gating networks (Top-K routing), expert capacity, and HBM weight-reading bottlenecks during decode.
  * *Efficiency Metrics:* Per-layer arithmetic intensity (QKV proj, Attention, FFN/MoE), MFU (Model FLOPs Utilization), and MBU (Model Bandwidth Utilization).
* **Study Deliverable:** Implement an analytical calculator (in Python) that predicts exact KV Cache VRAM consumption and theoretical MFU given batch size and sequence length.
  * *Step 1 — Parameter & KV Cache Modeling:* Implement a Hugging Face `config.json` parser calculating static weight VRAM and the dynamic KV Cache growth curve per batch and context length.
  * *Step 2 — FLOPs & Bytes Moved Estimation:* Compute exact FLOPs and HBM bytes transferred per forward pass for Prefill and Decode separately (supporting both dense and MoE models).
  * *Step 3 — PyTorch Calibration:* Validate memory predictions against `torch.cuda.max_memory_allocated()` and compare the theoretical minimum time (via Roofline) against real measured latency.

---

## Phase 2: Serving Engines, Memory & Compression

*Focus: Master production serving frameworks, optimize memory usage, and shrink models without destroying output quality.*

### Stage 4: Serving Engines & Constrained Decoding

* **Topics:** Internal architecture of vLLM, SGLang, TGI, and TensorRT-LLM; *Continuous Batching*, *PagedAttention*, structured output decoding, and grammar-guided generation.
  * *Dynamic Scheduling:* Static batching vs. Continuous Batching (iteration-level scheduling), prefill/decode queues, and request preemption.
  * *Paged Memory Management:* PagedAttention, physical/logical block tables, elimination of internal/external fragmentation, and Copy-on-Write for beam search / parallel sampling.
  * *Engine Architectures:* vLLM execution loop, SGLang RadixAttention and frontend DSL, TensorRT-LLM ahead-of-time graph compilation, and TGI Rust/Python architecture.
  * *Constrained Decoding (Structured Outputs):* Regular expressions, Context-Free Grammars (CFG), deterministic finite automata (DFA), pushdown automata, and compiling JSON/Pydantic schemas into vocabulary masks.
  * *Logit Masking & Jump-Forward Decoding:* Applying $-\infty$ to invalid tokens pre-softmax and deterministically advancing multiple structural tokens without a forward pass.
* **Practical Project 4 — Grammar-Constrained Decoding Engine:** Implement a custom sampling loop using *Context-Free Grammars* (CFG) or finite state machines to enforce 100% valid JSON/Pydantic outputs at the logit/token level.
  * *What it demonstrates:* Ability to guarantee deterministic data contracts for agents without relying on fragile prompt engineering.

### Stage 5: KV Cache & Memory Hierarchies

* **Topics:** *Prefix Caching* (RadixAttention), *Chunked Prefill*, KV Cache quantization (FP8/INT8), hierarchical offloading (`GPU → CPU → SSD`), multi-turn session reuse, and eviction policies (LRU).
  * *Context Reuse:* Block-hash-based Automatic Prefix Caching (APC in vLLM) vs. dynamic prefix trees (RadixTree in SGLang).
  * *Hybrid Scheduling:* Chunked Prefill (slicing long prompts to mix prefill and decode tokens in the same batch and prevent ITL spikes).
  * *State Compression:* Dynamic/static KV Cache quantization (FP8 E4M3, INT8, INT4) and per-channel/per-token scaling to mitigate outliers.
  * *Offloading Hierarchy:* Asynchronous movement between VRAM, Pinned Host RAM (`cudaMemcpyAsync` on dedicated CUDA Streams), and NVMe storage (GPUDirect Storage / Linux `io_uring`).
  * *Management Policies:* Cache eviction policies (LRU, LFU, priority-aware eviction for active sessions) and preventing thrashing across memory tiers.
* **Practical Project 3 — Hierarchical KV-Cache Offloader:** Build a memory manager that asynchronously *spills* inactive KV blocks from GPU VRAM to CPU RAM and then to NVMe SSD without stalling the *decode* loop.
  * *What it demonstrates:* Ability to serve massive-context agents while keeping VRAM costs under control.

### Stage 6: Quantization & Compression at Scale

* **Topics:** FP4, FP8, and INT8 formats; post-training quantization methods (AWQ, GPTQ); sparsity; *mixed-precision* serving; and rigorous evaluation of perplexity drop vs. latency/throughput gains.
  * *Numerical Representation:* Low-precision floating point (FP8 E4M3/E5M2, FP4 / NF4 / MXFP4), symmetric/asymmetric integers (INT8, INT4), and scaling factors (per-tensor, per-channel, group-wise).
  * *Outliers & Calibration:* *Activation outliers* in large models, SmoothQuant (migrating quantization difficulty from activations to weights), AWQ (Activation-Aware Weight Quantization), and GPTQ (Hessian approximation).
  * *Hardware Execution:* Weight-only quantization (W4A16 — on-the-fly dequantization in registers to accelerate *memory-bound* decode) vs. Weight-Activation quantization (W8A8 / FP8 — direct Tensor Core usage to accelerate *compute-bound* prefill).
  * *Sparsity:* 2:4 structured sparsity (NVIDIA Sparse Tensor Cores) and post-training pruning.
  * *Scientific Quality Evaluation:* Perplexity calculation (WikiText-2 / C4), KL divergence on logit distributions, and reasoning/coding evaluation suites (`lm-evaluation-harness`).
* **Practical Project 6 — FP4 Quantization & Calibration Pipeline:** Calibrate an open model to FP4/INT4, measure exact perplexity degradation (and reasoning benchmarks), and compare latency gains and VRAM reduction against FP16/BF16.
  * *What it demonstrates:* Ability to compress models to fit on smaller hardware while mathematically proving the quality *trade-off*.

---

## Phase 3: Kernels, Advanced Parallelism & Distributed Systems

*Focus: Write low-level code close to the metal and scale giant models (dense and MoE) across multiple GPUs and nodes.*

### Stage 7: Speculative Decoding & Advanced Parallelism

* **Topics:** *Draft-Target* pipelines, Medusa and EAGLE heads, Tensor Parallelism (TP) vs. Pipeline Parallelism (PP) vs. Expert Parallelism (EP), and *Sequence Parallelism* (Ring Attention) for ultra-long contexts.
  * *Speculative Decoding Foundations:* Breaking the *memory-bound* decode bottleneck via parallel verification, and modified *rejection sampling* algorithm (mathematically guaranteeing exact preservation of the target model's distribution).
  * *Drafting Architectures:* Independent draft models (Draft-Target), auxiliary multi-token decoding heads (Medusa), feature-level extrapolation (EAGLE / EAGLE-2), and model-free drafting (n-gram / Lookahead decoding).
  * *Intra-Layer & Inter-Layer Parallelism:* Tensor Parallelism (Megatron-LM style, row/column-wise partitioning with `AllReduce`) vs. Pipeline Parallelism (micro-batching, reducing *pipeline bubbles*).
  * *Specialized Parallelism:* Expert Parallelism (EP for MoE) and Sequence / Context Parallelism (Ulysses, Ring Attention overlapping P2P communication and attention computation).
  * *Collective Communication:* NCCL primitives (`AllReduce`, `AllGather`, `ReduceScatter`, `AllToAll`) and the impact of interconnect topology on per-token latency.
* **Practical Project 7 — Speculative Decoding Harness:** Set up a lightweight *draft* model (e.g., 2B) running locally to propose multiple candidate tokens that are verified in a single *forward pass* by a heavy *target* model (e.g., 70B).
  * *What it demonstrates:* Mastery of *memory-bound* decode physics to achieve ~2x *speedup* without altering the mathematical output distribution.

### Stage 8: Kernel-Level Optimization

* **Topics:** Triton programming, FlashAttention-3, CUDA Graphs, *operator fusion*, custom attention kernels, deep profiling with NVIDIA Nsight Compute, and *Roofline* model analysis.
  * *Triton Programming Model:* *Block-level programming*, `tl.program_id`, pointer arithmetic, boundary masks, and autotuning `BLOCK_SIZE` and `num_warps`.
  * *Kernel Memory Optimization:* Coalesced HBM reads/writes, explicit SRAM (Shared Memory) usage, preventing *bank conflicts*, and vectorized loads.
  * *Operator Fusion & CUDA Graphs:* Kernel fusion (combining RMSNorm + Residual Add + Quantization into a single HBM read/write pass) and CUDA Graph capture to eliminate CPU kernel launch overhead.
  * *Exact Attention Algorithms:* Online Softmax, Q/K/V tiling in SRAM (FlashAttention-2), and warp-specialized asynchrony + FP8 with TMA on Hopper (FlashAttention-3).
  * *Profiling & Roofline Analysis:* NVIDIA Nsight Compute (`ncu`), reading *Memory Throughput*, *Compute (SM) Throughput*, and *Warp Stall Reasons* metrics, and empirical plotting on a Roofline chart.
* **Practical Project 1 — Custom Triton Kernel Optimizer:** Write a fused *RMSNorm* or *FlashAttention* kernel in Triton, profile memory bandwidth utilization via Nsight Compute, and benchmark against native PyTorch (`torch.compile`).
  * *What it demonstrates:* Proficiency in silicon-level optimization and *Roofline* analysis, going far beyond using out-of-the-box APIs.

### Stage 9: Distributed & Disaggregated Inference

* **Topics:** Multi-node serving, NVLink, InfiniBand, and RDMA interconnects, disaggregated *Prefill* and *Decode* pools, MoE expert parallelism, and tuning the Prefill/Decode node ratio.
  * *High-Speed Network Fabrics:* NVLink / NVSwitch (intra-node), InfiniBand, RoCEv2, RDMA (Remote Direct Memory Access), and GPUDirect RDMA (inter-node GPU-to-GPU transfer bypassing the CPU).
  * *Disaggregated Architecture (Prefill/Decode Disaggregation):* Physical separation of Prefill instances (optimized for dense compute) and Decode instances (optimized for memory bandwidth), eliminating *prefill* interference on ITL.
  * *State Transport:* Serialization and direct transfer of KV Cache blocks from the Prefill instance to the Decode instance via RDMA/NIXL.
  * *Distributed MoE Systems Engineering:* *All-to-All* token routing across GPUs, *expert load imbalance*, *capacity factor*, redundant routing of hot experts, and compute/communication overlap.
  * *Cluster Sizing:* Calculating *Prefill-to-Decode ratio* based on input vs. output length distributions and SLO targets (TTFT vs. ITL).
* **Practical Projects (Execute in sequence or integrated):**
  * **Project 2 — Disaggregated Prefill/Decode Cluster:** Orchestrate routing where compute-heavy *prefills* go to high-FLOP/s GPUs (e.g., H100) and memory-bound *decodes* go to bandwidth/cost-optimized GPUs (e.g., L40S), transferring the KV Cache via RDMA/InfiniBand. (*Demonstrates maximizing MFU by separating the two physical phases of inference*).
  * **Project 5 — MoE Expert-Parallel Router:** *Shard* the weights of a Mixture-of-Experts model across multiple GPUs, optimizing *all-to-all* communication overhead and balancing load across experts. (*Demonstrates capability to operate frontier sparse architectures at scale*).

---

## Phase 4: FinOps, Intelligent Routing, Edge & Portfolio

*Focus: Transform technical infrastructure into resilient, economically viable systems backed by public benchmarks.*

### Stage 10: Autoscaling, FinOps & Economics

* **Topics:** Queue-based autoscaling (KEDA), *cold start* mitigation (fast weight loading via GDS/S3), Spot GPU orchestration, cost-per-million-tokens modeling, MFU tracking, and per-*tenant* budgeting.
  * *Inference-Metric-Driven Autoscaling:* Why CPU/GPU utilization fails for LLMs; scaling via KEDA using request queue depth, tokens awaiting prefill, and KV Cache block saturation.
  * *Cold Start Mitigation:* Optimized container images (lazy pulling / eStargz), parallel checkpoint streaming from S3/NVMe directly to VRAM (GPUDirect Storage / Run:ai Model Streamer), and fast local storage caching.
  * *Spot Orchestration & Resilience:* Handling interruption signals (*spot preemption notices*), graceful draining of active SSE/streaming connections, and state migration.
  * *Inference Unit Economics:* Cost formula per 1M input/output tokens as a function of GPU hourly cost, actual MFU, and average occupancy rate; cost vs. batch size curves.
  * *Multi-Tenant Governance:* Accurate token accounting (prompt + completion + cached tokens), dynamic quotas, and resource isolation.
* **Practical Project 8 — Queue-Based GPU Autoscaler (FinOps):** Build a KEDA-driven autoscaler that provisions Spot GPU instances based on pending token queue depth and performs graceful *drain* when the queue empties.
  * *What it demonstrates:* *Unit economics* mindset, preventing idle GPUs from burning budget.

### Stage 11: AI Gateway, Routing & Reliability

* **Topics:** Intelligent model routing, *fallback* chains, semantic caching, token-based *rate limiting* (TPM/RPM), TTFT (Time to First Token) and ITL (Inter-Token Latency) SLOs, *session-aware* routing for agents, observability, and *chaos testing*.
  * *Dynamic Routing & Model Cascading:* Prompt complexity classification (token entropy, trained lightweight classifiers, structural rules) to route between small local models and large frontier models.
  * *Semantic Caching:* Low-latency embedding models, in-memory vector stores (HNSW), cosine similarity threshold calibration to avoid false positives, and context-based invalidation.
  * *Session-Aware Routing (Sticky/Prefix-Aware Routing):* Directing subsequent turns of the same agent or shared prefixes to the specific replica already holding the warm KV Cache in VRAM.
  * *Traffic Control & Reliability:* Token-bucket rate limiting based on token estimation (TPM), circuit breakers, retries with exponential backoff, and multi-provider/multi-region fallback chains.
  * *Observability & Chaos Engineering:* Granular metrics (TTFT, ITL/TPOT, E2E latency, cache hit ratio, queue time) via OpenTelemetry/Prometheus and controlled fault injection (HTTP 503, latency spikes, replica drops).
* **Practical Projects (Components of the same Gateway):**
  * **Project 9 — Semantic AI Gateway & Cache:** Generate embeddings for incoming prompts, query a vector index for >98% similarity, and return cached responses without hitting the GPU. (*Demonstrates that the fastest and cheapest token is the one never generated*).
  * **Project 10 — Multi-Model Complexity Router:** Analyze prompt entropy or complexity to route simple queries to a local quantized 8B model and complex reasoning to a frontier model (e.g., 405B). (*Demonstrates drastic cost reduction without UX degradation*).

### Stage 12: Edge Inference & Portfolio

* **Topics:** *On-device* inference (MLX for Apple Silicon, `llama.cpp`, WebGPU in the browser), publishing rigorous benchmarks (latency, throughput, cost), technical writing on kernel optimization, and architecture *teardowns*.
  * *Edge Runtimes & Unified Architecture:* `llama.cpp` and GGUF format (K-quants quantization, hybrid CPU/GPU execution), MLX (leveraging zero-copy unified memory on Apple Silicon), and WebGPU (WGSL compute shaders running models in the client browser).
  * *Hybrid Cloud-Edge Architectures:* Detecting cloud degradation/outage and transparently failing over to compact local models (1B–4B) while preserving application continuity.
  * *Scientific Benchmarking Methodology:* Proper GPU warmup, realistic traffic generation (Poisson arrival distributions, datasets with real input/output distributions like ShareGPT), noise isolation, and percentile measurement (P50, P95, P99).
  * *Pareto Frontier Analysis:* Plotting and interpreting Pareto curves relating **Cost per 1M tokens vs. TTFT vs. Total Throughput per GPU** under strict SLA constraints.
  * *Engineering Technical Communication:* Structuring architectural *teardowns*, analyzing source code/assembly (PTX/SASS), and publishing 100% reproducible artifacts.
* **Practical Projects (Capstone & Showcase):**
  * **Project 11 — WebGPU / MLX Edge Fallback:** Run a ~3B model directly in the browser (via WebGPU) or locally on Apple Silicon (via MLX) as an automatic survival mechanism when the cloud API returns HTTP 503. (*Demonstrates resilient architecture that pushes compute to the edge during cloud failures*).
  * **Project 12 — Public Inference Benchmark Teardown:** Publish a complete technical article containing Pareto curves (**Cost vs. TTFT vs. Throughput**) comparing vLLM, SGLang, and TensorRT-LLM on specific hardware configurations. (*Demonstrates scientific rigor in infrastructure evaluation and high-level technical communication*).

---

## Stage-to-Project Mapping

| Phase | Theoretical Stage | Corresponding Practical Project |
| --- | --- | --- |
| **1. Silicon & Anatomy** | Stages 1, 2, and 3 | KV Cache / MFU Calculators & Hardware Profiling |
| **2. Engines & Memory** | Stage 4: Serving Engines | **Project 4:** Grammar-Constrained Decoding Engine |
| **2. Engines & Memory** | Stage 5: KV Cache Hierarchies | **Project 3:** Hierarchical KV-Cache Offloader |
| **2. Engines & Memory** | Stage 6: Quantization | **Project 6:** FP4 Quantization & Calibration Pipeline |
| **3. Kernels & Scale** | Stage 7: Speculative Decoding | **Project 7:** Speculative Decoding Harness |
| **3. Kernels & Scale** | Stage 8: Kernel Optimization | **Project 1:** Custom Triton Kernel Optimizer |
| **3. Kernels & Scale** | Stage 9: Distributed Inference | **Project 2:** Disaggregated Prefill/Decode Cluster<br>**Project 5:** MoE Expert-Parallel Router |
| **4. Production & Portfolio** | Stage 10: Autoscaling & FinOps | **Project 8:** Queue-Based GPU Autoscaler |
| **4. Production & Portfolio** | Stage 11: AI Gateway & Reliability | **Project 9:** Semantic AI Gateway & Cache<br>**Project 10:** Multi-Model Complexity Router |
| **4. Production & Portfolio** | Stage 12: Edge & Portfolio | **Project 11:** WebGPU / MLX Edge Fallback<br>**Project 12:** Public Inference Benchmark Teardown |
