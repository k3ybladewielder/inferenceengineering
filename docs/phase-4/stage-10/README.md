# Stage 10: Autoscaling, FinOps & Economics

**Navigation:** [Curriculum Home](../../README.md) | [Complete Syllabus](../../syllabus.md) | **Phase 4: FinOps, Intelligent Routing, Edge & Portfolio**

---

## Stage Overview

High-performance inference kernels and distributed serving topologies solve *peak* efficiency on a single node or static cluster, but production LLM platforms live or die on **elasticity, resilience, and unit economics**. A cluster running at 90% MFU during a 15-minute traffic spike will still destroy gross margins if it spends the remaining 23 hours sitting 85% idle, takes 6 minutes to cold-start new replicas, loses active streaming generations to Spot preemptions, or allows a single noisy tenant to exhaust the shared PagedAttention KV cache.

Stage 10 bridges low-level inference systems physics with cloud-native orchestration and financial engineering:

1. **Inference-Metric-Driven Autoscaling:** Replacing broken OS/hardware utilization triggers (`GPU-Util`, CPU, static VRAM allocation) with engine-internal queue, prefill-token, KV-cache saturation, and SLO burn-rate signals via KEDA and Kubernetes HPA hysteresis.
2. **Cold Start Mitigation:** Compressing a 3–8 minute Kubernetes GPU pod cold start down to seconds using lazy container pulling (eStargz, SOCI, EROFS), NVMe DaemonSet caches, and parallel checkpoint streaming directly into GPU VRAM bypassing the host page cache (GPUDirect Storage, Run:ai Model Streamer, Tensorizer).
3. **Spot Orchestration & Resilience:** Capturing 60–80% cloud GPU cost reductions on preemptible capacity without breaking user-facing Server-Sent Events (SSE) streams, combining cloud interruption handlers, instant admission cut-off, graceful drain, token/KV state continuation, and multi-tier On-Demand + Spot capacity pools.
4. **Inference Unit Economics:** Deriving exact mathematical cost formulas ($/1M input and $/1M output tokens) from GPU hourly cost, MFU, MBU, KV-cache residency, and diurnal occupancy; quantifying the 3x–10x output-vs-input cost asymmetry, batch-size Pareto frontiers, and SKU selection across H100, H200, B200, and L40S.
5. **Multi-Tenant Governance:** Implementing exact token-type metering (cached prefix, visible output, hidden reasoning, tool calls), atomic two-phase TPM/RPM/concurrency quotas, scheduler-level noisy-neighbor isolation (Deficit Round Robin, per-tenant KV caps, S-LoRA isolation), and idempotent chargeback telemetry pipelines.

---

## Topic Modules

| # | Document | Core Focus & Key Mechanisms |
| :--- | :--- | :--- |
| **01** | [Inference-Metric-Driven Autoscaling](./01-inference-metric-driven-autoscaling.md) | Failure modes of `GPU-Util` & static VRAM metrics; `num_requests_waiting`, queued prefill tokens, `gpu_cache_usage_perc`, SLO burn rates; KEDA `ScaledObject` PromQL triggers, HPA math, asymmetric stabilization windows, and hysteresis. |
| **02** | [Cold Start Mitigation](./02-cold-start-mitigation.md) | 6-stage cold-start breakdown; lazy image pulling (eStargz, AWS SOCI, EROFS/SquashFS); bypassing host page cache with GPUDirect Storage (`cuFile`), Run:ai Model Streamer, and Tensorizer; local NVMe DaemonSets, JIT/CUDA Graph caching, and standby balloon pools. |
| **03** | [Spot Orchestration & Resilience](./03-spot-orchestration-and-resilience.md) | Spot pricing & 30s–120s interruption windows (AWS/GCP/Azure); IMDS polling, Node Termination Handler & Karpenter; readiness drop + `503` admission cut-off; SSE graceful drain vs. token-state continuation & RDMA KV migration; mixed On-Demand base + Spot burst pools. |
| **04** | [Inference Unit Economics](./04-inference-unit-economics.md) | First-principles derivation of $/1M input & output tokens from $C_{\text{gpu}}$, $\Phi_{\text{peak}}$, $\beta_{\text{HBM}}$, MFU, MBU, and occupancy $U_{\text{avg}}$; physical drivers of 3x–10x output cost; Cost vs. Batch Size vs. ITL Pareto curves; H100 vs. H200 vs. B200 vs. L40S SKU selection. |
| **05** | [Multi-Tenant Governance](./05-multi-tenant-governance.md) | Granular token taxonomy (uncached, prefix-cached, visible, reasoning, tool-call); two-phase Redis Lua TPM/RPM/concurrency reservation & reconciliation; Deficit Round Robin (DRR) fair queuing, KV quota caps, prefix-cache & LoRA isolation; Kafka/ClickHouse chargeback pipelines. |

---

## Navigation

* **Up:** [Curriculum Home](../../README.md) | [Complete Syllabus](../../syllabus.md)
* **Previous Stage:** [← Stage 9: Distributed & Disaggregated Inference](../../phase-3/stage-09/README.md)
* **Start Reading:** [01 — Inference-Metric-Driven Autoscaling](./01-inference-metric-driven-autoscaling.md)
* **Next Stage:** [Stage 11: AI Gateway, Routing & Reliability →](../stage-11/README.md)
