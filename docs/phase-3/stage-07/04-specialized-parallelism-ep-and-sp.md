# 04. Specialized Parallelism: Expert Parallelism (EP) & Sequence Parallelism (SP)

[Previous: Intra-Layer & Inter-Layer Parallelism](./03-intra-and-inter-layer-parallelism.md) | [Stage 7 Index](./README.md) | [Next: Collective Communication (NCCL)](./05-collective-communication-nccl.md)

---

## 1. Expert Parallelism (EP) for Mixture-of-Experts (MoE)

### 1.1 Why Standard Tensor Parallelism Fails for Frontier MoE Models

Modern frontier MoE models (e.g., Mixtral-8x22B, DeepSeek-V3 671B with 256 routed experts + 1 shared expert, Qwen3-MoE) activate only $k$ out of $E$ experts per token (e.g., $k = 8$ out of $E = 256$). While **total parameters** are huge (requiring multi-GPU or multi-node VRAM), **active FLOPs per token** are small.

If you apply pure **Tensor Parallelism ($\text{TP} = 8$ or $16$)** to an MoE layer:
1. Each expert's FFN intermediate dimension $d_{\text{expert}}$ is already small (e.g., $d_{\text{expert}} = 2048$ in DeepSeek-V3 vs. $18{,}432$ in dense 70B). Sharding $2048$ across $16$ GPUs leaves tiny $[B_{\text{tok}}, 128]$ matrices—producing **catastrophically low GEMM efficiency** and wasted Tensor Core utilization.
2. Worse, during decode at batch size $B$, each GPU still executes $2 \times L$ `AllReduce` barriers per layer while reading weights for *multiple* sparsely activated experts.

**Expert Parallelism (EP)** solves this by partitioning the $E$ experts **across $N_{\text{EP}}$ GPUs** without slicing individual expert weight matrices along $d_{\text{expert}}$:
* Each GPU rank $r \in \{0, \dots, N_{\text{EP}}-1\}$ holds $E / N_{\text{EP}}$ complete experts (e.g., for $E = 256, N_{\text{EP}} = 64$, each GPU holds $4$ full experts).
* Instead of replicating tokens and reducing weights, **tokens move to the GPUs holding their selected experts** via two `AllToAll` collectives per MoE layer.

### 1.2 Token Dispatch & Combine Lifecycle (`AllToAll`)

Consider a batch of $T$ tokens on each DP/attention rank (with hidden dimension $d$ and top-$k$ routing):

1. **Local Gate & Routing Metadata:** Each GPU computes router logits $g = \text{softmax}(X W_{\text{gate}})$ and selects top-$k$ experts $(e_{i,1}, \dots, e_{i,k})$ with gating weights $(w_{i,1}, \dots, w_{i,k})$ for every local token $i \in \{1, \dots, T\}$.
2. **Metadata Exchange (`AllGather` or small `AllToAll`):** Ranks exchange per-rank token destination counts `send_counts[r]` so every GPU knows how many variable-length tokens it will receive from each peer (required for `ncclAllToAllv` / dynamic buffer allocation, unless fixed capacity padding is used).
3. **Token Permute & Dispatch (`AllToAll` #1):**
   * Each rank packs tokens contiguous by destination GPU rank (duplicating token $i$ up to $k$ times—once for each selected expert on a distinct rank, or deduplicating per destination node in hierarchical node-limited routing!).
   * Executes `AllToAll` (specifically `AllToAllv` for variable counts) to route token hidden states $[T \cdot k, d]$ across the $N_{\text{EP}}$ GPUs.
4. **Local Grouped GEMM (Expert Execution):**
   * Each GPU receives all tokens assigned to its local $E/N_{\text{EP}}$ experts from across the entire cluster and runs a single **Grouped GEMM** (`torch._grouped_mm` / CUTLASS Grouped GEMM) over its local experts.
   * Because tokens from $N_{\text{EP}}$ different GPUs are pooled together, each expert receives $N_{\text{EP}}\times$ more tokens per batch—raising GEMM arithmetic intensity dramatically!
5. **Token Combine (`AllToAll` #2) & Weighted Reduction:**
   * A second `AllToAll` sends computed expert outputs back to their originating GPUs.
   * Each originating GPU multiplies each returned expert output by its gating weight $w_{i,j}$ and sums across $j = 1, \dots, k$ (plus any local shared expert output).

#### Exact Communication Volume per MoE Layer
For $T$ tokens per GPU, hidden dimension $d$, top-$k$ active experts, and element size $b$ bytes (e.g., $b = 1$ byte in FP8 dispatch, $b = 2$ bytes in BF16 combine):
* Assuming uniform routing across $N_{\text{EP}}$ ranks, a fraction $\frac{N_{\text{EP}} - 1}{N_{\text{EP}}}$ of the $T \cdot k$ dispatched activations leave the local GPU:
  $$V_{\text{dispatch}} = T \cdot k \cdot d \cdot b_{\text{dispatch}} \cdot \frac{N_{\text{EP}} - 1}{N_{\text{EP}}}, \qquad V_{\text{combine}} = T \cdot k \cdot d \cdot b_{\text{combine}} \cdot \frac{N_{\text{EP}} - 1}{N_{\text{EP}}}$$
* **Hardware Optimization (FP8 Dispatch):** Because activations are quantized right before the first expert linear layer anyway, engines like DeepSeek-V3 dispatch tokens in **FP8 + per-group scales** ($1\text{ byte/elem}$ instead of $2\text{ bytes}$), cutting `AllToAll` #1 network bandwidth by **50%**, while combining outputs in BF16 for summation precision.

### 1.3 Load Imbalance, Capacity Factor, and Stragglers

Unlike dense TP where every GPU performs identical FLOPs, EP suffers from **dynamic routing skew**:
* **The Straggler Problem:** If a hot expert (e.g., punctuation/syntax expert) on GPU 3 receives $40\%$ of all tokens in a batch while GPU 7 receives $2\%$, **all $N_{\text{EP}} - 1$ GPUs stall** at the `AllToAll` #2 combine barrier waiting for GPU 3!
* **Capacity Factor (Dropping) vs. Dropless (`AllToAllv`):**
  * *Fixed Capacity ($C$):* Caps tokens per expert at $C \cdot \frac{T \cdot k}{E}$ ($C \in [1.0, 1.5]$). Guarantees deterministic buffer sizes and fixed latency, **but drops tokens** when exceeded—degrading model accuracy unpredictably in production.
  * *Dropless Inference (Standard in vLLM/SGLang):* Uses dynamic `AllToAllv` (or NVSHMEM P2P kernels like DeepEP) to process 100% of routed tokens without dropping.
* **Production Mitigations:** (1) **Redundant Hot-Expert Replication:** Duplicate the top-$m$ hottest experts (tracked via rolling exponential moving average of router counts) onto multiple GPUs and split traffic among replicas; (2) **Node-Limited Routing:** Constrain top-$k$ routing during training (e.g., DeepSeek-V3 restricts each token to at most $M = 4$ nodes out of $N_{\text{nodes}}$) so `AllToAll` traffic stays localized; (3) **Shared Expert Overlap:** Run dense shared experts on the GPU simultaneously while `AllToAll` #1 is in flight on an RDMA stream.

### 1.4 Hybrid Parallelism: `DP (Attention) + EP (MoE)` vs. `TP + EP`

How should Attention and MoE layers be combined across $G$ GPUs?

1. **Classic `TP + EP` (or Pure `TP` on Attention, `EP` on MoE):**
   * Attention uses `TP` (say $\text{TP} = 8$ inside a node) so KV cache is split $8\times$, while MoE uses $\text{EP} = 8$ (or $\text{EP} = 64$ across nodes).
   * **Problem during Decode:** Under `TP = 8` for Attention, all $8$ GPUs process the *same* batch of $B$ requests (or replicate KV heads when $H_{\text{kv}} < \text{TP}$). Specifically, for **Multi-Head Latent Attention (MLA)** (DeepSeek-V2/V3), there is only **1 compressed latent KV vector** ($d_c = 512$) per token! Sharding MLA across $\text{TP} = 8$ forces you to **replicate the compressed KV cache $8\times$** across all 8 GPUs!

2. **Modern `DP (Attention) + EP (MoE)` (DeepSeek-V3 / SGLang Wide-EP Architecture):**
   * Run **Data Parallelism ($\text{DP} = G$)** for the Attention layers: each GPU handles *different* user requests ($B/G$ requests per GPU), holding **zero duplicated KV cache** (100% distinct KV entries per GPU)!
   * At the MoE boundary, switch seamlessly to **Expert Parallelism ($\text{EP} = G$)** via `AllToAll` dispatch/combine across all $G$ GPUs (often spanning $32$ to $128$ H100/H800 GPUs).
   * To eliminate `AllToAll` latency stalls, **Dual-Batch / Two-Microbatch Overlap** splits the local batch into two halves ($B_A, B_B$): while micro-batch $B_A$ is waiting on `AllToAll` network transfer, the GPU executes Attention/Shared-Expert compute for micro-batch $B_B$!

---

## 2. Sequence / Context Parallelism (SP) for Ultra-Long Contexts

When serving ultra-long context lengths ($S = 128\text{k}$ to $2\text{M}$ tokens), **neither TP nor PP** solves the single-request activation and KV cache memory explosion during Prefill:
* At $S = 1\text{M}$ tokens in Llama-3-70B, attention activations and the KV cache ($320\text{ GB}$ just for KV in BF16!) far exceed a single GPU's HBM—and even with $\text{TP}=8$, $H_{\text{kv}} = 8$ limits TP scaling beyond $8$ GPUs!
* **Context / Sequence Parallelism (CP / SP)** splits the sequence dimension $S$ into $P$ chunks of length $S/P$ across $P$ GPUs. The hard engineering challenge is **Self-Attention**, where every query token in chunk $i$ must attend to **every key/value token in all chunks $0 \dots i$**.

Two dominant algorithms solve cross-chunk attention: **DeepSpeed-Ulysses** and **Ring Attention**.

### 2.1 DeepSpeed-Ulysses (`AllToAll` Head-Switch)

**DeepSpeed-Ulysses** (Jacobs et al., 2023) avoids distributed attention math completely by **transposing the sharding axis** from *sequence* to *attention heads* right before FlashAttention:

1. **Sequence-Sharded Input:** Each of the $P$ GPUs holds a local sequence shard of length $S/P$, shape $[B, S/P, d]$, and computes local Q, K, V projections independently:
   $$Q^{(r)}, K^{(r)}, V^{(r)} \in \mathbb{R}^{B \times (S/P) \times H \times d_h}$$
2. **Pre-Attention `AllToAll` (Sequence-Split $\to$ Head-Split):**
   * Execute `AllToAll` on $Q, K, V$ across the $P$ GPUs.
   * Each GPU transitions from holding *(part of sequence, all $H$ heads)* to holding ***(entire sequence $S$, only $H/P$ heads)***:
     $$Q_{\text{head}}^{(r)}, K_{\text{head}}^{(r)}, V_{\text{head}}^{(r)} \in \mathbb{R}^{B \times S \times (H/P) \times d_h}$$
3. **Local Exact FlashAttention:** Because each GPU now possesses the **full sequence $S$** for its $H/P$ heads, it runs standard single-GPU FlashAttention-2/3 locally with **zero custom distributed attention kernels** and native causal masking!
4. **Post-Attention `AllToAll` (Head-Split $\to$ Sequence-Split):**
   * Execute one more `AllToAll` on the attention output $O_{\text{head}}^{(r)} \in \mathbb{R}^{B \times S \times (H/P) \times d_h}$ to switch back to sequence-sharded layout $[B, S/P, d]$ for the Row/MLP layers.

#### Exact Communication Volume & Hard Constraint
* **Communication Volume per GPU per Layer:**
  Let $M_{\text{seq}} = B \cdot S \cdot d \cdot b$ bytes be the size of one full activation tensor ($d = H \cdot d_h$). Across $4$ `AllToAll` calls ($Q, K, V$ pre-attention + $O$ post-attention; or $2 + 2\frac{H_{\text{kv}}}{H}$ for GQA):
  $$V_{\text{Ulysses}} = \left(2 + 2\frac{H_{\text{kv}}}{H}\right) \cdot \frac{P - 1}{P} \cdot \frac{M_{\text{seq}}}{P}$$
  Notice the crucial $\mathbf{1/P}$ scaling per link: as $P$ increases, the bytes sent *per GPU* actually **decrease**! (Total aggregate cluster traffic stays $O(S)$, distributed across $P$ links).
* **Hard Architectural Limitation ($P \le H_{\text{kv}}$):** Ulysses **cannot scale beyond the number of attention heads** (specifically $H_{\text{kv}}$ for KV unless KV heads are replicated prior to `AllToAll`). For a GQA model with $H_{\text{kv}} = 8$, Ulysses hits a hard wall at $P = 8$ GPUs!

### 2.2 Ring Attention (Overlapped P2P KV Ring + Online Softmax)

**Ring Attention** (Liu et al., 2023) keeps **queries $Q^{(r)} \in \mathbb{R}^{B \times (S/P) \times H \times d_h}$ stationary** on GPU $r$ and circulates the $K, V$ blocks in a **logical ring** across all $P$ GPUs using asynchronous point-to-point `Send`/`Recv` overlapped with FlashAttention compute.

#### Mathematical Mechanics (Blockwise Online Softmax)
Suppose GPU $r$ holds local query chunk $Q^{(r)}$ and receives block $K^{(j)}, V^{(j)}$ at ring step $t$:
1. **Local Tile Attention:** GPU $r$ runs FlashAttention on $(Q^{(r)}, K^{(j)}, V^{(j)})$ to obtain local block output $\tilde{O}_{r,j}$, local row-wise max $m_{r,j}$, and local log-sum-exp denominator $l_{r,j} = \sum \exp(S_{r,j} - m_{r,j})$.
2. **Exact Online Softmax Merge:** GPU $r$ updates its running accumulators $(O_r, m_r, l_r)$ using exact FlashAttention rescaling:
   $$m_{\text{new}} = \max(m_r, m_{r,j}), \qquad l_{\text{new}} = e^{m_r - m_{\text{new}}} l_r + e^{m_{r,j} - m_{\text{new}}} l_{r,j}$$
   $$O_r \leftarrow \frac{e^{m_r - m_{\text{new}}} l_r \cdot O_r + e^{m_{r,j} - m_{\text{new}}} l_{r,j} \cdot \tilde{O}_{r,j}}{l_{\text{new}}}$$
3. **Simultaneous Async P2P Ring Rotation:** *While* GPU $r$'s Tensor Cores are computing attention on $(Q^{(r)}, K^{(j)}, V^{(j)})$, its CUDA communication stream asynchronously sends $(K^{(j)}, V^{(j)})$ to rank $(r+1) \bmod P$ and receives $(K^{(j-1)}, V^{(j-1)})$ from rank $(r-1) \bmod P$ into a **double buffer**.
4. After $P$ ring steps, every GPU has attended over all $P$ KV chunks and holds the exact final output $O_r$.

#### Causal Masking Load Imbalance & Zig-Zag (Striped) Ring Attention
Naive contiguous sequence splitting causes severe **causal load imbalance**:
* Under causal attention, token $i$ only attends to tokens $\le i$.
* GPU $0$ (holding tokens $0 \dots \frac{S}{P}-1$) skips $P-1$ future KV blocks (doing $\frac{1}{P^2}$ of total FLOPs), while GPU $P-1$ (holding the end of the prompt) attends to **all $P$ KV blocks** (doing $\frac{2P-1}{P^2}$ of total FLOPs—up to $(2P-1)\times$ more work than GPU 0!).
* **Fix — Zig-Zag (Symmetric) Partitioning:** Split the sequence into $2P$ chunks $\{C_0, C_1, \dots, C_{2P-1}\}$ and assign GPU $r$ a **front + back pair** $(C_r, C_{2P-1-r})$! Every GPU's total causal attention area sums to the **exact same constant** $\frac{1}{P} \cdot \frac{S^2}{2}$, restoring $100\%$ balanced compute and compute/comm overlap across all $P$ GPUs.

#### Compute-Communication Overlap Condition
At each of the $P - 1$ ring steps, GPU $r$ transfers $K^{(j)}, V^{(j)}$ of size $2 \cdot B \cdot (S/P) \cdot H_{\text{kv}} \cdot d_h \cdot b$ bytes over interconnect bandwidth $BW_{\text{link}}$, while performing $4 \cdot B \cdot (S/P)^2 \cdot H \cdot d_h$ FLOPs (or $2\times$ under causal masking) at Tensor Core throughput $F_{\text{peak}}$:
$$T_{\text{comm}} \le T_{\text{comp}} \iff \frac{2 \cdot B (S/P) H_{\text{kv}} d_h b}{BW_{\text{link}}} \le \frac{2 \cdot B (S/P)^2 H d_h}{F_{\text{peak}}} \implies \frac{S}{P} \ge \frac{F_{\text{peak}}}{BW_{\text{link}}} \cdot \frac{H_{\text{kv}}}{H} \cdot b$$

**Concrete H100 Calculation:**
* $F_{\text{peak}} = 989\text{ TFLOP/s}$, NVLink $BW_{\text{link}} = 450\text{ GB/s}$ unidirectional, BF16 $b = 2\text{ B}$, GQA $H_{\text{kv}}/H = 1/8$:
  $$\frac{S}{P} \ge \frac{989 \times 10^{12}}{450 \times 10^9} \cdot \frac{1}{8} \cdot 2 \approx \mathbf{550\text{ tokens/GPU}}$$
* Over InfiniBand NDR ($BW_{\text{link}} = 50\text{ GB/s}$): $\frac{S}{P} \ge 4{,}945\text{ tokens/GPU}$. Thus, for long-context prefill ($S/P \ge 8\text{k}$), **Ring Attention achieves near-100% communication hiding even across InfiniBand nodes**!

---

## 3. Communication Volume & Interconnect Comparison

| Parallelism Strategy | Collective / Mechanism | Bytes Moved per GPU per Layer | Hard Scaling Constraint | Ideal Interconnect |
| :--- | :--- | :--- | :--- | :--- |
| **Expert Parallelism (EP)** | $2\times$ `AllToAllv` (Dispatch + Combine) | $2 \cdot T k d b_{\text{avg}} \frac{N_{\text{EP}}-1}{N_{\text{EP}}}$ (non-overlapped if single batch) | Load skew / hot experts; small batch latency | NVLink/NVSwitch (decode) or IB NDR w/ dual-batch overlap |
| **DeepSpeed-Ulysses (SP)** | $4\times$ `AllToAll` (Q, K, V, O head-switch) | $(2 + 2\frac{H_{\text{kv}}}{H}) \frac{P-1}{P} \frac{B S d b}{P}$ | **$P \le H_{\text{kv}}$** (cannot scale past KV head count without replication) | High-bisection NVLink / Fat-Tree IB (cannot hide `AllToAll` inside attention) |
| **Ring Attention (CP)** | $(P-1)\times$ async P2P `Send/Recv` of $(K, V)$ | $2 (P - 1) \cdot B \frac{S}{P} H_{\text{kv}} d_h b$ (total, **overlapped** with compute) | Requires local chunk $S/P \gtrsim 550\text{--}5\text{k}$ tokens to hide comm; **prefill/chunked-prefill only** | Tolerates **PCIe & cross-node InfiniBand** (ring topology matches P2P links) |
| **Unified Hybrid (USP)** | **Ulysses $\times$ Ring 2D Mesh** | Ulysses inside NVLink node ($P_U \le H_{\text{kv}}$); Ring across nodes ($P_R$) | Eliminates both Ulysses head limit and cross-node `AllToAll` bottleneck | NVLink intra-node + InfiniBand inter-node |

---

## 4. Architecture & Execution Diagrams

### 4.1 Ring Attention (P2P KV Ring) vs. DeepSpeed-Ulysses (`AllToAll` Head-Switch)

```mermaid
flowchart TD
    SeqIn["Long Sequence Sharded Across 4 GPUs: GPU r holds S/4 tokens, all H heads"]

    SeqIn --> U_Path["DeepSpeed-Ulysses Path"]
    SeqIn --> R_Path["Ring Attention Path"]

    U_Path --> U1["1. Local QKV Proj: [S/4, H, d_h]"]
    U1 --> U2["2. Pre-Attn AllToAll: Transpose Seq <-> Heads"]
    U2 --> U3["3. Local FlashAttention: Full Sequence S, only H/4 heads"]
    U3 --> U4["4. Post-Attn AllToAll: Transpose Heads <-> Seq -> [S/4, d]"]

    R_Path --> R1["1. Local QKV Proj: Q_r, K_r, V_r of shape [S/4, H, d_h]"]
    R1 --> R2["2. Hold Q_r stationary; Double-buffer K, V blocks"]
    R2 --> R3["3. Ring Step t: FlashAttention(Q_r, K_j, V_j) + Online Softmax Merge"]
    R3 --> R4["4. Overlapped Async P2P Send/Recv: Pass K_j, V_j to (r+1) mod 4"]
    R4 -->|"Repeat P-1 times"| R3
```

### 4.2 Overlapped Ring Attention Compute and P2P KV Rotation (4 GPUs)

```mermaid
sequenceDiagram
    participant G0 as "GPU 0 (Q_0)"
    participant G1 as "GPU 1 (Q_1)"
    participant G2 as "GPU 2 (Q_2)"
    participant G3 as "GPU 3 (Q_3)"

    Note over G0,G3: Step 0 - Compute local block + Start Async Ring Send/Recv #1
    par Step 0 Compute & Comm Overlap
        G0->>G0: FlashAttn(Q_0, K_0, V_0) -> init (O_0, m_0, l_0)
        G1->>G1: FlashAttn(Q_1, K_1, V_1) -> init (O_1, m_1, l_1)
        G2->>G2: FlashAttn(Q_2, K_2, V_2) -> init (O_2, m_2, l_2)
        G3->>G3: FlashAttn(Q_3, K_3, V_3) -> init (O_3, m_3, l_3)
        G0->>G1: Async Send (K_0, V_0)
        G1->>G2: Async Send (K_1, V_1)
        G2->>G3: Async Send (K_2, V_2)
        G3->>G0: Async Send (K_3, V_3)
    end

    Note over G0,G3: Step 1 - Online Softmax Merge on received KV + Async Ring Send/Recv #2
    par Step 1 Compute & Comm Overlap
        G0->>G0: FlashAttn(Q_0, K_3, V_3) + Online Softmax Update
        G1->>G1: FlashAttn(Q_1, K_0, V_0) + Online Softmax Update
        G2->>G2: FlashAttn(Q_2, K_1, V_1) + Online Softmax Update
        G3->>G3: FlashAttn(Q_3, K_2, V_2) + Online Softmax Update
        G0->>G1: Async Send (K_3, V_3)
        G1->>G2: Async Send (K_0, V_0)
        G2->>G3: Async Send (K_1, V_1)
        G3->>G0: Async Send (K_2, V_2)
    end

    Note over G0,G3: Steps 2 and 3 complete remaining 2 rotations -> Exact Full-Context Output O_0..O_3
```

---

[Previous: Intra-Layer & Inter-Layer Parallelism](./03-intra-and-inter-layer-parallelism.md) | [Stage 7 Index](./README.md) | [Next: Collective Communication (NCCL)](./05-collective-communication-nccl.md)
