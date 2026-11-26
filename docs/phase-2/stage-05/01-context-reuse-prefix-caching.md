# 01 — Context Reuse: Automatic Prefix Caching & RadixAttention

[← Stage 5 Overview](./README.md) · [Main Docs](../../README.md) · [Syllabus](../../syllabus.md) · [Next: Chunked Prefill →](./02-hybrid-scheduling-chunked-prefill.md)

---

When serving Large Language Models (LLMs), different users and multi-turn conversations often send requests that start with the exact same opening text—such as system instructions or shared documents. This lesson explains how inference engines save and reuse the intermediate Key and Value tensors from those shared openings so the GPU does not have to recompute them anew on every request.

## Core Concepts & Terminology

* **Token & Token ID:** The basic unit of text processed by an LLM (a word, subword, or punctuation mark), mapped to an integer called a **token ID**.
* **Autoregressive Generation & Logits:** LLMs generate text one token at a time, feeding each new token back in as input for the next step. At the end of each step, the model outputs **logits**—raw scores for every possible next token in its vocabulary—to pick the next token.
* **Transformer Layer ($L$), Hidden State, Weights, & Model Dimension ($d_{\text{model}}$):** An LLM is a stack of $L$ repeating processing blocks called **layers**. Each token is represented inside the model as a list of numbers (a vector) of length $d_{\text{model}}$ called a **hidden state**, which is transformed by the model's learned numerical parameters (**weights**).
* **Self-Attention, Query ($Q$), Key ($K$), Value ($V$), & Causal Attention:** Inside each layer, **self-attention** lets each token gather context from earlier tokens using three projected vectors: **Query ($Q$)** (what the token is looking for), **Key ($K$)** (a label describing what the token contains), and **Value ($V$)** (the actual information passed along if matched). The dot-product similarity between queries and keys forms the **attention score matrix ($QK^\top$)**. **Causal attention** enforces that token $t$ can only look at tokens $0 \dots t$, never future tokens.
* **Linear Projections & MLP:** A **linear projection** multiplies token vectors by a weight matrix to produce $Q$, $K$, $V$, or the attention output ($O$). An **MLP (Multi-Layer Perceptron)** is the feed-forward neural network sub-block in each layer (with up, gate, and down projections) that processes each token independently.
* **KV Cache (Key-Value Cache):** Storing previously computed $K$ and $V$ vectors in GPU memory so they do not have to be recalculated when generating subsequent tokens or handling later turns.
* **VRAM / HBM (High Bandwidth Memory):** The ultra-fast memory chips mounted directly alongside the GPU processor (often referred to as **VRAM** or **HBM**).
* **GEMM (General Matrix Multiply) & Tensor Cores:** A **GEMM** is a dense 2D matrix-by-matrix multiplication—the core operation behind linear projections and MLPs. **Tensor Cores** are specialized hardware units on modern GPUs (such as the NVIDIA H100 SXM5) built to run GEMMs at massive speed.
* **FLOPs, TFLOPS, BF16, & MFU:** A **FLOP** is a single floating-point arithmetic operation (e.g., one multiply or add); **TFLOPS** means trillions of FLOPs per second. **BF16 (Brain Float 16)** is a 16-bit number format commonly used to store LLM weights and activations. **MFU (Model FLOPs Utilization)** is the percentage of a GPU's theoretical peak TFLOPS actually achieved by useful model math.
* **Compute-Bound vs. Memory-Bandwidth-Bound:** A workload is **compute-bound** when speed is limited by how fast Tensor Cores can do math, and **memory-bandwidth-bound** when speed is limited by how fast weights and KV cache bytes can be fetched from HBM.
* **Prefill, Decode, & TTFT (Time-to-First-Token):** **Prefill** is the initial phase that processes all input prompt tokens in parallel (compute-bound). **Decode** is the generation phase that produces one new token per step per request while reading all weights and cached KV data from HBM (memory-bandwidth-bound). **TTFT** is the delay from when a user sends a prompt until the first output token arrives.
* **LoRA (Low-Rank Adaptation) Adapter:** A small, swappable set of fine-tuning weights added on top of a base model to customize its behavior for a specific task or user.
* **Prefix Caching & APC (Automatic Prefix Caching):** Saving and reusing the KV cache of identical starting token sequences (**prefixes**) across requests so the engine only computes the new ending tokens (**delta suffix**).
* **PagedAttention, Block ($B$), `block_table`, Zero-Copy, & `ref_count`:** **PagedAttention** divides KV cache memory into fixed-size chunks called **blocks** (each holding $B$ tokens, e.g., $16$ or $32$) stored non-contiguously in VRAM, mapped by a per-request list called a **`block_table`**. Multiple requests sharing a prefix point directly to the same physical VRAM block without copying bytes (**zero-copy**), and a reference counter (**`ref_count`**) tracks how many active requests rely on that block so it is never overwritten while in use.
* **Hash Function, Chained (Merkle-Style) Hashing, XXH3, SHA-256, & SipHash:** A **hash function** maps variable-length data (such as a block of token IDs) to a compact fixed-size fingerprint. **Chained hashing** feeds block $i-1$'s hash into block $i$'s hash calculation so a block matches *only* if the entire preceding prefix matches. **XXH3** is an ultra-fast non-cryptographic hash accelerated by CPU vector instructions (**AVX2** on x86, **NEON** on ARM); **SHA-256** is a slower, cryptographically collision-resistant hash; **SipHash** is Python's built-in `hash()`.
* **Namespace Salt / Cache Salt (`extra_keys` / `cache_salt`):** Extra metadata mixed into a block's hash (e.g., LoRA adapter ID, multimodal image content hash, or a per-tenant secret `cache_salt`) to prevent different contexts or tenants from falsely sharing cached blocks or leaking timing information.
* **LRU (Least Recently Used) Eviction:** A memory-reclamation rule that removes the unpinned cached block that has gone the longest without being accessed.
* **Trie, Radix Tree (Patricia Trie), Path Compression, RadixAttention, & LPM:** A **trie** is a tree where paths from the root represent prefixes. A **Radix Tree** applies **path compression** by merging chains of single-child nodes into one edge labeled with a multi-token span. **RadixAttention** (used by SGLang) stores KV cache pages in a dynamic Radix Tree, while **LPM (Longest-Prefix-Match) Scheduling** prioritizes waiting requests that share the longest already-cached prefix so they run before that prefix is evicted.
* **FlashAttention / FlashInfer, SWA, SSM, & Cache Thrashing:** **FlashAttention** and **FlashInfer** are high-speed GPU kernels that compute attention without writing the huge $QK^\top$ matrix to HBM. **SWA (Sliding Window Attention)** restricts each token to attend only to the last $W$ tokens. **SSM (State Space Models, e.g., Mamba)** compress history into a fixed-size recurrent state matrix $S_t$ updated in-place rather than an append-only KV cache. **Cache thrashing** happens when cached data is repeatedly evicted and recomputed because requests arrive in an order that exceeds cache capacity.

---

## 1. The Physics of Redundant Prefill in Multi-Turn & Agentic Workloads

During autoregressive LLM serving, request execution splits into two phases with opposite hardware bottlenecks:

1. **Prefill:** Processes all $N_{\text{prompt}}$ input tokens in parallel. In PyTorch terms, all token embeddings are stacked into a single matrix of shape `[N_prompt, d_model]`, so linear projections ($Q, K, V, O$, and MLP up/gate/down) execute as large dense matrix multiplications (`[N_prompt, d_model] @ [d_model, d_out]`). Because the GPU performs many math operations for every weight byte loaded from memory, prefill is **compute-bound** on Tensor Cores; its latency scales roughly linearly with $N_{\text{prompt}}$ (and quadratically $O(N_{\text{prompt}}^2)$ in the attention score matrix $QK^\top$ at ultra-long contexts).
2. **Decode:** Generates 1 token per sequence per forward pass. With $B$ active requests, the input matrix has shape `[B, d_model]` (typically a thin matrix), yet the GPU must still stream all model weights and all cached KV blocks from HBM for every single step. Because data movement dominates arithmetic, decode is **memory-bandwidth-bound**.

In real production traffic, requests rarely start from an empty, unique context:

* **System Prompts & Tool Schemas:** Enterprise agents prepend $2\text{K}$–$16\text{K}$ tokens of instructions, JSON schemas, and safety policies to *every* API call.
* **Few-Shot / In-Context Learning:** Static exemplars ($1\text{K}$–$8\text{K}$ tokens) precede variable user queries.
* **Multi-Turn Conversations & Tool-Calling Loops:** A 10-turn agent trajectory sends $[P_0, U_1, A_1, T_1, U_2, A_2, \dots, U_{10}]$. Without prefix caching, Turn 10 recomputes the KV tensors of turns $1\dots 9$ all over again.
* **Tree-of-Thoughts / Best-of-$N$ / Beam Search:** Multiple candidate continuations branch from an identical root prompt.

Because Transformer attention is **causal** (each token $t$ is masked from seeing any token $> t$, so row $t$ of the attention output depends *only* on rows $0 \dots t$), the Key and Value tensors $K_l[0:P], V_l[0:P]$ at every layer $l \in [0, L-1]$ for a prefix of length $P$ are **pure, deterministic functions of the prefix tokens $[t_0, t_1, \dots, t_{P-1}]$** (plus model weights and active LoRA adapter ID). Changing or appending tokens after position $P-1$ cannot retroactively alter $K_l[0:P]$ or $V_l[0:P]$. Therefore, if those KV tensors are retained in memory and keyed by the exact prefix identity, any subsequent request sharing the first $P$ tokens skips prefill computation for $[0, P)$ entirely and computes only the **delta suffix** $[P, N_{\text{prompt}})$.

### Quantitative TTFT Collapse

Let $\text{FLOP}_{\text{prefill}}(N) \approx 2 \cdot P_{\text{active}} \cdot N + 4 \cdot L \cdot d_{\text{model}} \cdot N^2$ (for dense models with $P_{\text{active}}$ parameters, $L$ layers, and attention term when $N$ is large). On an NVIDIA H100 SXM5 ($989\text{ TFLOPS}$ BF16 peak, ~70% Model FLOPs Utilization during prefill $\approx 692\text{ TFLOPS}$ effective) running a 70B dense model:

* **No Prefix Caching ($N = 16{,}384$ tokens):**
  $$\text{Compute} \approx 2 \times 70 \times 10^9 \times 16{,}384 \approx 2.29 \times 10^{15}\text{ FLOPs} \implies \text{TTFT} \approx \frac{2.29 \times 10^{15}}{6.92 \times 10^{14}} \approx 3.31\text{ s}$$
* **With Prefix Caching ($P = 16{,}000$ cached, $\Delta = 384$ new tokens):**
  $$\text{Compute} \approx 2 \times 70 \times 10^9 \times 384 + \text{attention}(\Delta \times 16{,}384) \approx 5.4 \times 10^{13}\text{ FLOPs} \implies \text{TTFT} \approx 78\text{ ms}$$

Prefix reuse achieves a **$\sim 42\times$ reduction in TTFT** and frees $3.23\text{ s}$ of Tensor Core occupancy for other requests.

---

## 2. Paradigm A: Block-Hash Automatic Prefix Caching (vLLM APC)

vLLM builds prefix reuse directly atop **PagedAttention**, where KV memory is partitioned into fixed-size physical blocks of $B$ tokens (typically $B = 16$ or $B = 32$ tokens per block). Instead of allocating one giant contiguous `torch.empty(S, H_kv, d_head)` tensor per request—which fragments memory and prevents sharing—the engine pre-allocates one global KV pool tensor of shape `[num_blocks, B, H_kv, d_head]` per layer. Each sequence's logical KV cache is simply an ordered list of integer indices into that global pool (`block_table`, e.g., `[42, 87, 103]`). Because `block_table` is just a list of pointers, two requests can include the same physical block ID `42` in their `block_table` without copying a single byte of tensor data (**zero-copy sharing**).

### 2.1 Why Naive Block Hashing Fails (The False-Match Hazard)

Suppose block size $B = 16$ and two requests contain the same 16 tokens `"def solve(x): ..."` at block index 1 (tokens $16\dots 31$), but **different** tokens at block index 0 (tokens $0\dots 15$).

* **Request A:** `[System_Prompt_Math (0..15)] [def solve(x): ... (16..31)]`
* **Request B:** `[System_Prompt_SQL  (0..15)] [def solve(x): ... (16..31)]`

Because self-attention is contextual, the hidden states—and thus the Key/Value projections—at positions $16\dots 31$ depend heavily on positions $0\dots 15$. If block 1 were hashed solely from its local 16 token IDs `H(tokens[16:32])`, Request B would falsely hit Request A's block 1 and attend over corrupted KV states conditioned on `[System_Prompt_Math]`!

### 2.2 Chained (Merkle-Style) Block Hashing

To enforce **exact causal prefix equivalence**, vLLM APC computes each block's cache key as a **cryptographic or high-entropy chained hash** incorporating the hash of the preceding block:

$$\text{hash}_0 = H\big(\text{NONE},\ \text{tokens}[0:B],\ \text{extra\_keys}\big)$$
$$\text{hash}_i = H\big(\text{hash}_{i-1},\ \text{tokens}[iB : (i+1)B],\ \text{extra\_keys}\big)$$

Where `extra_keys` (the **namespace salt**) prevents cross-contamination across non-equivalent execution contexts:
* **LoRA Adapter ID / Hash:** Different LoRA weights produce different $K/V$ projections for identical tokens.
* **Multimodal Image/Audio Content Hashes:** Placeholder image tokens (e.g., `<image>`) share identical token IDs across different images; the perceptual/content digest of the multimodal embedding patch falling inside block $i$ must be hashed into `extra_keys`.
* **Cache Salt / Tenant Isolation:** Optional per-tenant cryptographic salt (`cache_salt`) to prevent cross-tenant timing side-channel attacks (where an attacker probes TTFT to infer whether another user queried a sensitive prefix).
* **Prompt Embedding / Soft Prompt Hash:** Any non-token input embeddings overlapping the block.

> [!IMPORTANT]
> **Full-Block Granularity Requirement:** In vLLM APC, a block can only be inserted into the prefix hash table (`cached_blocks: Dict[BlockHash, PhysicalBlock]`) once it is **completely filled** with $B$ tokens. A partially filled tail block of $r \in [1, B-1]$ tokens has variable length and continues mutating during decode; hashing partial blocks would either create unbounded unaligned fragments or mutate keys in-place. Thus, if a prompt is $100$ tokens with $B = 16$, blocks $0\dots 5$ ($96$ tokens) are cached and matched, while the remaining $4$ tokens sit in an uncached tail block and are recomputed on the next turn (until enough tokens accumulate to seal block $6$).

### 2.3 Hash Function Choice: SHA-256 vs. XXH3

Earlier vLLM versions used `hash()` (Python's SipHash, unstable across processes and slow for large blocks) or `SHA-256`. At 100K+ tokens/sec, hashing every 16-token block with SHA-256 creates measurable CPU scheduler overhead. Modern vLLM uses **XXH3 (64-bit or 128-bit)** or **SHA-256** (configurable via `--prefix-caching-hash-algo`):

* **XXH3-128:** Vectorized (AVX2/NEON), runs at $>30\text{ GB/s}$ per CPU core (~$5\text{ ns}$ per 16-token block), collision probability $\approx 2^{-128}$ (negligible).
* **SHA-256:** Cryptographically collision-resistant ($2^{-256}$) when untrusted tenants could theoretically craft hash-collision inputs to poison shared caches, though slower (~$100$–$300\text{ ns}$ per block).

### 2.4 Lookup & Allocation Mechanics in vLLM

1. **Sequential Prefix Walk:** When a request arrives with tokens $T[0:N]$, the scheduler slices $T$ into $\lfloor N/B \rfloor$ full blocks and computes `hash_0, hash_1, ...` sequentially in Python/C++ before launching any GPU kernel.
2. **Contiguous Prefix Hit Rule:** Lookup checks `cached_blocks` from $i = 0$ upward and stops at the **first miss** at block index $k$. Because each hash chains from the previous block (`hash_{i} = H(hash_{i-1}, ...)`), a miss at block $k$ means every subsequent chained hash `hash_{k+1}` will also differ from any unrelated prefix, and leaf-first eviction guarantees that an earlier prefix `[0..k]` is never evicted while `[0..k+1]` remains cached. Thus, a single sequential walk from $i = 0$ safely identifies the exact longest cached prefix `[0 : kB)` and marks only `[kB : N)` for GPU prefill computation.
3. **Copy-on-Write / Refcount Pinning:** Hit blocks $[0 \dots k-1]$ have their `ref_count` incremented (analogous to Python's reference counter keeping a shared `torch.Tensor` storage alive) and are appended directly to the request's `block_table`. Their physical VRAM frames are shared zero-copy across all concurrent requests using that prefix, and the memory allocator cannot overwrite them while `ref_count > 0`.
4. **Edge Case — Last Token Logit Requirement:** If **100%** of the prompt hits in cache ($N = kB$, all blocks cached), skipping the forward pass entirely would leave the engine with no `logits[:, -1, :]` tensor for token $N-1$ from which to sample the *first* generated token! Therefore, vLLM (and SGLang) intentionally recomputes at least the **final 1 token** (or final partial block) whenever the entire prompt is an exact cached match, ensuring the model executes one forward pass to emit `logits[N-1]`.

---

## 3. Paradigm B: Dynamic Prefix Trees (SGLang RadixAttention)

While vLLM treats the KV cache as an unordered hash map of fixed-size $B$-token blocks, **SGLang's RadixAttention** maintains an explicit **compressed prefix trie (Radix Tree)** directly over token sequences.

### 3.1 Compressed Trie Structure (Path Compression)

In a standard trie, each edge represents 1 token, creating $N$ pointer-chasing hops for a prompt of length $N$. A **Radix Tree (Patricia Trie)** compresses any unbranched chain of single-child nodes into a single edge labeled with a variable-length **token span** $T[a:b]$ and its corresponding array of KV cache slot/page indices:

* **Node (`TreeNode`):**
  * `children: Dict[TokenID, TreeNode]` (keyed by the *first* token of the child's edge span, or a sub-block tuple)
  * `parent: TreeNode`
  * `edge_tokens: List[int]` (the sequence of tokens on the incoming edge from `parent`)
  * `kv_indices: Tensor[int32]` (physical KV pool slot/page IDs corresponding to `edge_tokens`)
  * `lock_ref: int` (reference count of currently executing requests pinning this subtree)
  * `last_access_time: float` (monotonic timestamp for LRU eviction)

### 3.2 Dynamic Operations: Match, Split, Insert, and Evict

Unlike fixed-block APC, a RadixTree operates at **token-level (or page-aligned sub-span) granularity**, eliminating internal fragmentation on partial tail blocks:

1. **Longest-Prefix Match (`match_prefix(tokens)`):**
   Starting at `root`, look up `children[tokens[pos]]`. Compare `edge_tokens` against `tokens[pos:]` to find the longest common prefix length $m$:
   * If $m == \text{len}(\text{edge\_tokens})$: consume the entire edge, collect `node.kv_indices`, advance `pos += m`, and descend into `node`.
   * If $0 < m < \text{len}(\text{edge\_tokens})$: **Edge Split!** A new request diverges mid-edge (e.g., two agents share 513 tokens of a 600-token edge and diverge at token 514). The tree splits `node` into:
     * `new_mid_node` with `edge_tokens[:m]` and `kv_indices[:m]`
     * `node` updated to `edge_tokens[m:]` and `kv_indices[m:]` as a child of `new_mid_node`.
     The request hits all $m$ tokens without waiting for $B$-token block alignment!
2. **Insertion (`insert(tokens, kv_indices)`):**
   After a request completes prefill/decode for the delta suffix, the newly computed `(tokens[pos:], kv_indices[pos:])` are attached as a new leaf node under the last matched/split node.
3. **Tree-Structured LRU Eviction:**
   Because a RadixTree explicitly encodes parent-child dependencies, eviction **only targets unlocked leaf nodes (`len(children) == 0` and `lock_ref == 0`)** ordered by oldest `last_access_time`. Evicting a leaf cleanly reclaims its `kv_indices` back to the free memory pool, and if its parent now has 0 children and `lock_ref == 0`, the parent becomes an evictable leaf. This structurally guarantees that a parent prefix is never evicted before its descendants.

### 3.3 Cache-Aware Request Scheduling (Longest-Prefix-First)

In a flat FIFO queue, if Request 1 (prefix `A`) and Request 3 (prefix `A`) are separated by Request 2 (prefix `B`), executing `[1, 2, 3]` may force `A` to be evicted during `2` and recomputed for `3`.

Because SGLang maintains the global RadixTree in the scheduler, it implements **Cache-Aware Scheduling (Longest-Shared-Prefix / LPM Scheduling)**:
1. For every waiting request in the queue, compute `matched_len = radix_tree.match_prefix(req.tokens)` (read-only traversal without modifying the tree).
2. Sort or prioritize waiting requests by descending `matched_len` (subject to starvation-prevention aging bounds so short/uncached requests are not starved indefinitely).
3. **Result:** Requests sharing common prefixes (e.g., batch evaluation prompts, multi-agent trees, system-prompt cohorts) naturally coalesce into the same execution wave *while their shared prefix is hot in VRAM*, maximizing cache hit rate and minimizing thrashing.

---

## 4. Architectural Comparison: vLLM APC vs. SGLang RadixTree

| Dimension | vLLM Automatic Prefix Caching (APC) | SGLang RadixAttention (RadixTree) |
| :--- | :--- | :--- |
| **Index Structure** | Flat hash table (`Dict[ChainedBlockHash, PhysicalBlock]`) | Compressed prefix trie (`RadixTree` of variable-length spans) |
| **Granularity** | Fixed block size $B$ (e.g., 16 or 32 tokens); partial tail block uncached | Token-level (or sub-block page) granularity via dynamic edge splitting |
| **Prefix Identity Guarantee** | Chained Merkle-style hash `H(prev_hash, tokens, salt)` | Explicit tree path from `root` to `node` |
| **Memory Fragmentation** | Up to $B - 1$ uncached prefix tokens per divergence point | Zero prefix-boundary waste (down to 1 token, or FlashInfer page size) |
| **Lookup Complexity** | $O(\lceil N / B \rceil)$ hash computations + $O(1)$ hash lookups | $O(\text{tree depth} + \text{matched tokens})$ span comparisons |
| **Eviction Ordering** | Refcount = 0 + LRU heap (must track leaf-first or block timestamps) | Structural leaf-only LRU (`children == 0 && lock_ref == 0`) |
| **Scheduler Synergy** | Typically FCFS / priority; hash lookup happens at block allocation | Native Longest-Prefix-Match (LPM) scheduling across waiting queue |
| **Best Workload Fit** | Fixed/aligned system prompts, large multi-turn blocks, simple PagedAttention integration | Tree-of-Thoughts, branching agents, structured SGLang programs, arbitrary divergence points |

### Structural Comparison Diagram

```mermaid
flowchart TD
    subgraph VLLM["vLLM APC: Chained Block-Hash Table (Block Size B = 16)"]
        direction TB
        VT0["Tokens 0..15 + Salt"] --> VH0["H0 = XXH3(NONE, T0..15, Salt)"]
        VH0 --> VT1["Tokens 16..31"] --> VH1["H1 = XXH3(H0, T16..31, Salt)"]
        VH1 --> VT2["Tokens 32..37 (Partial: 6 tokens)"] --> VSkip["Uncached Tail (Recomputed)"]
        VH0 --> VMap["Hash Table: H0 -> PhysBlock 42"]
        VH1 --> VMap2["Hash Table: H1 -> PhysBlock 87"]
    end

    subgraph SGLANG["SGLang RadixAttention: Compressed Prefix Trie"]
        direction TB
        Root["Root Node (ref=0)"] -->|"Edge: System Prompt (0..500), KV Slots 0..500"| Mid["Split Node (ref=2, LRU=t2)"]
        Mid -->|"Edge: Agent Tool A (501..612), KV Slots 501..612"| LeafA["Leaf A (ref=0, Evictable)"]
        Mid -->|"Edge: Agent Tool B (501..589), KV Slots 700..788"| LeafB["Leaf B (ref=1, Pinned)"]
    end
```

---

## 5. End-to-End Execution Timeline: Multi-Turn Agent Prefix Hit

Consider a multi-turn coding agent on Turn 4:
* **Turn 1–3 Cached Prefix:** $12{,}000$ tokens (`System + Tools + Repo Context + Turns 1-3`).
* **Turn 4 Delta Suffix:** $256$ tokens (`Tool Output + User Follow-up`).
* Assume block size $B = 16$: $12{,}000 / 16 = 750$ full blocks hit in VRAM; $256 / 16 = 16$ new blocks allocated for the delta suffix.

```mermaid
sequenceDiagram
    participant Client as "Agent Client"
    participant Sched as "Engine Scheduler"
    participant Index as "Prefix Index (APC / RadixTree)"
    participant Alloc as "KV Block Allocator"
    participant GPU as "GPU Compute (Prefill / Attn)"

    Client->>Sched: Submit Turn 4 (12,256 tokens)
    Sched->>Index: Match longest cached prefix (hashes 0..765 or trie walk)
    Index-->>Sched: Hit 750 blocks (12,000 tokens), Miss 256 suffix tokens
    Sched->>Alloc: Pin 750 hit blocks (ref_count++) and allocate 16 new blocks
    Alloc-->>Sched: Block Table [B0..B749 (shared), B750..B765 (new)]
    Note over Sched,GPU: Compute ONLY delta suffix (Q length = 256, KV length = 12,256)
    Sched->>GPU: Launch Prefill Kernel (q_tokens=256, block_table=B0..B765)
    GPU->>GPU: Project Q,K,V for tokens 12,000..12,255 only
    GPU->>GPU: Write new K,V into B750..B765
    GPU->>GPU: Causal Attention Q[12000..12255] x KV[0..12255] via Paged/FlashAttn
    GPU-->>Sched: Sample Token 12,256 (TTFT ~65ms instead of ~2.4s)
    Sched->>Index: Insert newly sealed blocks B750..B765 into prefix index
    Sched-->>Client: Stream Token 12,256 and begin Decode Loop
```

### Critical Edge Cases & Failure Modes

1. **Non-Deterministic Numeric Divergence Across Hit vs. Recompute:**
   Floating-point addition is not strictly associative. Computing KV tensors for tokens `[0..12000]` in a single batch vs. reusing cached KV tensors computed in an earlier batch of different size (or different Tensor Core tile split) can produce $10^{-3}$ BF16 rounding differences, occasionally flipping greedy/near-tie token choices. Systems requiring bitwise reproducibility must keep batch-invariant kernels or disable cross-batch prefix reuse.
2. **Sliding Window Attention (SWA) & Hybrid State (Mamba/SSM/Linear Attention):**
   * For **SWA** (e.g., Mistral), tokens beyond window $W$ do not attend to older tokens, *however* layer $l$ still depends on layer $l-1$ across $L \cdot W$ effective receptive field; chained hashing still holds, though evicted older blocks outside the window may not be needed if a sequence only decodes forward.
   * For **Hybrid SSM/Transformer models** (e.g., Jamba, Qwen3-Next/Linear Attention), the recurrent SSM state $S_t$ is a fixed-size matrix overwritten in-place at each token rather than an append-only KV sequence. Unless periodic **state checkpoints** are snapshotted at block boundaries, prefix reuse cannot roll back an in-place mutated SSM state to an earlier branch point!
3. **Cache Pollution by Single-Use Long Prompts:**
   Inserting every 100K-token one-off document immediately evicts heavily reused 4K system prompts unless governed by **admission policies and cost-aware eviction** (covered in [05 — Management Policies & Eviction](./05-management-policies-eviction.md)).

---

## Navigation

* [← Stage 5 Overview](./README.md)
* **Next:** [02 — Hybrid Scheduling: Chunked Prefill →](./02-hybrid-scheduling-chunked-prefill.md)
