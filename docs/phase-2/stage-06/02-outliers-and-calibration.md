# 02 — Outliers & Calibration

[← 01 — Numerical Representation](./01-numerical-representation.md) · [Stage 6 Index](./README.md) · **Next:** [03 — Hardware Execution (W4A16 vs. W8A8) →](./03-hardware-execution-w4a16-vs-w8a8.md)

---

Simply rounding a large language model's numbers to 8 bits or 4 bits often ruins its accuracy because a tiny fraction of internal signals spike far above the rest. Calibration algorithms solve this offline using a small sample of text to rescale or adjust weights before deployment, preserving accuracy without slowing down GPU execution.

## Core Concepts & Terminology

Every technical term, numerical format, hardware concept, and calibration algorithm used in this document is defined below in dependency order:

1. **Transformer & Linear Layer Foundations:**
   * **Weights ($W$), Activations ($X$), & Hidden States:** Weights are the learned parameters of a linear layer $Y = XW$ ($W \in \mathbb{R}^{K \times N}$); activations (or hidden states) are the intermediate tensors ($X \in \mathbb{R}^{M \times K}$) flowing between layers for $M$ tokens across $K$ input channels and $N$ output neurons.
   * **Residual Stream, `LayerNorm`, & `RMSNorm`:** The residual stream is the running vector sum passed across Transformer blocks; `LayerNorm` and `RMSNorm` are normalization layers that rescale activations to stabilize magnitudes.
   * **`QKV`, `Out_proj` (`O_proj`), `V_proj`, `Gate/Up`, `Down_proj`, `SwiGLU`, & `SiLU`:** Standard projections inside each Transformer block. `QKV` creates Query, Key, and Value vectors; `V_proj` and `O_proj` project Value and attention outputs; `Gate/Up` and `Down_proj` form the feed-forward MLP using the `SwiGLU` gated activation (`SiLU` nonlinearity).
   * **Perplexity (PPL) & Out-of-Distribution (OOD) Domains:** Perplexity measures language modeling uncertainty (lower is better). OOD domains are text types (such as code, math, or non-English text) not heavily represented in the calibration sample.

2. **Quantization & Outlier Concepts:**
   * **`RTN` (Round-to-Nearest):** Naive quantization that rounds every number to its nearest representable grid value without calibration adjustments.
   * **`FP16`, `BF16`, `INT8`, `INT4`, `UINT4`, `FP8`, & `MXFP8`:** 16-bit (`FP16`, `BF16`), 8-bit (`INT8`, `FP8`, `MXFP8` with 32-element block scales), and 4-bit (`INT4` signed, `UINT4` unsigned) formats.
   * **`W8A8`, `W4A16`, `W3A16`, & `W2A16`:** Bit-width shorthand where `W` is weight bits and `A` is activation bits (e.g., `W4A16` = 4-bit weights, 16-bit activations).
   * **Scale ($s$, $\Delta$), Per-Token, Per-Channel, & Group-Wise ($g=128$) Scaling:** Step sizes shared per token row, per output column, or per $g$-element contiguous slice along $K$.
   * **Activation Outliers & `LLM.int8()`:** Activation outliers are fixed feature channels ($\le 0.1\%$ of $K$) whose magnitudes reach $20\times$–$100\times$ normal values across every token. `LLM.int8()` isolated those outlier columns in `FP16` while keeping normal columns in `INT8`.
   * **Attention Sink:** Early/anchor tokens that accumulate disproportionately large attention or activation magnitudes.
   * **Calibration Dataset (Pile, C4, WikiText-2):** A small set of 128–512 representative text sequences run through the model forward pass to gather activation statistics without backpropagation.

3. **GPU Hardware & Kernel Execution:**
   * **GEMM, Reduction Dimension ($K$), & BLAS `GEMM`:** General Matrix Multiply ($Y = XW$) summing along the shared inner dimension $K$; BLAS `GEMM` is a standard high-performance dense matrix-multiply routine.
   * **Tensor Core (`IMMA`), GPU Warps, & FLOPs:** `IMMA` is the integer Tensor Core matrix-multiply unit; a warp is 32 GPU threads executing in lockstep; FLOPs count floating-point arithmetic operations.
   * **HBM Bandwidth, Coalesced Reads, & Kernel Fusion / Epilogue:** HBM is off-chip GPU memory. Coalesced reads occur when adjacent threads read contiguous memory addresses (scattered gathers stall HBM). Kernel fusion merges elementwise math into the end (epilogue) of a preceding GPU kernel to avoid extra memory passes.
   * **Marlin Kernel & `g_idx`:** A fast fused `W4A16` GPU kernel; `g_idx` is a group-index lookup table used when weight columns are reordered.

4. **Calibration Algorithms & Mathematical Tools:**
   * **`SmoothQuant` ($\alpha$, $\text{diag}(s)$):** A `W8A8` algorithm that migrates quantization difficulty from activations to weights via $Y = (X\text{diag}(s)^{-1})(\text{diag}(s)W)$, folding $s^{-1}$ into the preceding normalization layer.
   * **`AWQ` (Activation-Aware Weight Quantization) & Frobenius Norm ($\|\cdot\|_F$):** A `W4A16` algorithm that protects the top 0.1%–1% salient weight channels (selected by mean activation magnitude $\mathbf{s}_X$) by scaling them up before quantization, using a 20-point grid search on $\alpha$ to minimize the Frobenius norm (matrix $\ell_2$ reconstruction error).
   * **`GPTQ` & `OBQ` (Optimal Brain Quantization):** Second-order quantization methods that actively adjust remaining unquantized weights to compensate for the rounding error of already-quantized weights.
   * **Hessian ($H = 2XX^\top$), Inverse Hessian ($H^{-1}$), & Lagrange Multipliers:** $H$ is the second-order curvature matrix of layer-wise squared output error; $H^{-1}$ encodes cross-channel correlations used to solve constrained quadratic error minimization (via Lagrange multipliers) for optimal weight updates.
   * **Cholesky Decomposition & Dampening ($\lambda I$, $\text{Tr}(H)$):** Cholesky decomposition factors a positive-definite matrix into triangular factors ($L L^\top$) for numerical stability; dampening adds $\lambda I$ (proportional to the trace $\text{Tr}(H)$, the sum of diagonal entries) so eigenvalues stay positive.
   * **`actorder` (`desc_act`):** A GPTQ option that quantizes columns in descending order of $\text{diag}(H)$ (highest activation energy first) so remaining columns can compensate them.

---

## 1. The Activation Outliers Phenomenon in Large Transformers

If Transformer weights and activations were clean, isotropic Gaussian tensors, naive round-to-nearest (`RTN`) quantization to `INT8` or `FP8` would preserve accuracy effortlessly. In reality, once autoregressive Transformers scale beyond **~6.7 billion parameters**, a sharp phase transition occurs in their internal hidden states—first characterized systematically by Dettmers et al. (`LLM.int8()`, 2022).

### 1.1 Anatomy of Emergent Feature Outliers

1. **Extreme Magnitude Disparity ($20\times\text{--}100\times$):** While the vast majority (>99.9%) of activation values in $X \in \mathbb{R}^{M \times K}$ fall within $[-3.0, +3.0]$, a tiny subset of **fixed hidden-dimension channels** (typically $\le 0.1\%$ of $K$, e.g., 4 to 20 specific indices out of $K = 4096$ or $8192$) consistently exhibit magnitudes of **$20$ to $100+$**.
2. **Channel-Systematic, Not Token-Random:** These outliers are **not** isolated spikes on single tokens. Once a model crosses the ~6.7B emergence threshold, the **exact same channel indices $j \in \mathcal{O}$** fire with high magnitude across **every token in the sequence** and across **most Transformer layers** (driven by residual stream accumulation through `LayerNorm`/`RMSNorm` and attention sink / variance-encoding mechanisms).
3. **Asymmetry Between Weights and Activations:**
   * **Weights $W$** are regular, smooth, and well-behaved across channels (rarely exceeding $\pm 0.5$ to $\pm 2.0$), making `INT8` or `FP8` weight quantization straightforward.
   * **Activations $X$** exhibit severe **inter-channel variance** (up to $100\times$ across columns $k$), while remaining relatively uniform within any given channel $k$ across tokens $m$.

### 1.2 Why Outliers Break Standard Activation Quantization

Consider **per-token dynamic `INT8` quantization** on a single token row $x_m \in \mathbb{R}^{K}$ where channel $j^* = 1024$ has value $x_{m, 1024} = 64.0$ and all other 4095 channels lie in $[-1.0, +1.0]$:

$$s_x[m] = \frac{\max_k |x_{m, k}|}{127} = \frac{64.0}{127} \approx 0.504$$

Every normal channel with $|x_{m, k}| < 0.252$ rounds identically to **`0`**, and the remaining normal values in $[-1.0, +1.0]$ are compressed into just **5 integer levels** (`{-2, -1, 0, +1, +2}`)—an effective precision of **~2.3 bits**!

```text
Channel Index k:      0       1       2   ...   1024 (Outlier)  ...  4095
True Activation x: [ 0.21,  -0.48,   0.11, ... ,   64.00,       ..., -0.35 ]
Per-Token Scale s:  64.0 / 127 = 0.5039
Quantized INT8 q:  [   0,     -1,      0,  ... ,    127,        ...,   -1  ]
Dequantized x_hat: [ 0.00,  -0.50,   0.00, ... ,   64.00,       ..., -0.50 ] -> Normal features obliterated!
```

* **Why per-channel activation scaling ($s_x[k]$) cannot be used directly in GEMM:** In PyTorch notation, per-token scaling computes `X.abs().max(dim=-1, keepdim=True)` (shape `[M, 1]`), which factors cleanly outside `X @ W`. By contrast, scaling along the feature dimension $k$ (`X.abs().max(dim=0)`, shape `[1, K]`) sits along the **reduction dimension** of $Y_{mn} = \sum_{k=1}^K X_{mk} W_{kn}$. A scale $s_x[k]$ depending on $k$ cannot be factored outside the Tensor Core `IMMA`/`FP8` dot product without dequantizing inside the inner loop (unless using fine-grained hardware block scaling like `MXFP8` at $k=32$).
* **Why `LLM.int8()` mixed-precision decomposition is too slow:** Splitting $X$ and $W$ into `FP16` outlier columns ($\mathcal{O}$) and `INT8` normal columns ($[K] \setminus \mathcal{O}$) requires gathering non-contiguous columns from memory, running two separate GEMMs, and adding their results—overhead that makes `LLM.int8()` **slower than pure `FP16`** on modern GPUs.

To keep **pure, dense, fast integer/FP8/INT4 GEMMs**, calibration algorithms transform the weights and scales offline using a small calibration dataset (typically 128–512 sequences from Pile/C4/WikiText-2) so that standard, contiguous matrix multiplications preserve accuracy.

---

## 2. SmoothQuant: Migrating Difficulty from Activations to Weights

Introduced by Xiao et al. (2023), **SmoothQuant** solves the `W8A8` activation outlier bottleneck with **zero runtime mixed-precision overhead** by observing that activations are hard to quantize across channels while weights are easy to quantize across those same channels.

### 2.1 Mathematical Equivalence via Diagonal Scaling Migration

For any linear projection $Y = X W$ ($X \in \mathbb{R}^{M \times K}, W \in \mathbb{R}^{K \times N}$), we can insert an identity matrix $I = \text{diag}(s)^{-1} \text{diag}(s)$ for any positive per-channel smoothing vector $s \in \mathbb{R}_{>0}^{K}$ without changing the exact mathematical output:

$$Y = X W = \left( X \, \text{diag}(s)^{-1} \right) \cdot \left( \text{diag}(s) \, W \right) = \widehat{X} \, \widehat{W}$$

In PyTorch terms, this divides column `j` of `X` by `s[j]` (`X / s[None, :]`) and multiplies row `j` of `W` by `s[j]` (`W * s[:, None]`). Dividing column $j$ of $X$ by $s_j$ shrinks the outlier activation channel $j$ down into the normal range, while multiplying row $j$ of $W$ by $s_j$ increases the magnitude of the corresponding input-channel weight row—**migrating the quantization difficulty from $X$ into $W$** before either tensor is rounded.

### 2.2 Balancing Factor $\alpha$

How large should $s_j$ be?
* If $s_j$ is too small, $\widehat{X}_{:, j}$ still has outliers.
* If $s_j$ is too large, $\widehat{W}_{j, :}$ develops weight outliers that degrade per-tensor/per-channel weight quantization.

SmoothQuant defines the optimal migration strength using a hyperparameter $\alpha \in [0, 1]$ calibrated from per-channel activation maxima over a calibration set:

$$s_j = \frac{\left( \max(|X_{:, j}|) \right)^{\alpha}}{\left( \max(|W_{j, :}|) \right)^{1 - \alpha}}$$

* **$\alpha = 0.5$** (default for OPT, BLOOM, LLaMA-1/2): Splits dynamic range difficulty evenly so $\max(|\widehat{X}_{:, j}|)$ and $\max(|\widehat{W}_{j, :}|)$ have balanced scale.
* **$\alpha \in [0.75, 0.85]$** (e.g., GLM-130B or models with extreme activation spikes): Shifts more dynamic range into weights, which tolerate higher variance better when quantized per-channel.

### 2.3 Offline Weight Folding & Runtime Fusion

Where does $\text{diag}(s)^{-1}$ execute during inference?
1. **Attention `QKV` and MLP `Gate/Up` projections:** These layers are immediately preceded by `LayerNorm` or `RMSNorm`:
   $$\text{Norm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta$$
   We fuse $\text{diag}(s)^{-1}$ directly into the normalization affine parameters offline:
   $$\gamma' = \gamma \oslash s, \qquad \beta' = \beta \oslash s$$
   Result: **literally zero extra FLOPs or memory reads at runtime!**
2. **Attention `Out_proj` and MLP `Down_proj`:** No normalization layer sits between `V` $\to$ `O_proj` or `SwiGLU` $\to$ `Down_proj`. Here, either:
   * $s^{-1}$ is fused into the preceding `V_proj` / `Up_proj` output weights (for linear paths like `V` $\to$ `O_proj` across heads), OR
   * For non-linear activations (`SwiGLU`), a lightweight elementwise multiplication by $s^{-1}$ is fused into the SiLU/quantization epilogue kernel before `Down_proj`.

### 2.4 Failure Modes of SmoothQuant

* **Sub-4-bit collapse:** Migrating an $80\times$ activation outlier into weights forces weight rows up by $\sim 9\times\text{--}30\times$. In `INT8` (255 levels), weights absorb this easily; in `INT4` (15 levels), scaling up one row ruins precision for the rest of the tensor unless group-wise scaling is used.
* **Calibration-set peak sensitivity:** Because $s_j$ uses $\max(|X_{:, j}|)$, a misrepresentative calibration set (e.g., short dummy prompts missing code or multilingual tokens) underestimates $s_j$ for certain channels, causing runtime clipping.

---

## 3. AWQ: Activation-Aware Weight Quantization

Whereas SmoothQuant targets `W8A8` (quantizing both weights and activations), **AWQ** (Lin et al., 2023) targets **`W4A16`** (4-bit weights, 16-bit activations).

### 3.1 Core Insight: Weight Salience Lives in the Activations

In `W4A16`, activations stay in `FP16`/`BF16`, so activation outliers do **not** suffer quantization clipping. Yet uniform `RTN` `INT4` quantization of weights still degrades perplexity heavily. Why?

Consider one term of the dot product $y = \sum_{j=1}^K x_j w_j$. A weight rounding error $\Delta w_j = \hat{w}_j - w_j$ is amplified linearly by the **input activation magnitude** $|x_j|$:

$$|\hat{y} - y| = \left| \sum_{j=1}^K x_j (\hat{w}_j - w_j) \right|$$

If channel $j^*$ has $|x_{j^*}| = 40.0$ while normal channels have $|x_j| = 0.4$, a small rounding error in $w_{j^*}$ causes **$100\times$ larger output perturbation** than the same error in $w_j$! Empirically, protecting just the **top 0.1%–1% salient weight channels** (selected by average activation magnitude $\frac{1}{M}\sum_m |X_{m, j}|$, *not* weight magnitude $|W_{j,:}|$) recovers nearly full `FP16` accuracy.

### 3.2 Protecting Salient Weights Without Mixed Precision

Keeping 1% of weight columns in `FP16` and 99% in `INT4` creates irregular sparse-dense hardware layouts that stall GPU warps. AWQ eliminates mixed precision using **equivalent per-channel scaling** $s_j \ge 1$ on weights, derived via error analysis:

Let a weight group $w$ have group scale $\Delta = \frac{\max(|w|)}{2^{b-1}-1}$ (symmetric) or range/15 (asymmetric), and quantization error $\text{Err}(Q(w)) = \Delta \cdot \text{RoundErr}(w/\Delta)$, where $|\text{RoundErr}(\cdot)| \le 0.5$. Now scale salient channel $j$ up by $s_j > 1$ before quantizing, and divide activation $x_j$ by $s_j$:

$$Q(w_j \cdot s_j) \cdot \frac{x_j}{s_j} = \Delta' \cdot \left( \frac{w_j s_j}{\Delta'} + \text{RoundErr}\!\left(\frac{w_j s_j}{\Delta'}\right) \right) \cdot \frac{x_j}{s_j}$$

Assuming $w_j s_j$ does not become the dominant maximum of the entire group (or increases $\Delta' \approx \Delta$ moderately while reducing the relative step on channel $j$):

$$\text{Output Error on Channel } j = \left( \frac{\Delta'}{\Delta} \cdot \frac{1}{s_j} \right) \cdot \Delta \cdot \text{RoundErr}(\cdot) \cdot x_j$$

As long as $\Delta' / \Delta < s_j$, scaling up salient channel $j$ shrinks its output reconstruction error by up to **$1/s_j$**! However, making $s_j$ arbitrarily large inflates the group scale $\Delta'$ and harms all *non-salient* channels in the same group.

### 3.3 Grid-Search Calibration Objective

AWQ solves this trade-off cleanly without backpropagation or Hessian inversion. For each layer/block:
1. Compute per-channel mean activation magnitude vector $\mathbf{s}_X \in \mathbb{R}^K$, where $s_{X, j} = \frac{1}{M}\sum_{m=1}^{M} |X_{m, j}|$.
2. Parameterize candidate scale vector as $\mathbf{s} = \mathbf{s}_X^{\alpha}$ with $\alpha \in [0, 1]$ searched over a fast 20-point grid ($\alpha \in \{0.00, 0.05, \dots, 1.00\}$):
   $$\alpha^* = \arg\min_{\alpha \in [0, 1]} \left\| Q\!\left(W \cdot \text{diag}(\mathbf{s}_X^\alpha)\right) \left(\text{diag}(\mathbf{s}_X^\alpha)^{-1} X\right) - W X \right\|_F$$
3. Fold $\text{diag}(\mathbf{s}^*)^{-1}$ into the preceding `LayerNorm`/`RMSNorm` or preceding linear layer (or store as an `FP16` pre-scale vector if adjacent to a non-linear activation).

**Why AWQ Generalizes Exceptionally Well:** Because AWQ relies only on the **first-order average magnitude** $\mathbf{s}_X$ and searches a single scalar $\alpha$ per layer, it cannot overfit weights to the calibration text—making AWQ remarkably robust on out-of-distribution domains (code, multilingual, reasoning) even with just 128 calibration sequences.

---

## 4. GPTQ: Second-Order Optimal Brain Quantization

While AWQ optimizes a diagonal channel scale before rounding, **GPTQ** (Frantar et al., 2023) modifies the **quantized integer values themselves**, using second-order curvature to compensate quantization error in one weight by adjusting *subsequent unquantized weights* in the same row.

### 4.1 Layer-Wise Reconstruction & Hessian Derivation

Consider a single linear layer $W \in \mathbb{R}^{K \times N}$ (in row-major notation per output neuron $w \in \mathbb{R}^K$, we quantize each output row independently; here $X \in \mathbb{R}^{K \times M}$ denotes stacked calibration inputs). Rather than minimizing weight error $\|W - \widehat{W}\|_F^2$ in isolation, GPTQ minimizes the **actual layer output error** on the calibration inputs:

$$\min_{\widehat{W}} \| W X - \widehat{W} X \|_2^2 = \sum_{n=1}^{N} \| w_n^\top X - \hat{w}_n^\top X \|_2^2$$

Taking a second-order Taylor expansion of the loss $L(w) = \| w^\top X - \hat{w}^\top X \|_2^2$ around the unquantized optimal $w$ (where gradient $\nabla L(w) = 0$ because $\delta w = 0$ is already a minimum):

$$\delta L \approx \frac{1}{2} \delta w^\top H \, \delta w, \qquad \text{where } H = 2 X X^\top \in \mathbb{R}^{K \times K}$$

Here $H$ is the Hessian (curvature) matrix of the squared reconstruction loss—essentially the unnormalized input feature covariance matrix. Crucially, **$H = 2 X X^\top$ depends only on the layer inputs $X$ and is shared identically across all $N$ output channels** of that linear layer, so it only needs to be inverted once per layer rather than $N$ times!

### 4.2 Optimal Brain Quantization (OBQ) Update Formula

Suppose we quantize weight index $q$ from $w_q$ to $\text{quant}(w_q)$, introducing constraint $\delta w_q = \text{quant}(w_q) - w_q$. Solving the constrained quadratic minimization $\min_{\delta w} \frac{1}{2}\delta w^\top H \delta w \text{ s.t. } e_q^\top \delta w = \text{quant}(w_q) - w_q$ via Lagrange multipliers yields the exact minimum loss increase and optimal compensating update $\delta_F$ for all remaining unquantized weights:

$$\Delta L_q = \frac{\left(\text{quant}(w_q) - w_q\right)^2}{2 [H^{-1}]_{qq}}, \qquad \delta_F = - \frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}} \cdot H^{-1}_{:, q}$$

In words: if quantizing $w_q$ rounds it *up* by $+0.02$, GPTQ adjusts the remaining unquantized weights $w_{j > q}$ along column $q$ of $H^{-1}$ so that correlated input features cancel out the $+0.02 \cdot x_q$ perturbation in the final output $Y$!

### 4.3 Three Breakthroughs That Scaled OBQ to 175B Models

Classic OBQ took hours for a small ResNet because it greedily picked a *different* optimal weight order $q$ for every single row $n$, forcing $O(N \cdot K^3)$ separate Hessian inversions ($O(d_{\text{row}} \cdot d_{\text{col}}^3)$). GPTQ makes this $O(K^3 + N K^2)$ (minutes/hours for 70B–175B models) through three insights:

1. **Arbitrary Fixed Column Order:** For large $K$ (4096+), greedy per-row ordering yields negligible accuracy gain over quantizing columns in a **single shared order** across all $N$ rows simultaneously. Because all rows share the exact same $H^{-1}$ sequence, updates vectorize across all $N$ rows at once!
2. **Lazy Batch-Updates ($B = 128$):** Updating the entire remaining matrix $W_{:, q+1:}$ on every single column $q$ is severely HBM-bandwidth-bound. GPTQ partitions columns into blocks of $B = 128$:
   * Inside block $[i, i+B)$, apply error compensation immediately to columns within the block using the Cholesky factor of $H^{-1}$.
   * Accumulate the normalized errors $E = (W_{\text{block}} - Q_{\text{block}}) / \text{diag}(H^{-1}_{\text{block}})$ and update **all remaining columns** $W_{:, i+B:}$ with a **single dense matrix multiply (BLAS `GEMM`)** at the end of the 128-column block:
     $$W_{:, i+B:} \leftarrow W_{:, i+B:} - E \cdot H^{-1}_{i:i+B, \, i+B:}$$
3. **Cholesky Decomposition with Dampening:** Repeated rank-1 Gaussian elimination of $H^{-1}$ accumulates floating-point rounding drift that can flip eigenvalues negative and explode weight updates. GPTQ stabilizes $H$ by adding **dampening** proportional to average diagonal energy (typically $p = 0.01$):
   $$H_{\text{damp}} = H + \lambda I, \qquad \lambda = p \cdot \frac{\text{Tr}(H)}{K}$$
   and precomputing the upper-triangular Cholesky factor $L = \text{cholesky}(H_{\text{damp}}^{-1})^\top$ in a single pass, completely avoiding iterative matrix inversions.

### 4.4 `actorder` (`desc_act`) Trade-Off

Enabling `desc_act=True` sorts columns in **descending order of $\text{diag}(H)$** (largest input activation energy quantized first, while all remaining columns are still available to compensate!). This consistently improves `INT4` perplexity, especially on `g=128` or per-channel models, **but creates a major runtime penalty**: because group-wise scale indices now follow permuted column order rather than contiguous $K$-order, naive `W4A16` kernels must perform non-coalesced activation gathers (`X[:, perm]`), slowing decode by 15%–40% unless repacked carefully (e.g., Marlin's `g_idx` shuffle support).

---

## 5. Head-to-Head Comparison: SmoothQuant vs. AWQ vs. GPTQ

| Dimension | **SmoothQuant** | **AWQ** | **GPTQ** |
| :--- | :--- | :--- | :--- |
| **Target Regime** | **`W8A8`** (`INT8`/`FP8` prefill & large-batch) | **`W4A16`** (`INT4`/`UINT4` memory-bound decode) | **`W4A16`** / `W3A16` / `W2A16` |
| **Core Mechanism** | Diagonal scale migration $s = \frac{\max(\|X\|)^\alpha}{\max(\|W\|)^{1-\alpha}}$ | Activation-salience protection $\mathbf{s} = \mathbf{s}_X^\alpha$ + grid search | Second-order Hessian error compensation $\delta_F \propto H^{-1}_{:,q}$ |
| **Modifies Quantized Grid Values?** | No (scales $X$ and $W$ equivalently, then RTN) | No (scales $X$ and $W$ equivalently, then RTN) | **Yes** (actively shifts unquantized weights) |
| **Calibration Statistic** | Channel maxima $\max(\|X_j\|)$ | Channel mean abs $\text{mean}(\|X_j\|)$ | Second-order input covariance $H = 2XX^\top$ |
| **Calibration Compute Cost** | Seconds–Minutes (forward pass only) | Minutes (forward pass + 20-pt $\alpha$ grid) | Tens of minutes–Hours (Cholesky + block updates) |
| **Overfitting Risk to Calibration Set** | Low (sensitive mainly to peak outliers) | **Very Low** (only 1 scalar $\alpha$/layer) | **Moderate** (can overfit $H$ on small/biased calibration sets) |
| **Runtime Kernel Impact** | Zero overhead if folded into Norm; requires W8A8 GEMM | Zero overhead if folded; uses standard Marlin/AWQ W4A16 | Standard Marlin W4A16 (`desc_act=False`); gather penalty if `desc_act=True` |

---

## 6. Calibration Pipelines & Execution Trace

### 6.1 Calibration Pipeline Comparison

```mermaid
flowchart TD
    CAL["Calibration Dataset (128-512 Sequences, e.g., Pile / WikiText-2 / Code)"] --> FWD["Layer-by-Layer Forward Pass"]

    FWD --> SQ["SmoothQuant Path (Target: W8A8)"]
    FWD --> AWQ["AWQ Path (Target: W4A16)"]
    FWD --> GPTQ["GPTQ Path (Target: W4A16 / W3A16)"]

    SQ --> SQ1["Record Channel Peak max(|X_j|)"]
    SQ1 --> SQ2["Compute s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)"]
    SQ2 --> SQ3["Fold diag(s)^-1 into Prev Norm & diag(s) into W"]
    SQ3 --> SQ4["RTN Quantize W & Calibrate X Scales (INT8 / FP8)"]

    AWQ --> AWQ1["Record Channel Mean Salience s_X = mean(|X_j|)"]
    AWQ1 --> AWQ2["Grid Search alpha in [0,1] Minimizing ||Q(W*s)*X/s - WX||"]
    AWQ2 --> AWQ3["Fold diag(s)^-1 into Prev Layer & Scale W_salient Up"]
    AWQ3 --> AWQ4["Group-Wise RTN Quantize to UINT4 (g=128)"]

    GPTQ --> GPTQ1["Accumulate Hessian H = 2 * X * X^T"]
    GPTQ1 --> GPTQ2["Dampen H + lambda*I & Compute Cholesky of H^-1"]
    GPTQ2 --> GPTQ3["Block Loop (B=128): Quantize Col q, Compensate via H^-1"]
    GPTQ3 --> GPTQ4["Lazy GEMM Update Remaining Unquantized Columns"]
```

### 6.2 Calibration Dataset Pass: Hessian / Activation Statistics & Weight Update

```mermaid
sequenceDiagram
    autonumber
    participant DS as "Calibration Dataset"
    participant Mod as "Transformer Block (Layer L)"
    participant Stat as "Calibration Collector"
    participant Solver as "AWQ / GPTQ Solver"
    participant Quant as "Quantized Checkpoint"

    loop Over N calibration batches (e.g., 128 x 2048 tokens)
        DS->>Mod: Feed token batch X_in
        Mod->>Stat: Hook inputs X entering Linear(W)
        Stat->>Stat: Accumulate s_X += sum(|X|) and H += 2 * X * X^T
        Mod->>Mod: Execute original FP16 forward to produce next-layer X_out
    end

    Stat->>Solver: Pass accumulated s_X, H, and FP16 weights W

    alt AWQ Calibration
        Solver->>Solver: Sweep alpha in [0.0..1.0], evaluate ||Q(W*s)*X/s - WX||_F
        Solver->>Mod: Fold optimal diag(s^*)^-1 into preceding RMSNorm gamma
        Solver->>Quant: Pack UINT4 weights + FP16 group scales (g=128)
    else GPTQ Calibration
        Solver->>Solver: Add dampening lambda*I to H and compute Cholesky(H^-1)
        loop Over column blocks of size B=128
            Solver->>Solver: Quantize col q and apply error delta_F inside block
            Solver->>Solver: Apply lazy BLAS GEMM update to remaining columns W[:, i+B:]
        end
        Solver->>Quant: Pack INT4 weights + FP16 group scales (g=128)
    end

    Quant->>Mod: Propagate quantized layer outputs as inputs to Layer L+1
```

*(Note the crucial final step: sequential layer-by-layer calibration feeds the **quantized** output of Layer $L$ as the input $X$ to Layer $L+1$, allowing Layer $L+1$ to adapt to upstream quantization noise!)*

---

[← 01 — Numerical Representation](./01-numerical-representation.md) · [Stage 6 Index](./README.md) · **Next:** [03 — Hardware Execution (W4A16 vs. W8A8) →](./03-hardware-execution-w4a16-vs-w8a8.md)
