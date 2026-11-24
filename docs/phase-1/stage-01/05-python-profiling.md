# 05 — Python Profiling

[Previous: 04 — C/C++ and Rust on the Hot Path](./04-c-cpp-and-rust-on-the-hot-path.md) | [Stage 1 Index](./README.md)

---

## 1. The Python GIL Bottleneck in Inference Pipelines

In standard CPython (3.8–3.12, and default builds of 3.13+), memory management relies on non-atomic reference counting (`PyObject->ob_refcnt`). To prevent two OS threads from corrupting reference counts or internal dicts simultaneously, CPython enforces the **Global Interpreter Lock (GIL)**: a single mutex + condition variable allowing **only one thread to execute Python bytecode at any instant**.

*(Note: Python 3.13 introduced experimental free-threaded builds via PEP 703 `python3.13t`, though production CUDA/PyTorch inference ecosystems remain predominantly GIL-bound today.)*

### 1.1 How the GIL Starves GPUs

1. **Switching Interval (`sys.getswitchinterval()` = 5 ms):** A Python thread holds the GIL until it finishes 5 ms of bytecode, hits I/O, or enters a C extension that explicitly calls `Py_BEGIN_ALLOW_THREADS` (releasing the GIL).
2. **GIL Convoying:** Suppose Thread A runs an `asyncio`/FastAPI HTTP server parsing incoming JSON prompts while Thread B runs the PyTorch decode scheduler loop dispatching CUDA kernels. Even though **PyTorch releases the GIL inside its C++/CUDA kernels**, every Python line *between* kernels (looping over batch items, checking `if request.finished:`, slicing lists, Pydantic validation) requires reacquiring the GIL. If Thread A holds the GIL for 4 ms parsing JSON, Thread B stalls—and the **GPU finishes its current step and sits 100% idle** waiting for the next kernel launch!
3. **Architecture Mitigation:** Production serving engines isolate the GPU scheduler in a dedicated OS process (`multiprocessing` / ZeroMQ IPC), run HTTP/tokenization in separate worker processes or async Rust/C++ threads that drop the GIL, and profile continuously to detect hidden GIL serialization.

---

## 2. Sampling vs. Deterministic Profiling

| Profiler | Mechanism | Overhead | Visibility | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`cProfile`** | Deterministic (`sys.setprofile` hook on every Python `call`/`return`/`c_call`) | **High (2x–10x slowdown)** on tight Python loops | Exact call counts, `tottime`, `cumtime` for Python & C-call boundaries | Micro-benchmarking pure-Python logic; **unsafe in production** and distorts async GPU timing |
| **`py-spy`** | Out-of-process sampling via Linux `process_vm_readv` (reads target stack at 100–1000 Hz) | **Negligible (<1–2%)**; zero target code changes | Python stacks + native C/C++/Rust frames (`--native`) + GIL status (`--gil`) | Production flamegraphs, finding GIL bottlenecks & hidden native CPU hotspots |
| **`tracemalloc`** | Hooks CPython `PyMem_*` / `PyObject_*` allocators | **Moderate–High** (scales with `nframe` traceback depth) | Exact source line + size of **CPU Python heap** allocations (NOT GPU VRAM!) | Catching host RAM leaks in request queues, KV metadata, or tokenizer caches |
| **`torch.profiler`** | CUPTI (CUDA Profiling Tools Interface) + PyTorch ATen/Autograd CPU trace hooks | **Moderate** during active trace window (use `schedule=`) | Unified CPU + CUDA Runtime + GPU Kernel timeline (Chrome/Perfetto JSON) | Finding GPU kernel gaps, CPU launch overhead, and implicit `.item()` / `.cpu()` sync stalls |

---

## 3. Deterministic Profiling (`cProfile`) & Allocation Tracking (`tracemalloc`)

### 3.1 `cProfile`: Interpreting `tottime` vs. `cumtime`

```python
import cProfile
import pstats

with cProfile.Profile() as pr:
    run_inference_batch(requests)

stats = pstats.Stats(pr).sort_stats("cumtime")
stats.print_stats(20)
```

* **`ncalls`:** Exact invocation count (e.g., `500000` calls to a tiny helper indicates loop overhead dominating work).
* **`tottime` (Self Time):** Time spent *strictly inside* that function, excluding sub-calls. High `tottime` in a pure-Python function signals CPU-bound bytecode work to move to vectorized NumPy/Rust/C++.
* **`cumtime` (Cumulative Time):** Time spent in that function *plus* all callees. Note: C functions that call `cudaStreamSynchronize` appear as a single high `tottime` inside `built-in method item` or `method 'cpu' of 'torch._C.TensorBase'`—masking whether the time was CPU compute or GPU wait!
* **Heisenberg Distortion:** Because `cProfile` adds ~500 ns–2 $\mu$s per Python function call, a tight loop calling a tiny Python wrapper 100,000 times slows down artificially by 5x–10x, completely masking real GPU execution ratios.

### 3.2 `tracemalloc`: Finding Host Memory Leaks

Long-running inference servers frequently suffer gradual host RSS growth until Linux OOM-kills the worker. `tracemalloc` isolates which line of Python allocated unreleased CPU objects:

```python
import tracemalloc

tracemalloc.start(10)  # Capture up to 10 stack frames per allocation
snap1 = tracemalloc.take_snapshot()

for _ in range(1000):
    serve_one_request()

snap2 = tracemalloc.take_snapshot()
for stat in snap2.compare_to(snap1, "lineno")[:10]:
    print(f"{stat.traceback.format()[0]} | +{stat.size_diff / 1024:.1f} KiB ({stat.count_diff} blocks)")
```

> **Critical Scope Boundary:** `tracemalloc` tracks allocations routed through CPython's `PyMem` API (Python dicts, strings, lists, and NumPy arrays if NumPy's `PyDataMem` hooks are active). It **does not track** `cudaMalloc` GPU VRAM (use `torch.cuda.memory_snapshot()` / `torch.cuda.max_memory_allocated()` for VRAM) or raw C++/Rust `malloc` allocations inside extensions that bypass `PyMem`.

---

## 4. Sampling Flamegraphs with `py-spy`

`py-spy` attaches externally to a running PID (requiring `CAP_SYS_PTRACE` or same-user permissions) and samples call stacks asynchronously without pausing the GIL:

```bash
# 1. Capture native C/C++/Rust + Python frames at 250 Hz for 30 seconds into an interactive SVG flamegraph
py-spy record -p $(pgrep -f serve.py) --native --rate 250 --duration 30 -o flamegraph.svg

# 2. Isolate ONLY threads actively holding the GIL (exposes GIL contention!)
py-spy top -p $(pgrep -f serve.py) --gil

# 3. Include idle/sleeping threads (shows threads blocked waiting on locks, I/O, or cudaStreamSynchronize)
py-spy record -p $(pgrep -f serve.py) --native --idle -o flamegraph_with_idle.svg
```

### 4.1 How to Read a Flamegraph Correctly

1. **Y-Axis = Call Stack Depth:** Bottom bar is the process/thread entrypoint (`_start` / `run_loop`); moving upward traces caller $\rightarrow$ callee; the **topmost edge** is the leaf function executing on CPU at the instant of sampling.
2. **X-Axis = Sample Proportion (NOT Time!):** Frames at the same depth are sorted **alphabetically**, *not chronologically*. A wide bar does **not** mean "one slow 10-second call"; it means **that function was on the call stack for X% of all captured samples** (either called once slowly, or called 1,000,000 times!).
3. **Diagnostic Shapes:**
   * **Wide Flat Plateaus at the Top:** True CPU bottlenecks! Example: `re.findall` in a BPE pre-tokenizer, `json.dumps` during streaming SSE formatting, or `libc.so` `memcpy` from unpinned H2D staging.
   * **Wide `libpthread` / `take_gil` / `cudaStreamSynchronize` Towers:** Waiting/synchronization bottlenecks (inspect with `--idle` and `--native`).
   * **Deep Narrow Spires:** Framework/Pydantic/PyTorch dispatcher overhead; optimize by batching calls or fusing ops.

---

## 5. `torch.profiler` and Interpreting Chrome / Perfetto Traces

While `py-spy` sees only the CPU, **`torch.profiler`** fuses CPU execution with **NVIDIA CUPTI** hardware timestamps into a single synchronized timeline viewed at **`ui.perfetto.dev`** (or `chrome://tracing`).

### 5.1 Proper Instrumentation Pattern

Never profile step 0 (CUDA context init + cuBLAS heuristic tuning + `torch.compile` JIT skews traces). Always use `schedule(wait, warmup, active, repeat)`:

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

prof_sched = schedule(wait=5, warmup=2, active=3, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_sched,
    record_shapes=True,
    with_stack=True,
    profile_memory=False,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./traces"),
) as prof:
    for step in range(10):
        with torch.profiler.record_function("scheduler_and_prep"):
            batch = prepare_batch()
        with torch.profiler.record_function("model_forward"):
            logits = model(batch)
        with torch.profiler.record_function("sample"):
            next_tok = torch.argmax(logits[:, -1, :], dim=-1)
        prof.step()  # Marks iteration boundary for schedule
```

### 5.2 Reading the Perfetto / Chrome Trace Timeline

A healthy GPU-bound trace looks like a **solid wall of packed GPU kernels** on the CUDA Stream track, with CPU `cudaLaunchKernel` slices running **ahead** of the GPU (asynchronous queuing):

1. **CPU Thread Track (`python` / `pt_autograd_0`):**
   * Shows `record_function` labels, ATen C++ dispatchers (`aten::mm`, `aten::scaled_dot_product_attention`), and **CUDA Runtime calls** (`cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaStreamSynchronize`, `cudaDeviceSynchronize`).
   * **Kernel Launch Overhead:** Every `cudaLaunchKernel` takes **~3–15 $\mu$s** on CPU. If a decode batch launches 300 tiny kernels (RMSNorm, RoPE, QKV bias, residual add, SiLU) that each take only **4 $\mu$s** on an H100, CPU dispatch takes $300 \times 8\text{ }\mu\text{s} = 2.4\text{ ms}$ while total GPU compute is only $1.2\text{ ms}$! The GPU stream shows **50% idle gaps ("bubbles")** between kernels—classic **CPU-launch-bound** execution (fixed by operator fusion and CUDA Graphs in Stage 8).
2. **Correlation Flow Arrows:** Clicking a `cudaLaunchKernel` slice on the CPU track draws a **flow arrow** down to the exact GPU kernel slice on `stream 7` (matched by CUPTI `correlation_id`). Large horizontal distance between launch and execution means the CPU is successfully staying ahead (good queue depth); zero distance means the CPU fell behind and the GPU is starving!
3. **GPU Stream Track (`stream 7` / Compute & `stream 14` / Memcpy):**
   * Inspect **gaps (white space / bubbles)** between consecutive GPU kernel rectangles. Every microsecond of gap means Tensor Cores and HBM bandwidth are sitting at **0% utilization**.
4. **Implicit CUDA Synchronization Points (The #1 Silent Performance Killer):**
   PyTorch operations are normally **asynchronous**—the CPU enqueues a kernel onto the CUDA stream and immediately moves on. However, any operation requiring a GPU value on the host CPU silently injects a **`cudaMemcpyDtoH` + `cudaStreamSynchronize`**, blocking the CPU until the *entire queued GPU pipeline drains*:
   * **`tensor.item()`** / `int(tensor)` / `float(tensor)` (e.g., checking `if next_token.item() == EOS_ID:` inside the decode loop before launching the next step!)
   * **`tensor.cpu()`** / **`tensor.numpy()`** (synchronous D2H copy on the default stream)
   * **`print(tensor)`** or Python `bool(tensor > 0)` / `if tensor:`
   * **Dynamic-shape ops like `torch.nonzero(mask)` or `torch.unique()`** (CPU cannot allocate output tensor without knowing how many elements matched on GPU $\implies$ forced sync!).

---

## 6. GPU Starvation Timeline: GIL Contention + Implicit Sync

The sequence diagram below illustrates how a seemingly innocent decode loop suffers massive GPU starvation from two combined Stage 1 issues: another thread holding the Python GIL during tokenization/JSON parsing, followed by an implicit `.item()` synchronization stall.

```mermaid
sequenceDiagram
    autonumber
    participant HTTP as "Thread A - FastAPI / JSON (Holds GIL)"
    participant Sched as "Thread B - PyTorch Decode Loop"
    participant CUDA as "CUDA Driver / Stream Queue"
    participant GPU as "GPU SMs & Tensor Cores"

    Sched->>CUDA: Enqueue Layer 0..32 kernels (async, ~300 us CPU)
    CUDA->>GPU: Execute Layer 0..32 (~1.5 ms GPU work)
    Note over Sched,GPU: GPU is busy while CPU should immediately prepare Step t+1
    Sched->>Sched: Call next_tok.item() (IMPLICIT SYNC!)
    Sched->>CUDA: Silently issues cudaMemcpyDtoH + cudaStreamSynchronize
    Note over Sched: Thread B blocks 1.5 ms waiting for GPU to drain!
    GPU-->>CUDA: Step t finishes and GPU sits 100% IDLE (Bubble!)
    CUDA-->>Sched: Returns scalar token ID, Thread B wakes up
    HTTP->>HTTP: Acquires GIL to parse 50 KB JSON + regex tokenize (4 ms)
    Sched->>Sched: Blocked waiting on GIL! Cannot launch Step t+1
    Note over GPU: GPU remains completely idle for 4.0 ms!
    HTTP-->>Sched: Releases GIL after switch interval / C-call
    Sched->>CUDA: Finally launches Step t+1 kernels
```

---

## 7. Diagnostic Decision Flowchart: CPU-Bound vs. GPU-Bound Pipelines

Use the systematic triage flow below whenever inference throughput (tokens/s) or latency (TTFT / ITL) misses hardware Roofline expectations:

```mermaid
flowchart TD
    Start["Observed Slow Inference (Low Tokens/s or High ITL)"] --> CheckGPU["Check GPU SM / HBM Utilization (nvidia-smi dmon / Perfetto)"]
    CheckGPU -->|"GPU > 85% Busy, No Stream Gaps"| GPUBound["GPU-Bound (Compute or HBM Bandwidth)"]
    GPUBound --> Stage23["Proceed to Stage 2/3/6/8 (Quantization, FlashAttention, Batching)"]

    CheckGPU -->|"GPU < 75% Busy, Gaps Between Kernels"| Capture["Capture torch.profiler Chrome Trace (wait=5, active=3)"]
    Capture --> SyncCheck{"See Long cudaStreamSynchronize or cudaMemcpyDtoH?"}

    SyncCheck -->|"Yes"| SyncFix["Find Calling Python Stack in Perfetto: Remove .item() / .cpu() / nonzero() or Use Async Events + Pinned D2H"]
    SyncCheck -->|"No"| LaunchCheck{"CPU Track Packed with Tiny cudaLaunchKernel Calls?"}

    LaunchCheck -->|"Yes (CPU Launch Overhead > GPU Kernel Time)"| LaunchFix["CPU Launch-Bound: Fuse Ops (torch.compile / Triton) or Enable CUDA Graphs"]
    LaunchCheck -->|"No (Large Empty/Python Gaps on CPU Track)"| PySpy["Run py-spy record --native --gil -p PID"]

    PySpy --> GILCheck{"High Time in take_gil or Thread Contention?"}
    GILCheck -->|"Yes"| GILFix["GIL-Bound: Isolate GPU Loop in Dedicated Process, Release GIL in C++/Rust FFI"]
    GILCheck -->|"No (Wide CPU Plateau in Python / Native)"| CPUHot["CPU Hot-Path Bottleneck: Vectorize SIMD, Fix SoA/False Sharing, Move to PyO3/nanobind"]
    PySpy --> LeakCheck["Host RAM Growing Over Time? Run tracemalloc Snapshot Diff"]
```

---

## Navigation

* **Previous:** [04 — C/C++ and Rust on the Hot Path](./04-c-cpp-and-rust-on-the-hot-path.md)
* **Up:** [Stage 1 Index](./README.md)
