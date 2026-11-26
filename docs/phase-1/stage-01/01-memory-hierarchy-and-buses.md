# 01 — Memory Hierarchy & Buses

[Stage 1 Index](./README.md) | [Next: 02 — Hardware Execution Models](./02-hardware-execution-models.md)

---

This document explains why moving data—not performing math—is the primary bottleneck in modern servers, and traces the exact physical path data takes from long-term storage to processor math units. Before diving into the hardware numbers, the section below defines every hardware, operating-system, and AI-inference term used throughout this guide.

## Core Concepts & Terminology

### 1. Basic Computing & Silicon Building Blocks
* **Silicon Die & Transistors:** Modern computer chips are etched onto tiny flat squares of silicon called **dies**. Each die contains billions of microscopic electronic switches called **transistors**. **Moore's Law** observed that transistor count roughly doubles every two years, while **Dennard scaling** historically meant those smaller transistors also used proportionally less power (a trend that has largely ended, making energy per operation critical).
* **Clock Cycle & Clock Frequency (GHz):** Processors march to an internal electronic metronome. One tick of that metronome is a **clock cycle**. A 4.0 GHz processor ticks 4 billion times per second, meaning each clock cycle lasts **0.25 nanoseconds (ns)** (1 ns = one billionth of a second; 1 **$\mu$s** = one microsecond = 1,000 ns). Energy is measured in **picojoules (pJ)** ($10^{-12}$ J) and **microjoules ($\mu$J)** ($10^{-6}$ J).
* **Package Pins, PCB, & Signal Integrity:** To talk to external memory or other chips, electrical signals must leave the silicon die through metal **package pins** and travel across copper **traces** on the **Printed Circuit Board (PCB)**—typically made of fiberglass **FR-4 dielectric** material. Sending signals off-chip is physically limited by electrical **capacitance**, **signal integrity** (noise/distortion at high speeds), pin count, and the speed of electricity in copper (~15 cm per nanosecond).
* **CPU (Central Processing Unit) & Cores:** The general-purpose brain of a server, built to run operating systems and complex control logic fast. A modern server CPU contains dozens of independent processing units called **cores**, housed inside one physical chip **socket** (often subdivided internally into smaller chiplets called **CCDs**, or Core Complex Dies).
* **ALU (Arithmetic Logic Unit):** The electronic circuit inside a processor core that actually performs math (addition, multiplication) and logic operations.
* **MAC (Multiply-Accumulate):** A single combined math operation ($a \times b + c$) that is the fundamental building block of neural networks.
* **GPU (Graphics Processing Unit) & Tensor Cores:** A specialized accelerator chip containing thousands of simple math cores designed for massive parallel math. Modern AI GPUs also include **Tensor Cores**, which are dedicated hardware units that multiply entire small matrices in a single clock cycle.
* **Number Formats (FP32, FP16, BF16):** Computers store decimal numbers in binary floating-point formats. **FP32** (32-bit Single Precision) uses 4 bytes per number. **FP16** (16-bit Half Precision) and **BF16** (Brain Float 16, which trades decimal precision for a wider exponent range) use 2 bytes per number, halving memory traffic and doubling math throughput for AI workloads.

### 2. Processor Memory & Cache Mechanics
* **Memory Hierarchy:** Because fast memory is expensive and physically tiny, computers stack multiple tiers of memory—from tiny, ultra-fast storage right next to the ALUs to huge, slower storage further away.
* **Registers (GPRs, PRF, AVX-512 ZMM):** The fastest, smallest storage tier inside a CPU core, holding the immediate numbers an ALU is working on right now. **Architectural Registers** (such as 16–32 **General-Purpose Registers / GPRs** and wide 512-bit **AVX-512 ZMM** vector registers) are the names visible to software, while the hardware transparently maps them onto a larger pool of **Physical Register File (PRF)** entries via **hardware register renaming**.
* **SRAM vs. DRAM:** **SRAM (Static Random-Access Memory)** uses 6 transistors per bit; it is blazingly fast and lives directly on the CPU/GPU silicon die, but takes up lots of chip area. **DRAM (Dynamic Random-Access Memory)** uses 1 transistor + 1 tiny capacitor per bit; it is much denser and cheaper, so it serves as main system RAM on external sticks (**DIMMs**), at the cost of much higher access time.
* **Cache (L1i, L1d, L2, L3 / LLC):** Small pools of on-die SRAM that automatically hold copies of recently used DRAM data so the CPU does not have to wait for slow off-chip reads. **L1 Cache** is split into **L1i** (instructions) and **L1d** (data) per core. **L2 Cache** is a larger private cache per core. **L3 Cache** (also called **LLC**, Last-Level Cache) is a large pool shared across multiple cores (sometimes stacked vertically using **3D V-Cache**).
* **Temporal & Spatial Locality:** Caches work because programs exhibit **temporal locality** (if you read a byte now, you will likely read it again soon) and **spatial locality** (if you read a byte at address $X$, you will likely read nearby addresses $X+1, X+2$ soon).
* **Cache Line (64-Byte Aligned Block):** Caches do not track single bytes. Instead, hardware fetches and stores data in fixed **64-byte chunks** called **cache lines**.
* **Set-Associative Cache, Tag, Set Index, & Byte Offset:** A cache organizes its 64-byte lines into **sets** (e.g., an 8-way set-associative cache holds 8 lines per set). To look up a memory address, the hardware splits the address into three parts: the **Byte Offset** (which of the 64 bytes inside the line is wanted), the **Set Index** (which set to check), and the **Tag** (the upper address bits used to verify a match).
* **Virtual Memory, Pages, TLB, & VIPT:** Operating systems give every program an isolated **virtual address** space, divided into fixed-size blocks called **pages** (typically 4 KB or 2 MB), and translate them to real hardware **physical addresses**. A **TLB (Translation Lookaside Buffer)** is a tiny hardware cache of recent virtual-to-physical translations. **VIPT (Virtually Indexed, Physically Tagged)** is a trick where L1d uses the virtual address bits to pick the cache set *at the same time* the TLB looks up the physical tag, hiding translation delay.
* **Line Fill Buffer (LFB / MSHR):** A small hardware queue inside a CPU core that tracks outstanding cache misses waiting for data to arrive from lower memory tiers.
* **Hardware Prefetcher & Pointer Chasing:** A **hardware prefetcher** watches memory access patterns (like sequential streams or fixed strides) and speculatively fetches future cache lines before the CPU asks for them. **Pointer chasing** (following memory addresses scattered randomly across RAM, such as in linked lists or Python object graphs) is unpredictable and defeats prefetchers.
* **Cache Coherence (MESI / MOESI), RFO, & False Sharing:** Because each core has its own private L1/L2 cache, hardware runs a **coherence protocol** (such as **MESI** or **MOESI**) to ensure all cores see a consistent view of memory. Before Core 0 can modify a single byte, it must claim exclusive ownership of the entire 64-byte cache line via **RFO (Read-For-Ownership)**, invalidating copies on other cores. **False sharing** occurs when two cores modify *different* variables that happen to sit inside the *same* 64-byte cache line, forcing the line to bounce back and forth between cores.
* **Non-Temporal Stores & Write-Combining (WC) Buffers:** Normal memory writes first load the target cache line into L1/L2, evicting useful data. **Non-temporal store** instructions (`MOVNTDQ` / `VMOVNTDQ`) bypass the caches completely, accumulating writes in a **Write-Combining (WC) buffer** and flushing full 64-byte lines straight to RAM.

### 3. Main System Memory (DDR5) & Multi-Socket Servers
* **DDR5 (Double Data Rate 5) & MT/s:** The current server standard for main system RAM. "Double Data Rate" transfers data on both the rising and falling edges of a clock signal, so **DDR5-4800** runs a 2,400 MHz clock to achieve 4,800 **MT/s (Megatransfers per second)**. Server modules include **ECC (Error-Correcting Code)** bits to detect and fix bit flips.
* **Integrated Memory Controller (IMC) & STREAM Triad:** The **IMC** is the circuit on the CPU die that talks to DDR5 DIMMs. **STREAM Triad** is a standard benchmark used to measure *real sustained* memory bandwidth (as opposed to theoretical peak).
* **DRAM Internal Organization (Channel, Rank, Bank Group, Bank, Row, Column):** A DDR5 subsystem is split into parallel **channels** (and 32-bit **sub-channels**). Inside each chip, memory is arranged into **ranks**, **bank groups**, and **banks**, where each bank is a 2D grid of **rows** (also called DRAM pages, 8–16 KB wide) and **columns**. Reading a row copies it into **sense amplifiers** (the row buffer). A **row-buffer hit** reads directly from the open row (**`CAS` / `tCL`**); a **row-buffer miss** must close the old row (**`PRECHARGE` / `tRP`**) and open the new one (**`ACTIVATE` / `tRCD`**) before reading, traversing the physical **PHY** and **uncore** (non-core chip logic).
* **NUMA (Non-Uniform Memory Access) & UPI / xGMI:** In dual-socket servers (two physical CPUs on one motherboard), each CPU has its own directly attached DDR5 RAM (**Local NUMA**). Accessing RAM attached to the *other* CPU (**Remote NUMA**) must hop across an inter-socket link—called **UPI (Ultra Path Interconnect)** on Intel or **xGMI (Infinity Fabric)** on AMD—doubling latency and slashing bandwidth.
* **HBM / HBM3e (High Bandwidth Memory):** A specialized type of DRAM stacked vertically in 3D layers right next to a GPU die, offering massive parallel buses (3–5+ TB/s) compared to DDR5 (~400 GB/s).

### 4. Buses, PCIe, DMA, & NVMe Storage
* **PCIe (Peripheral Component Interconnect Express):** The high-speed serial expansion bus connecting the CPU's **Root Complex (RC)** to add-in devices like GPUs, **NICs (Network Interface Cards)**, and SSDs. Each link consists of bidirectional wire pairs called **lanes** (e.g., **x4** = 4 lanes, **x16** = 16 lanes). Generations (**Gen3, Gen4, Gen5, Gen6**) double per-lane speed each step, measured in **GT/s (Gigatransfers per second)**.
* **Line Coding (`8b/10b`, `128b/130b`, PAM4/FLIT):** Physical wires need frequent voltage transitions to keep sender and receiver clocks synchronized. Gen3–Gen5 use **`128b/130b` encoding** (2 sync bits added per 128 data bits). Gen6 switches to **PAM4** (4-level voltage signaling) and fixed **FLIT** frames protected by **CRC** (error detection) and **FEC** (Forward Error Correction).
* **TLP, DLLP, MPS, & MRRS:** On PCIe, data travels inside packets: **TLPs (Transaction Layer Packets)** carry the payload plus headers, sequence numbers, and **LCRC** checksums; **DLLPs (Data Link Layer Packets)** handle link acknowledgments (**ACKs**) and flow control. **MPS (Max Payload Size)** caps how many payload bytes fit in a single TLP, while **MRRS (Max Read Request Size)** caps how much data a single read request can ask for. Every 4 bytes of header is called a **DWORD**.
* **MMIO, DMA, & H2D / D2H:** **MMIO (Memory-Mapped I/O)** lets the CPU send small control commands to a device by writing to special physical addresses (measured in **RTT**, Round-Trip Time). For bulk data, **DMA (Direct Memory Access)** lets a GPU's **Copy Engine** or storage controller move gigabytes directly between RAM and the device without the CPU copying each byte. Moving data from CPU RAM to GPU memory is **H2D (Host-to-Device)**; moving it back is **D2H (Device-to-Host)**. **Pinned memory** is host RAM locked in place so DMA hardware can safely read/write it directly, enabling asynchronous transfers (`cudaMemcpyAsync`) instead of slower unpinned `cudaMemcpy`.
* **NVMe SSD & NAND Flash:** **NVMe (Non-Volatile Memory Express)** is a fast storage protocol built directly on PCIe, replacing older **SATA/AHCI** disk interfaces. An **SSD (Solid-State Drive)** stores data in **NAND flash** cells—fast **SLC** (1 bit/cell, often used as a write cache), standard **TLC** (3 bits/cell), or dense **QLC** (4 bits/cell)—protected by **LDPC ECC**. Data is addressed in 4 KB **LBAs (Logical Block Addresses)** using **Submission Queue Entries (SQEs)**, **Completion Queue Entries (CQEs)**, and **MSI-X** interrupts. **Queue Depth (QD)** is how many I/O requests are in flight at once, and **IOPS** is I/O operations per second. **RAID-0** stripes data across multiple SSDs to multiply bandwidth.
* **Linux Page Cache:** Unused host DDR5 RAM that the Linux kernel automatically uses to cache recently read SSD file blocks.

### 5. Performance Laws & LLM Inference Terms
* **Latency vs. Throughput/Bandwidth & Little's Law:** **Latency ($L$)** is how long one operation takes; **bandwidth ($B$)** is how many bytes per second flow through a pipe. **Little's Law** ($\text{Concurrency} = B \times L$) proves how many bytes must be simultaneously "in flight" to keep a high-bandwidth pipe 100% full.
* **LLM (Large Language Model), Tokens, Weights, & Logits:** An **LLM** generates text one **token** (word piece) at a time using billions of learned numerical parameters called **weights**. At each step, the model outputs **logits** (raw unnormalized probability scores across the entire token vocabulary). **Grammar-constrained decoding** applies a **vocabulary bitmask** to zero out invalid tokens (e.g., to enforce valid JSON).
* **Prefill, Decode, & TTFT:** Processing the user's input prompt in parallel is **Prefill**; generating output tokens one by one is **Decode**. **TTFT (Time To First Token)** is how long the user waits before seeing the first generated token, a key **SLA (Service-Level Agreement)** metric.
* **KV Cache, GQA, & `head_dim`:** During generation, LLMs cache intermediate **Key and Value (KV)** attention tensors from past tokens so they do not have to be recomputed. **GQA (Grouped-Query Attention)** reduces KV cache size by sharing KV heads across query heads (each of dimension `head_dim`).
* **MoE (Mixture of Experts):** An LLM architecture where each token only activates a small subset of specialized sub-networks ("experts") rather than the entire model's weights.
* **KEDA & Ray Serve:** Kubernetes/distributed **autoscaling** tools that spin up new GPU inference pods on demand—where cold-start speed is governed by SSD and PCIe weight-loading time.

---

## 1. First Principles: Why Memory Dominates Compute

Modern silicon has evolved asymmetrically. Arithmetic Logic Units (ALUs) and Tensor Cores scale rapidly with transistor density (Moore's Law / Dennard post-scaling), whereas moving electrons off-die across package pins, PCB traces, and PCIe connectors is constrained by signal integrity, capacitance, pin count, and speed-of-light propagation (~$15\text{ cm/ns}$ in FR-4 PCB dielectric).

Consequently, **data movement costs orders of magnitude more time and energy than arithmetic**:

* A single FP16 Multiply-Accumulate (MAC) in 5nm–7nm silicon consumes **~0.2–0.9 pJ** and completes in **1 clock cycle** (~0.25–0.5 ns).
* Reading that operand from off-chip DDR5 DRAM consumes **~300–600 pJ** and takes **70–100 ns** (equivalent to ~300–400 CPU clock cycles).
* Fetching it from an NVMe SSD consumes **microjoules** and takes **10–80 $\mu$s** (~50,000–300,000 CPU clock cycles).

To bridge this 5-order-of-magnitude gap between compute speed and storage latency, computer architectures implement a strict **memory hierarchy** governed by the principles of **temporal locality** (recently accessed addresses are likely to be accessed again soon) and **spatial locality** (addresses adjacent to recently accessed addresses are likely to be accessed soon).

---

## 2. Quantitative Anatomy of the Memory Hierarchy

The table below gives concrete hardware metrics representative of modern server platforms (e.g., dual-socket AMD EPYC Genoa/Turin or Intel Xeon Emerald/Granite Rapids paired with PCIe Gen4/Gen5 GPUs and NVMe drives):

| Tier | Typical Capacity | Access Latency (Cycles / Wall Time) | Sustained Bandwidth | Management / Unit of Transfer |
| :--- | :--- | :--- | :--- | :--- |
| **CPU Architectural Registers** | ~1–4 KB (16–32 GPRs + 32 AVX-512 ZMMs = ~2.2 KB; backed by ~200–300+ physical PRF entries) | 0–1 cycle (~0.2–0.3 ns @ 3.5–4.5 GHz) | >20–50 TB/s aggregate across cores | Compiler / Hardware Register Renaming; 4–64 bytes |
| **L1 Cache (L1i + L1d)** | 32–64 KB per core (split I/D) | 3–5 cycles (~0.8–1.4 ns) | ~200–400 GB/s per core | Hardware; **64-byte cache line** |
| **L2 Cache** | 512 KB – 2 MB per core (private) | 12–16 cycles (~3.5–5 ns) | ~100–200 GB/s per core | Hardware; 64-byte cache line |
| **L3 Cache (LLC)** | 32 MB – 384+ MB (shared per CCD/socket; up to >1 GB with 3D V-Cache) | 40–80 cycles (~12–25 ns) | ~400–1,200+ GB/s aggregate | Hardware (ring/mesh interconnect); 64-byte cache line |
| **DDR5 System RAM (Local NUMA)** | 128 GB – 4+ TB per socket | ~250–350 cycles (**70–100 ns**; ~14–20 ns row-hit, ~100+ ns row-miss) | **~300–460 GB/s** (8–12 channels DDR5-4800/5600) | Hardware Memory Controller + OS Pages (4 KB / 2 MB) |
| **DDR5 System RAM (Remote NUMA)** | Remote socket capacity | ~450–750 cycles (**140–220 ns**) | **~64–128 GB/s** (limited by UPI / xGMI) | Cross-socket interconnect |
| **PCIe Gen4 x16 Bus** | Interconnect (CPU Root Complex $\leftrightarrow$ GPU/NIC) | ~0.8–1.5 $\mu$s RTT (MMIO/DMA descriptor) | **31.51 GB/s** theoretical (~25–28 GB/s sustained DMA) | TLP packets (128B–512B MPS) |
| **PCIe Gen5 x16 Bus** | Interconnect (CPU Root Complex $\leftrightarrow$ GPU/NIC) | ~0.7–1.2 $\mu$s RTT | **63.02 GB/s** theoretical (~50–56 GB/s sustained DMA) | TLP packets (128B–512B MPS) |
| **NVMe SSD (PCIe Gen4 / Gen5 x4)** | 1 TB – 61+ TB per drive | **10–80 $\mu$s** (SLC cache ~10 $\mu$s; TLC NAND read ~25–75 $\mu$s) | **~7 GB/s** (Gen4 x4) / **~14 GB/s** (Gen5 x4) | 4 KB logical blocks (LBA), 128 KB–1 MB sequential bursts |

```mermaid
flowchart TD
    CPU["CPU Core Execution Units (0.25 ns clock)"]
    REG["Architectural & Physical Registers (~2 KB, 0-1 cycle, >20 TB/s)"]
    L1["L1d / L1i Cache (32-64 KB/core, ~1 ns, ~300 GB/s/core)"]
    L2["L2 Private Cache (1-2 MB/core, ~4 ns, ~150 GB/s/core)"]
    L3["L3 Shared Last-Level Cache (32-384 MB, ~15 ns, ~800 GB/s)"]
    MC["Integrated Memory Controller (IMC)"]
    DDR5["DDR5 Local RAM (8-12 ch, 70-100 ns, 300-460 GB/s)"]
    NUMA["Remote Socket DDR5 via UPI / xGMI (140-220 ns, 64-128 GB/s)"]
    RC["PCIe Root Complex (Gen4/Gen5)"]
    NVME["NVMe SSD x4 (10-80 us, 7-14 GB/s)"]
    GPU["GPU via PCIe x16 (Gen4 ~26 GB/s, Gen5 ~54 GB/s) -> HBM3e (3-5 TB/s)"]

    CPU --> REG
    REG --> L1
    L1 --> L2
    L2 --> L3
    L3 --> MC
    MC --> DDR5
    MC -.->|"Cross-Socket Hop"| NUMA
    L3 --> RC
    RC --> NVME
    RC --> GPU
```

---

## 3. CPU Caches & 64-Byte Cache Line Mechanics

Caches do **not** track individual bytes or 4-byte floats; the tag overhead would dwarf the stored data. Instead, x86-64 and server ARM64 CPUs manage memory in **64-byte cache lines** (note: Apple Silicon M-series uses 128-byte cache lines).

### 3.1 Physical Address Decomposition

For a 64-byte cache line, the lowest 6 bits of every physical address specify the byte offset within the line ($2^6 = 64$):

$$\text{Physical Address} = [\underbrace{\text{Tag}}_{\text{remaining upper bits}} \mid \underbrace{\text{Set Index}}_{\log_2(\text{Sets})\text{ bits}} \mid \underbrace{\text{Byte Offset}}_{6\text{ bits}}]$$

Consider a classic **32 KB, 8-way set-associative L1d cache** with 64-byte lines:
* Total lines: $\frac{32,768}{64} = 512$ lines.
* Number of sets: $\frac{512}{8\text{-way}} = 64$ sets $\implies 6$ bits for Set Index (bits 6–11).
* Offset bits: 6 bits (bits 0–5).
* Notice that bits 0–11 ($6 + 6 = 12$ bits = 4 KB) fit entirely inside a **4 KB virtual memory page**! This allows L1d to be **VIPT (Virtually Indexed, Physically Tagged)**: the hardware indexes the L1 sets using virtual address bits `[11:6]` *in parallel* with the TLB lookup for the physical tag `[47:12]`, hiding TLB latency on an L1 hit.

### 3.2 What Happens on a Cache Miss

1. **Coalesced Fill (Critical-Word-First):** If a core reads 4 bytes at offset 24 and misses L1/L2/L3, the memory controller fetches the **entire 64-byte aligned block** (`addr & ~0x3F`). DDR5 bursts 64 bytes per transaction (in DDR5, each sub-channel is 32 bits + 8 ECC bits wide with a burst length of 16, yielding $32\text{ bits} \times 16 / 8 = 64\text{ bytes}$ per independent sub-channel!). The requested 8-byte word is forwarded immediately to the core ("critical word first") while the remaining 56 bytes populate the Line Fill Buffer (LFB).
2. **Spatial Utilization Penalty:** If a poorly structured scheduler scans an array of 128-byte structs only reading a single 4-byte `status` flag per struct, **96.875% of the fetched DRAM bandwidth is wasted** ($4 / 64$ bytes used, or $4 / 128$ if strided).
3. **Hardware Prefetchers:** Stream and stride prefetchers monitor consecutive L1/L2 misses (e.g., `addr`, `addr+64`, `addr+128`) and speculatively issue reads ahead of execution. Linked-list / pointer-chasing data structures (such as naive Python object graphs or unbalanced trees) defeat hardware prefetchers, exposing the full ~80–100 ns DRAM latency on every hop.
4. **Coherence Protocol (MESI / MOESI):** Every core's private L1/L2 caches maintain global coherence at **64-byte granularity**. A write to a single byte on Core 0 requires invalidating that entire 64-byte line in Core 1–63 via RFO (Read-For-Ownership), causing severe latency spikes if threads share adjacent variables (**false sharing**, detailed in [04 — C/C++ and Rust on the Hot Path](./04-c-cpp-and-rust-on-the-hot-path.md)).
5. **Non-Temporal Stores:** When copying gigabytes of weights or zeroing large buffers that will not be read immediately by the CPU, standard stores pollute L1/L2/L3 caches (Read-For-Ownership pulls the old 64B line from RAM just to overwrite it). Non-temporal instructions (`MOVNTDQ` / `VMOVNTDQ`) write directly to Write-Combining (WC) buffers and flush full 64B lines to RAM without cache pollution.

---

## 4. DDR5 RAM Physics & Bandwidth Calculations

DDR5 (Double Data Rate 5) transfers data on both the rising and falling edges of the clock. The marketed speed—e.g., **DDR5-4800** or **DDR5-6400**—is measured in **MT/s (Megatransfers per second)**, not MHz (the actual I/O clock is half: 2400 MHz or 3200 MHz).

### 4.1 Theoretical Peak Bandwidth Formula

Each DDR5 DIMM channel presents a **64-bit (8-byte) data bus** (internally split into two independent 32-bit sub-channels, improving concurrency for 64-byte cache-line requests):

$$\text{Peak Bandwidth} = \text{Transfer Rate (MT/s)} \times 8\text{ bytes/channel} \times N_{\text{channels}}$$

Concrete server configurations:
* **Single DDR5-4800 channel:** $4.8 \times 10^9 \times 8\text{ B} = \mathbf{38.4\text{ GB/s}}$.
* **8-channel Intel Xeon (DDR5-5600):** $5.6 \times 10^9 \times 8 \times 8 = \mathbf{358.4\text{ GB/s}}$ theoretical (~270–300 GB/s sustained STREAM Triad).
* **12-channel AMD EPYC Genoa (DDR5-4800):** $4.8 \times 10^9 \times 8 \times 12 = \mathbf{460.8\text{ GB/s}}$ theoretical (~360–390 GB/s sustained).

### 4.2 DRAM Row-Buffer Mechanics (Why Random Access Is Slow)

A DRAM chip is organized into **Channels $\rightarrow$ Ranks $\rightarrow$ Bank Groups $\rightarrow$ Banks $\rightarrow$ 2D Rows (Pages, typically 8–16 KB) $\times$ Columns**.
1. **Row-Buffer Hit:** The target row is already open in the bank's sense amplifiers. Only `CAS` (Column Address Strobe, `tCL`) is needed: **~14–20 ns**.
2. **Row-Buffer Miss (Closed/Conflict):** The active row must be written back (`PRECHARGE`, `tRP`), the new row opened (`ACTIVATE`, `tRCD`), and then column read (`tCL`): **~45–60 ns** DRAM-internal + memory controller queueing + PHY/uncore traversal = **70–100 ns** observed by a CPU core.

---

## 5. PCIe Gen4 / Gen5 Bandwidth & Protocol Overhead

Peripheral Component Interconnect Express (PCIe) is a **point-to-point, full-duplex serial interconnect** composed of $N$ bidirectional differential pairs called **lanes** (x1, x4, x8, x16).

### 5.1 GT/s, Encoding Overhead (`128b/130b`), and Raw Link Bandwidth

Raw signaling speed is expressed in **GT/s (Gigatransfers per second)**—the number of raw bits transmitted per lane per second, *including* physical-layer line-coding overhead.

* **PCIe Gen1 / Gen2** used `8b/10b` encoding (20% overhead: 10 bits sent for 8 bits of payload).
* **PCIe Gen3, Gen4, and Gen5** use **`128b/130b` encoding**: every 128 payload bits are prefixed with a 2-bit sync header (`01` for Data Block, `10` for Ordered Set) to ensure clock recovery and DC balance.
  $$\text{Line Coding Efficiency} = \frac{128}{130} \approx \mathbf{98.46\%} \quad (\text{1.538\% overhead})$$
* *(Note: PCIe Gen6 moves to PAM4 signaling at 64 GT/s with fixed 256-byte FLITs carrying 242B payload + 6B TLP/DLP + 8B CRC/FEC, ~94.5% FLIT efficiency, yielding ~121 GB/s unidirectional x16.)*

Exact unidirectional raw payload bandwidth per lane and x16 slot:

| PCIe Generation | Signaling Rate | Encoding | Unidirectional / Lane | Unidirectional x16 Theoretical | Typical Sustained x16 DMA Payload |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PCIe Gen3** | 8.0 GT/s | 128b/130b | $0.985\text{ GB/s}$ | $15.75\text{ GB/s}$ | ~12–13 GB/s |
| **PCIe Gen4** | 16.0 GT/s | 128b/130b | $1.969\text{ GB/s}$ | **$31.51\text{ GB/s}$** | **~25–28 GB/s** |
| **PCIe Gen5** | 32.0 GT/s | 128b/130b | $3.938\text{ GB/s}$ | **$63.02\text{ GB/s}$** | **~50–56 GB/s** |

Calculation verification for PCIe Gen5 x16:
$$\text{BW}_{\text{Gen5 x16}} = 16\text{ lanes} \times \frac{32 \times 10^9\text{ bits/s}}{8\text{ bits/byte}} \times \frac{128}{130} = 64 \times 0.984615 = \mathbf{63.015\text{ GB/s}}$$

### 5.2 Why Real DMA Bandwidth Is ~85–88% of Theoretical

Even after `128b/130b` encoding, PCIe transmits data wrapped in **TLPs (Transaction Layer Packets)** and **DLLPs (Data Link Layer Packets)**:
* Every TLP adds **12–16 bytes of TLP header** (3–4 DWORDs for 32-bit/64-bit addressing) + **2-byte Sequence Number** + **4-byte LCRC** + **2-byte framing** = **20–24 bytes of packet overhead**.
* **Max Payload Size (MPS):** Negotiated during BIOS/OS enumeration between Root Complex and GPU/NVMe (commonly **256 bytes** or **512 bytes**, sometimes capped at 128B on misconfigured platforms!).
  * At `MPS = 256B` with 24B packet overhead: $\frac{256}{256 + 24} = 91.4\%$. Combined with `128b/130b` ($0.9846$), peak wire efficiency is $0.914 \times 0.9846 = 90.0\% \implies 64 \times 0.90 = 57.6\text{ GB/s}$, and flow-control ACK DLLPs + root-complex turnaround drop sustained throughput to **~50–56 GB/s**.
  * At `MPS = 128B` (a common hidden BIOS misconfiguration): $\frac{128}{128 + 24} = 84.2\%$ before DLLPs/turnaround, silently cutting real H2D bandwidth down toward ~45 GB/s on Gen5 or ~21 GB/s on Gen4.
* **PCIe Switch / Root Complex Topology:** If two GPUs or an NVMe drive and a GPU share the same upstream PCIe switch port or Root Complex link, concurrent traffic contends for the same lanes. You can inspect negotiated link speed, width, and MPS via:
  ```bash
  sudo lspci -vvv -s 0000:c1:00.0 | grep -E "LnkCap:|LnkSta:|MaxPayload|MaxReadReq"
  ```

---

## 6. NVMe Storage Physics

NVMe (Non-Volatile Memory Express) attaches directly to PCIe lanes (typically **PCIe x4**, occasionally x8 in enterprise add-in cards), replacing legacy SATA/AHCI's single queue of 32 commands with up to **64,000 submission/completion queues**, each supporting **64,000 commands** in lock-free ring buffers mapped directly to CPU cores.

* **Bandwidth Ceiling:**
  * PCIe Gen4 x4 NVMe: Theoretical $7.88\text{ GB/s}$, real sequential read **~7.0–7.4 GB/s**.
  * PCIe Gen5 x4 NVMe: Theoretical $15.75\text{ GB/s}$, real sequential read **~13.8–14.5 GB/s**.
* **Latency Profile:**
  * NAND flash cell sensing + ECC (LDPC) decoding takes **~25–75 $\mu$s** for TLC NAND (and >100 $\mu$s for QLC); controller SLC write buffers return writes in **~10–15 $\mu$s**.
  * **Queue Depth (QD) Sensitivity:** At QD=1 with 4 KB random reads, a 50 $\mu$s drive delivers only $\frac{1}{50\times 10^{-6}} = 20,000\text{ IOPS} \approx \mathbf{80\text{ MB/s}}$—less than 1% of its 14 GB/s rating! Reaching peak sequential bandwidth requires large I/O sizes (128 KB–1 MB) and deep asynchronous queues.

---

## 7. Access Latency vs. Throughput & Little's Law

Engineers frequently conflate **latency** (time for one operation to complete, $L$) and **throughput/bandwidth** (rate of bytes transferred per second, $B$). A system can have terrible latency and massive bandwidth simultaneously (a truck full of NVMe drives driving down the highway).

**Little's Law** provides the exact mathematical bridge linking bandwidth, latency, and the **concurrency (bytes in flight)** required to saturate any hardware link:

$$\text{Concurrency (Bytes in Flight)} = \text{Bandwidth (Bytes/s)} \times \text{Latency (s)}$$

To achieve 100% of a bus or memory tier's bandwidth, the requester must keep at least $B \times L$ bytes simultaneously outstanding (pipelined) in hardware queues.

### Concrete Little's Law Calculations

1. **Why 1 CPU Core Cannot Saturate DDR5:**
   * Suppose 12-channel DDR5 delivers $B = 400\text{ GB/s}$ at $L = 90\text{ ns}$:
     $$\text{In-Flight Bytes} = 400 \times 10^9\text{ B/s} \times 90 \times 10^{-9}\text{ s} = \mathbf{36,000\text{ bytes}} = \mathbf{562.5\text{ cache lines (64B)}}$$
   * A single modern x86 core typically has only **10 to 16 Line Fill Buffers (LFBs / MSHRs)** tracking outstanding cache misses. Even if all 16 LFBs are saturated simultaneously, one core's maximum DDR5 bandwidth is bounded by:
     $$B_{\text{1 core}} \le \frac{16 \times 64\text{ bytes}}{90\text{ ns}} = \frac{1024\text{ B}}{90 \times 10^{-9}\text{ s}} \approx \mathbf{11.4\text{ GB/s}}$$
   * **Takeaway:** You need **~35+ active cores** (or hardware prefetchers + non-blocking streams across cores) to saturate a 400 GB/s DDR5 subsystem!

2. **Saturating PCIe Gen5 x4 NVMe:**
   * $B = 14\text{ GB/s}$, $L = 60\text{ }\mu\text{s}$:
     $$\text{In-Flight Bytes} = 14 \times 10^9 \times 60 \times 10^{-6} = \mathbf{840\text{ KB}}$$
   * With 4 KB reads, you need a Queue Depth of $\frac{840\text{ KB}}{4\text{ KB}} = \mathbf{210\text{ concurrent I/Os}}$. With 128 KB sequential reads, you only need $\text{QD} \ge \mathbf{7}$.

3. **Saturating PCIe Gen5 x16 GPU DMA:**
   * $B = 55\text{ GB/s}$, $L = 1.2\text{ }\mu\text{s}$:
     $$\text{In-Flight Bytes} = 55 \times 10^9 \times 1.2 \times 10^{-6} = \mathbf{66\text{ KB}}$$
   * Small per-token transfers (e.g., sending a 4-byte token ID or a 4 KB activation slice via synchronous `cudaMemcpy`) never fill the 66 KB pipeline—they are **100% latency-bound** (~1–5 $\mu$s per call regardless of Gen4 vs Gen5). Bulk transfers (>1–4 MB) are **bandwidth-bound**.

---

## 8. Execution Timeline: Cache Hit vs. DDR5 Miss vs. NVMe-to-GPU Transfer

The sequence diagram below contrasts the exact hardware interactions across three regimes: an L1d hit (~1 ns), a local DDR5 miss (~85 ns), and a cold NVMe-to-GPU weight load (~100 $\mu$s+ per chunk).

```mermaid
sequenceDiagram
    autonumber
    participant Core as "CPU Core"
    participant L1 as "L1d Cache (VIPT)"
    participant L3 as "L3 Shared Cache"
    participant IMC as "DDR5 Memory Controller"
    participant RC as "PCIe Root Complex"
    participant NVMe as "NVMe SSD"
    participant GPU as "GPU Copy Engine"

    Note over Core,L1: Case 1 - L1d Cache Hit (~1 ns)
    Core->>L1: Read 8B (VIPT index + TLB tag check)
    L1-->>Core: Return word in 4 cycles (~1 ns)

    Note over Core,IMC: Case 2 - L1/L2/L3 Miss to DDR5 (~85 ns)
    Core->>L1: Read 8B (Miss, allocate LFB entry)
    L1->>L3: Lookup 64B cache line (Miss, ~15 ns)
    L3->>IMC: Issue 64B read request
    IMC->>IMC: PRECHARGE + ACTIVATE + CAS (~55 ns)
    IMC-->>L1: Stream 64B line (critical word first)
    L1-->>Core: Unblock stalled instruction (~85 ns total)

    Note over Core,GPU: Case 3 - Cold NVMe Weight Load to GPU VRAM
    Core->>RC: Submit NVMe Read SQE via MMIO doorbell
    RC->>NVMe: PCIe TLP Read Request
    NVMe->>NVMe: Sense NAND flash + LDPC ECC (25-75 us)
    NVMe->>RC: DMA write 128 KB blocks into Host DDR5 Page Cache
    RC-->>Core: MSI-X Completion Interrupt / Poll CQE
    Core->>GPU: Enqueue cudaMemcpyAsync H2D DMA descriptor
    GPU->>RC: Issue PCIe Memory Read TLPs (MRRS=512B-4KB)
    RC->>IMC: Fetch pinned Host RAM lines
    IMC-->>GPU: Stream TLP Completions at ~26-54 GB/s into HBM
```

---

## 9. Direct LLM Inference Implications

### 9.1 Model Weight Loading & Autoscaling Cold Starts
Consider deploying a **Llama-3 70B** model in BF16 (**140 GB** of weights):
* **From Single PCIe Gen4 x4 NVMe (7.0 GB/s):** Minimum physics floor = $\frac{140\text{ GB}}{7.0\text{ GB/s}} = \mathbf{20.0\text{ seconds}}$ just for disk reads (before tensor deserialization or H2D copies!).
* **From RAID-0 4x Gen5 NVMe (~50 GB/s) or Warm Linux Page Cache:** Disk read drops to **0–2.8 seconds**.
* **Over PCIe Gen4 x16 to GPU (~26 GB/s) vs. Gen5 x16 (~54 GB/s):** H2D transfer takes **5.4 seconds** on Gen4 vs **2.6 seconds** on Gen5—assuming **pinned memory** is used (unpinned `cudaMemcpy` stalls around ~10–14 GB/s, inflating H2D to **10–14 seconds**).
* For serverless/autoscaling GPU pods (KEDA / Ray Serve), PCIe generation, NVMe striping, and warm host caching directly dictate whether cold-start SLA is 5 seconds or 45 seconds.

### 9.2 H2D / D2H Transfers in KV-Cache Offloading & MoE
Suppose a long-context serving engine offloads KV Cache blocks for a 70B GQA model (8 KV heads, 80 layers, `head_dim=128`, BF16 $\implies$ **327.68 KB per token**):
* Offloading or restoring a **32,768-token context** moves $32,768 \times 327.68\text{ KB} = \mathbf{10.74\text{ GB}}$.
* Over **PCIe Gen4 x16 (~26 GB/s)**: takes **413 ms** (almost half a second of TTFT!).
* Over **PCIe Gen5 x16 (~54 GB/s)**: takes **199 ms**.
* If the engine accidentally uses unpinned host buffers or suffers cross-NUMA hops, effective bandwidth drops to **~10 GB/s**, turning that restore into **1.07 seconds** and stalling the GPU decode loop if not overlapped on an asynchronous CUDA copy stream.
* Similarly, **CPU-offloaded MoE** (streaming active experts from DDR5 to GPU every token) is strictly gated by PCIe bandwidth: streaming just 10 GB of active expert weights per token over PCIe Gen5 x16 limits decode speed to **<5.4 tokens/s** ($\sim 185\text{ ms/token}$), completely independent of how fast the GPU Tensor Cores are.

### 9.3 CPU-Side Hot Paths (Tokenization, Scheduling, Logit Processing)
A serving engine's Python/C++/Rust control loop runs continuously between GPU iterations:
* Vocabulary masking (e.g., 128,256-token vocabulary bitmask/logits in FP32 = **512 KB** per request) fits comfortably inside a single core's **L2/L3 cache** (512 KB–32 MB) if stored contiguously.
* However, a poorly designed grammar-constrained decoder that chases scattered pointer trees across >50 MB of heap objects falls out of L3 into **80–100 ns DDR5 misses**, multiplying per-step CPU latency by 10x–50x and creating bubbles on the GPU timeline.

---

## Navigation

* **Up:** [Stage 1 Index](./README.md)
* **Next:** [02 — Hardware Execution Models](./02-hardware-execution-models.md)
