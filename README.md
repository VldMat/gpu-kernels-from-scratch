# GPU Kernels from Scratch — CUDA C++ & Triton

A from-scratch path through GPU programming: from a first vector add to tiled
online-softmax attention, written in CUDA C++ (plus a start in Triton). Each step
isolates one hardware idea and is checked against a reference result.

> **Status:** learning project, actively extended. Developed and tested on a local
> NVIDIA laptop GPU with the CUDA toolkit.

---

## Kernels

| # | Kernel | File | Idea it demonstrates | Correctness check |
|---|--------|------|----------------------|-------------------|
| 01 | Vector add | `vector add/vadd.cu` | Execution model: grid, blocks, threads; host ↔ device memory | CPU result |
| 02 | SAXPY | `saxpy/saxpy.cu` | Scalars passed by value; read-modify-write | CPU result |
| 03 | ReLU | `relu/relu.cu` | Element-wise kernel, write-only output | CPU result |
| 04 | Block reduction | `reduction/reduction.cu` | Shared-memory tree reduction, `__syncthreads()` | Exact sum |
| 05 | Warp-shuffle reduction | `reduction/warp_shuffle.cu` | Reduce inside a warp with `__shfl_down_sync`, then across warps | Exact sum |
| 06 | Softmax | `softmax/softmax.cu` | Two block reductions (max, then sum) for a numerically stable row softmax | Row sums = 1 |
| 07 | Online softmax | `online_softmax/online_softmax.cu` | Merge (max, sum) pairs in a single parallel reduction | Row sums = 1 |
| 08 | Naive matmul | `matmul/matmul.cu` | 2-D indexing; every operand read from global memory | Analytic result |
| 09 | Shared-memory tiled matmul | `matmul/matmul_tiled.cu` | Load 16×16 tiles into shared memory and reuse them | Analytic result |
| 10 | Register-tiled matmul | `matmul/matmul_registertiled.cu` | Each thread computes 8 outputs from registers (64×64 block tile, BK = 8) | cuBLAS |
| 11 | Attention (online softmax) | `attention/attention.cu` | One thread per query row; single pass over keys with running max / sum | CPU attention |
| 12 | Tiled attention | `attention/attention_tiled.cu` | K/V tiles staged in shared memory — the FlashAttention forward idea | CPU attention |
| T1 | Triton vector add / ReLU | `triton/` | Block-level programming with masked loads/stores | PyTorch |

---

## Online softmax — why one pass is enough

Softmax over a row of scores `x₁ … x_N` is

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

**Stability.** `exp` overflows in FP32 for inputs above ~88, so every practical
implementation subtracts the row maximum `m = max_j x_j` first. This changes nothing
mathematically (the factor `exp(−m)` cancels between numerator and denominator) but keeps
every exponent ≤ 0. The catch: the naive version needs **three passes** over the data —
one to find `m`, one to compute the sum `l = Σ exp(x_j − m)`, one to normalise.

**The online trick.** Keep a running maximum `m` and a running sum `l` that is always
expressed *relative to the current maximum*. When a new value `x` arrives:

```
m_new = max(m, x)
l_new = l · exp(m − m_new) + exp(x − m_new)
```

If `x` is not a new maximum, `m_new = m`, the correction factor is 1, and we just add
`exp(x − m)`. If `x` *is* a new maximum, everything accumulated so far was scaled by
`exp(−m)` but should now be scaled by `exp(−m_new)`; multiplying by `exp(m − m_new)` (≤ 1)
fixes all past terms at once. Max and sum are computed in a **single pass**.

**Why it parallelises.** Two partial results `(m₁, l₁)` and `(m₂, l₂)` from disjoint
chunks combine with the same rule:

```
m = max(m₁, m₂)
l = l₁ · exp(m₁ − m) + l₂ · exp(m₂ − m)
```

The operation is associative, so it can be used in a tree reduction — exactly what
`online_softmax/online_softmax.cu` does in shared memory, merging `(max, sum)` pairs instead
of plain sums.

**Why it matters for attention.** Attention computes `softmax(q·Kᵀ/√d) · V` for each
query. The same rescaling applies to the output accumulator `o`, a weighted sum of value
rows:

```
o_new = o · exp(m − m_new) + exp(s − m_new) · v
```

where `s` is the new score and `v` the matching value row. After the last key, `o / l` is
the exact attention output. The full N × N score matrix is never written to memory: each
query keeps only `m`, `l` and a `d`-sized accumulator. `attention/attention.cu` applies
this one key at a time; `attention/attention_tiled.cu` additionally stages blocks of K and V
in shared memory so each tile is read from global memory once per block of queries instead
of once per query — the core idea of the FlashAttention forward pass.

---

## Running it

Requires an NVIDIA GPU and the CUDA toolkit:

```bash
nvcc -O3 attention/attention_tiled.cu -o attention_tiled
./attention_tiled
```

Each program prints its kernel time (CUDA events) and its maximum error against the
reference. The register-tiled matmul needs cuBLAS: add `-lcublas`.

Triton kernels (PyTorch + Triton, NVIDIA GPU):

```bash
python triton/vector_add.py
```

---

## Benchmarks

Benchmarks against cuBLAS and PyTorch (`scaled_dot_product_attention`) across problem
sizes are in progress and will be added here.

---

## Known limitations

- FP32 only; no tensor cores (no WMMA / MMA), no FP16 / BF16.
- `attention_tiled.cu` assumes N is a multiple of the tile size (64) and `d = 64` fixed at compile time; one query row per thread, so per-thread register use is high.
- Attention rescales after every key; FlashAttention computes a tile of scores and rescales once per tile.
- Tile loads in `attention_tiled.cu` are not yet coalesced (each thread loads a whole row).
- Matmul kernels assume N is a multiple of the tile sizes.

## Next steps

- [ ] Benchmarks against cuBLAS and PyTorch across sizes
- [ ] Per-tile rescaling and coalesced K/V loads in tiled attention
- [ ] FP16 + tensor cores (WMMA)
- [ ] Profile matmul / attention with Nsight Compute (memory throughput, occupancy)
- [ ] Softmax and matmul in Triton; compare with the CUDA versions
- [ ] Causal masking; backward pass

---

## License

MIT
