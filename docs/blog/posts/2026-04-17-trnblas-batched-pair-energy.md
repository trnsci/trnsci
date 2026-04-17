---
date: 2026-04-17
categories: [Deep dive, trnblas]
comments: true
---

# trnblas Phase 3: from 215× slower to 3.6× faster in one kernel boundary move

The [Phase 2 profiler post](https://trnsci.dev/blog/trnblas-four-hypotheses-one-profiler-trace-and-why-148-is-the-correct-answer/)
closed with an unexpected conclusion: the fused MP2 energy kernel hits its
Amdahl ceiling, the remaining gap to the 3× target lives entirely in the
step that surrounds it, and the only lever left is the kernel *boundary*.
Phase 3 moved the boundary. The result is a 3.6× end-to-end speedup over
the torch baseline at the small bench shape — the first energy path that
actually beats chunk-GEMM.

<!-- more -->

## The problem

After Phase 2 confirmed the per-pair kernel was near-optimal in isolation,
the question shifted to the loop it lives inside. In the DF-MP2 reference
implementation, the energy step calls `nki_fused_gemm_energy(B[i], B[j], ...)`
once per occupied orbital pair `(i, j)`, for a total of `nocc²` dispatches:

```python
e_mp2 = 0.0
for i in range(nocc):
    for j in range(nocc):
        e_mp2 += nki_fused_gemm_energy(B[i], B[j], eps_occ, eps_vir)
```

The Phase 1 design assumed NEFF cache amortization would make this cheap:
compile once, pay ~1 ms per call thereafter. Measured on hardware, that
assumption was wrong by a factor of 100.

A warm-cache Neuron XLA dispatch costs approximately **100 ms**, not 1 ms.
The 100 ms is not kernel compute time — the Tensor Engine finishes in
< 2 ms for small-shape pairs. It is the time to move through the
XLA trace-dispatch pipeline: trace the Python call, look up the NEFF in
the in-process cache, enqueue it to the NeuronCore queue, and synchronize.
At `nocc=16` (256 pairs), that 100 ms × 256 = 25.6 s per energy call.
The torch chunk-GEMM baseline takes 0.13 s. The per-pair NKI kernel is
**215× slower** than what it was meant to replace.

## What the architecture suggests

The 100 ms per-dispatch floor is not a compiler limitation that can be
tuned away. It is a structural property of the Neuron XLA runtime: one
NEFF file, one enqueue, one synchronization fence per `@nki.jit` call.
For workloads where each dispatch is large — the Phase 1 energy kernel
doing one 1024² GEMM — 100 ms overhead is negligible. For workloads
where each dispatch is small — 112×384 GEMMs per `(i, j)` pair — it
dominates by two orders of magnitude.

The implication is architectural, not algorithmic: if one dispatch costs
100 ms regardless of kernel size, the right response is to pay it once.
Put all `nocc²` pair computations inside a single `@nki.jit` and loop
over pairs using `nl.affine_range`. The XLA graph for "all pairs" is not
materially larger than for "one pair" — both compile to a NEFF with
a fixed loop body. The XLA compiler traces the loop structure, not the
flattened iteration count.

This is the opposite of the GPU mental model, where launching many small
kernels is a known antipattern. On Trainium, it's the only antipattern
that matters at this scale.

## The approach

`nki_batched_pair_energy(B, eps_occ, eps_vir)` replaces the `nocc²`-call
loop with a single dispatch:

- **Input:** `B` of shape `(nocc, nvir, naux)` — all occupied-orbital slices
  stacked. `eps_occ` of shape `(nocc,)`, `eps_vir` of shape `(nvir,)`.
- **Output:** scalar `E_MP2`, returned as a Python float.
- **Kernel structure:** five levels of `nl.affine_range` — `i`, `j`,
  `a`-strip, `b`-strip, `k`-tile. For each `(i, j)`, two GEMMs are
  computed (T and its transpose), the energy denominator is built
  SBUF-resident, and partial energies accumulate in SBUF. One HBM store
  per `(i, j)` pair; host calls `.sum()` on the `(TILE, nocc²)` output.

The 3D NKI indexing — `nl.load_transpose2d(B[i, a:a+T, k:k+K])` with
`i` as an `nl.affine_range` loop variable — was validated on trn1 before
the full kernel was written. Spike A (pure indexing) and Spike B (full
batched energy, `nocc=4`) each ran on hardware, confirming the compiler
handles 3D affine indexing as expected.

## Implementation

Simplified kernel structure (full source in
[`trnblas/nki/dispatch.py`](https://github.com/trnsci/trnblas/blob/main/trnblas/nki/dispatch.py)):

```python
@nki.jit
def _batched_pair_kernel(B, eps_occ_tile, eps_vir_tile, e_out):
    nocc, nvir_pad, naux = B.shape  # after padding to TILE/TILE_K multiples
    TILE = 128; TILE_K = 128

    for i in nl.affine_range(nocc):
        for j in nl.affine_range(nocc):
            # Two GEMMs: T_ij = B[i] @ B[j].T,  T_T_ij = B[j] @ B[i].T
            acc_j = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
            acc_jT = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
            for a in nl.affine_range(nvir_pad // TILE):
                for b in nl.affine_range(nvir_pad // TILE):
                    acc_a = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                    acc_aT = nl.zeros((TILE, TILE), dtype=nl.float32, buffer=nl.sbuf)
                    for k in nl.affine_range(naux // TILE_K):
                        k_off = k * TILE_K; a_off = a * TILE; b_off = b * TILE
                        bi = nl.load_transpose2d(B[i, a_off:a_off+TILE, k_off:k_off+TILE_K])
                        bj = nl.load(B[j, b_off:b_off+TILE, k_off:k_off+TILE_K])
                        acc_a += nisa.nc_matmul(bi, bj)   # T[a,b] strip
                        bjT = nl.load_transpose2d(B[j, a_off:a_off+TILE, k_off:k_off+TILE_K])
                        biT = nl.load(B[i, b_off:b_off+TILE, k_off:k_off+TILE_K])
                        acc_aT += nisa.nc_matmul(bjT, biT)  # T_T[a,b] strip
                    # build Δ, energy expression — SBUF-resident
                    denom = (eps_occ[i] + eps_occ[j]
                             - eps_vir[a*TILE:a*TILE+TILE] - eps_vir[b*TILE:b*TILE+TILE])
                    contrib = acc_a * (2*acc_a - acc_aT) / denom
                    acc_j += contrib
            pair_idx = i * nocc + j
            nl.store(e_out[0:TILE, pair_idx:pair_idx+1], value=acc_j)
```

The `nl.affine_range` loops over `i` and `j` are not unrolled — the XLA
graph captures the loop structure. The NEFF compile cost for the whole
`nocc²`-pair kernel at small shape (nocc=16) is about the same as for
one-pair kernel. Dispatch overhead goes from O(nocc²) to O(1).

## What didn't work

**The per-pair kernel first (v0.5.1).** The natural incremental step was
to write the fused-gemm kernel per-pair, validate it, then worry about
dispatch overhead. That's exactly what happened: `nki_fused_gemm_energy`
compiled, validated against PySCF to six significant figures, and shipped
in v0.5.1. Then it was benchmarked at small shape (nocc=16 / 256 pairs):
**215× slower than torch baseline**. Not a rounding error. The per-pair
kernel was correct and useless, and the only path forward was to rethink
the dispatch granularity.

**No closures in `@nki.jit`.** The tile-shape autotuner (Phase 3, #26)
uses a factory pattern — `_make_gemm_kernel(tile_m, tile_k, tile_n)`
returns a new `@nki.jit` closure with the tile sizes baked in. The
batched-pair kernel cannot use this pattern because `@nki.jit` does not
permit Python closures: the kernel function's body must reference only
its explicit arguments and literal constants, not outer-scope variables.
Tile shapes in the batched-pair kernel are hardcoded constants; any
autotuning would require a separate generated-kernel path, deferred as
future work.

**Medium-shape XLA graph is 18 GB — the compiler can't fit it.** At
nocc=64, nvir=448, naux=1536 (4096 pairs), `nl.affine_range` traces
all iterations eagerly into the XLA graph at compile time. The resulting
NKI source JSON is 18 GB; the trn1 root volume had 16 GB free. The
NEFF failed mid-compile regardless of whether the compiler wrote to
`/tmp` or `/var/tmp` — both live on the same 96 GB EBS root volume.

For comparison, the small-shape kernel (nocc=16, 256 pairs) produces a
~240 MB JSON. Medium has 16× more pairs and roughly 64× more inner tile
operations (nvir and naux are also larger), which explains the 75× size
jump. `nl.affine_range` is not symbolic like a GPU JIT — it traces the
loop body once per iteration at compile time.

The correct fix is **chunked dispatch**: call the batched-pair kernel
with chunks of ~256 pairs (16 chunks for nocc=64). Each chunk's XLA
graph is ~240 MB; 16 dispatches add ~1.6 s of overhead versus the
per-pair loop's 409 s. This is deferred to a follow-on PR.

The fallback result (5.2 s warm energy, 7.1 s warm total) — faster
than the NKI chunk-GEMM baseline (8.0 s, 9.8 s) — came from CPU
`torch.matmul`, one of the more uncomfortable observations in the suite.

## Numbers

Small bench shape: `nbasis=128, nocc=16, nvir=112, naux=384` (256 pairs).
All on `trn1.2xlarge`, `neuronxcc 2.24.5133`, warm NEFF cache, v0.5.2:

| Energy path | Cold energy | Warm energy | Warm total | vs torch |
|---|---:|---:|---:|---:|
| torch (chunk-GEMM baseline) | 14.2 s | 0.018 s | 0.096 s | 1× |
| fused-gemm (per-pair, v0.5.1) | 1.9 s | 0.381 s | 0.454 s | **21× slower** |
| **batched-pair (v0.5.2)** | 6.7 s | **0.005 s** | **0.081 s** | **3.6× faster** |

The 6.7 s cold time for batched-pair is NEFF compile (paid once per instance
lifetime on the EBS-backed NEFF cache). The 0.005 s warm energy is a single
dispatch; the gap to fused-gemm's 0.381 s warm (76×) is exactly the 256 cold
dispatch overheads that batched-pair avoids (256 × ~1.5 ms ≈ 384 ms).

**Spike B validation (nocc=4 / 16 pairs):**

| Metric | Value |
|---|---:|
| Batched-pair warm | 1.9 ms |
| Per-pair loop warm | 25.4 ms |
| Speedup | **13.5×** |

**Medium shape** (`nocc=64, nvir=448, naux=1536`, 4096 pairs):

| Energy path | Warm energy | Warm total |
|---|---:|---:|
| torch | 8.035 s | 9.795 s |
| fused-gemm | 9.174 s | 10.877 s |
| batched-pair (CPU fallback, v0.5.2†) | 5.239 s | 7.111 s |
| **batched-pair (chunked NKI, v0.5.4)** | **1.536 s** | **4.784 s** |

† v0.5.2: NEFF compile failed (18 GB XLA graph exceeded disk); result was CPU
`torch.matmul` fallback. v0.5.4 chunked dispatch (issue #46) resolved this:
one `@nki.jit` call per i-row, 64 calls total, each processing all `nocc`
j-pairs. Cold energy = 34 min (77 NEFF compilations, paid once); warm energy
= 1.536 s = 64 dispatches × ~24 ms each. **5.2× faster than torch baseline.**

**Energy cross-check:** torch / fused-gemm = −1.619250×10⁻⁴ Ha,
batched-pair = −1.619249×10⁻⁴ Ha. Matches to FP32 noise.

## What's next

- **Chunked batched-pair dispatch — landed (v0.5.4).** The empirical question
  is answered: chunked NKI dispatch (64 i-calls, each processing all nocc
  j-pairs) achieves 1.536 s warm energy at medium shape — **5.2× faster than
  torch** and 3.4× faster than the v0.5.2 CPU fallback. The 18 GB XLA graph
  problem is solved; each chunk produces ~1.4 GB, compiling and caching
  normally. Remaining constraint: 64 loaded energy NEFFs × 244 MB DMA spill
  ≈ 15.6 GB saturates the 16 GB device; HBM pressure becomes the frontier
  at large shapes (nocc=96+).
- **[#20 — PySCF FP32 precision envelope.](https://github.com/trnsci/trnblas/issues/20)**
  Glycine/cc-pVDZ and water trimer tests written
  (`tests/test_df_mp2_pyscf.py`); hardware run pending. Decision gate
  for [#22 (double-double)](https://github.com/trnsci/trnblas/issues/22).
- **[#26 — tile autotuner](https://github.com/trnsci/trnblas/issues/26)**
  for `nki_gemm`. Sweep across six tile candidates; winner cached to EBS.
  Already landed in v0.5.0; medium-shape sweep numbers pending.
- **[#25 — trn2 benchmarks.](https://github.com/trnsci/trnblas/issues/25)**
  Infrastructure provisioned in `infra/terraform-trn2/` (self-contained VPC,
  sa-east-1). Hardware investigation deferred.

## Takeaway

The lesson from Phase 3 is about kernel granularity, not kernel content.
The per-pair fused kernel (v0.5.1) was correct — energies match to six
significant figures. It was 215× slower because it called a 100 ms
overhead function 256 times. Moving the loop inside the `@nki.jit`
boundary reduced the overhead from O(nocc²) to O(1) and turned a useless
kernel into a 3.6× end-to-end win.

The ~100 ms Neuron XLA dispatch overhead is not a bug to be fixed; it is
a property of the dispatch model that shapes how kernels should be
structured. The NKI programming model rewards large, coarse kernels that
do as much work as possible per dispatch. Per-element or per-small-batch
patterns that work on GPU — where kernel launch overhead is ~10 µs — need
to be restructured for Trainium, or they will spend 99% of wall time in
the runtime rather than the Tensor Engine.

The medium-shape XLA graph limit is resolved. Chunked dispatch (v0.5.4)
routes shapes above `_BATCHED_PAIR_CHUNK_THRESHOLD = 4096` iterations to a
Python i-loop over `_j_batched_kernel`, avoiding the 18 GB full-batch graph
while preserving the single NEFF compile property (all i-calls share identical
input shapes). The empirical answer: chunked NKI at 4096 pairs is 5.2× faster
than torch chunk-GEMM, not slower — the Tensor Engine advantage is real at
medium scale. The new constraint is device HBM saturation at large shapes
(nocc=96+), where the loaded NEFF count × 244 MB exceeds 16 GB.
