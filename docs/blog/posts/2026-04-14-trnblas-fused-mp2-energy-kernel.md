---
date: 2026-04-14
categories: [Deep dive, trnblas]
comments: true
---

# trnblas: fusing DF-MP2 energy reduction into one NKI kernel

trnblas v0.4.0 shipped hardware-validated NKI kernels for GEMM, SYRK,
and a fused MP2 energy reduction on trn1, with end-to-end density-fitted
MP2 matching PySCF to 10 µHa (1×10⁻⁵ Ha) on H₂O, CH₄, and NH₃ at
cc-pVDZ. The interesting story isn't the GEMM. It's the fused energy
kernel — a single NKI pass that holds the contraction, the orbital-
denominator division, and the scalar sum-reduction SBUF-resident, and
how the choice to build it looks nothing like a cuBLAS port. This post
is for anyone curious about what Trainium's whole-program compilation
model affords that per-call GEMM libraries cannot express.

<!-- more -->

## The problem

Density-fitted MP2 is a second-order perturbation-theory method used
across quantum chemistry for computing correlation energies on systems
too large for coupled-cluster. Its computational signature is dominated
by a single expression, evaluated per pair of occupied orbitals `(i, j)`
and summed over the virtual orbital pair `(a, b)`:

```
E_MP2 = Σ_{i,j,a,b} T[i,j,a,b] · (2·T[i,j,a,b] − T[i,j,b,a]) / Δ[i,j,a,b]
```

where `T[i,j,a,b] = Σ_P B[i,a,P]·B[j,b,P]` (a three-center tensor
contraction over the auxiliary basis) and `Δ[i,j,a,b] = ε_i + ε_j − ε_a
− ε_b` is the orbital-energy denominator. The tensor `T` at realistic
basis sizes is dense and large — for a medium benchmark shape
(nocc=64, nvir=448), the full per-chunk `T` is 16.6 GB in fp32.

The cuBLAS / PyTorch-on-GPU mental model for this pattern is a chain
of per-op kernel calls: one GEMM to produce `T`, three elementwise
kernels for `2T − Tᵀ` and the division, one reduction. Each call
writes its intermediate to HBM before the next one reads it back.
For the medium shape, that's ~50 GB of intermediate HBM traffic per
chunk — dominated not by arithmetic but by memory round-trips.
Profiled on trn1 before any fusion, the energy-reduction step takes
8.03 s of a 9.79 s total wall time, ~82 % of the workload.

cuBLAS cannot collapse this into one call. Its primitive is
`gemmStridedBatched` and similar — matrix products. The multiply-
subtract-divide-sum chain sits *between* cuBLAS calls, in a
separately written CUDA reduction kernel a user ships alongside. A
cuBLAS-shaped library has no natural place for it.

## What the architecture suggests

NKI kernels compile to NEFF — Neuron Executable File Format —
which expresses the whole kernel as a single scheduled program
across four engines (Tensor, Vector, Scalar, GpSimd), with explicit
SBUF (on-chip 128-partition scratchpad) and PSUM (32-bit tensor
accumulator) as named memories. Three properties of this model
point at a different design than cuBLAS+reduction:

**SBUF-resident intermediates cost nothing extra.** Once a tile
lands in SBUF, every engine that reads it reads from on-chip memory
at orders of magnitude higher bandwidth than HBM. An expression
chain `T * (2T − Tᵀ) / Δ` that would ping-pong through HBM on a
per-op model sits inside one SBUF tile for its entire lifetime if
the kernel is written as one `@nki.jit` function.

**Partition dim is the native parallel axis, not a thread block.**
NKI's partition dim is 128 physical lanes. A single `nl.subtract(a,
b)` on a `(128, N)` tile runs across all 128 partitions
simultaneously; there is no intra-tile thread-block boundary to
cross. Reductions have an asymmetry: the free dim is reducible,
the partition dim is not (directly). This shapes the tile layout
more than anything else.

**NEFF cache amortizes compile across invocations.** Every
`@nki.jit` function with the same trace-time shape compiles once,
then every subsequent invocation hits the cache and pays only
dispatch + execution. For a pair-energy loop invoking the same
kernel `nocc²` times (4 096 to 9 216 pairs at the bench shapes),
the compile cost divides across thousands of calls. The NEFF cache
is what makes "one kernel per pair" cheap in the first place.

These three combined suggest the shape: a single kernel that takes
`T_chunk` plus the three ε vectors, iterates `(i, j)` internally,
materializes each tile's chunk of the expression SBUF-resident,
reduces along the free dim, and returns per-partition partials that
the caller sums host-side.

## The approach

trnblas's fused MP2 kernel —
[`_mp2_energy_kernel`](https://github.com/trnsci/trnblas/blob/main/trnblas/nki/dispatch.py) —
is that design made concrete:

- **One `@nki.jit` function, one NEFF.** Signature takes `T_flat`,
  the three ε vectors, returns a `(P_TILE, IC, NOCC)` partial.
- **Tile layout fixes partition-dim alignment through the whole
  computation.** The virtual-orbital index `a` sits on the partition
  dim; `b` sits on the free dim. `T` and its transpose both load
  with the same partition semantics. Nothing reshapes across
  partition axes mid-kernel.
- **`Δ` is built SBUF-resident per strip** from the three ε vectors,
  never materialized in HBM. This is the step that cuBLAS can't
  express at all — `Δ` is not a matrix product, just an outer-sum
  of three small vectors.
- **Free-dim sum reduces each strip.** The result is `(P_TILE, 1)`
  per strip. Strips across the virtual-orbital range accumulate
  into an SBUF `(P_TILE, NSTRIP)` tile and reduce to `(P_TILE, 1)`
  after the strip loop. One HBM store per `(i, j)` pair — not per
  strip.
- **Partition-axis reduce happens host-side.** NKI does not support
  reduction along the partition dim, so the caller does one final
  `.sum()` on the `(P_TILE, IC, NOCC)` partial. The partial is tiny
  (≤ 258 KB at the large bench shape), and host-side reduction is
  noise.

A deliberate tradeoff: a true 3D-batched NKI kernel for the three-
center contraction is deferred to Phase 3. v0.4.0's `batched_gemm`
is a host-side loop around the 2D NKI GEMM, one slice per iteration.
Every slice after the first hits the NEFF cache, so per-slice cost
is HBM transfer plus dispatch. Tradeoff stated explicitly: the
Phase 1 scope focuses on single-call kernels with measurable
correctness; cross-call kernel fusion lands where a measured
benchmark justifies it.

## Implementation

The core shape, simplified (full source linked above):

```python
@nki.jit
def _mp2_energy_kernel(T_flat, eps_occ_chunk, eps_occ_full,
                       eps_vir_col, eps_vir_row):
    NVIR = eps_vir_row.shape[1]
    IC = eps_occ_chunk.shape[1]
    NOCC = eps_occ_full.shape[1]
    P_TILE = min(NVIR, 128)
    while NVIR % P_TILE != 0:
        P_TILE -= 1
    NSTRIP = NVIR // P_TILE

    e_partial = nl.ndarray((P_TILE, IC, NOCC),
                           dtype=nl.float32,
                           buffer=nl.shared_hbm)
    ev_row = nl.load(eps_vir_row[0:1, 0:NVIR])

    for i in nl.affine_range(IC):
        eo_i = nl.load(eps_occ_chunk[0:1, i:i+1])
        for j in nl.affine_range(NOCC):
            eo_j = nl.load(eps_occ_full[0:1, j:j+1])
            eo_sum = nl.add(eo_i, eo_j)
            acc_rows = nl.zeros((P_TILE, NSTRIP),
                                dtype=nl.float32, buffer=nl.sbuf)
            for s in nl.affine_range(NSTRIP):
                a_off = s * P_TILE
                t = nl.load(T_flat[i*NVIR + a_off : i*NVIR + a_off + P_TILE,
                                    j*NVIR : (j+1)*NVIR])
                t_T = nl.load_transpose2d(T_flat[
                    i*NVIR : (i+1)*NVIR,
                    j*NVIR + a_off : j*NVIR + a_off + P_TILE])
                ev_col = nl.load(eps_vir_col[a_off:a_off+P_TILE, 0:1])

                # Δ built SBUF-resident: all three eps operands
                # lifted to (P_TILE, NVIR) for partition-matched arith.
                eo_sum_bc = nl.broadcast_to(eo_sum, (P_TILE, NVIR))
                ev_col_bc = nl.broadcast_to(ev_col, (P_TILE, NVIR))
                ev_row_bc = nl.broadcast_to(ev_row, (P_TILE, NVIR))
                denom = nl.subtract(nl.subtract(eo_sum_bc, ev_col_bc),
                                    ev_row_bc)

                # The whole expression chain, SBUF-resident.
                term = nl.multiply(
                    nl.multiply(t, nl.subtract(nl.multiply(t, 2.0), t_T)),
                    nl.reciprocal(denom),
                )
                strip_partial = nl.sum(term, axis=1, keepdims=True)
                acc_rows[0:P_TILE, s:s+1] = strip_partial

            acc_row = nl.sum(acc_rows, axis=1, keepdims=True)
            nl.store(e_partial[0:P_TILE, i:i+1, j:j+1], value=acc_row)

    return e_partial
```

Four architectural moves sit in those lines: the `(P_TILE, NVIR)`
tile as the unit of work, `Δ` materialized SBUF-resident from three
loads rather than passed in from HBM, the full multiply-subtract-
multiply-reciprocal-multiply chain running before any HBM traffic,
and the per-pair HBM store at the end — one write for the whole
pair's contribution rather than three intermediates and a result.

## What didn't work

The honest record, in four parts.

**The `examples/df_mp2.py` revert.** The first post-M2 attempt
([#15](https://github.com/trnsci/trnblas/issues/15)) flipped the
DF-MP2 example's default to `nki_mp2_energy`. The 1.48× speedup on
the energy step was real but below the RFC's 3× minimum target,
and the post-flip CHANGELOG framing over-claimed. The example's
default was reverted back to the torch reduction, `--fused-energy`
added as an opt-in flag, and the CHANGELOG rewritten with the
measured numbers honestly. The kernel works; the speedup isn't yet
large enough to justify being the default.

**NKI 0.3.0 partition-broadcast strictness.** Neuron SDK 2.29's
MLIR verifier is stricter than 2.28 on tensor-tensor arithmetic
with mismatched partition dims. The M1 kernel built `Δ` with two
subtracts of shape `(1,1) − (P_TILE, 1)` — rejected wholesale in
0.3.0. The 5 MP2 test cases were re-skipped for a release before
the fix landed (commit
[`c1769c6`](https://github.com/trnsci/trnblas/commit/c1769c6)):
lift all three eps operands to `(P_TILE, NVIR)` via
`nl.broadcast_to` before subtracting. The simulator gate catches
four of the five 0.3.0 breaking-change classes; this one is MLIR-
verifier-level and still requires hardware.

**The v0.4.x "NKI" numbers were silent torch.matmul fallback.**
The SSM runners in v0.4.0 through v0.4.2 invoked the Neuron venv's
python without prepending its `bin/` to `$PATH`. `torch_neuronx`
calls `libneuronpjrt-path` at import to locate the PJRT plugin;
that binary lives in the venv's `bin/`, so it failed with
`FileNotFoundError`, the `_nki_*_impl` try/except wrappers swallowed
it, and every reported "trn1 NKI" warm number was actually trn1's
8-vCPU Xeon running `torch.matmul`. Correctness tests still passed
— torch gives the same answer as the kernel — but perf attribution
was wrong for three releases. v0.4.3 published the full retraction,
added `NkiFallbackWarning` so future regressions surface without
requiring `TRNBLAS_REQUIRE_NKI=1`, and added
`tests/test_nki_really_runs.py` as an anti-regression gate.

**The post-M2 perf-tuning rewrite didn't help.** A follow-up PR
([#32](https://github.com/trnsci/trnblas/pull/32)) hoisted the
pair-invariant part of `Δ` out of the `(i, j)` loop, collapsing
denom construction from 5 scheduled ops per iteration to 1 plus a
pre-loop precompute. Measured result: 1.48× → 1.50× at medium,
1.47× → 1.49× at large. Within noise. The NEFF compiler was
already doing this work on trnblas's behalf; the explicit hoist was
a no-op. The profiler investigation
([#33](https://github.com/trnsci/trnblas/issues/33)) meant to
diagnose the remaining gap hit its own blocker: the 2.29 DLAMI's
`neuron-profile show-session` tool rejects the trace format its
own `inspect` command produces, and `view --disable-ui` requires
InfluxDB the DLAMI doesn't pre-install. The remaining hypothesis
(a cross-pair HBM-store fence) is queued as
[#35](https://github.com/trnsci/trnblas/issues/35); until it's
tested, the gap is documented, not solved.

## Numbers

All on `trn1.2xlarge`, `neuronxcc 2.24.5133`, warm NEFF cache,
v0.4.3-measured under real NKI dispatch.

**Per-call kernel timings:**

| Op                  | Shape              | Warm    |
|---------------------|--------------------|--------:|
| NKI GEMM            | 512 × 512 × 512    | 1.3 ms  |
| NKI GEMM            | 1024 × 1024 × 1024 | 2.3 ms  |
| NKI SYRK            | 1024 × 1024        | 5.71 ms |
| NKI TRSM (DF-MP2)   | 2048 × 512         | 35.82 ms |

**Fused MP2 energy kernel (end-to-end energy step):**

| Shape                                     | Torch reduction | Fused kernel | Speedup |
|-------------------------------------------|----------------:|-------------:|--------:|
| medium (nbasis=512, nocc=64, nvir=448)    | 8.03 s          | 5.43 s       | 1.48×   |
| large (nbasis=768, nocc=96, nvir=672)     | 44.57 s         | 30.27 s      | 1.47×   |

**Cross-platform DF-MP2 medium end-to-end:**

| Platform      | Warm wall | Flops rate   |
|---------------|----------:|-------------:|
| trn1.2xlarge  | 9.91 s    | 0.28 TFLOPS  |
| A10G g5.xlarge | 0.266 s   | 10.3 TFLOPS  |

A10G's cuBLAS is ~37× faster per-call on medium. Ampere's tensor
cores are further along their per-watt optimization curve for dense
GEMM than trn1's first-generation NeuronCores at the sizes these
benches touch; the cost story (trn1.2xlarge at $1.34/hr vs g5.xlarge
at $1.006/hr) only shifts at scales where memory bandwidth and
multi-chip topology matter more than single-GPU throughput.

**Chemistry validation** (via
[`test_df_mp2_pyscf.py`](https://github.com/trnsci/trnblas/blob/main/tests/test_df_mp2_pyscf.py)):
H₂O/STO-3G matches PySCF's `mp.dfmp2.DFMP2` to 1 µHa
(1×10⁻⁶ Ha). H₂O/cc-pVDZ, CH₄/cc-pVDZ, and NH₃/cc-pVDZ match to
10 µHa (1×10⁻⁵ Ha). These tolerances come from the accumulated
rounding of FP32 accumulation in the Tensor Engine against PySCF's
FP64 reference. For closed-shell correlation energies in the
~10⁻¹ to 10⁻⁰ Ha range, 10⁻⁵ Ha relative error sits comfortably
above chemical-accuracy thresholds (~1 mHa ≈ 2.6 kJ/mol). The
FP32-vs-FP64 question is real and open — Phase 2 tracks
double-double GEMM for workloads where the relative error floor
matters more than these benches exercise.

## What's next

- **[#35](https://github.com/trnsci/trnblas/issues/35) — cross-pair
  batching.** The remaining perf hypothesis for the 1.48× gap:
  batching K pair partials in SBUF before one HBM store relaxes
  the per-pair synchronization fence. Phase 3 perf work.
- **[#26](https://github.com/trnsci/trnblas/issues/26) — NKI GEMM
  tile-shape autotuner.** Phase 3. Plan-time tile selection feeds
  the NEFF cache key; measured-best tile per shape replaces the
  current fixed `(128, 128, 512)`.
- **Phase 2 — [double-double FP64
  GEMM](https://github.com/trnsci/trnblas/issues/22).** Emulated
  FP64 via two FP32 values in the Tensor Engine. Opens the door
  to chemistry workloads where 10⁻⁵ Ha relative error is not
  enough — coupled-cluster correlation, geometry optimizations
  near stationary points.
- **Phase 4 — tensor-parallel GEMM across
  NeuronCores.** trn1.32xlarge has 16 chips. A single DF-MP2 chunk
  sharded across chips is the obvious next architectural exercise.

Phase tracker: [trnsci ROADMAP](https://trnsci.dev/roadmap/).

## Takeaway

The fused MP2 energy kernel is what happens when a library takes
Trainium's architecture seriously instead of porting cuBLAS
primitives one-to-one. The whole `T * (2T − Tᵀ) / Δ + sum`
expression is one kernel, one NEFF, one dispatch per `(i, j)` pair,
with every intermediate SBUF-resident. cuBLAS cannot express this
shape — not because it's hard, but because cuBLAS's primitive is
matrix products, and this expression isn't one. NKI's whole-program
DAG compilation is what makes "one kernel per pair" the natural
unit of work on Trainium. Phase 1's measurable win is smaller than
the RFC predicted (1.48× vs 3×), and that gap is documented and
queued as the Phase 3 concrete next step. But the architectural
shape of the kernel is the thing to notice — a shape cuBLAS doesn't
suggest and a whole-program compilation model does.
