---
date: 2026-04-15
categories: [Deep dive, trntensor]
comments: true
---

# trntensor: when the kernel boundary is the API

trntensor Phase 1 landed this week: the 2-index and batched `nc_matmul` NKI kernels validate on trn1, `ContractionPlan.backend` now reports `"nki"` when shapes qualify, and two fused multi-step primitives — the DF-MP2 correlation-energy kernel and the 4-index AO→MO integral transform — run the full contract → elementwise → reduce and contract → SBUF-resident → contract patterns in single NKI programs. The architectural point isn't "we implemented einsum on Trainium." It's that on Trainium the kernel boundary is the design surface: what cuTENSOR encapsulates behind a `Plan` object, NKI asks you to lay out in source. That's more work, but it's also where a tensor library can become a cuTENSOR superset rather than a cuTENSOR port.

<!-- more -->

## The problem

Einstein summation generalizes matrix multiplication to arbitrary tensor contractions. A quantum-chemistry workload like density-fitted MP2 is five or six contractions in sequence — auxiliary-basis 4-index transform to go from AO to MO, pair-energy contractions over occupied-occupied indices, elementwise denominators, reductions. Every post-Hartree-Fock method has the same shape.

cuTENSOR's model for this is one plan per contraction: `cutensorInitContractionDescriptor`, then `cutensorContractionExecute`. Each plan becomes a CUDA kernel. Multiple contractions compose in Python (or host C) between plans, with intermediate tensors landing in HBM between calls. The programmer rarely thinks about where the plan ends and the next one starts — cuTENSOR hides the kernel boundary.

The naive port of that shape to Trainium produces correct results and leaves most of the performance on the floor. DF-MP2 with 25 pair contractions compiles to 25 NKI dispatches; each dispatch pays a fixed XLA overhead. Profiling a 2048² matmul on trn1 (reported in [#33](https://github.com/trnsci/trntensor/issues/33#issuecomment-4239594619)) measured 4,081 µs host→XLA transfer, 2,994 µs XLA→host transfer, and only 568 µs of actual kernel time — the wrapper is doing an order of magnitude more work than the Tensor Engine. The per-kernel compile is fine. The per-dispatch surrounding work is not.

## What the architecture suggests

The NKI programming model exposes four engines (Tensor, Vector, Scalar, GpSimd), two partitioned memory regions (PSUM for accumulation, SBUF for staging), and a compile-once-run-many NEFF cache. A single `@nki.jit` program can span arbitrary control flow over those resources. That's what matters for a tensor library: one NKI program can contain multiple `nisa.nc_matmul` calls, interleaved Vector Engine elementwise ops, and a scalar reduction at the end, all without intermediate tensors ever touching HBM.

cuTENSOR's `Plan` is an opaque object that maps to one kernel. NKI's equivalent is the source of the `@nki.jit`-decorated function — and that function can encapsulate a DAG of contractions rather than a single one. Two concrete patterns:

- **Contract → elementwise → reduce.** Compute a contraction in PSUM. Copy to SBUF. Apply element-wise math (multiply, subtract, reciprocal). Fold via `nl.sum` into a scalar accumulator tile. Store one HBM value per outer iteration. This is the DF-MP2 energy pattern.

- **Contract → SBUF intermediate → contract.** Compute a first matmul whose PSUM output copies to SBUF. Use that SBUF tile as the stationary operand of a second matmul. Store the second matmul's PSUM output. The intermediate tensor never exists in the user's Python or in HBM. This is the 4-index AO→MO transform pattern.

Neither pattern is new to HPC — fused GEMMs and fused reductions exist on GPUs too. The difference is who writes them. cuTENSOR hands you a planner and hides the code; NKI hands you the code and expects you to plan. When the workload matches one of these patterns, the NKI version is a cuTENSOR superset — a primitive that cuTENSOR's plan abstraction doesn't name.

## The approach

trntensor has three public-API layers that correspond to three levels of user engagement with this kernel-boundary discipline:

1. **Generic `einsum(subscripts, *operands)`** — an opinionated router. `ContractionPlan` analyzes the subscript, classifies the contraction (`matmul`, `bmm`, `torch`), computes a FLOP estimate, and chooses a dispatch target. Small shapes skip NKI entirely via a 2-GFLOP threshold — the per-call dispatch overhead rules out NKI at sizes where the Tensor Engine couldn't earn it back. Subscripts that don't match a known pattern (3+ operands, broadcasting, patterns not yet specialized) fall through to `torch.einsum`.

2. **Named fused primitives** — `trntensor.mp2_energy(B, eps_occ, eps_vir)` and `trntensor.ao_to_mo_transform(eri, C_occ, C_vir)`. These aren't generic; they're named chemistry-domain operations that compile to one fused NKI program each. Users who know what they want reach for these; users who don't stay with `einsum`.

3. **Operand residency** — `trntensor.to_xla(tensor)` and `trntensor.from_xla(tensor)`. The kernel-boundary discipline applies between trntensor calls too: XLA-resident operands skip the per-dispatch transfer. The DF-MP2 pipeline becomes "transfer once, compute, pull back the scalar," matching at the program level what the kernels do at the intra-kernel level.

The deliberate tradeoff: trntensor doesn't try to detect fusion opportunities automatically. A generic multi-`einsum` detector that compiles arbitrary contraction DAGs to fused NKI programs is tracked for v0.3.0 and would be genuinely useful, but building it now means building the abstractions before they've been validated against enough concrete kernels. Two named primitives and a routed generic surface is a cleaner starting point than a one-shot DAG compiler.

## Implementation

The `mp2_energy_kernel` is the most complete demonstration. For each `(i, j)` orbital pair:

```python
# trntensor/nki/_kernels.py
@nki.jit
def mp2_energy_kernel(B, eps_occ, eps_vir):
    NOCC, NVIR, NAUX = B.shape
    partial = nl.ndarray((NOCC, NOCC), dtype=nl.float32, buffer=nl.shared_hbm)
    ev = nl.load(eps_vir[0:NVIR, 0:1])  # 2D for unambiguous partition dim

    for i in nl.affine_range(NOCC):
        Bi_t = nl.load_transpose2d(B[i, 0:NVIR, 0:NAUX])
        eo_i = nl.load(eps_occ[i : i + 1, 0:1])

        for j in nl.affine_range(NOCC):
            Bj_t = nl.load_transpose2d(B[j, 0:NVIR, 0:NAUX])
            eo_sum = nl.add(eo_i, nl.load(eps_occ[j : j + 1, 0:1]))

            # Two nc_matmul in PSUM: T = Bi @ Bj.T and T^T = Bj @ Bi.T
            psum_T  = nl.zeros((NVIR, NVIR), dtype=nl.float32, buffer=nl.psum)
            psum_Tt = nl.zeros((NVIR, NVIR), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_T,  stationary=Bi_t, moving=Bj_t, accumulate=True)
            nisa.nc_matmul(dst=psum_Tt, stationary=Bj_t, moving=Bi_t, accumulate=True)

            # PSUM → SBUF (NKI 0.3.0 requires explicit copy)
            t   = nl.ndarray((NVIR, NVIR), dtype=B.dtype, buffer=nl.sbuf)
            t_T = nl.ndarray((NVIR, NVIR), dtype=B.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(src=psum_T,  dst=t)
            nisa.tensor_copy(src=psum_Tt, dst=t_T)

            # Vector Engine: Δ_ab = (ε_i+ε_j) - ε_a - ε_b
            denom = nl.subtract(nl.subtract(eo_sum, ev), ev.reshape((1, NVIR)))

            # term = T * (2T - T^T) / Δ  (divide-as-reciprocal for 0.3.0)
            term = nl.multiply(
                nl.multiply(t, nl.subtract(nl.multiply(t, 2.0), t_T)),
                nl.reciprocal(denom),
            )

            # Scalar accumulator — avoids 0-D SBUF allocation
            acc = nl.zeros((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            acc[...] = nl.add(acc, nl.sum(term, axis=(0, 1)))
            nl.store(partial[i : i + 1, j : j + 1], value=acc)

    return partial
```

One program. Two `nc_matmul` per pair, reusing SBUF tiles. All elementwise math on the Vector Engine. Reduction into a `(1, 1)` SBUF accumulator. One HBM store per `(i, j)`. A cuTENSOR port would package this as three plans — contract, elementwise, reduce — with intermediates `T` and `term` materialized between them. Here nothing between `nl.load` and `nl.store` appears as a tensor in Python or HBM.

The `ContractionPlan.backend` report threads through:

```python
# trntensor/plan.py
def _backend_for(strategy: str, operands: tuple) -> str:
    if strategy not in ("matmul", "bmm"):
        return "pytorch"
    from .nki.dispatch import HAS_NKI, _MIN_NKI_FLOPS
    if not HAS_NKI:
        return "pytorch"
    flops = operand_flops(strategy, operands)
    return "nki" if flops >= _MIN_NKI_FLOPS else "pytorch"
```

`plan.backend` reflects what will actually run, not what the algorithm classification says. A 64×64 matmul is matmul-strategy but pytorch-backend; a 2048×2048 matmul is matmul-strategy and nki-backend. Plan-time transparency about where work will land.

## What didn't work

Several paths that looked reasonable didn't survive contact with the NKI compiler.

**0-D SBUF allocation.** The first version of `mp2_energy_kernel` wrote `e_ij = nl.sum(term, axis=(0, 1))` and stored `e_ij` directly. NKI 0.3.0 rejects this with `SBUF and PSUM tensors must have at least 2 dimensions (partition-dim and free-dim)`. The fix is the `acc = nl.zeros((1, 1), ...)` pattern above — broadcast the 0-D sum into an explicit `(1, 1)` tile via `nl.add`. The trnblas reference had this pattern; we didn't discover why until the assertion fired.

**`nl.copy` returns a view in 0.3.0.** In 2.24, `c_sbuf = nl.copy(psum, dtype=...)` allocated a fresh SBUF tile. In 0.3.0 it returns a view and the subsequent `nl.store` produces silently wrong results. The fix is explicit: allocate with `nl.ndarray(..., buffer=nl.sbuf)` and copy with `nisa.tensor_copy(src=psum, dst=...)`. This is mentioned in the release notes but easy to miss.

**1D tensor loads and partition-dim inference.** `mp2_energy_kernel` originally loaded `eps_vir` with a 1D slice `nl.load(eps_vir[0:NVIR])`. That compiled and ran when `eps_vir` arrived freshly transferred from CPU. It compile-failed when the same tensor arrived pre-pinned on XLA via `to_xla`. The 1D slice leaves partition-dim inference ambiguous, and the two residency states presented different tensor metadata to the compiler. Reshaping to `(N, 1)` at the dispatch boundary — what trnblas already does — fixed it.

**Cross-kernel XLA graph fusion.** The full DF-MP2 pipeline with everything pre-pinned (`ao_to_mo_transform → mp2_energy` without a `from_xla` between them) triggers an NKI compiler bug: the combined XLA lazy graph spanning both kernels generates `Shared memory is only supported on trn2, but inst__I-9-0:_mem_0_0_set is using Shared memory on an unsupported target` on trn1. Inserting `xm.mark_step()` between calls didn't resolve it — the flush itself is what produces the trn2-only code. Tracked in [#39](https://github.com/trnsci/trntensor/issues/39) for upstream escalation; users currently `from_xla(B)` between the two calls. This is the one place Phase 1's residency story has a hole.

### Fit assessment

Small contractions are not what Trainium wants. DF-MP2 pair contractions are 200–300 kilo-FLOP per call; the dispatch wrapper spends longer than that moving data before the kernel starts. cuBLAS has the same problem at similar sizes on NVIDIA — it's not a Trainium defect — but on Trainium the absolute overhead is higher because the device transfer crosses a less tightly-integrated boundary than a GPU's PCIe path. Trainium is over-indexed for large GEMMs and under-indexed for tight loops of small contractions. The practical answer is residency + fusion, not doing more per-call work.

**The planner isn't a path-search engine yet.** `ContractionPlan` handles one contraction at a time. For a 3+ operand einsum, the current fallback is `torch.einsum`, which loses the chance to pick a better contraction order and loses the fused-DAG opportunity entirely. Phase 3 adds the path search; Phase 1 admits it doesn't have one.

**Documentation gaps we discovered empirically.** The NKI 0.3.0 transition notes list `nc_matmul` signature changes and `nl.copy`'s new view semantics, but the partition-dim strictness of 1D loads wasn't called out. The `SBUF and PSUM tensors must have at least 2 dimensions` assertion error is accurate but doesn't hint at which SBUF allocation is 0-D — you have to walk the trace to `nki/language/core.py:51`. These are the kinds of gaps that get filled in once a few projects have tripped over them; documenting them here is part of that.

## Numbers

trn1.2xlarge, neuronxcc 2.29 / NKI 0.3.0. Same machine for both columns (`TRNTENSOR_FORCE_BACKEND=pytorch` for the CPU baseline).

| Op | Shape | FLOPs | PyTorch (trn1) | NKI (trn1) | Notes |
|---|---|---:|---:|---:|---|
| `einsum ap,bp->ab` (DF-MP2 pair) | 48×128 × 48×128 | 295 K | **19.6 µs** | 1047 µs | CPU 53× — dispatch overhead dominates |
| `einsum mi,mnP->inP` (4-index) | 32×8, 32×32×64 | 524 K | 35.4 µs | **35.1 µs** | break-even |
| `einsum ij,jk->ik` | 512³ | 134 M | **481 µs** | 1452 µs | CPU 3.0× |
| `einsum bij,bjk->bik` | 16×256³ | 268 M | **953 µs** | 2162 µs | CPU 2.3× |
| `einsum ij,jk->ik` | 1024³ | 1.07 G | **3402 µs** | 4022 µs | CPU 1.2× |
| `einsum ij,jk->ik` | 2048³ | 8.6 G | 27.4 ms | **16.9 ms** | NKI 1.6× |
| `einsum bij,bjk->bik` | 32×1024³ | 34.4 G | **126.3 ms** | 190.8 ms | CPU 1.5× |
| `mp2_energy` fused vs Python loop | 5×19×72 | — | **1.5 ms** | 16 ms | loop 10× — same overhead story |
| `mp2_energy` fused vs Python loop | 16×128×128 | — | **25.5 ms** | 41 ms | loop 1.6× — gap closing |
| 5-iter matmul_2048 with residency | 2048³ | 8.6 G | cold loop | **≥ 3× faster** | `to_xla` pre-pin — v0.3.0 baseline |

NKI wins one benchmark outright (2048² matmul, 1.6× faster). The residency test is where the story becomes interesting: pre-pinning operands eliminates the dispatch overhead that dominates every other number on this table. The fused kernels are architecturally correct; the residency API is what lets them earn their keep.

## What's next

Phase trackers for the library, in order:

- [Phase 2 — precision-aware contraction path selection](https://github.com/trnsci/trntensor/issues/28)
- [Phase 3 — opt_einsum-style path planner + plan cache reuse](https://github.com/trnsci/trntensor/issues/29)
- [Phase 4 — sharded tensor contractions across chips](https://github.com/trnsci/trntensor/issues/30)
- [Phase 5 — trn2 fused multi-contraction paths](https://github.com/trnsci/trntensor/issues/31)

Concrete v0.3.0 follow-ups already filed: [K-tiling for `ao_to_mo_transform` when nbasis > 128](https://github.com/trnsci/trntensor/issues/37), [generic `multi_einsum` shared-operand detection](https://github.com/trnsci/trntensor/issues/19), [α/β scaling to match cuTENSOR's interface](https://github.com/trnsci/trntensor/issues/20), [the cross-kernel NKI compiler bug](https://github.com/trnsci/trntensor/issues/39) pending upstream.

## Takeaway

A tensor contraction library on Trainium looks different from one on a GPU because the kernel boundary is writable. cuTENSOR's `Plan` encapsulates one contraction behind an opaque handle and lets the runtime pick a kernel; trntensor's named fused primitives (`mp2_energy`, `ao_to_mo_transform`) span multiple contractions in one NKI program and expose the composition. That's a cuTENSOR superset when the workload matches a named pattern and a cuTENSOR-equivalent generic path for everything else.

The design lesson Phase 1 delivers: fused, pattern-specific kernels are a normal mode of operation on Trainium, not an optimization pass. The library should name them as first-class primitives rather than try to detect them at dispatch time.
