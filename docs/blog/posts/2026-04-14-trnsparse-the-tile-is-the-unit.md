---
date: 2026-04-14
categories: [Deep dive, trnsparse]
comments: true
---

# trnsparse: the tile is the unit, not the nonzero

trnsparse shipped its first hardware-validated NKI SpMM kernel in v0.2.0 last week, and the benchmark table was publicly worse than scipy across every configuration that was run. That's not a failure — it's the piece of evidence that led to the reframe that v0.3.0 ships on: **Trainium's sparse primitive isn't CSR, it's the 128×128 Tensor Engine tile**. This post is the retrospective — what shipped, what the numbers actually said, and why the honest story is that the CUDA sparse playbook is the wrong starting point for this hardware.

<!-- more -->

## The problem

[trnsparse](https://trnsci.dev/trnsparse/) is trnsci's cuSPARSE-equivalent: CSR and COO formats, SpMV, SpMM, integral screening for quantum chemistry. The workloads that motivate it — Schwarz-screened Fock builds, block-structured Hamiltonians, graph Laplacians, block-sparse attention — share an awkward property. They are *sparse* in a general sense (most entries are zero), but the distribution of those zeros is almost never uniform. Fock matrices after Schwarz screening are dense in diagonal blocks and sparse off-diagonal. FEM stiffness matrices track mesh connectivity. Block-sparse transformer attention is a structured mask.

cuSPARSE handles all of this with a CSR-native design: row pointers, column indices, scalar values, a bag of kernels tuned for a GPU's ability to do arbitrary-pattern indirect gather cheaply. Thousands of threads each chasing one pointer is what GPU memory hierarchies are built for. A naive port of that design to Trainium is where trnsparse started, and where the numbers told it to stop.

## What the architecture suggests

Trainium's Tensor Engine is a 128-partition × 512-moving systolic array. `nisa.nc_matmul` is the hot op, and it operates on a tile — partition dim ≤ 128 on the stationary operand, free dim ≤ 512 on the moving operand. A single `nc_matmul` is a dense 128×K×N multiply. The DMA engine handles memory movement between HBM and SBUF, but — and this matters for sparse workloads — **as of NKI 2.24 / 0.3.0, the DMA engine does not expose an indirect-gather primitive at the kernel level**. Scatter-gather is a pattern the silicon supports in principle; the language doesn't expose it yet.

That combination — tile-shaped compute, no per-element indirect gather — has a specific consequence. The natural unit of sparse work on Trainium is not a single nonzero. It's a 128×128 block.

A format that stores sparse matrices as 128×128 dense blocks, with a block-level CSR pattern over which blocks are nonzero, maps one-to-one onto the Tensor Engine. Each nonzero block is already in the shape `nc_matmul` wants — no gather step. The block-level pattern is a much smaller sparse structure that fits cleanly in host-side dispatch. Zero blocks are skipped in the dispatch loop.

This is Block-Sparse Row (BSR) at `block_size = 128`. cuSPARSE has a BSR implementation, but it's a secondary format there — a specialization. On Trainium it's not a specialization; it's the format the hardware asks for. cuSPARSE's BSR is a port back. Trainium's BSR is native.

**CSR and COO stop being compute formats** on the NKI path. They're interop — how scipy users hand matrices in, how ERIs come out of a chemistry code, how graph adjacencies arrive from PyTorch Geometric. The compute path converts to BSR at dispatch time, runs at tile granularity, returns dense. And **block density matters much more than element density**: a Fock matrix at 99.5% zero might have 30% of its 128×128 blocks storing something. That's the density BSR cares about, and the one that stays modest for the structured workloads that motivated the library.

## The approach

v0.2.0 shipped as the correctness path. The NKI kernel — [`_spmm_dense_kernel`](https://github.com/trnsci/trnsparse/blob/main/trnsparse/nki/kernels.py) — does the simplest thing that validates the Neuron toolchain: materialize CSR into a dense `(M, K)` tile on the host, pad to tile multiples, dispatch a stationary-A GEMM. Effectively the [trnblas GEMM pattern](https://github.com/trnsci/trnblas/blob/main/trnblas/nki/dispatch.py) with a trivial preamble. No sparsity exploitation.

The deliberate tradeoff: publicly slow. At 1024×1024 / density 0.001 / N=128, this does roughly 1000× more arithmetic than scipy. The benchmark table in v0.2.0 documents that directly. The reason to ship it: the full toolchain — compile, NEFF cache, XLA dispatch, PyTorch integration, `torch.autograd.Function`-wrapped backward — had to be wired end-to-end before the project could credibly commit to BSR. v0.2.0 is the evidence that said "the pipeline works; the shape of the work is wrong."

v0.3.0 introduced [`BSRMatrix`](https://github.com/trnsci/trnsparse/blob/main/trnsparse/formats.py) and [`bsr_spmm`](https://github.com/trnsci/trnsparse/blob/main/trnsparse/ops.py) as the headline. CSR stays in the API — users bring CSR from elsewhere, and the vectorized CPU fallback is within 2× of scipy — but the NKI compute story runs through BSR. v0.4.0 layered on [`screened_spmm`](https://github.com/trnsci/trnsparse/blob/main/trnsparse/ops.py): a single `@nki.jit` kernel fuses Schwarz bounds, mask application, and matmul. The unfused equivalent is four host passes plus a separate CSR construction before the matmul. The fused flow is one dispatch, no mask tensor on HBM, no separate format conversion. That's the second architectural pattern Trainium makes natural and CUDA doesn't reach for.

## Implementation

The BSR kernel, stripped of docstrings and padding:

```python
@nki.jit
def _bsr_spmm_kernel(blocks_pad, b_gathered):
    """Block-sparse × dense matmul via per-block nc_matmul."""
    M_tiles, K_max, _, _ = blocks_pad.shape
    _, _, _, N = b_gathered.shape

    TILE_M = 128  # fixed by BSR block_size
    TILE_N = N if N <= 512 else 512

    out = nl.ndarray((M_tiles * TILE_M, N),
                     dtype=blocks_pad.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M_tiles):
        for n in nl.affine_range(N // TILE_N):
            psum = nl.zeros((TILE_M, TILE_N),
                            dtype=nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K_max):
                a_t    = nl.load_transpose2d(blocks_pad[m, k, :, :])
                b_tile = nl.load(b_gathered[m, k, :,
                                            n * TILE_N:(n + 1) * TILE_N])
                psum[...] += nisa.nc_matmul(a_t, b_tile)
            c_sbuf = nl.copy(psum, dtype=blocks_pad.dtype)
            nl.store(out[m * TILE_M:(m + 1) * TILE_M,
                          n * TILE_N:(n + 1) * TILE_N], value=c_sbuf)
    return out
```

The host-side preamble pads each block-row to the same `K_max` with zero blocks so the kernel's `affine_range` bounds are fixed. That padding is the honest admission: block-rows with fewer stored blocks pay for the max. The alternative — row-bucketing by nnz ([#15](https://github.com/trnsci/trnsparse/issues/15)) — requires an indirect-DMA primitive that NKI 2.24/0.3.0 does not expose.

The autograd wrapper, which is the suite's reference pattern for satisfying the cross-project differentiability requirement in [trnsci/trnsci#3](https://github.com/trnsci/trnsci/issues/3):

```python
class _BSRSpMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_blocks, A_block_col_indices, A_block_row_ptrs,
                A_shape, A_block_size, B):
        # Host-side pad + gather + NKI dispatch (or simulator / PyTorch fallback)
        ...
        ctx.save_for_backward(A_blocks, A_block_col_indices, A_block_row_ptrs, B)
        ctx.A_shape = A_shape
        ctx.A_block_size = A_block_size
        return C

    @staticmethod
    def backward(ctx, grad_out):
        # dA_blocks[k] = grad_out[rowblock*b:(rowblock+1)*b]
        #                @ B[col*b:(col+1)*b].T  (pattern stays fixed)
        # dB           = A.T @ grad_out   (reconstruct A dense)
        ...
```

Backward runs at the PyTorch level. The mask (which blocks exist) is non-differentiable by construction. `torch.autograd.gradcheck` at `atol=1e-4` passes on hardware; this pattern now ships for three NKI kernels (CSR SpMM in v0.2.0, BSR SpMM in v0.3.0, fused screened SpMM in v0.4.0).

## What didn't work

This is the richest section of the post, because several things went sideways in instructive ways.

**v0.2.0's benchmark numbers were worse than planning assumed.** The expectation was "dense-materialization will be slower at low densities, but the high-N dispatch win narrows the gap." The measured gap at 1024×1024 / density 0.01 / N=128 was scipy at 257 μs against trnsparse NKI at 2212 μs. The reason turned out to be dispatch overhead, not arithmetic — NKI times are roughly constant at 1.3–2.5 ms across all configurations, because the Neuron dispatch + HBM round-trip floor is flat and big. The [full benchmark table](https://trnsci.dev/trnsparse/benchmarks/) ships with all the entries where NKI loses by 100×. None were removed before release.

**CG-in-kernel isn't buildable on NKI 2.24/0.3.0.** [#24](https://github.com/trnsci/trnsparse/issues/24) was filed early as the v0.4.0 architectural follow-up: a fused Conjugate Gradient kernel with `A` SBUF-resident across all `max_iter` iterations, x/r/p cycled inside the kernel. This was the story. The audit that killed it found three hard walls: `nl.affine_range` has no `break`/`continue`, so no in-kernel convergence exit; no iteration-carried scalar state across `affine_range` levels (documented in the `trnblas _mp2_energy_kernel` source at `dispatch.py:586-588` as "`in-place += across affine_range hits NKI's 'Unexpected output dependencies'`"); and no nested kernel calls, so the BSR matvec would have to be inlined rather than invoked. #24 is closed as not-buildable in current NKI. An honest close comment explains the reframe and leaves the door open for a future NKI release that adds persistent SBUF across calls, at which point the full loop-in-kernel design becomes reachable again. What shipped instead was v0.3.2 `cg_bsr` — Python-loop around the existing `bsr_spmm` matvec, correct and differentiable but without the SBUF-resident win.

**NKI 0.3.0 migration breaking changes are MLIR-level, not Python-level.** The library migrated to the top-level `nki.*` namespace on 2026-04-14 ([trnsci/trnsci#5](https://github.com/trnsci/trnsci/issues/5)). The moves to watch — `nc_matmul` keyword-only arguments, `nl.copy` returning a view of PSUM, `nl.divide` dropped in favor of `nl.multiply` + `nl.reciprocal` — manifest at the MLIR verifier layer, not in a Python trace. The trnsparse kernels were already compliant by coincidence (modelled on trnblas, which did the audit first), so no functional change. For kernels written less cautiously, hardware CI stays the gate that catches these.

**The autograd wrapper's block-gradient projection needed care.** The natural first cut — differentiate through `BSRMatrix.from_dense` — flowed gradients back through the block-selection logic, which is structurally wrong because block selection is non-differentiable. The shipped wrapper stores block-col-indices and block-row-ptrs in `ctx` and routes `grad_out[rowblock*b:(rowblock+1)*b] @ B[col*b:(col+1)*b].T` into exactly the stored blocks. `torch.autograd.gradcheck` on a tiny synthetic system was the only thing that would have caught the first version.

**Simulator coverage is narrower than the headline suggests.** The [NKI 0.3.0 simulator write-up](https://trnsci.dev/blog/the-dev-loop-just-got-a-lot-shorter/) covers this in detail; the trnsparse-specific note is that `nki.simulate` catches Python-layer errors but not MLIR verifier errors, so hardware CI stays load-bearing for anything touching partition-dim broadcasting or shared-memory barriers. A device-free NEFF compile entry point would close this gap; a concrete request for the Neuron team.

**Candid fit-assessment.** Trainium is well-indexed for dense-GEMM-heavy training — its original motivating workload. trnsparse's Fock-build and block-sparse attention cases are a decent fit because they're block-dense at 128×128. Truly irregular sparse matmul (random CSR at density 0.001, highly variable nnz per row) is a shape mismatch with the silicon, not a library limitation. When a workload doesn't fit BSR — graph neural networks over non-uniform adjacency, for instance — the library recommends the `torch.sparse` fallback, not the NKI path. A future silicon generation that exposes indirect DMA gather would unblock a real gather-matmul-scatter path; that's a concrete hardware request.

## Numbers

All numbers below are on `trn1.2xlarge` with the DLAMI `ami-07f81955eadf5b89c` (2026-04-10 build, `neuronxcc==2.24.5133` alongside `nki==0.3.0`). CPU baselines run on the same instance's Xeon.

**v0.2.0 CSR SpMM, mean time in μs.** Lower is better. Columns are size `M=K`, density, `N` (RHS width).

| Size | Density | N | scipy | torch.sparse | trnsparse pytorch | trnsparse nki |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.001 | 32 | 8.8 | 15.4 | 48.3 | 1397 |
| 256 | 0.001 | 128 | 7.6 | 29.1 | 44.7 | 1365 |
| 256 | 0.01 | 32 | 8.8 | 15.6 | 28.9 | 1248 |
| 256 | 0.01 | 128 | 20.4 | 27.5 | 41.7 | 1370 |
| 256 | 0.1 | 32 | 40.0 | 18.5 | 31.7 | 1278 |
| 256 | 0.1 | 128 | 137 | 34.2 | 48.6 | 1428 |
| 1024 | 0.001 | 32 | 14.4 | 28.9 | 43.4 | 1732 |
| 1024 | 0.001 | 128 | 39.8 | 27.2 | 47.0 | 2067 |
| 1024 | 0.01 | 32 | 72.4 | 31.8 | 48.1 | 1847 |
| 1024 | 0.01 | 128 | 257 | 46.5 | 72.5 | 2212 |
| 1024 | 0.1 | 32 | 609 | 75.0 | 95.5 | 2151 |
| 1024 | 0.1 | 128 | 2475 | 248 | 274 | 2479 |

At every data point, the NKI column is slower than both CPU backends. This is the v0.2.0 shipping posture: correctness first, not speed. Two structural reasons, both honest:

1. **No sparsity exploitation.** The v0.2.0 kernel materializes the CSR into a dense `(M, K)` tile before the matmul. At density 0.001 on a 1024×1024 matrix, this is 1000× more work than scipy does. The fused path from v0.3.0 onward operates on BSR, which doesn't pay this cost.
2. **Dispatch overhead dominates.** The NKI column is roughly constant at 1.3–2.5 ms across densities. That's the Neuron dispatch + HBM round-trip floor, not the arithmetic. At these sizes, a kernel launch is more expensive than the matmul itself.

**v0.3.0 BSR SpMM.** At 10% block density on `4 × 4` block grids with `N=128`, BSR-NKI is 1.85 ms vs BSR-PyTorch 0.11 ms — NKI still loses because dispatch overhead dominates at these sizes, but the trend flips with scale: at `8 × 8` / 50% block density / `N=256`, NKI is 3.11 ms vs PyTorch 1.05 ms (3× slower), and PyTorch itself is approaching the dense-GEMM ceiling of 0.47 ms. The architectural claim — BSR is Trainium-native — is validated at the level of "runs correctly, differentiable, on the Tensor Engine." The performance win against CPU is still ahead of the library; that's Phase 3 work.

Phase 1 (correctness) landed. Phase 3 (performance) has a clear architectural shape and is not shipped.

## What's next

- **[Phase 3 — #15](https://github.com/trnsci/trnsparse/issues/15)** was originally scoped as row-bucketing CSR + gather-matmul-scatter. Under the BSR reframe it's backlog: CSR row-bucketing requires an indirect-DMA primitive that NKI doesn't expose. BSR at 128×128 covers the structured-sparse workloads without needing it. If NKI later exposes per-row DMA gather, #15 reopens.
- **[#22 — on-chip iterative solvers](https://github.com/trnsci/trnsparse/issues/22)** is half-shipped: the Python-loop plumbing (`cg_bsr`, `power_iteration_bsr`) is in v0.3.2. The architectural-win half (CG loop fused inside one NKI kernel, A SBUF-resident across iterations) is parked on [#24](https://github.com/trnsci/trnsparse/issues/24) pending NKI capability additions — closed honestly as not-buildable under current constraints.
- **[Phase 4 — #16](https://github.com/trnsci/trnsparse/issues/16)** is sharded BSR across multiple NeuronCores. Gated on suite-level multi-chip collective primitives that don't exist in any trnsci library yet; this is pioneering territory.
- **[Phase 5 — #17](https://github.com/trnsci/trnsparse/issues/17)** is trn2-specific DMA bandwidth exploitation. Blocked on trn2 silicon being more widely available and on the indirect-DMA story from Phase 3 reopening.
- **Block-sparse attention as a primitive** — BSR at 128×128 is architecturally identical to a local-attention mask in a sparse transformer. Tracked in [#21](https://github.com/trnsci/trnsparse/issues/21); the writeup is an unshipped doc task.

## Takeaway

The v0.2.0 benchmark table is the most important thing trnsparse has published so far. Not because the numbers are good — they're not — but because they're the evidence that anchored the reframe. The CUDA sparse playbook assumes a memory hierarchy where arbitrary-pattern gather is cheap; Trainium has a tile-shaped compute unit and a DMA engine that doesn't yet expose indirect gather at the kernel level. Under those constraints, the native sparse representation isn't a list of nonzeros; it's a list of 128×128 blocks. BSR isn't a cuSPARSE port. It's what the hardware asks for. And the honest way to find that out was to ship the naive port, publish the numbers it produced, and let the shape of the failure tell the story.
