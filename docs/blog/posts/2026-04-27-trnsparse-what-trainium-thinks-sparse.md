---
date: 2026-04-27
categories: [Deep dive, trnsparse]
comments: true
---

# trnsparse: what Trainium thinks a sparse matrix is

Block-sparse attention on a systolic array requires rethinking the data structure before touching the kernel. trnsparse v0.6.0 ships forward and backward NKI attention kernels, K-tiling for head_dim > 128, and — after a week fighting NKI 0.3.0's changed API — a simulator CI gate that actually tests the kernels rather than silently substituting PyTorch.

<!-- more -->

## The problem

Sparse matrix libraries are traditionally organized around CSR: sorted arrays of column indices and values, row pointers for fast row access. This structure makes sense for graph problems, FEM meshes, and integral screening — workloads where nonzero density varies row-to-row and the pattern isn't known until runtime.

Block-sparse attention is a different animal. The nonzero structure is determined by the attention mask — local window, dilated, global-token — which is regular, structured, and known at dispatch time. Each nonzero is a 128×128 block of scores or values, not a single float. And the workhorse operation is `scores @ V`, a matmul, not a scalar accumulation.

CSR's per-element granularity is wrong for this. The scatter-gather overhead of extracting individual rows from HBM for a scalar multiply kills throughput on a systolic array.

## What the architecture suggests

Trainium's Tensor Engine is a 128×128 systolic array. One `nisa.nc_matmul` call consumes one 128×128 tile pair and produces one 128×128 output tile. This isn't a constraint to route around — it's the unit of work.

The natural sparse format on Trainium is therefore Block Sparse Row (BSR) at `block_size=128`. Each nonzero block in the BSR pattern is exactly one Tensor Engine tile: zero gather overhead, one `nc_matmul` per block, PSUM accumulates across blocks within a row. The DMA engine gathers the B-column slices in parallel with computation on the already-gathered blocks.

CSR and COO survive as *input* formats. They're supported for compatibility and for the cases — graph traversal, unstructured FEM — where per-element sparsity is genuine and block structure doesn't exist. But the NKI compute path requires BSR materialization. If the nonzeros don't tile into 128×128 blocks, the kernel passes through PyTorch.

Block-sparse attention is where this maps cleanly. The attention mask determines which 128-row × 128-column score blocks to compute. Everything else is zero and never allocated.

## The approach

trnsparse implements block-sparse attention in three layers:

**Host gather** (`_attn_gather`): builds a column grid `(M_tiles, K_max)` from the BSR pattern, fancy-indexes K and V into padded `(M_tiles, K_max, b, head_dim)` tensors, and scales Q. The padding uses column 0 as a sentinel for empty slots.

**Two-pass NKI kernels**: Pass 1 (`_attn_stats_kernel`) computes per-block row-wise max and stable exp-sum for the softmax normalizer — independently per block, no carry between iterations. A host reduction combines per-block stats into global `row_max` and `row_denom`. Pass 2 (`_attn_out_kernel`) recomputes scores from Q and K, applies stable softmax using the global stats loaded per block-row, and accumulates `weights @ V` into a PSUM tile that spans all K-blocks for the current query row.

**Backward pass**: The Flash Attention delta identity `D_i = dO_i · O_i` eliminates the need to store the full `(seq_len, seq_len)` attention matrix. A row-first kernel computes dQ; a column-first kernel (built from a transposed BSC gather) computes dK and dV without atomic scatter.

K-tiling extends this to head_dim > 128 by splitting the Q·K.T computation into TILE_K=128 chunks accumulated into the same score PSUM. The head_dim=256 and head_dim=512 paths multiply the inner loop without changing the outer structure.

## Implementation

The forward pass-2 kernel accumulates `weights @ V` across K-blocks in a single outer PSUM:

```python
@nki.jit
def _attn_out_kernel(q_scaled_blocks, k_gathered_pad, v_gathered_pad, row_max, row_denom):
    M_tiles, K_max, _, head_dim = k_gathered_pad.shape
    out = nl.ndarray((M_tiles * _TILE_M, head_dim), dtype=..., buffer=nl.shared_hbm)

    for m in nl.affine_range(M_tiles):
        row_max_m = nl.load(row_max[m, :, :])   # (128, 1)
        row_denom_m = nl.load(row_denom[m, :, :])
        out_psum = nl.zeros((_TILE_M, head_dim), dtype=nl.float32, buffer=nl.psum)

        for ki in nl.affine_range(K_max):
            v_tile = nl.load(v_gathered_pad[m, ki, :, :])
            score_psum = nl.zeros((_TILE_M, _TILE_M), dtype=nl.float32, buffer=nl.psum)
            q_t = nl.load_transpose2d(q_scaled_blocks[m, :, :])   # stationary
            k_t = nl.load_transpose2d(k_gathered_pad[m, ki, :, :])  # moving
            nisa.nc_matmul(score_psum, q_t, k_t, accumulate=True)

            # Drain PSUM → SBUF for VectorE ops
            _sp = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
            _sn = nl.ndarray((_TILE_M, _TILE_M), dtype=nl.float32)
            nisa.activation(_sp, nl.relu, score_psum)
            nisa.activation(_sn, nl.relu, score_psum, scale=-1.0)
            stable = nl.subtract(nl.subtract(_sp, _sn), row_max_m)
            weights = nl.divide(nl.exp(stable), row_denom_m)

            _wh = nl.ndarray((_TILE_M, _TILE_M), dtype=..., buffer=nl.shared_hbm)
            nl.store(_wh, value=weights)
            weights_t = nl.load_transpose2d(_wh)   # SBUF stationary
            nisa.nc_matmul(out_psum, weights_t, v_tile, accumulate=True)

        # Drain out_psum and store
        ...
```

The nested PSUM strategy — `score_psum` per (m, ki) block drained immediately, `out_psum` accumulating across all ki for block-row m — is the memory-efficient core. No O(seq_len²) score tensor is ever materialized.

## What didn't work

**Fused CG iteration** (issue #22) was the obvious next target after `cg_bsr` shipped. `nl.affine_range` doesn't support iteration-carried scalar state — the convergence check (`‖r‖ < tol`) can't live inside the loop. The workaround is fixed-K Chebyshev and Richardson iteration, which map cleanly to `affine_range` without scalar carry. For production use cases where sparsity is high enough that a fixed-K method converges, this is fine. For general CG, it's a genuine gap that needs a Neuron SDK enhancement.

**`nl.load_transpose2d` as the moving tile** was rejected by NKI 0.3.0's simulator with "moving must be in ['sbuf']." The simulator requires the moving argument to `nc_matmul` to come from `nl.load` or `nl.transpose(nl.load(...))`, not directly from `nl.load_transpose2d`. This appears to be a simulator-only constraint — hardware compiles the same kernel correctly. The distinction between load paths based on the memory operation type (not the resulting tensor shape) isn't documented.

**`nl.transpose` returns PSUM**, not SBUF, in NKI 0.3.0. Any tensor produced by `nl.transpose` can't be passed to `nc_matmul` as stationary, because stationary must be SBUF. The workaround is an HBM round-trip: `nl.store(tmp_hbm, value=sbuf_tensor)` then `nl.load_transpose2d(tmp_hbm)`. For the `weights @ V` computation, where `weights` is computed in SBUF and needs to be transposed for `nc_matmul`, this is the only correct path. The extra HBM write-then-read is an overhead with no hardware-architecture justification — it's a simulator API constraint.

**NKI 0.3.0 changed `nc_matmul`'s calling convention** without any deprecation path. The old `psum[...] += nisa.nc_matmul(stationary, moving)` stopped working; the new form is `nisa.nc_matmul(psum, stationary, moving, accumulate=True)`. Every kernel in the library needed updating. The Neuron SDK version bump wasn't flagged in the release notes as a breaking change. Discovery was by CI failure, not documentation. This is a concrete ask for the AWS Neuron team: breaking NKI ISA changes warrant a dedicated migration guide, not silence.

**PSUM drain** changed simultaneously. `nl.copy(psum, dtype=...)` is no longer valid; `nisa.activation(dst, nl.relu, psum, scale=-1.0)` combined with a relu identity (`relu(x) - relu(-x) = x`) is the only validated drain path. The correct drain for intermediate computation exists and works; getting there took several CI iterations because the error messages weren't diagnostic.

**The simulator's K < TILE_K constraint** means any `nc_matmul` where the partition dimension is less than 128 (e.g., head_dim=32 using K=32) fails in the CPU simulator but compiles and runs correctly on hardware. This divergence is a problem: the simulator is supposed to be the fast correctness-iteration path. The workaround is zero-padding head_dim to TILE_K=128 in the simulator dispatch path. This adds latency to the dispatch code and will need cleanup when the simulator constraint is fixed.

**dQ backward systematic error** in the simulator remains under investigation. All other backward quantities (dK, dV) match the PyTorch reference at atol=1e-3. dQ has a systematic error of ~1.0 absolute at 99% of positions in the simulator, despite the formula (`dS @ K * scale`) being analytically correct and the same formula computing correctly in PyTorch. Hardware validation is the authoritative gate until this is diagnosed.

## Numbers

Performance numbers for the SpMM path reflect the v0.2.0 correctness-first approach:

| Shape (M×K, K×N) | scipy (CPU) | torch.sparse (CPU) | trnsparse NKI (trn1) |
|---|---|---|---|
| 1024×1024, 1024×64 (10% density) | 0.8 ms | 1.1 ms | 4.2 ms |
| 4096×4096, 4096×256 (1% density) | 8.3 ms | 9.7 ms | 11.4 ms |

The NKI path is slower at current shapes. This is expected: v0.2.0 materializes the CSR pattern as dense, feeds it to the Tensor Engine as a full 4096×4096 tile, and discards the zero-block work at the PSUM level. The v0.3.0 row-bucketing path — gather nonzero rows into TILE_M-aligned blocks, nc_matmul per bucket — is the architecturally correct approach. That benchmark will look different.

For block-sparse attention, the relevant metric is memory: a (256, 256) full attention matrix at float32 is 256 KB. The two-pass BSR implementation never allocates it — tile_max and tile_sumexp together are O(n_blocks × b) ≈ 2 KB for the same sequence length with a local-window pattern. At seq_len=2048 with 10% sparsity, the difference is 16 MB vs 320 KB.

## What's next

Issue [#15](https://github.com/trnsci/trnsparse/issues/15) (row-bucketing CSR) requires NKI indirect-DMA gather, which the current SDK exposes incompletely. This is the path to SpMM that actually beats scipy.

Issue [#16](https://github.com/trnsci/trnsparse/issues/16) (sharded BSR across NeuronCores) waits on suite-level multi-chip collectives from trnsci/trnsci.

The dQ backward simulator error needs root-cause analysis. The hypothesis is a floating-point reordering difference between the NKI simulator's 128-dimension padded matmul and PyTorch's 32-dimension unpadded matmul, but it hasn't been confirmed.

BLR/HODLR off-diagonal compression — storing off-diagonal low-rank blocks in BF16, diagonal blocks in FP32 — is the Phase 2+ direction for integral equations and covariance matrices. Carson–Chen–Liu (SIAM J. Sci. Comput., 2024) provides the error bound.

## Takeaway

Trainium's native sparse format is a 128×128 block, not a CSR row. The BSR materialization cost in v0.2.0 is an admission of this: the Tensor Engine wants dense tiles, and v0.2.0 hands it one by naively densifying the CSR pattern. The architecturally correct answer — gather nonzero tiles and nc_matmul per tile — is what v0.3.0 will ship. Block-sparse attention is where the BSR structure pays immediately: the attention mask is already a block-structured binary matrix, so the gap between "CSR pattern" and "Tensor Engine tile" closes to zero. What didn't fit was the toolchain: NKI 0.3.0 changed three call conventions simultaneously, and the simulator enforces constraints that hardware doesn't. Getting 12 of 15 simulator tests to actually exercise NKI kernels — rather than silently substituting PyTorch — took longer than the K-tiling feature itself.
