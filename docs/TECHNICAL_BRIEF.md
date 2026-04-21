# trnsci technical brief — post-FP64 positioning and architectural principles

*For sub-project maintainers. Paste the shared section plus your library's section into any agent context that involves architectural decisions, new kernel design, API additions, or benchmarking.*

---

## Shared context (paste for all six libraries)

### The thesis

trnsci is the reference implementation for **post-FP64 scientific computing** on AWS Trainium — not just cuX equivalents ported to Neuron. The numerical-analysis community has been building the framework for this hardware for over a decade (Carson–Higham iterative refinement, Ozaki-scheme FP64 emulation, Croci–Higham–Mary stochastic rounding, Halko–Martinsson–Tropp randomized NLA); trnsci is the library that treats these ideas as the default API rather than exotic overlays on an FP64-centric LAPACK stack.

The competitive context: H100 FP64-tensor:BF16-tensor throughput is 1:14.8. B200 is 1:61. B300 Ultra is under 1:5000; cuBLAS in CUDA 13 now emulates FP64 DGEMM via the Ozaki scheme on FP8 tensor cores. Trainium was designed without an FP64 legacy to protect. It is not behind this trend — it codified it in 2020.

The CUDA-equivalents framing gets users in the door and stays valid. The post-FP64 thesis is what the project is actually about.

### Four architectural principles

These affect how kernels should be designed, not just how the project is positioned.

**1. PSUM is a free FP32 accumulator.** PSUM is wider than SBUF (FP32 vs BF16/FP8), exclusively written by the systolic array (deterministic order), and addressable by every other engine. After a BF16 matmul writes into PSUM at FP32, VectorE can compute the BF16 split and the residual in-place — an error-free split — while the next TensorEngine matmul proceeds on the next tile. This is the Ogita–Rump–Oishi Dot2 construction at systolic scale. For iterative refinement, PSUM is the residual buffer at higher precision than the working computation. Kernels should use this deliberately, not accidentally.

**2. Engine concurrency is free adaptivity.** The four engines (Tensor, Vector, Scalar, GpSimd) run independently. Adaptive decisions — Ozaki stopping criteria, GMRES-IR residual norms, randomized sketch quality estimates — can run on VectorE/GpSimdE while the TensorEngine runs the next matmul. The marginal cost of adaptivity is near zero when it runs on the idle engine.

**3. Stochastic rounding is in the ISA and should be used.** Per-instruction SR is a first-class NKI primitive (`nisa.activation(..., round_mode="stochastic")`). Connolly–Higham–Mary (SIAM J. Sci. Comput., 2021) proved SR rounding errors are mean-zero, replacing Wilkinson's worst-case n·u error bound with a √n·u probabilistic bound. For BF16 (u ≈ 2⁻⁸), any dot product or reduction of length n ≥ ~300 needs SR for correct convergence — without it, BF16 Krylov stagnates. This is a correctness requirement, not an optimization.

**4. Determinism is structural.** The TensorEngine has a fixed reduction order; the compiler statically schedules all instructions. Bitwise reproducibility is a structural property of every trnsci kernel given a fixed seed — not a ReproBLAS-style overlay. Kernels should document their reproducibility guarantees explicitly.

### The API direction

The suite is building toward a **solve-to-target-error contract**, not a precision knob:

```python
x, info = trnsolver.solve(A, b, target_forward_error=1e-10)
```

trnsci selects factorization precision, Ozaki split count, IR iteration count, and SR policy automatically. The `info` struct returns the achieved error bound, the theorems invoked, and the precision trajectory.

The `precision=` kwarg (trnfft, trntensor), `iterative_refinement=True` (trnsolver), and return tuples `(x, iters, res)` (cg/gmres) are scaffolding for this. Every API addition should ask: does this fit the `target_forward_error` contract, or is it a dead end that adds precision knobs without accuracy guarantees?

### What changes about how you write code

- **Admit what doesn't fit.** Long MD trajectories, general unstructured CSR SpMV, classical CCSD(T) — these are honest non-fits for this hardware. Saying so is more credible than pretending otherwise.
- **Name SR when you use it.** If a kernel relies on stochastic rounding for convergence, say so in the docstring and in the blog post.
- **Benchmark achieved forward error, not just wall clock.** A 10× faster result at 4 bits less precision is a different answer, not a 10× win.
- **Vintage-match comparisons.** A10G vs trn1, H100 vs trn2, GB200 vs trn3. Not cherry-picked shapes where Trainium looks best.
- **The "What didn't work" section is more important than the benchmark table.** Specific, nameable failures — 303 NEFF compile workdirs, distribution mean 0.31 vs 0.5, 215× slower than baseline — are the most valuable things the blog publishes.

---

## Library-specific sections

### trnfft

**Fit:** Partial. Fits well at small N via DFT-as-GEMM (the DFT matrix IS a matmul; one Tensor-engine call replaces log₂(N) butterfly stages). At larger N, error accumulates across log₂(N) butterfly stages — O(u log N) even before overflow. Compensated butterfly (`precision="kahan"`) addresses this; iterative FFT refinement (compute in BF16, estimate residual at FP32 via PSUM, refine) is the research direction.

**Architectural fit:** The partition-dim flattening across `total_groups × batch` is the right idiom for parallel butterfly stages. The four-engine concurrency (TE for twiddle multiply, VE for butterfly combination, DMA prefetching next twiddles) is the right kernel structure. Radix-4 exploits W₄ = {1,−1,i,−i} needing no multiplications; radix-8 exploits W₈'s irrational entries to earn Tensor Engine work.

**What doesn't fit honestly:** Non-power-of-2 FFTs via Bluestein chain three power-of-2 FFTs and accumulate error multiplicatively. `precision="double"` is the escape hatch, at the cost of CPU roundtripping. Don't claim Bluestein-at-BF16 is production-accurate for large N.

**Phase 2 direction:** Complete Kahan butterfly hardware characterization (#58). Iterative FFT refinement is a genuine research contribution — no one has published it on a production deterministic systolic array.

---

### trnblas

**Fit:** Clean. Dense LU/QR/Cholesky/GEMM are the showcase workload. The fused DF-MP2 energy kernel is the current flagship: one NEFF, SBUF-resident intermediates through the full T*(2T−Tᵀ)/Δ expression, one HBM store per (i,j) pair.

**Phase 2 keystone:** trnblas#22 — double-double FP64 GEMM (compensated matmul exploiting PSUM as the FP32 accumulator). This unblocks the entire suite's Phase 2: trnsolver iterative refinement, trntensor `precision="dd"`, and the path to HPL-MxP. The Ozaki-scheme direction (adaptive split decomposition with VectorE residual estimation running concurrently with TensorEngine splits) is the long-term target.

**HPL-MxP:** A trnsci HPL-MxP submission on Trn2 UltraServer is a Phase 2/3 goal — the first Trainium-era HPL-MxP result. The recipe: LU factorization in BF16, FP32 PSUM residual via compensated matmul, GMRES-IR outer loop.

**What doesn't fit honestly:** The per-call 100 ms Neuron XLA dispatch overhead is real and shapes kernel granularity decisions (batched-pair vs per-pair). Don't claim dispatch improvements that aren't measured.

---

### trnrand

**Fit:** Clean. Monte Carlo is famously low-precision-tolerant because statistical error (1/√N) usually dominates arithmetic error. SR is the correct default for any accumulation of random draws.

**Architectural fit:** GpSimd is the right home for Philox and Threefry — 8 fully-programmable 512-bit vector cores running straight-line integer arithmetic, none of which requires the Tensor Engine. Box-Muller transcendentals belong on VectorE. SBUF-resident output for downstream fusion (trnfft noise injection, trnblas stochastic trace) is the key performance argument.

**Philox status:** Blocked on aws-neuron-sdk#1308 (no exact integer arithmetic above 2²⁴). Threefry4×32-20 is the hardware-validated path — designed for exactly this class of hardware constraint. Philox validation reopens when AWS ships a bitwise-exact primitive.

**What doesn't fit honestly:** Don't publish Philox hardware benchmarks until aws-neuron-sdk#1308 resolves. The distribution mean 0.31 vs expected 0.5 is documented; pretending otherwise would be a credibility problem.

---

### trnsolver

**Fit:** Clean for dense eigensolvers and Krylov. Jacobi's batched-sweep form (n/2 rotations commuting on disjoint pairs → one kernel dispatch per round → stable NEFF cache) is the canonical NKI-native eigensolver. Householder-QR for eigh and Schur fits once the NEFF compile graph is stable. GMRES-IR (BF16 inner solve, FP32 residual via PSUM, outer refinement) is the direction for general linear systems.

**Carson–Higham framing:** trnsolver's `solve_spd(iterative_refinement=True)` is already in the Carson–Higham three-precision regime (κ ≲ 10⁷). The Phase 2 goal is extending this to general systems + GMRES inner solve, and exposing it as `target_forward_error`.

**Phase 2/3 opportunity:** Randomized eigensolvers (Nakatsukasa–Tropp sRR, SIAM SIMAX 45(2), 2024) use non-orthogonal bases and only need subspace-embedding-quality arithmetic — exactly what BF16+FP32-accumulate provides. Reported 10× over MATLAB eigs at FP64; 50–100× play at BF16. Natural next API: `eigh_randomized(A, k, oversampling=10)`.

**What doesn't fit honestly:** NEFF compile overhead at 303 workdirs per test run is a documented result. Every architecture post should be honest that the simulator validates kernel math but only hardware validates host-integration shape.

---

### trnsparse

**Fit:** Clean after BSR reformulation. The key architectural fact: the Tensor Engine is a 128×128 systolic array; `nisa.nc_matmul` consumes a 128×K×N tile. CSR's arbitrary-pattern indirect gather is hostile to systolic arrays. BSR at 128×128 — one `nc_matmul` per nonzero block — is the native format.

**New pattern worth codifying:** `chebyshev_bsr` and `richardson_bsr` are fixed-K iterations with **no inner products**. Unlike CG or GMRES, they do not require dot products of length n at each step. For low-precision regimes, SR-tolerant convergence without accumulating inner-product error is the correct design pattern. This should be documented as a principled choice, not an omission.

**BLR/HODLR direction:** For kernels, integral equations, covariance matrices, and dense-PDE discretizations, hierarchical-matrix compression maps cleanly — store off-diagonal low-rank blocks in BF16, diagonal blocks in FP32. Carson–Chen–Liu (SIAM J. Sci. Comput., 2024) proves the mixed-precision HODLR error stays bounded. This is the Phase 2+ opportunity.

**What doesn't fit honestly:** Fully unstructured CSR SpMV with near-diagonal structure (random graphs, FEM on unstructured meshes) is a shape mismatch. Recommend CPU fallback for those cases and say so.

---

### trntensor

**Fit:** Clean. The DF-MP2 end-to-end pipeline — AO→MO transform, pair-energy contractions, elementwise denominators, reductions — is the current flagship workflow. The fused multi-step primitives (contract → SBUF-resident → contract) are the key design: what cuTENSOR hides behind an opaque Plan, NKI exposes as writable kernel source.

**Dispatch granularity is the architecture.** The ~100 ms Neuron XLA dispatch overhead (current, pre-SDK-2.30) determines kernel boundary decisions: one dispatch per loop, not one per iteration. The `_try_batched_multi_einsum` detection and K/M/N-tiling are both responses to this. Under `torch.compile` + neuron backend (SDK 2.30), the overhead drops to ~µs — kernel boundaries can be revisited then.

**`precision=` as scaffolding.** The `precision=` kwarg (`fast`/`kahan`/`dd`) is scaffolding for the `target_forward_error` API direction. `precision="dd"` is gated on trnblas#22. When that lands, trntensor's multi-contraction path gets FP32-accuracy output from BF16 inputs via compensated matmul.

**What doesn't fit honestly:** 3+ operand einsums that fall through to `torch.einsum` lose both the optimal-order choice and the fused-DAG opportunity. Document this explicitly rather than claiming `einsum()` handles all cases.
