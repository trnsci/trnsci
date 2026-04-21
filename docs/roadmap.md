# Roadmap

trnsci develops in five phases. The ordering is intentional — later phases only pay off when earlier ones are solid. Not every sub-project moves through every phase at the same pace; what matters is that a given sub-project has a credible Phase 1 before anyone starts caring about its Phase 3.

The detailed version of this roadmap, with tracking labels and cross-project dependencies, lives in [`ROADMAP.md`](https://github.com/trnsci/trnsci/blob/main/ROADMAP.md) on the umbrella repo.

## Phase 1 — Single-chip correctness

Every public API runs through a real NKI kernel on a trn1 or trn2 NeuronCore, matching the PyTorch reference within float tolerance. Phase 1 replaces dispatch stubs with real kernels and validates them against published test vectors and cross-library integration demos.

**Data-parallel multi-chip** (replica per chip, embarrassingly parallel) falls out of Phase 1 automatically — no new code beyond a worker loop.

## Phase 2 — Precision and numerical validation

The Carson–Higham three-precision iterative-refinement framework, Ozaki-scheme FP64 emulation on BF16/FP8 tensor units, and stochastic-rounding-enabled Krylov solvers. Concretely:

- **trnblas#22 — double-double FP64 GEMM** (keystone). A `nc_matmul_compensated` kernel exploiting PSUM as a free FP32 accumulator to deliver FP32-accuracy output from BF16 inputs at ~2× the work of naive BF16 matmul. This unblocks trnsolver iterative refinement, trntensor `precision="dd"`, and the Phase 2 direction for every library that does linear algebra.
- **Compensated reduction chains** (Kahan/Neumaier) in trnsolver cg/gmres inner products and trntensor contractions.
- **SR-enabled BF16 Krylov** — stochastic rounding is already in the NKI ISA; wiring it as the default for long reductions (n ≫ √(1/u) ≈ 300 for BF16) turns convergence from aspirational to structural.
- **`target_forward_error` API contract** — unify trnsolver's `iterative_refinement=True`, trntensor's `precision=`, and trnfft's `precision=` into a cross-package `target_forward_error` kwarg. The user states "solve to ε"; trnsci selects factorization precision, Ozaki split count, and IR iterations automatically, returning an achieved-error bound.

Validation targets: DF-MP2 to nanohartree against PySCF at aug-cc-pVDZ (currently passing at cc-pVDZ), and HPL-MxP single-node on Trn2.

## Phase 3 — Single-chip performance + randomized NLA flagship

**Performance:** NKI path becomes meaningfully faster than the PyTorch fallback and competitive per-dollar against vintage-matched NVIDIA instances (A10G vs Trn1, H100 vs Trn2, GB200 vs Trn3). Work per kernel: tile-shape sweeps, operand-stationarity choices, multi-engine scheduling, operation fusion, NEFF compile-cache reuse.

**Randomized NLA flagship:** ship `rsvd`, `hutch_plus_plus`, and `sRR` (sketched Rayleigh–Ritz, Nakatsukasa–Tropp 2024) as named entry points — natural home in trnsolver or a new `trnrandla` sibling, using existing trnblas, trnrand, and trnsolver.qr primitives. The flagship benchmark: randomized SVD of a 10⁶-dimensional kernel matrix on Trn2 UltraServer at BF16, verified to FP32 accuracy via iterative refinement, with bitwise-reproducible output given a seed. This is the highest-visibility Phase 3 differentiator — no GPU library can match ISA-level SR + documented-deterministic systolic reduction + Ozaki adaptivity in one package.

Additional Phase 3 targets: HPL-MxP multi-node submission on Trn2 UltraServer, BLR/HODLR direct solvers (trnsparse + trnblas frontal kernels), iterative FFT refinement research (trnfft Phase 2 research thread).

## Phase 4 — Model-parallel multi-chip

Workloads whose tensors exceed a single chip's HBM — large-basis DF-MP2, N > 2²⁴ FFTs, sparse systems with >1B nonzeros — sharded across NeuronCores within a chip and chips within an instance. Introduces `ShardedTensor` abstractions, collective operations (blocked on `nki.collectives`, expected in Neuron SDK 2.30+), and dispatch glue transparent to the user. HPL-MxP UltraServer submission.

Cross-instance (EFA-interconnected) multi-chip is a Phase 4 follow-up.

## Phase 5 — Generation-specific optimization + research

Trn1 (NeuronCore-v2) and Trn2/Trn3 (NeuronCore-v3) generation-specific fast paths: Trn3 MXFP8/MXFP4 microscaling as a native codepath for block-structured problems (directly applicable to mixed-precision HODLR and progressive-precision multigrid). Common Phase 3 paths stay the default; generation-specific paths are opt-in.

Research contributions back to the community: iterative FFT refinement theory, compensated systolic matmul error bounds (formal analysis for BF16+FP32-accum with ISA-level SR on a documented-deterministic array), SR-as-preconditioner analysis (Dexter et al. 2024 showed SR regularizes tall-skinny matrices — can this be formalized as a zero-memory preconditioner?), mixed-precision CCSD(T) cancellation analysis.

New domains: mixed-precision multigrid (`trnpde` or trnsolver submodule, Phase 3+), scientific optimization — LBFGS/ADMM/trust-region Newton (trnsolver submodule, Phase 3+), FEAST eigensolver via contour integration (Phase 3, maps naturally onto Trn3 UltraServer's 144 chips across quadrature points).

---

## Where each library is today

*As of April 2026.*

| Package | Version | Phase | Status |
|---|---|---|---|
| **trnfft** | v0.15.0 | Phase 1 ✓ / Phase 2 active | Hardware-validated on trn1.2xlarge (70/70 cases). DFT-as-GEMM fast path (up to 14× at N ≤ 256), Stockham radix-4/8 with Tensor-engine W₈, compensated butterfly in `precision="kahan"` mode. Iterative FFT refinement open as research. |
| **trnblas** | v0.5.4 | Phase 1 ✓ | GEMM, SYRK, batched GEMM, fused DF-MP2 energy reduction hardware-validated. DF-MP2 matches PySCF to 10 µHa at cc-pVDZ on H₂O/CH₄/NH₃. FP32 precision sufficient to nanohartree at target molecules (trnblas#20 closed). Phase 2 next: double-double FP64 GEMM (trnblas#22). |
| **trnsolver** | v0.9.0 | Phase 1 hardware-pending | Full API shipped: eigh, eigh_generalized, Cholesky/LU/QR/SVD/pinv, cg/gmres with iterative refinement (κ ≲ 10⁷, Carson–Higham regime), Schur decomposition. NKI Jacobi kernel simulator-validated; hardware validation pending. |
| **trnsparse** | v0.4.3 | Phase 1 partial | BSRMatrix (128×128), bsr_spmm, screened_spmm, block-sparse attention, cg_bsr, chebyshev_bsr, richardson_bsr shipped. Hardware-validated via densify-then-GEMM path. Native BSR NKI kernel hardware validation pending. |
| **trntensor** | v0.3.0 | Phase 1 hardware-pending | Einsum with greedy path planner, CP/Tucker/TT decompositions, homogeneous-contraction batching, `ShardedTensor` for output-parallel multi-chip, `to_xla`/`from_xla` residency. Hardware validation pending. |
| **trnrand** | v0.4.1 | Phase 1 hardware-pending | Philox 4×32 and Threefry4×32-20 NKI kernels (Threefry hardware-validated for uniforms); Box-Muller NKI kernel; full distribution suite (normal, exponential, Bernoulli, etc.); Sobol/Halton/LHS QMC. Philox blocked on aws-neuron-sdk#1308. |

## How to follow along

- Suite-wide coordination and cross-project dependencies: [issues on `trnsci/trnsci`](https://github.com/trnsci/trnsci/issues)
- Per-library roadmap tracking: labels `phase-1-correctness` / `phase-2-precision` / `phase-3-perf` / `phase-4-multichip` / `phase-5-generation` on each sub-project's issue tracker
- NKI-native migration tracking: [trnsci/trnsci#35](https://github.com/trnsci/trnsci/issues/35) — removing torch-xla, targeting Neuron SDK 2.30
- Releases: each library versions independently. Current PyPI versions at [pypi.org/project/trnsci/](https://pypi.org/project/trnsci/)
- Blog: [trnsci.dev/blog/](https://trnsci.dev/blog/) — retrospectives on shipped work, honest about what worked and what didn't
