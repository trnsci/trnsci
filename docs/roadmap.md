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

Workloads whose tensors exceed a single chip's HBM — large-basis DF-MP2, N > 2²⁴ FFTs, sparse systems with >1B nonzeros — sharded across NeuronCores within a chip and chips within an instance. Introduces `ShardedTensor` abstractions, collective operations (`nki.collectives`, which trntensor's reduce-parallel einsum has used across the four logical cores of a trn2.3xlarge since v0.17.0), and dispatch glue transparent to the user. HPL-MxP UltraServer submission.

Cross-instance (EFA-interconnected) multi-chip is a Phase 4 follow-up.

## Phase 5 — Generation-specific optimization + research

Trn1 (NeuronCore-v2) and Trn2/Trn3 (NeuronCore-v3) generation-specific fast paths: Trn3 MXFP8/MXFP4 microscaling as a native codepath for block-structured problems (directly applicable to mixed-precision HODLR and progressive-precision multigrid). Common Phase 3 paths stay the default; generation-specific paths are opt-in.

Research contributions back to the community: iterative FFT refinement theory, compensated systolic matmul error bounds (formal analysis for BF16+FP32-accum with ISA-level SR on a documented-deterministic array), SR-as-preconditioner analysis (Dexter et al. 2024 showed SR regularizes tall-skinny matrices — can this be formalized as a zero-memory preconditioner?), mixed-precision CCSD(T) cancellation analysis.

New domains: mixed-precision multigrid (`trnpde` or trnsolver submodule, Phase 3+), scientific optimization — LBFGS/ADMM/trust-region Newton (trnsolver submodule, Phase 3+), FEAST eigensolver via contour integration (Phase 3, maps naturally onto Trn3 UltraServer's 144 chips across quadrature points).

---

## Where each library is today

*As of 2026-11-01.* Every library ships native TorchNeuron dispatch on Trn2/Trn3 (Neuron SDK 2.33) next to the legacy PyTorch/XLA path on Trn1, and the releases below were validated on a trn2.3xlarge ([#35](https://github.com/trnsci/trnsci/issues/35)). On the native device, `auto` routes each op to an NKI kernel only where it measured faster than aten.

| Package | Version | Phase | Status |
|---|---|---|---|
| **trnfft** | v0.26.0 | Phase 1 ✓ / Phase 2–3 active | Single DFT-GEMM up to N = 1024, batch-fused four-step to N = 4096, six-step by default for power-of-two FP32 N ≥ 8192. Compensated butterfly in `precision="kahan"`. 0.25.0 fixed dropped GEMM tail tiles (small-N DFT-GEMM, bf16 modes, `nn.ComplexLinear`); 0.26.0 fixed `irfft`/`hfft`/`irfftn` bin selection and float64 on NKI paths. |
| **trnblas** | v0.9.1 | Phase 1 ✓ / Phase 3 active | On the native device `auto` sends dense GEMMs to aten (the NKI GEMM is 2.4–3.2× slower). The fused `pair_energy_sym` kernel is 1.8× faster than aten (an earlier 92× claim was wrong); DF-MP2 is 1.34× faster than aten end to end on synthetic inputs and 1.1–2.1× on real molecules. Host fp64 metric Cholesky / J⁻¹ᐟ² for naux ≤ 1024. Multi-core DF-MP2 in `trnblas.parallel`. Phase 2 next: double-double FP64 GEMM (trnblas#22). |
| **trnsolver** | v0.20.0 | Phase 1 ✓ / Phase 3 active | Full API: eigh, eigh_generalized, eigh_randomized (sRR), Cholesky/LU/QR/SVD/pinv, Schur, cg/gmres with iterative refinement (κ ≲ 10⁷). Householder `eigh` is single-tile (n ≤ 128); above that the aten cutover is the default. Batched Jacobi `eigh` with opt-in multi-core (`TRNSOLVER_EIGH_CORES`, worker-process pool): single core is 1.13× a thread-split CPU baseline at n = 128, B = 1024, and 4 cores are at most 1.25× vs 8 CPU threads. DF-RHF SCF example (`examples/scf_df_rhf.py`). |
| **trnsparse** | v0.12.1 | Phase 1 ✓ / Phase 3–4 active | BSRMatrix (128×128) with a static-structure SpMM kernel that beats dense aten on one core up to ~82–98% block density; `ShardedBSR` / `ShardedAttention` across cores; fused CG / PCG (`cg_bsr_fused`). Block sizes other than 128 run on torch. `block_sparse_attention_tiled` raises `ValueError` when `seq_len` is not a multiple of the block. |
| **trntensor** | v0.22.0 | Phase 1 ✓ / Phase 3 active | Einsum with greedy path planner, CP/Tucker/TT, `ShardedTensor`. Native fused kernels: MP2 energy (1.25–1.9× vs aten), AO→MO (1.3–1.6×), DF exchange / Coulomb (SCF at n_ao = 970: 9.68 s → 4.71 s). The NKI matmul / bmm kernels are deprecated on the native device and route to aten. 0.22.0 fixed einsum plans that ignored the output index order (every earlier release, every backend). |
| **trnrand** | v0.12.1 | Phase 1 ✓ / Phase 3–4 active | Threefry4×32-20 is the NKI generator (bit-identical across kernel widths, launch plans and 1–4 cores via opt-in `set_cores`); Philox 4×32 and Box-Muller kernels; full distribution suite; Sobol/Halton/LHS QMC. 0.12.0 fixed partitioned generators reusing counters (0.7.0–0.11.0); 0.12.1 fixed `get_state`/`set_state` dropping the stream counter. |

## How to follow along

- Suite-wide coordination and cross-project dependencies: [issues on `trnsci/trnsci`](https://github.com/trnsci/trnsci/issues)
- Per-library roadmap tracking: labels `phase-1-correctness` / `phase-2-precision` / `phase-3-perf` / `phase-4-multichip` / `phase-5-generation` on each sub-project's issue tracker
- Native TorchNeuron migration and hardware log: [trnsci/trnsci#35](https://github.com/trnsci/trnsci/issues/35) — native dispatch shipped in all six libraries; the issue stays open for the XLA-path deprecation at TorchNeuron general availability
- Releases: each library versions independently. Current PyPI versions at [pypi.org/project/trnsci/](https://pypi.org/project/trnsci/)
- Blog: [trnsci.dev/blog/](https://trnsci.dev/blog/) — retrospectives on shipped work, honest about what worked and what didn't
