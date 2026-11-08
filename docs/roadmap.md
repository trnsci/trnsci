# Roadmap

trnsci develops in five phases. The ordering is intentional — later phases only pay off when earlier ones are solid. Not every sub-project moves through every phase at the same pace; what matters is that a given sub-project has a credible Phase 1 before anyone starts caring about its Phase 3.

The live roadmap is on GitHub: one milestone per phase, an issue per unit of work, and the [trnsci suite project board](https://github.com/orgs/trnsci/projects/1). This page describes the phases and where each library stands; the milestones carry the work itself.

| Phase | Umbrella milestone | Per-library milestones |
|---|---|---|
| 1 — Single-chip correctness | [trnsci/milestone/1](https://github.com/trnsci/trnsci/milestone/1) | same name in each library repo |
| 2 — Precision and numerical validation | [trnsci/milestone/2](https://github.com/trnsci/trnsci/milestone/2) | [trnfft](https://github.com/trnsci/trnfft/milestones) · [trnblas](https://github.com/trnsci/trnblas/milestones) · [trnrand](https://github.com/trnsci/trnrand/milestones) · [trnsolver](https://github.com/trnsci/trnsolver/milestones) · [trnsparse](https://github.com/trnsci/trnsparse/milestones) · [trntensor](https://github.com/trnsci/trntensor/milestones) |
| 3 — Single-chip performance | [trnsci/milestone/3](https://github.com/trnsci/trnsci/milestone/3) | as above |
| 4 — Model-parallel multi-chip | [trnsci/milestone/4](https://github.com/trnsci/trnsci/milestone/4) | as above |
| 5 — Generation-specific optimization | [trnsci/milestone/5](https://github.com/trnsci/trnsci/milestone/5) | as above |
| Cross-cutting (exemplars, infrastructure, docs) | [trnsci/milestone/7](https://github.com/trnsci/trnsci/milestone/7) | — |
| Native PyTorch migration | [trnsci/milestone/6](https://github.com/trnsci/trnsci/milestone/6) | — |

The phase definitions ("done means") are also kept in [`ROADMAP.md`](https://github.com/trnsci/trnsci/blob/main/ROADMAP.md); the suite-wide tracker is [trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## Phase 1 — Single-chip correctness

Every public API runs through a real NKI kernel on a trn1 or trn2 NeuronCore, matching the PyTorch reference within float tolerance. Phase 1 replaces dispatch stubs with real kernels and validates them against published test vectors and cross-library integration demos.

**Data-parallel multi-chip** (replica per chip, embarrassingly parallel) falls out of Phase 1 automatically — no new code beyond a worker loop.

**Status:** done. The Phase 1 trackers are closed in all six libraries. Since November 2026 "matching the PyTorch reference" is no longer the bar on its own: each library also runs a conformance suite against independent references ([trnsci#105](https://github.com/trnsci/trnsci/issues/105)), because the pytorch-vs-NKI comparison passed bugs that both paths shared ([the silent wrong answers post](blog/posts/2026-11-04-suite-silent-wrong-answers.md)).

## Phase 2 — Precision and numerical validation

The Carson–Higham three-precision iterative-refinement framework, Ozaki-scheme FP64 emulation on the fp32-accumulating Tensor Engine, and compensated reductions. Concretely:

- **trnblas#22 — fp64-class GEMM on the device** (keystone). Error-compensated GEMM with PSUM as the FP32 accumulator: Ozaki splitting into slices whose products are exact in fp32, accumulated with compensation. The [device-fp64 epic, trnsci#129](https://github.com/trnsci/trnsci/issues/129), tracks the three consumers whose host fp64 step is now the measured bottleneck: the canonical Fock of large-basis DF-RHF/MP2, the GP preconditioner build, and GP refinement. CPU design results so far: a plain fp32 hi/lo double-double stays at fp32 accuracy on an fp32 accumulator, and Ozaki slices reach 1e-12 at 15 slice GEMMs ([trnblas#22 comment](https://github.com/trnsci/trnblas/issues/22#issuecomment-6718019677)). Device measurements are pending.
- **Compensated reduction chains** (Kahan/Neumaier) in trnsolver cg/gmres inner products and trntensor contractions.
- **Stochastic rounding where it applies.** On NKI 0.7 (Neuron SDK 2.33) the rounding mode is a process-wide runtime setting (`NEURON_RT_STOCHASTIC_ROUNDING_EN`), not a per-instruction one, and results with it on are not bitwise reproducible. SR removes bias from long accumulations in a narrow format (BF16); it does not lift an fp32 pipeline above the fp32 floor ([the fp32-floor post](blog/posts/2026-11-08-dfmp2-fp32-floor.md)). trntensor's `precision="sr"` documents the runtime requirement.
- **`target_forward_error` API contract** — unify trnsolver's `iterative_refinement=True`, trntensor's `precision=`, and trnfft's `precision=` into a cross-package `target_forward_error` kwarg. The user states "solve to ε"; trnsci selects factorization precision, Ozaki split count, and IR iterations automatically, returning an achieved-error bound.

**Shipped:** the suite float64 policy and a dtype guard at every NKI kernel boundary (all six libraries, November 2026), and the conformance suites. See [precision and platform limits](precision_and_platform_limits.md).

**Validation targets.** The fp32 device route for DF-MP2 is gated at 1e-6 Eh against an fp64 reference; 1e-7 Eh and below is assigned to device fp64 ([trnsci#129](https://github.com/trnsci/trnsci/issues/129)). DF-RHF energy within 1e-8 Eh of host fp64 at 1658 AOs is the epic's chemistry acceptance target. HPL-MxP single-node on Trn2 remains a target.

## Phase 3 — Single-chip performance + randomized NLA flagship

**Performance:** NKI path becomes meaningfully faster than the PyTorch fallback and competitive per-dollar against vintage-matched NVIDIA instances (A10G vs Trn1, H100 vs Trn2, GB200 vs Trn3). Work per kernel: tile-shape sweeps, operand-stationarity choices, multi-engine scheduling, operation fusion, NEFF compile-cache reuse.

**Application exemplars.** Phase 3 is now measured against whole workloads on one Trn2 chip rather than per-kernel speedups: [exact GP regression at n = 10⁶](exemplars.md#gp), the [spherical harmonic transform](exemplars.md#sht), [large-basis DF-RHF/MP2 to 3508 AOs](exemplars.md#large-basis), and [DeepVariant `call_variants`](exemplars.md#deepvariant), with [plane-wave DFT](exemplars.md#plane-wave) planned. Each reports three CPU baselines ([trnsci#112](https://github.com/trnsci/trnsci/issues/112)). Off-box NEFF compile-cache warming ([trnsci#118](https://github.com/trnsci/trnsci/issues/118), `scripts/warm_neuron_cache.py`) is the infrastructure piece.

**Randomized NLA flagship:** ship `rsvd`, `hutch_plus_plus`, and `sRR` (sketched Rayleigh–Ritz, Nakatsukasa–Tropp 2024) as named entry points — natural home in trnsolver or a new `trnrandla` sibling, using existing trnblas, trnrand, and trnsolver.qr primitives. trnsolver 0.22 already ships `cg_block`, pivoted-Cholesky preconditioning and `slq_logdet` for the GP exemplar. The flagship benchmark: randomized SVD of a 10⁶-dimensional kernel matrix on Trn2 UltraServer, verified to FP32 accuracy via iterative refinement, with bitwise-reproducible output given a seed.

Additional Phase 3 targets: HPL-MxP multi-node submission on Trn2 UltraServer, BLR/HODLR direct solvers (trnsparse + trnblas frontal kernels), iterative FFT refinement research (trnfft Phase 2 research thread).

## Phase 4 — Model-parallel multi-chip

Workloads whose tensors exceed a single chip's HBM — large-basis DF-MP2, N > 2²⁴ FFTs, sparse systems with >1B nonzeros — sharded across NeuronCores within a chip and chips within an instance. Introduces `ShardedTensor` abstractions, collective operations (`nki.collectives`, which trntensor's reduce-parallel einsum has used across the four logical cores of a trn2.3xlarge since v0.17.0), and dispatch glue transparent to the user. HPL-MxP UltraServer submission.

**Next step:** a trn2.48xlarge window ([trnsci#132](https://github.com/trnsci/trnsci/issues/132), purchase needs sign-off) for SCF at ≥ 3500 AOs with the integrals sharded across HBM ([trnsolver#125](https://github.com/trnsci/trnsolver/issues/125)), GP at n = 10⁷ ([trnsparse#116](https://github.com/trnsci/trnsparse/issues/116)) and DeepVariant whole-genome runs. Cross-chip exchange measurements ([trntensor#118](https://github.com/trnsci/trntensor/issues/118)) decide host staging against device collectives; a cross-core `.to("neuron:k")` currently does not move data ([aws-neuron-sdk#1451](https://github.com/aws-neuron/aws-neuron-sdk/issues/1451)).

Cross-instance (EFA-interconnected) multi-chip is a Phase 4 follow-up.

## Phase 5 — Generation-specific optimization + research

Trn1 (NeuronCore-v2) and Trn2/Trn3 (NeuronCore-v3) generation-specific fast paths: Trn3 MXFP8/MXFP4 microscaling as a native codepath for block-structured problems (directly applicable to mixed-precision HODLR and progressive-precision multigrid). Common Phase 3 paths stay the default; generation-specific paths are opt-in.

Research contributions back to the community: iterative FFT refinement theory, compensated systolic matmul error bounds (formal analysis for BF16+FP32-accum on a documented-deterministic array), SR-as-preconditioner analysis (Dexter et al. 2024 showed SR regularizes tall-skinny matrices — can this be formalized as a zero-memory preconditioner?), mixed-precision CCSD(T) cancellation analysis.

New domains: mixed-precision multigrid (`trnpde` or trnsolver submodule, Phase 3+), scientific optimization — LBFGS/ADMM/trust-region Newton (trnsolver submodule, Phase 3+), FEAST eigensolver via contour integration (Phase 3, maps naturally onto Trn3 UltraServer's 144 chips across quadrature points).

---

## Where each library is today

*As of 2026-11-08.* Every library ships native TorchNeuron dispatch on Trn2/Trn3 (public Neuron SDK 2.33) next to the legacy PyTorch/XLA path on Trn1, and each release below was validated on a trn2.3xlarge ([validation status](nki_validation_status.md), [#35](https://github.com/trnsci/trnsci/issues/35)). On the native device, `auto` routes each op to an NKI kernel only where it measured faster than aten, and float64 follows the [suite policy](precision_and_platform_limits.md#fp64-policy). The version column is the release each row describes; the site build checks it against PyPI and names any newer release. Each CHANGELOG has the detail, and each README a "Known issues in earlier releases" table.

| Package | Version | Phase | State |
|---|---|---|---|
| **trnfft** | [[described:trnfft=0.29.0]] · [changes](https://github.com/trnsci/trnfft/blob/main/CHANGELOG.md) | Phase 1 ✓ / Phase 2–3 active | The only FFT on the device (`torch.fft` doesn't run there). New `trnfft.sht` spherical harmonic transform with SBUF-stationary Legendre kernels; `irfft` no longer calls `torch.flip`, which took the SHT inverse at L = 511 from 1002 ms to 34 ms. |
| **trnblas** | [[described:trnblas=0.12.1]] · [changes](https://github.com/trnsci/trnblas/blob/main/CHANGELOG.md) | Phase 1 ✓ / Phase 2–3 active | Dense GEMM goes to aten except the DF-MP2 half-transform, where an LNC-split NKI GEMM is 1.5–1.7× faster; fused `pair_energy_sym` is 1.8× aten. 0.12.1 row-blocks device matmuls with rows ≥ 32 MiB ([aws-neuron-sdk#1457](https://github.com/aws-neuron/aws-neuron-sdk/issues/1457)). Next: fp64-class GEMM ([trnblas#22](https://github.com/trnsci/trnblas/issues/22)). |
| **trnrand** | [[described:trnrand=0.14.0]] · [changes](https://github.com/trnsci/trnrand/blob/main/CHANGELOG.md) | Phase 1 ✓ / Phase 3–4 active | Threefry4×32-20 NKI generator, bit-identical across 1–4 cores; Philox, full distribution suite, QMC. 0.14.0 makes `sobol(scramble=True)` real Owen scrambling (`scramble="shift"` keeps the old behaviour) and fixes `Generator(seed=None)` repeating its stream. |
| **trnsolver** | [[described:trnsolver=0.22.1]] · [changes](https://github.com/trnsci/trnsolver/blob/main/CHANGELOG.md) | Phase 1 ✓ / Phase 2–4 active | `cg_block`, pivoted low-rank preconditioner and `slq_logdet` for matrix-free GP; `occupied_subspace`, `orthogonalizer` and a minao guess for SCF; DF-RHF/MP2 drivers in `examples/` that ran to 3508 AOs. Householder `eigh` is single-tile; batched Jacobi `eigh` is opt-in multi-core. |
| **trnsparse** | [[described:trnsparse=0.15.0]] · [changes](https://github.com/trnsci/trnsparse/blob/main/CHANGELOG.md) | Phase 1 ✓ / Phase 3–4 active | `KernelOperator` / `kernel_matmul`, a fused K(X, Y)·V that never forms K (7.3 s per apply at n = 2²⁰, 33 right-hand sides, 4 cores); BSR-128 SpMM; `screened_spmm` skips screened-out tiles; fused CG / PCG. |
| **trntensor** | [[described:trntensor=0.25.1]] · [changes](https://github.com/trnsci/trntensor/blob/main/CHANGELOG.md) | Phase 1 ✓ / Phase 3–4 active | `AuxStream` streams DF integrals over the aux index; the blocked packed AO→MO kernel runs at any n_ao (about 100× a 2-thread host per chunk at 3508 AOs); fused MP2, DF J/K. 0.25.1 row-blocks device matmuls with rows ≥ 32 MiB. |

## How to follow along

- Suite-wide coordination and cross-project dependencies: [issues on `trnsci/trnsci`](https://github.com/trnsci/trnsci/issues) and the [project board](https://github.com/orgs/trnsci/projects/1)
- Per-library tracking: the phase milestones in each sub-project's repository (table above)
- Native TorchNeuron migration and hardware log: [trnsci/trnsci#35](https://github.com/trnsci/trnsci/issues/35) — native dispatch shipped in all six libraries; the issue stays open for the XLA-path deprecation at TorchNeuron general availability ([GA checklist, #136](https://github.com/trnsci/trnsci/issues/136))
- Upstream Neuron SDK issues the suite depends on: [trnsci#135](https://github.com/trnsci/trnsci/issues/135)
- Releases: each library versions independently; the table above links each CHANGELOG
- Blog: [trnsci.dev/blog/](https://trnsci.dev/blog/) — retrospectives on shipped work, honest about what worked and what didn't
