# trnsci roadmap

This document describes the multi-phase plan for the trnsci scientific-computing suite on AWS Trainium. It applies at the suite level; per-sub-project status tracking lives in labeled GitHub issues inside each sub-project repository.

The phases are ordered such that later phases only make sense once the earlier ones are solid. Nothing prevents parallel work on later phases for parts of the suite where earlier phases are already complete.

## Phase 1 — Single-chip correctness

**Goal:** every public API runs through a real NKI kernel (not a PyTorch fallback) and produces output that matches the PyTorch reference within float tolerance on a trn1 or trn2 NeuronCore.

**Done means:**

- NKI kernels replace stubs in each sub-project's `nki/dispatch.py`.
- Hardware CI (`scripts/run_neuron_tests.sh`) is green against both PyTorch fallback and NKI path for representative test matrices.
- Cross-library integration demo (`examples/quantum_chemistry/df_mp2_synthetic.py` and the nvidia_samples ports) runs end-to-end with `set_backend("nki")`.
- Published canonical test vectors pass (e.g. Philox SC'11 reference, known eigenvalue decompositions, Bluestein round-trip).

**Data-parallel multi-chip falls out of this phase automatically** — replica-per-chip needs no new code beyond a worker loop. Model-parallel is its own phase (see Phase 4).

## Phase 2 — Precision and numerical validation

**Goal:** numerical results are good enough for the motivating scientific workloads, not just within FP32 tolerance of the PyTorch reference.

**Done means:**

- Double-double FP64 emulation for GEMM (`trnblas`) validated against scipy / LAPACK on chemistry benchmarks (DF-MP2 energies to nanohartree tolerance).
- Compensated summation (Kahan / Neumaier) in long reduction chains — Bluestein FFT (`trnfft`), CG / GMRES (`trnsolver`), tensor contractions with high inner-dimension (`trntensor`).
- Newton-Schulz and other iterative-refinement paths where they beat the eigendecomposition-based approach on the target hardware.
- Cross-library fuzz tests that catch catastrophic cancellation regressions in later optimization work.

Not every sub-project needs Phase 2 work: `trnrand` is precision-neutral, and `trnsparse` / `trntensor` inherit the BLAS precision story from `trnblas`.

## Phase 3 — Single-chip performance

**Goal:** the NKI path is meaningfully faster than the PyTorch fallback on a single NeuronCore, and competitive on a per-dollar basis with NVIDIA on vintage-matched instances.

**Done means:**

- Per-kernel tuning: tile shapes, stationary vs moving operand choice, dual-engine scheduling (Tensor + Vector + Scalar + GpSimd).
- Operation fusion where the dataflow allows (e.g. DF-MP2 pair-energy kernel fuses the contraction with the orbital-energy denominator).
- NEFF compile-cache reuse in all examples and benchmarks — no paying for recompiles across runs.
- Published benchmarks on `trnsci.dev/<pkg>/benchmarks/` with CPU baseline, vintage-matched GPU baseline (A10G vs trn1, H100 vs trn2), and per-shape TFLOPS.
- Plan / cache re-execution paths (FFT plans, contraction plans) validated to skip re-planning on repeated calls.

## Phase 4 — Model-parallel multi-chip

**Goal:** workloads whose tensors don't fit on one chip's HBM can be sharded across multiple NeuronCores within a chip and multiple chips within an instance.

**Done means:**

- Sharded tensor abstractions in each sub-project (`trnblas`, `trntensor`, `trnsparse`, `trnsolver`) — at least a working pattern, not necessarily a generic solution.
- Collective operations (all-reduce, all-gather, reduce-scatter) either using Neuron Collective Compute Library or equivalent.
- Large-basis DF-MP2 (>3000 basis functions) runs end-to-end on trn1.32xlarge without OOM.
- Large FFT (N > 2^24) runs with multi-chip butterflies.

Data-parallel workloads (single replicas running independently) are out of scope for this phase — they work from Phase 1 onward with a worker loop.

Cross-instance (EFA-interconnected) multi-chip is a follow-up to Phase 4 if the motivating workloads warrant it.

## Phase 5 — Generation-specific optimization

**Goal:** trn1 and trn2 each get fast paths that exploit their respective architectural strengths, without maintaining two separately tuned codebases.

**Done means:**

- Runtime capability detection (e.g. NeuronCore version, SBUF size, PSUM width) wired into the dispatch layer.
- Generation-specific NKI kernels where the payoff is meaningful (larger SBUF → fewer tile loads, wider PSUM → reduced partial-sum rounds, FP16 accumulate on trn2 → faster GEMMs where precision allows).
- Common path stays the default; generation-specific paths are opt-in via backend selection or automatic based on detected hardware.
- Forward-planning for future NeuronCore generations documented in each sub-project's architecture.md.

## Phase overview by sub-project

Rough projection of what each phase means for each library. Not every phase applies equally.

| Phase | trnfft | trnblas | trnrand | trnsolver | trnsparse | trntensor |
|---|---|---|---|---|---|---|
| 1 correctness | butterfly + complex GEMM land | GEMM + batched_gemm land | Philox + Box-Muller land | Jacobi `eigh` lands | SpMM gather-matmul-scatter lands | einsum fusion lands |
| 2 precision | Kahan for Bluestein chain | double-double GEMM | — | iterative refinement | — | precision-aware path choice |
| 3 perf | plan reuse, streaming | tile sweep, DF-MP2 hot path | batched-tile RNG | preconditioned CG / GMRES | nnz-bucketing SpMM | `opt_einsum`-style planner |
| 4 multi-chip | N-chip large FFT | tensor-parallel GEMM | stream-partitioned RNG | parallel Jacobi sweeps | sharded sparse | sharded contraction |
| 5 generation | trn2 larger SBUF path | trn2 FP16-accumulate path | trn2 PSUM sizing | trn2 rotation blocks | trn2 DMA bandwidth | trn2 fused paths |

## Current state (2026-04-13)

- **Phase 1 complete:** `trnfft` (butterfly + complex GEMM kernels, hardware-validated on trn1.2xlarge in v0.8.0) and `trnblas` (GEMM, SYRK, MP2 energy reduction, hardware-validated with end-to-end DF-MP2 timings).
- **Phase 1 scaffolded, awaiting hardware validation:** `trnrand`, `trnsolver`, `trnsparse`, `trntensor`. All four have NKI code in `nki/dispatch.py`, a provisioned Terraform module under `infra/terraform/`, and a `scripts/run_neuron_tests.sh` runner. The remaining work is a hardware-validation sweep.
- **Touching Phase 2:** `trnblas` has PySCF validation against real chemistry at nanohartree tolerance on small molecules, which is early Phase 2 precision work for dense GEMM.
- **Setting up Phase 3:** `trnsolver` has published CPU baselines against LAPACK and scipy, which is the infrastructure Phase 3 benchmarks will compare against.

Live progress counter in [`trnsci/trnsci#1`](https://github.com/trnsci/trnsci/issues/1).

## Tracking

Each sub-project uses labels `phase-1-correctness`, `phase-2-precision`, `phase-3-perf`, `phase-4-multichip`, `phase-5-generation` on issues. A sub-project's README links to its active phase-tracker issues under a "Current phase" section.

The [suite-level coordination issue](https://github.com/trnsci/trnsci/issues) thread tracks cross-project dependencies (e.g. `trnblas` double-double unblocks the chemistry precision story across `trnsolver`, `trnsparse`, `trntensor`).
