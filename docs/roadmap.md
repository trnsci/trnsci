# Roadmap

trnsci develops in five phases. The ordering is intentional — later phases only pay off when earlier ones are solid. Not every sub-project moves through every phase at the same pace; what matters is that a given sub-project has a credible Phase 1 before anyone starts caring about its Phase 3.

The detailed version of this roadmap, with tracking labels and cross-project dependencies, lives in [`ROADMAP.md`](https://github.com/trnsci/trnsci/blob/main/ROADMAP.md) on the umbrella repo.

## Phase 1 — Single-chip correctness

Every public API runs through a real NKI kernel on a trn1 or trn2 NeuronCore, matching the PyTorch reference within float tolerance. Today, across the suite, the NKI kernels are scaffolded but dispatch falls back to PyTorch. Phase 1 replaces the stubs with real kernels and validates them against published test vectors and cross-library integration demos.

**Data-parallel multi-chip** (replica per chip, embarrassingly parallel) falls out of Phase 1 automatically. No new code beyond a worker loop.

## Phase 2 — Precision and numerical validation

Trainium's Tensor Engine is FP32-only. For workloads where that matters — quantum chemistry in particular — Phase 2 delivers double-double FP64 emulation, Kahan / Neumaier compensated summation for long reduction chains, and iterative-refinement variants where they beat the eigendecomposition-based approaches.

Validation targets include DF-MP2 energies to nanohartree tolerance against PySCF, and long-chain FFT round-trips against scipy.

Not every library needs Phase 2. `trnrand` is precision-neutral; `trnsparse` and `trntensor` inherit their precision story from `trnblas`.

## Phase 3 — Single-chip performance

The NKI path becomes meaningfully faster than the PyTorch fallback and competitive on a per-dollar basis with NVIDIA on vintage-matched instances (A10G vs trn1, H100 vs trn2).

Work in this phase is per-kernel: tile-shape sweeps, operand-stationarity choices, multi-engine scheduling, operation fusion, NEFF compile-cache reuse, plan and contraction-plan re-execution. Published benchmarks on each sub-project's docs page provide the CPU baseline, the vintage-matched GPU baseline, and the Trainium numbers side-by-side.

## Phase 4 — Model-parallel multi-chip

Workloads whose tensors exceed a single chip's HBM — large-basis DF-MP2, N > 2^24 FFTs, sparse systems with >1B nonzeros — are sharded across NeuronCores within a chip and chips within an instance.

This phase introduces sharded tensor abstractions, collective operations, and the dispatch glue that makes model-parallel operation transparent to the library user.

Cross-instance (EFA-interconnected) multi-chip is a Phase 4 follow-up if demand warrants it.

## Phase 5 — Generation-specific optimization

trn1 (NeuronCore v2) and trn2 (NeuronCore v3) get fast paths that exploit their respective architectural strengths — trn2's larger SBUF, wider PSUM, and FP16 accumulate are the main levers today — without requiring the maintainer to track two separately tuned code paths.

The common Phase 3 path stays the default. Generation-specific paths are opt-in via backend selection or automatic based on runtime capability detection.

## Where each library is today

All six sub-projects are in **late Phase 1** — APIs stable, PyTorch fallback wired everywhere, NKI dispatch layer in place, hardware validation pending for each kernel. `trnblas` has published DF-MP2 results against PySCF at nanohartree tolerance on small molecules, which touches Phase 2 territory. `trnsolver` has published CPU baselines against LAPACK and scipy, which sets up Phase 3.

See each sub-project's docs (`trnsci.dev/<name>/`) for library-specific status.

## How to follow along

- Suite-wide coordination and cross-project dependencies: [issues on `trnsci/trnsci`](https://github.com/trnsci/trnsci/issues).
- Per-library roadmap tracking: labels `phase-1-correctness` / `phase-2-precision` / `phase-3-perf` / `phase-4-multichip` / `phase-5-generation` on each sub-project's issue tracker.
- Releases: each library versions independently. Current PyPI versions at [pypi.org/project/trnsci/](https://pypi.org/project/trnsci/).
