# NKI validation status

A snapshot of where each sub-project stands on
[Phase 1](roadmap.md#phase-1-single-chip-correctness) — the "NKI
kernels replace stubs and match the PyTorch reference on real trn1/trn2
hardware" milestone.

**Three gates**, in order of how cheaply they can be run:

1. **Simulator gate** — `nki.simulate(kernel)(numpy_args)` runs the NKI program on CPU with no device and no NEFF compile. Exercised in every PR via the `nki-simulator` CI job on `ubuntu-latest` across all six libraries. Catches Python-layer correctness, shape mismatches, and API drift.
2. **Hardware gate** — `@pytest.mark.neuron` tests dispatched to a per-repo `trn1.2xlarge` via SSM, run manually by maintainers via `scripts/run_neuron_tests.sh`. Catches MLIR-verifier issues, numerical behavior, and real NEFF compile.
3. **Canonical reference** — published spec vectors, PyTorch-parity tolerance, or scipy / PySCF / LAPACK agreement, depending on the library.

Per-kernel details live in each sub-project's `docs/architecture.md`. Per-phase definitions live in the suite [roadmap](roadmap.md).

## Status by sub-project

| Sub-project | Phase 1 tracker | Status | NKI kernels | Blog retrospective |
|---|---|---|---|---|
| [trnfft](https://github.com/trnsci/trnfft) | [#51](https://github.com/trnsci/trnfft/issues/51) | ✅ Hardware-validated | butterfly FFT, batched FFT, complex GEMM, DFT-as-GEMM fast path, Kahan butterfly | [trnfft: FFT on hardware that doesn't want to be an FFT engine](https://trnsci.dev/blog/trnfft-fft-on-hardware-that-doesnt-want-to-be-an-fft-engine/) |
| [trnblas](https://github.com/trnsci/trnblas) | [#21](https://github.com/trnsci/trnblas/issues/21) | ✅ Hardware-validated | GEMM, SYRK, fused DF-MP2 energy reduction | [trnblas: fusing DF-MP2 energy reduction into one NKI kernel](https://trnsci.dev/blog/trnblas-fusing-df-mp2-energy-reduction-into-one-nki-kernel/) |
| [trnrand](https://github.com/trnsci/trnrand) | [#18](https://github.com/trnsci/trnrand/issues/18) | ✅ Hardware-validated | Threefry4×32-20 (the NKI generator), Philox 4×32-10, Box-Muller, Sobol / Halton | [trnrand: RNG is a four-engine workload](https://trnsci.dev/blog/trnrand-rng-is-a-four-engine-workload-if-the-silicon-lets-you-say-so/) |
| [trnsolver](https://github.com/trnsci/trnsolver) | [#26](https://github.com/trnsci/trnsolver/issues/26) | ✅ Hardware-validated | Householder `eigh` (single tile, n ≤ 128), batched and block Jacobi `eigh` (`method="jacobi"`, opt-in multi-core), Newton–Schulz inverse-sqrt, CG / GMRES with Jacobi preconditioner | [trnsolver: Jacobi for Trainium](https://trnsci.dev/blog/trnsolver-jacobi-for-trainium--when-the-hardware-inverts-the-algorithm-choice/) |
| [trnsparse](https://github.com/trnsci/trnsparse) | [#14](https://github.com/trnsci/trnsparse/issues/14) | ✅ Hardware-validated | BSR-128 SpMM (static-structure and gather), fused screened SpMM, block-sparse attention, fused CG / PCG (`cg_bsr_fused`) | [trnsparse: the tile is the unit, not the nonzero](https://trnsci.dev/blog/trnsparse-the-tile-is-the-unit-not-the-nonzero/) |
| [trntensor](https://github.com/trnsci/trntensor) | [#27](https://github.com/trnsci/trntensor/issues/27) | ✅ Hardware-validated | fused MP2 energy, AO→MO transform, fused DF exchange / Coulomb (`df_exchange`, `df_coulomb`); the matmul / bmm kernels are deprecated on the native device | [trntensor: when the kernel boundary is the API](https://trnsci.dev/blog/trntensor-when-the-kernel-boundary-is-the-api/) |

## Trn2 / native TorchNeuron (Neuron SDK 2.33)

All six libraries ship native TorchNeuron dispatch (`nki_op` on the `neuron` device) next to the PyTorch/XLA path; the native ports merged on 2026-10-30. The current PyPI releases were validated on a trn2.3xlarge with Neuron SDK 2.33.0 (TorchNeuron public beta, NKI 0.7.0, neuronx-cc 2.28) on 2026-11-01, with the NKI trace cache at its TorchNeuron default (on): trnfft 0.26.0, trnblas 0.9.1, trnrand 0.12.1, trnsolver 0.20.0, trnsparse 0.12.1 and trntensor 0.22.0.

On the native device each library's `auto` dispatch sends an op to an NKI kernel only where the kernel measured faster than aten on the device, and to aten otherwise. Dense GEMMs (trnblas) and plain einsum matmul / bmm (trntensor) go to aten; the fused kernels (DF-MP2 pair energy, AO→MO, BSR SpMM, block-sparse attention) are where NKI wins. trnfft has no aten alternative to route to, because aten has no FFT on the device. In most libraries a kernel error falls back to torch with a warning unless `<LIB>_REQUIRE_NKI=1` is set.

Scoreboard and evidence: [trnsci#35](https://github.com/trnsci/trnsci/issues/35). Platform policy: [installation](installation.md#hardware-compatibility).

**Legend:**

- ✅ **Hardware-validated** — kernels pass `@pytest.mark.neuron` on hardware (trn1 on the XLA path, Trn2 on the native path). On the native device, `auto` picks the NKI kernel where it measured faster than aten.
- 🕑 **Simulator-validated** — kernels pass `nki-simulator` CI; hardware runs queued. PyTorch is the default until hardware passes.
- 🚧 **Simulator-validated, upstream-blocked** — kernels compile and run; the library has a named, trackable NKI primitive gap preventing numerically-correct hardware output. See the blog retrospective for specifics.

## Cross-suite infrastructure

- **NKI 0.3.0 migration** complete across all six libraries. Coordination tracked in [trnsci/trnsci#5](https://github.com/trnsci/trnsci/issues/5). Narrative: [The dev loop just got a lot shorter](https://trnsci.dev/blog/the-dev-loop-just-got-a-lot-shorter/).
- **`nki-simulator` CI gate** on `ubuntu-latest` for every library. Fast iteration for Python-layer correctness; does not replace hardware for MLIR-verifier or numerical-behavior checks.
- **Hardware CI** runs manually via per-repo `scripts/run_neuron_tests.sh` against a `<repo>-ci-trn1` instance.

## Looking ahead

- **Phase 1 trackers are closed for all six libraries.** trnrand's Philox kernel, long attributed to [aws-neuron-sdk#1308](https://github.com/aws-neuron/aws-neuron-sdk/issues/1308), was failing on a saturating int32/uint32 cast in the kernel itself; trnrand 0.8.0 fixed it (trnrand#56).
- Phases 2–5 (precision, single-chip perf, multi-chip, generation-specific optimization) build on top of Phase 1 per the [roadmap](roadmap.md) and are tracked per sub-project via the matching `phase-N` labels.

## Design RFCs

Sub-projects with published design docs for upcoming phases:

- [trnblas: fused DF-MP2 pair-energy kernel (Phase 3)](https://trnsci.dev/trnblas/design/fused_df_mp2_energy_kernel/) — collapse `(T*(2T−T.T)/denom).sum()` into one SBUF-resident Vector+Scalar pass; targeted 3–6× on the DF-MP2 hot path. Shipped as `pair_energy_sym`; on Trn2 it measures 1.8× faster than aten (an earlier 92× figure was wrong, trnblas#57).
- [trnrand: SBUF-resident streaming Generator (Phase 3)](https://trnsci.dev/trnrand/design/sbuf_resident_generator/) — pre-compiled multi-distribution kernel with pipelined GpSimd / Vector / Scalar engines.
- [trnrand: counter-partitioned multi-chip RNG (Phase 4)](https://trnsci.dev/trnrand/design/counter_partitioned_multichip/) — bit-exact cross-cluster-size RNG via Philox counter-space partitioning.

## Maintenance

This page is updated when a Phase 1 tracker closes (or opens, for any
new sub-project added to the suite). Historic state lives in the git
history — no versioning beyond that.
