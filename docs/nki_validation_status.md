# NKI validation status

A snapshot of how each sub-project's current release was validated, and where it stands on
[Phase 1](roadmap.md#phase-1-single-chip-correctness) — the "NKI
kernels replace stubs and match the PyTorch reference on real trn1/trn2
hardware" milestone.

**Four gates**, in order of how cheaply they can be run:

1. **Simulator gate** — `nki.simulate(kernel)(numpy_args)` runs the NKI program on CPU with no device and no NEFF compile. Exercised in every PR via the `nki-simulator` CI job on `ubuntu-latest` across all six libraries. Catches Python-layer correctness, shape mismatches, and API drift.
2. **Conformance gate** — `tests/test_conformance.py` in every library checks each public API against an independent reference (see [below](#conformance)). Runs on CPU in every PR, and on the device by hand.
3. **Hardware gate** — `@pytest.mark.neuron` tests, run manually by maintainers on Trainium (Trn2 for the native path, trn1 for the XLA path) via `scripts/run_neuron_tests.sh`. There is no hardware CI. Catches MLIR-verifier issues, numerical behavior, and real NEFF compile.
4. **Canonical reference** — published spec vectors, or scipy / PySCF / LAPACK / ducc0 agreement, depending on the library.

Per-kernel details live in each sub-project's `docs/architecture.md`. Per-phase definitions live in the suite [roadmap](roadmap.md).

## Current releases on Trn2 {#current-releases}

Each current PyPI release was validated on a trn2.3xlarge with public Neuron SDK 2.33 (TorchNeuron public beta, torch 2.14, NKI 0.7.0, neuronx-cc 2.28), LNC=2, one logical core per process. Each validation ran the release's `pytest -m neuron` suite with the NKI trace cache off and on, the device conformance suite (`--backend nki --conformance-device neuron`), the CPU suite in a clean CPU-only venv, and the installed PyPI wheel. The records are comments on [#35](https://github.com/trnsci/trnsci/issues/35):

| release | validated | record | `-m neuron` | device conformance failures, all classified |
|---|---|---|---|---|
| trnfft [[described:trnfft=0.29.0]] | 2026-11-04, window 4 | [comment](https://github.com/trnsci/trnsci/issues/35#issuecomment-6625324136) | 250 passed, 0 failed | 1: a device `aten::convolution` compile failure ([trnfft#155](https://github.com/trnsci/trnfft/issues/155), platform) |
| trnblas [[described:trnblas=0.13.0]] | 2026-11-10, window 6 | [comment](https://github.com/trnsci/trnsci/issues/35#issuecomment-6756567385) | 263 passed, 0 failed | 2022: complex `mv` unsupported and device float64 at fp32 (2012), test tolerance (8), harness (2); no fp32 / bf16 / fp16 failures |
| trnrand [[described:trnrand=0.14.0]] | 2026-11-03, window 4 | [comment](https://github.com/trnsci/trnsci/issues/35#issuecomment-6616562992) | 407 passed, 0 failed | 5: the documented Poisson normal approximation at λ ≥ 20 ([trnrand#88](https://github.com/trnsci/trnrand/issues/88)) |
| trnsolver [[described:trnsolver=0.22.1]] | 2026-11-07, window 5 | [comment](https://github.com/trnsci/trnsci/issues/35#issuecomment-6710207391) ([main probe](https://github.com/trnsci/trnsci/issues/35#issuecomment-6708079805)) | 135 passed, 0 failed | 100: complex linalg unsupported (41), device float64 at fp32 (59) |
| trnsparse [[described:trnsparse=0.16.0]] | 2026-11-10, window 6 | [comment](https://github.com/trnsci/trnsci/issues/35#issuecomment-6756919429) | 315 passed, 0 failed (4 GiB opt-in 2/2) | 85: complex `mv` unsupported (64), device float64 at fp32 (12), harness (8), fp16 `sqrt` ulp (1) |
| trntensor [[described:trntensor=0.25.1]] | 2026-11-07, window 5 | [comment](https://github.com/trnsci/trnsci/issues/35#issuecomment-6711392931) (0.25.0: [comment](https://github.com/trnsci/trnsci/issues/35#issuecomment-6624197485)) | 122 passed, 0 failed | 10 at 0.25.0: complex SVD / solve unsupported (9), device float64 at fp32 (1) |

Every device conformance failure in these runs is in a documented platform category, a test-harness limit or a documented approximation, and none is new against the window-4 baselines. The platform categories are on [precision and platform limits](precision_and_platform_limits.md). The window-3 [audit](https://github.com/trnsci/trnsci/issues/35#issuecomment-6570060466) and [scoreboard](https://github.com/trnsci/trnsci/issues/35#issuecomment-6574951854), and the window-4 [conformance triage](https://github.com/trnsci/trnsci/issues/35#issuecomment-6604543845), set those baselines.

### Conformance against independent references {#conformance}

The suites used to test the NKI path against the library's own pytorch path. That passes any bug in code both paths share: argument parsing, a planner, a host wrapper. Since November 2026 every library also has `tests/test_conformance.py` ([trnsci#105](https://github.com/trnsci/trnsci/issues/105)). Each public API is checked against a reference that shares no code with it (`torch.fft`, `torch.einsum`, `torch.linalg`, scipy or a dense formula, in float64 on the host), at awkward sizes (1, 2, primes, 127 and 129 around the tile, 513, 1000) and with non-default keywords, dtypes, strides, empty shapes, `out=`, aliasing and autograd. The same cases run in CPU CI and, by hand, on the device. The audits that led to them found about 60 silent wrong-result bugs in released versions; each README lists them under "Known issues in earlier releases" ([the post](blog/posts/2026-11-04-suite-silent-wrong-answers.md)).

## Status by sub-project

| Sub-project | Phase 1 tracker | Status | NKI kernels | Blog retrospective |
|---|---|---|---|---|
| [trnfft](https://github.com/trnsci/trnfft) | [#51](https://github.com/trnsci/trnfft/issues/51) | ✅ Hardware-validated | butterfly FFT, batched FFT, complex GEMM, DFT-as-GEMM fast path, batch-fused four-step / six-step, Kahan butterfly, SHT Legendre (`trnfft.sht`) | [trnfft: FFT on hardware that doesn't want to be an FFT engine](https://trnsci.dev/blog/trnfft-fft-on-hardware-that-doesnt-want-to-be-an-fft-engine/) |
| [trnblas](https://github.com/trnsci/trnblas) | [#21](https://github.com/trnsci/trnblas/issues/21) | ✅ Hardware-validated | GEMM (incl. LNC-split GEMM for the DF-MP2 half-transform), SYRK, fused DF-MP2 energy reduction | [trnblas: fusing DF-MP2 energy reduction into one NKI kernel](https://trnsci.dev/blog/trnblas-fusing-df-mp2-energy-reduction-into-one-nki-kernel/) |
| [trnrand](https://github.com/trnsci/trnrand) | [#18](https://github.com/trnsci/trnrand/issues/18) | ✅ Hardware-validated | Threefry4×32-20 (the NKI generator), Philox 4×32-10, Box-Muller, Sobol / Halton | [trnrand: RNG is a four-engine workload](https://trnsci.dev/blog/trnrand-rng-is-a-four-engine-workload-if-the-silicon-lets-you-say-so/) |
| [trnsolver](https://github.com/trnsci/trnsolver) | [#26](https://github.com/trnsci/trnsolver/issues/26) | ✅ Hardware-validated | Householder `eigh` (single tile, n ≤ 128), batched and block Jacobi `eigh` (`method="jacobi"`, opt-in multi-core), Newton–Schulz inverse-sqrt, CG / GMRES with Jacobi preconditioner | [trnsolver: Jacobi for Trainium](https://trnsci.dev/blog/trnsolver-jacobi-for-trainium--when-the-hardware-inverts-the-algorithm-choice/) |
| [trnsparse](https://github.com/trnsci/trnsparse) | [#14](https://github.com/trnsci/trnsparse/issues/14) | ✅ Hardware-validated | BSR-128 SpMM (static-structure and gather), fused screened SpMM with tile skipping, fused kernel-matrix product (`kernel_matmul`, the GP operator), block-sparse attention, fused CG / PCG (`cg_bsr_fused`) | [trnsparse: the tile is the unit, not the nonzero](https://trnsci.dev/blog/trnsparse-the-tile-is-the-unit-not-the-nonzero/) |
| [trntensor](https://github.com/trnsci/trntensor) | [#27](https://github.com/trnsci/trntensor/issues/27) | ✅ Hardware-validated | fused MP2 energy, AO→MO transform (incl. the blocked packed kernel for any n_ao), fused DF exchange / Coulomb (`df_exchange`, `df_coulomb`); the matmul / bmm kernels are deprecated on the native device | [trntensor: when the kernel boundary is the API](https://trnsci.dev/blog/trntensor-when-the-kernel-boundary-is-the-api/) |

## Native TorchNeuron dispatch

All six libraries ship native TorchNeuron dispatch (`nki_op` on the `neuron` device) next to the PyTorch/XLA path; the native ports merged on 2026-10-30. On the native device each library's `auto` dispatch sends an op to an NKI kernel only where the kernel measured faster than aten on the device, and to aten otherwise. Dense GEMMs (trnblas, except the DF-MP2 half-transform) and plain einsum matmul / bmm (trntensor) go to aten; the fused kernels (DF-MP2 pair energy, AO→MO, DF J/K, BSR SpMM, the GP kernel operator, block-sparse attention, the SHT Legendre stage) are where NKI wins. trnfft has no aten alternative to route to, because `torch.fft` doesn't run on the device. float64 follows the [suite policy](precision_and_platform_limits.md#fp64-policy). In most libraries a kernel error falls back to torch with a warning unless `<LIB>_REQUIRE_NKI=1` is set.

Scoreboard and evidence: [trnsci#35](https://github.com/trnsci/trnsci/issues/35). Platform policy: [installation](installation.md#hardware-compatibility).

**Legend:**

- ✅ **Hardware-validated** — kernels pass `@pytest.mark.neuron` on hardware (trn1 on the XLA path, Trn2 on the native path). On the native device, `auto` picks the NKI kernel where it measured faster than aten.
- 🕑 **Simulator-validated** — kernels pass `nki-simulator` CI; hardware runs queued. PyTorch is the default until hardware passes.
- 🚧 **Simulator-validated, upstream-blocked** — kernels compile and run; the library has a named, trackable NKI primitive gap preventing numerically-correct hardware output. See the blog retrospective for specifics.

## Cross-suite infrastructure

- **NKI 0.3.0 migration** complete across all six libraries. Coordination tracked in [trnsci/trnsci#5](https://github.com/trnsci/trnsci/issues/5). Narrative: [The dev loop just got a lot shorter](https://trnsci.dev/blog/the-dev-loop-just-got-a-lot-shorter/).
- **`nki-simulator` CI gate** on `ubuntu-latest` for every library. Fast iteration for Python-layer correctness; does not replace hardware for MLIR-verifier or numerical-behavior checks.
- **Hardware validation** runs manually: per-repo `scripts/run_neuron_tests.sh` against a `<repo>-ci-trn1` instance for the XLA path, and booked Trn2 windows for the native path, logged on [#35](https://github.com/trnsci/trnsci/issues/35).
- **Off-box compile caches.** NEFFs compiled on a CPU host are reused on the device ([trnsci#118](https://github.com/trnsci/trnsci/issues/118), `scripts/warm_neuron_cache.py`), so hardware windows don't start cold.

## Looking ahead

- **Phase 1 trackers are closed for all six libraries.** trnrand's Philox kernel, long attributed to [aws-neuron-sdk#1308](https://github.com/aws-neuron/aws-neuron-sdk/issues/1308), was failing on a saturating int32/uint32 cast in the kernel itself; trnrand 0.8.0 fixed it (trnrand#56).
- Phases 2–5 (precision, single-chip perf, multi-chip, generation-specific optimization) build on top of Phase 1 per the [roadmap](roadmap.md) and are tracked in the phase milestones of each repository. The workload-level results are on [exemplars](exemplars.md).
- **At TorchNeuron general availability** the XLA path is deprecated on Trn2/Trn3 and every library is re-validated on the GA SDK ([GA checklist, #136](https://github.com/trnsci/trnsci/issues/136)).

## Design RFCs

Sub-projects with published design docs for upcoming phases:

- [trnblas: fused DF-MP2 pair-energy kernel (Phase 3)](https://trnsci.dev/trnblas/design/fused_df_mp2_energy_kernel/) — collapse `(T*(2T−T.T)/denom).sum()` into one SBUF-resident Vector+Scalar pass; targeted 3–6× on the DF-MP2 hot path. Shipped as `pair_energy_sym`; on Trn2 it measures 1.8× faster than aten (an earlier 92× figure was wrong, trnblas#57).
- [trnrand: SBUF-resident streaming Generator (Phase 3)](https://trnsci.dev/trnrand/design/sbuf_resident_generator/) — pre-compiled multi-distribution kernel with pipelined GpSimd / Vector / Scalar engines.
- [trnrand: counter-partitioned multi-chip RNG (Phase 4)](https://trnsci.dev/trnrand/design/counter_partitioned_multichip/) — bit-exact cross-cluster-size RNG via Philox counter-space partitioning.

## Maintenance

This page is updated when a library releases (the current-releases table) and when a Phase 1 tracker closes (or opens, for any
new sub-project added to the suite). Historic state lives in the git
history — no versioning beyond that.
