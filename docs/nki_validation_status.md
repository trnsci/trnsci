# NKI validation status

A snapshot of where each sub-project stands on
[Phase 1](roadmap.md#phase-1-single-chip-correctness) — the "NKI
kernels replace stubs and match the PyTorch reference on real trn1/trn2
hardware" milestone.

**What "validated" means here:**

1. The NKI kernel is in-tree under `<pkg>/nki/` (not a stub).
2. `set_backend("nki")` exercises it for the relevant public API calls.
3. On-hardware tests (`@pytest.mark.neuron`) pass on trn1 and/or trn2 via
   `scripts/run_neuron_tests.sh` — **run manually** by the project
   maintainers, not in GitHub Actions.
4. A canonical reference passes: published spec vectors, PyTorch-parity
   tolerance, or scipy-equivalent output depending on the library.

Per-kernel details live in each sub-project's `docs/architecture.md`.
Per-phase definitions live in the suite [roadmap](roadmap.md).

## Status by sub-project

| Sub-project | Phase 1 tracker | Status | Validated kernels | First shipped |
|---|---|---|---|---|
| [trnfft](https://github.com/trnsci/trnfft) | [#51](https://github.com/trnsci/trnfft/issues/51) | ✅ Validated | butterfly FFT, complex GEMM | v0.10.0 |
| [trnblas](https://github.com/trnsci/trnblas) | [#21](https://github.com/trnsci/trnblas/issues/21) | ✅ Validated | GEMM, batched_gemm | ≥ v0.4.0 |
| [trnrand](https://github.com/trnsci/trnrand) | [#18](https://github.com/trnsci/trnrand/issues/18) | 🕑 Pending | Philox 4×32-10, Box-Muller *(scaffolded, CPU-reference oracles pass)* | — |
| [trnsolver](https://github.com/trnsci/trnsolver) | [#26](https://github.com/trnsci/trnsolver/issues/26) | 🕑 Pending | Jacobi `eigh` *(scaffolded)* | — |
| [trnsparse](https://github.com/trnsci/trnsparse) | [#14](https://github.com/trnsci/trnsparse/issues/14) | ✅ Validated | SpMM gather-matmul-scatter | ≥ v0.2.0 |
| [trntensor](https://github.com/trnsci/trntensor) | [#27](https://github.com/trnsci/trntensor/issues/27) | ✅ Validated | fused einsum | ≥ v0.2.0 |

**Legend:**

- ✅ **Validated** — Phase 1 tracker is closed; NKI path is the default when `neuronxcc` is available, falls back to PyTorch otherwise.
- 🕑 **Pending** — NKI kernel exists in-tree with a CPU reference oracle, but on-hardware tests haven't been run yet. Users today get the PyTorch fallback; behavior and API are stable.

## Looking ahead

- **trnrand** and **trnsolver** are the two remaining pending Phase 1
  validations. Both await access to a trn1.2xlarge via their
  respective `scripts/run_neuron_tests.sh`.
- Phases 2–5 (precision, single-chip perf, multi-chip, generation-
  specific optimization) build on top of Phase 1 per the
  [roadmap](roadmap.md) and are tracked per sub-project via the
  matching `phase-N` labels.

## Design RFCs

Sub-projects with published design docs for upcoming phases:

- [trnblas: fused DF-MP2 pair-energy kernel (Phase 3)](https://trnsci.dev/trnblas/design/fused_df_mp2_energy_kernel/) — collapse `(T*(2T−T.T)/denom).sum()` into one SBUF-resident Vector+Scalar pass; targets 3–6× speedup on the DF-MP2 hot path.
- [trnrand: SBUF-resident streaming Generator (Phase 3)](https://trnsci.dev/trnrand/design/sbuf_resident_generator/) — pre-compiled multi-distribution kernel with pipelined GpSimd / Vector / Scalar engines.
- [trnrand: counter-partitioned multi-chip RNG (Phase 4)](https://trnsci.dev/trnrand/design/counter_partitioned_multichip/) — bit-exact cross-cluster-size RNG via Philox counter-space partitioning.

## Maintenance

This page is updated when a Phase 1 tracker closes (or opens, for any
new sub-project added to the suite). Historic state lives in the git
history — no versioning beyond that.
