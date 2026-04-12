# Workflow: quantum chemistry

A significant share of the motivation for `trnblas`, `trnsolver`, `trnsparse`, and `trntensor` comes from **post-Hartree-Fock correlation methods** in computational quantum chemistry — specifically density-fitted MP2 (DF-MP2), DF-CCSD, and related approaches on molecules with several thousand basis functions.

These calculations are GEMM-dominated: most of the wall time is spent in dense matrix multiplications inside tensor contractions. On conventional CPU clusters, a single DF-MP2 energy evaluation on a large-basis system can run for many hours. Accelerator hardware helps, but only if the library stack exists to express the workload.

## The DF-MP2 shape

A canonical DF-MP2 pipeline, reduced to its linear-algebra skeleton:

1. **DF metric Cholesky** — factor the auxiliary-basis metric `J = L L^T`.
2. **Half-transform the three-center integrals** from AO to MO basis: `(μν|P) → (ia|P)` via two GEMMs.
3. **Contract with J⁻¹ᐟ²** to form the DF coefficient tensor `B_{iaP}`.
4. **Pair-energy contraction**: for each pair of occupied orbitals `(i, j)`, compute `T_{ab} = B_{ia}^P B_{jb}^P`, then sum with the orbital-energy denominator.
5. *(For >3000 basis functions)* **Integral screening** — the vast majority of shell quartets contribute below threshold and can be dropped via Schwarz inequality bounds.

Every step lives in a different `trnsci` library:

| Step | Library | API |
|---|---|---|
| 1. Cholesky | `trnsolver` | `cholesky` |
| 2. Half-transform | `trnblas` | `gemm`, `batched_gemm` |
| 3. Metric contraction | `trnblas` | `trsm`, `batched_gemm` |
| 4. Pair energies | `trntensor` | `einsum`, `estimate_flops` |
| 5. Screening | `trnsparse` | `schwarz_bounds`, `screen_quartets` |

This is the single workload that motivates *four* of the six libraries. Getting DF-MP2 to run end-to-end on Trainium is the suite's proving ground.

## Why Trainium

The DF-MP2 hot path is dense-GEMM-heavy — exactly what the Trainium Tensor Engine's systolic array is optimized for. The regular shapes (AO, MO, and auxiliary basis dimensions are fixed per calculation) are friendly to the Neuron compiler. And the cost economics of Trainium relative to comparable NVIDIA instances meaningfully change the wall-clock budget for exploratory chemistry — which is the scientific case.

## Example

A minimal synthetic version of the DF-MP2 pipeline is in [`examples/quantum_chemistry/`](https://github.com/trnsci/trnsci/tree/main/examples/quantum_chemistry). It constructs a synthetic `(AO × AO × aux)` three-center integral tensor with random MO coefficients, runs each stage through its respective library, reports per-stage timing, and prints a FLOPs estimate from the contraction planner. The energy is not physically meaningful — this is a plumbing demo. A real calculation would feed AO integrals and converged SCF MO coefficients from a driver such as PySCF.

See also [`examples/quantum_chemistry/pyscf_bridge_sketch.md`](https://github.com/trnsci/trnsci/tree/main/examples/quantum_chemistry/pyscf_bridge_sketch.md) for notes on wiring a real chemistry driver to the `trnsci` stack.

## Related CUDA samples

- [`cuBLAS/Level-3/gemm`](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS/Level-3/gemm) — the canonical dense GEMM sample; the DF-MP2 half-transform is several applications of this primitive.
- [`cuSOLVER/potrf`](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSOLVER/potrf) — Cholesky of an SPD matrix; directly corresponds to the DF metric factorization.
- [`cuTENSOR/contraction`](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR/contraction) — general tensor contraction; the pair-energy step is a natural `cutensorContract` call, and maps onto `trntensor.einsum`.
