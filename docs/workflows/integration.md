# Cross-project integration: DF-MP2

The canonical integration demo is `examples/quantum_chemistry/df_mp2_synthetic.py`. It computes a density-fitted MP2 correlation energy on a small synthetic system, with each stage handled by a different library in the suite:

| Stage | Library | API used |
|---|---|---|
| Random occupied / virtual coefficients (demo only) | `trnrand` | `normal()` |
| Cholesky of the DF metric $J$ | `trnsolver` | `cholesky()` |
| Half-transform $\mu\nu P \to ia P$ via GEMM | `trnblas` | `gemm()`, `batched_gemm()` |
| Pair energy contraction $B_{iaP} B_{jbP}$ | `trntensor` | `einsum("ap,bp->ab", ...)` |
| FLOPs estimate | `trntensor` | `estimate_flops()` |

The same pattern could substitute `trnsparse` for the DF tensor when shell screening is enabled (not shown in the minimal demo — see `trnsparse/examples/sparse_fock.py`).

## Run it

```bash
python examples/quantum_chemistry/df_mp2_synthetic.py --demo
```

Output reports per-stage wall time and a FLOPs estimate from the contraction planner.

## Why this example

DF-MP2 is the natural proving ground for the suite: it exercises dense GEMM (trnblas), small linear algebra (trnsolver), and tensor contractions with an FLOPs budget (trntensor). On Trainium, this is the path toward NKI-backed post-Hartree-Fock correlation methods.
