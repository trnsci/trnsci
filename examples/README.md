# trnsci examples

Cross-project integration demos exercising multiple libraries from the suite in a single pipeline.

## df_mp2_integrated.py

Density-fitted MP2 correlation energy on a small synthetic system.

| Stage | Library | APIs |
|---|---|---|
| Random MO coefficients | `trnrand` | `manual_seed`, `normal` |
| Cholesky of DF metric $J$ | `trnsolver` | `cholesky` |
| Half-transform $(\mu\nu\|P) \to (ia\|P)$ | `trnblas` | `gemm`, `batched_gemm`, `trsm` |
| Pair-energy contraction $T_{ab} = B_{ia}^P B_{jb}^P$ | `trntensor` | `einsum`, `estimate_flops` |

```bash
python examples/df_mp2_integrated.py --demo
```

The synthetic system is not physically meaningful — this is a plumbing demo that verifies the APIs across libraries compose end-to-end. For a chemistry-valid DF-MP2 pipeline, drive it with AO integrals and SCF MO coefficients from a quantum-chemistry driver.
