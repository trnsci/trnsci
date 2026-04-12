# Example workflow: quantum chemistry

Synthetic, end-to-end DF-MP2 plumbing demo. Uses `trnrand` + `trnsolver` + `trnblas` + `trntensor` in one pipeline — see [`../../docs/workflows/quantum_chemistry.md`](../../docs/workflows/quantum_chemistry.md) for workflow context.

## `df_mp2_synthetic.py`

Density-fitted MP2 correlation energy on a small synthetic system.

| Stage | Library | APIs |
|---|---|---|
| Random MO coefficients | `trnrand` | `manual_seed`, `normal` |
| Cholesky of DF metric J | `trnsolver` | `cholesky` |
| Half-transform (μν\|P) → (ia\|P) | `trnblas` | `gemm`, `batched_gemm`, `trsm` |
| Pair-energy contraction T_ab = B_iaP B_jbP | `trntensor` | `einsum`, `estimate_flops` |

```bash
python examples/quantum_chemistry/df_mp2_synthetic.py --demo
```

The inputs are random; the energy number is not physically meaningful. This is a **plumbing** demo that exercises the API surfaces across four libraries end-to-end. For a chemistry-valid calculation, drive the pipeline with AO integrals and SCF MO coefficients from a real quantum-chemistry driver (see `pyscf_bridge_sketch.md` for notes on what that looks like).

## `pyscf_bridge_sketch.md`

Notes on how a PySCF-driven pipeline would feed real integrals into this demo. Documentation only — not code.
