# Cross-project integration: DF-MP2

The canonical integration demo is `examples/quantum_chemistry/df_mp2_synthetic.py`. It computes a density-fitted MP2 correlation energy on a small synthetic system, with each stage handled by a different library in the suite:

| Stage | Library | API used |
|---|---|---|
| Random occupied / virtual coefficients (demo only) | `trnrand` | `normal()` |
| AO-pair Schwarz screening (report only) | `trnsparse` | `schwarz_bounds()`, `sparsity_stats()` |
| Cholesky of the DF metric $J$ | `trnsolver` | `cholesky()` |
| Half-transform $\mu\nu P \to ia P$ via GEMM, metric contraction | `trnblas` | `gemm()`, `batched_gemm()`, `trsm()` |
| Closed-shell MP2 energy $\sum T(2T - T^\top)/\Delta$, $T_{iajb} = \sum_P B_{iaP} B_{jbP}$ | `trntensor` | `mp2_energy()` |
| FLOPs estimate | `trntensor` | `estimate_flops()` |

The demo only reports the Schwarz screening; it does not drop pairs, because random inputs aren't chemistry-realistic. For screening inside a Fock build, see `trnsparse/examples/sparse_fock.py`.

## Run it

```bash
python examples/quantum_chemistry/df_mp2_synthetic.py --demo
python examples/quantum_chemistry/df_mp2_synthetic.py --demo --device neuron   # Trn2/Trn3, TorchNeuron
```

Output reports per-stage wall time, a FLOPs estimate from the contraction planner, and the energy. An integration test pins that energy to the float64 closed-shell reference (agreement is about 1e-7). `examples/quantum_chemistry/df_mp2_profile.py` runs the pipeline across a ladder of sizes and reports per-stage time, the route each library's `auto` dispatch picked, and error against float64.

## Why this example

DF-MP2 is the natural proving ground for the suite: it exercises dense GEMM (trnblas), small linear algebra (trnsolver), and tensor contractions with an FLOPs budget (trntensor). On Trainium, this is the path toward NKI-backed post-Hartree-Fock correlation methods.
