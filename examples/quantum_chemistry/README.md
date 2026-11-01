# Example workflow: quantum chemistry

Synthetic, end-to-end DF-MP2 plumbing demo. Uses `trnrand` + `trnsparse` + `trnsolver` + `trnblas` + `trntensor` in one pipeline — see [`../../docs/workflows/quantum_chemistry.md`](../../docs/workflows/quantum_chemistry.md) for workflow context.

## `df_mp2_synthetic.py`

Density-fitted MP2 correlation energy on a small synthetic system.

| Stage | Library | APIs |
|---|---|---|
| Random MO coefficients | `trnrand` | `manual_seed`, `normal` |
| AO-pair Schwarz screening (report only) | `trnsparse` | `schwarz_bounds`, `sparsity_stats` |
| Cholesky of DF metric J | `trnsolver` | `cholesky` |
| Half-transform (μν\|P) → (ia\|P) | `trnblas` | `gemm`, `batched_gemm`, `trsm` |
| Closed-shell MP2 energy Σ T(2T − Tᵀ)/Δ, T_iajb = Σ_P B_iaP B_jbP | `trntensor` | `mp2_energy`, `estimate_flops` |

```bash
python examples/quantum_chemistry/df_mp2_synthetic.py --demo
python examples/quantum_chemistry/df_mp2_synthetic.py --demo --device neuron   # Trn2/Trn3, TorchNeuron
```

With `--device neuron` the inputs are built on the host and copied to the device once; each library's `auto` dispatch then picks an NKI kernel or aten for the device-resident tensors. The first call per shape includes compilation.

The inputs are random; the energy number is not physically meaningful. This is a **plumbing** demo that exercises the API surfaces across five libraries end-to-end. For a chemistry-valid calculation, drive the pipeline with AO integrals and SCF MO coefficients from a real quantum-chemistry driver (see `pyscf_bridge_sketch.md` for notes on what that looks like).

## `df_mp2_profile.py`

End-to-end profile of the pipeline above across a ladder of synthetic sizes: per-stage time, the route each library's `auto` dispatch picked, and stage-output and energy error against a float64 host reference. Modes `cpu` (all libraries on their pytorch backend), `host-auto` (host tensors, `auto` backends) and `neuron` (device-resident). Pipelines `example` (this example's stage functions), `fused` (`trntensor.ao_to_mo_transform`, `trnblas.metric_inv_half`, `trntensor.mp2_energy`) and `packed` (as `fused`, with the integrals packed to their μ≥ν lower triangle on the host first). Routes are recorded while each stage runs, so host inputs that a library uploads to the device show as `upload+aten` / `upload -> ...`.

```bash
NEURON_RT_VISIBLE_CORES=2 OMP_NUM_THREADS=2 taskset -c 4,10 \
    python examples/quantum_chemistry/df_mp2_profile.py \
    --sizes s,m,l,xl --modes cpu,host-auto,neuron --json df_mp2_profile.jsonl
```

## `pyscf_bridge_sketch.md`

Notes on how a PySCF-driven pipeline would feed real integrals into this demo. Documentation only — not code.
