# PySCF bridge sketch

This file is notes, not code. It describes how a real quantum-chemistry driver like [PySCF](https://pyscf.org/) would feed its outputs into the `trnsci` DF-MP2 pipeline implemented in [`df_mp2_synthetic.py`](df_mp2_synthetic.py).

## What the synthetic demo fakes

- **MO coefficients** — `C_occ`, `C_vir` in the synthetic demo come from `trnrand.normal(...)`. In a real calculation they come from a converged SCF.
- **AO integrals** — the three-center tensor `(μν|P)` is synthesized with random values in the demo. In reality it's computed from an auxiliary basis via analytic integral routines.
- **DF metric J** — built as `M M^T + n·I` in the demo. In reality it's `(P|Q)` over the auxiliary basis.
- **Orbital energies** — fabricated `linspace` in the demo. In reality they're SCF Fock-matrix eigenvalues.

## Real pipeline outline

```python
import pyscf
import pyscf.df

mol = pyscf.M(atom=..., basis=..., auxbasis=...)
mf = pyscf.scf.RHF(mol).density_fit()
mf.kernel()

# AO-basis DF metric J_{PQ} and 3-center tensor (μν|P)
naux = mol.auxmol_set().nao
J = mf.with_df.get_naoaux_2c()
eri_3c = mf.with_df.get_naoaux_3c().reshape(-1, mol.nao, mol.nao)
eri_3c = eri_3c.transpose(1, 2, 0)   # → (nao, nao, naux)

# Occupied / virtual MO coefficients and orbital energies
nocc = mol.nelec[0]
C_occ = mf.mo_coeff[:, :nocc]
C_vir = mf.mo_coeff[:, nocc:]
eps_occ = mf.mo_energy[:nocc]
eps_vir = mf.mo_energy[nocc:]

# Now hand the tensors to the trnsci pipeline
import torch
from examples.quantum_chemistry import df_mp2_synthetic as demo

ia_P = demo.half_transform(
    torch.from_numpy(eri_3c).float(),
    torch.from_numpy(C_occ).float(),
    torch.from_numpy(C_vir).float(),
)
B = demo.metric_contract(ia_P, torch.from_numpy(J).float())
e_mp2, flops = demo.pair_energy(
    B,
    torch.from_numpy(eps_occ).float(),
    torch.from_numpy(eps_vir).float(),
)
```

## What's missing to call this "production"

- **FP64 accuracy** — PySCF uses FP64 everywhere. The `trnsci` stack is currently FP32-only. Chemistry applications needing tight energy thresholds would need double-double arithmetic (documented, not yet implemented in `trnblas`).
- **Integral screening** — large-basis systems benefit enormously from Schwarz screening to drop negligible shell quartets. `trnsparse.screen_quartets` exists for this; the sketch above does not wire it in.
- **Memory blocking** — `pair_energy` in the synthetic demo uses explicit Python loops over occupied-pair indices. A production version would block over batches and fuse the denominator division into the contraction, as `trnblas/examples/df_mp2.py` does.
- **SCF step** — the synthetic demo skips SCF entirely. `trnsolver.eigh_generalized` covers the SCF Fock diagonalization step, but a production driver is a significant project on its own.

## Not in scope

A full PySCF integration is not in scope for the umbrella repo. PySCF's own DF-MP2 already runs on CPU; the value of `trnsci` is to replace the hot GEMM path with a Trainium-backed implementation. A lighter integration (PySCF produces the tensors, `trnsci` consumes them for the contraction-heavy steps) is the pragmatic path and is what this sketch describes.
