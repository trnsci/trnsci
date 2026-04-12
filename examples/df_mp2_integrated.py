"""
Cross-project DF-MP2 integration example.

Each stage is handled by a different library in the trnsci suite:

    trnrand   — synthetic occupied/virtual MO coefficients
    trnsolver — Cholesky of the DF metric J
    trnblas   — half-transforms (μν|P) → (ia|P) via GEMM
    trntensor — pair-energy contraction via einsum, with FLOPs estimate

Run:

    python examples/df_mp2_integrated.py --demo
"""

from __future__ import annotations

import argparse
import time

import torch

import trnblas
import trnrand
import trnsolver
import trntensor


def synthetic_system(*, n_ao: int, n_aux: int, n_occ: int, seed: int):
    """Build a small synthetic DF-MP2 problem.

    Returns (eri_ao, J, C_occ, C_vir, eps_occ, eps_vir) where
        eri_ao : (n_ao, n_ao, n_aux)   — AO-basis three-center integrals
        J      : (n_aux, n_aux)        — DF metric (SPD)
        C_occ  : (n_ao, n_occ)         — occupied MO coefficients
        C_vir  : (n_ao, n_vir)         — virtual MO coefficients
        eps_*  : orbital energies (ascending / descending)
    """
    g = trnrand.manual_seed(seed)
    n_vir = n_ao - n_occ

    eri_ao = trnrand.normal(n_ao, n_ao, n_aux, generator=g)
    # Symmetrize in the AO indices (physical integrals have μν ↔ νμ symmetry)
    eri_ao = 0.5 * (eri_ao + eri_ao.transpose(0, 1))

    # SPD metric by construction: J = M Mᵀ + nI
    M = trnrand.normal(n_aux, n_aux, generator=g)
    J = M @ M.transpose(0, 1) + n_aux * torch.eye(n_aux)

    C_occ = trnrand.normal(n_ao, n_occ, generator=g)
    C_vir = trnrand.normal(n_ao, n_vir, generator=g)

    # Orbital energies — occupied below zero, virtuals above
    eps_occ = -torch.linspace(1.0, 0.1, n_occ)
    eps_vir = torch.linspace(0.1, 1.0, n_vir)

    return eri_ao, J, C_occ, C_vir, eps_occ, eps_vir


def half_transform(eri_ao, C_occ, C_vir):
    """(μν|P) → (ia|P) via two trnblas GEMMs."""
    n_ao, _, n_aux = eri_ao.shape
    n_occ = C_occ.shape[1]
    n_vir = C_vir.shape[1]

    # Reshape to (n_ao, n_ao * n_aux), contract index μ with C_occ
    flat = eri_ao.reshape(n_ao, n_ao * n_aux)
    iv_P = trnblas.gemm(1.0, C_occ, flat, transA=True)
    iv_P = iv_P.reshape(n_occ, n_ao, n_aux)

    # Contract second AO index ν with C_vir per occupied i.
    # iv_P : (n_occ, n_ao, n_aux), C_vir : (n_ao, n_vir)
    # Want: (n_occ, n_vir, n_aux) = C_vir.T @ iv_P[i]  stacked over i.
    C_vir_T = C_vir.transpose(0, 1).unsqueeze(0).expand(n_occ, -1, -1)
    ia_P = trnblas.batched_gemm(1.0, C_vir_T, iv_P)
    return ia_P   # (n_occ, n_vir, n_aux)


def metric_contract(ia_P, J):
    """Factor J = L Lᵀ via trnsolver, then B_ia^P = (ia|Q) · J^{-1/2}_{QP} via trnblas."""
    n_occ, n_vir, n_aux = ia_P.shape
    L = trnsolver.cholesky(J)
    # J^{-1/2} = L^{-T}: solve Lᵀ X = I  →  X = L^{-T}
    J_inv_half = trnblas.trsm(
        1.0, L, torch.eye(n_aux, dtype=J.dtype), uplo="lower", trans=True
    )
    # One batched GEMM over the occupied axis: B[i] = ia_P[i] @ J_inv_half
    J_b = J_inv_half.unsqueeze(0).expand(n_occ, -1, -1)
    B = trnblas.batched_gemm(1.0, ia_P, J_b)
    return B


def pair_energy(B, eps_occ, eps_vir):
    """MP2 pair energies via trntensor.einsum, summed with orbital-energy denominator."""
    n_occ, n_vir, n_aux = B.shape
    e = 0.0
    flops = 0
    # Canonical DF-MP2: E = -sum_ijab |T_iajb|^2 / (eps_i + eps_j - eps_a - eps_b)
    for i in range(n_occ):
        for j in range(i, n_occ):
            T = trntensor.einsum("ap,bp->ab", B[i], B[j])
            flops += trntensor.estimate_flops("ap,bp->ab", B[i], B[j])
            denom = (eps_occ[i] + eps_occ[j]).item() - (eps_vir.unsqueeze(1) + eps_vir.unsqueeze(0))
            pair = (T * T / denom).sum().item()
            e += pair if i == j else 2.0 * pair
    return e, flops


def run_demo(*, n_ao: int, n_aux: int, n_occ: int, seed: int) -> None:
    eri_ao, J, C_occ, C_vir, eps_occ, eps_vir = synthetic_system(
        n_ao=n_ao, n_aux=n_aux, n_occ=n_occ, seed=seed
    )

    print(f"system: n_ao={n_ao} n_aux={n_aux} n_occ={n_occ} n_vir={n_ao - n_occ}")
    print(f"backends: trnblas={trnblas.get_backend()} "
          f"trnsolver={trnsolver.get_backend()} "
          f"trntensor={trntensor.get_backend()}")
    print()

    stages = []

    t = time.perf_counter()
    ia_P = half_transform(eri_ao, C_occ, C_vir)
    stages.append(("half-transform (trnblas)", time.perf_counter() - t))

    t = time.perf_counter()
    B = metric_contract(ia_P, J)
    stages.append(("metric contract (trnsolver+trnblas)", time.perf_counter() - t))

    t = time.perf_counter()
    e_mp2, flops = pair_energy(B, eps_occ, eps_vir)
    stages.append(("pair energy (trntensor)", time.perf_counter() - t))

    width = max(len(name) for name, _ in stages)
    for name, dt in stages:
        print(f"  {name:<{width}}  {dt * 1e3:8.2f} ms")
    print()
    print(f"E(MP2,corr) = {e_mp2:+.6e} Ha  (synthetic — not physically meaningful)")
    print(f"contraction FLOPs estimate: {flops:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run the small synthetic demo")
    parser.add_argument("--n-ao", type=int, default=32)
    parser.add_argument("--n-aux", type=int, default=64)
    parser.add_argument("--n-occ", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    run_demo(n_ao=args.n_ao, n_aux=args.n_aux, n_occ=args.n_occ, seed=args.seed)


if __name__ == "__main__":
    main()
