"""Smoke test for the cross-project DF-MP2 integration example."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))


@pytest.fixture(autouse=True)
def _ensure_imports():
    # Fail fast if any sub-project is missing
    for pkg in ("trnblas", "trnrand", "trnsolver", "trntensor"):
        pytest.importorskip(pkg)


def test_df_mp2_smoke():
    """Runs the integration example on a tiny system and asserts finiteness."""
    import df_mp2_integrated as demo
    import math

    eri, J, C_occ, C_vir, eps_occ, eps_vir = demo.synthetic_system(
        n_ao=8, n_aux=12, n_occ=3, seed=42
    )
    ia_P = demo.half_transform(eri, C_occ, C_vir)
    B = demo.metric_contract(ia_P, J)
    e_mp2, flops = demo.pair_energy(B, eps_occ, eps_vir)

    assert math.isfinite(e_mp2), "energy must be finite"
    assert flops > 0, "FLOPs estimate must be positive"
    assert ia_P.shape == (3, 5, 12)
    assert B.shape == (3, 5, 12)


def test_df_mp2_deterministic():
    """Same seed must give the same energy on two runs."""
    import df_mp2_integrated as demo

    def run():
        eri, J, C_occ, C_vir, eps_occ, eps_vir = demo.synthetic_system(
            n_ao=8, n_aux=12, n_occ=3, seed=7
        )
        ia_P = demo.half_transform(eri, C_occ, C_vir)
        B = demo.metric_contract(ia_P, J)
        e, _ = demo.pair_energy(B, eps_occ, eps_vir)
        return e

    assert run() == run()
