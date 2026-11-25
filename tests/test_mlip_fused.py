"""CPU tests for the fused ops of the MLIP exemplar (examples/mlip, trnsci/trnsci#178).

- The ZBL pair energy per edge (``trnsci_fused.zbl_edge_energy``) against
  MACE's ``ZBLBasis`` formula, fp64 ``gradcheck`` / ``gradgradcheck``, and the
  padding contract (edges past the ZBL cutoff contribute exactly zero). Needs
  torch only.
- Live (skipped without mace-torch and a trntensor with the fused ops): the
  energy + forces step with every fused op matches the unmodified model, and
  ``check_fused.py`` passes at the 16-atom structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MLIP_DIR = Path(__file__).resolve().parent.parent / "examples" / "mlip"
sys.path.insert(0, str(MLIP_DIR))


def _zbl_case(E=40, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    Zu = torch.randint(1, 80, (E, 1), generator=g).to(torch.float64)
    Zv = torch.randint(1, 80, (E, 1), generator=g).to(torch.float64)
    cr = 0.3 + 1.5 * torch.rand(100, generator=g, dtype=torch.float64)
    rmax = cr[Zu.long()] + cr[Zv.long()]
    r = 0.3 + (rmax.max() + 1.0) * torch.rand(E, 1, generator=g, dtype=torch.float64)
    k = {
        "a": 0.4543 * 0.529 / (Zu**0.3 + Zv**0.3),
        "zz": 14.3996 * Zu * Zv,
        "rmax": rmax,
        "c": (0.1818, 0.5099, 0.2802, 0.02817),
        "p": 6.0,
    }
    k = {n: (v.to(dtype) if torch.is_tensor(v) else v) for n, v in k.items()}
    return r.to(dtype), k


def _mace_zbl(r, k):
    """MACE ZBLBasis.forward's per-edge energy (mace-torch 0.3.15), before the sum."""
    c, a = k["c"], k["a"]
    r_over_a = r / a
    phi = (
        c[0] * torch.exp(-3.2 * r_over_a)
        + c[1] * torch.exp(-0.9423 * r_over_a)
        + c[2] * torch.exp(-0.4028 * r_over_a)
        + c[3] * torch.exp(-0.2016 * r_over_a)
    )
    v = k["zz"] / r * phi
    p, rmax = k["p"], k["rmax"]
    u = r / rmax
    env = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(u, p)
        + p * (p + 2.0) * torch.pow(u, p + 1)
        - (p * (p + 1.0) / 2) * torch.pow(u, p + 2)
    ) * (r < rmax)
    return 0.5 * v * env


def test_zbl_matches_mace_formula():
    import trnsci_fused as tf

    for dtype in (torch.float64, torch.float32):
        r, k = _zbl_case(dtype=dtype)
        got = tf.zbl_edge_energy(r, k)
        torch.testing.assert_close(got, _mace_zbl(r, k), rtol=1e-6, atol=1e-6)
        r1 = r.clone().requires_grad_(True)
        r2 = r.clone().requires_grad_(True)
        g = torch.randn_like(r)
        (d1,) = torch.autograd.grad(tf.zbl_edge_energy(r1, k), r1, g)
        (d2,) = torch.autograd.grad(_mace_zbl(r2, k), r2, g)
        torch.testing.assert_close(d1, d2, rtol=2e-5, atol=1e-5)


def test_zbl_gradcheck_fp64():
    import trnsci_fused as tf

    r, k = _zbl_case(E=12, seed=3)
    r.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: tf.zbl_edge_energy(t, k), (r,))
    assert torch.autograd.gradgradcheck(lambda t: tf.zbl_edge_energy(t, k), (r,))


def test_zbl_far_edges_are_exactly_zero():
    import trnsci_fused as tf

    r, k = _zbl_case(E=8)
    far = (
        k["rmax"] + torch.tensor([[0.0], [1.0], [5.0], [10.0]] * 2, dtype=r.dtype)
    ).requires_grad_(True)
    v = tf.zbl_edge_energy(far, k)
    (g,) = torch.autograd.grad(v.sum(), far)
    assert bool((v == 0).all()) and bool((g == 0).all())


def _live():
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    pytest.importorskip("trnsparse")
    trntensor = pytest.importorskip("trntensor")
    if not hasattr(trntensor, "edge_chain"):
        pytest.skip("trntensor without the fused MLIP ops (trnsci/trnsci#178)")
    return trntensor


def test_live_fused_step_matches_mace():
    _live()
    import mace_setup as ms
    import mace_trnsci_step as mts
    import trnsparse
    import trntensor

    trnsparse.set_backend("pytorch")
    trntensor.set_backend("pytorch")
    model_cpu = ms.load_model()
    atoms = [ms.build_structure("molecule_box_16", s) for s in (0, 1)]
    refs = [
        mts.step(model_cpu, ms.atoms_to_batch(a, model_cpu), torch.device("cpu"))
        for a in atoms
    ]
    # two structures in one bucketed batch: padding edges (far self-edges) and
    # padding atoms must add nothing through the fused ops either
    st = mts.Step(
        ms.load_model(), atoms, "cpu", mode="trnsci", fuse=["all"], model_cpu=model_cpu
    )
    assert st.info.e_pad > st.info.e_real
    parts = st.split(st())
    for p, (e_ref, f_ref) in zip(parts, refs, strict=True):
        assert abs(p["energy"] - e_ref) / 16 < 1e-5  # eV per atom
        assert float((p["forces"] - f_ref).norm() / f_ref.norm()) < 1e-5


def test_live_check_fused_passes():
    _live()
    import check_fused
    import mace_setup as ms

    model = ms.load_model()
    rows = check_fused.check_structure(
        model,
        "molecule_box_16",
        0,
        torch.device("cpu"),
        {"sh", "edge", "zbl", "linear", "skip"},
    )
    assert len(rows) == 1 + 1 + 1 + 9 + 2
    for r in rows:
        assert r["out_vs_mace32"] < 1e-6, r
        if "grad_ours_vs_64" in r:
            assert r["grad_ours_vs_64"] <= max(2e-6, 3 * r["grad_mace32_vs_64"]), r
