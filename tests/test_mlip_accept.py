"""CPU tests for the MLIP model-level acceptance tooling (examples/mlip/accept.py).

The core runs on any ``evaluate(structures, stress)``, so these tests use a
small pair potential written in torch (exact symmetries, forces and stress by
autograd) and broken variants of it. The live test runs the suite on the
public MACE-MP-0b2 checkpoint where mace-torch is installed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MLIP_DIR = Path(__file__).resolve().parent.parent / "examples" / "mlip"
sys.path.insert(0, str(MLIP_DIR))

accept = importlib.import_module("accept")


def _structure(n=6, seed=0, name="toy"):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g, dtype=torch.float64) * 4.0
    cell = torch.eye(3, dtype=torch.float64) * 8.0
    return accept.Structure(name, [1] * n, pos, cell, (True, True, True))


def _toy_evaluator(field=0.0, force_scale=1.0, stress_scale=1.0):
    """Morse-like pair energy over all pairs (no images): invariant, with
    forces -dE/dx and stress (1/V) dE/d strain by autograd. ``field`` adds an
    external potential (breaks translation and rotation invariance);
    ``force_scale`` / ``stress_scale`` break the gradient consistency."""

    def energy(pos):
        d = pos[:, None, :] - pos[None, :, :]
        r = torch.sqrt((d**2).sum(-1) + torch.eye(pos.shape[0], dtype=pos.dtype))
        pair = torch.exp(-2.0 * (r - 1.5)) - 2.0 * torch.exp(-(r - 1.5))
        pair = pair * (1 - torch.eye(pos.shape[0], dtype=pos.dtype))
        node = 0.5 * pair.sum(1) + field * pos[:, 0]
        return node

    def evaluate(structures, stress=False):
        outs = []
        for s in structures:
            eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
            pos = s.positions.clone().requires_grad_(True)
            node = energy(pos @ (torch.eye(3, dtype=torch.float64) + eps))
            e = node.sum()
            gx, ge = torch.autograd.grad(e, (pos, eps))
            outs.append(
                {
                    "energy": float(e.detach()),
                    "forces": -force_scale * gx,
                    "node_energy": node.detach(),
                    "stress": stress_scale * ge / s.volume if stress else None,
                }
            )
        return outs

    return evaluate


def _run(evaluate, tol=None, structures=None, **kw):
    tol = tol or accept.Tolerances(eps=float(torch.finfo(torch.float64).eps))
    plan = accept.plan_cases(structures or [_structure()], tol=tol, **kw)
    return plan, accept.run_cases(evaluate, plan, tol)


def test_random_rotation_is_proper():
    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        R = accept.random_rotation(g)
        assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
        assert abs(float(torch.linalg.det(R)) - 1.0) < 1e-12


def test_exact_potential_accepted_everywhere():
    _, cases = _run(_toy_evaluator())
    tests = {c.test for c in cases}
    assert tests == set(accept.TESTS)
    bad = [c for c in cases if not c.accepted]
    assert not bad, bad
    rot = [c for c in cases if c.test == "rotation"]
    assert all("stress_max_abs" in c.metrics for c in rot)


def test_external_field_breaks_invariance_only():
    _, cases = _run(_toy_evaluator(field=0.05))
    by = {}
    for c in cases:
        by.setdefault(c.test, []).append(c.accepted)
    assert not any(by["translation"]) and not any(by["rotation"])
    # the field term is still a gradient: forces stay consistent
    assert all(by["fd_forces"])


def test_inconsistent_forces_fail_finite_difference():
    _, cases = _run(_toy_evaluator(force_scale=1.05))
    fd = [c for c in cases if c.test == "fd_forces"]
    assert fd and not any(c.accepted for c in fd)
    assert all(c.accepted for c in cases if c.test in ("translation", "rotation"))


def test_inconsistent_stress_fails_finite_difference():
    _, cases = _run(_toy_evaluator(stress_scale=1.1))
    fds = [c for c in cases if c.test == "fd_stress"]
    assert fds and not any(c.accepted for c in fds)


def test_tolerances_are_parameters():
    # zero tolerance on the finite difference: truncation error alone rejects it
    tol = accept.Tolerances(fd_rtol=0.0, fd_atol=0.0, eps=2.2e-16)
    _, cases = _run(_toy_evaluator(), tol=tol)
    assert not any(c.accepted for c in cases if c.test == "fd_forces")
    loose = accept.Tolerances(energy_meV_atom=1e9, forces_rel=1e9, stress_abs=1e9)
    _, cases = _run(_toy_evaluator(field=0.05), tol=loose)
    assert all(c.accepted for c in cases if c.test == "translation")


def test_no_stress_skips_stress_cases():
    _, cases = _run(_toy_evaluator(), stress=False)
    assert not any(c.test == "fd_stress" for c in cases)
    nonperiodic = _structure()
    nonperiodic.pbc = (False, False, False)
    _, cases = _run(_toy_evaluator(), structures=[nonperiodic])
    assert not any(c.test == "fd_stress" for c in cases)


def test_paired_table():
    tol = accept.Tolerances(eps=2.2e-16)
    plan = accept.plan_cases([_structure(), _structure(seed=1, name="b")], tol=tol)
    good = accept.run_cases(_toy_evaluator(), plan, tol)
    bad = accept.run_cases(_toy_evaluator(force_scale=1.05), plan, tol)
    t = accept.paired(good, bad)
    n_fd = sum(1 for c in good if c.test == "fd_forces")
    assert t["fd_forces"] == {
        "both": 0,
        "baseline_only": n_fd,
        "target_only": 0,
        "neither": 0,
    }
    assert t["rotation"]["both"] == sum(1 for c in good if c.test == "rotation")
    assert sum(t["all"].values()) == len(good)
    rev = accept.paired(bad, good)
    assert rev["fd_forces"]["target_only"] == n_fd
    assert "| fd_forces | 0 | " in accept.format_table(t)
    with pytest.raises(ValueError):
        accept.paired(good, bad[:-1])


def test_evaluate_without_forces_raises():
    def broken(structures, stress=False):
        return [{"energy": 0.0, "forces": None} for _ in structures]

    plan = accept.plan_cases([_structure()])
    with pytest.raises(RuntimeError):
        accept.run_cases(broken, plan, accept.Tolerances())


def test_live_mace_suite_cpu():
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    trnsparse = pytest.importorskip("trnsparse")
    trntensor = pytest.importorskip("trntensor")
    import mace_setup as ms

    trnsparse.set_backend("pytorch")
    trntensor.set_backend("pytorch")
    model = ms.load_model()
    s = accept.Structure.from_ase(ms.build_structure("molecule_box_16"), "mol16")
    tol = accept.Tolerances()
    plan = accept.plan_cases([s], tol=tol, rotations=1, directions=2, strains=1)
    base_eval = accept.mace_evaluator(model, "cpu", bucket={"pad": False})
    tgt_eval = accept.mace_evaluator(model, "cpu", mode="trnsci", batch=4)
    base_outs = accept.evaluate_plan(base_eval, plan)
    tgt_outs = accept.evaluate_plan(tgt_eval, plan)
    base = accept.run_cases(base_eval, plan, tol, outs=base_outs)
    target = accept.run_cases(tgt_eval, plan, tol, outs=tgt_outs)
    table = accept.paired(base, target)
    assert table["all"]["both"] == len(base), (table, base, target)
    agree = accept.agreement(plan, base_outs, tgt_outs, 0.1, 1e-4)
    assert all(r["accepted"] for r in agree), agree
