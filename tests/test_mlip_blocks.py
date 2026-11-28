"""CPU tests for the MLIP block contracts (examples/mlip).

The committed fixture is a slice of one MACE-MP-0b2 medium capture, so the
reference check runs on any host with torch. The live test re-captures
from the public checkpoint and runs only where mace-torch is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MLIP_DIR = Path(__file__).resolve().parent.parent / "examples" / "mlip"
FIXTURE = MLIP_DIR / "fixtures" / "mace_mp0b2_tiny.pt"
sys.path.insert(0, str(MLIP_DIR))


def _run(cap, impl=None, edge=False):
    import blocks
    import check

    chk = check.Checker(impl or blocks, "cpu", 2e-5)
    if cap is not None:
        chk.conv_tp(cap)
        chk.symcon(cap)
        chk.graph(cap)
    if edge:
        check.edge_cases(impl or blocks, "cpu", chk)
    return chk


def test_fixture_is_small():
    assert FIXTURE.stat().st_size < 400 * 1024


def test_reference_reproduces_fixture():
    import check

    cap = check.load(str(FIXTURE))
    chk = _run(cap)
    names = {r[0] for r in chk.rows}
    assert any(n.startswith("A1") for n in names)
    assert any(n.startswith("A2") for n in names)
    assert any(n.startswith("B2 force") for n in names)
    bad = [r for r in chk.rows if r[3] != "ok"]
    assert not bad, bad


def test_reference_edge_cases():
    chk = _run(None, edge=True)
    bad = [r for r in chk.rows if r[3] != "ok"]
    assert not bad, bad


def test_fixture_contract_shapes():
    import check

    cap = check.load(str(FIXTURE))
    assert cap["sender"].dtype == torch.int64 and cap["receiver"].dtype == torch.int64
    assert cap["stats"]["sender_sorted"] and not cap["stats"]["receiver_sorted"]
    for a in cap["A1"]:
        assert a["x"].shape[0] == a["y"].shape[0] == a["w"].shape[0]
        assert a["y"].shape[1] == 16
    for a in cap["A2"]:
        assert a["x"].dim() == 3 and a["x"].shape[2] == 16
        assert all(c["correlation"] == 3 for c in a["spec"]["contractions"])


def test_live_capture_matches_reference():
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    import capture
    import check
    import mace_setup as ms

    model = ms.load_model()
    cap = capture.capture(
        model, ms.build_structure("molecule_box_16"), "molecule_box_16"
    )
    chk = _run(cap)
    bad = [r for r in chk.rows if r[3] != "ok"]
    assert not bad, bad
    # the committed fixture regenerates from the public checkpoint
    fx = capture.make_fixture(cap)
    ref = check.load(str(FIXTURE))
    for blk in ("A1", "A2"):
        for a, b in zip(fx[blk], ref[blk]):
            assert check.rel(a["out"], b["out"]) < 1e-4
