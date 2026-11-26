"""CPU tests for batched MLIP steps and the CPU reference (examples/mlip).

- ``batching.bucket`` and the reference comparison need only torch.
- The batched step tests build block-diagonal batches of the small public
  structures and run MACE-MP-0b2 with the trnsci blocks on their pytorch
  paths; they run where mace-torch, trntensor and trnsparse are installed.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MLIP_DIR = Path(__file__).resolve().parent.parent / "examples" / "mlip"
sys.path.insert(0, str(MLIP_DIR))

batching = importlib.import_module("batching")
cpu_ref = importlib.import_module("cpu_ref")


@pytest.mark.parametrize("shift", [2, 4, 6])
def test_bucket_bounds(shift):
    prev = 0
    for n in range(1, 5000, 7):
        b = batching.bucket(n, shift)
        assert b >= n
        assert b - n < max(1, 2 ** (n.bit_length() - 1 - shift))
        assert b <= n * (1 + 2.0**-shift) + 1
        assert b >= prev  # monotone
        prev = b
    # nearby sizes share a bucket: 10392 and 10378 edges (the 216-atom slab)
    assert batching.bucket(10392) == batching.bucket(10378) == 10752
    assert batching.bucket(96278) == 98304  # 1728 atoms, quantum 4096
    assert batching.bucket(5, 4, 128) == 128


def _ref(n=4, **kw):
    f = torch.arange(3 * n, dtype=torch.float64).reshape(n, 3) + 1.0
    args = {"energy": -10.0, "forces": f, "cpu_ref_s": 0.5, "threads": 2} | kw
    return cpu_ref.Reference(**args)


def test_compare_against_reference():
    ref = _ref()
    f = ref.forces.float() * (1 + 1e-6)
    r = cpu_ref.compare(-10.004, f, ref)
    assert r["dE_per_atom_meV"] == pytest.approx(1.0, rel=1e-6)
    assert 0 < r["forces_rel"] < 2e-6
    assert r["cpu_ref_s"] == 0.5
    w = cpu_ref.worst([r, r | {"forces_rel": 1.0}])
    assert w["forces_rel"] == 1.0 and w["cpu_ref_s"] == 1.0


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "no_forces",
        "empty_forces",
        "wrong_shape",
        "nan_forces",
        "no_run_time",
        "zero_run_time",
        "zero_forces",
        "nan_energy",
    ],
)
def test_missing_reference_raises(bad):
    """The 1728-atom window-7 gap: no reference must never report 0 error."""
    f = torch.ones(4, 3)
    if bad is None:
        ref = None
    else:
        ref = {
            "no_forces": _ref(forces=None),
            "empty_forces": _ref(forces=torch.zeros(0, 3)),
            "wrong_shape": _ref(n=5),
            "nan_forces": _ref(forces=torch.full((4, 3), float("nan"))),
            "no_run_time": _ref(cpu_ref_s=None),
            "zero_run_time": _ref(cpu_ref_s=0.0),
            "zero_forces": _ref(forces=torch.zeros(4, 3)),
            "nan_energy": _ref(energy=float("nan")),
        }[bad]
    with pytest.raises(cpu_ref.MissingReferenceError):
        cpu_ref.compare(-10.0, f, ref)
    with pytest.raises(cpu_ref.MissingReferenceError):
        cpu_ref.worst([])


# ── live: block-diagonal MACE batches (mace-torch + trntensor + trnsparse) ──


@pytest.fixture(scope="module")
def live():
    pytest.importorskip("mace")
    pytest.importorskip("ase")
    trnsparse = pytest.importorskip("trnsparse")
    trntensor = pytest.importorskip("trntensor")
    import mace_setup as ms

    trnsparse.set_backend("pytorch")
    trntensor.set_backend("pytorch")
    return ms.load_model()


def _atoms(names, seed0=0):
    import mace_setup as ms

    return [ms.build_structure(n, seed0 + k) for k, n in enumerate(names)]


def test_make_batch_layout(live):
    names = ["molecule_box_16", "cu111_slab_64", "molecule_box_16"]
    b, info = batching.make_batch(_atoms(names), live)
    assert info.atoms == [16, 64, 16]
    assert info.n_pad > info.n_real and info.e_pad >= info.e_real
    assert info.n_pad <= batching.bucket(info.n_real + 1, 4, 8)
    assert b["ptr"].numel() - 1 == info.num_structures + 1  # padding structure
    ei = b["edge_index"]
    # offsets: structure 1's edges point into its own atom range
    e0, a0 = info.edge_start[1], info.atom_start[1]
    blk = ei[:, e0 : e0 + info.edges[1]]
    assert int(blk.min()) >= a0 and int(blk.max()) < a0 + 64
    # padding edges: masked, self-edges on padding atoms, past the cutoff
    pe = info.edge_padding
    assert int(pe.sum()) == info.e_pad - info.e_real
    assert bool((b["shifts"][pe].norm(dim=1) >= 3 * float(live.r_max)).all())
    # a second geometry of the same structures lands in the same bucket
    _, info1 = batching.make_batch(_atoms(names, seed0=10), live)
    assert info1.key == info.key
    # corrupting the layout is caught
    bad = dict(b)
    bad["edge_index"] = ei.clone()
    bad["edge_index"][1, 0] = a0
    with pytest.raises(ValueError):
        batching.check_batch(bad, info)


def test_plan_masks_padding(live):
    import mace_trnsci_step as mts

    b, info = batching.make_batch(_atoms(["molecule_box_16"] * 2), live)
    g = mts.Graph(b, b, info.edge_padding)
    assert g.rplan.num_padding == g.splan.num_padding == info.e_pad - info.e_real


@pytest.mark.parametrize("stress", [False, True])
def test_batched_split_is_bitwise_and_matches_cpu(live, stress):
    """Batch of 3 on the trnsci path: each structure's slice equals the
    structure run alone at the batch's shapes bitwise, and matches unmodified
    MACE run alone and unpadded to fp32 rounding."""
    import mace_trnsci_step as mts

    atoms = _atoms(["molecule_box_16", "cu111_slab_64", "molecule_box_16"])
    st = mts.Step(live, atoms, "cpu", mode="trnsci", stress=stress)
    chk = mts.split_check(st, bitwise=True)
    assert chk["bitwise"]
    out = st()
    refs = cpu_ref.cpu_reference(live, atoms, stress=stress)
    for k, (p, r) in enumerate(zip(st.split(out), refs)):
        c = cpu_ref.compare(p["energy"], p["forces"], r, f"s{k}")
        assert c["dE_per_atom_meV"] < 1e-3 and c["forces_rel"] < 1e-5, c
        if stress:
            assert float((p["stress"].double() - r.stress).abs().max()) < 1e-6


def test_reference_cache_round_trip(live, tmp_path, monkeypatch):
    """MLIP_REF_CACHE: the second call loads the stored reference (same
    energy, forces and measured cpu_ref_s) and marks it cached; a different
    geometry misses."""
    import mace_setup as ms

    monkeypatch.setenv("MLIP_REF_CACHE", str(tmp_path / "refs"))
    a0 = ms.build_structure("molecule_box_16", 0)
    (r1,) = cpu_ref.cpu_reference(live, [a0], 1)
    assert not r1.cached and r1.cpu_ref_s > 0
    (r2,) = cpu_ref.cpu_reference(live, [ms.build_structure("molecule_box_16", 0)], 1)
    assert r2.cached and r2.cpu_ref_s == r1.cpu_ref_s and r2.energy == r1.energy
    assert torch.equal(r2.forces, r1.forces)
    assert cpu_ref.compare(r1.energy, r1.forces.float(), r2)["cpu_ref_cached"]
    (r3,) = cpu_ref.cpu_reference(live, [ms.build_structure("molecule_box_16", 1)], 1)
    assert not r3.cached


def _main(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", [mod.__name__ + ".py", *argv])
    mod.main()


def test_step_script_batch(live, tmp_path, monkeypatch):
    import mace_trnsci_step as mts

    out = tmp_path / "step.json"
    argv = ["--structure", "molecule_box_16", "--batch", "2", "--backend", "pytorch"]
    argv += ["--reps", "2", "--warm", "0", "--json", str(out)]
    _main(mts, argv, monkeypatch)
    res = json.loads(out.read_text())
    assert res["batch"] == 2 and len(res["per_structure"]) == 2
    assert res["split_check"]["bitwise"]
    for r in res["per_structure"]:
        assert r["cpu_ref_s"] > 0 and 0 < r["forces_rel"] < 1e-5


def _drop_forces(monkeypatch):
    real = cpu_ref.cpu_reference

    def no_forces(*a, **k):
        refs = real(*a, **k)
        for r in refs:
            r.forces = None
        return refs

    monkeypatch.setattr(cpu_ref, "cpu_reference", no_forces)


@pytest.mark.parametrize("script", ["mace_trnsci_step", "profile_device_step"])
def test_step_scripts_raise_without_reference(live, monkeypatch, script):
    """Regression for the window-7 1728-atom runs (forces_rel = 0.0 with no
    CPU reference): both step scripts must raise, not report 0."""
    import importlib

    mod = importlib.import_module(script)
    _drop_forces(monkeypatch)
    argv = ["--structure", "molecule_box_16", "--backend", "pytorch"]
    argv += ["--reps", "1", "--warm", "0"]
    if script == "profile_device_step":
        argv += ["--no-profile"]
    with pytest.raises(cpu_ref.MissingReferenceError):
        _main(mod, argv, monkeypatch)
