"""Model-level acceptance for an MLIP evaluator: symmetry, force and stress consistency.

The per-block checks (``check.py``) and the per-step comparison against the
CPU (``mace_trnsci_step.py``) say a candidate matches MACE on the inputs we
tried. These tests say whether the candidate behaves like a potential at all,
the properties molecular dynamics and relaxations rely on:

- ``translation``: the energy is invariant and the forces unchanged when every
  atom moves by the same random vector;
- ``rotation``: the energy is invariant under a random rotation of the atoms
  and the cell, and the forces (and the stress) are equivariant: rotate the
  inputs, rotate the outputs back, compare;
- ``fd_forces``: the forces are the negative gradient of the energy. A
  central difference of the energy along a random direction ``d`` (every
  atom moved, ``|d| = sqrt(N)``, step ``h``) against ``-F . d``;
- ``fd_stress``, when the model returns a stress and the cell has a volume:
  a central difference of the energy under a random symmetric strain
  against ``V * sum(sigma * S)`` (``sigma = (1/V) dE/d strain``, the ASE
  convention MACE uses).

Each test runs on the **baseline** (unmodified MACE on the host CPU, alone
and unpadded: the reference) and on the **target** (``--target trnsci``: the
three blocks on the trnsci kernels on ``--device``; ``--target mace``: MACE
unmodified on ``--device``), on the same transformed inputs. The paired
table counts, per test, the cases both accept, only the baseline accepts,
only the target accepts and neither accepts. "Baseline only" is the
interesting column: the target broke something the reference model has.
The target is also compared with the baseline directly on the untransformed
inputs (``reference``).

Every tolerance is a parameter (``--help``); the defaults are starting points
for fp32, not claims. For the finite differences the default absolute
tolerance is a multiple (``--fd-noise``) of the estimated rounding floor of
the energy difference: ``eps * sqrt(sum e_i^2) / h`` with ``e_i`` the
per-atom energies and ``eps`` of the evaluation dtype.

    python accept.py                                         # CPU MACE vs CPU MACE
    python accept.py --target trnsci --device neuron --backend nki --batch 8 \\
        --structures molecule_box_16,cu111_slab_64,tic001_slab_216 --json acc.json

The core (``Structure``, ``plan_cases``, ``run_cases``, ``paired``) needs only
torch, so it runs with any ``evaluate(structures, stress)`` callable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field

import torch

TESTS = ("translation", "rotation", "fd_forces", "fd_stress")


@dataclass
class Structure:
    """Atoms as torch tensors (fp64): enough to transform and to rebuild ASE."""

    name: str
    numbers: list
    positions: torch.Tensor  # [N, 3]
    cell: torch.Tensor  # [3, 3], rows are cell vectors
    pbc: tuple

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    @property
    def volume(self) -> float:
        return abs(float(torch.linalg.det(self.cell)))

    @classmethod
    def from_ase(cls, atoms, name: str) -> Structure:
        return cls(
            name=name,
            numbers=[int(z) for z in atoms.numbers],
            positions=torch.as_tensor(atoms.positions, dtype=torch.float64).clone(),
            cell=torch.as_tensor(atoms.cell.array, dtype=torch.float64).clone(),
            pbc=tuple(bool(p) for p in atoms.pbc),
        )

    def to_ase(self):
        from ase import Atoms

        return Atoms(
            numbers=self.numbers,
            positions=self.positions.numpy(),
            cell=self.cell.numpy(),
            pbc=self.pbc,
        )

    def moved(self, positions=None, cell=None, tag="") -> Structure:
        return Structure(
            name=self.name + tag,
            numbers=list(self.numbers),
            positions=self.positions if positions is None else positions,
            cell=self.cell if cell is None else cell,
            pbc=self.pbc,
        )


def random_rotation(gen: torch.Generator) -> torch.Tensor:
    """A uniformly random proper rotation (fp64 [3, 3], det +1)."""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def translated(s: Structure, t: torch.Tensor) -> Structure:
    return s.moved(positions=s.positions + t, tag="+t")


def rotated(s: Structure, R: torch.Tensor) -> Structure:
    """Atoms and cell rotated by R (row vectors: x -> x R^T)."""
    return s.moved(positions=s.positions @ R.T, cell=s.cell @ R.T, tag="*R")


def displaced(s: Structure, d: torch.Tensor, h: float) -> Structure:
    return s.moved(positions=s.positions + h * d, tag=f"{h:+g}d")


def strained(s: Structure, S: torch.Tensor, h: float) -> Structure:
    """Homogeneous deformation by (I + h S), atoms and cell."""
    D = torch.eye(3, dtype=torch.float64) + h * S
    return s.moved(positions=s.positions @ D, cell=s.cell @ D, tag=f"{h:+g}S")


@dataclass
class Tolerances:
    """Acceptance limits. Every one is a parameter, none a claim."""

    energy_meV_atom: float = 1e-2  # |dE| per atom, invariance tests
    forces_rel: float = 1e-4  # relative Frobenius error, (equi)variance tests
    stress_abs: float = 1e-5  # max |d sigma| (eV/A^3), rotation test
    fd_step: float = 3e-3  # A per atom (rms), force finite difference
    fd_rtol: float = 1e-2
    fd_atol: float | None = None  # eV/A; None: fd_noise x the rounding floor
    fd_noise: float = 16.0
    stress_step: float = 1e-3  # strain, stress finite difference
    stress_rtol: float = 1e-2
    stress_atol: float | None = None  # eV; None: fd_noise x the rounding floor
    eps: float = float(torch.finfo(torch.float32).eps)  # of the evaluation dtype


@dataclass
class Case:
    structure: str
    test: str
    case: int
    accepted: bool
    metrics: dict = field(default_factory=dict)


@dataclass
class Plan:
    """Every input the suite evaluates, and how the cases use them."""

    inputs: list  # Structure, evaluated once each
    cases: list  # (structure name, test, case index, kind, indices, extra)
    originals: list  # index in ``inputs`` of each untransformed structure


def plan_cases(
    structures: list,
    *,
    seed: int = 0,
    rotations: int = 2,
    directions: int = 3,
    strains: int = 2,
    tol: Tolerances | None = None,
    stress: bool = True,
) -> Plan:
    """The transformed inputs, drawn once so baseline and target see the same."""
    tol = tol or Tolerances()
    gen = torch.Generator().manual_seed(seed)
    inputs, cases, originals = [], [], []

    def add(s):
        inputs.append(s)
        return len(inputs) - 1

    for s in structures:
        i0 = add(s)
        originals.append(i0)
        t = (torch.rand(3, generator=gen, dtype=torch.float64) - 0.5) * 10.0
        cases.append((s.name, "translation", 0, "inv", (i0, add(translated(s, t))), {}))
        for k in range(rotations):
            R = random_rotation(gen)
            cases.append(
                (s.name, "rotation", k, "rot", (i0, add(rotated(s, R))), {"R": R})
            )
        for k in range(directions):
            d = torch.randn(s.n, 3, generator=gen, dtype=torch.float64)
            d = d * (math.sqrt(s.n) / d.norm())
            h = tol.fd_step
            ip, im = add(displaced(s, d, h)), add(displaced(s, d, -h))
            cases.append((s.name, "fd_forces", k, "fd", (i0, ip, im), {"d": d, "h": h}))
        if stress and s.volume > 0 and all(s.pbc):
            for k in range(strains):
                S = torch.randn(3, 3, generator=gen, dtype=torch.float64)
                S = 0.5 * (S + S.T)
                S = S / S.norm()
                h = tol.stress_step
                ip, im = add(strained(s, S, h)), add(strained(s, S, -h))
                cases.append(
                    (s.name, "fd_stress", k, "fds", (i0, ip, im), {"S": S, "h": h})
                )
    return Plan(inputs=inputs, cases=cases, originals=originals)


def _energy64(out: dict) -> tuple[float, float]:
    """(energy, its rounding scale sqrt(sum e_i^2)) in fp64; from the per-atom
    energies when the evaluator returns them (their fp64 sum has no fp32
    rounding of the total), else from the total."""
    ne = out.get("node_energy")
    if ne is not None:
        ne = ne.double()
        return float(ne.sum()), float(ne.norm())
    e = float(out["energy"])
    return e, abs(e)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    den = float(b.norm())
    num = float((a - b).norm())
    return num / den if den > 0 else (0.0 if num == 0 else math.inf)


def evaluate_plan(evaluate, plan: Plan, stress: bool = True) -> list:
    """Every input of ``plan``, evaluated once. ``evaluate(structures, stress)``
    returns one dict per structure: ``energy``, ``forces`` [N, 3], optional
    ``node_energy`` [N] and ``stress`` [3, 3]."""
    outs = evaluate(plan.inputs, stress)
    if len(outs) != len(plan.inputs):
        raise RuntimeError("evaluate returned the wrong number of results")
    for s, o in zip(plan.inputs, outs):
        if o.get("forces") is None or tuple(o["forces"].shape) != (s.n, 3):
            raise RuntimeError(f"{s.name}: evaluate returned no [N, 3] forces")
    return outs


def run_cases(
    evaluate, plan: Plan, tol: Tolerances, stress: bool = True, outs=None
) -> list:
    """Judge every case of ``plan`` (evaluating it first unless ``outs``)."""
    if outs is None:
        outs = evaluate_plan(evaluate, plan, stress)
    has_stress = stress and all(o.get("stress") is not None for o in outs)
    res = []
    for name, test, k, kind, idx, extra in plan.cases:
        s0 = plan.inputs[idx[0]]
        o0 = outs[idx[0]]
        m: dict = {}
        if kind in ("inv", "rot"):
            o1 = outs[idx[1]]
            e0, _ = _energy64(o0)
            e1, _ = _energy64(o1)
            m["dE_meV_atom"] = 1e3 * abs(e1 - e0) / s0.n
            f1 = o1["forces"].double()
            if kind == "rot":
                R = extra["R"]
                f1 = f1 @ R  # rotate back: F(R x) = F(x) R^T
            m["forces_rel"] = _rel(f1, o0["forces"])
            ok = (
                m["dE_meV_atom"] <= tol.energy_meV_atom
                and m["forces_rel"] <= tol.forces_rel
            )
            if kind == "rot" and has_stress and s0.volume > 0 and all(s0.pbc):
                s1 = R.T @ o1["stress"].double() @ R
                m["stress_max_abs"] = float((s1 - o0["stress"].double()).abs().max())
                ok = ok and m["stress_max_abs"] <= tol.stress_abs
        elif kind == "fd":
            h, d = extra["h"], extra["d"]
            ep, sp = _energy64(outs[idx[1]])
            em, sm = _energy64(outs[idx[2]])
            fd = (ep - em) / (2 * h)
            an = -float((o0["forces"].double() * d).sum())
            floor = tol.eps * math.hypot(sp, sm) / (2 * h)
            atol = tol.fd_atol if tol.fd_atol is not None else tol.fd_noise * floor
            lim = atol + tol.fd_rtol * abs(an)
            m |= {"fd_eV_A": fd, "analytic_eV_A": an, "abs_err": abs(fd - an)}
            m |= {"limit": lim, "rounding_floor": floor, "h": h}
            ok = abs(fd - an) <= lim
        elif kind == "fds":
            if not has_stress:
                continue
            h, S = extra["h"], extra["S"]
            ep, sp = _energy64(outs[idx[1]])
            em, sm = _energy64(outs[idx[2]])
            fd = (ep - em) / (2 * h)
            an = s0.volume * float((o0["stress"].double() * S).sum())
            floor = tol.eps * math.hypot(sp, sm) / (2 * h)
            st_atol = tol.stress_atol
            atol = st_atol if st_atol is not None else tol.fd_noise * floor
            lim = atol + tol.stress_rtol * abs(an)
            m |= {"fd_eV": fd, "analytic_eV": an, "abs_err": abs(fd - an)}
            m |= {"limit": lim, "rounding_floor": floor, "h": h}
            ok = abs(fd - an) <= lim
        else:
            raise ValueError(kind)
        res.append(Case(name, test, k, bool(ok), m))
    return res


def agreement(plan: Plan, base: list, target: list, e_tol, f_tol) -> list:
    """Target against baseline on the untransformed inputs (``evaluate_plan``
    outputs of each)."""
    rows = []
    for i in plan.originals:
        s, b, t = plan.inputs[i], base[i], target[i]
        de = 1e3 * abs(_energy64(t)[0] - _energy64(b)[0]) / s.n
        fr = _rel(t["forces"], b["forces"])
        rows.append(
            {
                "structure": s.name,
                "dE_meV_atom": de,
                "forces_rel": fr,
                "accepted": bool(de <= e_tol and fr <= f_tol),
            }
        )
    return rows


OUTCOMES = ("both", "baseline_only", "target_only", "neither")


def paired(base: list, target: list) -> dict:
    """Per test: how many cases both / only the baseline / only the target /
    neither accept. Cases pair by (structure, test, case)."""
    tb = {(c.structure, c.test, c.case): c.accepted for c in base}
    tt = {(c.structure, c.test, c.case): c.accepted for c in target}
    if tb.keys() != tt.keys():
        missing = sorted(set(tb) ^ set(tt))
        raise ValueError(f"baseline and target cases do not pair: {missing[:5]}")
    table = {}
    for key, b in tb.items():
        t = tt[key]
        row = table.setdefault(key[1], dict.fromkeys(OUTCOMES, 0))
        o = (
            "both"
            if b and t
            else "baseline_only"
            if b
            else "target_only"
            if t
            else "neither"
        )
        row[o] += 1
    total = dict.fromkeys(OUTCOMES, 0)
    for row in table.values():
        for o in OUTCOMES:
            total[o] += row[o]
    table["all"] = total
    return table


def format_table(table: dict) -> str:
    lines = [
        "| test | both accepted | baseline only | target only | neither |",
        "|---|---|---|---|---|",
    ]
    for test, row in table.items():
        lines.append(f"| {test} | " + " | ".join(str(row[o]) for o in OUTCOMES) + " |")
    return "\n".join(lines)


# ── MACE evaluators ──────────────────────────────────────────────────────────


def mace_evaluator(
    model,
    device="cpu",
    *,
    mode="mace",
    batch=1,
    bucket=None,
    torch_path=(),
    fuse=(),
    model_cpu=None,
    compile_backend="neuron",
):
    """``evaluate(structures, stress)`` for MACE: ``batch`` structures per step
    through ``mace_trnsci_step.Step`` (``mode`` "mace" or "trnsci"), padded to
    a shape bucket unless ``bucket={"pad": False}``. ``fuse`` applies the
    ``fusions`` prototypes on top of the trnsci blocks (rebuilt per batch: the
    edge constants and the species masks belong to the neighbour list)."""
    import mace_trnsci_step as mts

    bucket = dict(bucket or {})
    if fuse and (mode != "trnsci" or model_cpu is None):
        raise ValueError("fuse needs mode='trnsci' and the host model_cpu")

    def evaluate(structures, stress=False):
        outs = []
        for i in range(0, len(structures), batch):
            chunk = [s.to_ase() for s in structures[i : i + batch]]
            st = mts.Step(
                model,
                chunk,
                device,
                mode=mode,
                torch_path=torch_path,
                stress=stress,
                node_energy=True,
                **bucket,
            )
            with st:
                undo = None
                if fuse:
                    import fusions

                    ctx = {
                        "dev": st.dev,
                        "model_cpu": model_cpu,
                        "batch_host": st.batch_host,
                        "g": st.g,
                        "compile_backend": compile_backend,
                    }
                    undo = fusions.apply(model, list(fuse), ctx)
                try:
                    out = st.run()
                finally:
                    if undo is not None:
                        undo()
            outs += st.split(out)
        return outs

    return evaluate


def _cases_json(cases):
    return [asdict(c) for c in cases]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structures", default="molecule_box_16,cu111_slab_64")
    ap.add_argument("--seeds", default="0", help="comma list of rattle seeds")
    ap.add_argument("--target", choices=["mace", "trnsci"], default="mace")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--backend", default="auto", help="trntensor / trnsparse backend (trnsci)"
    )
    ap.add_argument("--torch-path", default="", help="A1,A2,B on the pytorch path")
    ap.add_argument(
        "--fuse", default="", help="fusions prototypes on the target: edge,dense"
    )
    ap.add_argument("--compile-backend", default="neuron", help="for --fuse edge")
    ap.add_argument("--batch", type=int, default=1, help="target structures per step")
    ap.add_argument("--pad", choices=["bucket", "none"], default="bucket")
    ap.add_argument("--bucket-shift", type=int, default=4)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--baseline-dtype", default="float32", help="dtype of the CPU baseline"
    )
    ap.add_argument("--rotations", type=int, default=2)
    ap.add_argument("--directions", type=int, default=3)
    ap.add_argument("--strains", type=int, default=2)
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seed of the random transforms")
    d = Tolerances()
    for f in (
        "energy_meV_atom",
        "forces_rel",
        "stress_abs",
        "fd_step",
        "fd_rtol",
        "fd_atol",
        "fd_noise",
        "stress_step",
        "stress_rtol",
        "stress_atol",
    ):
        ap.add_argument("--" + f.replace("_", "-"), type=float, default=getattr(d, f))
    ap.add_argument(
        "--ref-energy-meV-atom", type=float, default=0.1, help="target vs baseline"
    )
    ap.add_argument("--ref-forces-rel", type=float, default=1e-4)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    import mace_setup as ms

    tol = Tolerances(
        **{f: getattr(args, f) for f in Tolerances.__dataclass_fields__ if f != "eps"}
    )
    stress = not args.no_stress
    structures = [
        Structure.from_ase(ms.build_structure(n, int(sd)), f"{n}/s{sd}")
        for n in args.structures.split(",")
        for sd in args.seeds.split(",")
    ]
    plan = plan_cases(
        structures,
        seed=args.seed,
        rotations=args.rotations,
        directions=args.directions,
        strains=args.strains,
        tol=tol,
        stress=stress,
    )

    base_model = ms.load_model(args.checkpoint, dtype=args.baseline_dtype)
    base_eval = mace_evaluator(base_model, "cpu", bucket={"pad": False})
    base_tol = Tolerances(
        **{
            **asdict(tol),
            "eps": float(torch.finfo(getattr(torch, args.baseline_dtype)).eps),
        }
    )
    base_outs = evaluate_plan(base_eval, plan, stress)
    base = run_cases(base_eval, plan, base_tol, stress, base_outs)

    dev = torch.device(args.device)
    if args.target == "trnsci":
        import trnsparse
        import trntensor

        trnsparse.set_backend(args.backend)
        trntensor.set_backend(args.backend)
    model = ms.load_model(args.checkpoint).to(dev)
    bucket = {"pad": args.pad != "none", "shift": args.bucket_shift}
    tgt_eval = mace_evaluator(
        model,
        dev,
        mode=args.target,
        batch=args.batch,
        bucket=bucket,
        torch_path=[t for t in args.torch_path.split(",") if t],
        fuse=[f for f in args.fuse.split(",") if f],
        model_cpu=base_model if args.baseline_dtype == "float32" else None,
        compile_backend=args.compile_backend,
    )
    tgt_outs = evaluate_plan(tgt_eval, plan, stress)
    target = run_cases(tgt_eval, plan, tol, stress, tgt_outs)
    agree = agreement(
        plan, base_outs, tgt_outs, args.ref_energy_meV_atom, args.ref_forces_rel
    )
    table = paired(base, target)
    res = {
        "target": args.target,
        "device": args.device,
        "backend": args.backend if args.target == "trnsci" else None,
        "batch": args.batch,
        "pad": args.pad,
        "fuse": args.fuse,
        "structures": [s.name for s in structures],
        "tolerances": asdict(tol),
        "baseline_tolerances": asdict(base_tol),
        "paired": table,
        "reference": agree,
        "baseline": _cases_json(base),
        "target_cases": _cases_json(target),
    }
    print(format_table(table))
    print()
    for r in agree:
        print(
            f"reference {r['structure']:22s} dE {r['dE_meV_atom']:.2e} meV/atom  "
            f"forces rel {r['forces_rel']:.2e}  {'ok' if r['accepted'] else 'FAIL'}"
        )
    for who, cases in (("baseline", base), ("target", target)):
        for c in cases:
            if not c.accepted:
                print(f"{who} rejects {c.structure} {c.test}[{c.case}]: {c.metrics}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
