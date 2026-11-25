"""CPU profile of MACE-MP-0b2 medium: energy + forces, attributed to blocks.

Forces are -dE/dpositions through autograd, so the trace holds the forward
and the backward. Every forward ATen op is attributed to the innermost
named range (module forward hooks plus wrappers around ``scatter_sum``,
the graph preparation and the force gradient). A backward node is
attributed to the range of the forward op that created it, matched on the
autograd sequence number. Block labels:

- ``A1.conv_tp``  interaction tensor product (e3nn ``TensorProduct``, uvu)
- ``A2.symcon``   symmetric contraction (product basis, correlation 3)
- ``B.gather``    ``node_feats[sender]`` inside each interaction
- ``B.scatter``   ``scatter_sum`` onto receivers (messages, density, ZBL, ...)

Usage:
    python profile_mace.py --threads 8 --json profile.json
    python profile_mace.py --structures tic_bulk_6 --threads 8   # scaling rung
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict

import mace_setup as ms
import torch
from torch.profiler import ProfilerActivity, profile, record_function

BWD = "autograd::engine::evaluate_function: "
STEP = "mlip.step"


class Labeller:
    """Forward hooks that open/close a record_function per tagged module."""

    def __init__(self) -> None:
        self.stack: list = []
        self.labels: set[str] = {STEP}
        self.handles: list = []

    def tag(self, module, label: str) -> None:
        self.labels.add(label)

        def pre(mod, args, label=label):
            rf = record_function(label)
            rf.__enter__()
            self.stack.append((label, rf))

        def post(mod, args, out):
            _, rf = self.stack.pop()
            rf.__exit__(None, None, None)

        self.handles += [
            module.register_forward_pre_hook(pre),
            module.register_forward_hook(post),
        ]

    def wrap(self, fn, label_fn):
        def wrapped(*a, **k):
            label = label_fn(self)
            self.labels.add(label)
            with record_function(label):
                return fn(*a, **k)

        return wrapped

    def current(self) -> str:
        return self.stack[-1][0] if self.stack else "model"


def instrument(model) -> tuple[Labeller, list]:
    from mace.modules import blocks, models, radial

    lab = Labeller()
    lab.tag(model.node_embedding, "embed.node")
    lab.tag(model.spherical_harmonics, "edge.sph_harm")
    lab.tag(model.radial_embedding, "edge.radial")
    if hasattr(model, "pair_repulsion_fn"):
        lab.tag(model.pair_repulsion_fn, "edge.zbl")
    lab.tag(model.atomic_energies_fn, "embed.e0")
    for i, it in enumerate(model.interactions):
        lab.tag(it, f"interaction{i}")
        lab.tag(it.linear_up, f"interaction{i}.linear_up")
        lab.tag(it.conv_tp_weights, f"interaction{i}.radial_mlp")
        lab.tag(it.density_fn, f"interaction{i}.density_mlp")
        lab.tag(it.conv_tp, f"A1.conv_tp{i}")
        lab.tag(it.linear, f"interaction{i}.linear")
        lab.tag(it.skip_tp, f"interaction{i}.skip_tp")
    for i, p in enumerate(model.products):
        lab.tag(p, f"product{i}")
        lab.tag(p.symmetric_contractions, f"A2.symcon{i}")
        lab.tag(p.linear, f"product{i}.linear")
    for i, r in enumerate(model.readouts):
        lab.tag(r, f"readout{i}")

    restore = []
    for mod in (blocks, models, radial):
        if hasattr(mod, "scatter_sum"):
            orig = mod.scatter_sum
            restore.append((mod, "scatter_sum", orig))
            mod.scatter_sum = lab.wrap(orig, lambda s: f"B.scatter[{s.current()}]")
    for name, label in (
        ("prepare_graph", "graph.prepare"),
        ("get_outputs", "forces.autograd"),
    ):
        orig = getattr(models, name)
        restore.append((models, name, orig))
        setattr(models, name, lab.wrap(orig, lambda s, label=label: label))
    return lab, restore


def uninstrument(lab: Labeller, restore: list) -> None:
    for h in lab.handles:
        h.remove()
    for mod, name, orig in restore:
        setattr(mod, name, orig)


def block_of(label: str) -> str:
    """Collapse a range label to a reporting bucket."""
    if label.startswith("A1.conv_tp"):
        return "A1.conv_tp"
    if label.startswith("A2.symcon"):
        return "A2.symcon"
    if label.startswith("B."):
        return "B.scatter"
    if label.endswith(("radial_mlp", "density_mlp")):
        return "radial_mlp"
    if label.startswith(("edge.", "graph.")):
        return label
    if label.startswith(("interaction", "product")):
        return "linear/skip/other"
    return "other"


def attribute(events, labels: set[str]):
    """Return {bucket: {"fwd": us, "bwd": us}} from a profiler event list."""
    by_id = {}
    for e in events:
        by_id[id(e)] = e

    def innermost(e):
        p = e.cpu_parent
        while p is not None:
            if p.name in labels and p.name != STEP:
                return p.name
            p = p.cpu_parent
        return "model"

    seq_label: dict[int, str] = {}
    fwd = defaultdict(float)
    bwd = defaultdict(float)
    for e in events:
        if e.name in labels or e.name.startswith(BWD):
            continue
        # skip ops running inside a backward node
        p, in_bwd = e.cpu_parent, False
        while p is not None:
            if p.name.startswith(BWD):
                in_bwd = True
                break
            p = p.cpu_parent
        if in_bwd:
            continue
        lab = innermost(e)
        if lab.startswith("interaction") and "." not in lab and e.name == "aten::index":
            lab = "B.gather"
        # Ops that do not create an autograd node report the *pending*
        # sequence number, so the node's creator is the last forward op
        # seen with that number.
        if e.sequence_nr >= 0:
            seq_label[e.sequence_nr] = lab
        fwd[lab] += e.self_cpu_time_total
    for e in events:
        if e.name.startswith(BWD):
            lab = seq_label.get(e.sequence_nr, "unmatched")
            bwd[lab] += e.cpu_time_total
    return fwd, bwd


def neighbour_list_time(atoms, cutoff: float, reps: int = 5) -> float:
    from mace.data.neighborhood import get_neighborhood

    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        get_neighborhood(
            positions=atoms.positions.copy(),
            cutoff=cutoff,
            pbc=tuple(bool(p) for p in atoms.pbc),
            cell=atoms.cell.array.copy(),
        )
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def run(name: str, model, reps: int, warmup: int) -> dict:
    atoms = ms.build_structure(name)
    batch = ms.atoms_to_batch(atoms, model)
    n, e = len(atoms), int(batch["edge_index"].shape[1])
    t_nl = neighbour_list_time(atoms, float(model.r_max))
    t0 = time.perf_counter()
    ms.atoms_to_batch(atoms, model)
    t_batch = time.perf_counter() - t0

    for _ in range(warmup):
        ms.energy_forces(model, batch)
    walls, fwd_walls = [], []
    for _ in range(reps):
        t = time.perf_counter()
        ms.energy_forces(model, batch)
        walls.append(time.perf_counter() - t)
        b = dict(batch)
        t = time.perf_counter()
        with torch.no_grad():
            model(b, compute_force=False, training=False)
        fwd_walls.append(time.perf_counter() - t)

    lab, restore = instrument(model)
    try:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            for _ in range(reps):
                with record_function(STEP):
                    ms.energy_forces(model, batch)
    finally:
        uninstrument(lab, restore)

    events = prof.events()
    step_us = sum(ev.cpu_time_total for ev in events if ev.name == STEP) / reps
    fwd, bwd = attribute(events, lab.labels)
    buckets = defaultdict(lambda: {"fwd_ms": 0.0, "bwd_ms": 0.0})
    for d, key in ((fwd, "fwd_ms"), (bwd, "bwd_ms")):
        for label, us in d.items():
            bucket = "B.gather" if label == "B.gather" else block_of(label)
            buckets[bucket][key] += us / reps / 1e3
    for v in buckets.values():
        v["share"] = (v["fwd_ms"] + v["bwd_ms"]) / (step_us / 1e3)

    top = []
    for avg in sorted(
        prof.key_averages(group_by_input_shape=True),
        key=lambda a: a.self_cpu_time_total,
        reverse=True,
    )[:25]:
        if avg.key in lab.labels or avg.key.startswith(BWD):
            continue
        top.append(
            {
                "op": avg.key,
                "shapes": str(avg.input_shapes)[:160],
                "calls": avg.count // reps,
                "self_ms": avg.self_cpu_time_total / reps / 1e3,
                "share": avg.self_cpu_time_total / reps / step_us,
            }
        )
    # Top-level ATen calls per step (each is one dispatch on a device backend).
    calls = sum(
        1
        for ev in events
        if ev.name.startswith("aten::")
        and (ev.cpu_parent is None or not ev.cpu_parent.name.startswith("aten::"))
    )
    deg = torch.bincount(batch["edge_index"][1], minlength=n)
    return {
        "structure": name,
        "atoms": n,
        "edges": e,
        "neighbors_per_atom": {
            "min": int(deg.min()),
            "mean": float(deg.float().mean()),
            "max": int(deg.max()),
        },
        "threads": torch.get_num_threads(),
        "dtype": str(next(model.parameters()).dtype),
        "neighbour_list_ms": t_nl * 1e3,
        "atoms_to_batch_ms": t_batch * 1e3,
        "energy_forces_ms": statistics.median(walls) * 1e3,
        "energy_only_ms": statistics.median(fwd_walls) * 1e3,
        "profiled_step_ms": step_us / 1e3,
        "aten_calls_per_step": calls // reps,
        "blocks": dict(sorted(buckets.items(), key=lambda kv: -kv[1]["share"])),
        "top_ops": top,
    }


def print_report(r: dict) -> None:
    print(
        f"\n== {r['structure']}: {r['atoms']} atoms, {r['edges']} edges, "
        f"neighbours/atom {r['neighbors_per_atom']}, {r['dtype']}, {r['threads']} threads"
    )
    print(
        f"   E+F {r['energy_forces_ms']:.1f} ms (E only {r['energy_only_ms']:.1f} ms, "
        f"profiled {r['profiled_step_ms']:.1f} ms); neighbour list {r['neighbour_list_ms']:.2f} ms, "
        f"atoms->batch {r['atoms_to_batch_ms']:.2f} ms; {r['aten_calls_per_step']} top-level ATen calls"
    )
    print(f"   {'block':22s} {'fwd ms':>8s} {'bwd ms':>8s} {'share':>7s}")
    for k, v in r["blocks"].items():
        print(
            f"   {k:22s} {v['fwd_ms']:8.2f} {v['bwd_ms']:8.2f} {100 * v['share']:6.1f}%"
        )
    print("   top ATen ops by self time:")
    for t in r["top_ops"][:15]:
        print(
            f"   {100 * t['share']:5.1f}% {t['self_ms']:7.2f} ms x{t['calls']:<4d} {t['op']:28s} {t['shapes']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structures", default=",".join(ms.STRUCTURES))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model = ms.load_model(args.checkpoint, args.dtype)
    results = []
    for name in args.structures.split(","):
        r = run(name, model, args.reps, args.warmup)
        print_report(r)
        results.append(r)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
