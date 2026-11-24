"""Where the aten remainder of the trnsci MACE step goes, by MACE module, on device.

``mace_trnsci_step.py --mode trnsci`` runs the three blocks (conv_tp,
symmetric contraction, gather / segment sum) on the trnsci kernels; the
rest of MACE-MP-0b2 stays aten on the same device. This script splits that
rest by module:

1. One step under forward hooks that record every tagged module call (its
   inputs, kwargs and, through tensor hooks, the cotangent each output got
   in the force backward), plus the trnsci block calls (``install(rec=...)``).
2. Each recorded call is replayed in isolation: forward only, then forward +
   the input gradients the step needed, each synchronised and timed, and
   once more under a dispatch mode that counts the aten ops (forward and
   backward separately).
3. A whole-step op count by innermost tagged module (forward ops; backward
   ops land under ``backward``) closes the books against the step total.

Module groups: radial basis (Agnesi transform, Bessel, polynomial cutoff),
radial MLPs (``conv_tp_weights``, ``density_fn``), spherical harmonics,
linears (``linear_up``, ``linear``, product ``linear``), skip tensor product,
readouts, ZBL, node embedding, E0, scale-shift.

    python profile_device_step.py --device neuron --json prof216.json
    python profile_device_step.py --device neuron --fuse radial --json fuse.json
    python profile_device_step.py --device neuron --pad-edges 512 --seeds 0,1

``--fuse`` swaps a group for a prototype (see ``fusions``): ``radial`` puts
the per-edge radial chain (transform, Bessel, cutoff, both layers' radial and
density MLPs) in one ``torch.compile`` region; ``skip`` replaces e3nn's
``FullyConnectedTensorProduct`` against the one-hot species by a per-species
weight gather and a batched GEMM. ``--pad-edges B`` pads the edge list to a
multiple of ``B`` with far self-edges (length 3 r_max, so the cutoff zeroes
them exactly) to test whether the aten graphs survive a geometry change.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import os
import statistics
import time

import mace_setup as ms
import mace_trnsci_step as mts
import torch
from torch.utils._python_dispatch import TorchDispatchMode

GROUPS = {
    "radial_basis": "radial basis (Agnesi, Bessel, cutoff)",
    "radial_mlp": "radial MLP (conv_tp_weights)",
    "density_mlp": "density MLP (density_fn)",
    "sph_harm": "spherical harmonics",
    "linear": "linears (linear_up, linear, product linear)",
    "skip_tp": "skip tensor product (one-hot species)",
    "readout": "readouts",
    "zbl": "ZBL pair repulsion",
    "embed": "node embedding + E0",
    "scale_shift": "scale-shift",
    "radial_fused": "compiled per-edge region (fusions.py edge)",
}


def sync(dev):
    if dev.type == "neuron":
        torch.neuron.synchronize()


def tag_modules(model):
    """(label, group, module) for every module call to replay."""
    out = [
        ("node_embedding", "embed", model.node_embedding),
        ("atomic_energies", "embed", model.atomic_energies_fn),
        ("spherical_harmonics", "sph_harm", model.spherical_harmonics),
        ("radial_embedding", "radial_basis", model.radial_embedding),
    ]
    if hasattr(model, "pair_repulsion_fn"):
        out.append(("zbl", "zbl", model.pair_repulsion_fn))
    for i, it in enumerate(model.interactions):
        out += [
            (f"L{i}.linear_up", "linear", it.linear_up),
            (f"L{i}.conv_tp_weights", "radial_mlp", it.conv_tp_weights),
            (f"L{i}.density_fn", "density_mlp", it.density_fn),
            (f"L{i}.linear", "linear", it.linear),
            (f"L{i}.skip_tp", "skip_tp", it.skip_tp),
        ]
    for i, p in enumerate(model.products):
        out.append((f"L{i}.product_linear", "linear", p.linear))
    for i, r in enumerate(model.readouts):
        out.append((f"readout{i}", "readout", r))
    if hasattr(model, "scale_shift"):
        out.append(("scale_shift", "scale_shift", model.scale_shift))
    return out


def sub_modules(model):
    """Parts of the radial embedding, reported as detail rows (not summed)."""
    re = model.radial_embedding
    out = [("radial.cutoff", re.cutoff_fn), ("radial.bessel", re.bessel_fn)]
    if hasattr(re, "distance_transform"):
        out.append(("radial.agnesi", re.distance_transform))
    return out


def _tensors(obj):
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for o in obj for t in _tensors(o)]
    return []


class Recorder:
    """Forward hooks that keep each tagged module call for replay, and a
    label stack for the whole-step op count."""

    def __init__(self):
        self.calls = []
        self.stack = []
        self.handles = []
        self.recording = False

    def add(self, label, group, module, record=True):
        def pre(mod, args, kwargs, label=label):
            self.stack.append(label)

        def post(mod, args, kwargs, out, label=label, group=group):
            self.stack.pop()
            if not (self.recording and record):
                return
            outs = _tensors(out)
            e = {
                "label": label,
                "group": group,
                "module": mod,
                "args": [a.detach() if torch.is_tensor(a) else a for a in args],
                "kwargs": {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in kwargs.items()
                },
                "need_args": [torch.is_tensor(a) and a.requires_grad for a in args],
                "need_kwargs": {
                    k: torch.is_tensor(v) and v.requires_grad for k, v in kwargs.items()
                },
                "grad_out": [None] * len(outs),
            }
            for j, t in enumerate(outs):
                if t.requires_grad:
                    t.register_hook(
                        lambda g, e=e, j=j: e["grad_out"].__setitem__(
                            j, None if g is None else g.detach()
                        )
                    )
            self.calls.append(e)

        self.handles += [
            module.register_forward_pre_hook(pre, with_kwargs=True),
            module.register_forward_hook(post, with_kwargs=True),
        ]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def current(self):
        return self.stack[-1] if self.stack else "glue"


class OpCount(TorchDispatchMode):
    def __init__(self, label_fn=None):
        super().__init__()
        self.n = 0
        self.by = collections.Counter()
        self.label_fn = label_fn

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        if self.label_fn is not None:
            self.by[self.label_fn(func)] += 1
        return func(*args, **(kwargs or {}))


def timed(fn, dev, warm, reps, budget=10.0):
    fn()
    sync(dev)
    for _ in range(warm):
        fn()
    sync(dev)
    ts, t_start = [], time.perf_counter()
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
        if time.perf_counter() - t_start > budget and len(ts) >= 3:
            break
    return statistics.median(ts)


def replay(e, dev, warm, reps):
    """Forward and forward+backward of one recorded module call, isolated."""
    mod = e["module"]
    go = [g for g in e["grad_out"]]
    has_bwd = any(g is not None for g in go) and (
        any(e["need_args"]) or any(e["need_kwargs"].values())
    )

    def inputs():
        args = [
            a.detach().requires_grad_(True) if n else a
            for a, n in zip(e["args"], e["need_args"])
        ]
        kw = {
            k: (v.detach().requires_grad_(True) if e["need_kwargs"][k] else v)
            for k, v in e["kwargs"].items()
        }
        wrt = [a for a, n in zip(args, e["need_args"]) if n]
        wrt += [kw[k] for k, n in e["need_kwargs"].items() if n]
        return args, kw, wrt

    def fo():
        with torch.no_grad():
            return mod.forward(*e["args"], **e["kwargs"])

    def fb():
        args, kw, wrt = inputs()
        outs = _tensors(mod.forward(*args, **kw))
        pairs = [(o, g) for o, g in zip(outs, go) if g is not None and o.requires_grad]
        if pairs and wrt:
            torch.autograd.grad(
                [o for o, _ in pairs], wrt, [g for _, g in pairs], allow_unused=True
            )

    t_f = timed(fo, dev, warm, reps)
    t_fb = timed(fb, dev, warm, reps) if has_bwd else t_f
    with torch.autograd.set_multithreading_enabled(False):
        c = OpCount()
        with c:
            fo()
        n_f = c.n
        n_fb = n_f
        if has_bwd:
            c = OpCount()
            with c:
                fb()
            n_fb = c.n
    return {
        "fwd_ms": 1e3 * t_f,
        "fwd_bwd_ms": 1e3 * t_fb,
        "ops_fwd": n_f,
        "ops_bwd": n_fb - n_f,
        "backward": has_bwd,
    }


def pad_batch(batch_host, multiple, r_max):
    """Pad the edge list to a multiple of ``multiple`` with self-edges on node 0
    shifted by (0, 0, 3 r_max): length 3 r_max > r_max, so the polynomial
    cutoff (``envelope * (x < r_max)``) and the ZBL envelope are exactly 0 and
    the padded edges add nothing to the energy or the forces."""
    b = dict(batch_host)
    E = int(b["edge_index"].shape[1])
    Ep = -(-E // multiple) * multiple
    k = Ep - E
    if k == 0:
        return b, E, Ep
    ei = b["edge_index"]
    b["edge_index"] = torch.cat([ei, ei.new_zeros(2, k)], dim=1)
    sh = b["shifts"].new_zeros(k, 3)
    sh[:, 2] = 3.0 * r_max
    b["shifts"] = torch.cat([b["shifts"], sh], 0)
    if "unit_shifts" in b:
        b["unit_shifts"] = torch.cat(
            [b["unit_shifts"], b["unit_shifts"].new_zeros(k, 3)], 0
        )
    return b, E, Ep


@contextlib.contextmanager
def label_gev(rec):
    """Label the (patched) edge-vector function for the whole-step count."""
    from mace.modules import utils as mutils

    inner = mutils.get_edge_vectors_and_lengths

    def gev(*a, **k):
        rec.stack.append("edge_vectors")
        try:
            return inner(*a, **k)
        finally:
            rec.stack.pop()

    mutils.get_edge_vectors_and_lengths = gev
    try:
        yield
    finally:
        mutils.get_edge_vectors_and_lengths = inner


def whole_step_count(model, batch, dev, rec):
    def label(func):
        ns = str(func.overloadpacket).split(".")[0]
        if ns in ("trnsparse", "trntensor"):
            return "trnsci:" + str(func.overloadpacket)
        if torch._C._current_autograd_node() is not None:
            return "backward"
        return rec.current()

    with torch.autograd.set_multithreading_enabled(False), label_gev(rec):
        c = OpCount(label)
        with c:
            mts.step(model, batch, dev)
    return c.n, dict(c.by.most_common())


def build(args, dev, seed):
    model_cpu = ms.load_model()
    atoms = ms.build_structure(args.structure, seed)
    batch_host = ms.atoms_to_batch(atoms, model_cpu)
    e_ref, f_ref = mts.step(model_cpu, batch_host, torch.device("cpu"))
    E0 = int(batch_host["edge_index"].shape[1])
    if args.pad_edges:
        batch_host, E0, _ = pad_batch(
            batch_host, args.pad_edges, float(model_cpu.r_max)
        )
    model = ms.load_model().to(dev)
    batch = mts.to_dev(batch_host, dev)
    return model, model_cpu, atoms, batch_host, batch, e_ref, f_ref, E0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structure", default="tic001_slab_216")
    ap.add_argument("--seeds", default="0", help="comma list; each one a new geometry")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", default="nki")
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-profile", action="store_true", help="step timing only")
    ap.add_argument("--fuse", default="", help="comma list of prototypes: radial, skip")
    ap.add_argument("--pad-edges", type=int, default=0)
    ap.add_argument("--compile-backend", default="neuron", help="for --fuse edge")
    ap.add_argument(
        "--count", action="store_true", help="whole-step op count with --no-profile"
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    import fusions
    import trnsparse
    import trntensor

    trnsparse.set_backend(args.backend)
    trntensor.set_backend(args.backend)
    results = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        model, model_cpu, atoms, batch_host, batch, e_ref, f_ref, E0 = build(
            args, dev, seed
        )
        g = mts.Graph(batch_host, batch)
        restore = mts.install(model, g)
        ctx = {
            "dev": dev,
            "model_cpu": model_cpu,
            "batch_host": batch_host,
            "g": g,
            "compile_backend": args.compile_backend,
        }
        undo_fuse = fusions.apply(model, [f for f in args.fuse.split(",") if f], ctx)
        first, med, ts, (e, f) = mts.timed(
            lambda m=model, b=batch: mts.step(m, b, dev),
            dev,
            args.warm,
            args.reps,
            60.0,
        )
        N = len(atoms)
        f = f[:N]
        res = {
            "structure": args.structure,
            "seed": seed,
            "atoms": N,
            "edges": E0,
            "edges_padded": int(batch_host["edge_index"].shape[1]),
            "fuse": args.fuse,
            "first_s": first,
            "step_ms": 1e3 * med,
            "step_ms_all": [1e3 * t for t in ts],
            "energy": e,
            "energy_ref_cpu": e_ref,
            "dE_per_atom_meV": 1e3 * abs(e - e_ref) / N,
            "forces_rel": float(
                (f.double() - f_ref.double()).norm() / f_ref.double().norm()
            ),
            "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2 (default)"),
        }
        print(json.dumps(res), flush=True)
        if args.count or not args.no_profile:
            rec = Recorder()
            for label, group, mod in tag_modules(model):
                fwd = mod.__dict__.get("forward")
                if getattr(fwd, "stash_stub", False):
                    continue  # its work runs inside the fused region
                if getattr(fwd, "fused_region", False):
                    label, group = "edge_region (compiled)", "radial_fused"
                rec.add(label, group, mod)
            if "edge" not in args.fuse:  # inside the compiled region otherwise
                for label, mod in sub_modules(model):
                    rec.add(label, "detail", mod)
            n_ops, by_label = whole_step_count(model, batch, dev, rec)
            res["ops_total"] = n_ops
            res["ops_by_label"] = by_label
            print(f"  ops {n_ops}: {by_label}", flush=True)
            if args.no_profile:
                rec.remove()
        if not args.no_profile:
            # record one step: module calls + trnsci block calls
            restore()
            blk = []
            restore = mts.install(model, g, blk)
            rec.recording = True
            mts.step(model, batch, dev)
            rec.recording = False
            # stop recording block calls: the module replays call them again
            restore()
            restore = mts.install(model, g)
            rows = []
            for e in rec.calls:
                r = replay(e, dev, args.warm, max(5, args.reps // 2))
                r.update(label=e["label"], group=e["group"])
                rows.append(r)
                print(
                    f"  {e['label']:24s} fwd {r['fwd_ms']:7.3f} ms  fwd+bwd {r['fwd_bwd_ms']:7.3f} ms"
                    f"  ops {r['ops_fwd']}+{r['ops_bwd']}",
                    flush=True,
                )
            for r in mts.replay_shares(blk, dev, args.warm, max(5, args.reps // 2)):
                rows.append(
                    {
                        "label": r["name"],
                        "group": "trnsci." + r["family"],
                        "fwd_ms": r["fwd_ms"],
                        "fwd_bwd_ms": r["fwd_bwd_ms"],
                    }
                )
            rec.remove()
            groups = collections.defaultdict(lambda: collections.Counter())
            for r in rows:
                if r["group"] == "detail":
                    continue
                gr = groups[r["group"]]
                gr["calls"] += 1
                for k in ("fwd_ms", "fwd_bwd_ms", "ops_fwd", "ops_bwd"):
                    gr[k] += r.get(k, 0)
            res["calls"] = rows
            res["groups"] = {k: dict(v) for k, v in groups.items()}
            res["sum_isolated_ms"] = sum(v["fwd_bwd_ms"] for v in groups.values())
            print(
                f"  step {res['step_ms']:.1f} ms, {n_ops} ops; isolated sum {res['sum_isolated_ms']:.1f} ms"
            )
            for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["fwd_bwd_ms"]):
                print(
                    f"  {k:20s} x{v['calls']:<3d} fwd {v['fwd_ms']:7.2f}  fwd+bwd {v['fwd_bwd_ms']:7.2f} ms"
                    f"  ops {v['ops_fwd']}+{v['ops_bwd']}",
                    flush=True,
                )
        undo_fuse()
        restore()
        results.append(res)
        del model, batch
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
