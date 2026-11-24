"""One MACE-MP-0b2 energy + forces step with the trnsci kernels in all three blocks.

The model is the public MACE-MP-0b2 medium checkpoint (``mace_setup.py``).
``--mode trnsci`` swaps, inside MACE's own forward:

- A1 ``conv_tp``: ``trntensor.conv_tp`` (spec read from each e3nn ``TensorProduct``);
- A2 ``symmetric_contraction``: ``trntensor.symmetric_contraction``;
- B  every edge gather and edge -> node sum: ``trnsparse.gather`` /
  ``trnsparse.segment_sum`` with one ``SegmentPlan`` per index (senders,
  receivers), built once per neighbour list. That covers ``node_feats[sender]``,
  the density and message sums, the ZBL pair sum and the edge vectors
  ``pos[receiver] - pos[sender]``, whose backward is the force assembly.

Everything else (radial MLP, spherical harmonics, e3nn ``Linear`` /
``FullyConnectedTensorProduct``, readouts, ZBL) stays MACE's code, run as
aten on the same device. Forces are ``-dE/dpositions`` by autograd through
the kernels' registered backwards. ``--mode mace`` runs the unmodified model.

    python mace_trnsci_step.py --structure tic001_slab_216 --mode mace --device cpu
    python mace_trnsci_step.py --structure tic001_slab_216 --mode trnsci --device neuron \
        --shares --json step.json

The result is checked against the unmodified model on the host CPU (fp32):
energy difference per atom and the relative Frobenius error of the forces.
``--shares`` replays every block call of one step in isolation (forward +
the input gradients the step needed, at the step's own shapes and values)
and reports each block family's share of the step time.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import time

import mace_setup as ms
import torch


def sync(dev: torch.device) -> None:
    if dev.type == "neuron":
        torch.neuron.synchronize()


def to_dev(batch: dict, dev: torch.device) -> dict:
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Graph:
    """Per-neighbour-list state: indices on the device and their plans."""

    def __init__(self, batch_host: dict, batch_dev: dict) -> None:
        import trnsparse

        ei = batch_host["edge_index"]
        self.N = int(batch_host["positions"].shape[0])
        self.E = int(ei.shape[1])
        dev = batch_dev["edge_index"].device
        t0 = time.perf_counter()
        self.splan = trnsparse.segment_plan(ei[0], self.N).to(dev)
        self.rplan = trnsparse.segment_plan(ei[1], self.N).to(dev)
        self.plan_s = time.perf_counter() - t0
        self.edge_index = batch_dev["edge_index"]
        self.sender = self.edge_index[0]
        self.receiver = self.edge_index[1]
        self.species = batch_host["node_attrs"].argmax(-1).to(dev)


@contextlib.contextmanager
def backend(lib, name: str | None):
    if name is None:
        yield
        return
    old = lib.get_backend()
    lib.set_backend(name)
    try:
        yield
    finally:
        lib.set_backend(old)


def install(model, g: Graph, rec: list | None = None, torch_path=()):
    """Patch the three blocks of ``model`` to the trnsci kernels; returns an undo.

    Families in ``torch_path`` ("A1", "A2", "B") call the library with its
    ``pytorch`` backend instead (the library's own aten formulation, on the
    same device)."""
    import trnsparse as B
    import trntensor as A
    from mace.modules import radial
    from mace.modules import utils as mutils

    undo = []
    lib = {"A1": A, "A2": A, "B": B}

    def blk(family, name, fn, *ts):
        def run(*a, fn=fn, family=family):
            with backend(lib[family], "pytorch" if family in torch_path else None):
                return fn(*a)

        out = run(*ts)
        fn = run
        if rec is not None:
            e = {
                "family": family,
                "name": name,
                "fn": fn,
                "inputs": [t.detach() for t in ts],
                "need": [bool(t.requires_grad) for t in ts],
                "grad_out": None,
            }
            rec.append(e)
            if out.requires_grad:
                out.register_hook(
                    lambda go, e=e: e.__setitem__("grad_out", go.detach())
                )
        return out

    def gather_s(h):
        return B.gather(h, g.sender, plan=g.splan)

    def gather_r(h):
        return B.gather(h, g.receiver, plan=g.rplan)

    def segsum_r(m):
        return B.segment_sum(m, g.receiver, g.N, plan=g.rplan)

    # edge vectors: gathers of positions; their backward is the force assembly
    orig_gev = mutils.get_edge_vectors_and_lengths

    def gev(positions, edge_index, shifts, normalize=False, eps=1e-9):
        vr = blk("B", "gather pos[receiver] (F=3)", gather_r, positions)
        vs = blk("B", "gather pos[sender] (F=3)", gather_s, positions)
        vectors = vr - vs + shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        if normalize:
            return vectors / (lengths + eps), lengths
        return vectors, lengths

    mutils.get_edge_vectors_and_lengths = gev
    undo.append(lambda: setattr(mutils, "get_edge_vectors_and_lengths", orig_gev))

    # ZBL pair energy: edge -> receiver sum
    orig_ss = radial.scatter_sum

    def zbl_ss(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
        # MACE 0.3.15 sums v_edges [E, 1] over dim 0 (an [E] src over dim -1 in
        # older versions); both are the receiver segment sum.
        if (
            ((src.dim() == 1 and dim in (0, -1)) or (src.dim() == 2 and dim == 0))
            and src.shape[0] == g.E
            and index.data_ptr() == g.receiver.data_ptr()
            and out is None
            and reduce == "sum"
        ):
            return blk("B", "segment_sum ZBL (F=1)", segsum_r, src)
        return orig_ss(src, index, dim=dim, out=out, dim_size=dim_size, reduce=reduce)

    radial.scatter_sum = zbl_ss
    undo.append(lambda: setattr(radial, "scatter_sum", orig_ss))

    # interactions: gather -> conv_tp -> segment_sum (+ density sum)
    for li, it in enumerate(model.interactions):
        kind = type(it).__name__
        if kind not in (
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
        ) or hasattr(it, "conv_fusion"):
            raise NotImplementedError(f"interaction {li}: {kind} not covered")
        residual = kind.endswith("ResidualInteractionBlock")
        spec = A.conv_tp_spec(module=it.conv_tp)

        def tp(x, y, w, spec=spec):
            return A.conv_tp(x, y, w, spec)

        def fwd(
            node_attrs,
            node_feats,
            edge_attrs,
            edge_feats,
            edge_index,
            cutoff=None,
            lammps_class=None,
            lammps_natoms=(0, 0),
            first_layer=False,
            it=it,
            li=li,
            residual=residual,
            tp=tp,
        ):
            if lammps_class is not None:
                raise NotImplementedError("LAMMPS path not covered")
            sc = it.skip_tp(node_feats, node_attrs) if residual else None
            h = it.linear_up(node_feats)
            tp_w = it.conv_tp_weights(edge_feats)
            dens = torch.tanh(it.density_fn(edge_feats) ** 2)
            if cutoff is not None:
                tp_w = tp_w * cutoff
                dens = dens * cutoff
            density = blk("B", f"segment_sum density L{li} (F=1)", segsum_r, dens)
            x = blk(
                "B", f"gather node_feats[sender] L{li} (F={h.shape[1]})", gather_s, h
            )
            mji = blk("A1", f"conv_tp L{li}", tp, x, edge_attrs, tp_w)
            message = blk(
                "B", f"segment_sum messages L{li} (F={mji.shape[1]})", segsum_r, mji
            )
            message = it.linear(message) / (density + 1)
            if not residual:
                message = it.skip_tp(message, node_attrs)
            return it.reshape(message), sc

        it.forward = fwd
        undo.append(lambda it=it: delattr(it, "forward"))

    for li, p in enumerate(model.products):
        sc_mod = p.symmetric_contractions
        spec = A.symmetric_contraction_spec(module=sc_mod)

        def symcon(x, spec=spec):
            return A.symmetric_contraction(x, g.species, spec)

        def sc_fwd(x, y, symcon=symcon, li=li):
            return blk("A2", f"symmetric_contraction L{li}", symcon, x)

        sc_mod.forward = sc_fwd
        undo.append(lambda m=sc_mod: delattr(m, "forward"))

    def restore():
        for u in reversed(undo):
            u()

    return restore


def step(model, batch, dev):
    e, f = ms.energy_forces(model, batch)
    return float(e.detach().to("cpu").sum()), f.detach().to("cpu")


def timed(fn, dev, warm: int, reps: int, budget_s: float):
    """(first call s, median s, all s). Every call ends with a host copy."""
    t0 = time.perf_counter()
    res = fn()
    sync(dev)
    first = time.perf_counter() - t0
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    t_start = time.perf_counter()
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
        if time.perf_counter() - t_start > budget_s and len(ts) >= 3:
            break
    return first, statistics.median(ts), ts, res


def count_ops(fn) -> dict:
    """Operators dispatched by one call of ``fn`` (forward and backward), by name."""
    import collections

    from torch.utils._python_dispatch import TorchDispatchMode

    counts: collections.Counter = collections.Counter()

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counts[str(func.overloadpacket)] += 1
            return func(*args, **(kwargs or {}))

    with torch.autograd.set_multithreading_enabled(False), Count():
        fn()
    kernels = {
        k: v for k, v in counts.items() if k.split(".")[0] in ("trnsparse", "trntensor")
    }
    return {
        "total": sum(counts.values()),
        "trnsci_kernels": sum(kernels.values()),
        "by_kernel": kernels,
        "top": counts.most_common(15),
    }


def replay_shares(rec: list, dev, warm: int, reps: int) -> list:
    """Time each recorded block call in isolation: forward + the input gradients
    the step needed (the step's grad_out as cotangent)."""
    rows = []
    for e in rec:
        need = [n for n in e["need"]]
        go = e["grad_out"]

        def fb(e=e, need=need, go=go):
            ins = [
                t.detach().requires_grad_(True) if n else t
                for t, n in zip(e["inputs"], need)
            ]
            out = e["fn"](*ins)
            wrt = [t for t, n in zip(ins, need) if n]
            if wrt and go is not None:
                torch.autograd.grad(out, wrt, go)
            return out

        def fo(e=e):
            with torch.no_grad():
                return e["fn"](*e["inputs"])

        _, t_fb, _, _ = timed(fb, dev, warm, reps, 10.0)
        _, t_f, _, _ = timed(fo, dev, warm, reps, 10.0)
        rows.append(
            {
                "family": e["family"],
                "name": e["name"],
                "shapes": [list(t.shape) for t in e["inputs"]],
                "grads": need if go is not None else [False] * len(need),
                "fwd_ms": 1e3 * t_f,
                "fwd_bwd_ms": 1e3 * t_fb,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structure", default="tic001_slab_216")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["mace", "trnsci"], default="trnsci")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--budget", type=float, default=60.0, help="timing budget, s")
    ap.add_argument(
        "--backend",
        default="nki",
        help="trntensor / trnsparse backend for --mode trnsci (pytorch on a CPU host)",
    )
    ap.add_argument(
        "--torch-path",
        default="",
        help="comma list of A1,A2,B to run on the library's pytorch path (same device)",
    )
    ap.add_argument("--shares", action="store_true")
    ap.add_argument(
        "--count-ops", action="store_true", help="count dispatched ops in one step"
    )
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)

    model_cpu = ms.load_model(args.checkpoint)
    atoms = ms.build_structure(args.structure, args.seed)
    batch_host = ms.atoms_to_batch(atoms, model_cpu)
    e_ref, f_ref = step(model_cpu, batch_host, torch.device("cpu"))

    model = ms.load_model(args.checkpoint).to(dev)
    batch = to_dev(batch_host, dev)
    g, restore, rec = None, None, None
    if args.mode == "trnsci":
        import trnsparse
        import trntensor

        trnsparse.set_backend(args.backend)
        trntensor.set_backend(args.backend)
        g = Graph(batch_host, batch)
        torch_path = {t for t in args.torch_path.split(",") if t}
        restore = install(model, g, torch_path=torch_path)

    first, med, ts, (e, f) = timed(
        lambda: step(model, batch, dev), dev, args.warm, args.reps, args.budget
    )
    N = len(atoms)
    res = {
        "structure": args.structure,
        "seed": args.seed,
        "atoms": N,
        "edges": int(batch_host["edge_index"].shape[1]),
        "mode": args.mode,
        "device": args.device,
        "threads": torch.get_num_threads(),
        "first_s": first,
        "step_ms": 1e3 * med,
        "step_ms_all": [1e3 * t for t in ts],
        "plan_ms": 1e3 * g.plan_s if g else None,
        "energy": e,
        "energy_ref_cpu": e_ref,
        "dE_per_atom_meV": 1e3 * abs(e - e_ref) / N,
        "forces_rel": float(
            (f.double() - f_ref.double()).norm() / f_ref.double().norm()
        ),
        "forces_max_abs_eV_A": float((f - f_ref).abs().max()),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "backend": args.backend if args.mode == "trnsci" else None,
        "torch_path": args.torch_path if args.mode == "trnsci" else None,
        "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2 (default)"),
    }
    if args.count_ops:
        res["ops"] = count_ops(lambda: step(model, batch, dev))
    if args.shares and args.mode == "trnsci":
        restore()
        rec = []
        restore = install(model, g, rec, torch_path=torch_path)
        step(model, batch, dev)
        rows = replay_shares(rec, dev, args.warm, max(5, args.reps // 2))
        fam = {}
        for r in rows:
            fam[r["family"]] = fam.get(r["family"], 0.0) + r["fwd_bwd_ms"]
        res["calls"] = rows
        res["family_ms"] = fam
        res["family_share"] = {k: v / res["step_ms"] for k, v in fam.items()}
        res["rest_share"] = 1 - sum(res["family_share"].values())
    print(json.dumps({k: v for k, v in res.items() if k not in ("calls",)}, indent=1))
    if rec is not None:
        for r in res["calls"]:
            print(
                f"  {r['family']:3s} {r['name']:44s} fwd {r['fwd_ms']:8.3f} ms  "
                f"fwd+bwd {r['fwd_bwd_ms']:8.3f} ms  grads {r['grads']}"
            )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
