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
aten on the same device, unless ``--fuse`` swaps it for the fused library
ops in ``trnsci_fused.py`` (trnsci/trnsci#178): ``sh`` (spherical
harmonics), ``edge`` (radial basis + both layers' radial and density MLPs),
``zbl`` (ZBL per edge), ``linear`` (e3nn ``Linear``s) and ``skip`` (one-hot
``skip_tp``), or ``all``. ``--region-backend neuron`` runs the fused ops'
PyTorch bodies as one compiled graph per direction; ``--backend nki`` puts
the spherical harmonics and the edge chain on their NKI kernels. Forces are
``-dE/dpositions`` by autograd through the registered backwards.
``--mode mace`` runs the unmodified model.

    python mace_trnsci_step.py --structure tic001_slab_216 --mode mace --device cpu
    python mace_trnsci_step.py --structure tic001_slab_216 --mode trnsci --device neuron \
        --shares --json step.json
    python mace_trnsci_step.py --mode trnsci --device neuron --fuse all \
        --region-backend neuron --count-ops --json fused.json

``--batch B`` evaluates B structures (the ``--structure`` list cycled, seeds
``seed .. seed+B-1``) in one step, as one block-diagonal graph
(``batching.make_batch``): atoms and edges concatenated with offset indices,
one sender and one receiver ``segment_plan`` for the whole batch, and the
totals padded to a shape bucket so a new geometry or batch of similar size
reuses the compiled shapes (``--pad``, ``--bucket-shift``). The padding edges
are masked out of every segment sum (``segment_plan(..., padding=mask)``).
The batch is split back into per-structure energies and forces;
``--split-check`` reruns each structure alone at the batch's own shapes and
compares (bitwise for the trnsci path on the CPU).

    python mace_trnsci_step.py --structure tic001_slab_216 --batch 8 --device neuron

Each structure is checked against the unmodified model on the host CPU
(fp32), run alone and unpadded (``cpu_ref.py``): energy difference per atom
and the relative Frobenius error of the forces. A missing reference raises.
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

    def __init__(self, batch_host: dict, batch_dev: dict, edge_padding=None) -> None:
        import trnsparse

        ei = batch_host["edge_index"]
        self.N = int(batch_host["positions"].shape[0])
        self.E = int(ei.shape[1])
        self.edge_padding = edge_padding
        dev = batch_dev["edge_index"].device
        t0 = time.perf_counter()
        kw = {} if edge_padding is None else {"padding": edge_padding}
        self.splan = trnsparse.segment_plan(ei[0], self.N, **kw).to(dev)
        self.rplan = trnsparse.segment_plan(ei[1], self.N, **kw).to(dev)
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


def _recorder(rec: list):
    """``blk(family, name, fn, *tensors)`` recording into ``rec`` (as ``install``)."""

    def blk(family, name, fn, *ts):
        out = fn(*ts)
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
            out.register_hook(lambda go, e=e: e.__setitem__("grad_out", go.detach()))
        return out

    return blk


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
            fused = getattr(it, "_trnsci_edge", None)  # trnsci_fused "edge"
            if fused is not None:
                tp_w, dens = fused()
            else:
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


def step_outputs(model, batch, stress=False, keys=("energy", "forces")) -> dict:
    """One energy + forces (+ stress) step; the outputs in ``keys`` on the host.

    ``energy`` is per structure of the batch (the padding structure last),
    ``node_energy`` per atom (MACE returns it in fp64), ``stress`` per
    structure when ``stress``."""
    b = dict(batch)
    b["positions"] = b["positions"].detach().clone()
    out = model(b, compute_force=True, compute_stress=stress, training=False)
    host = {}
    for k in keys:
        v = out.get(k)
        host[k] = None if v is None else v.detach().to("cpu")
    return host


def bucket_kwargs(args) -> dict:
    """``batching.make_batch`` options from the shared command-line flags."""
    return {
        "pad": args.pad != "none",
        "shift": args.bucket_shift,
        "atom_multiple": args.atom_multiple,
        "edge_multiple": args.edge_multiple,
    }


def add_batch_args(ap) -> None:
    """The batching / bucket flags shared by the step scripts and accept.py."""
    ap.add_argument(
        "--batch",
        type=int,
        default=1,
        help="structures per step (block-diagonal graph; seeds seed..seed+B-1)",
    )
    ap.add_argument(
        "--pad",
        choices=["bucket", "none"],
        default="bucket",
        help="pad atoms and edges to a shape bucket (masked padding edges)",
    )
    ap.add_argument(
        "--bucket-shift",
        type=int,
        default=4,
        help="bucket quantum 2^(floor(log2 n) - shift): padding <= 2^-shift",
    )
    ap.add_argument("--atom-multiple", type=int, default=0, help="fixed atom bucket")
    ap.add_argument("--edge-multiple", type=int, default=0, help="fixed edge bucket")


class Step:
    """B structures as one padded block-diagonal batch on ``dev``, ready to run.

    ``mode="trnsci"`` builds one ``Graph`` (a sender and a receiver
    ``segment_plan``, with the padding mask) for the whole batch and patches
    the three blocks while the step runs (``with step:``); ``mode="mace"``
    runs MACE unmodified. ``batch_host`` / ``info`` come from
    ``batching.make_batch``, or are passed in (``isolate``).
    """

    def __init__(
        self,
        model,
        atoms_list,
        dev,
        *,
        mode="trnsci",
        torch_path=(),
        stress=False,
        node_energy=False,
        batch=None,
        fuse=(),
        model_cpu=None,
        **bucket,
    ):
        import batching

        self.model, self.dev, self.mode = model, torch.device(dev), mode
        self.torch_path, self.stress = set(torch_path), stress
        if fuse and mode != "trnsci":
            raise ValueError("fused ops (--fuse) need mode='trnsci'")
        # trnsci_fused builds its specs from a host copy of the model, once
        self.fuse, self.model_cpu, self._fuse_ctx = list(fuse), model_cpu, {}
        self.rec = None
        self.keys = ("energy", "forces")
        self.keys += ("node_energy",) * bool(node_energy) + ("stress",) * bool(stress)
        self.atoms_list = atoms_list
        if batch is None:
            batch = batching.make_batch(atoms_list, model, **bucket)
        self.batch_host, self.info = batch
        self.batch = to_dev(self.batch_host, self.dev)
        self.g = None
        self._restore = None
        if mode == "trnsci":
            pad = self.info.edge_padding
            self.g = Graph(
                self.batch_host,
                self.batch,
                pad if self.info.e_pad > self.info.e_real else None,
            )

    def __enter__(self):
        if self.mode == "trnsci":
            undo = [install(self.model, self.g, self.rec, torch_path=self.torch_path)]
            if self.fuse:
                import trnsci_fused

                if self.model_cpu is None:
                    import copy

                    self.model_cpu = copy.deepcopy(self.model).to("cpu")
                ctx = self._fuse_ctx
                ctx.update(
                    dev=self.dev, model_cpu=self.model_cpu, batch_host=self.batch_host
                )
                blk = _recorder(self.rec) if self.rec is not None else None
                undo.append(
                    trnsci_fused.install_fused(self.model, self.g, self.fuse, ctx, blk)
                )

            def restore():
                for u in reversed(undo):
                    u()

            self._restore = restore
        return self

    @contextlib.contextmanager
    def recording(self, rec: list):
        """Patched as ``with step:``, recording every block and fused call into ``rec``."""
        self.rec = rec
        try:
            with self:
                yield self
        finally:
            self.rec = None

    def __exit__(self, *exc):
        if self._restore is not None:
            self._restore()
            self._restore = None

    def run(self) -> dict:
        """One step on the device; outputs on the host (patch with ``with``)."""
        return step_outputs(self.model, self.batch, self.stress, self.keys)

    def __call__(self) -> dict:
        with self:
            return self.run()

    def split(self, out: dict) -> list:
        """Per-structure ``{"energy", "forces", "node_energy", "stress"}``."""
        info, rows = self.info, []
        for i in range(info.num_structures):
            a0, n = info.atom_start[i], info.atoms[i]
            ne = out.get("node_energy")
            st = out.get("stress")
            rows.append(
                {
                    "energy": float(out["energy"][i]),
                    "forces": out["forces"][a0 : a0 + n],
                    "node_energy": None if ne is None else ne[a0 : a0 + n],
                    "stress": None if st is None else st[i],
                }
            )
        return rows

    def isolated(self, i: int) -> Step:
        """Structure ``i`` alone, at the same shapes and offsets (``isolate``)."""
        import batching

        return Step(
            self.model,
            [self.atoms_list[i]],
            self.dev,
            mode=self.mode,
            torch_path=self.torch_path,
            stress=self.stress,
            node_energy="node_energy" in self.keys,
            batch=batching.isolate(self.batch_host, self.info, i),
            fuse=self.fuse,
            model_cpu=self.model_cpu,
        )


def split_check(step: Step, bitwise: bool, warm: int = 2) -> dict:
    """Each structure of the batch run alone at the batch's own shapes
    (``Step.isolated``) against its slice of the batched step. With
    ``bitwise`` any difference raises: nothing of the other structures, and
    nothing of the padding, may reach a structure's energy or forces.

    Every step is warmed up first (``warm`` calls): e3nn's TorchScript
    modules run a profiling pass before their optimised graph, and on the
    CPU the first call's forces differ from later calls' in the last bits."""
    with step:
        for _ in range(warm):
            step.run()
        full = step.split(step.run())
    rows = []
    for i in range(step.info.num_structures):
        one = step.isolated(i)
        with one:
            for _ in range(warm):
                one.run()
            alone = one.split(one.run())[i]
        e_eq = alone["energy"] == full[i]["energy"]
        f_eq = torch.equal(alone["forces"], full[i]["forces"])
        rows.append(
            {
                "structure": i,
                "energy_bitwise": bool(e_eq),
                "forces_bitwise": bool(f_eq),
                "dE_eV": abs(alone["energy"] - full[i]["energy"]),
                "dF_max_eV_A": float(
                    (alone["forces"].double() - full[i]["forces"].double()).abs().max()
                ),
            }
        )
    res = {"bitwise": all(r["energy_bitwise"] and r["forces_bitwise"] for r in rows)}
    res["rows"] = rows
    if bitwise and not res["bitwise"]:
        bad = [r for r in rows if not (r["energy_bitwise"] and r["forces_bitwise"])]
        raise AssertionError(f"batched split differs from the isolated runs: {bad}")
    return res


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


def _count(fn) -> int:
    """Ops one call of ``fn`` dispatches (forward and backward; as ``count_ops``)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    n = [0]

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            n[0] += 1
            return func(*args, **(kwargs or {}))

    with torch.autograd.set_multithreading_enabled(False), Count():
        fn()
    return n[0]


def launch_estimate(rec: list) -> dict:
    """Per family (A1, A2, B: the trnsci blocks; F: the fused ops), the ops its
    calls dispatch on this run (each replayed alone, forward + the input
    gradients the step needed) and the launches they cost as kernels or
    compiled regions on the device: one per direction."""
    fam: dict = {}
    for e in rec:
        need, go = e["need"], e["grad_out"]

        def fb(e=e, need=need, go=go):
            ins = [
                t.detach().requires_grad_(True) if n else t
                for t, n in zip(e["inputs"], need)
            ]
            out = e["fn"](*ins)
            wrt = [t for t, n in zip(ins, need) if n]
            if wrt and go is not None:
                torch.autograd.grad(out, wrt, go)

        f = fam.setdefault(e["family"], {"calls": 0, "ops": 0, "launches": 0})
        f["calls"] += 1
        f["ops"] += _count(fb)
        f["launches"] += 1 + int(go is not None and any(need))
    return fam


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


def measured_dispatches(fn) -> dict:
    """Dispatches of one call of ``fn`` as the host issues them, from the
    PyTorch profiler: top-level operators (aten, the trnsci custom ops) plus
    one per compiled-region call (``Torch-Compiled Region`` / compiled
    autograd functions), not counting anything nested inside those. Unlike
    ``count_ops`` this works with compiled regions on, since it needs no
    dispatch mode (TorchDynamo refuses to trace under one)."""
    import collections

    from torch.profiler import ProfilerActivity, profile

    def is_region(n):
        return n.startswith(("Torch-Compiled Region", "CompiledFunction"))

    def is_op(n):
        return "::" in n and n.split("::")[0] in (
            "aten",
            "trntensor",
            "trnsparse",
            "neuron",
        )

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    counts: collections.Counter = collections.Counter()
    for e in prof.events():
        n = e.name
        if not (is_op(n) or is_region(n)):
            continue
        p, nested = e.cpu_parent, False
        while p is not None:
            if is_op(p.name) or is_region(p.name):
                nested = True
                break
            p = p.cpu_parent
        if not nested:
            counts["compiled_region" if is_region(n) else n] += 1
    kernels = sum(
        v for k, v in counts.items() if k.split("::")[0] in ("trntensor", "trnsparse")
    )

    def launches(name):
        # views and composite ops that alias or return the input when nothing
        # changes issue no device work by themselves; everything else may
        if not name.startswith("aten::"):
            return True
        op = name.split("::", 1)[1]
        if op in _COMPOSITE_MAYBE_COPY:
            return False
        packet = getattr(torch.ops.aten, op, None)
        if packet is None:
            return True
        try:
            return not any(getattr(packet, o).is_view for o in packet.overloads())
        except Exception:  # noqa: BLE001 - unknown overload metadata: count it
            return True

    return {
        "total": sum(counts.values()),
        "non_view": sum(v for k, v in counts.items() if launches(k)),
        "compiled_region_calls": counts.get("compiled_region", 0),
        "trnsci_kernels": kernels,
        "top": counts.most_common(15),
        "top_non_view": [kv for kv in counts.most_common() if launches(kv[0])][:15],
    }


# Composite aten ops that are views or no-ops unless they must copy (reshape of
# a non-contiguous tensor, ``to`` a new dtype/device, ...): left out of
# ``measured_dispatches()["non_view"]``; any copy they do shows up as a nested
# ``aten::copy_`` / ``aten::_to_copy``, which is not counted either, so
# ``non_view`` is a lower bound and ``total`` an upper bound.
_COMPOSITE_MAYBE_COPY = frozenset(
    {"to", "reshape", "contiguous", "flatten", "type_as", "view_as", "expand_as"}
    | {"split", "chunk", "unbind", "narrow", "unflatten", "movedim", "swapaxes"}
)


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


def structure_list(names: str, seed: int, batch: int) -> list:
    """``batch`` structures: the comma list ``names`` cycled, seeds
    ``seed, seed + 1, ...`` (``batch=1``: the one structure at ``seed``)."""
    ns = [n for n in names.split(",") if n]
    return [ms.build_structure(ns[k % len(ns)], seed + k) for k in range(batch)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structure", default="tic001_slab_216", help="comma list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["mace", "trnsci"], default="trnsci")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="threads for the CPU reference (default: --threads)",
    )
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
    add_batch_args(ap)
    ap.add_argument(
        "--fuse",
        default="",
        help="comma list of fused ops for the aten remainder: sh,edge,zbl,linear,skip or all",
    )
    ap.add_argument(
        "--region-backend",
        default=None,
        help="torch.compile backend for the fused ops' bodies (neuron on the device)",
    )
    ap.add_argument(
        "--split-check",
        choices=["auto", "bitwise", "report", "off"],
        default="auto",
        help="run each structure alone at the batch's shapes and compare with its "
        "slice of the batch (auto: bitwise for trnsci on the cpu, off elsewhere; "
        "report: record the differences; always off for --batch 1)",
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
    import cpu_ref

    model_cpu = ms.load_model(args.checkpoint)
    atoms_list = structure_list(args.structure, args.seed, args.batch)
    refs = cpu_ref.cpu_reference(
        model_cpu, atoms_list, args.cpu_threads or args.threads
    )

    model = ms.load_model(args.checkpoint).to(dev)
    torch_path = {t for t in args.torch_path.split(",") if t}
    fuse = [f for f in args.fuse.split(",") if f]
    if fuse and args.mode != "trnsci":
        raise SystemExit("--fuse needs --mode trnsci")
    if args.mode == "trnsci":
        import trnsparse
        import trntensor

        trnsparse.set_backend(args.backend)
        trntensor.set_backend(args.backend)
        if fuse or args.region_backend:
            trntensor.set_region_backend(args.region_backend)
    st = Step(
        model,
        atoms_list,
        dev,
        mode=args.mode,
        torch_path=torch_path,
        fuse=fuse,
        model_cpu=model_cpu if fuse else None,
        **bucket_kwargs(args),
    )
    with st:
        first, med, ts, out = timed(st.run, dev, args.warm, args.reps, args.budget)
    parts = st.split(out)
    names = [n for n in args.structure.split(",") if n]
    per = [
        {"structure": names[k % len(names)], "seed": args.seed + k}
        | cpu_ref.compare(p["energy"], p["forces"], r, f"structure {k}")
        for k, (p, r) in enumerate(zip(parts, refs))
    ]
    info = st.info
    res = {
        "structure": args.structure,
        "seed": args.seed,
        "batch": args.batch,
        "atoms": info.n_real,
        "edges": info.e_real,
        "bucket": list(info.key),
        "layout": info.as_dict(),
        "fuse": fuse,
        "region_backend": args.region_backend,
        "mode": args.mode,
        "device": args.device,
        "threads": torch.get_num_threads(),
        "first_s": first,
        "step_ms": 1e3 * med,
        "step_ms_all": [1e3 * t for t in ts],
        "per_structure_ms": 1e3 * med / args.batch,
        "atoms_per_s": info.n_real / med,
        "plan_ms": 1e3 * st.g.plan_s if st.g else None,
        **cpu_ref.worst(per),
        "per_structure": per,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "backend": args.backend if args.mode == "trnsci" else None,
        "torch_path": args.torch_path if args.mode == "trnsci" else None,
        "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2 (default)"),
    }
    mode = args.split_check
    if mode == "auto":
        on_cpu = dev.type == "cpu"
        mode = "bitwise" if on_cpu and args.mode == "trnsci" else "off"
    if args.batch > 1 and mode != "off" and args.pad != "none":
        res["split_check"] = split_check(st, bitwise=mode == "bitwise")
    if args.count_ops:
        regions = args.mode == "trnsci" and hasattr(trntensor, "get_region_backend")
        measured = None
        if args.mode == "trnsci":
            # Measured with the regions on. The dispatch-mode counts below
            # (count_ops, launch_estimate) run with them off: TorchDynamo does
            # not trace under a dispatch mode.
            with st:
                measured = measured_dispatches(st.run)
        prev_region = trntensor.get_region_backend() if regions else None
        if regions:
            trntensor.set_region_backend(None)
        try:
            with st:
                res["ops"] = count_ops(st.run)
            if args.mode == "trnsci":
                res["ops"]["measured_dispatches"] = measured
                # Expected device dispatches: every recorded block / fused call as
                # one kernel or compiled graph per direction, the rest as counted.
                rec_c = []
                with st.recording(rec_c):
                    st.run()
                fam = launch_estimate(rec_c)
                res["ops"]["by_family"] = fam
                res["ops"]["device_estimate"] = (
                    res["ops"]["total"]
                    - sum(f["ops"] for f in fam.values())
                    + sum(f["launches"] for f in fam.values())
                )
        finally:
            if regions:
                trntensor.set_region_backend(prev_region)
    if args.shares and args.mode == "trnsci":
        rec = []
        with st.recording(rec):
            st.run()
        rows = replay_shares(rec, dev, args.warm, max(5, args.reps // 2))
        fam = {}
        for r in rows:
            fam[r["family"]] = fam.get(r["family"], 0.0) + r["fwd_bwd_ms"]
        res["calls"] = rows
        res["family_ms"] = fam
        res["family_share"] = {k: v / res["step_ms"] for k, v in fam.items()}
        res["rest_share"] = 1 - sum(res["family_share"].values())
    show = {k: v for k, v in res.items() if k not in ("calls", "per_structure")}
    print(json.dumps(show, indent=1))
    for r in per:
        print(
            f"  {r['structure']:12s} seed {r['seed']:3d}  dE {r['dE_per_atom_meV']:.4f}"
            f" meV/atom  forces rel {r['forces_rel']:.2e}  ref {r['cpu_ref_s']:.2f} s"
        )
    for r in res.get("calls", []):
        print(
            f"  {r['family']:3s} {r['name']:44s} fwd {r['fwd_ms']:8.3f} ms  "
            f"fwd+bwd {r['fwd_bwd_ms']:8.3f} ms  grads {r['grads']}"
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
