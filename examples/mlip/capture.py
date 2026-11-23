"""Capture block inputs, outputs and gradients from MACE-MP-0b2 medium.

One energy + forces evaluation per structure. Wrappers around the module
forwards alias each block input (``view_as``) and register tensor hooks, so
the backward of the force computation fills in the cotangent of every block
output and the gradient of every block input that depends on positions.
Captured per structure:

- ``A1``: per interaction, conv_tp inputs (x, y, w), output, grad_out and
  the input gradients; the e3nn instruction list and ``w3j`` constants;
- ``A2``: per product, symmetric-contraction input x [N, C, 16], species,
  output, grad_out, grad_x; the U coupling tensors and per-element weights;
- ``B``: every gather by sender and every edge -> node ``scatter_sum``
  (src, index, out, grad_out, grad_src), plus dE/d(edge vector) and forces
  for the force assembly; index statistics.

Writes ``<out>/<structure>.pt`` (full; ~1 GB for the 216-atom slab, kept
out of git, regenerate with this script) and, with ``--fixture``, the small
committed CI fixture (see ``make_fixture``).

    python capture.py --out ~/.xtmp/mlip/captures --fixture fixtures/mace_mp0b2_tiny.pt
"""

from __future__ import annotations

import argparse
import os
import platform

import torch

import mace_setup as ms


def _hook(store: dict, key: str, t: torch.Tensor) -> torch.Tensor:
    """Alias ``t`` and record the gradient that reaches the alias."""
    if not t.requires_grad:
        return t
    a = t.view_as(t)

    def h(g):
        store[key] = g.detach().clone()

    a.register_hook(h)
    return a


def _irreps(irreps) -> list:
    return [(int(mul), int(ir.l)) for mul, ir in irreps]


def tp_spec(tp) -> dict:
    from e3nn import o3

    ins = [
        (int(i.i_in1), int(i.i_in2), int(i.i_out), float(i.path_weight))
        for i in tp.instructions
    ]
    for i in tp.instructions:
        assert (
            i.connection_mode == "uvu" and i.has_weight and tuple(i.path_shape)[1] == 1
        )
    in1, in2, out = (
        _irreps(tp.irreps_in1),
        _irreps(tp.irreps_in2),
        _irreps(tp.irreps_out),
    )
    w3j = {}
    for i1, i2, io, _ in ins:
        key = f"{in1[i1][1]},{in2[i2][1]},{out[io][1]}"
        w3j[key] = o3.wigner_3j(in1[i1][1], in2[i2][1], out[io][1]).to(torch.float64)
    return {
        "irreps_in1": in1,
        "irreps_in2": in2,
        "irreps_out": out,
        "instructions": ins,
        "w3j": w3j,
        "weight_numel": int(tp.weight_numel),
    }


def sc_spec(sc) -> dict:
    cons = []
    for irrep_out, con in zip(sc.irreps_out, sc.contractions):
        corr = int(con.correlation)
        U = {str(nu): con.U_tensors(nu).detach().clone() for nu in range(1, corr + 1)}
        # weights_max is nu = corr; weights[j] is nu = corr - 1 - j
        W = {str(corr): con.weights_max.detach().clone()}
        for j, w in enumerate(con.weights):
            W[str(corr - 1 - j)] = w.detach().clone()
        cons.append(
            {
                "irrep_out": str(irrep_out.ir),
                "d_out": int(irrep_out.ir.dim),
                "correlation": corr,
                "U": U,
                "W": W,
            }
        )
    return {
        "irreps_in": str(sc.irreps_in),
        "irreps_out": str(sc.irreps_out),
        "contractions": cons,
    }


def capture(model, atoms, name: str) -> dict:
    import mace
    import mace.modules.blocks as blocks
    import mace.modules.models as models
    import mace.modules.radial as radial

    batch = ms.atoms_to_batch(atoms, model)
    N = len(atoms)
    E = int(batch["edge_index"].shape[1])
    grads: dict = {}
    rec = {"A1": [], "A2": [], "B_scatter": [], "B_gather": []}
    restore = []
    ctx_box: dict = {}

    # A1: conv_tp
    for li, it in enumerate(model.interactions):
        tp = it.conv_tp
        orig = tp.forward

        def tp_fwd(x, y, w, _orig=orig, li=li, tp=tp):
            k = f"A1.L{li}"
            xa, ya, wa = (
                _hook(grads, k + ".grad_x", x),
                _hook(grads, k + ".grad_y", y),
                _hook(grads, k + ".grad_w", w),
            )
            out = _orig(xa, ya, wa)
            rec["A1"].append(
                {
                    "layer": li,
                    "x": x.detach(),
                    "y": y.detach(),
                    "w": w.detach(),
                    "spec": tp_spec(tp),
                }
            )
            return _hook(grads, k + ".grad_out", out)

        tp.forward = tp_fwd
        restore.append((tp, "forward", None))

        # gather input: linear_up output feeds only node_feats[sender]
        lu = it.linear_up
        orig_lu = lu.forward

        def lu_fwd(x, _orig=orig_lu, li=li):
            out = _orig(x)
            rec["B_gather"].append(
                {"name": f"node_feats[sender] L{li}", "src": out.detach()}
            )
            return _hook(grads, f"Bg.L{li}.grad_src", out)

        lu.forward = lu_fwd
        restore.append((lu, "forward", None))

    # A2: symmetric contraction
    for li, p in enumerate(model.products):
        sc = p.symmetric_contractions
        orig = sc.forward

        def sc_fwd(x, y, _orig=orig, li=li, sc=sc):
            k = f"A2.L{li}"
            xa = _hook(grads, k + ".grad_x", x)
            out = _orig(xa, y)
            rec["A2"].append(
                {
                    "layer": li,
                    "x": x.detach(),
                    "species": y.argmax(-1),
                    "out": out.detach(),
                    "spec": sc_spec(sc),
                }
            )
            return _hook(grads, k + ".grad_out", out)

        sc.forward = sc_fwd
        restore.append((sc, "forward", None))

    # B: scatter_sum onto receivers (edge -> node only)
    def wrap_scatter(mod, where):
        orig = mod.scatter_sum

        def ss(src, index, dim=-1, out=None, dim_size=None, reduce="sum", _orig=orig):
            if (
                index.dim() == 1
                and index.shape[0] == E
                and src.shape[0] == E
                and dim == 0
            ):
                k = f"Bs.{len(rec['B_scatter'])}"
                sa = _hook(grads, k + ".grad_src", src)
                res = _orig(
                    sa, index, dim=dim, out=out, dim_size=dim_size, reduce=reduce
                )
                rec["B_scatter"].append(
                    {
                        "name": where,
                        "src": src.detach(),
                        "index": index,
                        "dim_size": dim_size,
                        "out": res.detach(),
                        "key": k,
                    }
                )
                return _hook(grads, k + ".grad_out", res)
            return _orig(src, index, dim=dim, out=out, dim_size=dim_size, reduce=reduce)

        mod.scatter_sum = ss
        restore.append((mod, "scatter_sum", orig))

    wrap_scatter(blocks, "interaction")
    wrap_scatter(radial, "zbl")

    orig_pg = models.prepare_graph

    def pg(*a, **k):
        ctx = orig_pg(*a, **k)
        ctx_box["vectors"] = ctx.vectors.detach()
        ctx.vectors.register_hook(
            lambda g: grads.__setitem__("edge_vectors.grad", g.detach().clone())
        )
        return ctx

    models.prepare_graph = pg
    restore.append((models, "prepare_graph", orig_pg))

    try:
        energy, forces = ms.energy_forces(model, batch)
    finally:
        for obj, attr, orig in restore:
            if attr == "forward":
                del obj.forward
            else:
                setattr(obj, attr, orig)

    # attach outputs / grads
    for i, a in enumerate(rec["A1"]):
        k = f"A1.L{a['layer']}"
        for g in ("grad_x", "grad_y", "grad_w", "grad_out"):
            a[g] = grads.get(f"{k}.{g}")
    for a in rec["A2"]:
        k = f"A2.L{a['layer']}"
        a["grad_x"], a["grad_out"] = (
            grads.get(k + ".grad_x"),
            grads.get(k + ".grad_out"),
        )
    sender, receiver = batch["edge_index"][0], batch["edge_index"][1]
    for li, g in enumerate(rec["B_gather"]):
        g.update({"index": sender, "grad_src": grads.get(f"Bg.L{li}.grad_src")})
        a1 = rec["A1"][li]
        g["out"], g["grad_out"] = a1["x"], a1["grad_x"]
    names = ["density L0", "messages L0", "density L1", "messages L1"]
    inter = [s for s in rec["B_scatter"] if s["name"] == "interaction"]
    for s, nm in zip(inter, names):
        s["name"] = f"scatter_sum {nm}"
    for s in rec["B_scatter"]:
        if s["name"] == "zbl":
            s["name"] = "scatter_sum ZBL pair energy"
        s["index_role"] = (
            "receiver"
            if torch.equal(s["index"], receiver)
            else ("sender" if torch.equal(s["index"], sender) else "other")
        )
        s["grad_src"] = grads.get(s["key"] + ".grad_src")
        s["grad_out"] = grads.get(s["key"] + ".grad_out")
        del s["key"]
    # Messages share storage with A1 out / grad_out; keep the A1 view as the source of truth.
    for li, a in enumerate(rec["A1"]):
        msg = [
            s for s in rec["B_scatter"] if s["name"] == f"scatter_sum messages L{li}"
        ][0]
        a["out"], a["grad_out"] = msg["src"], msg["grad_src"]

    deg_r = torch.bincount(receiver, minlength=N)
    deg_s = torch.bincount(sender, minlength=N)
    stats = {
        "atoms": N,
        "edges": E,
        "index_dtype": str(sender.dtype),
        "sender_sorted": bool((sender[1:] >= sender[:-1]).all()),
        "receiver_sorted": bool((receiver[1:] >= receiver[:-1]).all()),
        "receiver_degree": {
            "min": int(deg_r.min()),
            "mean": float(deg_r.float().mean()),
            "max": int(deg_r.max()),
        },
        "sender_degree": {
            "min": int(deg_s.min()),
            "mean": float(deg_s.float().mean()),
            "max": int(deg_s.max()),
        },
        "empty_receivers": int((deg_r == 0).sum()),
        "self_image_edges": int((sender == receiver).sum()),
        "max_flat_index_scatter": int(
            N * max(s["src"].shape[1] for s in rec["B_scatter"])
        ),
        "max_flat_index_edge": int(
            E * max(s["src"].shape[1] for s in rec["B_scatter"])
        ),
    }
    return {
        "meta": {
            "structure": name,
            "symbols": atoms.get_chemical_formula(),
            "mace_version": str(mace.__version__),
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
            "checkpoint_url": ms.CHECKPOINT_URL,
            "checkpoint_sha256": ms.CHECKPOINT_SHA256,
            "dtype": str(next(model.parameters()).dtype),
        },
        "positions": batch["positions"].detach(),
        "cell": batch["cell"].detach(),
        "species": batch["node_attrs"].argmax(-1),
        "sender": sender,
        "receiver": receiver,
        "shifts": batch["shifts"].detach(),
        "edge_vectors": ctx_box["vectors"],
        "edge_vectors_grad": grads["edge_vectors.grad"],
        "energy": energy.detach(),
        "forces": forces.detach(),
        "stats": stats,
        **rec,
    }


# --------------------------------------------------------------------- fixture


def _coo(u: torch.Tensor) -> dict:
    nz = u.nonzero(as_tuple=False).t().contiguous()
    return {
        "shape": list(u.shape),
        "indices": nz.to(torch.int16),
        "values": u[tuple(nz)].clone(),
    }


def make_fixture(cap: dict, channels: int = 8, edges: int = 24) -> dict:
    """Small, committed slice of one capture (CI runs it without MACE).

    Every block is independent across channels except the A1 gradient
    w.r.t. y, which sums over channels; so the fixture keeps
    ``channels`` channels of each irrep block, all nodes, and ``edges``
    edges for A1. Captured tensors are sliced exactly; the one quantity
    that cannot be sliced (A1 grad_y) is recomputed with the reference
    on the slice, and ``check.py --capture`` checks it against MACE on
    the full tensors. U is stored as COO; weights keep only the elements
    present."""
    from blocks import conv_tp

    def chan_idx(irreps, c):
        idx, o = [], 0
        for mul, ell in irreps:
            d = 2 * ell + 1
            idx += [o + u * d + m for u in range(c) for m in range(d)]
            o += mul * d
        return torch.tensor(idx)

    def sliced_irreps(irreps, c):
        return [(c, ell) for _, ell in irreps]

    fx = {
        "meta": dict(cap["meta"], fixture=f"channels={channels}, A1 edges={edges}"),
        "stats": cap["stats"],
    }
    for k in (
        "positions",
        "cell",
        "species",
        "sender",
        "receiver",
        "shifts",
        "edge_vectors",
        "edge_vectors_grad",
        "energy",
        "forces",
    ):
        fx[k] = cap[k]

    E = cap["stats"]["edges"]
    eidx = torch.linspace(0, E - 1, edges).round().long().unique()
    fx["A1"] = []
    for a in cap["A1"]:
        sp = a["spec"]
        ix, io = (
            chan_idx(sp["irreps_in1"], channels),
            chan_idx(sp["irreps_out"], channels),
        )
        wi = torch.cat(
            [
                torch.arange(
                    p * sp["irreps_in1"][i1][0], p * sp["irreps_in1"][i1][0] + channels
                )
                for p, (i1, *_r) in enumerate(sp["instructions"])
            ]
        )
        spec = dict(
            sp,
            irreps_in1=sliced_irreps(sp["irreps_in1"], channels),
            irreps_out=sliced_irreps(sp["irreps_out"], channels),
            weight_numel=len(wi),
        )
        x, y, w = a["x"][eidx][:, ix], a["y"][eidx], a["w"][eidx][:, wi]
        g = a["grad_out"][eidx][:, io]
        yv = y.clone().requires_grad_(True)
        gy = torch.autograd.grad(conv_tp(x, yv, w, spec), yv, g)[0]
        fx["A1"].append(
            {
                "layer": a["layer"],
                "edges": eidx,
                "x": x,
                "y": y,
                "w": w,
                "out": a["out"][eidx][:, io],
                "grad_out": g,
                "grad_x": None if a["grad_x"] is None else a["grad_x"][eidx][:, ix],
                "grad_w": a["grad_w"][eidx][:, wi],
                "grad_y": gy,
                "grad_y_from": "reference on the slice",
                "spec": spec,
            }
        )

    present = torch.unique(cap["species"])
    remap = torch.full((int(present.max()) + 1,), -1, dtype=torch.long)
    remap[present] = torch.arange(len(present))
    fx["A2"] = []
    for a in cap["A2"]:
        C = a["x"].shape[1]
        cons = []
        oidx, o = [], 0
        for con in a["spec"]["contractions"]:
            d = con["d_out"]
            oidx += [o + c * d + m for c in range(channels) for m in range(d)]
            o += C * d
            cons.append(
                dict(
                    con,
                    U={k: _coo(v) for k, v in con["U"].items()},
                    W={
                        k: v[present][:, :, :channels].clone()
                        for k, v in con["W"].items()
                    },
                )
            )
        oidx = torch.tensor(oidx)
        fx["A2"].append(
            {
                "layer": a["layer"],
                "x": a["x"][:, :channels].clone(),
                "species": remap[a["species"]],
                "species_z_index": present,
                "out": a["out"][:, oidx],
                "grad_out": a["grad_out"][:, oidx],
                "grad_x": a["grad_x"][:, :channels].clone(),
                "spec": dict(a["spec"], contractions=cons),
            }
        )

    def keep(t, c):
        return None if t is None else t[:, :c].clone()

    fx["B_scatter"] = [
        dict(
            s,
            src=keep(s["src"], channels * 2),
            out=keep(s["out"], channels * 2),
            grad_src=keep(s["grad_src"], channels * 2),
            grad_out=keep(s["grad_out"], channels * 2),
        )
        for s in cap["B_scatter"]
    ]
    fx["B_gather"] = [
        dict(
            g,
            src=keep(g["src"], channels * 2),
            out=keep(g["out"], channels * 2),
            grad_src=keep(g["grad_src"], channels * 2),
            grad_out=keep(g["grad_out"], channels * 2),
        )
        for g in cap["B_gather"]
    ]
    return fx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structures", default=",".join(ms.STRUCTURES))
    ap.add_argument("--out", required=True, help="directory for full captures")
    ap.add_argument(
        "--fixture",
        default=None,
        help="write the small CI fixture from the first structure",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="rattle seed (1 gives a second geometry)"
    )
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    model = ms.load_model(args.checkpoint, args.dtype)
    os.makedirs(args.out, exist_ok=True)
    for i, name in enumerate(args.structures.split(",")):
        cap = capture(model, ms.build_structure(name, args.seed), name)
        suffix = "" if args.seed == 0 else f"_seed{args.seed}"
        path = os.path.join(args.out, f"{name}{suffix}.pt")
        torch.save(cap, path)
        print(f"{path}: {os.path.getsize(path) / 2**20:.1f} MiB, {cap['stats']}")
        if args.fixture and i == 0:
            torch.save(make_fixture(cap), args.fixture)
            print(f"{args.fixture}: {os.path.getsize(args.fixture) / 2**10:.0f} KiB")


if __name__ == "__main__":
    main()
