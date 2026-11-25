"""Check the fused ops of ``trnsci_fused.py`` against MACE-MP-0b2 at the exemplar structures.

One energy + forces evaluation of the unmodified model per structure and
seed records, at every site a fused op replaces, the inputs, MACE's output
and the cotangent that output received in the force backward:

- ``sh``: the edge vectors and MACE's spherical harmonics;
- ``edge``: the edge lengths and both layers' ``conv_tp_weights`` and edge
  densities ``tanh(density_fn(·)²)`` (cotangents from the density sums);
- ``zbl``: the ZBL pair energy per edge (cotangent from its receiver sum);
- ``linear``: every node ``o3.Linear`` (``linear_up``, ``linear``, product
  ``linear``, readouts);
- ``skip``: both one-hot ``skip_tp``.

Each fused op then runs on the recorded inputs, and its output and its input
gradient for the *recorded* cotangent (the vector-Jacobian product the force
step needs) are compared with MACE's own modules re-run on the same inputs,
in fp32 and in fp64. Reported per op: relative error against MACE fp32
(the bar is 1e-6 for outputs) and, for both MACE fp32 and the fused op, the
relative error against MACE fp64; a fused gradient is acceptable when it is
no further from fp64 than MACE's own fp32 gradient.

    python check_fused.py                                   # 3 structures x 2 seeds, CPU
    python check_fused.py --device neuron --backend nki     # the NKI kernels
    python check_fused.py --device neuron --region-backend neuron  # compiled regions
    python check_fused.py --trntensor-fixture ../../../trntensor/tests/fixtures/mace_mp0b2_edge.pt
"""

from __future__ import annotations

import argparse
import json
import math

import mace_setup as ms
import torch


def _rel(a, b) -> float:
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    n = float(b.norm())
    return float((a - b).norm()) / n if n > 0 else float((a - b).norm())


def record(model, batch) -> dict:
    """One E+F step of ``model`` with hooks at every fused-op site."""
    from e3nn import o3
    from mace.modules import blocks, radial

    E = int(batch["edge_index"].shape[1])
    rec = {"linear": [], "skip": [], "density": [], "tp_w": []}
    hooks = []

    def keep(t, d, key):
        d[key] = t.detach()
        if t.requires_grad:
            t.register_hook(lambda g: d.__setitem__("grad_" + key, g.detach()))

    def fwd_hook(kind, name, mod, nargs=1):
        def h(m, args, out):
            d = {
                "name": name,
                "module": mod,
                "inputs": [a.detach() for a in args[:nargs]],
            }
            keep(out, d, "out")
            rec[kind].append(d)

        return mod.register_forward_hook(h)

    def sh_hook(m, args, out):
        d = {"inputs": [args[0].detach()]}
        keep(out, d, "out")
        rec["sh"] = d

    hooks.append(model.spherical_harmonics.register_forward_hook(sh_hook))

    def re_hook(m, args, out):
        rec["lengths"] = args[0].detach()

    hooks.append(model.radial_embedding.register_forward_hook(re_hook))
    for li, it in enumerate(model.interactions):
        hooks.append(fwd_hook("tp_w", f"tp_w{li}", it.conv_tp_weights))
        hooks.append(fwd_hook("skip", f"skip_tp L{li}", it.skip_tp, nargs=2))
    for name, mod in model.named_modules():
        if isinstance(mod, o3.Linear) and name.split(".")[0] in (
            "interactions",
            "products",
            "readouts",
        ):
            hooks.append(fwd_hook("linear", name, mod))

    restore = []

    def wrap(modl, kind):
        orig = modl.scatter_sum

        def ss(src, index, dim=-1, out=None, dim_size=None, reduce="sum", _o=orig):
            if src.dim() == 2 and src.shape == (E, 1) and dim == 0:
                d = {}
                keep(src, d, "out")
                if kind == "zbl":
                    rec["zbl"] = d
                else:
                    rec["density"].append(d)
            return _o(src, index, dim=dim, out=out, dim_size=dim_size, reduce=reduce)

        modl.scatter_sum = ss
        restore.append((modl, orig))

    wrap(blocks, "density")
    wrap(radial, "zbl")
    try:
        ms.energy_forces(model, batch)
    finally:
        for h in hooks:
            h.remove()
        for modl, orig in restore:
            modl.scatter_sum = orig
    return rec


def _vjp(fn, inputs, need, cotangents):
    ins = [t.detach().clone().requires_grad_(n) for t, n in zip(inputs, need)]
    outs = fn(*ins)
    outs = outs if isinstance(outs, (tuple, list)) else (outs,)
    pairs = [(o, g) for o, g in zip(outs, cotangents) if g is not None]
    wrt = [t for t, n in zip(ins, need) if n]
    grads = torch.autograd.grad(
        [o for o, _ in pairs], wrt, [g for _, g in pairs], allow_unused=True
    )
    return [o.detach() for o in outs], grads


def compare(name, mace_fn, ours_fn, inputs, need, cot, dev):
    """Output and VJP of ``ours_fn`` (on ``dev``) against ``mace_fn`` in fp32 and fp64."""
    m32, g32 = _vjp(mace_fn(torch.float32), inputs, need, cot)
    m64, g64 = _vjp(
        mace_fn(torch.float64),
        [t.double() if t.is_floating_point() else t for t in inputs],
        need,
        [None if g is None else g.double() for g in cot],
    )
    o, go = _vjp(
        ours_fn,
        [t.to(dev) for t in inputs],
        need,
        [None if g is None else g.to(dev) for g in cot],
    )

    def cat(ts):
        return torch.cat([t.reshape(t.shape[0], -1).double().cpu() for t in ts], 1)

    row = {
        "op": name,
        "out_vs_mace32": _rel(cat(o), cat(m32)),
        "out_mace32_vs_64": _rel(cat(m32), cat(m64)),
        "out_ours_vs_64": _rel(cat(o), cat(m64)),
    }
    if any(need) and any(g is not None for g in g32):
        gm, gt, g6 = (cat([g for g in gs if g is not None]) for gs in (g32, go, g64))
        row.update(
            grad_vs_mace32=_rel(gt, gm),
            grad_mace32_vs_64=_rel(gm, g6),
            grad_ours_vs_64=_rel(gt, g6),
        )
    return row


def check_structure(model, name, seed, dev, ops) -> list:
    import trnsci_fused as tf
    import trntensor

    atoms = ms.build_structure(name, seed)
    b = ms.atoms_to_batch(atoms, model)
    rec = record(model, b)
    ei, na = b["edge_index"], b["node_attrs"]
    sp = na.argmax(-1)
    rows = []

    def as_dtype(mod, dt):
        return mod.to(dt)

    if "sh" in ops:
        s = rec["sh"]
        rows.append(
            compare(
                "spherical_harmonics",
                lambda dt: as_dtype(model, dt).spherical_harmonics,
                trntensor.spherical_harmonics,
                s["inputs"],
                [True],
                [s.get("grad_out")],
                dev,
            )
        )
        model.to(torch.float32)
    if "edge" in ops:
        spec = trntensor.edge_chain_spec(module=model)
        Z = model.atomic_numbers.to(torch.int64)[sp]
        r0 = spec.agnesi_r0(Z[ei[0]], Z[ei[1]]).float()
        cot = []
        for li in range(len(model.interactions)):
            cot += [rec["tp_w"][li].get("grad_out"), rec["density"][li].get("grad_out")]

        def mace_edge(dt):
            m = model.to(dt)

            def f(lengths, r0_):
                ef, _ = m.radial_embedding(lengths, na.to(dt), ei, m.atomic_numbers)
                outs = []
                for it in m.interactions:
                    outs += [it.conv_tp_weights(ef), torch.tanh(it.density_fn(ef) ** 2)]
                return tuple(outs)

            return f

        rows.append(
            compare(
                "edge_chain",
                mace_edge,
                lambda r, r0_: trntensor.edge_chain(r, spec, r0_),
                [rec["lengths"], r0],
                [True, False],
                cot,
                dev,
            )
        )
        model.to(torch.float32)
    if "zbl" in ops and "zbl" in rec:
        k32 = tf.zbl_edge_constants(model, b, dev)

        def mace_zbl(dt):
            m = model.to(dt)
            zb = m.pair_repulsion_fn
            from mace.modules import radial

            def f(lengths):
                # the per-edge part of ZBLBasis.forward, MACE's own ops
                box = {}
                orig = radial.scatter_sum

                def grab(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
                    box["v"] = src
                    return orig(
                        src, index, dim=dim, out=out, dim_size=dim_size, reduce=reduce
                    )

                radial.scatter_sum = grab
                try:
                    zb(lengths, na.to(dt), ei, m.atomic_numbers)
                finally:
                    radial.scatter_sum = orig
                return box["v"]

            return f

        rows.append(
            compare(
                "zbl_edge_energy",
                mace_zbl,
                lambda r: tf.zbl_edge_energy(r, k32),
                [rec["lengths"]],
                [True],
                [rec["zbl"].get("grad_out")],
                dev,
            )
        )
        model.to(torch.float32)
    if "linear" in ops:
        for d in rec["linear"]:
            spec = trntensor.equivariant_linear_spec(d["module"])
            mod = d["module"]
            rows.append(
                compare(
                    f"equivariant_linear {d['name']}",
                    lambda dt, mod=mod: mod.to(dt),
                    lambda x, spec=spec: trntensor.equivariant_linear(x, spec),
                    d["inputs"],
                    [True],
                    [d.get("grad_out")],
                    dev,
                )
            )
            mod.to(torch.float32)
    if "skip" in ops:
        elems = sorted(int(z) for z in sp.unique())
        for d in rec["skip"]:
            mod = d["module"]
            spec = trntensor.equivariant_linear_spec(mod, elements=elems)
            x, attrs = d["inputs"]
            need_x = d.get("grad_out") is not None
            rows.append(
                compare(
                    f"equivariant_linear {d['name']}",
                    lambda dt, mod=mod, attrs=attrs: (
                        lambda x_: mod.to(dt)(x_, attrs.to(dt))
                    ),
                    lambda x_, spec=spec: trntensor.equivariant_linear(
                        x_, spec, sp.to(x_.device)
                    ),
                    [x],
                    [need_x],
                    [d.get("grad_out")],
                    dev,
                )
            )
            mod.to(torch.float32)
    for r in rows:
        r.update(structure=name, seed=seed, atoms=len(atoms), edges=int(ei.shape[1]))
    return rows


def trntensor_fixture(model, path: str, tp_cols: int = 16) -> None:
    """The trntensor CI fixture: MACE-MP-0b2 spherical harmonics and edge chain
    at the 16-atom box (all 150 edges), with the real force-step cotangents.

    The two ``conv_tp_weights`` heads are sliced to their first ``tp_cols``
    outputs (output columns are independent, so the slice is exact); the
    reference input gradients use the sliced cotangents. Also two small
    e3nn-built linears (synthetic weights, e3nn's layout and normalisation)."""
    import trntensor
    from e3nn import o3

    torch.manual_seed(0)
    b = ms.atoms_to_batch(ms.build_structure("molecule_box_16", 0), model)
    rec = record(model, b)
    ei, na = b["edge_index"], b["node_attrs"]
    spec = trntensor.edge_chain_spec(module=model)
    Z = model.atomic_numbers.to(torch.int64)[na.argmax(-1)]
    r0 = spec.agnesi_r0(Z[ei[0]], Z[ei[1]])
    heads, cots = [], []
    for li, it in enumerate(model.interactions):
        h_tp = next(h for h in spec.heads if h.name == f"tp_w{li}")
        heads.append(
            {
                "name": h_tp.name,
                "post": None,
                "weights": list(h_tp.weights[:-1]) + [h_tp.weights[-1][:, :tp_cols]],
            }
        )
        h_d = next(h for h in spec.heads if h.name == f"density{li}")
        heads.append({"name": h_d.name, "post": h_d.post, "weights": list(h_d.weights)})
        cots += [
            rec["tp_w"][li]["grad_out"][:, :tp_cols],
            rec["density"][li]["grad_out"],
        ]

    def mace_edge(dt):
        m = model.to(dt)

        def f(lengths):
            ef, _ = m.radial_embedding(lengths, na.to(dt), ei, m.atomic_numbers)
            o = []
            for it in m.interactions:
                o += [
                    it.conv_tp_weights(ef)[:, :tp_cols],
                    torch.tanh(it.density_fn(ef) ** 2),
                ]
            return tuple(o)

        return f

    L = rec["lengths"]
    o32, (g32,) = _vjp(mace_edge(torch.float32), [L], [True], cots)
    _o64, (g64,) = _vjp(
        mace_edge(torch.float64), [L.double()], [True], [c.double() for c in cots]
    )
    model.to(torch.float32)
    sh = rec["sh"]
    v = sh["inputs"][0]
    y32, (dv32,) = _vjp(model.spherical_harmonics, [v], [True], [sh["grad_out"]])
    fx = {
        "source": "MACE-MP-0b2 medium (mace-torch 0.3.15, e3nn 0.4.4), molecule_box_16 seed 0, "
        "examples/mlip/check_fused.py --trntensor-fixture (trnsci/trnsci#178)",
        "edge_chain": {
            "r_max": spec.r_max,
            "bessel_weights": list(spec.bessel_weights),
            "prefactor": spec.prefactor,
            "cutoff_p": spec.cutoff_p,
            "agnesi": list(spec.agnesi),
            "apply_cutoff": spec.apply_cutoff,
            "heads": heads,
            "lengths": L,
            "r0": r0.float(),
            "out": list(o32),
            "grad_out": cots,
            "grad_lengths": g32,
            "grad_lengths_fp64": g64,
        },
        "sh": {
            "vectors": v,
            "out": y32[0],
            "grad_out": sh["grad_out"],
            "grad_vectors": dv32,
        },
    }
    lin = []
    for irr_in, irr_out in (
        ("8x0e+8x1o+8x2e+8x3o", "8x0e+8x1o+8x2e+8x3o"),
        ("16x0e+8x1o+8x1o+4x2e", "8x0e+4x1o+4x2e"),
    ):
        mod = o3.Linear(irr_in, irr_out)
        x = torch.randn(12, mod.irreps_in.dim)
        g = torch.randn(12, mod.irreps_out.dim)
        y, (dx,) = _vjp(mod, [x], [True], [g])
        M = mod(torch.eye(mod.irreps_in.dim)).detach()  # no bias: f(0) = 0
        lin.append(
            {
                "irreps_in": irr_in,
                "irreps_out": irr_out,
                "matrix": M,
                "x": x,
                "out": y[0],
                "grad_out": g,
                "grad_x": dx,
            }
        )
    fx["linear"] = lin
    torch.save(_compact(fx), path)
    print(f"wrote {path}")


def _compact(obj):
    """Fresh storage for every tensor (a saved view keeps its whole base)."""
    if torch.is_tensor(obj):
        return obj.detach().clone().contiguous()
    if isinstance(obj, dict):
        return {k: _compact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_compact(v) for v in obj)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structures", default=",".join(ms.STRUCTURES))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--ops", default="sh,edge,zbl,linear,skip")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--backend", default="pytorch", help="trntensor backend (nki on the device)"
    )
    ap.add_argument(
        "--region-backend",
        default=None,
        help="torch.compile backend for the fused bodies",
    )
    ap.add_argument("--trntensor-fixture", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    import trntensor

    model = ms.load_model()
    if args.trntensor_fixture:
        trntensor_fixture(model, args.trntensor_fixture)
        return
    trntensor.set_backend(args.backend)
    trntensor.set_region_backend(args.region_backend)
    dev = torch.device(args.device)
    ops = set(args.ops.split(","))
    rows = []
    for name in args.structures.split(","):
        for seed in (int(s) for s in args.seeds.split(",")):
            rows += check_structure(model, name, seed, dev, ops)
    worst: dict = {}
    for r in rows:
        key = (
            r["op"].split(" ")[0]
            if r["op"].startswith("equivariant_linear")
            else r["op"]
        )
        w = worst.setdefault(key, {})
        for k, v in r.items():
            if isinstance(v, float) and not math.isnan(v):
                w[k] = max(w.get(k, 0.0), v)
    print(
        f"{'op':28s} {'out/MACE32':>11s} {'out MACE32/64':>13s} {'out ours/64':>11s} "
        f"{'grad/MACE32':>11s} {'grad MACE32/64':>14s} {'grad ours/64':>12s}"
    )
    for k, w in worst.items():

        def f(n, w=w):
            return f"{w[n]:.2e}" if n in w else "-"

        print(
            f"{k:28s} {f('out_vs_mace32'):>11s} {f('out_mace32_vs_64'):>13s} {f('out_ours_vs_64'):>11s} "
            f"{f('grad_vs_mace32'):>11s} {f('grad_mace32_vs_64'):>14s} {f('grad_ours_vs_64'):>12s}"
        )
    ok = all(w["out_vs_mace32"] <= 1e-6 for w in worst.values()) and all(
        w.get("grad_ours_vs_64", 0.0)
        <= max(1e-6, 1.5 * w.get("grad_mace32_vs_64", 0.0))
        for w in worst.values()
    )
    print("PASS" if ok else "FAIL", "(worst over", len(rows), "op instances)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "worst": worst, "pass": ok}, fh, indent=1)


if __name__ == "__main__":
    main()
