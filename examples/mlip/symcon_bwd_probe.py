"""Why the trntensor PyTorch-path backward of ``symmetric_contraction`` is slow on device.

trnsci/trntensor#140: on Trn2 the PyTorch path (the library's own aten
formulation, not the NKI kernel) took 4.2 s for the layer-0 backward against
a 3.3 ms forward. This probe runs the MACE-MP-0b2 layer-0 and layer-1
contractions at 216 atoms on one device and reports, for the forward and
for the backward (``torch.autograd.grad``):

- the wall time (median, synchronised, after warm-up);
- every dispatched aten op with its shape and its synchronised time, so the
  slow op stands out;
- the same backward with the ``index_add_`` scatter over the monomial
  factors replaced by one GEMM against the 0/1 factor-selection matrix
  (``--variant``), checked against the fp64 host reference.

    python symcon_bwd_probe.py --device neuron --json symcon_bwd.json
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time

import mace_setup as ms
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def sync(dev):
    if dev.type == "neuron":
        torch.neuron.synchronize()


def timed(fn, dev, warm=2, reps=5):
    t0 = time.perf_counter()
    fn()
    sync(dev)
    first = time.perf_counter() - t0
    for _ in range(warm):
        fn()
    sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    return first, statistics.median(ts)


class OpTimer(TorchDispatchMode):
    """Each dispatched op, synchronised: (name, shapes, seconds)."""

    def __init__(self, dev):
        super().__init__()
        self.dev = dev
        self.rows = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        sync(self.dev)
        t0 = time.perf_counter()
        out = func(*args, **(kwargs or {}))
        sync(self.dev)
        dt = time.perf_counter() - t0
        shapes = [tuple(a.shape) for a in args if torch.is_tensor(a)]
        self.rows.append((str(func.overloadpacket), str(shapes)[:90], dt))
        return out


def op_table(rows, top=12):
    agg = collections.defaultdict(lambda: [0, 0.0])
    for name, shapes, dt in rows:
        a = agg[(name, shapes)]
        a[0] += 1
        a[1] += dt
    out = sorted(agg.items(), key=lambda kv: -kv[1][1])[:top]
    return [
        {"op": k[0], "shapes": k[1], "calls": v[0], "ms": 1e3 * v[1]} for k, v in out
    ]


def backward_gemm(go, xr, wr, c, L):
    """The PyTorch-path backward with the factor scatter as one GEMM.

    ``dxa[:, a] = sum_j sum_alpha [factor j of alpha == a] * gm_j[:, alpha]``,
    i.e. ``[gm_0 | gm_1 | gm_2] @ sel_stack`` with ``sel_stack [3A, L+1]``
    the 0/1 selection matrix, instead of three ``index_add_`` along dim 1."""
    from trntensor.equivariant import _monomials

    x1, x2, x3 = _monomials(xr, c)
    gm = ((go @ c["ST"]) * wr) @ c["usymT"]
    parts = torch.cat([gm * x2 * x3, gm * x1 * x3, gm * x1 * x2], dim=1)
    sel = c["selT"].permute(1, 0, 2).reshape(-1, L + 1)  # [3A, L+1]
    return (parts @ sel)[:, :L]


def run_layer(layer, model, species, dev, variant):
    import trntensor
    from trntensor import equivariant as eq

    sc = model.products[layer].symmetric_contractions
    spec = trntensor.symmetric_contraction_spec(module=sc)
    N = species.shape[0]
    C, L = spec.channels, spec.num_features
    g = torch.Generator().manual_seed(layer)
    x_h = torch.randn(N, C, L, generator=g) * 0.5
    go_h = torch.randn(N, spec.dim_out, generator=g)
    lay = spec.layout

    # fp64 host reference of the input gradient
    xd = x_h.double().requires_grad_(True)
    with trntensor_backend("pytorch"):
        out_ref = trntensor.symmetric_contraction(xd, species, spec)
        (dx_ref,) = torch.autograd.grad(out_ref, xd, go_h.double())

    x = x_h.to(dev)
    go = go_h.to(dev)
    sp = species.to(dev)
    res = {
        "layer": layer,
        "N": N,
        "C": C,
        "L": L,
        "A": lay.A,
        "MK": lay.MK,
        "Dsum": lay.Dsum,
    }

    def fwd():
        with torch.no_grad():
            return trntensor.symmetric_contraction(x, sp, spec)

    def fb():
        xx = x.detach().requires_grad_(True)
        out = trntensor.symmetric_contraction(xx, sp, spec)
        return torch.autograd.grad(out, xx, go)[0]

    def bwd_only():
        return eq._symcon_backward_torch(go, x, sp, spec)

    def bwd_gemm():
        c = spec.constants(x.device, x.dtype)
        gop = go.index_select(1, c["out_gather_inv"]).reshape(-1, lay.Dsum)
        xr = x.reshape(-1, L)
        wr = c["W_zcm"].index_select(0, sp).reshape(-1, lay.MK)
        return backward_gemm(gop, xr, wr, c, L).reshape(N, C, L)

    with trntensor_backend("pytorch"):
        for name, fn in (("fwd", fwd), ("bwd_only", bwd_only), ("fwd_bwd", fb)):
            first, med = timed(fn, dev, reps=3 if name != "fwd" else 5)
            res[name] = {"first_s": first, "ms": 1e3 * med}
            print(
                f"L{layer} {name:9s} first {first:8.2f} s  median {1e3 * med:9.2f} ms",
                flush=True,
            )
        dx = fb().cpu()
        res["fwd_bwd"]["rel_err_vs_fp64"] = float(
            (dx.double() - dx_ref).norm() / dx_ref.norm()
        )
        with torch.autograd.set_multithreading_enabled(False):
            t = OpTimer(dev)
            with t:
                bwd_only()
            res["bwd_only"]["ops"] = len(t.rows)
            res["bwd_only"]["sum_sync_ms"] = 1e3 * sum(r[2] for r in t.rows)
            res["bwd_only"]["top"] = op_table(t.rows)
            t = OpTimer(dev)
            with t:
                fwd()
            res["fwd"]["ops"] = len(t.rows)
            res["fwd"]["top"] = op_table(t.rows, 6)
        if variant:
            first, med = timed(bwd_gemm, dev, reps=5)
            dxg = bwd_gemm().cpu()
            res["bwd_gemm"] = {
                "first_s": first,
                "ms": 1e3 * med,
                "rel_err_vs_fp64": float(
                    (dxg.double() - dx_ref).norm() / dx_ref.norm()
                ),
            }
            t = OpTimer(dev)
            with t:
                bwd_gemm()
            res["bwd_gemm"]["ops"] = len(t.rows)
            res["bwd_gemm"]["top"] = op_table(t.rows, 6)
            print(
                f"L{layer} bwd_gemm  first {first:8.2f} s  median {1e3 * med:9.2f} ms",
                flush=True,
            )
    for k in ("fwd", "bwd_only", "bwd_gemm"):
        if k in res and "top" in res[k]:
            print(f"  L{layer} {k}: {res[k].get('ops')} ops")
            for r in res[k]["top"]:
                print(
                    f"    {r['ms']:9.3f} ms x{r['calls']:<3d} {r['op']:24s} {r['shapes']}"
                )
    return res


class trntensor_backend:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        import trntensor

        self.old = trntensor.get_backend()
        trntensor.set_backend(self.name)

    def __exit__(self, *a):
        import trntensor

        trntensor.set_backend(self.old)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structure", default="tic001_slab_216")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layers", default="0,1")
    ap.add_argument(
        "--variant", action="store_true", help="also time the GEMM backward"
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    model = ms.load_model()
    atoms = ms.build_structure(args.structure)
    batch = ms.atoms_to_batch(atoms, model)
    species = batch["node_attrs"].argmax(-1)
    out = [
        run_layer(int(li), model, species, dev, args.variant)
        for li in args.layers.split(",")
    ]
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
