"""The aten remainder of the trnsci MACE step on fused library ops (trnsci/trnsci#178, item 1).

``install_fused(model, g, names, ctx)`` patches a device model already set up
by ``mace_trnsci_step.install`` and returns an undo. The fusions are the
maintained counterparts of the prototypes in ``fusions.py``:

- ``sh``: the edge attributes as ``trntensor.spherical_harmonics`` (e3nn's
  real basis, ``l ≤ 3``, component normalisation; analytic backward).
- ``edge``: the radial basis (Agnesi transform, Bessel, polynomial cutoff) and
  every radial MLP that reads it, i.e. both layers' ``conv_tp_weights`` and
  edge densities ``tanh(density_fn(·)²)``, as one ``trntensor.edge_chain``.
  The Agnesi species constant ``r0`` is built once per neighbour list.
- ``zbl``: the ZBL pair repulsion per edge as one fused op with an analytic
  backward (``zbl_edge_energy`` below; its species constants are built once
  per neighbour list), then the receiver ``trnsparse.segment_sum``.
- ``linear``: every e3nn ``o3.Linear`` on nodes (``linear_up``, ``linear``,
  product ``linear``, readouts) as ``trntensor.equivariant_linear``, one
  matmul per irrep type.
- ``skip``: each one-hot ``skip_tp`` as a per-species
  ``trntensor.equivariant_linear``.

Every fused op is a ``torch.autograd.Function`` with a registered analytic
backward, so forces stay ``-dE/dpositions`` through autograd. On the device
the trntensor ops run their NKI kernels under ``set_backend("nki")``
(spherical harmonics and the edge chain), and the PyTorch bodies of all of
them run as one compiled graph per direction under
``trntensor.set_region_backend("neuron")`` (``--region-backend``).
"""

from __future__ import annotations

import torch

FUSE_ALL = ("sh", "edge", "zbl", "linear", "skip")

# ZBL screening-function exponents (MACE ZBLBasis): φ(s) = Σ c_i exp(−k_i s).
ZBL_K = (3.2, 0.9423, 0.4028, 0.2016)


# ─────────────────────────────────────────────────────── ZBL per-edge energy


def zbl_edge_constants(model, batch_host, dev, dtype=torch.float32) -> dict:
    """Per-edge ZBL constants, once per neighbour list (fp64 on the host).

    ``a [E, 1]`` screening length, ``zz [E, 1] = 14.3996 Z_u Z_v`` and
    ``rmax [E, 1]`` (sum of covalent radii), exactly as MACE's ``ZBLBasis``
    builds them on every call."""
    zb = model.pair_repulsion_fn
    ei = batch_host["edge_index"]
    Z = model.atomic_numbers.to(torch.int64)[batch_host["node_attrs"].argmax(-1)]
    Zu, Zv = Z[ei[0]].unsqueeze(-1), Z[ei[1]].unsqueeze(-1)
    fu, fv = Zu.to(torch.float64), Zv.to(torch.float64)
    ae = float(zb.a_exp)
    crz = zb.covalent_radii.to(torch.float64)
    k = {
        "a": float(zb.a_prefactor) * 0.529 / (fu**ae + fv**ae),
        "zz": 14.3996 * fu * fv,
        "rmax": crz[Zu] + crz[Zv],
    }
    k = {n: v.to(dev, dtype) for n, v in k.items()}
    k["c"] = tuple(float(c) for c in zb.c)
    k["p"] = float(zb.p)
    return k


def _envelope(r, rmax, p):
    u = r / rmax
    A, B, C = (p + 1.0) * (p + 2.0) / 2.0, p * (p + 2.0), p * (p + 1.0) / 2.0
    mask = r < rmax
    env = (
        1.0 - A * torch.pow(u, p) + B * torch.pow(u, p + 1) - C * torch.pow(u, p + 2)
    ) * mask
    denv = (
        (
            -A * p * torch.pow(u, p - 1)
            + B * (p + 1) * torch.pow(u, p)
            - C * (p + 2) * torch.pow(u, p + 1)
        )
        / rmax
        * mask
    )
    return env, denv


def _zbl_forward(r, a, zz, rmax, c, p):
    s = r / a
    phi = c[0] * torch.exp(-ZBL_K[0] * s)
    for ci, ki in zip(c[1:], ZBL_K[1:], strict=True):
        phi = phi + ci * torch.exp(-ki * s)
    env, _ = _envelope(r, rmax, p)
    return 0.5 * (zz / r * phi) * env


def _zbl_backward(g, r, a, zz, rmax, c, p):
    s = r / a
    phi = torch.zeros_like(r)
    dphi = torch.zeros_like(r)
    for ci, ki in zip(c, ZBL_K, strict=True):
        e = ci * torch.exp(-ki * s)
        phi = phi + e
        dphi = dphi - ki * e
    env, denv = _envelope(r, rmax, p)
    # v = ½ zz φ(r/a) env(r) / r
    dv = 0.5 * zz * ((dphi / a) * env / r + phi * denv / r - phi * env / (r * r))
    return g * dv


def _region(fn):
    """trntensor's compiled region (one graph under ``set_region_backend``), if present."""
    try:
        from trntensor import compiled_region
    except ImportError:  # trntensor without the fused ops: eager
        return fn
    return compiled_region(fn)


class _ZBLEdge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, a, zz, rmax, c, p):
        ctx.save_for_backward(r, a, zz, rmax)
        ctx.args = (c, p)
        return _region(_zbl_forward)(r, a, zz, rmax, c, p)

    @staticmethod
    def backward(ctx, g):
        r, a, zz, rmax = ctx.saved_tensors
        c, p = ctx.args
        dr = _region(_zbl_backward)(g.contiguous(), r, a, zz, rmax, c, p)
        return dr, None, None, None, None, None


def zbl_edge_energy(lengths: torch.Tensor, k: dict) -> torch.Tensor:
    """MACE's ZBL pair energy per edge, ``v_edges [E, 1]``, as one fused op.

    ``v = ½ · 14.3996 Z_u Z_v / r · φ(r / a) · envelope(r; R_u + R_v, p)``,
    differentiable in ``lengths [E, 1]`` (analytic backward); ``k`` from
    :func:`zbl_edge_constants`. MACE sums it onto the receivers.
    """
    return _ZBLEdge.apply(lengths, k["a"], k["zz"], k["rmax"], k["c"], k["p"])


# ─────────────────────────────────────────────────────────────── installing


def _species(batch_host, dev):
    return batch_host["node_attrs"].argmax(-1).to(dev)


def install_fused(model, g, names, ctx, blk=None):
    """Patch ``model`` (already on the trnsci blocks) to the fused ops in ``names``.

    ``ctx``: ``dev``, ``model_cpu``, ``batch_host``; the specs and per-edge
    constants are cached in it (``ctx["_cache"]``), so a ``ctx`` kept with one
    neighbour list (as ``mace_trnsci_step.Step`` does) builds them once.
    ``blk(family, name, fn, *tensors)``, when given, wraps each fused call
    (``mace_trnsci_step``'s share recorder). Returns an undo.
    """
    import trnsparse
    import trntensor
    from e3nn import o3

    names = list(FUSE_ALL) if "all" in names else list(names)
    bad = sorted(set(names) - set(FUSE_ALL))
    if bad:
        raise ValueError(f"unknown fusions {bad}; have {list(FUSE_ALL)} or 'all'")
    dev, mcpu, bh = ctx["dev"], ctx["model_cpu"], ctx["batch_host"]

    cache = ctx.setdefault("_cache", {})

    def cached(key, build):
        if key not in cache:
            cache[key] = build()
        return cache[key]

    def call(name, fn, *ts):
        return blk("F", name, fn, *ts) if blk is not None else fn(*ts)

    undo = []

    def patch(mod, fwd):
        mod.forward = fwd
        undo.append(lambda m=mod: delattr(m, "forward"))

    if "sh" in names:
        sh = model.spherical_harmonics

        def sh_fwd(vectors):
            return call("spherical_harmonics", trntensor.spherical_harmonics, vectors)

        patch(sh, sh_fwd)

    if "edge" in names:
        spec = cached("edge_spec", lambda: trntensor.edge_chain_spec(module=mcpu))

        def build_r0():
            if spec.agnesi is None:
                return None
            ei = bh["edge_index"]
            Z = mcpu.atomic_numbers.to(torch.int64)[bh["node_attrs"].argmax(-1)]
            return spec.agnesi_r0(Z[ei[0]], Z[ei[1]]).to(dev, torch.float32)

        r0 = cached("agnesi_r0", build_r0)
        stash = {}
        heads = [h.name for h in spec.heads]

        def chain(lengths):
            if r0 is None:
                return torch.cat(trntensor.edge_chain(lengths, spec), 1)
            return torch.cat(trntensor.edge_chain(lengths, spec, r0), 1)

        def re_fwd(lengths, node_attrs=None, edge_index=None, atomic_numbers=None):
            if blk is None:
                outs = trntensor.edge_chain(lengths, spec, r0)
            else:  # the recorder wants one output tensor
                outs = torch.split(
                    call("edge_chain", chain, lengths), list(spec.dims_out), 1
                )
            stash.update(zip(heads, outs, strict=True))
            return (
                lengths,
                None,
            )  # edge_feats are not read: the interactions use the stash

        patch(model.radial_embedding, re_fwd)
        for li, it in enumerate(model.interactions):
            if f"tp_w{li}" not in heads:
                raise NotImplementedError(f"interaction {li}: no conv_tp_weights head")
            dens = f"density{li}" if f"density{li}" in heads else None
            it._trnsci_edge = lambda li=li, dens=dens: (
                stash[f"tp_w{li}"],
                stash[dens] if dens else None,
            )
            undo.append(lambda it=it: delattr(it, "_trnsci_edge"))

    if "zbl" in names and hasattr(model, "pair_repulsion_fn"):
        k = cached("zbl", lambda: zbl_edge_constants(mcpu, bh, dev))

        def zbl_fwd(x, node_attrs, edge_index, atomic_numbers):
            v = call("zbl_edge_energy", lambda r: zbl_edge_energy(r, k), x)
            return trnsparse.segment_sum(v, g.receiver, g.N, plan=g.rplan).squeeze(-1)

        patch(model.pair_repulsion_fn, zbl_fwd)

    if "linear" in names:
        named_cpu = dict(mcpu.named_modules())
        for name, mod in model.named_modules():
            if isinstance(mod, o3.Linear) and name.split(".")[0] in (
                "interactions",
                "products",
                "readouts",
            ):
                lspec = cached(
                    ("linear", name),
                    lambda name=name: trntensor.equivariant_linear_spec(
                        named_cpu[name]
                    ),
                )

                def lin(x, lspec=lspec, name=name):
                    return call(
                        f"linear {name}",
                        lambda t: trntensor.equivariant_linear(t, lspec),
                        x,
                    )

                patch(mod, lin)

    if "skip" in names:
        sp = cached("species", lambda: _species(bh, dev))
        elems = sorted(int(z) for z in bh["node_attrs"].argmax(-1).unique())
        for li, it in enumerate(model.interactions):
            sspec = cached(
                ("skip", li, tuple(elems)),
                lambda li=li: trntensor.equivariant_linear_spec(
                    mcpu.interactions[li].skip_tp, elements=elems
                ),
            )

            def skip(x, attrs, sspec=sspec, li=li):
                return call(
                    f"skip_tp L{li}",
                    lambda t: trntensor.equivariant_linear(t, sspec, sp),
                    x,
                )

            patch(it.skip_tp, skip)

    def restore():
        for u in reversed(undo):
            u()

    return restore
