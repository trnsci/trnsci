"""Fusion prototypes for the aten remainder of the trnsci MACE step (trnsci/trnsci#178).

``apply(model, names, ctx)`` patches a device model already set up by
``mace_trnsci_step.install`` and returns an undo. ``ctx`` carries the device,
the host model, the host batch and the ``Graph`` (neighbour-list state).

- ``edge``: everything MACE computes per edge from the edge vectors (the
  spherical harmonics, the Agnesi-transformed Bessel basis with its
  polynomial cutoff, the ZBL pair energy per edge, and both layers' radial
  and density MLPs) runs as one ``torch.compile(backend="neuron")`` region.
  The species-dependent per-edge constants (Agnesi ``r_0``, ZBL screening
  length, ``Z_u Z_v`` and ZBL cutoff radius) depend only on the neighbour
  list, so they are built once with it, like the segment plans; MACE
  recomputes them from integer gathers every step. The ZBL edge -> node sum
  goes to ``trnsparse.segment_sum``.
- ``dense``: every e3nn ``o3.Linear`` on nodes (``linear_up``, ``linear``,
  product ``linear``, readouts) becomes one GEMM against its dense
  equivalent matrix (block-diagonal ``W_l (x) I_{2l+1}``, probed once from
  the module), and each ``skip_tp`` (an e3nn ``FullyConnectedTensorProduct``
  against the one-hot species) one GEMM over the species present.
"""

from __future__ import annotations

import torch


def _probe_linear(mod, d_in):
    """Dense ``M``, ``b`` (fp64) with ``mod(x) == x @ M + b``; ``mod`` is linear in x."""
    mod.to(torch.float64)
    try:
        with torch.no_grad():
            z = mod(torch.zeros(1, d_in, dtype=torch.float64))
            M = mod(torch.eye(d_in, dtype=torch.float64)) - z
    finally:
        mod.to(torch.float32)
    return M, z[0]


def _dense_linear(mod_cpu, dev):
    M, b = _probe_linear(mod_cpu, mod_cpu.irreps_in.dim)
    M = M.to(dev, torch.float32)
    b = b.to(dev, torch.float32) if bool(b.abs().max() > 0) else None

    def fwd(x, M=M, b=b):
        y = x @ M
        return y if b is None else y + b

    return fwd


def _dense_skip(mod_cpu, node_attrs, dev):
    """``skip_tp(x, node_attrs)`` with one-hot ``node_attrs``: ``x @ M_z`` for
    each species z present, masked onto that species' nodes."""
    d_in, n_el = mod_cpu.irreps_in1.dim, mod_cpu.irreps_in2.dim
    sp = node_attrs.argmax(-1)
    present = sorted(int(z) for z in sp.unique())
    mod_cpu.to(torch.float64)
    Ms = []
    try:
        with torch.no_grad():
            eye = torch.eye(d_in, dtype=torch.float64)
            for z in present:
                oh = torch.zeros(d_in, n_el, dtype=torch.float64)
                oh[:, z] = 1.0
                Ms.append(mod_cpu(eye, oh))
    finally:
        mod_cpu.to(torch.float32)
    d_out, S = Ms[0].shape[1], len(present)
    Mcat = torch.cat(Ms, dim=1).to(dev, torch.float32)  # [d_in, S * d_out]
    mask = torch.stack([(sp == z) for z in present], dim=1).to(dev, torch.float32)

    def fwd(x, attrs, Mcat=Mcat, mask=mask):
        y = x @ Mcat
        if S == 1:
            return y
        return (y.view(x.shape[0], S, d_out) * mask.unsqueeze(-1)).sum(1)

    return fwd


def apply_dense(model, ctx):
    from e3nn import o3

    named_cpu = dict(ctx["model_cpu"].named_modules())
    undo = []
    for name, mod in model.named_modules():
        if isinstance(mod, o3.Linear) and name.split(".")[0] in (
            "interactions",
            "products",
            "readouts",
        ):
            mod.forward = _dense_linear(named_cpu[name], ctx["dev"])
            undo.append(lambda mod=mod: delattr(mod, "forward"))
    for i, it in enumerate(model.interactions):
        it.skip_tp.forward = _dense_skip(
            ctx["model_cpu"].interactions[i].skip_tp,
            ctx["batch_host"]["node_attrs"],
            ctx["dev"],
        )
        undo.append(lambda m=it.skip_tp: delattr(m, "forward"))
    return undo


def edge_constants(model, batch_host, dev):
    """Per-edge species constants, once per neighbour list (fp64 on the host)."""
    ei = batch_host["edge_index"]
    Z = model.atomic_numbers.to(torch.int64)[batch_host["node_attrs"].argmax(-1)]
    Zu, Zv = Z[ei[0]].unsqueeze(-1), Z[ei[1]].unsqueeze(-1)
    out = {}
    re = model.radial_embedding
    if hasattr(re, "distance_transform"):
        cr = re.distance_transform.covalent_radii.to(torch.float64)
        out["r0"] = 0.5 * (cr[Zu] + cr[Zv])
    if hasattr(model, "pair_repulsion_fn"):
        zb = model.pair_repulsion_fn
        fu, fv = Zu.to(torch.float64), Zv.to(torch.float64)
        ae = float(zb.a_exp)
        out["a"] = float(zb.a_prefactor) * 0.529 / (fu**ae + fv**ae)
        out["zz"] = 14.3996 * fu * fv
        crz = zb.covalent_radii.to(torch.float64)
        out["rmax"] = crz[Zu] + crz[Zv]
    return {k: v.to(dev, torch.float32) for k, v in out.items()}


def edge_function(model):
    """The per-edge chain as one function of the edge vectors and constants."""
    from mace.modules.radial import PolynomialCutoff

    sh, re = model.spherical_harmonics, model.radial_embedding
    ag = getattr(re, "distance_transform", None)
    zb = getattr(model, "pair_repulsion_fn", None)
    # the modules' own forwards: apply_edge stubs the instances' ``forward``
    mlps = [
        (type(m).forward, m)
        for it in model.interactions
        for m in (it.conv_tp_weights, it.density_fn)
    ]
    sh_fwd = type(sh).forward
    if ag is not None:
        ag_a, ag_q, ag_p = float(ag.a), float(ag.q), float(ag.p)
    if zb is not None:
        zc = [float(c) for c in zb.c]
        zp = float(zb.p)

    def edge_fn(vectors, r0, a, zz, rmax):
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        outs = [sh_fwd(sh, vectors)]
        cutoff = re.cutoff_fn(lengths)
        x = lengths
        if ag is not None:
            t = lengths / r0
            x = 1.0 / (1 + ag_a * torch.pow(t, ag_q) / (1 + torch.pow(t, ag_q - ag_p)))
        ef = re.bessel_fn(x) * cutoff
        outs.append(ef)
        if zb is not None:
            ra = lengths / a
            phi = (
                zc[0] * torch.exp(-3.2 * ra)
                + zc[1] * torch.exp(-0.9423 * ra)
                + zc[2] * torch.exp(-0.4028 * ra)
                + zc[3] * torch.exp(-0.2016 * ra)
            )
            env = PolynomialCutoff.calculate_envelope(lengths, rmax, zp)
            outs.append(0.5 * (zz / lengths * phi) * env)
        outs += [f(m, ef) for f, m in mlps]
        return tuple(outs)

    return edge_fn, zb is not None


def _stub(f):
    """Mark a forward that only returns a tensor of the fused region."""
    f.stash_stub = True
    return f


def apply_edge(model, ctx):
    import trnsparse

    dev, g = ctx["dev"], ctx["g"]
    k = edge_constants(ctx["model_cpu"], ctx["batch_host"], dev)
    dummy = torch.zeros(1, 1, device=dev)
    consts = [k.get(n, dummy) for n in ("r0", "a", "zz", "rmax")]
    edge_fn, has_zbl = edge_function(model)
    backend = ctx.get("compile_backend", "neuron")
    fn = (
        edge_fn
        if backend == "eager-nocompile"
        else torch.compile(edge_fn, backend=backend, dynamic=False)
    )
    sh, re = model.spherical_harmonics, model.radial_embedding
    stash = {}

    def sh_stub(vectors):
        outs = list(fn(vectors, *consts))
        stash["ef"] = outs[1]
        j = 2
        if has_zbl:
            stash["v"], j = outs[2], 3
        stash["tpw"], stash["dens"] = outs[j::2], outs[j + 1 :: 2]
        return outs[0]

    undo = []
    sh.forward = sh_stub
    undo.append(lambda: delattr(sh, "forward"))
    sh_stub.fused_region = True
    re.forward = _stub(lambda *a, **kw: (stash["ef"], None))
    undo.append(lambda: delattr(re, "forward"))
    if has_zbl:
        zb = model.pair_repulsion_fn
        zb.forward = _stub(
            lambda *a, **kw: trnsparse.segment_sum(
                stash["v"], g.receiver, g.N, plan=g.rplan
            ).squeeze(-1)
        )
        undo.append(lambda: delattr(zb, "forward"))
    for li, it in enumerate(model.interactions):
        it.conv_tp_weights.forward = _stub(lambda x, li=li: stash["tpw"][li])
        it.density_fn.forward = _stub(lambda x, li=li: stash["dens"][li])
        undo.append(lambda m=it.conv_tp_weights: delattr(m, "forward"))
        undo.append(lambda m=it.density_fn: delattr(m, "forward"))
    return undo


FUSIONS = {"edge": apply_edge, "dense": apply_dense}


def apply(model, names, ctx):
    undo = []
    for n in names:
        if n not in FUSIONS:
            raise ValueError(f"unknown fusion {n!r}; have {sorted(FUSIONS)}")
        undo += FUSIONS[n](model, ctx)

    def restore():
        for u in reversed(undo):
            u()

    return restore
