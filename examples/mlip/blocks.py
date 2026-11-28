"""Plain-PyTorch reference implementations of the two MLIP block contracts.

No e3nn or MACE import: the Clebsch-Gordan (``w3j``) and coupling (``U``)
constants travel with the captured fixtures, so this file and ``check.py``
run on any host with torch.

Contract A (trntensor): equivariant contractions
    A1  ``conv_tp``               per-edge channel-wise ("uvu") tensor product
    A2  ``symmetric_contraction`` per-node product basis, correlation order nu <= 3

Contract B (trnsparse): graph gather / segment sum
    B1  ``gather``                out[e] = h[idx[e]]
    B2  ``segment_sum``           out[i] = sum over e with idx[e] == i of m[e]

The two B ops are each other's backward, so a device pair (gather,
segment_sum) with registered autograd closes under differentiation.

Layouts follow e3nn: a feature vector for irreps ``mul x l`` blocks is the
concatenation over blocks of ``[mul, 2l+1]`` row-major (channel-major).
"""

from __future__ import annotations

import torch

# --------------------------------------------------------------------- B


def gather(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """B1: ``out[e, :] = h[idx[e], :]``. ``h`` [N, F], ``idx`` [E] int64.

    Backward w.r.t. ``h``: ``segment_sum(grad_out, idx, N)``."""
    return h.index_select(0, idx)


def segment_sum(m: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
    """B2: ``out[i, :] = sum_{e : idx[e] == i} m[e, :]``; rows with no edge are 0.

    ``m`` [E, F], ``idx`` [E] int64 (unsorted, repeated), ``num_segments`` N.
    Backward w.r.t. ``m``: ``gather(grad_out, idx)``. This is MACE's
    ``scatter_sum(src, index, dim=0, dim_size=N)``, which calls
    ``zeros(N, F).scatter_add_(0, index.expand_as(src), src)``."""
    out = m.new_zeros((num_segments,) + tuple(m.shape[1:]))
    return out.index_add_(0, idx, m)


def force_assembly(
    edge_grad: torch.Tensor, sender: torch.Tensor, receiver: torch.Tensor, n: int
):
    """Forces from dE/d(edge vector): MACE builds ``vec[e] = pos[receiver[e]] -
    pos[sender[e]] + shift[e]``, so ``F = -(segsum(g, receiver) - segsum(g, sender))``."""
    return segment_sum(edge_grad, sender, n) - segment_sum(edge_grad, receiver, n)


# --------------------------------------------------------------------- A1


def _offsets(irreps):
    out, o = [], 0
    for mul, ell in irreps:
        d = mul * (2 * ell + 1)
        out.append((o, o + d))
        o += d
    return out, o


def conv_tp(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, spec: dict
) -> torch.Tensor:
    """A1: e3nn ``TensorProduct`` with "uvu" instructions and per-edge weights.

    x [E, dim(irreps_in1)]  sender features (gathered), C channels per irrep
    y [E, dim(irreps_in2)]  spherical harmonics, 1 x (0e + 1o + 2e + 3o) = 16
    w [E, weight_numel]     radial-MLP path weights (x cutoff), C per path
    -> [E, dim(irreps_out)]

    For instruction p = (i1, i2, io, alpha) with l1, l2, l3:
        out_io[e, u, k] = alpha * w_p[e, u] * sum_{i,j} C^{l1 l2 l3}_{ijk} x_i1[e, u, i] y_i2[e, j]
    where C is e3nn's real ``wigner_3j(l1, l2, l3)``. Each io has one path here.
    """
    in1, in2, out_irreps = spec["irreps_in1"], spec["irreps_in2"], spec["irreps_out"]
    s1, _ = _offsets(in1)
    s2, _ = _offsets(in2)
    E = x.shape[0]
    outs: list = [None] * len(out_irreps)
    woff = 0
    for i1, i2, io, alpha in spec["instructions"]:
        (mul1, l1), (mul2, l2), (mulo, l3) = in1[i1], in2[i2], out_irreps[io]
        x1 = x[:, s1[i1][0] : s1[i1][1]].reshape(E, mul1, 2 * l1 + 1)
        x2 = y[:, s2[i2][0] : s2[i2][1]].reshape(E, mul2, 2 * l2 + 1)
        wp = w[:, woff : woff + mul1 * mul2].reshape(E, mul1, mul2)
        woff += mul1 * mul2
        C = spec["w3j"][f"{l1},{l2},{l3}"].to(x.device, x.dtype)
        r = alpha * torch.einsum("euv,ijk,eui,evj->euk", wp, C, x1, x2)
        r = r.reshape(E, mulo * (2 * l3 + 1))
        outs[io] = r if outs[io] is None else outs[io] + r
    for io, (mulo, l3) in enumerate(out_irreps):
        if outs[io] is None:
            outs[io] = x.new_zeros(E, mulo * (2 * l3 + 1))
    return torch.cat(outs, dim=1)


# --------------------------------------------------------------------- A2


def _dense_u(u) -> torch.Tensor:
    """U is stored dense (full captures) or as COO {"shape", "indices", "values"}."""
    if torch.is_tensor(u):
        return u
    dense = torch.zeros(u["shape"], dtype=u["values"].dtype)
    dense[tuple(u["indices"].long())] = u["values"]
    return dense


def symmetric_contraction(
    x: torch.Tensor, species: torch.Tensor, spec: dict, chunk: int = 4096
) -> torch.Tensor:
    """A2: MACE ``SymmetricContraction`` (product basis).

    x       [N, C, L]  node features, L = (l_max + 1)^2 = 16 (0e+1o+2e+3o)
    species [N] int64  element index into the weight tables
    -> [N, sum_out C * d_out], blocks in ``spec["contractions"]`` order

    For one output irrep of dimension d (d = 1 is squeezed in U):
        out[n, c, m] = sum_{nu=1..corr} sum_k W_nu[species[n], k, c]
                       * sum_{a_1..a_nu} U_nu[m, a_1, ..., a_nu, k] * prod_j x[n, c, a_j]
    U_nu is a fixed, very sparse coupling tensor (generalised CG); W_nu is
    learned, per element. Rows (n, c) are independent: no reduction over
    nodes or channels, in the forward or in the input gradient.
    """
    N, C, L = x.shape
    blocks = []
    for con in spec["contractions"]:
        d = con["d_out"]
        parts = []
        for n0 in range(0, N, max(1, chunk // C)):
            n1 = min(N, n0 + max(1, chunk // C))
            xb, zb = x[n0:n1], species[n0:n1]
            acc = None
            mono = None
            for nu in range(1, con["correlation"] + 1):
                U = _dense_u(con["U"][str(nu)]).to(x.device, x.dtype)
                if d == 1:
                    U = U.unsqueeze(0)
                K = U.shape[-1]
                mono = (
                    xb
                    if mono is None
                    else (
                        mono.unsqueeze(-1)
                        * xb.reshape(xb.shape[:2] + (1,) * (nu - 1) + (L,))
                    )
                )
                # mono [n, c, L, ..., L] (nu axes) -> [n*c, L^nu]
                q = mono.reshape(-1, L**nu) @ U.reshape(d, L**nu, K).permute(
                    1, 0, 2
                ).reshape(L**nu, d * K)
                q = q.reshape(n1 - n0, C, d, K)
                wz = con["W"][str(nu)].to(x.device, x.dtype)[zb]  # [n, K, C]
                t = torch.einsum("ncdk,nkc->ncd", q, wz)
                acc = t if acc is None else acc + t
            parts.append(acc.reshape(n1 - n0, C * d))
        blocks.append(torch.cat(parts, 0))
    return torch.cat(blocks, dim=1)
