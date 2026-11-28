"""Batched MACE graphs: block-diagonal batches, shape buckets and padding masks.

B structures become one graph, as MACE's own data loader builds them: atoms
and edges concatenated, each structure's edge indices offset by the atoms
before it, ``batch`` naming the structure of every atom. One energy + forces
step then evaluates all B structures; the per-structure energies are
``energy[:B]`` and the forces the slices at ``BatchInfo.atom_start``
(``mace_trnsci_step.Step.split``).

**Shape buckets.** Device kernels and the aten remainder compile per shape,
so the totals are rounded up to a bucket: ``bucket(n)`` is the next multiple
of ``2^(floor(log2 n) - shift)`` (at least ``minimum``), so the padding is at
most ``2^-shift`` of n (6.25% at the default shift 4). The key of a batch is
``(atoms, edges)`` after padding; a new geometry or a new batch of the same
sizes lands in the same bucket and reuses the compiled shapes.

**Padding.** The padding atoms form one extra structure (index B), which
MACE evaluates and whose energy is dropped: no real atom is ever its
neighbour, so real atoms see no padding. There is always at least one
padding atom, so padding edges always have somewhere to point. The padding
edges are self-edges spread round-robin over the padding atoms, shifted by
the padding structure's third cell vector (at least ``3 r_max`` long, past
every cutoff, so their messages are exactly zero, and never zero-length, so
nothing is NaN; ``unit_shifts`` match, so the stress path rebuilds the same
shift). The ``[E]`` bool mask ``edge_padding`` marks them; it is what
``trnsparse.segment_plan(idx, N, padding=mask)`` takes, so the segment sums
leave them out entirely (no slots, the real rows' summation order unchanged),
instead of piling them onto node 0.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import torch


def bucket(n: int, shift: int = 4, minimum: int = 1) -> int:
    """``n`` rounded up to a multiple of ``max(minimum, 2^(floor(log2 n) - shift))``."""
    if n <= 0:
        return max(minimum, 1)
    q = max(minimum, 1 << max(0, math.floor(math.log2(n)) - shift))
    return -(-n // q) * q


@dataclass
class BatchInfo:
    """Layout of one padded batch. Index ``i < B`` is a real structure; the
    padding atoms all belong to structure ``B``. ``isolated`` is set on the
    batches ``isolate`` makes: only that structure is real."""

    num_structures: int
    atoms: list = field(default_factory=list)  # real atoms per structure
    edges: list = field(default_factory=list)  # real edges per structure
    atom_start: list = field(default_factory=list)  # first atom of each structure
    edge_start: list = field(default_factory=list)  # first edge of each structure
    n_real: int = 0
    e_real: int = 0
    n_pad: int = 0  # total atoms after padding
    e_pad: int = 0  # total edges after padding
    edge_padding: torch.Tensor | None = None  # [e_pad] bool, True at padding edges
    node_padding: torch.Tensor | None = None  # [n_pad] bool, True at padding atoms
    isolated: int | None = None

    @property
    def key(self) -> tuple:
        """The compile bucket: (atoms, edges) after padding."""
        return (self.n_pad, self.e_pad)

    def as_dict(self) -> dict:
        return {
            "structures": self.num_structures,
            "atoms": self.atoms,
            "edges": self.edges,
            "atoms_total": self.n_real,
            "edges_total": self.e_real,
            "atoms_padded": self.n_pad,
            "edges_padded": self.e_pad,
            "bucket": list(self.key),
        }


def atoms_to_data(atoms, model):
    """One ``AtomicData`` (matscipy neighbour list at r_max), as ``MACECalculator``."""
    from mace import data as mace_data
    from mace.tools import utils

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = list(getattr(model, "heads", ["default"]))
    keyspec = mace_data.KeySpecification(info_keys={}, arrays_keys={})
    config = mace_data.config_from_atoms(
        atoms, key_specification=keyspec, head_name=heads[0]
    )
    return mace_data.AtomicData.from_config(
        config, z_table=z_table, cutoff=float(model.r_max), heads=heads
    )


def padding_atoms(p: int, model):
    """``p`` atoms of the model's first element on a cubic grid ``3 r_max`` apart
    (no neighbours within r_max), non-periodic, in a cubic cell of side at
    least ``3 r_max``. Its cell vectors give the padding edges' shift."""
    from ase import Atoms

    r = 3.0 * float(model.r_max)
    m = max(1, math.ceil(p ** (1.0 / 3.0) - 1e-9))
    pos = [(r * (i % m), r * ((i // m) % m), r * (i // (m * m))) for i in range(p)]
    z = int(model.atomic_numbers[0])
    return Atoms(numbers=[z] * p, positions=pos, cell=[r, r, r], pbc=False)


def collate(datas: list) -> dict:
    """MACE's own block-diagonal collation of a list of ``AtomicData``."""
    from mace.tools import torch_geometric

    loader = torch_geometric.dataloader.DataLoader(
        datas, batch_size=len(datas), shuffle=False
    )
    return next(iter(loader)).to_dict()


def _up(n: int, a: int) -> int:
    return -(-n // a) * a


def make_batch(
    atoms_list: list,
    model,
    *,
    shift: int = 4,
    atom_min: int = 8,
    edge_min: int = 128,
    atom_multiple: int = 0,
    edge_multiple: int = 0,
    pad: bool = True,
) -> tuple[dict, BatchInfo]:
    """B structures as one padded block-diagonal batch.

    Layout: the B structures' atoms back to back, then the padding atoms; the
    B structures' edges back to back (indices offset), then the padding edges.
    The totals are rounded with ``bucket(., shift, minimum)``; a nonzero
    ``atom_multiple`` / ``edge_multiple`` sets a fixed multiple instead.
    ``pad=False`` gives MACE's plain collation (no padding structure).
    Returns the host batch dict (in the model's dtype) and its ``BatchInfo``.
    """
    dtype = next(model.parameters()).dtype
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # AtomicData takes positions in this dtype
    try:
        datas = [atoms_to_data(a, model) for a in atoms_list]
    finally:
        torch.set_default_dtype(old)
    B = len(datas)
    atoms = [int(d.num_nodes) for d in datas]
    edges = [int(d.edge_index.shape[1]) for d in datas]
    n_real, e_real = sum(atoms), sum(edges)
    a_start = [sum(atoms[:i]) for i in range(B)]
    e_start = [sum(edges[:i]) for i in range(B)]

    def up(n, mult, mn):
        return _up(n, mult) if mult else bucket(n, shift, mn)

    if pad:
        n_pad = up(n_real + 1, atom_multiple, atom_min)  # >= 1 padding atom
        e_pad = up(max(e_real, 1), edge_multiple, edge_min)
        p = n_pad - n_real
        extra = atoms_to_data(padding_atoms(p, model), model)
        if extra.edge_index.shape[1]:
            raise RuntimeError("padding atoms must have no neighbours")
        b = collate([*datas, extra])  # MACE's collation: offsets, batch, ptr
    else:
        n_pad, e_pad, p = n_real, e_real, 0
        b = collate(datas)

    k = e_pad - e_real
    if pad:
        c = b["cell"].view(-1, 3, 3)[B]
        if not float(c[2].norm()) >= 3.0 * float(model.r_max):
            raise RuntimeError(f"padding cell {c.tolist()} is too small for r_max")
    if k:
        own = n_real + torch.arange(k, dtype=torch.int64) % p
        b["edge_index"], b["unit_shifts"], b["shifts"] = _padding_edges(
            b, B, own, b["edge_index"], b["unit_shifts"], b["shifts"]
        )
    for kk, v in b.items():
        if torch.is_tensor(v) and v.is_floating_point():
            b[kk] = v.to(dtype)
    edge_padding = torch.zeros(e_pad, dtype=torch.bool)
    edge_padding[e_real:] = True
    node_padding = torch.zeros(n_pad, dtype=torch.bool)
    node_padding[n_real:] = True
    info = BatchInfo(
        num_structures=B,
        atoms=atoms,
        edges=edges,
        atom_start=a_start,
        edge_start=e_start,
        n_real=n_real,
        e_real=e_real,
        n_pad=n_pad,
        e_pad=e_pad,
        edge_padding=edge_padding,
        node_padding=node_padding,
    )
    check_batch(b, info)
    return b, info


def _padding_edges(b, B, own, ei, us, sh):
    """Append self-edges on the padding atoms ``own``, shifted by the padding
    structure's third cell vector (consistent with ``unit_shifts @ cell``,
    which the stress path recomputes)."""
    k = own.shape[0]
    pus = torch.zeros(k, 3, dtype=us.dtype)
    pus[:, 2] = 1.0
    cell = b["cell"].view(-1, 3, 3)[B]
    psh = (pus @ cell.to(pus.dtype)).to(sh.dtype)
    return (
        torch.cat([ei, torch.stack([own, own])], 1),
        torch.cat([us, pus], 0),
        torch.cat([sh, psh], 0),
    )


def isolate(b: dict, info: BatchInfo, i: int) -> tuple[dict, BatchInfo]:
    """A copy of the batch with every structure but ``i`` turned into padding:
    their atoms join structure B and their edges become padding self-edges.
    Structure ``i`` keeps its slot, and every tensor keeps its shape, so a
    step on this batch is structure ``i`` evaluated alone at exactly the
    shapes and offsets of the full batch. Splitting the full batch must
    reproduce it bitwise: nothing of the other structures may reach ``i``."""
    B = info.num_structures
    if info.n_pad == info.n_real:
        raise ValueError("isolate needs a padded batch (make_batch(pad=True))")
    a0, n = info.atom_start[i], info.atoms[i]
    e0, e = info.edge_start[i], info.edges[i]
    keep_n = torch.zeros(info.n_pad, dtype=torch.bool)
    keep_n[a0 : a0 + n] = True
    keep_e = torch.zeros(info.e_pad, dtype=torch.bool)
    keep_e[e0 : e0 + e] = True
    c = dict(b)
    c["batch"] = torch.where(keep_n, b["batch"], torch.full_like(b["batch"], B))
    ei, us, sh = b["edge_index"], b["unit_shifts"], b["shifts"]
    drop = ~keep_e
    r = ei[1, drop]  # former or padding atoms, all in structure B now
    pei, pus, psh = _padding_edges(c, B, r, ei[:, :0], us[:0], sh[:0])
    ei, us, sh = ei.clone(), us.clone(), sh.clone()
    ei[:, drop], us[drop], sh[drop] = pei, pus, psh
    c["edge_index"], c["unit_shifts"], c["shifts"] = ei, us, sh
    one = dataclasses.replace(info, edge_padding=drop, node_padding=~keep_n, isolated=i)
    check_batch(c, one)
    return c, one


def check_batch(b: dict, info: BatchInfo) -> None:
    """Invariants of a padded batch: block-diagonal edges, padding isolated."""
    ei = b["edge_index"]
    if tuple(ei.shape) != (2, info.e_pad) or b["positions"].shape[0] != info.n_pad:
        raise ValueError("batch shape does not match its BatchInfo")
    gid = b["batch"]
    if not torch.equal(gid[ei[0]], gid[ei[1]]):
        raise ValueError("an edge crosses two structures (not block-diagonal)")
    B = info.num_structures
    if not torch.equal(gid[ei[0]] == B, info.edge_padding):
        raise ValueError("padding edges must be exactly the edges on padding atoms")
    if not torch.equal(gid == B, info.node_padding):
        raise ValueError("padding atoms must be exactly structure B")
    if bool(info.node_padding.all()):
        raise ValueError("a batch needs at least one real atom")
    pad_atoms = torch.nonzero(info.node_padding).flatten()
    if info.edge_padding.any():
        pe = ei[:, info.edge_padding]
        if not torch.equal(pe[0], pe[1]):
            raise ValueError("padding edges must be self-edges")
        if not bool(torch.isin(pe[0], pad_atoms).all()):
            raise ValueError("padding edges must sit on padding atoms")
    real = range(B) if info.isolated is None else [info.isolated]
    for i in real:
        a0, e0 = info.atom_start[i], info.edge_start[i]
        if not bool((gid[a0 : a0 + info.atoms[i]] == i).all()):
            raise ValueError(f"structure {i}: atoms are not at their offsets")
        if not bool((gid[ei[1, e0 : e0 + info.edges[i]]] == i).all()):
            raise ValueError(f"structure {i}: edges are not at their offsets")
