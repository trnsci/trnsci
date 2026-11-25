"""The CPU reference for the MLIP step scripts, and the comparison against it.

Every device result is compared against unmodified MACE on the host CPU,
one structure at a time and unpadded. ``cpu_reference`` runs it and records
how long it took (``cpu_ref_s``). ``compare`` refuses to report an error
without a reference: a missing, empty, mis-shaped or non-finite reference
energy or force array raises ``MissingReferenceError`` rather than turning
into an error of 0.

Why this exists (trnsci/trnsci#178): in window 7 the 1728-atom JSONs reported
``forces_rel = 0.0`` next to an empty ``cpu_ref_s``, while their energy error
was non-zero. So those runs had a reference energy but no reference forces,
and the harness wrote a number anyway. The scripts as pushed allowed exactly
that: they computed ``forces_rel`` from whatever ``f_ref`` held and recorded
nothing about whether, or how long, the reference ran, so a skipped or
defaulted force reference was indistinguishable from a perfect match. Now
every per-structure error comes from ``compare``, which needs the reference
forces and the reference's own run time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch


class MissingReferenceError(RuntimeError):
    """A comparison was asked for without a usable CPU reference."""


@dataclass
class Reference:
    """Unmodified MACE on the host CPU for one structure."""

    energy: float
    forces: torch.Tensor  # [N, 3] fp64 on the host
    cpu_ref_s: float  # wall time of the reference step
    threads: int
    stress: torch.Tensor | None = None  # [3, 3], when asked for


def cpu_reference(model_cpu, atoms_list, threads=None, stress=False) -> list:
    """One ``Reference`` per structure: unmodified MACE on the host CPU, each
    structure alone and unpadded (``mace_setup.atoms_to_batch``)."""
    import mace_setup as ms

    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(int(threads))
    try:
        refs = []
        for atoms in atoms_list:
            b = ms.atoms_to_batch(atoms, model_cpu)
            b["positions"] = b["positions"].detach().clone()
            t0 = time.perf_counter()
            out = model_cpu(
                b, compute_force=True, compute_stress=stress, training=False
            )
            dt = time.perf_counter() - t0
            s = out.get("stress") if stress else None
            refs.append(
                Reference(
                    energy=float(out["energy"].detach().sum()),
                    forces=out["forces"].detach().double().clone(),
                    cpu_ref_s=dt,
                    threads=torch.get_num_threads(),
                    stress=None if s is None else s.detach().double()[0].clone(),
                )
            )
        return refs
    finally:
        torch.set_num_threads(old)


def _require(ref, n_atoms: int, what: str) -> None:
    if ref is None:
        raise MissingReferenceError(f"{what}: no CPU reference")
    e = getattr(ref, "energy", None)
    if e is None or not math.isfinite(float(e)):
        raise MissingReferenceError(f"{what}: reference energy missing or not finite")
    f = getattr(ref, "forces", None)
    if not torch.is_tensor(f) or f.numel() == 0:
        raise MissingReferenceError(f"{what}: reference forces missing")
    if tuple(f.shape) != (n_atoms, 3):
        raise MissingReferenceError(
            f"{what}: reference forces are {tuple(f.shape)}, the structure has "
            f"{n_atoms} atoms"
        )
    if not bool(torch.isfinite(f).all()):
        raise MissingReferenceError(f"{what}: reference forces are not finite")
    t = getattr(ref, "cpu_ref_s", None)
    if t is None or not t > 0:
        raise MissingReferenceError(f"{what}: the reference step did not run")
    if float(f.double().norm()) == 0.0:
        raise MissingReferenceError(
            f"{what}: reference forces are all zero (relative error undefined)"
        )


def compare(energy, forces, ref, what: str = "structure") -> dict:
    """Errors of one structure's ``energy`` (eV) and ``forces`` ([N, 3]) against
    its ``Reference``. Raises ``MissingReferenceError`` instead of reporting
    an error without a reference."""
    if forces is None or not torch.is_tensor(forces):
        raise ValueError(f"{what}: no candidate forces to compare")
    n = int(forces.shape[0])
    _require(ref, n, what)
    f = forces.detach().to("cpu").double()
    fr = ref.forces.double()
    if not bool(torch.isfinite(f).all()):
        raise ValueError(f"{what}: candidate forces are not finite")
    e = float(energy)
    return {
        "atoms": n,
        "energy": e,
        "energy_ref_cpu": float(ref.energy),
        "dE_per_atom_meV": 1e3 * abs(e - float(ref.energy)) / n,
        "forces_rel": float((f - fr).norm() / fr.norm()),
        "forces_max_abs_eV_A": float((f - fr).abs().max()),
        "cpu_ref_s": float(ref.cpu_ref_s),
        "cpu_ref_threads": int(ref.threads),
    }


def worst(rows: list) -> dict:
    """The worst per-structure errors of a batch."""
    if not rows:
        raise MissingReferenceError("no per-structure comparisons")
    return {
        "dE_per_atom_meV": max(r["dE_per_atom_meV"] for r in rows),
        "forces_rel": max(r["forces_rel"] for r in rows),
        "forces_max_abs_eV_A": max(r["forces_max_abs_eV_A"] for r in rows),
        "cpu_ref_s": sum(r["cpu_ref_s"] for r in rows),
    }
