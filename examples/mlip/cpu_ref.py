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

**Reference cache (``MLIP_REF_CACHE=<dir>``, opt-in).** At 1728 atoms the
reference takes ~39 s per structure on 2 threads, so a batch of 32 needs
~21 min of CPU before the device step starts. With the variable set, each
``Reference`` is stored under a key built from the exact inputs (atomic
numbers, positions, cell, pbc), the model parameters, the dtype, the thread
count and ``stress``. A later run with the same key loads it instead of
recomputing, so the references can be filled in a separate CPU-only process
(``python cpu_ref.py --structure tic_bulk_6 --seeds 0-63``). A cached
reference keeps the ``cpu_ref_s`` it was measured with, and ``compare``
reports ``cpu_ref_cached`` so a reader can tell it was not measured in the
same process.
"""

from __future__ import annotations

import hashlib
import math
import os
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
    cached: bool = False  # loaded from MLIP_REF_CACHE, not measured in this process


def _model_fingerprint(model_cpu) -> str:
    fp = getattr(model_cpu, "_mlip_ref_fingerprint", None)
    if fp is None:
        h = hashlib.sha256()
        for name, p in sorted(model_cpu.state_dict().items()):
            h.update(name.encode())
            h.update(str(p.dtype).encode())
            h.update(
                p.detach()
                .cpu()
                .contiguous()
                .view(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
        fp = h.hexdigest()
        model_cpu._mlip_ref_fingerprint = fp
    return fp


def _cache_key(model_fp: str, atoms, threads: int, stress: bool) -> str:
    import numpy as np

    h = hashlib.sha256(model_fp.encode())
    h.update(np.ascontiguousarray(atoms.get_atomic_numbers(), dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(atoms.get_positions(), dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(atoms.cell.array, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(atoms.pbc, dtype=np.int8).tobytes())
    h.update(
        f"threads={threads};stress={bool(stress)};torch={torch.__version__}".encode()
    )
    return h.hexdigest()


def _cache_load(path: str):
    if not os.path.exists(path):
        return None
    d = torch.load(path, map_location="cpu", weights_only=True)
    return Reference(
        energy=float(d["energy"]),
        forces=d["forces"],
        cpu_ref_s=float(d["cpu_ref_s"]),
        threads=int(d["threads"]),
        stress=d.get("stress"),
        cached=True,
    )


def _cache_store(path: str, ref: Reference) -> None:
    d = {
        "energy": ref.energy,
        "forces": ref.forces,
        "cpu_ref_s": ref.cpu_ref_s,
        "threads": ref.threads,
    }
    if ref.stress is not None:
        d["stress"] = ref.stress
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)


def cpu_reference(model_cpu, atoms_list, threads=None, stress=False) -> list:
    """One ``Reference`` per structure: unmodified MACE on the host CPU, each
    structure alone and unpadded (``mace_setup.atoms_to_batch``)."""
    import mace_setup as ms

    old = torch.get_num_threads()
    if threads:
        torch.set_num_threads(int(threads))
    cache = os.environ.get("MLIP_REF_CACHE") or None
    try:
        refs = []
        for atoms in atoms_list:
            path = None
            if cache:
                os.makedirs(cache, exist_ok=True)
                key = _cache_key(
                    _model_fingerprint(model_cpu),
                    atoms,
                    torch.get_num_threads(),
                    stress,
                )
                path = os.path.join(cache, key + ".pt")
                hit = _cache_load(path)
                if hit is not None:
                    refs.append(hit)
                    continue
            b = ms.atoms_to_batch(atoms, model_cpu)
            b["positions"] = b["positions"].detach().clone()
            t0 = time.perf_counter()
            out = model_cpu(
                b, compute_force=True, compute_stress=stress, training=False
            )
            dt = time.perf_counter() - t0
            s = out.get("stress") if stress else None
            ref = Reference(
                energy=float(out["energy"].detach().sum()),
                forces=out["forces"].detach().double().clone(),
                cpu_ref_s=dt,
                threads=torch.get_num_threads(),
                stress=None if s is None else s.detach().double()[0].clone(),
            )
            if path is not None:
                _cache_store(path, ref)
            refs.append(ref)
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
        "cpu_ref_cached": bool(getattr(ref, "cached", False)),
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


def _seeds(spec: str) -> list:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None) -> None:
    """Fill ``MLIP_REF_CACHE`` (CPU only): one reference per structure and seed."""
    import argparse

    import mace_setup as ms

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--structure", required=True)
    ap.add_argument("--seeds", required=True, help="e.g. 0-63 or 0,1,5")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args(argv)
    if not os.environ.get("MLIP_REF_CACHE"):
        raise SystemExit("set MLIP_REF_CACHE=<dir>")
    model = ms.load_model(args.checkpoint)
    for s in _seeds(args.seeds):
        (r,) = cpu_reference(
            model, [ms.build_structure(args.structure, s)], args.threads, args.stress
        )
        print(
            f"{args.structure} seed {s}: cpu_ref_s {r.cpu_ref_s:.2f} "
            f"threads {r.threads} cached {r.cached}",
            flush=True,
        )


if __name__ == "__main__":
    main()
