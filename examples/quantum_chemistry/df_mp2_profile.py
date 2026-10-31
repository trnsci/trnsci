"""
End-to-end profile of the DF-MP2 integration example (df_mp2_synthetic.py).

Runs the same synthetic pipeline across a ladder of problem sizes and reports,
per stage: wall time (device synchronized), which library entry point ran and
which route its ``auto`` dispatch picked (NKI kernel or aten), and the error of
the stage output against a float64 host reference.

Modes (``--modes``):

    cpu        every library on its pytorch backend, host tensors: the all-CPU
               baseline (pin it to the same host threads as the device runs)
    host-auto  ``auto`` backends with host tensors: what the example does on a
               Neuron box without ``--device``
    neuron     ``auto`` backends, inputs copied once to the TorchNeuron device

Pipelines (``--pipelines``):

    example    the example's own stage functions: trnblas gemm + batched_gemm
               half-transform, trnsolver.cholesky + trnblas.trsm metric,
               trntensor.mp2_energy
    fused      trntensor.ao_to_mo_transform, trnblas.metric_inv_half + one
               trnblas.gemm, trntensor.mp2_energy
    packed     as fused, with the integrals packed to their mu>=nu lower
               triangle (npair, naux) on the host before timing, the layout
               drivers store; ao_to_mo_transform takes it as is (trntensor#55)

Routes are recorded while each stage runs (see ``RouteLog``), so a host input
that a library uploads and computes on the device shows as ``upload+aten``
(trnblas) or ``upload -> ...`` (trntensor), not as ``aten``.

The inputs are random (see df_mp2_synthetic.py); energies are not physically
meaningful, but their agreement with float64 is.

Run (Trn2, one logical core, host threads pinned)::

    NEURON_RT_VISIBLE_CORES=2 OMP_NUM_THREADS=2 taskset -c 4,10 \\
        python examples/quantum_chemistry/df_mp2_profile.py \\
        --sizes s,m,l,xl --modes cpu,host-auto,neuron \\
        --pipelines example,fused,packed --json df_mp2_profile.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import df_mp2_synthetic as ex
import trnblas
import trnsolver
import trntensor

# name -> (n_ao, n_aux, n_occ). n_aux = 3 n_ao, n_occ = n_ao / 8: the
# proportions of a small-molecule cc-pVDZ / cc-pVDZ-RI calculation. The
# ladder spans trntensor's `auto` crossovers (ao_to_mo_transform from
# n_occ·n_ao²·n_aux ≥ 5e8, mp2_energy from 3 GFLOP per row launch).
LADDER = {
    "demo": (32, 64, 8),  # df_mp2_synthetic.py --demo defaults
    "xs": (32, 96, 4),
    "s": (64, 192, 8),
    "m": (128, 384, 16),
    "l": (256, 768, 32),
    "xl": (384, 1152, 48),
    "xxl": (512, 1536, 64),
}

LIBS = (trnblas, trnsolver, trntensor)


# ---------------------------------------------------------------- reference


def reference_fp64(eri, J, C_occ, C_vir, eps_occ, eps_vir):
    """Float64 host reference for every stage output (plain torch, LAPACK)."""
    eri, J, C_occ, C_vir, eps_occ, eps_vir = (
        t.double() for t in (eri, J, C_occ, C_vir, eps_occ, eps_vir)
    )
    ia_P = torch.einsum("na,inP->iaP", C_vir, torch.einsum("mi,mnP->inP", C_occ, eri))
    L = torch.linalg.cholesky(J)
    X = torch.linalg.solve_triangular(
        L.T, torch.eye(J.shape[0], dtype=J.dtype), upper=True
    )
    B = ia_P @ X
    ev = eps_vir[:, None] + eps_vir[None, :]
    e = 0.0
    for i in range(B.shape[0]):
        T = torch.einsum("ap,jbp->jab", B[i], B)
        denom = (eps_occ[i] + eps_occ)[:, None, None] - ev
        e += float((T * (2 * T - T.transpose(1, 2)) / denom).sum())
    return ia_P, B, e


def rel_err(x: torch.Tensor, ref: torch.Tensor) -> float:
    x = x.detach().to("cpu", torch.float64)
    return float(torch.linalg.norm(x - ref) / torch.linalg.norm(ref))


# ------------------------------------------------------------------ routes
#
# Routes are recorded, not re-derived: `RouteLog` wraps each library's routing
# helpers for the duration of a run and logs what they return for the operands
# the stage actually passed. Re-asking the helpers afterwards on the stage's
# device tensors misses decisions taken on the way (trntensor's host-input
# offload, trntensor#56; trnblas's host-vs-upload choice, trnblas#65) and the
# intermediates' real residency.
#
# Labels:
#   host          torch on the host CPU
#   aten          aten on the `neuron` device, operands already there
#   upload+aten   host operands uploaded, aten on the device, result downloaded
#   nki / trn2 / rows   an NKI kernel (trntensor: Trn2 AO->MO / MP2 row kernel)
#   upload -> X   trntensor uploaded host inputs, then the device route X ran
#   unpack -> X   packed integrals unpacked (index_select) for route X


def _residency(*tensors) -> str:
    types = {t.device.type for t in tensors}
    return "host" if types == {"cpu"} else "device" if "cpu" not in types else "mixed"


def _blas_label(route: str, *tensors) -> str:
    where = _residency(*tensors)
    if route == "aten" and where == "host":
        return "upload+aten"
    if route == "nki" and where == "host":
        return "upload+nki"
    if route == "torch":
        return "host" if where == "host" else "aten"
    return route


class RouteLog:
    """Context manager that records the routing decisions of trnblas and
    trntensor (and the residency of trnsolver.cholesky) per pipeline stage."""

    def __init__(self):
        self.stage = None
        self.events: dict[str, list[str]] = {}
        self._saved = []

    def _add(self, label: str) -> None:
        if self.stage is None:
            return
        ev = self.events.setdefault(self.stage, [])
        if not ev or ev[-1] != label:
            ev.append(label)

    def _wrap(self, mod, name, make_label):
        orig = getattr(mod, name)

        def wrapper(*args, **kwargs):
            out = orig(*args, **kwargs)
            label = make_label(out, *args, **kwargs)
            if label:
                self._add(label)
            return out

        self._saved.append((mod, name, orig))
        setattr(mod, name, wrapper)

    def _wrap_entry(self, mod, name, label):
        """Log `label(*args)` for a call that consulted no routing helper
        (e.g. trnblas.gemm without NKI goes straight to torch.matmul)."""
        orig = getattr(mod, name)

        def wrapper(*args, **kwargs):
            before = sum(map(len, self.events.values()))
            out = orig(*args, **kwargs)
            if sum(map(len, self.events.values())) == before:
                self._add(label(*args))
            return out

        self._saved.append((mod, name, orig))
        setattr(mod, name, wrapper)

    def __enter__(self):
        import trnsolver.factor as sf
        import trntensor.quantum as tq
        from trnblas.nki import dispatch as bd
        from trntensor.nki import dispatch as td

        def on(op, *ts):
            return f"{op}=" + ("host" if _residency(*ts) == "host" else "aten")

        def gemm(r, *ts, op="gemm", **_):
            return f"{op}={_blas_label(r, *ts)}"

        def trsm(r, A, B):
            blocked = " blocked" if r in ("aten", "nki") and B.shape[0] > 128 else ""
            return f"trsm={_blas_label(r, A, B)}{blocked}"

        def metric(r, J):
            if r == "host":
                return "metric_inv_half=host fp64 (download J, upload X)"
            return on("cholesky", J)  # torch cholesky, then trnblas trsm

        def offload(r, *args, **_):
            return "upload" if r else None

        def ao2mo(r, eri, nocc, nvir, nbasis=None):
            where = _residency(eri)
            if eri.dim() == 2 and r != "trn2":
                return "unpack"
            if r == "aten":
                return "ao_to_mo=" + ("host" if where == "host" else "aten")
            return f"ao_to_mo={r}" + ("(packed)" if eri.dim() == 2 else "")

        def mp2(r, B):
            if r == "aten":
                return "mp2_energy=" + ("host" if _residency(B) == "host" else "aten")
            return f"mp2_energy={r}"

        def chol(r, A):
            return on("cholesky", A)

        self._wrap(bd, "_gemm_route", gemm)
        self._wrap(bd, "_trsm_route", trsm)
        self._wrap(bd, "_metric_solve_route", metric)
        self._wrap(td, "_offload_host", offload)
        # quantum.py binds the two route helpers by name at import.
        self._wrap(tq, "_ao_to_mo_route", ao2mo)
        self._wrap(tq, "_mp2_energy_route", mp2)
        self._wrap(sf, "_cholesky_lower", chol)
        # Calls that never reach a routing helper: torch on the operands' device.
        self._wrap_entry(trnblas, "gemm", lambda _a, A, B, *r, **k: on("gemm", A, B))
        self._wrap(bd, "_trsm_torch", lambda r, _a, A, B, *x: on("trsm", A, B))
        return self

    def __exit__(self, *exc):
        for mod, name, orig in reversed(self._saved):
            setattr(mod, name, orig)
        self._saved.clear()

    def route(self, stage: str) -> str:
        return " -> ".join(self.events.get(stage, [])) or "-"


# ---------------------------------------------------------------- pipelines


def fused_half(eri, C_occ, C_vir):
    return trntensor.ao_to_mo_transform(eri, C_occ, C_vir)


def fused_metric(ia_P, J):
    n_occ, n_vir, n_aux = ia_P.shape
    X = trnblas.metric_inv_half(J)
    return trnblas.gemm(1.0, ia_P.reshape(n_occ * n_vir, n_aux), X).reshape(
        n_occ, n_vir, n_aux
    )


PIPELINES = {
    "example": {
        "half": ("trnblas.gemm+batched_gemm", ex.half_transform),
        "metric": ("trnsolver.cholesky+trnblas.trsm+batched_gemm", ex.metric_contract),
    },
    "fused": {
        "half": ("trntensor.ao_to_mo_transform", fused_half),
        "metric": ("trnblas.metric_inv_half+gemm", fused_metric),
    },
    # As "fused", but the integrals are the packed lower triangle (npair, naux)
    # that drivers store (trntensor.pack_tril, PySCF's _cderi.T; trntensor#55).
    # Packing happens once on the host before timing, like a driver's output;
    # in `neuron` mode only the packed rows are uploaded (half the bytes).
    "packed": {
        "half": ("trntensor.ao_to_mo_transform(packed)", fused_half),
        "metric": ("trnblas.metric_inv_half+gemm", fused_metric),
    },
}


# ------------------------------------------------------------------- runner


def set_backends(mode: str) -> None:
    for lib in LIBS:
        lib.set_backend("pytorch" if mode == "cpu" else "auto")


def run_once(pipeline, host_inputs, device, log):
    times = {}
    t = time.perf_counter()
    eri, J, C_occ, C_vir, eps_occ, eps_vir = ex.to_device(host_inputs, device)
    ex.synchronize(device)
    times["h2d"] = time.perf_counter() - t

    _, half = PIPELINES[pipeline]["half"]
    _, metric = PIPELINES[pipeline]["metric"]

    log.stage = "half"
    t = time.perf_counter()
    ia_P = half(eri, C_occ, C_vir)
    ex.synchronize(device)
    times["half"] = time.perf_counter() - t

    log.stage = "metric"
    t = time.perf_counter()
    B = metric(ia_P, J)
    ex.synchronize(device)
    times["metric"] = time.perf_counter() - t

    log.stage = "energy"
    t = time.perf_counter()
    e = float(trntensor.mp2_energy(B, eps_occ, eps_vir))  # includes the scalar readback
    times["energy"] = time.perf_counter() - t
    log.stage = None
    times["total"] = sum(times.values())
    return times, ia_P, B, e


def profile_case(size, mode, pipeline, *, repeats, seed, ref_cache):
    n_ao, n_aux, n_occ = LADDER[size]
    device = "neuron" if mode == "neuron" else "cpu"
    set_backends(mode)

    eri, J, C_occ, C_vir, eps_occ, eps_vir, _Q = ex.synthetic_system(
        n_ao=n_ao, n_aux=n_aux, n_occ=n_occ, seed=seed
    )
    host_inputs = tuple(
        t.contiguous() for t in (eri, J, C_occ, C_vir, eps_occ, eps_vir)
    )
    if size not in ref_cache:
        ref_cache[size] = reference_fp64(*host_inputs)
    ia_ref, B_ref, e_ref = ref_cache[size]
    if pipeline == "packed":
        host_inputs = (trntensor.pack_tril(host_inputs[0]), *host_inputs[1:])
    del eri, J, C_occ, C_vir, eps_occ, eps_vir

    runs = []
    energies = []
    caught = []
    for k in range(repeats + 1):
        with warnings.catch_warnings(record=True) as w, RouteLog() as log:
            warnings.simplefilter("always")
            times, ia_P, B, e = run_once(pipeline, host_inputs, device, log)
        caught.extend(str(x.message).splitlines()[0][:120] for x in w)
        runs.append(times)
        energies.append(e)
        if k != repeats:
            del ia_P, B
            gc.collect()

    # Routes the libraries took in the last (warm) run.
    if mode == "cpu":
        half_route = metric_route = mp2_route = "torch (cpu, pytorch backends)"
    else:
        half_route, metric_route, mp2_route = (
            log.route(st) for st in ("half", "metric", "energy")
        )

    warm = runs[1:] if repeats else runs
    med = {k: statistics.median(r[k] for r in warm) for k in runs[0]}
    rec = {
        "size": size,
        "n_ao": n_ao,
        "n_aux": n_aux,
        "n_occ": n_occ,
        "n_vir": n_ao - n_occ,
        "mode": mode,
        "pipeline": pipeline,
        "threads": torch.get_num_threads(),
        "cold_total_ms": runs[0]["total"] * 1e3,
        **{f"{k}_ms": v * 1e3 for k, v in med.items()},
        "route_half": f"{PIPELINES[pipeline]['half'][0]} [{half_route}]",
        "route_metric": f"{PIPELINES[pipeline]['metric'][0]} [{metric_route}]",
        "route_energy": f"trntensor.mp2_energy [{mp2_route}]",
        "err_ia": rel_err(ia_P, ia_ref),
        "err_B": rel_err(B, B_ref),
        "e": e,
        "e_fp64": e_ref,
        "err_e": abs(e - e_ref) / abs(e_ref),
        "err_e_max_over_runs": max(abs(x - e_ref) / abs(e_ref) for x in energies),
        "warnings": sorted(set(caught)),
    }
    del ia_P, B, host_inputs
    gc.collect()
    return rec


HEADER = (
    "| size (ao/aux/occ) | mode | pipeline | h2d | half | metric | energy | total (warm) "
    "| cold | rel err B | max rel err E |"
)


def row(r) -> str:
    return (
        f"| {r['size']} ({r['n_ao']}/{r['n_aux']}/{r['n_occ']}) | {r['mode']} | {r['pipeline']} "
        f"| {r['h2d_ms']:.1f} | {r['half_ms']:.1f} | {r['metric_ms']:.1f} | {r['energy_ms']:.1f} "
        f"| {r['total_ms']:.1f} | {r['cold_total_ms']:.0f} | {r['err_B']:.1e} "
        f"| {r['err_e_max_over_runs']:.1e} |"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--sizes", default="xs,s,m", help=f"comma list from {list(LADDER)}")
    p.add_argument(
        "--modes", default="cpu", help="comma list from cpu,host-auto,neuron"
    )
    p.add_argument("--pipelines", default="example,fused,packed")
    p.add_argument(
        "--repeats", type=int, default=3, help="warm runs after one cold run"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="append one JSON record per case to this file")
    args = p.parse_args()

    if os.environ.get("OMP_NUM_THREADS"):
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

    print(HEADER)
    print("|" + "---|" * (HEADER.count("|") - 1))
    ref_cache: dict = {}
    for size in args.sizes.split(","):
        for mode in args.modes.split(","):
            for pipeline in args.pipelines.split(","):
                rec = profile_case(
                    size,
                    mode,
                    pipeline,
                    repeats=args.repeats,
                    seed=args.seed,
                    ref_cache=ref_cache,
                )
                print(row(rec), flush=True)
                print(
                    f"|   routes: half={rec['route_half']}; metric={rec['route_metric']}; "
                    f"energy={rec['route_energy']}; warnings={len(rec['warnings'])}",
                    flush=True,
                )
                if args.json:
                    with open(args.json, "a") as f:
                        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
