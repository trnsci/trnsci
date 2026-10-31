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

The inputs are random (see df_mp2_synthetic.py); energies are not physically
meaningful, but their agreement with float64 is.

Run (Trn2, one logical core, host threads pinned)::

    NEURON_RT_VISIBLE_CORES=2 OMP_NUM_THREADS=2 taskset -c 8,9 \\
        python examples/quantum_chemistry/df_mp2_profile.py \\
        --sizes s,m,l,xl --modes cpu,neuron --pipelines example,fused \\
        --json df_mp2_profile.jsonl
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


def _route_gemm(*tensors) -> str:
    from trnblas.nki import dispatch as d

    if not d._use_nki():
        return "aten"
    return "aten" if d._gemm_via_aten(*tensors) else "nki"


def _route_ao2mo(eri, nocc, nvir) -> str:
    from trntensor.nki import dispatch as d

    return d._ao_to_mo_route(eri, nocc, nvir)


def _route_mp2(B) -> str:
    from trntensor.nki import dispatch as d

    return d._mp2_energy_route(B)


def _route_metric(J) -> str:
    from trnblas.nki import dispatch as d

    return d._metric_solve_route(J)


def _route_trsm(J) -> str:
    from trnblas.nki import dispatch as d

    if not d._use_nki():
        return "torch solve_triangular"
    blocked = J.shape[0] > 128
    return f"blocked, gemm={_route_gemm(J, J)}" if blocked else "solve_triangular"


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
}


def stage_routes(pipeline, eri, J, C_occ, C_vir, ia_shape):
    nocc, nvir = C_occ.shape[1], C_vir.shape[1]
    if pipeline == "example":
        half = f"gemm={_route_gemm(C_occ, eri)}"
        metric = f"cholesky=aten, trsm={_route_trsm(J)}, bmm={_route_gemm(J, J)}"
    else:
        half = _route_ao2mo(eri, nocc, nvir)
        metric = f"inv_half={_route_metric(J)}, gemm={_route_gemm(J, J)}"
    return half, metric


# ------------------------------------------------------------------- runner


def set_backends(mode: str) -> None:
    for lib in LIBS:
        lib.set_backend("pytorch" if mode == "cpu" else "auto")


def run_once(pipeline, host_inputs, device):
    eri, J, C_occ, C_vir, eps_occ, eps_vir = host_inputs
    times = {}
    t = time.perf_counter()
    eri, J, C_occ, C_vir, eps_occ, eps_vir = ex.to_device(host_inputs, device)
    ex.synchronize(device)
    times["h2d"] = time.perf_counter() - t

    _, half = PIPELINES[pipeline]["half"]
    _, metric = PIPELINES[pipeline]["metric"]

    t = time.perf_counter()
    ia_P = half(eri, C_occ, C_vir)
    ex.synchronize(device)
    times["half"] = time.perf_counter() - t

    t = time.perf_counter()
    B = metric(ia_P, J)
    ex.synchronize(device)
    times["metric"] = time.perf_counter() - t

    t = time.perf_counter()
    e = float(trntensor.mp2_energy(B, eps_occ, eps_vir))  # includes the scalar readback
    times["energy"] = time.perf_counter() - t
    times["total"] = sum(times.values())
    return times, (eri, J, C_occ, C_vir), ia_P, B, e


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

    runs = []
    energies = []
    caught = []
    for _ in range(repeats + 1):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            times, dev_in, ia_P, B, e = run_once(pipeline, host_inputs, device)
        caught.extend(str(x.message).splitlines()[0][:120] for x in w)
        runs.append(times)
        energies.append(e)
        last = (dev_in, ia_P, B, e)
        if _ != repeats:
            del dev_in, ia_P, B
            gc.collect()

    (eri_d, J_d, C_occ_d, C_vir_d), ia_P, B, e = last
    half_route, metric_route = stage_routes(
        pipeline, eri_d, J_d, C_occ_d, C_vir_d, ia_P.shape
    )
    mp2_route = _route_mp2(B)
    if mode == "cpu":
        half_route = metric_route = mp2_route = "aten (cpu)"

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
    del eri_d, J_d, C_occ_d, C_vir_d, ia_P, B, last
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
    p.add_argument("--pipelines", default="example,fused")
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
