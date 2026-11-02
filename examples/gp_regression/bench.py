"""Time-to-solution for the matrix-free GP posterior mean (#109, milestone 1).

Columns, one row per n (d = 3, R = 1 + probes right-hand sides):

- device: trnsci on Trn2 (`--device`): fused kernel operator + block CG
- cpu2:   the same algorithm and code on this host's CPU, 2 threads (the
          suite's same-box convention, trnsci/trnsci#35)
- cpuall: the same on all host vCPUs
- gpytorch: the reference package's CG exact GP on the CPU (all vCPUs)

A CPU solve above `--cpu-full-max` points is *projected*, not run: the
measured CPU rate of one operator application (kernel entries per second,
timed on a block of rows) times the iteration count of the run in the same
row, plus the measured preconditioner build. Projected cells are marked
"~". gpytorch runs up to `--gpytorch-max`. GPU numbers are not part of this
script; quote them only if measured.

Usage:
    python bench.py --device --sizes 100000,1000000 --json gp_bench.json
    python bench.py --sizes 20000,50000                 # CPU columns only
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import trnsparse
from gp_regression import posterior_mean, synthetic_field
from trnsparse.kernel_matmul import _torch_kernel_matmul


def _cpu_rate(X, R, lengthscale, threads, rows=2048, reps=3):
    """Kernel entries/s of one CPU operator application (PyTorch path)."""
    torch.set_num_threads(threads)
    trnsparse.set_backend("pytorch")
    op = trnsparse.KernelOperator(X, "rbf", lengthscale, 1.0, 1e-2, dtype=torch.float32)
    V = torch.randn(X.shape[0], R)
    Xa = op.X[: min(rows, X.shape[0])].contiguous()
    _torch_kernel_matmul(Xa, op.X, V, op.kind, op._a, 1.0, 0.0)
    t0 = time.perf_counter()
    for _ in range(reps):
        _torch_kernel_matmul(Xa, op.X, V, op.kind, op._a, 1.0, 0.0)
    return Xa.shape[0] * X.shape[0] * reps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--sizes", default="20000,100000,1000000")
    ap.add_argument("--modes", default="device,cpu2,cpuall,gpytorch")
    ap.add_argument("--probes", type=int, default=32)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--lengthscale", type=float, default=0.15)
    ap.add_argument("--noise", type=float, default=1e-2)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--cpu-full-max", type=int, default=20000)
    ap.add_argument("--gpytorch-max", type=int, default=100000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    modes = [m for m in args.modes.split(",") if m]
    if not args.device:
        modes = [m for m in modes if m != "device"]
    nvcpu = os.cpu_count()
    rows = []
    for n in [int(s) for s in args.sizes.split(",")]:
        X, y, Xt, ft = synthetic_field(
            n, 3, lengthscale=args.lengthscale, noise=args.noise
        )
        rec = {"n": n, "R": 1 + args.probes, "rank": args.rank}
        iters = None
        precond_s = None
        common = {
            "lengthscale": args.lengthscale,
            "noise": args.noise,
            "probes": args.probes,
            "precond_rank": args.rank,
            "tol": args.tol,
        }
        for mode in modes:
            if mode == "device":
                trnsparse.set_backend("nki")
                t0 = time.perf_counter()
                res = posterior_mean(X, y, Xt, device="neuron", **common)
                rec["device_s"] = time.perf_counter() - t0
                rec["device_iterations"] = iters = res.iterations
                rec["device_timings"] = res.timings
                rec["device_sampled_residual"] = res.sampled_residual
                rec["device_rmse"] = float((res.mean - ft).pow(2).mean().sqrt())
                precond_s = res.timings.get("precond_s", 0.0)
            elif mode in ("cpu2", "cpuall"):
                threads = 2 if mode == "cpu2" else nvcpu
                if n <= args.cpu_full_max:
                    torch.set_num_threads(threads)
                    trnsparse.set_backend("pytorch")
                    t0 = time.perf_counter()
                    res = posterior_mean(X, y, Xt, device="cpu", **common)
                    rec[f"{mode}_s"] = time.perf_counter() - t0
                    rec[f"{mode}_iterations"] = res.iterations
                    iters = iters or res.iterations
                    precond_s = (
                        precond_s
                        if precond_s is not None
                        else res.timings.get("precond_s", 0.0)
                    )
                else:
                    rate = _cpu_rate(X, 1 + args.probes, args.lengthscale, threads)
                    it = iters if iters is not None else float("nan")
                    # iterations + the final true-residual application + prediction (small)
                    rec[f"{mode}_s"] = (it + 1) * n * n / rate + (precond_s or 0.0)
                    rec[f"{mode}_projected"] = True
                    rec[f"{mode}_entries_per_s"] = rate
            elif mode == "gpytorch":
                if n <= args.gpytorch_max:
                    from validate import gpytorch_reference

                    torch.set_num_threads(nvcpu)
                    _, t = gpytorch_reference(
                        X,
                        y,
                        Xt,
                        args.lengthscale,
                        1.0,
                        args.noise,
                        tol=args.tol,
                        precond=args.rank,
                    )
                    rec["gpytorch_s"] = t
        rows.append(rec)

        def cell(m, rec=rec):
            v = rec.get(f"{m}_s")
            if v is None:
                return "n/a"
            return ("~" if rec.get(f"{m}_projected") else "") + f"{v:.1f}"

        print(
            f"n={n:>8} R={rec['R']} iters={iters} | "
            + " | ".join(f"{m} {cell(m)} s" for m in modes),
            flush=True,
        )
    print("\n| n | iterations | " + " | ".join(modes) + " |")
    print("|---|---|" + "---|" * len(modes))
    for rec in rows:
        it = rec.get(
            "device_iterations",
            rec.get("cpu2_iterations", rec.get("cpuall_iterations", "")),
        )
        cells = []
        for m in modes:
            v = rec.get(f"{m}_s")
            cells.append(
                "n/a"
                if v is None
                else ("~" if rec.get(f"{m}_projected") else "") + f"{v:.1f} s"
            )
        print(f"| {rec['n']} | {it} | " + " | ".join(cells) + " |")
    print(f"\n~ projected from the measured CPU entries/s; host vCPUs = {nvcpu}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1, default=str)


if __name__ == "__main__":
    main()
