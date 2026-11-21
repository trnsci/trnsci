"""Exact Gaussian-process regression, matrix-free, on Trainium (trnsci/trnsci#109, milestone 1).

The posterior mean of a GP with an RBF (or Matern) kernel and Gaussian noise
at test points X*:

    mean(X*) = K(X*, X) alpha,   (K(X, X) + noise I) alpha = y.

No low-rank or inducing-point approximation of the model: K is never formed.
The solve is block CG (`trnsolver.cg_block`) on the matrix-free operator
`trnsparse.KernelOperator`, one fused kernel launch per iteration covering
every right-hand side: [y, z_1 .. z_p], the data plus p random-sign probe
vectors (the probe solves are what the log-marginal-likelihood estimate of
milestone 3 needs; here they ride along at no extra launches). An optional
pivoted-Cholesky preconditioner (`trnsolver.pivoted_lowrank_preconditioner`)
cuts the iteration count; it is built from the kernel's diagonal and rows
only. Device iterations are fp32; the host checks an fp64 residual on
sampled rows, and `refine=True` runs fp64 iterative refinement
(`trnsolver.cg_block_ir`) where an fp64 operator on the host is affordable.

Synthetic data: inputs uniform in the unit cube, outputs a random-Fourier-
feature draw from GP(0, s * RBF(l)) plus noise, so the generating
hyperparameters are known.

Usage:
    python gp_regression.py --n 20000                     # CPU
    python gp_regression.py --n 1000000 --device          # Trn2, one logical core
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import torch
import trnsolver
import trnsparse

_RFF_BLOCK = 16384


def synthetic_field(
    n: int,
    d: int = 3,
    *,
    n_test: int = 1000,
    lengthscale: float = 0.15,
    outputscale: float = 1.0,
    noise: float = 1e-2,
    features: int = 2048,
    seed: int = 0,
):
    """(X, y, X_test, f_test): a known-hyperparameter synthetic GP field.

    f(x) = sum_k a_k cos(w_k . x + b_k), w_k ~ N(0, I / l^2), b_k ~ U(0, 2 pi),
    a_k ~ N(0, 2 s / features): random Fourier features, whose covariance
    tends to s * exp(-|x - x'|^2 / (2 l^2)) as features grows. y = f(X) + eps,
    eps ~ N(0, noise). float64 on the host.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(features, d, generator=g, dtype=torch.float64) / lengthscale
    b = torch.rand(features, generator=g, dtype=torch.float64) * (2 * math.pi)
    a = torch.randn(features, generator=g, dtype=torch.float64) * math.sqrt(
        2 * outputscale / features
    )

    def f(P):
        out = torch.empty(P.shape[0], dtype=torch.float64)
        for i in range(0, P.shape[0], _RFF_BLOCK):
            out[i : i + _RFF_BLOCK] = (
                torch.cos(torch.addmm(b, P[i : i + _RFF_BLOCK], W.T)) @ a
            )
        return out

    X = torch.rand(n, d, generator=g, dtype=torch.float64)
    Xt = torch.rand(n_test, d, generator=g, dtype=torch.float64)
    y = f(X) + math.sqrt(noise) * torch.randn(n, generator=g, dtype=torch.float64)
    return X, y, Xt, f(Xt)


@dataclass
class GPResult:
    mean: torch.Tensor  # posterior mean at the test points, float64 host
    alpha: torch.Tensor  # (K + noise I)^-1 y, float64 host
    iterations: int
    converged: bool
    true_residual: float  # ||y - K^ alpha|| / ||y|| from the solver (fp32 device)
    sampled_residual: float  # fp64 host residual on sampled rows
    timings: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def _rademacher(n: int, p: int, seed: int) -> torch.Tensor:
    # Random-sign probes. trnrand.rademacher (trnsci/trnrand#87) will generate
    # these on the device; a host generator keeps this example reproducible now.
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, 2, (n, p), generator=g, dtype=torch.int8).to(torch.float64) * 2
        - 1
    )


def _sync(device):
    if torch.device(device).type == "neuron" and hasattr(torch, "neuron"):
        torch.neuron.synchronize()


def posterior_mean(
    X: torch.Tensor,
    y: torch.Tensor,
    X_test: torch.Tensor,
    *,
    kernel: str = "rbf",
    lengthscale: float = 0.15,
    outputscale: float = 1.0,
    noise: float = 1e-2,
    probes: int = 32,
    precond_rank: int = 1024,
    tol: float = 1e-6,
    maxiter: int = 1000,
    check_every: int = 10,
    refine: bool = False,
    device="cpu",
    sample_rows: int = 2048,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    predict: str = "host64",
) -> GPResult:
    """Posterior mean at X_test by preconditioned block CG on the matrix-free operator.

    `dtype` is the operator and CG precision: float32 (the device's, and the
    default on the CPU too, so CPU baselines run the same arithmetic) or
    float64 (CPU only). `predict` is "host64" (the posterior mean
    K(X*, X) alpha in float64 on the host) or "operator" (the fp32 fused
    operator, on the device when there is one).
    """
    t = {}
    t0 = time.perf_counter()
    op = trnsparse.KernelOperator(
        X, kernel, lengthscale, outputscale, noise, device=device, dtype=dtype
    )
    n = X.shape[0]
    B64 = torch.cat([y[:, None], _rademacher(n, probes, seed + 1)], 1)
    B = B64.to(device=device, dtype=op.dtype)
    t["setup_s"] = time.perf_counter() - t0

    M = None
    if precond_rank:
        t0 = time.perf_counter()
        L = trnsolver.pivoted_cholesky(op.diag, op.rows, min(precond_rank, n))
        # Applied in float64 on the host for an fp32 solve: the fp32 Woodbury
        # form cancels and breaks CG at GP noise levels (see
        # trnsolver.PivotedLowRankPreconditioner). One (n, R) round trip per
        # iteration; a stable on-device form is milestone-2 work.
        mode = "native" if op.dtype == torch.float64 and device == "cpu" else "host64"
        M = trnsolver.PivotedLowRankPreconditioner(L, noise, apply=mode)
        t["precond_s"] = time.perf_counter() - t0
        t["precond_rank"] = L.shape[1]

    t0 = time.perf_counter()
    if refine:

        def A_hi(V):
            return trnsparse.kernel_matmul_reference(
                X, None, V, kernel, lengthscale, outputscale, noise
            )

        sol, info = trnsolver.cg_block_ir(
            op,
            B,
            A_hi=A_hi,
            M=M,
            tol=tol,
            inner_tol=1e-4,
            maxiter=maxiter,
            check_every=check_every,
        )
        iters = int(sum(info["inner_iterations"]))
        converged = bool(info["converged"][0])
        true_res = float(info["true_residual"][0])
    else:
        sol, info = trnsolver.cg_block(
            op, B, M=M, tol=tol, maxiter=maxiter, check_every=check_every
        )
        iters = int(info["iterations"])
        converged = bool(info["converged"][0])
        true_res = float(info["true_residual"][0])
    _sync(device)
    t["solve_s"] = time.perf_counter() - t0

    alpha = sol[:, 0].detach().to("cpu", torch.float64)
    t0 = time.perf_counter()
    if predict == "host64":
        # K(X*, X) alpha in float64 on the host: alpha is large (about
        # ||y|| / noise) and the mean is a cancelling sum of it, so the fp32
        # operator's per-entry error (~u |x|^2 / l^2 from the distance
        # expansion) would cap the mean's accuracy near 1e-4. q * n entries,
        # once.
        mean = trnsparse.kernel_matmul_reference(
            X_test, X, alpha, kernel, lengthscale, outputscale
        )
    else:
        mean = op.cross(X_test, alpha.to(device=device, dtype=op.dtype))
        mean = mean.to("cpu", torch.float64)
    _sync(device)
    t["predict_s"] = time.perf_counter() - t0

    # fp64 residual of the alpha solve on sampled rows (all rows when n is small).
    idx = (
        torch.arange(n)
        if n <= sample_rows
        else torch.linspace(0, n - 1, sample_rows).long()
    )
    Ka = trnsparse.kernel_matmul_reference(
        X[idx], X, alpha, kernel, lengthscale, outputscale
    )
    r = y[idx] - Ka - noise * alpha[idx]
    sampled = float(r.norm() / y[idx].norm())
    return GPResult(mean, alpha, iters, converged, true_res, sampled, t, info)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--kernel", default="rbf")
    ap.add_argument("--lengthscale", type=float, default=0.15)
    ap.add_argument("--outputscale", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=1e-2)
    ap.add_argument("--probes", type=int, default=32)
    ap.add_argument(
        "--rank", type=int, default=1024, help="pivoted-Cholesky rank (0: none)"
    )
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--check-every", type=int, default=10)
    ap.add_argument(
        "--refine", action="store_true", help="fp64 host iterative refinement"
    )
    ap.add_argument(
        "--device", action="store_true", help="run on the neuron device (nki backend)"
    )
    ap.add_argument(
        "--threads", type=int, default=0, help="CPU threads (0: torch default)"
    )
    ap.add_argument("--predict", default="host64", choices=["host64", "operator"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cpu"
    if args.device:
        trnsparse.set_backend("nki")
        device = "neuron"
    else:
        trnsparse.set_backend("pytorch")
    X, y, Xt, ft = synthetic_field(
        args.n, args.d, n_test=args.n_test, lengthscale=args.lengthscale,
        outputscale=args.outputscale, noise=args.noise, seed=args.seed,
    )  # fmt: skip
    res = posterior_mean(
        X, y, Xt, kernel=args.kernel, lengthscale=args.lengthscale, outputscale=args.outputscale,
        noise=args.noise, probes=args.probes, precond_rank=args.rank, tol=args.tol,
        maxiter=args.maxiter, check_every=args.check_every, refine=args.refine, device=device,
        seed=args.seed, predict=args.predict,
    )  # fmt: skip
    rmse = float((res.mean - ft).pow(2).mean().sqrt())
    print(
        f"n={args.n} d={args.d} kernel={args.kernel} device={device} R={1 + args.probes}"
    )
    print(f"iterations={res.iterations} converged={res.converged}")
    print(
        f"solver true residual (fp32)={res.true_residual:.2e}  fp64 sampled residual={res.sampled_residual:.2e}"
    )
    print(f"test RMSE vs latent f={rmse:.4e}  (noise sd {math.sqrt(args.noise):.3e})")
    print(
        "timings: "
        + ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in res.timings.items()
        )
    )


if __name__ == "__main__":
    main()
