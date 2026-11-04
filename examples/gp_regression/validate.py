"""Validate the matrix-free GP posterior mean against independent references (#109).

1. Exact, CPU float64 (default n = 2x10^4): scikit-learn
   `GaussianProcessRegressor` with the kernel fixed (`optimizer=None`) and
   `scipy.linalg.cho_factor` / `cho_solve`. Requirement: posterior-mean
   relative error <= 1e-4.
2. Same algorithm, independent code (default n = 2x10^4): gpytorch's
   CG-based exact GP on the CPU (`max_cholesky_size(0)`), at fixed
   hyperparameters. Agreement is to CG tolerance. Without pykeops gpytorch
   evaluates the full n x n kernel inside CG (8 n^2 bytes: 80 GB at n = 10^5),
   and `checkpoint_kernel` ran out of memory on a 48 GB host even at
   n = 5000 (gpytorch 1.15), so n = 10^5 needs pykeops or a large-memory
   host: `--n-gpytorch 100000`.

The trnsci side runs on the PyTorch backend on the CPU, or on the device
with --device. Relative error is ||m - m_ref||_2 / ||m_ref||_2 over the test
points (max-abs relative is printed too).

Usage:
    python validate.py                       # exact and gpytorch at 2e4
    python validate.py --n-gpytorch 0        # exact only
    python validate.py --device              # trnsci on Trn2
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import trnsparse
from gp_regression import posterior_mean, synthetic_field


def _rel(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return float((a - b).norm() / b.norm()), float((a - b).abs().max() / b.abs().max())


def exact_references(X, y, Xt, lengthscale, outputscale, noise, sklearn_too=True):
    """Posterior means from scipy cho_solve and (optionally) scikit-learn, float64."""
    import scipy.linalg as sla
    from scipy.spatial.distance import cdist

    Xn, yn, Xtn = X.numpy(), y.numpy(), Xt.numpy()
    out = {}
    t0 = time.perf_counter()
    K = outputscale * np.exp(-cdist(Xn, Xn, "sqeuclidean") / (2 * lengthscale**2))
    K[np.diag_indices_from(K)] += noise
    c = sla.cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
    alpha = sla.cho_solve(c, yn, check_finite=False)
    del K, c
    Ks = outputscale * np.exp(-cdist(Xtn, Xn, "sqeuclidean") / (2 * lengthscale**2))
    out["scipy"] = (Ks @ alpha, time.perf_counter() - t0)
    if sklearn_too:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel

        t0 = time.perf_counter()
        k = ConstantKernel(outputscale, "fixed") * RBF(lengthscale, "fixed")
        gpr = GaussianProcessRegressor(
            k, alpha=noise, optimizer=None, normalize_y=False, copy_X_train=False
        )
        gpr.fit(Xn, yn)
        out["sklearn"] = (gpr.predict(Xtn), time.perf_counter() - t0)
    return out


def gpytorch_reference(
    X, y, Xt, lengthscale, outputscale, noise, *, tol=1e-6, precond=512
):
    """Posterior mean from gpytorch's CG-based exact GP (CPU, float64)."""
    import gpytorch

    class _GP(gpytorch.models.ExactGP):
        def __init__(self, x, t, likelihood):
            super().__init__(x, t, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-8)
    ).double()
    model = _GP(X, y, likelihood).double()
    model.covar_module.base_kernel.lengthscale = lengthscale
    model.covar_module.outputscale = outputscale
    likelihood.noise = noise
    model.eval()
    likelihood.eval()
    t0 = time.perf_counter()
    with (
        torch.no_grad(),
        gpytorch.settings.max_cholesky_size(0),
        gpytorch.settings.eval_cg_tolerance(tol),
        gpytorch.settings.cg_tolerance(tol),
        gpytorch.settings.max_cg_iterations(10000),
        gpytorch.settings.max_preconditioner_size(precond),
        gpytorch.settings.skip_posterior_variances(True),
    ):
        mean = model(Xt).mean
    return mean, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-exact", type=int, default=20000)
    ap.add_argument("--n-gpytorch", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--lengthscale", type=float, default=0.15)
    ap.add_argument("--noise", type=float, default=1e-2)
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument(
        "--refine", action="store_true", help="fp64 refinement in the exact case"
    )
    ap.add_argument("--no-sklearn", action="store_true")
    ap.add_argument("--predict", default="host64", choices=["host64", "operator"])
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    device = "neuron" if args.device else "cpu"
    trnsparse.set_backend("nki" if args.device else "pytorch")
    s = 1.0
    report = {"device": device}
    ok = True

    if args.n_exact:
        n = args.n_exact
        X, y, Xt, ft = synthetic_field(
            n, 3, n_test=args.n_test, lengthscale=args.lengthscale, noise=args.noise
        )
        t0 = time.perf_counter()
        res = posterior_mean(
            X, y, Xt, lengthscale=args.lengthscale, noise=args.noise, precond_rank=args.rank,
            tol=args.tol, refine=args.refine, device=device, predict=args.predict,
        )  # fmt: skip
        t_ours = time.perf_counter() - t0
        refs = exact_references(
            X, y, Xt, args.lengthscale, s, args.noise, not args.no_sklearn
        )
        print(f"== exact fp64 references, n = {n}, d = 3, R = 33, rank {args.rank}")
        print(
            f"trnsci: {res.iterations} iterations, {t_ours:.1f} s, sampled fp64 residual {res.sampled_residual:.2e}"
        )
        for name, (m, t) in refs.items():
            rel2, relmax = _rel(res.mean, m)
            passed = rel2 <= 1e-4
            ok &= passed
            print(
                f"  vs {name:8s} ({t:6.1f} s): rel err {rel2:.2e} (max-abs rel {relmax:.2e}) {'PASS' if passed else 'FAIL'} (<= 1e-4)"
            )
            report[f"exact_{name}"] = {
                "n": n,
                "rel_err": rel2,
                "max_rel_err": relmax,
                "ref_time_s": t,
            }
        rmse = float((res.mean - ft).pow(2).mean().sqrt())
        report["exact_trnsci"] = {
            "iterations": res.iterations,
            "time_s": t_ours,
            "rmse_vs_latent": rmse,
            **res.timings,
        }

    if args.n_gpytorch:
        try:
            import gpytorch  # noqa: F401
        except ImportError:
            print("gpytorch not installed; skipping the 1e5 comparison")
        else:
            n = args.n_gpytorch
            X, y, Xt, _ = synthetic_field(
                n,
                3,
                n_test=args.n_test,
                lengthscale=args.lengthscale,
                noise=args.noise,
                seed=1,
            )
            t0 = time.perf_counter()
            res = posterior_mean(
                X, y, Xt, lengthscale=args.lengthscale, noise=args.noise, precond_rank=args.rank,
                tol=args.tol, device=device, predict=args.predict,
            )  # fmt: skip
            t_ours = time.perf_counter() - t0
            m_ref, t_ref = gpytorch_reference(
                X,
                y,
                Xt,
                args.lengthscale,
                s,
                args.noise,
                tol=args.tol,
                precond=args.rank,
            )
            rel2, relmax = _rel(res.mean, m_ref)
            print(f"== gpytorch (CPU CG), n = {n}")
            print(
                f"trnsci: {res.iterations} iterations, {t_ours:.1f} s; gpytorch {t_ref:.1f} s"
            )
            print(
                f"  rel err {rel2:.2e} (max-abs rel {relmax:.2e}); CG tolerance {args.tol:.0e}"
            )
            report["gpytorch"] = {
                "n": n,
                "rel_err": rel2,
                "max_rel_err": relmax,
                "time_s": t_ref,
                "trnsci_time_s": t_ours,
                "trnsci_iterations": res.iterations,
            }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
