"""Smoke and accuracy tests for the matrix-free GP regression exemplar (#109)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "gp_regression"
sys.path.insert(0, str(EXAMPLES_DIR))

trnsparse = pytest.importorskip("trnsparse")
trnsolver = pytest.importorskip("trnsolver")
if not hasattr(trnsparse, "KernelOperator") or not hasattr(trnsolver, "cg_block"):
    pytest.skip(
        "needs trnsparse.KernelOperator (trnsci/trnsparse#93) and trnsolver.cg_block "
        "(trnsci/trnsolver#97)",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _pytorch_backend():
    old = trnsparse.get_backend()
    trnsparse.set_backend("pytorch")
    yield
    trnsparse.set_backend(old)


def _exact_mean(X, y, Xt, ell, s, noise):
    K = s * torch.exp(-torch.cdist(X, X).square() / (2 * ell**2))
    K.diagonal().add_(noise)
    alpha = torch.cholesky_solve(y[:, None], torch.linalg.cholesky(K))[:, 0]
    return s * torch.exp(-torch.cdist(Xt, X).square() / (2 * ell**2)) @ alpha


@pytest.mark.parametrize("refine", [False, True])
def test_posterior_mean_matches_exact_fp64(refine):
    import gp_regression as gp

    X, y, Xt, ft = gp.synthetic_field(
        3000, 3, n_test=200, lengthscale=0.15, noise=1e-2, seed=3
    )
    res = gp.posterior_mean(
        X,
        y,
        Xt,
        lengthscale=0.15,
        noise=1e-2,
        probes=8,
        precond_rank=256,
        refine=refine,
    )
    ref = _exact_mean(X, y, Xt, 0.15, 1.0, 1e-2)
    rel = float((res.mean - ref).norm() / ref.norm())
    assert res.converged
    assert rel < (1e-6 if refine else 1e-4)
    assert res.sampled_residual < (1e-6 if refine else 1e-3)
    # The model recovers the latent field to about the noise level.
    assert float((res.mean - ft).pow(2).mean().sqrt()) < 0.1


def test_preconditioner_reduces_iterations():
    import gp_regression as gp

    X, y, Xt, _ = gp.synthetic_field(
        2000, 3, n_test=10, lengthscale=0.15, noise=1e-2, seed=4
    )
    plain = gp.posterior_mean(
        X, y, Xt, lengthscale=0.15, noise=1e-2, probes=2, precond_rank=0
    )
    pre = gp.posterior_mean(
        X, y, Xt, lengthscale=0.15, noise=1e-2, probes=2, precond_rank=256
    )
    assert pre.iterations < plain.iterations / 2


def test_deterministic():
    import gp_regression as gp

    X, y, Xt, _ = gp.synthetic_field(1500, 3, n_test=50, seed=5)
    a = gp.posterior_mean(X, y, Xt, probes=4, precond_rank=128)
    b = gp.posterior_mean(X, y, Xt, probes=4, precond_rank=128)
    assert torch.equal(a.mean, b.mean)
