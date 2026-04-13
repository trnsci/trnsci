"""Port of NVIDIA CUDALibrarySamples/cuSOLVER/syevd to trnsci.

Upstream sample:
    https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSOLVER/syevd

The upstream sample computes all eigenvalues and eigenvectors of a symmetric
dense matrix via cusolverDnSsyevd, then checks A V = V diag(w). This port does
the same using trnsolver.eigh.

Run:
    python examples/nvidia_samples/cusolver_syevd.py --demo
"""

from __future__ import annotations

import argparse

import torch

import trnsolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--n", type=int, default=256)
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    torch.manual_seed(0)
    # Build a symmetric matrix from a random one.
    M = torch.randn(args.n, args.n, dtype=torch.float32)
    A = 0.5 * (M + M.T)

    w, V = trnsolver.eigh(A)

    # Check A V = V diag(w)
    recon = V @ torch.diag(w) @ V.T
    err = (A - recon).abs().max().item()
    orth = (V.T @ V - torch.eye(args.n)).abs().max().item()

    print(f"symmetric eigendecomposition: n={args.n}")
    print(f"backend: {trnsolver.get_backend()}")
    print(f"  max |A - V diag(w) V^T|: {err:.3e}")
    print(f"  max |V^T V - I|:         {orth:.3e}")
    print(f"  smallest eigenvalue:     {w.min().item():+.4f}")
    print(f"  largest eigenvalue:      {w.max().item():+.4f}")
    print(
        f"  status:                  {'OK' if err < 1e-3 and orth < 1e-4 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
