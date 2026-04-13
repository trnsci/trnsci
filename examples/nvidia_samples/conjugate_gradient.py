"""Port of NVIDIA cuda-samples conjugateGradient to trnsci.

Upstream sample:
    https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/conjugateGradient

The upstream sample solves A x = b with the conjugate-gradient method, where A
is a sparse symmetric positive-definite matrix in CSR format. The canonical
test problem is the 2D Laplacian on a uniform grid with Dirichlet boundary
conditions. This port does the same using trnsparse for the CSR storage and
SpMV, and trnsolver.cg for the iteration.

Run:
    python examples/nvidia_samples/conjugate_gradient.py --demo
"""

from __future__ import annotations

import argparse

import torch

import trnsolver
import trnsparse


def build_2d_laplacian(grid: int) -> tuple[torch.Tensor, int]:
    """Construct the 5-point 2D Laplacian on a grid x grid mesh as a dense
    SPD matrix, then hand it to trnsparse as CSR.

    Returns (A_csr, n) where n = grid * grid.
    """
    n = grid * grid
    A = torch.zeros(n, n, dtype=torch.float32)
    for i in range(grid):
        for j in range(grid):
            idx = i * grid + j
            A[idx, idx] = 4.0
            if i > 0:
                A[idx, idx - grid] = -1.0
            if i < grid - 1:
                A[idx, idx + grid] = -1.0
            if j > 0:
                A[idx, idx - 1] = -1.0
            if j < grid - 1:
                A[idx, idx + 1] = -1.0
    A_csr = trnsparse.CSRMatrix.from_dense(A)
    return A_csr, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--grid", type=int, default=16, help="grid size (n = grid*grid unknowns)"
    )
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=500)
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    A_csr, n = build_2d_laplacian(args.grid)
    torch.manual_seed(0)
    b = torch.randn(n, dtype=torch.float32)

    # trnsolver.cg expects a matvec callable when given a sparse operator.
    def matvec(x: torch.Tensor) -> torch.Tensor:
        return trnsparse.spmv(A_csr, x)

    x, info = trnsolver.cg(matvec, b, tol=args.tol, max_iter=args.max_iter)

    # Residual check
    r = b - matvec(x)
    res = r.norm().item() / b.norm().item()

    print(f"CG on 2D Laplacian: grid={args.grid}x{args.grid} n={n} nnz={A_csr.nnz}")
    print(f"  iterations:   {info.get('iterations', '?')}")
    print(f"  converged:    {info.get('converged', '?')}")
    print(f"  ||r||/||b||:  {res:.3e}")
    print(f"  status:       {'OK' if res < 10 * args.tol else 'FAIL'}")


if __name__ == "__main__":
    main()
