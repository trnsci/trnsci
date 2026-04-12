"""Port of NVIDIA CUDALibrarySamples/cuTENSOR/contraction to trnsci.

Upstream sample:
    https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR/contraction

The upstream sample performs a generic tensor contraction via cutensorContract,
e.g. C[m,u,n,v] = alpha * A[m,h,k,n] * B[u,k,v,h] + beta * C[m,u,n,v], then
checks the result against a reference. This port does the same using
trntensor.einsum and reports the FLOPs estimate from the contraction planner.

Run:
    python examples/nvidia_samples/cutensor_contraction.py --demo
"""

from __future__ import annotations

import argparse

import torch

import trntensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--u", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--v", type=int, default=16)
    parser.add_argument("--h", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    torch.manual_seed(0)
    A = torch.randn(args.m, args.h, args.k, args.n, dtype=torch.float32)
    B = torch.randn(args.u, args.k, args.v, args.h, dtype=torch.float32)

    expr = "mhkn,ukvh->munv"

    # trnsci path
    C = trntensor.einsum(expr, A, B)
    plan = trntensor.plan_contraction(expr, A, B)
    flops = trntensor.estimate_flops(expr, A, B)

    # Reference path
    C_ref = torch.einsum(expr, A, B)

    err = (C - C_ref).abs().max().item()
    rel = err / C_ref.abs().max().item()

    print(f"contraction: {expr}")
    print(f"  A shape: {tuple(A.shape)}  B shape: {tuple(B.shape)}  C shape: {tuple(C.shape)}")
    print(f"  dispatch:   {getattr(plan, 'dispatch', '?')}")
    print(f"  FLOPs:      {flops:,}")
    print(f"  max error:  {err:.3e}  (rel {rel:.3e})")
    print(f"  status:     {'OK' if rel < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()
