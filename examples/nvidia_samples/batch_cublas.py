"""Port of NVIDIA cuda-samples Samples/4_CUDA_Libraries/batchCUBLAS to trnsci.

Upstream sample:
    https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/batchCUBLAS

The upstream sample benchmarks batched SGEMM via cublasSgemmBatched, comparing
performance to a reference implementation. This port does the same using
trnblas.batched_gemm, measures wall time, and reports effective FP32 TFLOPS.

Run:
    python examples/nvidia_samples/batch_cublas.py --demo
"""

from __future__ import annotations

import argparse
import time

import torch

import trnblas


def bench(batch: int, m: int, n: int, k: int, iters: int) -> tuple[float, float]:
    A = torch.randn(batch, m, k, dtype=torch.float32)
    B = torch.randn(batch, k, n, dtype=torch.float32)

    # One warmup pass so any backend initialization isn't counted.
    _ = trnblas.batched_gemm(1.0, A, B)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = trnblas.batched_gemm(1.0, A, B)
    dt = (time.perf_counter() - t0) / iters

    flops = 2.0 * batch * m * n * k
    tflops = flops / dt / 1e12
    return dt, tflops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    print(f"batched SGEMM: batch={args.batch} M={args.m} N={args.n} K={args.k}")
    print(f"backend: {trnblas.get_backend()}")
    dt, tflops = bench(args.batch, args.m, args.n, args.k, args.iters)
    print(f"  per-iter:   {dt * 1e3:8.3f} ms")
    print(f"  throughput: {tflops:8.2f} TFLOPS")


if __name__ == "__main__":
    main()
