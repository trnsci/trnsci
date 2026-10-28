#!/usr/bin/env python
"""Leak soak: repeat each library's public op many times; device memory must not grow.

Each op runs SOAK_ITERS times (default 2000) with set_backend("nki"). Device
memory (torch_neuronx.memory_allocated) is sampled every SOAK_EVERY iterations;
growth from the first sample after warm-up to the last is reported. A steady
climb means a per-call leak (buffers kept alive by caches, autograd, or the
dispatch layer).

    python scripts/spike_soak.py            # native device
    python scripts/spike_soak.py --cpu      # pre-flight: pytorch backend, no memory stats
"""

import os
import sys
import time
import traceback

import torch

CPU = "--cpu" in sys.argv or os.environ.get("SPIKE_DEVICE") == "cpu"
ITERS = int(os.environ.get("SOAK_ITERS", "200" if CPU else "2000"))
EVERY = int(os.environ.get("SOAK_EVERY", "50" if CPU else "200"))
LEAK_BYTES = int(os.environ.get("SOAK_LEAK_BYTES", str(8 << 20)))  # 8 MiB


def mem():
    if CPU:
        return 0
    import torch_neuronx

    torch_neuronx.synchronize()
    return torch_neuronx.memory_allocated()


def ops():
    import trnblas
    import trnfft
    import trnrand
    import trntensor

    be = "pytorch" if CPU else "nki"
    for lib in (trnblas, trnfft, trnrand, trntensor):
        lib.set_backend(be)
    torch.manual_seed(0)
    A, B = torch.randn(256, 256), torch.randn(256, 512)
    x = torch.randn(8, 1024)
    T1, T2 = torch.randn(128, 64), torch.randn(64, 128)
    return {
        "trnblas.gemm": lambda: trnblas.gemm(1.0, A, B),
        "trnfft.fft": lambda: trnfft.fft(x),
        "trnrand.normal": lambda: trnrand.normal(65536),
        "trntensor.einsum": lambda: trntensor.einsum("ij,jk->ik", T1, T2),
    }


def soak(name, fn):
    samples = []
    t0 = time.perf_counter()
    for i in range(1, ITERS + 1):
        fn()
        if i % EVERY == 0:
            samples.append(mem())
    secs = time.perf_counter() - t0
    growth = samples[-1] - samples[0] if len(samples) > 1 else 0
    ok = CPU or growth < LEAK_BYTES
    print(
        f"RESULT [{name}]: {'PASS' if ok else 'FAIL'}  iters={ITERS}  "
        f"{secs / ITERS * 1e3:.2f} ms/iter  mem first={samples[0] if samples else 0}  "
        f"last={samples[-1] if samples else 0}  growth={growth} bytes",
        flush=True,
    )
    return ok


if __name__ == "__main__":
    results = {}
    for name, fn in ops().items():
        try:
            results[name] = "PASS" if soak(name, fn) else "FAIL"
        except Exception as e:  # noqa: BLE001 - report and continue
            results[name] = "ERROR"
            print(
                f"RESULT [{name}]: ERROR {type(e).__name__}: {str(e).splitlines()[0][:200]}"
            )
            traceback.print_exc(limit=2)
    print("\n===== SUMMARY =====")
    for k, v in results.items():
        print(f"  {v:5}  {k}")
