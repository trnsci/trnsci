#!/usr/bin/env python
"""Spike B — per-dispatch overhead on public TorchNeuron (SDK 2.33).

Two suite designs exist because of the XLA path's ~100 ms warm dispatch floor:
trnblas's batched-pair energy kernel and trntensor's homogeneous batching plus
its ``_MIN_NKI_FLOPS`` / ``_MIN_NKI_FLOPS_PINNED`` NKI-vs-CPU cutover. This spike
re-measures the floor on the native stack so those thresholds can be re-derived.

For square fp32 matmuls N in {128, 512, 1024, 2048} it times (warm medians):
  * nki_op(trntensor matmul_kernel), eager, operands resident on device
  * same, including host->device and device->host copies (the "unpinned" case)
  * same op under torch.compile(backend="neuron")
  * TorchNeuron's own aten matmul (A @ B on device) — native-op comparator
  * torch.matmul on host CPU — the fallback the cost model compares against

Requires Trn2/Trn3 + torch 2.14 + torch-neuronx 2.14.* + trntensor installed.

    python scripts/spike_torchneuron_dispatch_overhead.py 2>&1 | tee ~/spike_overhead.log
"""

import os

os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")

import statistics
import time

import torch
import torch_neuronx
from torch_neuronx import nki_op

from trntensor.nki._kernels import matmul_kernel

DEV = "neuron"
# SPIKEB_SIZES=128,512,... overrides the default sweep.
SIZES = tuple(
    int(n) for n in os.environ.get("SPIKEB_SIZES", "128,512,1024,2048").split(",")
)
REPS = 30


@nki_op("trntensor_spike::mm", mutates_args={})
def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return matmul_kernel(a, b)


def warm_median_ms(fn, sync=True):
    fn()  # cold call: NEFF compile / cache load
    if sync:
        torch_neuronx.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch_neuronx.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


if __name__ == "__main__":
    print("torch", torch.__version__, "| torch_neuronx", torch_neuronx.__version__)

    rows = []
    for n in SIZES:
        A, B = torch.randn(n, n), torch.randn(n, n)
        Ad, Bd = A.to(DEV), B.to(DEV)
        # Static shapes: TorchNeuron's compile backend doesn't support dynamic
        # (symbolic) shapes, and a second size would otherwise retrace dynamically.
        compiled = torch.compile(
            lambda a, b: mm(a, b), backend="neuron", fullgraph=True, dynamic=False
        )
        row = {
            "N": n,
            "nki_resident": warm_median_ms(lambda: mm(Ad, Bd)),
            "nki_host_roundtrip": warm_median_ms(
                lambda: mm(A.to(DEV), B.to(DEV)).cpu()
            ),
            "nki_compiled": warm_median_ms(lambda: compiled(Ad, Bd)),
            "aten_resident": warm_median_ms(lambda: Ad @ Bd),
            "cpu": warm_median_ms(lambda: A @ B, sync=False),
        }
        rows.append(row)
        print(f"N={n} done")

    cols = [
        "nki_resident",
        "nki_host_roundtrip",
        "nki_compiled",
        "aten_resident",
        "cpu",
    ]
    print("\nwarm median ms (n=%d)" % REPS)
    print(f"{'N':>6} {'GFLOP':>8} " + " ".join(f"{c:>19}" for c in cols))
    for r in rows:
        gflop = 2 * r["N"] ** 3 / 1e9
        print(f"{r['N']:>6} {gflop:>8.3f} " + " ".join(f"{r[c]:>19.3f}" for c in cols))

    floor = rows[0]["nki_resident"]
    print(
        f"\nnative eager dispatch floor (N=128, resident) ~ {floor:.3f} ms  (XLA path: ~100 ms)"
    )
    for kind in ("nki_resident", "nki_host_roundtrip"):
        cross = next((r for r in rows if r[kind] < r["cpu"]), None)
        where = (
            f"N={cross['N']} ({2 * cross['N'] ** 3 / 1e9:.3f} GFLOP)"
            if cross
            else f"> N={SIZES[-1]}"
        )
        print(f"first size where {kind} beats CPU: {where}")
    print(
        "\nIf the floor is << 10 ms, re-derive trntensor _MIN_NKI_FLOPS (host round-trip "
        "crossover) and _MIN_NKI_FLOPS_PINNED (resident crossover), and revisit trnblas "
        "batched-pair granularity."
    )
