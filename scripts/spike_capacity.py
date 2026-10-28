#!/usr/bin/env python
"""Device-memory limits: scale DF-MP2 and FFT until they fail.

Each case runs in its own subprocess with a timeout, so an out-of-memory crash
or a hang can't take down the sweep. Reports, per case, PASS with peak device
memory (torch_neuronx.max_memory_allocated) and wall time, or the first error
line. Phase 4 (model-parallel) sizing input; the roadmap target is DF-MP2 with
>3000 basis functions.

    python scripts/spike_capacity.py                       # native device
    python scripts/spike_capacity.py --cpu                 # pre-flight: tiny sizes, pytorch backend
    CAP_NAO=256,512 CAP_FFT_LOG2=18,20 python scripts/spike_capacity.py
"""

import json
import os
import subprocess
import sys

CPU = "--cpu" in sys.argv or os.environ.get("SPIKE_DEVICE") == "cpu"
NAO = [
    int(n)
    for n in os.environ.get(
        "CAP_NAO", "64" if CPU else "256,512,1024,1536,2048,3072"
    ).split(",")
]
FFT_LOG2 = [
    int(n)
    for n in os.environ.get("CAP_FFT_LOG2", "12" if CPU else "18,20,22,24,26").split(
        ","
    )
]
TIMEOUT = int(os.environ.get("CAP_TIMEOUT", "900"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def case(kind, size):
    """Run one case in this process; print one JSON line."""
    import time

    import torch

    be = "pytorch" if CPU else "nki"
    out = {"kind": kind, "size": size}
    t0 = time.perf_counter()
    if kind == "df_mp2":
        import runpy

        import trnblas
        import trnsolver
        import trntensor

        for lib in (trnblas, trnsolver, trntensor):
            lib.set_backend(be)
        mod = runpy.run_path(
            os.path.join(REPO, "examples/quantum_chemistry/df_mp2_synthetic.py")
        )
        mod["run_demo"](n_ao=size, n_aux=2 * size, n_occ=max(8, size // 8), seed=0)
    else:
        import trnfft

        trnfft.set_backend(be)
        n = 1 << size
        trnfft.fft(torch.randn(n))
    out["secs"] = round(time.perf_counter() - t0, 2)
    if not CPU:
        import torch_neuronx

        torch_neuronx.synchronize()
        out["peak_bytes"] = torch_neuronx.max_memory_allocated()
    print("CASE " + json.dumps(out), flush=True)


def sweep():
    cases = [("df_mp2", n) for n in NAO] + [("fft", k) for k in FFT_LOG2]
    for kind, size in cases:
        label = f"{kind}_{size}" if kind == "df_mp2" else f"fft_2^{size}"
        try:
            p = subprocess.run(
                [sys.executable, __file__, "--case", kind, str(size)]
                + (["--cpu"] if CPU else []),
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"RESULT [{label}]: TIMEOUT after {TIMEOUT}s", flush=True)
            continue
        line = next((x for x in p.stdout.splitlines() if x.startswith("CASE ")), None)
        if p.returncode == 0 and line:
            d = json.loads(line[5:])
            peak = d.get("peak_bytes")
            peak_s = f"  peak={peak / 2**30:.2f} GiB" if peak is not None else ""
            print(f"RESULT [{label}]: PASS  {d['secs']}s{peak_s}", flush=True)
        else:
            err = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1][:220]
            print(f"RESULT [{label}]: FAIL rc={p.returncode}  {err}", flush=True)


if __name__ == "__main__":
    if "--case" in sys.argv:
        i = sys.argv.index("--case")
        case(sys.argv[i + 1], int(sys.argv[i + 2]))
    else:
        sweep()
