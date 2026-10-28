#!/usr/bin/env python
"""Spike B7/B8 — per-library public-op latency: native NKI vs TorchNeuron aten vs CPU.

B7 (default mode). For each library's headline public op, over a size sweep,
warm medians (``LAT_REPS``, default 20; ``torch_neuronx.synchronize()`` after
every call) of:

  * ``nki_resident``  library op, ``set_backend("nki")``, operands already on
                      the ``neuron`` device, result left on device
  * ``nki_host``      same, including host->device copies of the inputs and
                      the device->host copy of the result
  * ``aten_resident`` the TorchNeuron aten/torch equivalent on ``neuron``
  * ``cpu``           the torch equivalent on host CPU (the fallback)

Ops: trnblas.gemm, trnfft.fft (batch ``LAT_FFT_BATCH`` x N), trnsolver.eigh
(N <= 128), trnsparse.spmm (CSR A NxN at ``LAT_SPMM_DENSITY`` x dense Nx128),
trntensor.einsum ``ij,jk->ik`` (NKI forced at every size via
TRNTENSOR_MIN_NKI_FLOPS*=0), trnrand.normal (n elements). Per op it prints the
first size where NKI beats CPU, resident and with host copies — the NKI-vs-CPU
cutover thresholds per library. Lines: ``RESULT [<op> N=<n>]: OK|ERROR ...``,
``CUTOVER [<op>] resident=... host=...``, then ``SUMMARY``.

    python scripts/spike_op_latency.py 2>&1 | tee ~/b7_latency.log

Size env vars (comma lists): LAT_GEMM_SIZES (128,256,512,1024,2048),
LAT_FFT_SIZES (64,256,1024,4096), LAT_EIGH_SIZES (16,32,64,128),
LAT_SPMM_SIZES (256,512,1024,2048), LAT_EINSUM_SIZES (128,256,512,1024,2048),
LAT_NORMAL_SIZES (16384,262144,1048576,4194304); LAT_OPS selects ops
(gemm,fft,eigh,spmm,einsum,normal).

B8 mode ``--throughput SECONDS``: warm up, then loop a fixed workload (one
trnblas.gemm at ``TP_GEMM_N`` (1024) + one trnfft.fft of ``TP_FFT_BATCH`` (32)
x ``TP_FFT_N`` (1024), resident, synchronized every iteration) for SECONDS and
print exactly one line ``THROUGHPUT ops_per_sec=<x>`` (each gemm and each fft
counts as one op). The harness launches 1, 2, 4 copies on different cores
(NEURON_RT_VISIBLE_CORES) to measure data-parallel scaling:

    python scripts/spike_op_latency.py --throughput 60

CPU dry run: add ``--cpu`` (or SPIKE_DEVICE=cpu); pytorch backends, device cpu.
"""

import argparse
import os
import statistics
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", action="store_true", help="run on CPU with pytorch backends")
ap.add_argument("--throughput", type=float, metavar="SECONDS", default=None)
ARGS = ap.parse_args()
CPU = ARGS.cpu or os.environ.get("SPIKE_DEVICE", "").lower() == "cpu"

if not CPU:
    os.environ.setdefault("TRNTENSOR_MIN_NKI_FLOPS", "0")
    os.environ.setdefault("TRNTENSOR_MIN_NKI_FLOPS_PINNED", "0")
    os.environ.setdefault("TRNTENSOR_REQUIRE_NKI", "1")
    os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")

import torch  # noqa: E402

if CPU:
    DEV = "cpu"

    def sync():
        pass

else:
    import torch_neuronx

    DEV = "neuron"

    def sync():
        torch_neuronx.synchronize()


BACKEND = "pytorch" if CPU else "nki"
REPS = int(os.environ.get("LAT_REPS", "20"))


def ints(name, default):
    return [int(v) for v in os.environ.get(name, default).split(",") if v.strip()]


def first_line(e):
    msg = str(e).strip().splitlines()
    return f"{type(e).__name__}: {msg[0][:160] if msg else ''}"


def warm_median_ms(fn, device_sync=True):
    fn()  # cold call: NEFF compile / cache load
    if device_sync:
        sync()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        if device_sync:
            sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def timed(fn, device_sync=True):
    try:
        return warm_median_ms(fn, device_sync), None
    except Exception as e:
        return None, first_line(e)


def to_host(x):
    # ComplexTensor / tuple results -> host
    if torch.is_tensor(x):
        return x.cpu()
    if isinstance(x, tuple):
        return tuple(to_host(t) for t in x)
    return (x.real.cpu(), x.imag.cpu())


# ---------------------------------------------------------------- op builders
# Each builder(n) returns dict of zero-arg callables: nki_resident, nki_host,
# aten_resident, cpu.  Inputs are created outside the timed region.
def b_gemm(n):
    import trnblas

    A, B = torch.randn(n, n), torch.randn(n, n)
    Ad, Bd = A.to(DEV), B.to(DEV)
    return {
        "nki_resident": lambda: trnblas.gemm(1.0, Ad, Bd),
        "nki_host": lambda: trnblas.gemm(1.0, A.to(DEV), B.to(DEV)).cpu(),
        "aten_resident": lambda: Ad @ Bd,
        "cpu": lambda: A @ B,
    }


def b_fft(n):
    import trnfft

    bsz = int(os.environ.get("LAT_FFT_BATCH", "32"))
    xr, xi = torch.randn(bsz, n), torch.randn(bsz, n)
    xd = trnfft.ComplexTensor(xr.to(DEV), xi.to(DEV))
    xc = torch.complex(xr, xi)
    xcd = xc.to(DEV)
    return {
        "nki_resident": lambda: trnfft.fft(xd),
        "nki_host": lambda: to_host(
            trnfft.fft(trnfft.ComplexTensor(xr.to(DEV), xi.to(DEV)))
        ),
        "aten_resident": lambda: torch.fft.fft(xcd),
        "cpu": lambda: torch.fft.fft(xc),
    }


def b_eigh(n):
    import trnsolver

    M = torch.randn(n, n)
    A = M @ M.T + n * torch.eye(n)
    Ad = A.to(DEV)
    return {
        "nki_resident": lambda: trnsolver.eigh(Ad),
        "nki_host": lambda: to_host(trnsolver.eigh(A.to(DEV))),
        "aten_resident": lambda: torch.linalg.eigh(Ad),
        "cpu": lambda: torch.linalg.eigh(A),
    }


def b_spmm(n):
    import trnsparse

    dens = float(os.environ.get("LAT_SPMM_DENSITY", "0.05"))
    D = torch.randn(n, n) * (torch.rand(n, n) < dens)
    B = torch.randn(n, 128)
    S = trnsparse.from_dense(D)
    Sd = trnsparse.CSRMatrix(
        S.values.to(DEV), S.col_indices.to(DEV), S.row_ptrs.to(DEV), S.shape
    )
    Bd, Dd = B.to(DEV), D.to(DEV)
    Scpu = D.to_sparse_csr()

    def host():
        s = trnsparse.CSRMatrix(
            S.values.to(DEV), S.col_indices.to(DEV), S.row_ptrs.to(DEV), S.shape
        )
        return trnsparse.spmm(s, B.to(DEV)).cpu()

    return {
        "nki_resident": lambda: trnsparse.spmm(Sd, Bd),
        "nki_host": host,
        "aten_resident": lambda: Dd @ Bd,  # dense matmul (no sparse aten on neuron)
        "cpu": lambda: torch.sparse.mm(Scpu, B),
    }


def b_einsum(n):
    import trntensor

    A, B = torch.randn(n, n), torch.randn(n, n)
    Ad, Bd = A.to(DEV), B.to(DEV)
    return {
        "nki_resident": lambda: trntensor.einsum("ij,jk->ik", Ad, Bd),
        "nki_host": lambda: trntensor.einsum("ij,jk->ik", A.to(DEV), B.to(DEV)).cpu(),
        "aten_resident": lambda: torch.einsum("ij,jk->ik", Ad, Bd),
        "cpu": lambda: torch.einsum("ij,jk->ik", A, B),
    }


def b_normal(n):
    import trnrand

    def res():
        out = trnrand.normal(n)
        return out if out.device.type == torch.device(DEV).type else out.to(DEV)

    return {
        "nki_resident": res,
        "nki_host": lambda: trnrand.normal(n).cpu(),
        "aten_resident": lambda: torch.randn(n, device=DEV),
        "cpu": lambda: torch.randn(n),
    }


OPS = {
    "gemm": ("trnblas", b_gemm, "LAT_GEMM_SIZES", "128,256,512,1024,2048"),
    "fft": ("trnfft", b_fft, "LAT_FFT_SIZES", "64,256,1024,4096"),
    "eigh": ("trnsolver", b_eigh, "LAT_EIGH_SIZES", "16,32,64,128"),
    "spmm": ("trnsparse", b_spmm, "LAT_SPMM_SIZES", "256,512,1024,2048"),
    "einsum": ("trntensor", b_einsum, "LAT_EINSUM_SIZES", "128,256,512,1024,2048"),
    "normal": ("trnrand", b_normal, "LAT_NORMAL_SIZES", "16384,262144,1048576,4194304"),
}
COLS = ("nki_resident", "nki_host", "aten_resident", "cpu")


def set_backend(libname):
    import importlib

    mod = importlib.import_module(libname)
    mod.set_backend(BACKEND)
    return getattr(mod, "__version__", "?")


def latency_mode():
    selected = os.environ.get("LAT_OPS", ",".join(OPS)).split(",")
    table = {}
    cutovers = {}
    for op in selected:
        lib, builder, env, default = OPS[op]
        print(f"\n===== {op} ({lib}) =====", flush=True)
        try:
            print(f"{lib} {set_backend(lib)} backend={BACKEND}")
        except Exception as e:
            print(f"RESULT [{op} setup]: ERROR {first_line(e)}")
            continue
        rows = []
        for n in ints(env, default):
            try:
                fns = builder(n)
            except Exception as e:
                print(f"RESULT [{op} N={n}]: ERROR build: {first_line(e)}", flush=True)
                continue
            row, errs = {"N": n}, {}
            for c in COLS:
                row[c], err = timed(fns[c], device_sync=(c != "cpu"))
                if err:
                    errs[c] = err
            rows.append(row)
            cells = " ".join(
                f"{c}={row[c]:.3f}ms" if row[c] is not None else f"{c}=ERR"
                for c in COLS
            )
            status = "ERROR" if row["nki_resident"] is None else "OK"
            errtxt = "".join(f" | {c}: {e}" for c, e in errs.items())
            print(f"RESULT [{op} N={n}]: {status} {cells}{errtxt}", flush=True)
        table[op] = rows
        cut = {}
        for kind in ("nki_resident", "nki_host"):
            ok = [r for r in rows if r[kind] is not None and r["cpu"] is not None]
            wins = [r["N"] for r in ok if r[kind] < r["cpu"]]
            # sustained: smallest N from which NKI wins at every larger measured size
            sustained = None
            for r in reversed(ok):
                if r[kind] < r["cpu"]:
                    sustained = r["N"]
                else:
                    break
            if not ok:
                cut[kind] = "n/a(all ERROR)"
            elif not wins:
                cut[kind] = f">N={ok[-1]['N']}"
            else:
                cut[kind] = f"N={wins[0]}" + (
                    ""
                    if sustained == wins[0]
                    else f"(sustained:{'N=' + str(sustained) if sustained else 'no'})"
                )
        cutovers[op] = cut
        print(
            f"CUTOVER [{op}] resident={cut['nki_resident']} host={cut['nki_host']}",
            flush=True,
        )

    print(f"\nSUMMARY  device={DEV} backend={BACKEND} reps={REPS} (warm median ms)")
    for op, rows in table.items():
        print(f"\n{op}:")
        print(f"{'N':>9} " + " ".join(f"{c:>14}" for c in COLS))
        for r in rows:
            print(
                f"{r['N']:>9} "
                + " ".join(
                    f"{r[c]:>14.3f}" if r[c] is not None else f"{'ERR':>14}"
                    for c in COLS
                )
            )
    print("\nNKI-vs-CPU cutover (first size where NKI is faster):")
    for op, cut in cutovers.items():
        print(
            f"  {op:7} resident={cut['nki_resident']:<26} with_host_copies={cut['nki_host']}"
        )


def throughput_mode(seconds):
    import trnblas
    import trnfft

    trnblas.set_backend(BACKEND)
    trnfft.set_backend(BACKEND)
    gn = int(os.environ.get("TP_GEMM_N", "1024"))
    fn_ = int(os.environ.get("TP_FFT_N", "1024"))
    fb = int(os.environ.get("TP_FFT_BATCH", "32"))
    Ad, Bd = torch.randn(gn, gn).to(DEV), torch.randn(gn, gn).to(DEV)
    xd = trnfft.ComplexTensor(
        torch.randn(fb, fn_).to(DEV), torch.randn(fb, fn_).to(DEV)
    )

    def step():
        trnblas.gemm(1.0, Ad, Bd)
        trnfft.fft(xd)
        sync()

    for _ in range(3):  # compile / cache load
        step()
    cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "unset")
    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        step()
        iters += 1
    dt = time.perf_counter() - t0
    print(
        f"throughput detail: cores={cores} device={DEV} gemm_n={gn} fft={fb}x{fn_} "
        f"iters={iters} elapsed={dt:.2f}s iter_ms={dt / max(iters, 1) * 1e3:.3f}",
        file=sys.stderr,
    )
    print(f"THROUGHPUT ops_per_sec={2 * iters / dt:.3f}", flush=True)


if __name__ == "__main__":
    print(
        f"python {sys.version.split()[0]} torch {torch.__version__} device={DEV}",
        file=sys.stderr,
    )
    if not CPU:
        print("torch_neuronx", torch_neuronx.__version__, file=sys.stderr)
    if ARGS.throughput is not None:
        throughput_mode(ARGS.throughput)
    else:
        latency_mode()
