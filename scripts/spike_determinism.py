#!/usr/bin/env python
"""Spike B1 — structural determinism of the suite's NKI kernels on Trn2.

Under round-to-nearest (``NEURON_RT_STOCHASTIC_ROUNDING_EN=0``, set here by
default) every NKI kernel in the suite should be bit-reproducible: the same
inputs give the same output bits run-to-run in one process, and across
NeuronCores. This script runs 2-4 representative public ops per library with
fixed seeds and fixed inputs, every library on ``set_backend("nki")``.

Modes:

    # run every op twice in-process (run1 == run2 bitwise?) and save outputs
    NEURON_RT_VISIBLE_CORES=0 python scripts/spike_determinism.py --save ~/det/core0
    NEURON_RT_VISIBLE_CORES=2 python scripts/spike_determinism.py --save ~/det/core2

    # cross-core comparison: bitwise-equal / max-abs-diff per op
    python scripts/spike_determinism.py --compare ~/det/core0 ~/det/core2

    # CPU pre-flight (every library on set_backend("pytorch"))
    python scripts/spike_determinism.py --cpu --save /tmp/det_cpu

Each op is isolated: an op that errors is recorded as ERROR (first line of the
message) and the script continues. Output lines are greppable:
``RESULT [<op>]: PASS|FAIL|ERROR ...`` followed by a ``SUMMARY`` block.

Env knobs: ``SPIKE_DET_N`` (base matrix size, default 256), ``SPIKE_DET_OPS``
(comma-separated op-name filter), ``SPIKE_DEVICE=cpu`` (same as ``--cpu``).

Post both ``--save`` logs and the ``--compare`` log to trnsci/trnsci#35.
"""

import os

# Determinism is a round-to-nearest property: pin RNE unless the caller overrides.
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
# Editable installs have no .dist-info, so the NKI trace cache can't version them.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")

import argparse
import importlib
import socket
import sys
import time
import traceback
from pathlib import Path

import torch

LIBS = ("trnblas", "trnfft", "trnsolver", "trnsparse", "trntensor", "trnrand")
N = int(os.environ.get("SPIKE_DET_N", "256"))
OP_FILTER = {s for s in os.environ.get("SPIKE_DET_OPS", "").split(",") if s}


def first_line(exc):
    msg = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {msg[0] if msg else ''}"[:300]


# ------------------------------------------------------------------ output flattening
def flatten(obj, prefix="out"):
    """Turn an op result into {name: cpu tensor}. Handles ComplexTensor, tuples, dicts."""
    out = {}
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        if t.is_complex():
            out[prefix + ".real"] = t.real.contiguous()
            out[prefix + ".imag"] = t.imag.contiguous()
        else:
            out[prefix] = t
    elif (
        hasattr(obj, "real")
        and hasattr(obj, "imag")
        and not isinstance(obj, (int, float))
    ):
        out.update(flatten(obj.real, prefix + ".real"))
        out.update(flatten(obj.imag, prefix + ".imag"))
    elif isinstance(obj, (tuple, list)):
        for i, o in enumerate(obj):
            out.update(flatten(o, f"{prefix}[{i}]"))
    elif isinstance(obj, dict):
        for k, o in obj.items():
            out.update(flatten(o, f"{prefix}.{k}"))
    elif isinstance(obj, (int, float, bool)):
        out[prefix] = torch.tensor(obj, dtype=torch.float64)
    elif obj is None:
        pass
    else:
        out[prefix] = torch.tensor(float(obj), dtype=torch.float64)
    return out


def bit_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    a = a.contiguous().reshape(-1)
    b = b.contiguous().reshape(-1)
    if a.dtype == torch.bool:
        return torch.equal(a, b)
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def max_abs_diff(a, b):
    if a.shape != b.shape:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    d = (a.double() - b.double()).abs()
    d = torch.where(
        torch.isnan(a.double()) & torch.isnan(b.double()), torch.zeros_like(d), d
    )
    return float(d.max().item())


def compare_dicts(da, db):
    """Return (all_bit_equal, max_abs_diff, detail)."""
    keys = sorted(set(da) | set(db))
    eq, worst, detail = True, 0.0, []
    for k in keys:
        if k not in da or k not in db:
            eq, worst = False, float("inf")
            detail.append(f"{k}:missing")
            continue
        e = bit_equal(da[k], db[k])
        m = max_abs_diff(da[k], db[k])
        eq = eq and e
        worst = max(worst, m)
        if not e:
            n_diff = (
                int((da[k].double() != db[k].double()).sum().item())
                if da[k].shape == db[k].shape
                else -1
            )
            detail.append(f"{k}:ndiff={n_diff}/{da[k].numel()},max={m:.3e}")
    return eq, worst, ";".join(detail)


# ------------------------------------------------------------------ op definitions
def _gen(seed):
    g = torch.Generator().manual_seed(seed)
    return g


def _randn(*shape, seed):
    return torch.randn(*shape, generator=_gen(seed), dtype=torch.float32)


def _spd(n, seed):
    a = _randn(n, n, seed=seed)
    return (a @ a.T) / n + torch.eye(n)


def build_ops():
    """Return list of (name, fn). Inputs are built once here (fixed seeds)."""
    ops = []

    # trnblas
    def trnblas_ops():
        import trnblas

        A, B = _randn(N, N, seed=1), _randn(N, N, seed=2)
        C = _randn(N, N, seed=3)
        Ab, Bb = _randn(4, 128, 128, seed=4), _randn(4, 128, 128, seed=5)
        S = _randn(N, 128, seed=6)
        return [
            ("trnblas.gemm", lambda: trnblas.gemm(1.0, A, B)),
            ("trnblas.gemm_beta", lambda: trnblas.gemm(0.5, A, B, beta=2.0, C=C)),
            ("trnblas.batched_gemm", lambda: trnblas.batched_gemm(1.0, Ab, Bb)),
            ("trnblas.syrk", lambda: trnblas.syrk(1.0, S)),
        ]

    # trnfft
    def trnfft_ops():
        import trnfft

        x = _randn(8, 1024, seed=11)
        xr = _randn(4, 2048, seed=12)
        sig = _randn(16000, seed=13)
        win = torch.hann_window(512)
        return [
            ("trnfft.fft", lambda: trnfft.fft(x)),
            ("trnfft.rfft", lambda: trnfft.rfft(xr)),
            (
                "trnfft.stft",
                lambda: trnfft.stft(sig, n_fft=512, hop_length=128, window=win),
            ),
        ]

    # trnsolver
    def trnsolver_ops():
        import trnsolver

        H = _spd(128, seed=21)
        P = _spd(N, seed=22)
        b = _randn(N, seed=23)
        return [
            ("trnsolver.eigh", lambda: trnsolver.eigh(H)),
            ("trnsolver.cholesky", lambda: trnsolver.cholesky(P)),
            ("trnsolver.cg", lambda: trnsolver.cg(P, b, tol=1e-6, maxiter=200)),
        ]

    # trnsparse
    def trnsparse_ops():
        import trnsparse

        n = max(N, 256)
        d = _randn(n, n, seed=31)
        mask = torch.rand(n, n, generator=_gen(32)) < 0.05
        csr = trnsparse.from_dense(d * mask)
        Bd = _randn(n, 128, seed=33)
        xv = _randn(n, seed=34)
        # block-sparse: keep ~half of the 128x128 blocks
        nb = n // 128
        bmask = torch.rand(nb, nb, generator=_gen(35)) < 0.5
        bmask.fill_diagonal_(True)
        dense_b = d * bmask.repeat_interleave(128, 0).repeat_interleave(128, 1)
        bsr = trnsparse.BSRMatrix.from_dense(dense_b, block_size=128)
        return [
            ("trnsparse.spmm", lambda: trnsparse.spmm(csr, Bd)),
            ("trnsparse.spmv", lambda: trnsparse.spmv(csr, xv)),
            ("trnsparse.bsr_spmm", lambda: trnsparse.bsr_spmm(bsr, Bd)),
        ]

    # trntensor
    def trntensor_ops():
        import trntensor

        A, B = _randn(N, N, seed=41), _randn(N, 512, seed=42)
        X, Y = _randn(4, 128, 256, seed=43), _randn(4, 256, 128, seed=44)
        nocc, nvir, naux = 8, 32, 64
        Bq = _randn(nocc, nvir, naux, seed=45) * 0.1
        eo = -1.0 - torch.rand(nocc, generator=_gen(46))
        ev = 0.5 + torch.rand(nvir, generator=_gen(47))
        return [
            ("trntensor.einsum_ij_jk", lambda: trntensor.einsum("ij,jk->ik", A, B)),
            (
                "trntensor.einsum_bij_bjk",
                lambda: trntensor.einsum("bij,bjk->bik", X, Y),
            ),
            ("trntensor.mp2_energy", lambda: trntensor.mp2_energy(Bq, eo, ev)),
        ]

    # trnrand (fresh Generator per call -> same seed each run)
    def trnrand_ops():
        import trnrand

        return [
            (
                "trnrand.uniform",
                lambda: trnrand.uniform(65536, generator=trnrand.Generator(seed=1234)),
            ),
            (
                "trnrand.normal",
                lambda: trnrand.normal(65536, generator=trnrand.Generator(seed=1234)),
            ),
            ("trnrand.sobol", lambda: trnrand.sobol(4096, 6, scramble=False)),
            ("trnrand.halton", lambda: trnrand.halton(4096, 5)),
        ]

    for lib, builder in (
        ("trnblas", trnblas_ops),
        ("trnfft", trnfft_ops),
        ("trnsolver", trnsolver_ops),
        ("trnsparse", trnsparse_ops),
        ("trntensor", trntensor_ops),
        ("trnrand", trnrand_ops),
    ):
        try:
            ops.extend(builder())
        except Exception as e:
            ops.append((f"{lib}.<setup>", _raiser(e)))
    if OP_FILTER:
        ops = [(n, f) for n, f in ops if n in OP_FILTER or n.split(".")[0] in OP_FILTER]
    return ops


def _raiser(exc):
    def f():
        raise exc

    return f


# ------------------------------------------------------------------ setup
def setup_backends(cpu):
    backend = "pytorch" if cpu else "nki"
    status = {}
    if not cpu:
        try:
            import torch_neuronx  # noqa: F401

            status["torch_neuronx"] = getattr(torch_neuronx, "__version__", "?")
        except Exception as e:
            status["torch_neuronx"] = "IMPORT-ERROR " + first_line(e)
    for name in LIBS:
        try:
            lib = importlib.import_module(name)
            lib.set_backend(backend)
            status[name] = (
                f"{getattr(lib, '__version__', '?')} backend={lib.get_backend()}"
            )
        except Exception as e:
            status[name] = "ERROR " + first_line(e)
    try:
        from trnrand.nki import dispatch as rd

        if hasattr(rd, "_get_dispatch_path"):
            status["trnrand.dispatch_path"] = rd._get_dispatch_path()
    except Exception as e:
        status["trnrand.dispatch_path"] = "ERROR " + first_line(e)
    return backend, status


def sync(cpu):
    if cpu:
        return
    try:
        import torch_neuronx

        torch_neuronx.synchronize()
    except Exception:
        pass


# ------------------------------------------------------------------ modes
def do_save(outdir, cpu):
    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    backend, status = setup_backends(cpu)
    core = os.environ.get("NEURON_RT_VISIBLE_CORES", "(unset)")
    meta = {
        "core": core,
        "host": socket.gethostname(),
        "backend": backend,
        "device": "cpu" if cpu else "neuron",
        "torch": torch.__version__,
        "sr_env": os.environ.get("NEURON_RT_STOCHASTIC_ROUNDING_EN"),
        "libs": status,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("META", meta, flush=True)
    results = {}
    for name, fn in build_ops():
        print(f"\n===== {name} =====", flush=True)
        rec = {"op": name, "meta": meta}
        try:
            t0 = time.perf_counter()
            r1 = flatten(fn())
            sync(cpu)
            t1 = time.perf_counter()
            r2 = flatten(fn())
            sync(cpu)
            t2 = time.perf_counter()
            eq, worst, detail = compare_dicts(r1, r2)
            rec.update(status="OK", outputs=r1, run2_equal=eq, run2_maxdiff=worst)
            finite = all(
                bool(torch.isfinite(v.double()).all()) for v in r1.values() if v.numel()
            )
            verdict = "PASS" if eq else "FAIL"
            results[name] = verdict
            shapes = ",".join(
                f"{k}{tuple(v.shape)}:{str(v.dtype)[6:]}" for k, v in r1.items()
            )
            print(
                f"RESULT [{name}]: {verdict} in-process run1==run2 bitwise={eq} "
                f"max_abs_diff={worst:.3e} finite={finite} "
                f"t1={t1 - t0:.2f}s t2={t2 - t1:.2f}s outs={shapes} {detail}",
                flush=True,
            )
        except Exception as e:
            traceback.print_exc(limit=3)
            msg = first_line(e)
            rec.update(status="ERROR", error=msg)
            results[name] = "ERROR"
            print(f"RESULT [{name}]: ERROR {msg}", flush=True)
        try:
            torch.save(rec, outdir / f"{name}.pt")
        except Exception as e:
            print(f"  (save failed: {first_line(e)})", flush=True)
    summary(results, f"save core={core} backend={backend} dir={outdir}")


def do_compare(dir_a, dir_b):
    da, db = Path(dir_a).expanduser(), Path(dir_b).expanduser()
    names = sorted(
        {p.stem for p in da.glob("*.pt")} | {p.stem for p in db.glob("*.pt")}
    )
    results = {}
    meta_printed = False
    for name in names:
        try:
            pa, pb = da / f"{name}.pt", db / f"{name}.pt"
            if not pa.exists() or not pb.exists():
                results[name] = "ERROR"
                print(
                    f"RESULT [{name}]: ERROR missing in {'A' if not pa.exists() else 'B'}"
                )
                continue
            ra = torch.load(pa, weights_only=False)
            rb = torch.load(pb, weights_only=False)
            if not meta_printed:
                print(
                    "META A", ra["meta"].get("core"), ra["meta"].get("host"), ra["meta"]
                )
                print(
                    "META B", rb["meta"].get("core"), rb["meta"].get("host"), rb["meta"]
                )
                meta_printed = True
            if ra.get("status") != "OK" or rb.get("status") != "OK":
                results[name] = "ERROR"
                print(
                    f"RESULT [{name}]: ERROR A={ra.get('status')} {ra.get('error', '')} | "
                    f"B={rb.get('status')} {rb.get('error', '')}"
                )
                continue
            eq, worst, detail = compare_dicts(ra["outputs"], rb["outputs"])
            verdict = "PASS" if eq else "FAIL"
            results[name] = verdict
            print(
                f"RESULT [{name}]: {verdict} cross-run bitwise={eq} max_abs_diff={worst:.3e} "
                f"A(core={ra['meta']['core']},in-proc-eq={ra['run2_equal']}) "
                f"B(core={rb['meta']['core']},in-proc-eq={rb['run2_equal']}) {detail}"
            )
        except Exception as e:
            results[name] = "ERROR"
            print(f"RESULT [{name}]: ERROR {first_line(e)}")
    summary(results, f"compare A={da} B={db}")


def summary(results, label):
    print("\n========== SUMMARY ==========")
    print(label)
    for k, v in results.items():
        print(f"  {v:5}  {k}")
    counts = {
        s: sum(1 for v in results.values() if v == s) for s in ("PASS", "FAIL", "ERROR")
    }
    print(f"  totals: {counts} of {len(results)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cpu", action="store_true", help="run on CPU with set_backend('pytorch')"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="DIR")
    g.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    args = ap.parse_args()
    cpu = args.cpu or os.environ.get("SPIKE_DEVICE", "neuron").lower() == "cpu"
    t0 = time.time()
    if args.save:
        do_save(args.save, cpu)
    else:
        do_compare(*args.compare)
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
