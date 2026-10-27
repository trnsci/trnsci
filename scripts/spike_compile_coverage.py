#!/usr/bin/env python
"""Spike B6: which public suite calls survive torch.compile on the native device?

For 2-5 representative public calls per library (all with ``set_backend("nki")``,
inputs on ``device="neuron"``), this script:

  1. runs the call eagerly (the reference),
  2. runs ``torch._dynamo.explain`` (fullgraph=False) to count graphs and graph
     breaks and list the first break reasons,
  3. compiles ``lambda *a: fn(*a)`` with
     ``torch.compile(backend="neuron", fullgraph=True, dynamic=False)``, calls it
     with the same inputs and compares the result to eager.

Status per call:
  OK           compiled with fullgraph=True, matches eager within tolerance
  MISMATCH     compiled, but differs from eager beyond tolerance
  GRAPH_BREAK  dynamo raised Unsupported/UserError etc. under fullgraph=True
               (the reason column is the first line of the message)
  BACKEND_ERR  dynamo traced a full graph, the neuron backend failed to compile it
  EAGER_ERR    the eager reference itself failed, so compile wasn't tried
  ERROR        anything else

Outputs that aren't tensors (trnfft ComplexTensor, trnsolver tuples) are
flattened to a tuple of tensors inside the compiled function. Sparse CSR
operands are rebuilt from their component tensors inside the function.
trnrand calls are reseeded before each run so eager and compiled draw the same
stream.

    python scripts/spike_compile_coverage.py 2>&1 | tee ~/spike_compile_coverage.log

CPU dry run (pytorch backends, device=cpu, backend="eager"):

    python scripts/spike_compile_coverage.py --cpu

Env: COV_N (problem size, default 256), COV_LIBS (comma list to restrict),
SPIKE_DEVICE=cpu (same as --cpu). Post the log (the COVERAGE table) to
trnsci/trnsci#35.
"""

# ruff: noqa: E402  (env vars must be set before the suite and NKI are imported)
import os
import sys

CPU = "--cpu" in sys.argv or os.environ.get("SPIKE_DEVICE", "").lower() == "cpu"
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")

import logging
import math
import time
import traceback

import torch
import torch._dynamo

torch._logging.set_logs(dynamo=logging.ERROR)

if not CPU:
    import torch_neuronx  # noqa: F401

import trnblas
import trnfft
import trnrand
import trnsolver
import trnsparse
import trntensor

DEV = "cpu" if CPU else "neuron"
BACKEND = "pytorch" if CPU else "nki"
COMPILE_BACKEND = "eager" if CPU else "neuron"
N = int(os.environ.get("COV_N", "256"))
LIBS = os.environ.get(
    "COV_LIBS", "trnblas,trnfft,trnsparse,trnsolver,trntensor,trnrand"
).split(",")
TOL = 1e-2

# torch._dynamo exceptions that mean "dynamo could not capture a full graph".
_BREAK_EXC = tuple(
    getattr(torch._dynamo.exc, n)
    for n in (
        "Unsupported",
        "UserError",
        "UncapturedHigherOrderOpError",
        "ObservedException",
    )
    if hasattr(torch._dynamo.exc, n)
)
_BACKEND_EXC = tuple(
    getattr(torch._dynamo.exc, n)
    for n in ("BackendCompilerFailed",)
    if hasattr(torch._dynamo.exc, n)
)

ROWS = []  # (lib, call, status, breaks, reason)


def sync():
    if not CPU:
        import torch_neuronx

        torch_neuronx.synchronize()


def first_line(e):
    lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
    return f"{type(e).__name__}: {lines[0][:160] if lines else ''}"


def flat(out):
    """Flatten an output to a tuple of tensors (ComplexTensor, tuples, scalars)."""
    if isinstance(out, trnfft.ComplexTensor):
        return (out.real, out.imag)
    if isinstance(out, torch.Tensor):
        return (out,)
    if isinstance(out, (tuple, list)):
        res = ()
        for o in out:
            res += flat(o)
        return res
    if isinstance(out, (int, float)):
        return (torch.tensor(float(out)),)
    raise TypeError(f"unflattenable output type {type(out).__name__}")


def max_rel(a, b):
    worst = 0.0
    for x, y in zip(a, b):
        x, y = x.detach().cpu().double(), y.detach().cpu().double()
        if x.shape != y.shape:
            return math.inf
        den = y.norm().item()
        worst = max(worst, (x - y).norm().item() / (den if den > 0 else 1.0))
    return worst


def r(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(DEV)


def spd(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g) / math.sqrt(n)
    return (M @ M.T + 2 * torch.eye(n)).to(DEV)


def csr_parts(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n, n, generator=g) * (torch.rand(n, n, generator=g) < 0.1)
    c = trnsparse.from_dense(D)
    return c.values.to(DEV), c.col_indices.to(DEV), c.row_ptrs.to(DEV)


def build_cases():
    n = N
    sym = r(n, n, seed=4)
    sym = (sym + sym.T) / 2
    eig_n = min(n, 128)  # trnsolver NKI eigh guard: n <= 128
    sym_e = sym[:eig_n, :eig_n].contiguous()
    v, ci, rp = csr_parts(n, seed=5)

    def csr(v, ci, rp):
        return trnsparse.CSRMatrix(values=v, col_indices=ci, row_ptrs=rp, shape=(n, n))

    def reseed(fn):
        def g(*a):
            return fn(*a)

        g.reseed = True
        return g

    # (lib, call label, fn, args)
    return [
        (
            "trnblas",
            "gemm(1.0, A, B)",
            lambda A, B: trnblas.gemm(1.0, A, B),
            (r(n, n, seed=1), r(n, n, seed=2)),
        ),
        (
            "trnblas",
            "gemv(1.0, A, x)",
            lambda A, x: trnblas.gemv(1.0, A, x),
            (r(n, n, seed=1), r(n, seed=3)),
        ),
        (
            "trnblas",
            "dot(x, y)",
            lambda x, y: trnblas.dot(x, y),
            (r(n, seed=3), r(n, seed=6)),
        ),
        (
            "trnblas",
            "batched_gemm(1.0, A, B)",
            lambda A, B: trnblas.batched_gemm(1.0, A, B),
            (r(4, n, n, seed=7), r(4, n, n, seed=8)),
        ),
        ("trnfft", "fft(x) n=N", lambda x: flat(trnfft.fft(x)), (r(4, n, seed=9),)),
        ("trnfft", "ifft(x) n=N", lambda x: flat(trnfft.ifft(x)), (r(4, n, seed=9),)),
        ("trnfft", "rfft(x) n=N", lambda x: flat(trnfft.rfft(x)), (r(4, n, seed=9),)),
        (
            "trnfft",
            "fft(x) n=127 (Bluestein)",
            lambda x: flat(trnfft.fft(x)),
            (r(4, 127, seed=10),),
        ),
        (
            "trnfft",
            "complex_matmul(a, b)",
            lambda ar, ai, br, bi: flat(
                trnfft.complex_matmul(
                    trnfft.ComplexTensor(ar, ai), trnfft.ComplexTensor(br, bi)
                )
            ),
            (r(n, n, seed=11), r(n, n, seed=12), r(n, n, seed=13), r(n, n, seed=14)),
        ),
        (
            "trnsparse",
            "spmm(CSR, B)",
            lambda v, ci, rp, B: trnsparse.spmm(csr(v, ci, rp), B),
            (v, ci, rp, r(n, 64, seed=15)),
        ),
        (
            "trnsparse",
            "spmv(CSR, x)",
            lambda v, ci, rp, x: trnsparse.spmv(csr(v, ci, rp), x),
            (v, ci, rp, r(n, seed=16)),
        ),
        (
            "trnsparse",
            "from_dense(D) -> spmm",
            lambda D, B: trnsparse.spmm(trnsparse.from_dense(D), B),
            (r(n, n, seed=17), r(n, 64, seed=18)),
        ),
        (
            "trnsolver",
            "cholesky(A)",
            lambda A: trnsolver.cholesky(A),
            (spd(n, seed=19),),
        ),
        (
            "trnsolver",
            "solve(A, B)",
            lambda A, B: trnsolver.solve(A, B),
            (spd(n, seed=19), r(n, 4, seed=20)),
        ),
        (
            "trnsolver",
            f"eigh(A) n={eig_n}",
            lambda A: flat(trnsolver.eigh(A)[0]),
            (sym_e,),
        ),
        ("trnsolver", "qr(A)", lambda A: flat(trnsolver.qr(A)), (r(n, n, seed=21),)),
        (
            "trnsolver",
            "cg(A, b)",
            lambda A, b: trnsolver.cg(A, b, tol=1e-6, maxiter=200)[0],
            (spd(n, seed=19), r(n, seed=22)),
        ),
        (
            "trntensor",
            "einsum('ij,jk->ik')",
            lambda a, b: trntensor.einsum("ij,jk->ik", a, b),
            (r(n, n, seed=23), r(n, n, seed=24)),
        ),
        (
            "trntensor",
            "einsum('bij,bjk->bik')",
            lambda a, b: trntensor.einsum("bij,bjk->bik", a, b),
            (r(4, n, n, seed=25), r(4, n, n, seed=26)),
        ),
        (
            "trntensor",
            "einsum('ijk,kl->ijl')",
            lambda a, b: trntensor.einsum("ijk,kl->ijl", a, b),
            (r(8, 32, n, seed=27), r(n, 64, seed=28)),
        ),
        ("trnrand", "normal(N*N)", reseed(lambda: trnrand.normal(n * n)), ()),
        ("trnrand", "uniform(N*N)", reseed(lambda: trnrand.uniform(n * n)), ()),
        ("trnrand", "sobol(N, 4)", lambda: trnrand.sobol(n, 4, seed=0), ()),
    ]


def explain(fn, args):
    """Return (graph_count, break_count, first reasons) via torch._dynamo.explain."""
    torch._dynamo.reset()
    if getattr(fn, "reseed", False):
        trnrand.manual_seed(0)
    ex = torch._dynamo.explain(fn)(*args)
    reasons = []
    for br in getattr(ex, "break_reasons", [])[:3]:
        reasons.append(str(getattr(br, "reason", br)).strip().splitlines()[0][:120])
    # explain can report 0 (or -1) breaks when the very first op is unsupported
    # while still listing a reason; count listed reasons as breaks in that case.
    n_breaks = max(ex.graph_break_count, len(getattr(ex, "break_reasons", [])))
    return ex.graph_count, n_breaks, reasons


def cover(lib, label, fn, args):
    print(f"\n===== {lib} :: {label} =====", flush=True)
    breaks = "?"
    try:
        if getattr(fn, "reseed", False):
            trnrand.manual_seed(0)
        ref = flat(fn(*args))
        sync()
    except Exception as e:
        traceback.print_exc(limit=-3)
        return lib, label, "EAGER_ERR", breaks, first_line(e)

    try:
        gc, bc, reasons = explain(fn, args)
        breaks = str(bc)
        print(f"explain: graphs={gc} graph_breaks={bc}")
        for i, rs in enumerate(reasons):
            print(f"  break[{i}]: {rs}")
    except Exception as e:
        print(f"explain failed: {first_line(e)}")

    torch._dynamo.reset()
    compiled = torch.compile(
        lambda *a: fn(*a), backend=COMPILE_BACKEND, fullgraph=True, dynamic=False
    )
    t0 = time.perf_counter()
    try:
        if getattr(fn, "reseed", False):
            trnrand.manual_seed(0)
        got = flat(compiled(*args))
        sync()
    except _BACKEND_EXC as e:
        return lib, label, "BACKEND_ERR", breaks, first_line(e)
    except _BREAK_EXC as e:
        return lib, label, "GRAPH_BREAK", breaks, first_line(e)
    except Exception as e:
        traceback.print_exc(limit=-3)
        return lib, label, "ERROR", breaks, first_line(e)
    dt = time.perf_counter() - t0
    err = max_rel(got, ref)
    status = "OK" if err < TOL else "MISMATCH"
    return (
        lib,
        label,
        status,
        breaks,
        f"rel_err_vs_eager={err:.2e} compile+run {dt:.1f}s",
    )


def main():
    print(
        "torch",
        torch.__version__,
        "| device",
        DEV,
        "| lib backend",
        BACKEND,
        "| compile backend",
        COMPILE_BACKEND,
    )
    print("N", N, "| libs", LIBS)
    for lib in (trnblas, trnfft, trnrand, trnsolver, trnsparse, trntensor):
        lib.set_backend(BACKEND)
    t_start = time.perf_counter()
    for lib, label, fn, args in build_cases():
        if lib not in LIBS:
            continue
        row = cover(lib, label, fn, args)
        ROWS.append(row)
        print(f"RESULT [{lib} {label}]: {row[2]} breaks={row[3]} {row[4]}", flush=True)

    print(
        f"\n===== COVERAGE ===== (fullgraph=True, dynamic=False, backend={COMPILE_BACKEND}, N={N})"
    )
    print(f"{'library':10} {'call':32} {'status':12} {'breaks':>6}  reason / detail")
    for lib, label, status, breaks, reason in ROWS:
        print(f"{lib:10} {label:32} {status:12} {breaks:>6}  {reason}")
    print("\n===== SUMMARY =====")
    for lib in dict.fromkeys(r[0] for r in ROWS):
        st = [r[2] for r in ROWS if r[0] == lib]
        counts = ", ".join(f"{s}={st.count(s)}" for s in dict.fromkeys(st))
        print(f"  {lib:10} {len(st)} calls: {counts}")
    print(f"total {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
