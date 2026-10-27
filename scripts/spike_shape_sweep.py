#!/usr/bin/env python
"""Spike B3: awkward-size sweep of padding/tiling edge paths on the native device.

Every library is switched to ``set_backend("nki")`` and called on deliberately
awkward sizes (1, primes, tile +/- 1, non-powers of two). Each result is checked
against a float64 host reference (torch on CPU), and one RESULT line is printed
per case:

  * trnblas.gemm             square (n,n,n) plus rectangular (M,K,N) triples
  * trnfft.fft / ifft        1-D lengths; primes / non-powers of 2 take Bluestein
  * trnsparse.spmm           random CSR (about 10 % dense) at (n x n) @ (n x 64)
  * trnsolver.eigh           n <= 128 only (NKI Householder single-tile guard)
  * trnsolver.cg             SPD n x n, well conditioned
  * trntensor.einsum         'ij,jk->ik' and the batched 'bij,bjk->bik'
  * trnrand.normal / uniform element counts; checks count, range, moments and
                             same-seed reproducibility (there is no bitwise
                             reference: the NKI stream is Threefry, not torch)

Error metric: relative Frobenius error ||got - ref|| / ||ref||, against a
tolerance chosen per dtype/op (fp32 operands that may go through BF16 Tensor
Engine passes with FP32 PSUM accumulation: 1e-2 for matmul-like ops and FFT).

So that a silent fallback to PyTorch can't pass as an NKI result, this script
sets every ``<LIB>_REQUIRE_NKI=1`` (kernel errors re-raise and show up as
ERROR), and sets ``TRNTENSOR_MIN_NKI_FLOPS[_PINNED]=0`` so small contractions
still go through the NKI kernel instead of the host cutover. Override with
``SWEEP_REQUIRE_NKI=0``.

Env knobs:
  SWEEP_SIZES        default 1,7,127,128,129,255,257,1000,4097
  SWEEP_OPS          comma list of op groups to run (default all):
                     gemm,fft,spmm,eigh,cg,einsum,rand
  SWEEP_INPUT_DEVICE neuron (default) | cpu: where inputs live before the call
  SWEEP_MAX_BATCHED  largest n for the batched einsum (default 1000)
  SPIKE_DEVICE=cpu   same as --cpu

Run on the box (from the umbrella root):

    python scripts/spike_shape_sweep.py 2>&1 | tee ~/spike_shape_sweep.log

CPU dry run (pytorch backends, device=cpu):

    python scripts/spike_shape_sweep.py --cpu

Post the log (RESULT lines plus the SUMMARY table) to trnsci/trnsci#35.
"""

# ruff: noqa: E402  (env vars must be set before the suite and NKI are imported)
import os
import sys

CPU = "--cpu" in sys.argv or os.environ.get("SPIKE_DEVICE", "").lower() == "cpu"

os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")
if not CPU and os.environ.get("SWEEP_REQUIRE_NKI", "1") == "1":
    for _lib in ("TRNBLAS", "TRNFFT", "TRNSOLVER", "TRNSPARSE", "TRNTENSOR", "TRNRAND"):
        os.environ.setdefault(f"{_lib}_REQUIRE_NKI", "1")
    os.environ.setdefault("TRNTENSOR_MIN_NKI_FLOPS", "0")
    os.environ.setdefault("TRNTENSOR_MIN_NKI_FLOPS_PINNED", "0")

import math
import time
import traceback
from collections import OrderedDict

import torch

if not CPU:
    import torch_neuronx  # noqa: F401  (registers the "neuron" device)

import trnblas
import trnfft
import trnrand
import trnsolver
import trnsparse
import trntensor

DEV = "cpu" if CPU else "neuron"
BACKEND = "pytorch" if CPU else "nki"
IN_DEV = "cpu" if CPU else os.environ.get("SWEEP_INPUT_DEVICE", "neuron")
SIZES = [
    int(s)
    for s in os.environ.get("SWEEP_SIZES", "1,7,127,128,129,255,257,1000,4097").split(
        ","
    )
]
OPS = os.environ.get("SWEEP_OPS", "gemm,fft,spmm,eigh,cg,einsum,rand").split(",")
MAX_BATCHED = int(os.environ.get("SWEEP_MAX_BATCHED", "1000"))
EIGH_NKI_MAX = 128  # trnsolver Householder tridiag: single-tile, n <= 128

TOL_MM = 1e-2  # fp32 in, possible BF16 PE passes, FP32 PSUM accumulate
TOL_FFT = 1e-2
TOL_EIGH = 1e-2  # eigenvalues rel. to ||A||; residual ||AV - VW|| / ||A||
TOL_CG = 1e-3  # cg tol=1e-6 on residual; well-conditioned -> small forward error

RESULTS = OrderedDict()  # op -> list of (case, status, detail)


def sync():
    if not CPU:
        import torch_neuronx

        torch_neuronx.synchronize()


def first_line(e):
    msg = str(e).strip().splitlines()
    return f"{type(e).__name__}: {msg[0][:200] if msg else ''}"


def rel(got, ref):
    got = got.detach().cpu().double()
    ref = ref.detach().cpu().double()
    if got.shape != ref.shape:
        raise ValueError(
            f"shape mismatch got {tuple(got.shape)} ref {tuple(ref.shape)}"
        )
    den = ref.norm().item()
    num = (got - ref).norm().item()
    return num / den if den > 0 else num


def record(op, case, status, detail=""):
    RESULTS.setdefault(op, []).append((case, status, detail))
    print(f"RESULT [{op} {case}]: {status} {detail}", flush=True)


def run_case(op, case, fn):
    """fn() returns (ok: bool, detail: str) or (None, reason) for SKIP."""
    t0 = time.perf_counter()
    try:
        ok, detail = fn()
        sync()
        dt = time.perf_counter() - t0
        if ok is None:
            record(op, case, "SKIP", detail)
        else:
            record(op, case, "PASS" if ok else "FAIL", f"{detail} ({dt:.1f}s)")
    except Exception as e:
        print(f"--- {op} {case} traceback (last frames):")
        traceback.print_exc(limit=-4)
        record(op, case, "ERROR", first_line(e))


def on(t):
    return t.to(IN_DEV)


# ------------------------------------------------------------------ cases
def gemm_triples():
    out = [(n, n, n) for n in SIZES]
    rot1, rot2 = SIZES[1:] + SIZES[:1], SIZES[2:] + SIZES[:2]
    for m, k, n in zip(SIZES, rot1, rot2):
        if (m, k, n) not in out:
            out.append((m, k, n))
    return out


def case_gemm(m, k, n):
    def f():
        g = torch.Generator().manual_seed(m * 7 + k * 3 + n)
        A, B = torch.randn(m, k, generator=g), torch.randn(k, n, generator=g)
        C = trnblas.gemm(1.0, on(A), on(B))
        e = rel(C, A.double() @ B.double())
        return e < TOL_MM, f"rel_err={e:.2e} tol={TOL_MM:.0e}"

    return f


def case_fft(n, inverse):
    def f():
        g = torch.Generator().manual_seed(n)
        xr, xi = torch.randn(4, n, generator=g), torch.randn(4, n, generator=g)
        x = trnfft.ComplexTensor(on(xr), on(xi))
        y = trnfft.ifft(x) if inverse else trnfft.fft(x)
        ref_c = (torch.fft.ifft if inverse else torch.fft.fft)(
            torch.complex(xr.double(), xi.double())
        )
        got = torch.stack([y.real.cpu(), y.imag.cpu()])
        ref = torch.stack([ref_c.real, ref_c.imag])
        e = rel(got, ref)
        return e < TOL_FFT, f"rel_err={e:.2e} tol={TOL_FFT:.0e} batch=4"

    return f


def case_spmm(n):
    def f():
        g = torch.Generator().manual_seed(n + 11)
        D = torch.randn(n, n, generator=g)
        D = D * (torch.rand(n, n, generator=g) < 0.1)
        if n == 1 or D.count_nonzero() == 0:
            D[0, 0] = 1.0
        csr = trnsparse.from_dense(D)
        if IN_DEV != "cpu":
            csr = trnsparse.CSRMatrix(
                values=on(csr.values),
                col_indices=on(csr.col_indices),
                row_ptrs=on(csr.row_ptrs),
                shape=csr.shape,
            )
        B = torch.randn(n, 64, generator=g)
        C = trnsparse.spmm(csr, on(B))
        e = rel(C, D.double() @ B.double())
        return (
            e < TOL_MM,
            f"rel_err={e:.2e} tol={TOL_MM:.0e} nnz={int(D.count_nonzero())}",
        )

    return f


def case_eigh(n):
    def f():
        if not CPU and n > EIGH_NKI_MAX:
            return (
                None,
                f"n={n} > {EIGH_NKI_MAX}: trnsolver NKI Householder tridiag is single-tile (documented guard)",
            )
        g = torch.Generator().manual_seed(n + 5)
        M = torch.randn(n, n, generator=g)
        A = (M + M.T) / 2
        w, V = trnsolver.eigh(on(A))
        w, V = w.cpu().double(), V.cpu().double()
        Ad = A.double()
        w_ref = torch.linalg.eigvalsh(Ad)
        nA = Ad.norm().item()
        ew = (w - w_ref).norm().item() / nA
        res = (Ad @ V - V * w).norm().item() / nA
        orth = (V.T @ V - torch.eye(n, dtype=torch.float64)).norm().item() / math.sqrt(
            n
        )
        ok = ew < TOL_EIGH and res < TOL_EIGH and orth < TOL_EIGH
        return (
            ok,
            f"eig_err={ew:.2e} resid={res:.2e} orth={orth:.2e} tol={TOL_EIGH:.0e}",
        )

    return f


def case_cg(n):
    def f():
        g = torch.Generator().manual_seed(n + 9)
        M = torch.randn(n, n, generator=g) / math.sqrt(n)
        A = M @ M.T + 2.0 * torch.eye(n)  # SPD, cond ~ (2+4)/2 = 3
        b = torch.randn(n, generator=g)
        x, iters, resid = trnsolver.cg(on(A), on(b), tol=1e-6, maxiter=500)
        x_ref = torch.linalg.solve(A.double(), b.double())
        e = rel(x, x_ref)
        return (
            e < TOL_CG,
            f"rel_err={e:.2e} tol={TOL_CG:.0e} iters={int(iters)} resid={float(resid):.1e}",
        )

    return f


def case_einsum(spec, shapes):
    def f():
        g = torch.Generator().manual_seed(sum(sum(s) for s in shapes))
        ops = [torch.randn(*s, generator=g) for s in shapes]
        got = trntensor.einsum(spec, *[on(t) for t in ops])
        ref = torch.einsum(spec, *[t.double() for t in ops])
        e = rel(got, ref)
        return e < TOL_MM, f"rel_err={e:.2e} tol={TOL_MM:.0e} shapes={shapes}"

    return f


def case_rand(kind, n):
    def f():
        x1 = (trnrand.normal if kind == "normal" else trnrand.uniform)(
            n, generator=trnrand.Generator(seed=1234)
        )
        x2 = (trnrand.normal if kind == "normal" else trnrand.uniform)(
            n, generator=trnrand.Generator(seed=1234)
        )
        x1, x2 = x1.cpu().double(), x2.cpu().double()
        problems = []
        if x1.numel() != n:
            problems.append(f"numel={x1.numel()}!={n}")
        if not torch.isfinite(x1).all():
            problems.append("non-finite")
        if not torch.equal(x1, x2):
            problems.append("same-seed not reproducible")
        mean = x1.mean().item()
        if kind == "normal":
            mu, sd = 0.0, 1.0
        else:
            mu, sd = 0.5, math.sqrt(1 / 12)
            if x1.min() < 0 or x1.max() >= 1:
                problems.append(f"out of [0,1): [{x1.min():.3g},{x1.max():.3g}]")
        if abs(mean - mu) > 6 * sd / math.sqrt(n):
            problems.append(f"mean={mean:.3g}")
        std = x1.std().item() if n > 1 else sd
        if n >= 100 and abs(std - sd) > 6 * sd / math.sqrt(2 * n) + 1e-3:
            problems.append(f"std={std:.3g}")
        if n >= 1000:  # distinct-value check: catches lane/tile replication
            frac = x1.unique().numel() / n
            if frac < 0.99:
                problems.append(f"unique_frac={frac:.3f}")
        return not problems, (
            f"mean={mean:.3g} std={std:.3g} " + "; ".join(problems)
        ).strip()

    return f


def main():
    print(
        "torch",
        torch.__version__,
        "| device",
        DEV,
        "| backend",
        BACKEND,
        "| inputs on",
        IN_DEV,
    )
    print("sizes", SIZES, "| ops", OPS)
    for lib in (trnblas, trnfft, trnrand, trnsolver, trnsparse, trntensor):
        lib.set_backend(BACKEND)
        print(
            f"  {lib.__name__} {getattr(lib, '__version__', '?')} HAS_NKI={getattr(lib, 'HAS_NKI', '?')}"
        )
    t_start = time.perf_counter()

    if "gemm" in OPS:
        for m, k, n in gemm_triples():
            run_case("trnblas.gemm", f"M={m} K={k} N={n}", case_gemm(m, k, n))
    if "fft" in OPS:
        for n in SIZES:
            run_case("trnfft.fft", f"n={n}", case_fft(n, False))
            run_case("trnfft.ifft", f"n={n}", case_fft(n, True))
    if "spmm" in OPS:
        for n in SIZES:
            run_case("trnsparse.spmm", f"n={n} x64", case_spmm(n))
    if "eigh" in OPS:
        for n in SIZES:
            run_case("trnsolver.eigh", f"n={n}", case_eigh(n))
    if "cg" in OPS:
        for n in SIZES:
            run_case("trnsolver.cg", f"n={n}", case_cg(n))
    if "einsum" in OPS:
        for m, k, n in gemm_triples():
            run_case(
                "trntensor.einsum ij,jk->ik",
                f"{m}x{k}x{n}",
                case_einsum("ij,jk->ik", [(m, k), (k, n)]),
            )
        for n in SIZES:
            if n > MAX_BATCHED:
                record(
                    "trntensor.einsum bij,bjk->bik",
                    f"b=3 n={n}",
                    "SKIP",
                    f"n > SWEEP_MAX_BATCHED={MAX_BATCHED}",
                )
                continue
            run_case(
                "trntensor.einsum bij,bjk->bik",
                f"b=3 n={n}",
                case_einsum("bij,bjk->bik", [(3, n, n), (3, n, n)]),
            )
    if "rand" in OPS:
        for kind in ("normal", "uniform"):
            for n in SIZES:
                run_case(f"trnrand.{kind}", f"count={n}", case_rand(kind, n))

    print(
        f"\n===== SUMMARY ===== (total {time.perf_counter() - t_start:.0f}s, device={DEV})"
    )
    print(f"{'op':34} {'PASS':>5} {'FAIL':>5} {'ERROR':>5} {'SKIP':>5}  failing cases")
    tot = dict.fromkeys(("PASS", "FAIL", "ERROR", "SKIP"), 0)
    for op, rows in RESULTS.items():
        c = dict.fromkeys(tot, 0)
        for _, s, _ in rows:
            c[s] += 1
            tot[s] += 1
        bad = ", ".join(f"{case}={s}" for case, s, _ in rows if s in ("FAIL", "ERROR"))
        print(
            f"{op:34} {c['PASS']:>5} {c['FAIL']:>5} {c['ERROR']:>5} {c['SKIP']:>5}  {bad}"
        )
    print(
        f"{'TOTAL':34} {tot['PASS']:>5} {tot['FAIL']:>5} {tot['ERROR']:>5} {tot['SKIP']:>5}"
    )


if __name__ == "__main__":
    main()
