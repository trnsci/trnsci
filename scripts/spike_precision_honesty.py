#!/usr/bin/env python
"""Spike B5 — do the suite's precision promises hold on the device?

Each library advertises accuracy modes with a stated (or derivable) error
bound. This script runs every mode on the device, measures the achieved error
against an FP64 reference, and checks it against the promise.

trntensor  ``einsum`` for every ``precision=`` mode (fast/sr/dd/kahan) and
           ``einsum`` / ``multi_einsum`` for ``target_forward_error`` in
           {1e-2, 1e-4, 1e-6, 1e-8}; fp32 and bf16 operands; square matmul
           ``ij,jk->ik`` over ``PREC_SIZES``. Bound = the formula
           ``select_precision_for_error`` uses for the selected mode
           (fast K*u, sr sqrt(K)*u, dd K*u^2, kahan K*2^-53). Error metric is
           the Wilkinson componentwise one the bounds are stated in:
           max_ij |C - C64|_ij / (|A||B|)_ij  (RMS of it for "sr", whose bound
           is an expectation). For targets, PASS also requires error <= target.
trnfft     ``set_precision`` modes (fast/kahan/double/bf16/bf16_refined/ozaki/
           ozaki_hq) vs FP64 torch.fft over N in ``FFT_SIZES``; normwise
           relative L2 error vs the bound each mode's docstring states (table
           ``FFT_BOUNDS`` below; FP32-output modes are floored at 1e-6).
           ozaki/ozaki_hq run with ``set_ozaki_product_precision_verified(True)``
           — whether trn2 really has FP32 product precision is the question.
trnblas    ``gemm`` fp32 vs K*u_f32; double-double GEMM if any public entry
           point exposes it (SKIP otherwise; trnblas#22).

SR note: on NKI 0.7 trntensor ``precision="sr"`` only rounds stochastically
when ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1`` is set at process start (default
round-to-nearest). The value is printed; when SR is not enabled the sr cases
are judged against the fast (K*u) bound and tagged ``SR_INACTIVE``.

Lines: ``RESULT [<case>]: PASS|FAIL|ERROR|SKIP ...`` then a ``SUMMARY`` block.

On the box (Trn2, SDK 2.33), both SR settings:

    NEURON_RT_STOCHASTIC_ROUNDING_EN=0 python scripts/spike_precision_honesty.py 2>&1 | tee ~/b5_rne.log
    NEURON_RT_STOCHASTIC_ROUNDING_EN=1 python scripts/spike_precision_honesty.py 2>&1 | tee ~/b5_sr.log

CPU dry run (pytorch backends): ``python scripts/spike_precision_honesty.py --cpu``.
Env: PREC_SIZES (default 128,256,512,1024), PREC_DTYPES (fp32,bf16),
FFT_SIZES (64,128,256,512,1024,2048,4096), FFT_BATCH (8), PREC_LIBS
(trntensor,trnfft,trnblas). Post both logs.
"""

import argparse
import math
import os
import sys
import time
import warnings

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", action="store_true", help="run on CPU with pytorch backends")
ARGS = ap.parse_args()
CPU = ARGS.cpu or os.environ.get("SPIKE_DEVICE", "").lower() == "cpu"

if not CPU:
    # Never let trntensor silently fall back to host torch.matmul: force the NKI
    # path at every size and raise on kernel failure (reported as ERROR).
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


def ints(name, default):
    return [int(v) for v in os.environ.get(name, default).split(",") if v.strip()]


SIZES = ints("PREC_SIZES", "128,256,512,1024")
FFT_SIZES = ints("FFT_SIZES", "64,128,256,512,1024,2048,4096")
FFT_BATCH = int(os.environ.get("FFT_BATCH", "8"))
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
DTYPES = {k: DTYPES[k] for k in os.environ.get("PREC_DTYPES", "fp32,bf16").split(",")}
LIBS = os.environ.get("PREC_LIBS", "trntensor,trnfft,trnblas").split(",")
TARGETS = (1e-2, 1e-4, 1e-6, 1e-8)
U = {torch.bfloat16: 2**-8, torch.float16: 2**-11, torch.float32: 2**-24}
U64 = 2**-53

SR_ENV = os.environ.get("NEURON_RT_STOCHASTIC_ROUNDING_EN")
SR_ACTIVE = CPU or SR_ENV == "1"  # CPU pytorch backend emulates SR in software

RESULTS = []  # (case, status, detail)


def record(case, status, detail=""):
    RESULTS.append((case, status, detail))
    print(f"RESULT [{case}]: {status} {detail}", flush=True)


def first_line(e):
    msg = str(e).strip().splitlines()
    return f"{type(e).__name__}: {msg[0][:200] if msg else ''}"


def mode_bound(mode, dtype, K):
    u = U.get(dtype, U[torch.float32])
    if mode == "fast":
        return K * u
    if mode == "sr":
        return math.sqrt(K) * u if SR_ACTIVE else K * u
    if mode == "dd":
        return K * u * u
    if mode == "kahan":
        return K * U64
    raise ValueError(mode)


def componentwise_err(C, A, B):
    """|C - C64| / (|A||B|) elementwise, in fp64 on host (inputs as rounded)."""
    A64, B64 = A.detach().cpu().double(), B.detach().cpu().double()
    ref = A64 @ B64
    denom = A64.abs() @ B64.abs()
    e = (C.detach().cpu().double() - ref).abs() / denom.clamp_min(1e-300)
    norm = ((C.detach().cpu().double() - ref).norm() / ref.norm()).item()
    return e.max().item(), e.pow(2).mean().sqrt().item(), norm


def judge(case, mode, dtype, K, C, A, B, target=None, extra=""):
    """PASS iff achieved <= accumulation bound + output-dtype rounding (u_out),
    and (for targets) achieved <= target. The selector's formulas cover the
    accumulation only; an fp32/bf16 result can never beat its own u_out, so a
    target below u_out is flagged TARGET_BELOW_OUTPUT_ROUNDOFF."""
    mx, rms, norm = componentwise_err(C, A, B)
    accum = mode_bound(mode, dtype, K)
    sr = mode == "sr" and SR_ACTIVE
    achieved, metric = (rms, "rms_cw") if sr else (mx, "max_cw")
    u_out = U.get(C.dtype, U64 if C.dtype == torch.float64 else U[torch.float32])
    bound = accum + u_out
    ok = achieved <= bound and (target is None or achieved <= target)
    tag = " SR_INACTIVE" if (mode == "sr" and not SR_ACTIVE) else ""
    if mode == "dd" and dtype not in (torch.bfloat16, torch.float16):
        tag += " DD_DOCUMENTED_FOR_BF16_FP16_ONLY"
    if target is not None and u_out > target:
        tag += " TARGET_BELOW_OUTPUT_ROUNDOFF"
    detail = (
        f"mode={mode} {metric}={achieved:.3e} bound={bound:.3e} (accum={accum:.3e} + u_out={u_out:.1e})"
        + (f" target={target:.0e}" if target is not None else "")
        + f" max_cw={mx:.3e} norm_rel={norm:.3e} out_dtype={str(C.dtype).replace('torch.', '')}"
        + tag
        + extra
    )
    if not ok:
        why = []
        if achieved > bound:
            why.append(f"achieved {achieved:.3e} > promised {bound:.3e}")
        if target is not None and achieved > target:
            why.append(f"achieved {achieved:.3e} > target {target:.0e}")
        detail += " | " + "; ".join(why)
    record(case, "PASS" if ok else "FAIL", detail)


def make_ab(n, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.float64).to(dtype)
    B = torch.randn(n, n, generator=g, dtype=torch.float64).to(dtype)
    return A, B


# ------------------------------------------------------------------ trntensor
def run_trntensor():
    import trntensor

    trntensor.set_backend("pytorch" if CPU else "nki")
    print(f"trntensor {trntensor.__version__} backend={trntensor.get_backend()}")
    for dname, dtype in DTYPES.items():
        for n in SIZES:
            A, B = make_ab(n, dtype, seed=n)
            Ad, Bd = A.to(DEV), B.to(DEV)
            K = n
            for mode in ("fast", "sr", "dd", "kahan"):
                case = f"trntensor einsum precision={mode} {dname} N={n}"
                try:
                    C = trntensor.einsum("ij,jk->ik", Ad, Bd, precision=mode)
                    sync()
                    extra = ""
                    if mode == "sr":
                        C2 = trntensor.einsum("ij,jk->ik", Ad, Bd, precision=mode)
                        sync()
                        diff = (C.cpu() != C2.cpu()).double().mean().item()
                        extra = f" run2run_diff_frac={diff:.3f}"
                    judge(case, mode, dtype, K, C, A, B, extra=extra)
                except Exception as e:
                    record(case, "ERROR", first_line(e))
            for t in TARGETS:
                case = f"trntensor einsum target={t:.0e} {dname} N={n}"
                try:
                    try:
                        picked = trntensor.select_precision_for_error(dtype, K, t)
                    except ValueError as e:
                        record(case, "ERROR", f"selector refused: {first_line(e)}")
                        continue
                    C = trntensor.einsum("ij,jk->ik", Ad, Bd, target_forward_error=t)
                    sync()
                    judge(
                        case,
                        picked,
                        dtype,
                        K,
                        C,
                        A,
                        B,
                        target=t,
                        extra=f" picked={picked}",
                    )
                except Exception as e:
                    record(case, "ERROR", first_line(e))
            # multi_einsum: homogeneous batch of 3 (exercises the batched path).
            Bs = [make_ab(n, dtype, seed=1000 + n + i)[1] for i in range(3)]
            Bds = [b.to(DEV) for b in Bs]
            for t in TARGETS:
                case = f"trntensor multi_einsum target={t:.0e} {dname} N={n}"
                try:
                    try:
                        picked = trntensor.select_precision_for_error(dtype, K, t)
                    except ValueError as e:
                        record(case, "ERROR", f"selector refused: {first_line(e)}")
                        continue
                    outs = trntensor.multi_einsum(
                        *[("ij,jk->ik", Ad, b) for b in Bds], target_forward_error=t
                    )
                    sync()
                    # judge the worst of the three
                    errs = [componentwise_err(o, A, b)[0] for o, b in zip(outs, Bs)]
                    w = max(range(3), key=lambda i: errs[i])
                    judge(
                        case,
                        picked,
                        dtype,
                        K,
                        outs[w],
                        A,
                        Bs[w],
                        target=t,
                        extra=f" picked={picked} worst_of=3",
                    )
                except Exception as e:
                    record(case, "ERROR", first_line(e))


# --------------------------------------------------------------------- trnfft
# Promised normwise relative error per mode, from trnfft/precision.py docstrings.
# (bound for N <= DFT-GEMM threshold 256, bound above it). FP32-output modes are
# floored at 1e-6 (an FP32 result cannot be more accurate than ~u_f32*sqrt(log N)).
FFT_BOUNDS = {
    "fast": (
        4e-6,
        4e-6,
    ),  # "fast's <= 4e-6" butterfly budget (N up to 4096 measured 3.6e-6)
    "kahan": (5e-7, 5e-7),  # measured <= 4.55e-7 to N=4096 ("<= 3e-7" budget text)
    "double": (
        1e-6,
        1e-6,
    ),  # FP64 DFT-GEMM N<=1024 (1e-14 pre-cast; fp32 output); see below
    "bf16": (2e-3, 4e-6),  # "~1e-3 at N=256"; above 256 falls through to fast
    "bf16_refined": (1e-5, 4e-6),  # "near-FP32"
    "ozaki": (1.6e-5, 4e-6),  # "O(u_bf16^2) ~ 1.6e-5 at N=256"
    "ozaki_hq": (1e-6, 4e-6),  # theory 2e-9, floored at fp32 output
}


def fft_bound(mode, n):
    if mode == "double":
        return 1e-6 if n <= 1024 else 1e-4  # >1024 falls to NKI Stockham "~1e-4"
    lo, hi = FFT_BOUNDS[mode]
    return lo if n <= 256 else hi


def run_trnfft():
    import trnfft

    trnfft.set_backend("pytorch" if CPU else "nki")
    print(f"trnfft {trnfft.__version__} backend={trnfft.get_backend()}")
    from trnfft.precision import _VALID as modes

    trnfft.set_ozaki_product_precision_verified(True)
    for n in FFT_SIZES:
        g = torch.Generator().manual_seed(n)
        xr = torch.randn(FFT_BATCH, n, generator=g, dtype=torch.float64)
        xi = torch.randn(FFT_BATCH, n, generator=g, dtype=torch.float64)
        ref = torch.fft.fft(torch.complex(xr, xi))
        x = trnfft.ComplexTensor(xr.float().to(DEV), xi.float().to(DEV))
        for mode in modes:
            case = f"trnfft fft precision={mode} N={n}"
            try:
                trnfft.set_precision(mode)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    y = trnfft.fft(x)
                    sync()
                got = torch.complex(
                    y.real.detach().cpu().double(), y.imag.detach().cpu().double()
                )
                err = ((got - ref).abs().norm() / ref.abs().norm()).item()
                bound = fft_bound(mode, n)
                warn = f" warn={str(w[0].message)[:60]!r}" if w else ""
                detail = f"norm_rel={err:.3e} bound={bound:.1e}{warn}"
                if err > bound:
                    detail += f" | achieved {err:.3e} > promised {bound:.1e}"
                record(case, "PASS" if err <= bound else "FAIL", detail)
            except Exception as e:
                record(case, "ERROR", first_line(e))
            finally:
                trnfft.set_precision("fast")


# -------------------------------------------------------------------- trnblas
def run_trnblas():
    import inspect

    import trnblas

    trnblas.set_backend("pytorch" if CPU else "nki")
    print(f"trnblas {trnblas.__version__} backend={trnblas.get_backend()}")
    for n in SIZES:
        A, B = make_ab(n, torch.float32, seed=7 * n)
        case = f"trnblas gemm fp32 N={n}"
        try:
            C = trnblas.gemm(1.0, A.to(DEV), B.to(DEV))
            sync()
            mx, _, norm = componentwise_err(C, A, B)
            bound = n * U[torch.float32]
            d = f"max_cw={mx:.3e} bound=K*u={bound:.3e} norm_rel={norm:.3e}"
            if mx > bound:
                d += f" | achieved {mx:.3e} > promised {bound:.3e}"
            record(case, "PASS" if mx <= bound else "FAIL", d)
        except Exception as e:
            record(case, "ERROR", first_line(e))

    dd_names = [
        nm
        for nm in dir(trnblas)
        if any(s in nm.lower() for s in ("dd", "double", "ozaki"))
    ]
    gemm_params = inspect.signature(trnblas.gemm).parameters
    if "precision" in gemm_params:
        for n in SIZES:
            A, B = make_ab(n, torch.float32, seed=11 * n)
            case = f"trnblas gemm precision=dd N={n}"
            try:
                C = trnblas.gemm(1.0, A.to(DEV), B.to(DEV), precision="dd")
                sync()
                mx, _, norm = componentwise_err(C, A, B)
                bound = n * U[torch.float32] ** 2
                ok = mx <= max(bound, U[torch.float32])
                record(
                    case, "PASS" if ok else "FAIL", f"max_cw={mx:.3e} bound={bound:.3e}"
                )
            except Exception as e:
                record(case, "ERROR", first_line(e))
    else:
        record(
            "trnblas dd_gemm",
            "SKIP",
            f"no public double-double GEMM (gemm has no precision=; dd-like names: {dd_names or 'none'}; trnblas#22)",
        )


if __name__ == "__main__":
    t0 = time.time()
    print(f"python {sys.version.split()[0]} torch {torch.__version__} device={DEV}")
    if not CPU:
        print("torch_neuronx", torch_neuronx.__version__)
    print(
        f"NEURON_RT_STOCHASTIC_ROUNDING_EN={SR_ENV!r} -> trntensor sr "
        + (
            "stochastic"
            if SR_ACTIVE
            else "round-to-nearest (SR_INACTIVE; judged vs K*u)"
        )
    )
    for lib, fn in (
        ("trntensor", run_trntensor),
        ("trnfft", run_trnfft),
        ("trnblas", run_trnblas),
    ):
        if lib not in LIBS:
            continue
        print(f"\n===== {lib} =====", flush=True)
        try:
            fn()
        except Exception as e:
            record(f"{lib} setup", "ERROR", first_line(e))

    print("\nSUMMARY")
    print(f"device={DEV} sr_env={SR_ENV!r} elapsed={time.time() - t0:.0f}s")
    counts = {}
    for _, s, _ in RESULTS:
        counts[s] = counts.get(s, 0) + 1
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for case, s, d in RESULTS:
        if s in ("FAIL", "ERROR"):
            print(f"  {s:5} {case}: {d}")
