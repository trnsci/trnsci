---
date: 2026-04-25
categories: [Deep dive, trnfft]
comments: true
---

# trnfft: eight hardware runs, one wrong argument, and the Ozaki scheme

v0.18 ships `precision="ozaki"` — three BF16 matmuls that together deliver ~1e-5
relative error where a single BF16 matmul gives ~1e-3. The implementation took an
afternoon. Getting the benchmark to record a single timing took eight hardware runs
and produced a "What didn't work" section that was, frankly, humbling.

## The problem

`precision="bf16"` (v0.17) gives ~1.4× throughput over FP32 at N=64–256. It also gives
~1e-3 relative error because the BF16 DFT matrix W has ~4e-3 per-entry quantisation
error that propagates through the matmul. For applications that can tolerate 0.1%
error, BF16 is fine. For spectral methods, iterative solvers, or chained FFT pipelines,
it isn't.

The question: can we get BF16 throughput with systematically better accuracy?

## What the architecture suggests

The Ogita–Rump–Oishi split decomposes any FP32 value into two BF16 parts:

```python
x_high = bfloat16(x)                       # BF16 representation
x_low  = bfloat16(x - float32(x_high))     # BF16 residual
```

The split satisfies x ≈ x_high + x_low with error O(u_bf16²) instead of O(u_bf16).
Applied to both W (the DFT matrix) and x (the input signal):

```
W @ x ≈ W_h @ x_h  +  W_h @ x_l  +  W_l @ x_h
```

Three terms. Each term is a BF16 matmul. The PSUM accumulates each in FP32 (hardware
invariant). Sum the three FP32 results — the accumulated error is now O(sqrt(N) × u²)
instead of O(sqrt(N) × u). For u_bf16 ≈ 4e-3: from ~1e-3 down to ~1.6e-5 at N=256.

No new NKI kernel. No new hardware feature. The split uses only PyTorch arithmetic on
CPU; each term reuses the same `complex_gemm_bf16` kernel that v0.17 already validated.

## The approach

`_fft_via_ozaki` calls `complex_gemm_bf16` three times with the high/low split parts,
then sums the three FP32 results on-device. The data-dependency trick ensures sequential
execution:

```python
hh = _term(x_r_h, x_i_h, W_r_h, W_i_h)
_dep_hh = hh.real.mean() * 0   # zero, but forces hh to evaluate before hl starts

hl = _term(x_r_h + _dep_hh, x_i_h + _dep_hh, W_r_l, W_i_l)
_dep_hl = hl.real.mean() * 0

lh = _term(x_r_l + _dep_hl, x_i_l + _dep_hl, W_r_h, W_i_h)

X_re = (hh.real + hl.real + lh.real).reshape(orig_shape)
```

The `0 * mean()` creates a graph dependency that forces the XLA compiler to execute
`hh` before `hl`, and `hl` before `lh`, without any sync API call.

## What didn't work

**Eight hardware benchmark attempts produced no timing data.** The first seven attempts
were the most thorough debugging-in-the-dark session in this project's history:

1. Dispatched through a new `complex_gemm_ozaki()` wrapper — silent failure.
2. Moved FP64 accumulation to CPU via `.detach().cpu().double()` — silent failure.
3. Switched to FP32 on-device accumulation — silent failure.
4. Rewrote to call `complex_gemm_bf16` directly per term — silent failure.
5. Added `.contiguous()` after each term — silent failure. (Correctly-contiguous XLA
   tensors don't materialise on `.contiguous()` — it's a no-op.)
6. Added explicit `torch_xla.sync()` — silent failure. (Unavailable or broken on this
   SDK version.)
7. Data-dependency trick (`0 * mean()`) — silent failure.

All seven produced `14 of 17 benchmarks recorded`. The three Ozaki tests ran and passed
correctness tests on CPU. Something was different about hardware.

The diagnostic: run the Ozaki path directly on the instance via SSM and capture stderr:

```
FAILED: TypeError _fft_via_ozaki() got an unexpected keyword argument 'levels'
```

The bench toggle `_FORCE_OZAKI` was calling `_fft_via_ozaki(x, inverse, levels=2)`.
The `levels` parameter had been removed from the function signature when we simplified
from a multi-level design to a single-level one. Every single attempt threw `TypeError`
before timing started. The test harness caught the exception, marked the test as failed,
and produced no benchmark record — no output, no traceback, nothing. Seven hardware runs
and seven SSM launches for a one-word fix: remove `levels=2`.

**The actual root cause of the XLA lazy graph issue** (which led to the first few
attempts): three independent `complex_gemm_bf16` calls with no data dependencies
between them build a graph with three independent branches that the Neuron runtime
processes differently from a single-kernel call. The data-dependency trick is the correct
fix for this, and it's in the final implementation — it just didn't matter until the
`TypeError` was fixed.

**Upstream ask for pytest-benchmark on Neuron**: a test that throws `TypeError` before
calling `benchmark(fn, ...)` should emit a warning rather than silently producing no
JSON record. The current behaviour (0 records, no error in the benchmark output) makes
hardware regressions of this type extremely hard to diagnose.

## Numbers

Hardware bench: trn1.2xlarge, Neuron SDK 2.29.0, NKI 0.3.0, 2026-04-25.

| N   | Ozaki (µs) | BF16 (µs) | FP32 DFT-GEMM | oz/bf16 | oz/fp32 | rel error |
| --- | ---------- | --------- | ------------- | ------- | ------- | --------- |
| 64  | 3 241      | 1 179     | ~1 833        | 2.75×   | 1.77×   | ~1.6e-5   |
| 128 | 3 291      | 1 216     | —             | 2.71×   | —       | ~2.3e-5   |
| 256 | 3 466      | 1 302     | ~1 882        | 2.66×   | 1.84×   | ~3.2e-5   |

CPU precision test (validated before hardware run):

| N   | BF16 rel error | Ozaki rel error | improvement |
| --- | -------------- | --------------- | ----------- |
| 64  | ~2.2e-3        | ~1.6e-5         | 140×        |
| 256 | ~3.9e-3        | ~3.2e-5         | 120×        |

**Where this is well-indexed:** spectral methods and pipelines that chain multiple FFTs
where accumulated BF16 error matters. The O(u²) error bound is provable and predictable,
unlike the iterative refinement approach (IR-1 from v0.17) whose convergence depends on
the specific signal.

**Where it is not well-indexed:** single unbatched FFT where FP32 (`precision="fast"`)
is already accurate enough; N > 256 where DFT-GEMM's O(N²) precision budget is
exhausted regardless; applications that need full FP64 accuracy (`precision="double"`).

## What's next

- **2-level Ozaki (FP32 residual staging):** a second split level requires keeping the
  BF16 residual in FP32 through the intermediate stage (not casting it back to BF16
  immediately). This would reach O(u^4) ≈ 2e-10 relative error at N=256 — genuine
  near-FP64 accuracy on-chip.
- **Multi-NeuronCore distribution** for N > 4096.

Issues tracking the above are open on [trnsci/trnsci](https://github.com/trnsci/trnsci/issues).

## Takeaway

Three BF16 matmuls from a one-line split: `x_high = bfloat16(x); x_low = bfloat16(x -
float32(x_high))`. The Ozaki scheme on Trainium is enabled by the PSUM being FP32
regardless of input dtype, and by the fact that three independent BF16 matmuls cost
exactly 3× one BF16 matmul — which is ~1.8× one FP32 matmul. The accuracy improvement
is 120–140× over single-pass BF16. The implementation is 40 lines of Python with no new
NKI kernel. The debugging took longer than the implementation. That's the project.
