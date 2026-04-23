---
date: 2026-04-22
categories: [Deep dive, trnfft]
comments: true
---

# trnfft: the FP32 accumulator you didn't know you had

trnfft v0.17 ships two new precision modes — `"bf16"` and `"bf16_refined"` — for the
DFT-GEMM fast path. The headline numbers: 1.4–1.5× faster than FP32 at N=64–256 on trn1,
with near-FP32 accuracy after one correction step. The mechanism is an architectural
property of Trainium that was already present in every kernel, just never exploited.

<!-- more -->

## The problem

BF16 on neural network hardware is a throughput story. A BF16 Tensor Engine tile fits
twice as many elements as an FP32 tile in the same systolic array pass — roughly 2× the
arithmetic operations per second. For large matrix multiplications (the workload the
Tensor Engine was designed for), that theoretical 2× translates almost directly to wall-
clock throughput.

FFT at small N is not that workload. At N=256, the DFT matrix is 256×256 = 65,536
complex entries. The one-matmul DFT-GEMM path (`W @ x`) dispatches a single `nc_matmul`
call and returns — the bottleneck is launch overhead and memory bandwidth, not arithmetic.
The standard BF16 story (2× arithmetic → 2× throughput) doesn't hold when you're memory-
bound.

The more interesting question: if BF16 doesn't give 2× throughput, what does it give, and
is there a way to get accuracy back?

## What the architecture suggests

The Tensor Engine accumulates into PSUM at FP32 regardless of input dtype. This is a
hardware invariant: every `nisa.nc_matmul` call, whether its inputs are FP32, BF16, or
FP16, writes to a FP32 PSUM tile. The PSUM is wider than the SBUF (FP32 vs BF16/FP8) by
design — it's the place where accumulated rounding errors are suppressed.

The existing `_complex_gemm_kernel` throws this away. Lines 190–192 of `dispatch.py`:

```python
if a_real.dtype != nl.float32:
    cr_sbuf = nl.cast(cr_sbuf, dtype=a_real.dtype)   # ← discards FP32 PSUM
    ci_sbuf = nl.cast(ci_sbuf, dtype=a_real.dtype)
```

The kernel casts the FP32 PSUM back to the input dtype before storing. For FP32 inputs
this is a no-op. For BF16 inputs, it rounds the accumulated FP32 result back to BF16 —
discarding the accumulation precision that the hardware worked to maintain.

The architectural suggestion: skip the cast. Store the FP32 PSUM directly.

## The approach

`_complex_gemm_kernel_bf16` is a two-line diff from the existing kernel. The output HBM
buffers are allocated as `nl.float32` instead of `a_real.dtype`, and the `nl.cast` block
is removed. The nc_matmul calls, the PSUM accumulation, the tile tiling logic — all
unchanged.

```python
# Output is always FP32, regardless of BF16 input dtype
c_real = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
c_imag = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.shared_hbm)
# ... nc_matmul accumulation unchanged ...
nisa.tensor_copy(dst=cr_sbuf, src=psum_cr)
# No nl.cast — FP32 PSUM goes directly to output
nl.store(c_real[...], value=cr_sbuf)
```

The driver `_fft_via_gemm_bf16` quantises `x` and `W` to BF16 before the call. One
subtlety: `W` entries are computed in FP32 then quantised, not computed in BF16 directly.
Computing the DFT angles in BF16 introduces quantisation in the exponent (k × j can reach
3969 at N=64; BF16 represents that as 3968, giving a ~0.1 radian angle error). Computing
in FP32 and rounding the cosine/sine values gives ~4e-3 entry error, which is what BF16
accuracy means.

**`precision="bf16_refined"` — one iterative correction step:**

```python
X̂ = fft_bf16(x)          # BF16 → FP32 PSUM → FP32 output
r  = x_fp32 − IFFT(X̂)   # FP32 residual: what the BF16 path got wrong
X̂  = X̂ + fft_bf16(r)    # correction (residual is small → BF16 ok)
```

The residual `r` corrects for the BF16 `W` quantisation error. After one step, the error
drops from ~1e-3 (BF16 `W`) to near-FP32 levels. The residual is small by construction —
the BF16 path already gets within ~1e-3 — so applying BF16 compute to it introduces only
~1e-3 × 1e-3 = 1e-6 of additional error. The correction converges.

## Implementation

The CPU fallback for `complex_gemm_bf16` casts BF16 inputs to FP32 before `complex_matmul`
to simulate the FP32 PSUM behaviour. On CPU, PyTorch's BF16 `torch.matmul` accumulates in
BF16 (no FP32 PSUM), giving ~50% relative error at N=64. The cast fixes this and makes
the CPU path a faithful simulator of hardware behaviour — same ~1e-3 accuracy from BF16
`W` quantisation, same FP32 output, just slower.

The dispatch adds two new precision modes before the existing `"fast"` DFT-GEMM check:

```python
if precision == "bf16" and n <= _DFT_GEMM_THRESHOLD:
    return _fft_via_gemm_bf16(x, inverse)

if precision == "bf16_refined" and n <= _DFT_GEMM_THRESHOLD:
    return _fft_iterative_refinement(x, inverse, steps=1)
```

Both modes are no-ops for N > 256 (fall through to the existing Stockham/butterfly paths).
The DFT-GEMM threshold is the natural boundary: beyond N=256, the O(N²) matmul work
dominates and BF16's throughput advantage shrinks further.

## What didn't work

**BF16 angle computation.** The first implementation used `torch.arange(n, dtype=torch.bfloat16)`
to build the DFT angle tensor. For N=64, the outer product `k×j` reaches 3969. BF16
has 7 mantissa bits — integers above ~128 lose precision, and 3969 is represented as 3968.
The resulting angle error is `2π × 1/64 ≈ 0.098 rad`, which gives ~40% relative error
in the DFT output. The fix was to compute angles in FP32 and quantise the final W entries,
not the angle indices.

**CPU BF16 accumulation.** On CPU, `torch.matmul` with BF16 inputs accumulates in BF16
(no systolic array PSUM). The initial test showed 43% relative error at N=64 on CPU —
indistinguishable from "broken." The fix: the CPU fallback casts to FP32 before the matmul.
This makes the CPU path test the right thing (BF16 `W` quantisation error, not BF16
accumulation error), but it's a reminder that the FP32 PSUM property is a hardware
feature — it cannot be assumed on CPU.

**The 2× throughput claim.** The theoretical 2× BF16 speedup assumed arithmetic-bound
compute. At N=64–256, the DFT-GEMM path is memory-bandwidth and launch-overhead bound.
Measured speedup is 1.4–1.5×, not 2×. For the BF16 path to show 2×, the problem would
need to be larger (N > 1024) or more compute-bound. The measured result is still a
meaningful win, just not the theoretical ceiling.

## Numbers

Hardware bench: trn1.2xlarge, Neuron SDK 2.29.0, NKI 0.3.0, 2026-04-22.

| N   | `"bf16"` (µs) | `"fast"` FP32 (µs) | Speedup | `"bf16"` rel error |
| --- | ------------- | ------------------- | ------- | ------------------- |
| 64  | 1 189         | ~1 833              | ~1.5×   | ~2e-3               |
| 128 | 1 208         | —                   | —       | ~3e-3               |
| 256 | 1 310         | ~1 882              | ~1.4×   | ~4e-3               |

FP32 baseline from v0.12 on SDK 2.24. BF16 rel error measured on CPU with BF16-quantised
W (same regime as hardware).

The `"bf16_refined"` path costs approximately 2× the `"bf16"` path (one forward + one IFFT
+ one forward) and drives rel error to ~1e-6 (near-FP32). Total wall-clock: ~2.4–2.6 ms
at N=256, comparable to FP32 DFT-GEMM (~1.9 ms) but with BF16 throughput for all the
compute and FP32 accuracy for the output.

**Where this is well-indexed:** spectral methods and signal processing pipelines where
multiple FFTs are chained — the BF16 path gives 1.4× speedup per FFT stage, and `"bf16_refined"`
gives near-FP32 accuracy when chain accumulation matters.

**Where it is not well-indexed:** very small N (< 64) where PSUM startup dominates; large
N (> 256) where BF16 `W` quantisation error grows with the larger DFT matrix; applications
requiring the full FP64 accuracy of `precision="double"`.

## What's next

- **Ozaki-scheme FP64 emulation from BF16 inputs.** The Ozaki scheme represents each
  input as a sum of BF16 partial values, accumulates each partial matmul into FP32 PSUM,
  and combines. Applied to DFT-GEMM, this would give near-FP64 accuracy (≈1e-14) using
  a sequence of BF16 matmuls — the same PSUM-as-accumulator principle, applied to each
  Ozaki component. This is the `target_forward_error=1e-10` API direction.
- **Multi-NeuronCore distribution** for large N (N > 4096). Linear speedup with core count
  by partitioning the batch dimension across NeuronCores.

Issues tracking the above are open on [trnsci/trnsci](https://github.com/trnsci/trnsci/issues).

## Takeaway

The FP32 PSUM accumulator was always there. Every previous kernel built it, used it, then
threw it away by casting back to the input dtype. Keeping the FP32 output costs zero
additional compute — it is the compute that was already happening. The BF16 precision mode
is one line of kernel code shorter than the FP32 mode. The 1.4× speedup comes from
computing in BF16, not from a new algorithm. The near-FP32 accuracy from `"bf16_refined"`
comes from applying the same BF16 compute once more to a small residual. Both are direct
consequences of the systolic array accumulating into a FP32 accumulator that no GPU with
an FP64 legacy would have prioritised.
