---
date: 2026-04-30
categories: [Deep dive, trnfft]
comments: true
---

# trnfft: correcting the Ozaki precision claim

Two earlier posts — [v0.18](https://trnsci.dev/blog/trnfft-eight-hardware-runs-one-wrong-argument-and-the-ozaki-scheme/) and
[v0.19](https://trnsci.dev/blog/trnfft-the-residual-must-stay-fp32/) — claimed that the Ozaki scheme
delivers O(sqrt(N)·u_bf16²) ≈ 1.6e-5 and O(sqrt(N)·u_bf16⁴) ≈ 2e-9 relative error
on Trainium hardware. The hardware measurements say otherwise. Both modes give ~1.7e-3
— the same as a single-pass BF16 DFT-GEMM. This post corrects the record and explains
why.

<!-- more -->

## The measurement

`TestOzakiHQCharacterization` on trn1.2xlarge (Neuron SDK 2.29.0, 2026-04-30):

| mode       | hardware rel error (N=64) | theoretical | CPU rel error (N=64) |
| ---------- | ------------------------- | ----------- | -------------------- |
| bf16       | ~1.7e-3                   | ~1.6e-2     | ~2.2e-3              |
| ozaki      | ~1.7e-3                   | ~1.6e-5     | ~1.6e-5              |
| ozaki_hq   | ~1.7e-3                   | ~2e-9       | ~1.4e-7              |

On CPU the scheme works exactly as theorised — ozaki is ~140× better than bf16, ozaki_hq
is a further ~100× better. On hardware, all three modes give the same result.

## What the architecture actually does

The Ogita–Rump–Oishi scheme improves precision by decomposing W and x into BF16 high/low
parts and summing multiple BF16 matmuls. The precision gain depends on a specific
assumption: **the error in each matmul is dominated by the BF16 quantization of the
inputs, not by arithmetic rounding inside the matmul itself.**

That assumption holds on CPU, where the `complex_gemm_bf16` fallback promotes BF16 inputs
to FP32 before calling `torch.matmul`. Every product `W_h[i,k] × x_h[k,j]` is computed
in FP32 — no per-product rounding beyond what BF16 quantization already introduced. The
split terms (W_h@x_l, W_l@x_h) accurately cancel the input quantization error.

On Trainium, `nc_matmul` with BF16 stationary and BF16 moving inputs accumulates into
FP32 PSUM. The PSUM does prevent accumulation error, as advertised. But each individual
product `W_h[i,k] × x_h[k,j]` is computed **in BF16** before being added to the FP32
accumulator. That per-product BF16 rounding introduces ~u_bf16 of error per product,
and N products sum to ~sqrt(N)·u_bf16 total error — indistinguishable from the single-pass
BF16 result. The split terms can't cancel errors they can't observe.

```
CPU:      BF16(W_h) × BF16(x_h) → FP32 product → FP32 PSUM  ✓ Ozaki works
Hardware: BF16(W_h) × BF16(x_h) → BF16 product → FP32 PSUM  ✗ per-product error dominates
```

The "free FP32 accumulator" is real and prevents accumulation error. But it is not enough:
the bottleneck is not accumulation, it is the per-product multiply.

## What this means for the library

**Throughput numbers are unaffected.** The timing benchmarks in v0.18/v0.19 are correct:
ozaki takes ~2.7× BF16 latency and ozaki_hq takes ~5.3×. Three and six BF16 matmuls
are genuinely being dispatched and timed. The data-dependency trick works. The XLA lazy
graph executes sequentially. The Ozaki path runs exactly as implemented.

**The precision claim is wrong for hardware.** On Trainium, `precision="ozaki"` and
`precision="ozaki_hq"` deliver ~1.7e-3 relative error — the same as `precision="bf16"`.
The extra matmuls cost time without buying accuracy. The modes are not harmful (the results
are correct FFTs), but the precision marketing was based on CPU measurements and does not
transfer.

**The right precision path for hardware remains `precision="double"`**: route through
CPU, use FP64 arithmetic, pay the roundtrip cost. That is the only mode that actually
delivers near-FP64 accuracy on this hardware today.

## What a hardware Ozaki would require

For the Ozaki scheme to work on Trainium, the Tensor Engine would need to compute
BF16 × BF16 products in FP32 (or higher) before accumulation into PSUM. That is:
mixed-precision multiply-accumulate where inputs are BF16 but the partial product is FP32.

This is a concrete hardware request: **BF16 input matmul with FP32 product precision
and FP32 PSUM.** The result would be identical to what Tensor Cores with TF32 accumulation
provide in CUDA. If Trainium 2 (trn2) exposes this path via NKI, the Ozaki scheme would
work as theorised without any code change — only the hardware behavior needs to change.

Until then, the ozaki and ozaki_hq modes are available for CPU correctness testing and
future hardware validation, but should not be used in production with the expectation of
hardware precision improvement.

## What's next

- **`precision="double"` remains the hardware precision path.** No change to the API.
- **NKI upstream ask:** filed. Request: `nc_matmul` option to compute BF16 × BF16
  products at FP32 precision before FP32 PSUM accumulation.
- **Ozaki as a CPU preprocessing step:** if inputs are small enough to fit in CPU SRAM,
  running the DFT-GEMM on CPU in FP64 and then transferring to Trainium is faster than
  `precision="double"` for batch workloads. This is a dispatch optimization, not an
  algorithm change. Tracked as a future improvement.

## Takeaway

The Ozaki scheme is a correct algorithm. The CPU measurements were correct. The
architectural reasoning — "PSUM prevents accumulation error, so splits correct the
remaining input-quantization error" — was incomplete. It missed the per-product multiply
rounding that happens before PSUM sees the result. On current Trainium hardware, that
rounding dominates, and the split provides no benefit. Throughput costs are real;
precision gains are not. The posts are wrong. The CHANGELOG now says so explicitly.
