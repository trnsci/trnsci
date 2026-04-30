---
date: 2026-04-30
categories: [Deep dive, trnfft]
comments: true
---

# trnfft: what trn1 tells us about the Ozaki frontier

The [v0.18](../2026-04-25-trnfft-ozaki-eight-runs/) and [v0.19](../2026-04-28-trnfft-two-level-ozaki/)
posts claimed hardware precision of O(sqrt(N)·u_bf16²) ≈ 1.6e-5 and O(sqrt(N)·u_bf16⁴) ≈ 2e-9
for the Ozaki modes. The trn1 hardware measurement says those numbers are wrong — both modes
deliver ~1.7e-3, equivalent to single-pass BF16. But the conclusion is not "Ozaki is a dead end."
NVIDIA shipped Ozaki-style FP64 emulation in cuBLAS in October 2025, with 1.5–3× speedups in
ecTrans, BerkeleyGW, and Quantum Espresso at maintained accuracy. The difference between
NVIDIA's working result and trnfft's non-result is one thing: hardware generation.

<!-- more -->

## The measurement

`TestOzakiHQCharacterization` on trn1.2xlarge (Neuron SDK 2.29.0, 2026-04-30):

| mode     | trn1 rel error (N=64) | theoretical | CPU rel error (N=64) |
| -------- | --------------------- | ----------- | -------------------- |
| bf16     | ~1.7e-3               | ~1.6e-2     | ~2.2e-3              |
| ozaki    | ~1.7e-3               | ~1.6e-5     | ~1.6e-5              |
| ozaki_hq | ~1.7e-3               | ~2e-9       | ~1.4e-7              |

On CPU the scheme works exactly as theorised — ozaki is ~140× better than bf16, ozaki_hq a
further ~100× better. On trn1 hardware, all three modes give the same result.

## The product-precision constraint

The Ozaki scheme improves precision by decomposing W and x into BF16 high/low parts and
summing correction matmuls. The precision gain depends on one property: **the error in each
matmul is dominated by BF16 input quantization, not by arithmetic rounding inside the
matmul itself.**

On trn1, `nc_matmul` with BF16 inputs rounds each product to BF16 before PSUM accumulation.
Each product `W_h[i,k] × x_h[k,j]` acquires ~u_bf16 of rounding error at the multiply —
before the FP32 PSUM sees it. The FP32 PSUM prevents accumulation error, which is real.
But the bottleneck moved upstream. The split terms can't cancel errors they can't observe.

```
CPU:   BF16(W) × BF16(x) → FP32 product → FP32 PSUM   ✓ input error captured by split
trn1:  BF16(W) × BF16(x) → BF16 product → FP32 PSUM   ✗ product rounding dominates
trn2+: BF16(W) × BF16(x) → FP32 product → FP32 PSUM   ? TF32 product precision — test needed
```

## Why NVIDIA's Ozaki works

NVIDIA's cuBLAS FP64 emulation on Blackwell uses INT8 Tensor Cores where each product of two
low-precision inputs accumulates at higher precision before reaching the FP32 accumulator.
The Ozaki correction terms have something to capture because the per-product precision
*exceeds* the input precision. The same logic held on A100/H100 TF32: 7-bit BF16 inputs,
10-bit TF32 products, FP32 PSUM. Ozaki works because the 3-bit product precision gap is
what the split addresses.

trn1 has no such gap: 7-bit BF16 products from 7-bit BF16 inputs leave nothing for the
correction to capture. trn2's TF32 support means it likely has the gap. "Likely" is not a
precision table — the same characterization test that exposed the trn1 constraint will answer
the trn2 question.

## What the API does now

The ozaki modes stay; the wrapper becomes hardware-aware.

On trn1, calling `precision="ozaki"` or `"ozaki_hq"` now emits a `RuntimeWarning` and falls
back to `precision="bf16"`. The warning text names the constraint (BF16-product PSUM), the
cost (3–6× BF16 latency with no accuracy gain), and the path forward (run
`TestOzakiHQCharacterization` on your instance type; call
`trnfft.set_ozaki_product_precision_verified(True)` to suppress after confirming).

On trn2, if the characterization test shows ozaki_err ≪ bf16_err, the modes enable natively
with no warning. That result also retroactively validates the v0.18/v0.19 accuracy claims —
they were describing hardware that didn't exist yet on the AWS Trainium roadmap.

trn3 with MXFP8/MXFP4 tensor cores is the Trainium analog of Blackwell's INT8 path: the
Ozaki-II higher-order splits that NVIDIA named as a continuing development priority become
the primary trnfft route to FP64-class accuracy.

This is the generational structure NVIDIA navigated: Ampere (BF16 products, Ozaki doesn't
work), Hopper (TF32 products, Ozaki works), Blackwell (INT8 products, Ozaki-II). trnsci is
at the Ampere moment on Trainium silicon.

## The post-FP64 thesis, sharpened

The reframe after this exercise is not "only FP32-sufficient workloads." It's sharper:
**FP64-class accuracy on accelerators is an algorithmic property of low-precision tensor
matmul plus correction, not a property of native FP64 hardware.** The entire industry is
moving this way. The FP64:FP32 ratio on consumer GPUs degraded from 1:8 to 1:64 on Ampere
while the AI boom eroded the datacenter-GPU exception. NVIDIA's October 2025 cuBLAS release
is proof the migration is complete at the top end — native FP64 is retreating from hardware
into algorithm.

trnsci is documenting the same migration on AWS silicon, one generation behind. The trnfft
saga isn't the project being wrong about post-FP64. It's hitting the same generational
constraint NVIDIA Ampere had before TF32 closed the product-precision gap. The thesis is
intact. The implementation needs a hardware version check.

## What's actually next

**Immediate:** hardware-gated warning in the ozaki dispatch (trn1 → warn + fallback; trn2 →
gated on characterization test; API handle to suppress warning after user verification).

**trn2 validation:** `AWS_PROFILE=aws ./scripts/run_precision_characterization.sh trn2`. If
ozaki_err ≪ bf16_err, the CHANGELOG gets a trn2 precision table and the modes enable
natively. That run is the single most informative experiment the project can run right now.

**Stochastic rounding:** Trainium2 advertises SR in its ISA; NKI exposes ISA-level control.
SR converts accumulated rounding error in iterative algorithms from O(N·u) systematic drift
to O(sqrt(N)·u) zero-mean noise — every reduction in trnfft (Bluestein chains, twiddle
accumulation, Stockham reductions) benefits. SR is more distinctive than Ozaki as a
Trainium angle and more broadly applicable: it works for arbitrary iterative algorithms, not
just structured matmul, and it uses a hardware feature Trainium actually exposes. The
numerical analysis of algorithms designed for BF16-product/FP32-PSUM hardware — the Higham
for systolic arrays — doesn't exist. SR is the technical core of the first chapter.

## Takeaway

trn1 is the pre-TF32 moment. The algorithm is correct; the hardware generation is wrong.
cuBLAS proved in October 2025 that Ozaki-style emulation works when product precision
exceeds input precision. trn2/trn3 close that gap on Trainium. The API stays
forward-compatible, the modes stay, the precision claims become conditional on a measurement
any user can run. The thesis isn't retreating — it's being validated one generation at a time,
on a different vendor's timeline.
