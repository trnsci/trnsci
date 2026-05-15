---
date: 2026-04-30
categories: [Deep dive, trnfft]
comments: true
---

# trnfft: what trn1 and trn2 tell us about the Ozaki frontier

The [v0.18](../2026-04-25-trnfft-ozaki-eight-runs/) and [v0.19](../2026-04-28-trnfft-two-level-ozaki/)
posts claimed hardware precision of O(sqrt(N)·u_bf16²) ≈ 1.6e-5 and O(sqrt(N)·u_bf16⁴) ≈ 2e-9
for the Ozaki modes. The trn1 hardware measurement says those numbers are wrong — both modes
deliver ~1.7e-3, equivalent to single-pass BF16. trn2 was then tested with the same characterization.
The result is identical. Both generations. The conclusion is still not "Ozaki is a dead end" —
but the generational gap theory needs revision.

<!-- more -->

## The measurements

`TestOzakiHQCharacterization` — actual Ozaki kernels, not the BF16 fallback:

| mode     | trn1 rel error (N=64) | trn2 rel error (N=64) | theoretical | CPU rel error (N=64) |
| -------- | --------------------- | --------------------- | ----------- | -------------------- |
| bf16     | ~1.5e-3               | ~1.5e-3               | ~1.6e-2     | ~2.2e-3              |
| ozaki    | ~1.7e-3               | ~1.7e-3               | ~1.6e-5     | ~1.6e-5              |
| ozaki_hq | ~1.7e-3               | ~1.7e-3               | ~2e-9       | ~1.4e-7              |

On CPU the scheme works exactly as theorised — ozaki is ~140× better than bf16. On both
Trainium generations, all three modes give the same result. Ozaki is marginally *worse* than
plain BF16 on hardware (0.9×), not better — the extra matmuls add a small amount of noise.

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
correction to capture. trn2's TF32 hardware support was the basis for thinking it might have
the gap. It doesn't. The NKI Bootcamp materials say "BF16 products are computed at full
precision internally" — this refers to the FP32 PSUM accumulator, not the individual multiply
precision. Both generations round BF16×BF16 products to BF16 before PSUM.

The generational structure assumed (Ampere → Hopper → Blackwell) doesn't map cleanly onto
the Trainium roadmap. NVIDIA's TF32 is a specific architectural choice to raise product
precision for exactly this class of algorithm. Trainium hasn't made that choice through trn2.
Whether trn3 MXFP8 changes the calculus is the next open question — but MXFP8 operates at a
different scale (group quantization with INT8 microscales), not TF32-style per-product promotion.

## What the API does now

The ozaki modes stay, hardware-gated. Calling `precision="ozaki"` or `"ozaki_hq"` emits a
`RuntimeWarning` on any unverified instance and falls back to `precision="bf16"`. The warning
names the constraint, the cost (3–6× BF16 latency, no accuracy gain), and the verification
path (`TestOzakiHQCharacterization` + `set_ozaki_product_precision_verified(True)`). On trn1
and trn2, there is no configuration under which verification should be set to True.

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

**trn3 MXFP8 (`nisa.nc_matmul_mx`):** The NKI Bootcamp documents `nisa.quantize_mx()` and
`nisa.nc_matmul_mx()` — group-quantized FP8 inputs with INT8 microscale factors and FP32
accumulation. This is structurally different from TF32 product-precision promotion and may
or may not give Ozaki the gap it needs. The characterization test can answer this on a trn3
instance when available.

**Stochastic rounding:** trn3's ISA includes a stochastic rounding instruction (NKI Bootcamp
Day 3). The bootcamp describes it as "load/store rounding state for checkpointing and replay"
— suggesting training reproducibility as the primary use case, but the instruction itself is
general. SR converts accumulated rounding error in iterative algorithms from O(N·u) systematic
drift to O(sqrt(N)·u) zero-mean noise. This is a different precision strategy than Ozaki
(which targets input quantization error); SR targets accumulation error in iterative loops.
For trnfft — Bluestein chains, Stockham reductions, twiddle accumulation — SR is more
broadly applicable and doesn't require the product-precision gap Ozaki does.

## Takeaway

trn1 is the pre-TF32 moment. The algorithm is correct; the hardware generation is wrong.
cuBLAS proved in October 2025 that Ozaki-style emulation works when product precision
exceeds input precision. trn2/trn3 close that gap on Trainium. The API stays
forward-compatible, the modes stay, the precision claims become conditional on a measurement
any user can run. The thesis isn't retreating — it's being validated one generation at a time,
on a different vendor's timeline.
