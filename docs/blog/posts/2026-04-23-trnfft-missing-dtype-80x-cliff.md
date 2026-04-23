---
date: 2026-04-23
categories: [Deep dive, trnfft]
comments: true
---

# trnfft: the missing dtype and the 80× cliff

The first working version of trnfft's NKI butterfly kernel passed every correctness test.
It was also 80× slower than the PyTorch fallback for batched STFT — a regression so large
the benchmark was assumed to be broken. It wasn't. The kernel was calling NKI once per
batch row in a Python loop, paying full XLA graph compilation overhead for each row.

That discovery, and the fix, is what Phase 1 is mostly about.

<!-- more -->

## The problem

Trainium has no complex dtype. The first consequence is obvious: every complex multiply
becomes four real multiplies. The second consequence is architectural: without a complex
dtype, you must choose how to represent complex tensors, and the choice propagates into
every kernel design decision.

The natural choice — and what trnfft uses — is split real/imaginary: `ComplexTensor`
wraps two real tensors. The alternatives (interleaved, strided) don't fit the 128-partition
systolic array constraint any better; the split representation at least makes the four-
real-matmul structure explicit.

The butterfly problem is different. The per-stage butterfly computes a fixed-size transform
for each batch row. A GPU does this with warp-level parallelism across rows. Trainium's
Tensor Engine works differently: its partition dimension is the "batch" axis of the systolic
array. Exploiting the hardware correctly means the batch rows go into the partition dim, not
into a Python loop.

## What the architecture suggests

**Stationary reuse from split representation.** The complex GEMM identity:

```
C_real = A_real @ B_real − A_imag @ B_imag
C_imag = A_real @ B_imag + A_imag @ B_real
```

Is four matmuls sharing two stationary tiles. `A_real` and `A_imag` each load once into
SBUF and stream against both `B_real` and `B_imag`. The result: 4 SBUF loads instead of
8 for the naïve four-matmul sequence. This reuse only becomes visible when you think in
terms of SBUF tile budgets rather than complex multiply counts.

**The partition dim absorbs the batch.** For the butterfly, the right structure is:

```
(B, n) → reshape → (B × num_groups, m)
```

where `B × num_groups` becomes the partition dimension. The Tensor Engine processes all
batch rows and all butterfly groups in the same systolic pass. The Python loop over `B`
is removed entirely — it was never the right structure for this hardware.

**STFT is a stress test for batching.** STFT decomposes a waveform into overlapping
frames, each of length `n_fft`, and applies FFT to each. The number of frames is
typically large (hundreds to thousands). Before the batched vectorisation fix, each frame
paid the NKI dispatch overhead independently. After the fix, all frames fold into the
partition dimension and disappear into a single kernel call.

## The approach

The core of Phase 1 is three NKI kernels:

**`_complex_gemm_kernel`** — four real matmuls with stationary reuse. The `A_real` tile
is loaded once and used for both `A_real @ B_real` and `A_real @ B_imag`; same for
`A_imag`. PSUM accumulates in FP32. The kernel is hardware-validated and forms the
foundation for the DFT-GEMM fast path (dispatched for N ≤ 256) and the complex linear
layer in `ComplexLinear`.

**`butterfly_stage_kernel`** — batched radix-2 butterfly stage, partition dim over
`B × num_groups`. The critical design decision is to reshape `(B, n)` into
`(total_groups, m)` before dispatch so the partition dimension absorbs all batch rows.
The kernel processes one butterfly stage; the driver calls it log₂(N) times.

**`butterfly_stage_kernel_kahan`** — Dekker 2Prod compensated variant. The compensation
is cheap: `_split` and `_two_prod` run on the Vector Engine, using the idle engine cycles
while the Tensor Engine accumulates the twiddle multiply in PSUM. The Kahan variant
doubles the op count in the compensated sections but barely affects wall-clock time.

A CPU simulator path (`TRNFFT_USE_SIMULATOR=1`) routes all kernels through
`nki.simulate(kernel)(numpy_args)`, catching Python-trace errors (bad kwargs, dropped
ops, shape mismatches) without a hardware round-trip.

## What didn't work

**Per-batch butterfly.** The first kernel called the NKI butterfly once per batch row.
It passed 70/70 benchmark cases. The STFT benchmark showed 80× worse-than-baseline
performance — each of the ~1000 frames was paying the XLA graph compilation cost
independently. The per-batch structure was never the right design; the benchmark needed
to be the one to show it.

**The Kahan kernel broke in NKI 0.3.0.** The `_split` and `_two_prod` helpers were
defined as inner functions inside the `@nki.jit` body, which NKI 0.2.x allowed silently.
NKI 0.3.0 added a restriction on inner function definitions. The runtime error:

```
RuntimeError: NKI does not support inner function definitions;
move function definition outside this function
```

appeared during a precision characterization run in 2026, not during initial development.
The fix was two lines — hoist the helpers to module scope — but finding it required
knowing what "inner function definition" meant in this context. SDK-pinning note: the
`+<hash>` suffix in NEFF cache paths (`neuronxcc-2.24.5133.0+58f8de22`) is the reliable
version key; semantic versions alone don't capture breaking changes in NKI's parser.

**The simulator doesn't enforce MLIR constraints.** `nki.simulate` runs through a Python
interpreter that approximates NKI semantics. It will not catch: partition-dim violations,
`nl.load_transpose2d` from kernel-local `shared_hbm` buffers, or any constraint that
the MLIR verifier enforces at compile time. Every kernel that passes the simulator needs
a hardware validation run before being committed to a dispatch path. This cost is
unavoidable; the simulator's value is in the iteration speed it provides for the cases
it does catch.

## Numbers

Hardware bench: trn1.2xlarge, Neuron SDK 2.29.0, NKI 0.3.0.

**Batched FFT after partition-dim vectorisation (v0.13, SDK 2.29):**

| Shape          | trnfft (µs) | Butterfly path (µs) | PyTorch (µs) | vs butterfly | vs PyTorch |
| -------------- | ----------- | ------------------- | ------------ | ------------ | ---------- |
| (B=32, N=128)  | 1 088       | 17 153              | 6 866        | 15.8×        | 6.3×       |
| (B=32, N=256)  | 1 214       | 17 375              | 13 710       | 14.3×        | 11.3×      |

**STFT after partition-dim vectorisation (v0.13, SDK 2.29):**

| n_fft | trnfft (µs) | n_fft=512 butterfly (µs) | PyTorch (µs) | vs butterfly | vs PyTorch |
| ----- | ----------- | ------------------------ | ------------ | ------------ | ---------- |
| 128   | 1 127       | 14 791                   | 6 987        | 13.1×        | 6.2×       |
| 256   | 1 214       | 15 196                   | 12 746       | 12.5×        | 10.5×      |

The 15.8× batched FFT win is what the partition-dim vectorisation was designed to produce.
The 13.1× STFT win is the reason it was worth finding: STFT is the highest-value real
workload for trnfft (speech enhancement, spectral analysis), and it was the workload that
found the 80× cliff first.

**Where this is well-indexed:** any workload that calls FFT many times at the same N — batched
FFT, STFT, spectral autoencoders, multi-channel signal processing. The partition-dim
absorption of `B × num_groups` is most effective when B × num_groups ≥ 128 (fills the
systolic array partition).

**Where it is not well-indexed:** single unbatched FFT at large N; non-power-of-2 N in
BF16/FP16 where Bluestein errors compound; batch sizes that don't fill the partition
dimension (B < 4 at N=256).

## What's next

Phase 1 established the correctness baseline. v0.12–v0.17 (published separately)
pushed the DFT-GEMM fast path, Stockham radix-4/8, BF16 PSUM-FP32 output, and iterative
refinement. The remaining open questions: Ozaki-scheme FP64 emulation from BF16 inputs
(accumulate BF16 Ozaki components into FP32 PSUM, combine for near-FP64 accuracy), and
multi-NeuronCore distribution for N > 4096.

Issues tracking the above are open on [trnsci/trnsci](https://github.com/trnsci/trnsci/issues).

## Takeaway

The 80× cliff happened because the kernel was right about the computation and wrong about
the structure. On Trainium the batch dimension belongs in the partition dim, not in a
Python loop — and the only way to discover that was to run the STFT benchmark and read
the number. The split real/imag representation was the constraint that forced the stationary
reuse design. Neither was obvious from the cuFFT source; both became obvious once the
hardware said no.
