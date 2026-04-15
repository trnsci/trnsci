---
date: 2026-04-14
categories: [Deep dive, trnfft]
comments: true
---

# trnfft: FFT on hardware that doesn't want to be an FFT engine

Between v0.7 and v0.12, [trnfft](https://trnsci.dev/trnfft/)'s NKI story moved from one per-row butterfly dispatch into a batched butterfly + fused DFT-as-GEMM stack with opt-in Kahan-compensated precision — all hardware-validated on trn1.2xlarge. What landed on silicon looks very little like cuFFT: no complex dtype, no thread-per-butterfly, no bit-reversal in the fast path. What Trainium's architecture — four programmable engines, a fixed 128-partition × 512-moving tile, explicit SBUF/PSUM memory — suggested was a different decomposition, and this post is the retrospective on what that turned out to be. Readers evaluating Trainium for spectral workloads or maintaining sibling NKI libraries will find the architectural framing directly portable.

<!-- more -->

## The problem

FFT is load-bearing for spectral PDE solvers, speech enhancement, radar, convolutional nets, Ewald sums. `torch.fft` and `cuFFT` are the reference APIs; both assume a native complex dtype and a massively threaded SIMT unit — a radix-2 Cooley-Tukey butterfly in that model is "one warp per butterfly, thousands in flight, one complex register per element, bit-reverse the input index".

Trainium has neither assumption. There is no hardware complex dtype — every complex tensor must be paired real tensors. The execution model isn't SIMT; it's four cooperating engines (Tensor, Vector, Scalar, DMA) with a fixed tile shape of 128 partitions × up to 512 free elements, and a whole-program NKI compiler that schedules their dependencies. A naive "one thread per butterfly" port runs headfirst into partition underutilization: filling 128 partitions with radix-2 pairs requires reaching across 64 simultaneous butterfly groups, not across threads within one. "Port cuFFT" is a bad starting point. The interesting question is what the architecture suggests if cuFFT didn't exist.

## What the architecture suggests

**No complex dtype.** Every complex tensor becomes two real tensors side by side — the `ComplexTensor(real, imag)` wrapper is a split-real/imag holder, not a new primitive. Complex multiply is four real multiplies and two real adds. Complex matmul is four real matmuls and two real matrix adds. On Trainium this is the native shape, because the real Tensor Engine op (stationary `nisa.nc_matmul` with PSUM accumulation) has no complex-typed equivalent and would unpack to four real matmuls in the compiler anyway.

**The 128-partition tile prefers many independent small operations.** A radix-2 butterfly acts on pairs, so filling the partition dim means *flattening across butterfly groups*. An FFT of size N at stage s has `N / 2^(s+1)` independent butterfly groups; all of them run in parallel along the partition dim. For batched FFT (`(B, N)` input) the math compounds: `B × num_groups` lands in the partition slot, naturally saturating the tile at reasonable B. The kernel runs once per stage, vectorized across every group in every batch row. What looks like a radix-2 algorithm on paper is really "one big elementwise-per-stage over a tall partition-dim".

**Four engines mean a butterfly stage issues two ops in parallel.** The Tensor Engine does the twiddle × odd-element multiply. The Vector Engine does the `e + prod` / `e - prod` butterfly combination. These aren't sequential the way they are on a GPU warp — the NKI compiler schedules them on separate engines with DMA prefetch overlapping from HBM. The architectural primitive isn't "one butterfly"; it's "fill the Tensor Engine pipeline with twiddle multiplies while the Vector Engine consumes their outputs".

**PSUM is fp32 and has a ceiling.** The Tensor Engine accumulates products in PSUM in fp32. Generous for one matmul, but a long dependency chain — three power-of-2 FFTs composed in Bluestein's chirp-z decomposition — compounds rounding error. At N ≥ 500 the default Bluestein path in fp32 accumulates roughly 2 × 10⁻² relative error against a scipy fp64 reference. That's a hardware-sizing observation as much as an algorithm choice, and it's what motivated the Kahan-compensated butterfly shipped in the `"kahan"` precision mode: the Vector Engine folds a 2Prod compensation step into the butterfly kernel cheaply, because the extra adds land on the engine that would otherwise be idle while the Tensor Engine is busy. Compensation is architecturally-free in a way it wouldn't be on a GPU.

**NEFF cache makes the first call expensive.** Every distinct kernel signature compiles to a NEFF binary on first invocation; subsequent calls are cache hits. This favors plan-based execution (FFTW/cuFFT-style) over kernel-per-call — `trnfft.plan` caches plans keyed on `(N, inverse)`.

## The approach

v0.8.0 landed four NKI kernels built on the above:

1. `_complex_gemm_kernel` — 4 real `nisa.nc_matmul` calls into 2 fp32 PSUM accumulators with stationary-tile reuse: A_real stays in the stationary slot while B_real and B_imag stream through as moving operands, then A_imag stationary while -B_imag and B_real stream. Four SBUF loads become two per PSUM pair, saving HBM traffic.
2. `_complex_mul_kernel` — fused elementwise complex multiply, single kernel, one SBUF round-trip instead of six for the naive implementation.
3. `butterfly_stage_kernel` — batched radix-2 DIT stage. Input `(B, N)` flattens to `(B * num_groups, m)`; partition dim is the combined batch-and-group axis, free dim is the intra-group element. Twiddles are host-broadcast across partition rows so partition dims match when the Vector Engine fires the complex-multiply + butterfly.
4. `butterfly_stage_kernel_kahan` — compensated variant. Dekker 2Prod split of each `t × o`, then adds the rounded-off low-order part back into the complex sum. Doubles the butterfly op count, runs mostly on the Vector Engine. Opt-in via `trnfft.set_precision("kahan")`.

STFT, batched FFT, and fft2/fftn all run through a single `_cooley_tukey_nki` dispatcher that flattens any leading batch dims into the partition slot and calls the kernel once per stage.

The deliberate tradeoff: radix-2, not radix-4 or larger. Radix-2 is simpler and fills the Vector Engine cleanly, but at large N the per-stage launch count is `log₂(N)` and each launch pays NKI dispatch overhead. For N ≥ 1024 this starts to dominate — which motivates Thread B (Stockham radix-4) under active development.

## Implementation

The butterfly stage kernel is the load-bearing piece. Stripped to its essential shape:

```python
@nki.jit
def butterfly_stage_kernel(x_re, x_im, tw_re_bcast, tw_im_bcast, n, stage):
    B, _ = x_re.shape
    m = 1 << (stage + 1)
    half = m >> 1
    num_groups = n // m
    total_groups = B * num_groups

    out_re = nl.ndarray((B, n), dtype=x_re.dtype, buffer=nl.shared_hbm)
    out_im = nl.ndarray((B, n), dtype=x_im.dtype, buffer=nl.shared_hbm)

    # Flatten (B, n) -> (total_groups, m). Partition dim = total_groups;
    # every partition row is one independent butterfly group.
    x_re_2d = x_re.reshape((total_groups, m))
    # ... same for x_im_2d, out_re_2d, out_im_2d ...

    groups_chunk = total_groups if total_groups <= PMAX else PMAX
    n_partition_tiles = total_groups // groups_chunk

    for p in nl.affine_range(n_partition_tiles):
        p_off = p * groups_chunk
        p_end = p_off + groups_chunk
        for k in nl.affine_range(half):
            t_re_col = nl.load(tw_re_bcast[p_off:p_end, k:k+1])
            t_im_col = nl.load(tw_im_bcast[p_off:p_end, k:k+1])
            e_re = nl.load(x_re_2d[p_off:p_end, k:k+1])
            e_im = nl.load(x_im_2d[p_off:p_end, k:k+1])
            o_re = nl.load(x_re_2d[p_off:p_end, k+half:k+half+1])
            o_im = nl.load(x_im_2d[p_off:p_end, k+half:k+half+1])

            # Complex multiply, one butterfly column at a time.
            prod_re = nl.subtract(
                nl.multiply(t_re_col, o_re), nl.multiply(t_im_col, o_im))
            prod_im = nl.add(
                nl.multiply(t_re_col, o_im), nl.multiply(t_im_col, o_re))

            # Even = e + prod, odd = e - prod.
            nl.store(out_re_2d[p_off:p_end, k:k+1],
                     value=nl.add(e_re, prod_re))
            # ... three more stores for out_im even, out_re odd, out_im odd ...
    return out_re, out_im
```

(Apache 2.0, full source: [`trnfft/nki/butterfly.py`](https://github.com/trnsci/trnfft/blob/main/trnfft/nki/butterfly.py).)

The partition-dim-is-total_groups pattern is the Trainium-native bit. A GPU would nest `k` as the outer loop and thread-parallelize over groups; here the partition dim *is* the group dim, and `k` iterates through butterfly positions within each group. For non-power-of-2 `B` (STFT's 33-frame case), the host pads to the next multiple of 128 — zero-padding is cheaper than supporting irregular partition counts in NKI 2.24, and the padding is discarded after the stage.

## What didn't work

**FP32 Bluestein precision.** The single largest surprise. Bluestein chains three power-of-2 FFTs and a pair of chirp multiplies to handle arbitrary-N FFT; composed in fp32 the relative error grows roughly as O(N). At N = 500 error reaches ~1.4 × 10⁻²; at N = 8193 ~2.2 × 10⁻³. The test suite had silently papered over this with `tol = 2e-2` for N ≥ 500 — a gap marked "expected fp32 degradation" rather than fixed. v0.11.0 shipped three precision modes in response: `"fast"` keeps the fp32 path, `"double"` promotes Bluestein host math to fp64 (~5 × 10⁻¹³ at any N, 10 orders of magnitude tighter, but host-side only — NKI stays fp32), and `"kahan"` uses the compensated butterfly. `"kahan"` on CPU is equivalent to `"fast"` because the chirp multiplies aren't where the error lives — the butterfly chain is — and only on NKI does the compensation actually engage.

**NKI kernels silently detached autograd.** Every `@nki.jit` kernel returns a tensor from `nl.shared_hbm` with no registered `grad_fn`. Forward passes work fine; `loss.backward()` raises `element 0 of tensors does not require grad and does not have a grad_fn` on the first backward call (issue #56). Invisible for inference-only users. v0.10.1 wrapped every kernel in a `torch.autograd.Function` subclass with analytic adjoints (`dA = dC @ conj(B)ᵀ` for GEMM, `dx = ifft(dy) × n` for FFT).

**`torch.Tensor.unfold` has no XLA backend.** `trnfft.stft` originally used `x.unfold(dim, size, step)` for frame extraction; that raised `aten::unfold not implemented for XLA` the moment anyone set `set_backend("nki")`. Replaced with explicit `torch.arange`-based frame-index construction in PR #44.

**The NKI 0.3.0 migration surfaced three API deltas that broke our kernels.** `nisa.nc_matmul` went kwargs-only with in-place accumulation (`dst=, stationary=, moving=, accumulate=True`). `nl.copy` now returns a view, so PSUM → SBUF materialization requires `nisa.tensor_copy(dst=, src=)` with a pre-allocated SBUF destination. Python `*` / `+` / `-` operators are no longer defined on `NkiTensor` — every complex multiply and butterfly expression rewrote with explicit `nl.multiply` / `nl.add` / `nl.subtract`. All three were surfaced by the CPU simulator (`NKI_SIMULATOR=1`) in minutes. AWS Neuron team: migration release notes calling out operator-overload changes would save downstream libraries a CI iteration each.

**Full-size DFT-as-GEMM capped at N = 256.** v0.12.0 introduced a DFT-as-GEMM fast path (`x @ W` where `W` is the `N × N` DFT matrix) that beats butterfly by 2.2–5.7× for N ∈ {8..128} and still wins 5.3× at N = 1024 — but fp32 `nisa.nc_matmul` accumulation at N = 1024 reaches ~2.2% relative error, breaking the 1e-3 test tolerance. The win is perf-real and precision-blocked. Routing past the 256 ceiling is what motivates Stockham radix-4 (Thread B), where a `log₄(N)` chain accumulates only O(r²) error per stage.

## Numbers

Steady-state on trn1.2xlarge, `neuronxcc 2.24.5133.0`, Deep Learning AMI Neuron PyTorch 2.9. All numbers after warmup — first call pays NEFF compile, these are mean of subsequent steady-state invocations.

### v0.7.0 → v0.8.0 — batched kernel landing

| Operation | v0.7.0 | v0.8.0 | Speedup |
| --- | ---: | ---: | ---: |
| fftn 32×64×64 | 52.3 s | 70.8 ms | 738× |
| fft2 1024×1024 | 32.4 s | 545 ms | 59× |
| batched FFT (128×1024) | 2.07 s | 52 ms | 39× |
| STFT (16 000 samples) | 765 ms | 28 ms | 27× |

v0.7.0 ran the butterfly kernel in a Python for-loop over batch rows — one kernel dispatch per (row, stage) pair. v0.8.0's batched kernel removes that loop; what looks like a 27× STFT speedup is "fix the dispatch pattern so the kernel sees the batch".

### v0.12.0 — DFT-as-GEMM for small N

| Shape | DFT-GEMM | Butterfly | Speedup |
| --- | ---: | ---: | ---: |
| N = 256, B = 1 | 1 882 μs | 9 862 μs | 5.2× |
| B = 32, N = 256 | 2 049 μs | 29 210 μs | 14.3× |
| STFT (n_fft = 256, 16 k samples) | 2 445 μs | 30 514 μs | 12.5× |

The batched/STFT columns are where the architectural thesis paid off. DFT-GEMM at `(B = 32, N = 256)` collapses the entire batch into one `nisa.nc_matmul` — the partition-dim finally saturates, and the per-batch cost drops 12–14× against the butterfly chain. `torch.fft.fft` on the x86 bench host (running MKL) is still faster than Trainium for isolated one-shot FFT calls — 10–80 μs cold, roughly 50–300× lower than the NKI path. The architectural story isn't "beat MKL on cold calls"; it's "keep data on-chip across long operator chains".

## What's next

- **Phase 2 #52 — Kahan / Neumaier summation for long Bluestein chains.** Partial delivery in v0.11.0 (precision-modes API + compensated butterfly kernel). The Kahan butterfly compiles on NKI 2.24.5133.0 and agrees with the stock kernel at FP32 rtol across the 17-test neuron suite; whether the 2Prod compensation *actually* reduces FP32 FFT error on silicon is the measurement question still open in issue #58.
- **Thread B — Stockham radix-4 (v0.13 candidate).** A POC kernel shipped this week, CPU reference + NKI port green under the CPU simulator. 5 kernel launches at N = 1024 vs 10 for butterfly, and log₄(N) fp32 accumulation vs the O(N²) ceiling that caps DFT-GEMM at N = 256. Hardware validation pending an AWS DLAMI with Neuron SDK 2.29.
- **Phase 3 #53 — perf.** Plan reuse across shapes, streaming large FFTs that exceed HBM, NEFF cache hit-rate instrumentation.
- **Phase 4 #54 — multi-chip FFT for N > 2²⁰** via cross-core collectives.
- **Phase 5 #55 — trn2.** Larger SBUF, different systolic array sizing; some constraints loosen.

## Toolchain observations worth flagging

- **The CPU simulator (`NKI_SIMULATOR=1`) is the biggest DX change since NKI 0.3.0 shipped.** It runs kernels on CPU without NEFF compile or hardware dispatch. Kernel iteration that used to need a 5–10 min SSM round-trip to trn1 now takes seconds locally; GitHub Actions can run simulator-backed correctness tests on `ubuntu-latest` with no AWS access. It surfaced every one of the NKI 0.3.0 API deltas that broke trnfft's kernels in a CI run lasting under two minutes. If you're shipping a library that depends on NKI, add a simulator job before you ship anything else.
- **`nisa.tensor_copy` signature.** The switch from `nl.copy(psum, dtype=...)` to `nisa.tensor_copy(dst=sbuf, src=psum)` is a meaningful behavioral change — `nl.copy` now returns a view — and the migration guide didn't call it out. Downstream libraries with PSUM → SBUF materialization patterns will hit the same issue.

## Takeaway

Trainium was designed for large-model training and inference; FFT is not the workload it was built for. A CUDA-style radix-2 butterfly with one thread per butterfly and thousands in flight maps badly onto a 128-partition systolic array with a fixed engine hierarchy. What works instead is treating the partition dim as "total independent butterfly groups across every batch row, vectorized", the four engines as a scheduling substrate that fires multiply and add in parallel, and the fp32 PSUM as a real ceiling that constrains algorithm choice. Every trnfft kernel shipped so far is a variation on that theme — split real/imag, flatten across groups, route compute between engines, and respect the precision ceiling. The rest of the roadmap is about pushing that thesis into medium and large N where it's not obvious the butterfly is still the right decomposition.
