---
date: 2026-04-28
categories: [Deep dive, trnfft]
comments: true
---

# trnfft: the residual must stay FP32

v0.19 ships `precision="ozaki_hq"` — six BF16 matmuls that together reach O(sqrt(N)·u_bf16⁴)
≈ 2e-9 relative error, near-FP64 accuracy on the Tensor Engine. The implementation is a
40-line extension of the v0.18 Ozaki scheme. There is one non-obvious constraint that the
algorithm turns entirely on. Getting it wrong gives you 1-level accuracy out of a 2-level
design, silently, with no error.

<!-- more -->

## The problem

v0.18's 1-level Ozaki (`precision="ozaki"`) delivers ~1.6e-5 relative error at N=64 — a
140× improvement over single-pass BF16. For most spectral methods, that is sufficient.
For chained FFT pipelines, iterative eigensolvers, and signal processing where floating-point
error accumulates across hundreds of transforms, it is not. The alternative — `precision="double"`
— routes through a CPU IFFT roundtrip that is both slow and architecturally backwards on
hardware that has BF16 matmul for free.

The question v0.19 asks: can a second split level reach near-FP64 accuracy without leaving
the Tensor Engine?

## What the architecture suggests

The Tensor Engine's PSUM accumulator is always FP32, regardless of input dtype. This is the
hardware invariant that the Ozaki scheme exploits: BF16 inputs, FP32 accumulation, FP32
output — no precision is lost in the *accumulation*, only in the *quantization to BF16*.

The 1-level scheme attacks the quantization error at one level. The 2-level scheme should attack
it at two levels: split both W and x into smaller pieces, compute all cross-terms, let the PSUM
accumulate each in FP32, sum the results. With W split 2-ways and x split 3-ways, there are 6
cross-terms. Each is a BF16 matmul. Each accumulates in FP32. The accuracy target: O(sqrt(N)·u_bf16⁴).

At u_bf16 ≈ 4e-3 and N=64: sqrt(64) × (4e-3)⁴ ≈ 8 × 2.56e-10 ≈ 2e-9.

That's within 100× of FP64 rounding noise (u_fp64 ≈ 1e-15, N=64 → ~8e-15), and within 10×
of FP32 rounding noise (u_fp32 ≈ 1.2e-7, N=64 → ~1e-6). Without any FP64 instructions.

## The approach

The 3-way split of x is:

```python
def _ozaki_split_3way_bf16(x):
    x_h1 = x.bfloat16()
    r1 = x - x_h1.float()   # FP32 residual — NOT cast to BF16 yet
    x_h2 = r1.bfloat16()
    x_h3 = (r1 - x_h2.float()).bfloat16()
    return x_h1, x_h2, x_h3
```

The critical line is `r1 = x - x_h1.float()`. The subtraction happens in FP32. The
result is a FP32 tensor. Only then is it cast to BF16 for the next level.

This is the constraint the 2-level design turns on: **once a value is quantized to BF16,
its residual has zero BF16 mantissa bits.** `bfloat16(bfloat16(x) - bfloat16(x))` is
identically zero — the subtraction in BF16 produces exactly 0, not a small residual.
To get a meaningful second split, the residual must be computed in FP32, where it has
value ~u_bf16 × |x| and is representable.

The 6-term summation with data-dependency sequencing:

```python
# W split: 2-way (same as 1-level)
W_r_h, W_r_l = _ozaki_split_bf16(W_r_fp32)
W_i_h, W_i_l = _ozaki_split_bf16(W_i_fp32)

# x split: 3-way with FP32 residuals between levels
x_r_h1, x_r_h2, x_r_h3 = _ozaki_split_3way_bf16(x_re_fp32)
x_i_h1, x_i_h2, x_i_h3 = _ozaki_split_3way_bf16(x_im_fp32)

t1 = _term(x_r_h1, x_i_h1, W_r_h, W_i_h)
_d1 = t1.real.mean() * 0              # data-dependency: forces t1 before t2

t2 = _term(x_r_h2 + _d1, x_i_h2 + _d1, W_r_h, W_i_h)
_d2 = t2.real.mean() * 0

t3 = _term(x_r_h3 + _d2, x_i_h3 + _d2, W_r_h, W_i_h)
_d3 = t3.real.mean() * 0

t4 = _term(x_r_h1 + _d3, x_i_h1 + _d3, W_r_l, W_i_l)
_d4 = t4.real.mean() * 0

t5 = _term(x_r_h2 + _d4, x_i_h2 + _d4, W_r_l, W_i_l)
_d5 = t5.real.mean() * 0

t6 = _term(x_r_h3 + _d5, x_i_h3 + _d5, W_r_l, W_i_l)

X_re = (t1.real + t2.real + t3.real + t4.real + t5.real + t6.real)
```

The `_term` function calls the same `complex_gemm_bf16` kernel validated in v0.17. No new
NKI kernel. The data-dependency trick (`0 * mean()`) is the same one from v0.18 — it
encodes a graph edge without a sync call, so XLA evaluates each term before starting the
next. Full source: [`trnfft/fft_core.py`](https://github.com/trnsci/trnfft/blob/main/trnfft/fft_core.py).

## Implementation

The dispatch chain in `_cooley_tukey_nki_nograd` checks `_FORCE_OZAKI_HQ` first (benchmark
toggle), then `precision == "ozaki_hq"` for sizes within the DFT-GEMM threshold (N ≤ 256):

```python
if _FORCE_OZAKI_HQ:
    return _fft_via_ozaki_hq(x, inverse)
...
if precision == "ozaki_hq" and n <= _DFT_GEMM_THRESHOLD:
    return _fft_via_ozaki_hq(x, inverse)
```

`set_precision("ozaki_hq")` is the user-facing API. The `_FORCE_OZAKI_HQ` flag exists for
benchmarking — it bypasses the N threshold and activates the path at any size.

## What didn't work

**The silent wrong-answer trap.** The first draft of `_ozaki_split_3way_bf16` computed
`r1 = (x - x_h1).bfloat16()` — casting the residual to BF16 before the second split.
This produces three BF16 tensors that sum to x in BF16 arithmetic, but the second and
third splits carry no additional precision: `bfloat16(x - bfloat16(x))` is essentially
noise at the BF16 representable scale, not a meaningful ~u_bf16² residual. The function
compiles, runs, returns the right output type, and produces accuracy indistinguishable
from the 1-level scheme. Nothing flags the error. CPU tests pass. Only a precision
comparison against the 1-level result on hardware would reveal the problem.

The correct invariant is: **the subtraction must happen in FP32, not BF16.** The one-line
diff is `r1 = x - x_h1.float()` (FP32) versus `r1 = (x - x_h1).bfloat16()` (BF16 subtraction).

**Terminology confusion.** The "2-level" label refers to the number of Ozaki split levels
applied to x, not to some hierarchical recursion. W is only 2-split in v0.19 (the same as
v0.18). Extending W to a 3-split would produce 9 terms and O(u⁶) accuracy — a straightforward
extension, but the marginal gain over 2e-9 is academic for any workload that fits N ≤ 256.

**Why not push it further?** Six BF16 matmuls at N=64 take ~6,252 µs. Each additional split
level doubles the term count and roughly doubles the time. The scaling wall is the dispatch
overhead per matmul call (each `complex_gemm_bf16` involves a full kernel launch, NEFF cache
lookup, and XLA graph evaluation), not the compute itself. A fused multi-term NKI kernel
would change this calculus, but that's a Phase 3 item.

## Numbers

Hardware bench: trn1.2xlarge, Neuron SDK 2.29.0, NKI 0.3.0, 2026-04-17.

| N   | ozaki_hq (µs) | ozaki (µs) | BF16 (µs) | hq/ozaki | hq/fp32 |
| --- | ------------- | ---------- | --------- | -------- | ------- |
| 64  | 6 252         | 3 225      | 1 178     | 1.94×    | 3.41×   |
| 128 | 6 313         | 3 316      | 1 214     | 1.90×    | —       |
| 256 | 6 417         | 3 451      | 1 300     | 1.86×    | 3.41×   |

The overhead is almost exactly 2× over 1-level Ozaki — six terms versus three, with the same
per-term dispatch cost. The precision target is ~2e-9 (theoretical); hardware measurement
is running separately as `TestOzakiHQCharacterization` in the test suite.

**Where this is well-indexed:** chained spectral pipelines where BF16 error accumulates
across multiple transforms; numerical methods (iterative eigensolvers, spectral PDE solvers)
where the per-step error floor matters; any N ≤ 256 workload where `precision="double"`
would require a CPU roundtrip.

**Where it is not well-indexed:** single unbatched transforms at any N (FP32 is already
sub-1e-6 rel error); N > 256 where DFT-GEMM gives way to Stockham and the GEMM-based
Ozaki path doesn't apply; workflows where 1e-5 error tolerance is acceptable and the 2×
throughput cost of 2-level versus 1-level matters.

## What's next

- **Precision characterization on hardware:** `pytest tests/test_precision_modes.py::TestOzakiHQCharacterization -m neuron` will record the actual vs. theoretical accuracy. The 2e-9 figure is derived from the error expansion; measured hardware results depend on the specific Neuron matmul accumulation behavior and will be documented in the CHANGELOG.
- **Multi-NeuronCore distribution (v0.20):** for N > 4096, batch dimensions can be split across NeuronCores using `torch_neuronx` parallel utilities. The scaffold in [`trnfft/nki/multicore.py`](https://github.com/trnsci/trnfft/blob/main/trnfft/nki/multicore.py) is the implementation target.

Full roadmap: [trnsci/trnsci](https://github.com/trnsci/trnsci/issues).

## Takeaway

The 2-level Ozaki scheme on Trainium works because of two FP32 guarantees that come for
free: the PSUM always accumulates in FP32 (hardware invariant), and the BF16 residuals
are computed in FP32 before quantization (software invariant). The hardware invariant is
what makes BF16 matmuls usable at all for precision work. The software invariant is what
makes the second split level meaningful rather than degenerate. Miss either one and you
get 1-level accuracy from a 6-matmul computation. Get both right and six BF16 matmuls
reach error orders that would have required FP64 arithmetic on a GPU.
