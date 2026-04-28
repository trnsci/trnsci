---
date: 2026-04-27
categories: [Deep dive, trntensor]
comments: true
---

# trntensor v0.15.0: the caller specifies the error; the library picks the mode

The precision scaffolding built across v0.11.0–v0.14.0 gave trntensor four accumulation modes. v0.15.0 gives the caller a single number. The question "which mode?" is now the library's problem, not the caller's.

<!-- more -->

## The problem

The four-mode precision surface — `"fast"`, `"sr"`, `"dd"`, `"kahan"` — was complete after v0.14.0, but complete in the way a toolbox is complete: every tool was there, and using the right one correctly was entirely on the caller. Selecting `precision="sr"` for a BF16 K=512 contraction is correct, but it requires knowing that K=512 with BF16 unit roundoff (u = 2⁻⁸) means fast rounding's Wilkinson bound reaches ~200% relative error, while stochastic rounding brings it down to ~8.8% in expectation. Most callers don't carry this arithmetic in their head.

The CUDA analogy is `cublasMath_t` — the caller picks `CUBLAS_TENSOR_OP_MATH` vs `CUBLAS_PEDANTIC_MATH` vs `CUBLAS_DEFAULT_MATH` based on some mix of empirical testing and vague intuition. trntensor's four modes had the same shape: named, documented, but without a bridge from "I need my contraction to be accurate to 1%" to "which mode achieves that."

The [v0.13.0 post](https://trnsci.dev/blog/trntensor-v0130-precision-all-the-way-down/) closes the precision-threading gaps. The [v0.14.0 post](https://trnsci.dev/blog/trntensor-v0140-four-modes-four-mocks--completing-the-precision-contract/) completes the CPU test surface. v0.15.0 closes the gap between a user's accuracy requirement and the mode that satisfies it.

## What the architecture suggests

The four precision modes map to four concrete facts about Trainium's hardware:

1. **`"fast"`**: BF16 multiply-adds accumulate into FP32 PSUM. One NKI matmul dispatch. Error: K·u_bf.
2. **`"sr"`**: same kernel, stochastic PSUM→SBUF downcast via `nisa.activation(round_mode="stochastic")`. No extra dispatch cost. Error: √K·u_bf (mean-zero, statistical).
3. **`"dd"`**: Ozaki-2 — four BF16 matmul dispatches whose sum approximates the FP32-accurate result. On hardware: one fused NKI program (trnblas#22) that holds all four correction tiles in PSUM before the single SBUF store. Error: K·u_bf².
4. **`"kahan"`**: FP64 promotion before contraction. Error: K·u_f64.

Each mode has a known theoretical error bound. Those bounds exist independent of any particular caller or workload. The natural step is to expose them as the selector itself: given a dtype, a K, and a target error, the cheapest-sufficient mode is a lookup, not a judgment call.

This is what makes `target_forward_error=` a hardware-informed API, not just a convenience wrapper. The table below is the hardware's cost/accuracy tradeoff expressed in the caller's units:

| K | `"fast"` (K·u) | `"sr"` (√K·u) | `"dd"` (K·u²) | `"kahan"` (K·u_f64) |
|---|---|---|---|---|
| 4 | 0.016 | 0.008 | 6×10⁻⁵ | 4.4×10⁻¹⁵ |
| 64 | 0.25 | 0.031 | 9.8×10⁻⁴ | 7.1×10⁻¹⁴ |
| 512 | 2.0 | 0.088 | 7.8×10⁻³ | 5.7×10⁻¹³ |

Values for BF16 (u = 2⁻⁸). `"sr"` and `"dd"` are only considered for BF16 and FP16 inputs — those are the dtypes where NKI stochastic rounding and Ozaki splitting are meaningful.

K=512, BF16, `target_forward_error=0.1`: fast bound is 2.0 (fails), sr bound is 0.088 (passes). Selection: `"sr"`.

## The approach

`select_precision_for_error(dtype, K, target)` walks the four modes in cost order and returns the first whose bound satisfies the target. A new `target_forward_error` kwarg on `einsum()` computes K from the subscript and operand shapes, calls the selector, then proceeds with the selected precision as if the caller had specified it explicitly. Passing both `precision=` and `target_forward_error=` raises `ValueError`.

The selector is a public API function (`trntensor.select_precision_for_error`), available independently of `einsum()`. Callers who want to inspect the selection — "what mode would a K=512 BF16 contraction get at target=0.05?" — can query it without running the full contraction.

One deliberate tradeoff: the K computation for multi-operand chains uses the product of all contracted indices in the full subscript. For `"ij,jk,kl->il"`, K = j_size × k_size. This is conservative for chains where the contractions happen in sequence — the actual worst-case error for a sequence of binary contractions is lower than the product-of-all-contracted-indices bound. The library accepts this conservatism because underestimating accuracy needs is the wrong failure mode. Choosing a mode that's one step more expensive than strictly necessary is an acceptable penalty; choosing one that's one step less expensive and silently delivering worse accuracy than the caller specified is not.

## Implementation

`select_precision_for_error` in `trntensor/plan.py`:

```python
def select_precision_for_error(dtype: torch.dtype, K: int, target_forward_error: float) -> str:
    u = _UNIT_ROUNDOFF.get(dtype, _UNIT_ROUNDOFF[torch.float32])
    u_f64 = _UNIT_ROUNDOFF[torch.float64]

    if K * u <= target_forward_error:
        return "fast"
    if dtype in _SR_DD_DTYPES and K**0.5 * u <= target_forward_error:
        return "sr"
    if dtype in _SR_DD_DTYPES and K * u * u <= target_forward_error:
        return "dd"
    if K * u_f64 <= target_forward_error:
        return "kahan"
    raise ValueError(
        f"Cannot achieve target_forward_error={target_forward_error:.2e} for "
        f"K={K}, dtype={dtype}. "
        f"Best available: 'kahan' (K·u_f64 ≈ {K * u_f64:.2e})."
    )
```

The dispatch block in `einsum()` that invokes it:

```python
if target_forward_error is not None:
    if precision != "fast":
        raise ValueError(
            "Cannot specify both 'precision' and 'target_forward_error'. "
            "Use one or the other."
        )
    _input_str, _output_str = _parse_subscripts(subscripts)
    _size_map = {}
    for _term, _op in zip(_input_str.split(","), operands, strict=False):
        for _c, _s in zip(_term, _op.shape, strict=False):
            _size_map[_c] = int(_s)
    _contracted = {c for c in _size_map if c not in _output_str}
    K = 1
    for c in _contracted:
        K *= _size_map[c]
    eff_dtype = _resolve_dtype(dtype) or (operands[0].dtype if operands else torch.float32)
    precision = select_precision_for_error(eff_dtype, K, target_forward_error)
```

After this block, `precision` holds the selected mode and the rest of `einsum()` proceeds normally — same routing, same dispatch paths, no special cases downstream.

## What didn't work

**SR's bound is statistical, not worst-case.** The Connolly–Higham–Mary result (SIAM J. Sci. Comput. 2021) gives `"sr"` a mean-zero error with standard deviation √K·u. `select_precision_for_error` uses √K·u as the SR threshold, but this is a probabilistic bound, not a Wilkinson worst-case. There is a small probability — formally e^{-c·K} for any constant c > 0 — of exceeding it. For workloads where a probabilistic bound is acceptable (DF-MP2 energy contributions, iterative solvers with redundancy), SR is the right choice at this target. For workloads where the bound must hold deterministically on every call, `"dd"` is the correct selection. The `select_precision_for_error` docstring makes this distinction explicit, and the table column is labeled "Expected (statistical)" for `"sr"`. The API doesn't hide the distinction, but it also doesn't force every caller to reason about it upfront — for most scientific computing workloads at intermediate K values, √K·u in expectation is a useful contract.

**The K overestimate for path strategy.** As noted above, K for a 3-operand chain is computed as the product of all contracted indices, which can be a conservative overestimate. A caller with `"ij,jk,kl->il"` at j=32, k=64 gets K=2048, which pushes the fast bound to 8.0 for BF16 — triggering SR or DD selection even if the sequential binary contractions would each have a much smaller effective K. Tighter K accounting per binary step is future work. The conservative choice is safe; the cost is occasionally using a more expensive mode than necessary.

**Toolchain note, unchanged.** The CPU simulator still does not support `round_mode="stochastic"` in `nisa.activation`. This gap was flagged in the [v0.11.0 post](https://trnsci.dev/blog/trntensor-v0110-stochastic-rounding-at-the-psumsbuf-boundary/) and remains open. `target_forward_error=` selections that route to `"sr"` run `_stochastic_round_cpu` in CI; the hardware path through `nisa.activation(..., round_mode="stochastic")` is unvalidated outside a real trn1 instance. The Neuron team request from v0.11.0 stands: simulator support for `round_mode` would close this for CI. This is worth repeating here because `target_forward_error=` is likely to route more code through SR than the explicit `precision="sr"` kwarg ever did — the automation makes the untested path easier to reach.

## Numbers

The error-bound table in "What the architecture suggests" is the quantitative content of this release. No hardware timing is available — v0.15.0 adds no new NKI kernel paths; it routes to existing ones.

The test that best demonstrates the end-to-end contract is `test_very_tight_selects_kahan`: K=16, BF16, `target_forward_error=1e-5`. The selector walks all four modes:

- fast: 16 × 2⁻⁸ ≈ 0.063 > 1e-5
- sr: 4 × 2⁻⁸ ≈ 0.016 > 1e-5
- dd: 16 × (2⁻⁸)² ≈ 2.4×10⁻⁴ > 1e-5
- kahan: 16 × 2⁻⁵³ ≈ 1.8×10⁻¹⁵ < 1e-5 → selected

The test then verifies the result is numerically identical to `precision="kahan"` output. The selector reached the right answer; the execution path is unchanged.

| New test | What it validates |
|---|---|
| `test_large_k_bf16_selects_sr` | K=512 BF16, target=0.1 → `"sr"` selected; correct shape and dtype |
| `test_small_k_selects_fast` | K=4 BF16, target=0.1 → `"fast"` selected |
| `test_tight_target_selects_dd` | K=64 BF16, target=0.003 → `"dd"` selected |
| `test_very_tight_selects_kahan` | K=16 BF16, target=1e-5 → `"kahan"`; result identical to explicit kahan |
| `test_ambiguous_raises` | `precision=` + `target_forward_error=` together raises `ValueError` |
| `test_impossible_target_raises` | target=1e-20 raises `ValueError` naming `"kahan"` as best available |

Total tests: 153, all passing (6 new in `TestTargetForwardError`).

## What's next

`target_forward_error=` in `einsum()` is the first concrete realization of the suite-level direction. The natural extensions:

- **`multi_einsum(*contractions, target_forward_error=ε)`**: each contraction selects its own mode based on its own K and dtype. The per-contraction loop in `multi_einsum` already calls `einsum()` for each step; threading `target_forward_error=` through is straightforward.
- **Adaptive error estimation**: rather than static Wilkinson bounds, measure actual residuals using Trainium's idle-engine concurrency (VectorE can run alongside Tensor Engine GEMMs). Escalate precision only when the measured residual exceeds the target, amortizing the cost of higher-precision modes across the cases that actually need them.
- **`solve(A, b, target_forward_error=ε)`** (trnsolver): the suite-level API direction. trntensor's contraction-level `target_forward_error=` is the building block; trnsolver's factorization and solve paths need the same contract at the linear-algebra level.
- **trnblas#22 and SDK 2.30+**: the two hardware gates that complete the NKI side of the precision arc. When trnblas#22 lands, `"dd"` on Trainium becomes a single fused NKI program instead of a `NotImplementedError`. When SDK 2.30+ arrives, reduce-parallel sharding completes. `target_forward_error=` selections that route to `"dd"` will automatically benefit from the trnblas#22 path with no API changes.

Live roadmap: [trnsci.dev/roadmap/](https://trnsci.dev/roadmap/). Suite tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## Takeaway

The precision scaffolding across v0.11.0–v0.14.0 — stochastic rounding, mixed sharding, precision threading, Ozaki-2 CPU mocks — was four releases of laying infrastructure. v0.15.0 is one release that puts a door in front of it. Callers who previously had to know that K=512 with BF16 means fast rounding's bound reaches 200% relative error now write `target_forward_error=0.1` and get stochastic rounding automatically. The hardware facts are still there — the table is still in the docstring — but they no longer have to live in the caller's head. For the suite-level arc toward `solve(A, b, target_forward_error=ε)`, the contraction layer needed to speak in error bounds before the solver layer could. That part is done.
