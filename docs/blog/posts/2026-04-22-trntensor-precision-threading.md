---
date: 2026-04-22
categories: [Deep dive, trntensor]
comments: true
---

# trntensor v0.13.0: precision all the way down

[v0.11.0](https://trnsci.dev/blog/trntensor-v0110-stochastic-rounding-at-the-psumsbuf-boundary/) introduced `precision="sr"` and wired it to `_execute_matmul`. Two dispatch paths were never connected: `_execute_bmm` silently ran at fast rounding regardless of what the caller asked for, and `multi_einsum` had no `precision=` parameter at all. v0.13.0 closes both gaps. The fix was small; the reason it matters is architectural.

<!-- more -->

## The problem

A precision parameter that propagates through some dispatch paths but not others is not a parameter — it is advice that the library may or may not take. `precision="sr"` was introduced in [v0.11.0](https://trnsci.dev/blog/trntensor-v0110-stochastic-rounding-at-the-psumsbuf-boundary/) because Trainium's PSUM buffer makes stochastic rounding a one-instruction hardware primitive at the end of each BF16 tile accumulation. The architectural argument — that PSUM→SBUF is the single controlled rounding point in a K-reduction — applied to every contraction that routes through an NKI matmul kernel. The implementation honored that argument for `einsum()` when the planner chose the `matmul` strategy. It did not honor it for the `bmm` strategy or for `multi_einsum` calls.

The bmm gap was invisible at the call site. `plan.precision` was set correctly on the `ContractionPlan` object; `_execute_bmm` never read it. The call to `nki_batched_matmul` was `nki_batched_matmul(A, B)` — `use_sr` defaulting to `False`, quietly ignoring whatever the caller had requested. No error. No warning. The kernel had supported `use_sr` since v0.11.0.

The `multi_einsum` gap was structural. The function had no `precision=` parameter. Callers who relied on batched multi-contraction paths — DF-MP2 (i,j) loops, AO→MO batches — had no way to opt in to SR even if they wanted to.

## What the architecture suggests

The PSUM→SBUF rounding argument from [v0.11.0](https://trnsci.dev/blog/trntensor-v0110-stochastic-rounding-at-the-psumsbuf-boundary/) is not specific to 2D matmul. It applies anywhere the Tensor Engine accumulates BF16 multiply-adds into FP32 PSUM and then downcasts to SBUF. That downcast happens in both `matmul_kernel` and `batched_matmul_kernel`. The tile boundary is the same. The `nisa.activation(..., round_mode="stochastic")` call is the same. The only meaningful distinction between the matmul and batched-matmul paths is that the batched kernel loops over a leading batch dimension before each tile accumulation — the PSUM structure per tile is identical.

`multi_einsum` adds a layer above this. Its internal optimization — stacking homogeneous 2D contractions into a single `nki_batched_matmul` call to amortize the ~0.67 ms XLA dispatch overhead — is an internal routing decision invisible to the caller. A caller asking for `precision="sr"` expects SR to apply whether their contraction goes through the batched kernel or the per-contraction fallback. If `precision=` doesn't propagate through `_try_batched_multi_einsum`, callers with large DF-MP2 (i,j) batches would silently receive round-nearest output whenever the batching optimization fires, even if they explicitly requested SR.

This matters for `target_forward_error`, the prospective API that would let callers state an accuracy bound and have the library select precision automatically. That API cannot make a selection that actually holds unless every dispatch path honors it.

## The approach

| Dispatch path | v0.11.0 | v0.13.0 |
|---|---|---|
| `einsum` → `_execute_matmul` | SR | SR |
| `einsum` → `_execute_bmm` | fast only (gap) | SR |
| `einsum` → `_execute_path` | SR (recursive) | SR |
| `multi_einsum` → batched kernel | no kwarg (gap) | SR |
| `multi_einsum` → per-contraction loop | no kwarg (gap) | SR |

The `"kahan"` and `"dd"` modes are handled separately. Both live in `_execute_contraction`'s fp64-promotion branch, upstream of NKI dispatch. `_try_batched_multi_einsum` returns `None` for those modes — the existing signal meaning "don't optimize, fall through" — and the per-contraction loop handles them via `einsum()`, which routes to `_execute_contraction`. This is not a gap; `"kahan"` promotes to FP64 before the contraction and has no PSUM rounding step to intercept.

The tradeoff made deliberately: `"sr"` is a no-op for strategies that don't reach an NKI matmul kernel. The `"torch"` and `"path"` strategies have no PSUM buffer to apply SR at. Injecting random quantization noise at the output of a `torch.einsum` call would not be SR — it would be noise. The `ContractionPlan` logs this when `precision="sr"` is requested but the planner selects a non-matmul strategy.

## Implementation

The BMM fix, in `_execute_bmm`:

```python
def _execute_bmm(subscripts: str, operands: tuple, plan: ContractionPlan) -> torch.Tensor:
    """Execute as batched matmul."""
    from .nki.dispatch import nki_batched_matmul

    A, B = operands
    return nki_batched_matmul(A, B, use_sr=(plan.precision == "sr"))
```

The v0.11.0 version was `nki_batched_matmul(A, B)`. The fix is one expression.

The `multi_einsum` change adds `precision: str = "fast"` to the signature and threads it through both paths:

```python
def multi_einsum(*contractions: tuple, precision: str = "fast") -> list[torch.Tensor]:
    ...
    batched = _try_batched_multi_einsum(subst, precision=precision)
    if batched is not None:
        return batched

    # Fallback: per-contraction loop
    for c in subst:
        result = einsum(c[0], *c[1:], precision=precision)
        ...
```

And in `_try_batched_multi_einsum`, the precision guard and the `use_sr` pass-through:

```python
def _try_batched_multi_einsum(
    contractions: list, precision: str = "fast"
) -> list[torch.Tensor] | None:
    if precision not in ("fast", "sr"):
        return None           # kahan/dd fall to per-contraction loop
    ...
    out_stack = nki_batched_matmul(A_stack, B_stack, use_sr=(precision == "sr"))
    ...
```

The `return None` for non-BF16 modes is three characters. It delegates `"kahan"` and `"dd"` to the path that already handles them correctly instead of attempting to pass unsupported modes into a kernel that only knows `use_sr`.

## What didn't work

**Silent failure was the whole problem.** The BMM gap produced no error at any level: not from the planner (which correctly stored `precision="sr"` on `ContractionPlan`), not from `_execute_bmm` (which received the plan but ignored its precision field), not from `nki_batched_matmul` (which received `use_sr=False` and rounded normally). A test at v0.11.0 release would have caught this immediately. `test_sr_noop_for_bmm` existed but tested the wrong thing — it verified that SR accepted without error on a bmm subscript, not that SR actually applied. That test passing was evidence of nothing. The test gap was `test_sr_bmm_shape_and_dtype` and `test_sr_bmm_close_to_fast`, both of which could have been written when `precision="sr"` first shipped. The code fix was one expression. The test-surface fix was three new tests.

**`"kahan"` in `_try_batched_multi_einsum`**: the first implementation passed `precision` through to `nki_batched_matmul` for all modes, which meant passing `"kahan"` to a kernel that only has a `use_sr` bool and no FP64 accumulation path. The kernel silently ignores unrecognized kwargs on CPU; on hardware it would error or produce wrong results. The fix — `return None` for any mode that isn't `"fast"` or `"sr"` — is the right structural choice. `"kahan"` FP64 promotion belongs in `_execute_contraction`, not in the batched kernel. Routing it there via the fallback path keeps the two concerns separated.

**Toolchain note.** The CPU simulator still does not support `round_mode="stochastic"` in `nisa.activation`. This was flagged in [v0.11.0](https://trnsci.dev/blog/trntensor-v0110-stochastic-rounding-at-the-psumsbuf-boundary/); it remains open after v0.13.0. All three new SR tests run under `_stochastic_round_cpu`, the named CPU stand-in for the PSUM→SBUF hardware primitive. The hardware path through `nisa.activation(..., round_mode="stochastic")` in `batched_matmul_kernel` is syntactically correct against the SDK 2.29 headers but has not been validated on hardware. The Neuron team request from v0.11.0 stands: `round_mode` support in the CPU simulator would close this gap for CI.

## Numbers

v0.13.0 adds no new hardware execution paths. The table is about test coverage and what it validates:

| New test | What it validates |
|---|---|
| `test_sr_bmm_shape_and_dtype` | `_execute_bmm` honors `use_sr`, output dtype preserved |
| `test_sr_bmm_close_to_fast` | SR bmm within atol=0.3 of fast (K=64 BF16, √64·u ≈ 0.032) |
| `test_sr_multi_einsum_threads_precision` | `multi_einsum(precision="sr")` threads to each contraction |

Total tests: 139 → 142. Tolerances: bmm atol=0.3 for K=64 BF16 (tighter than the original matmul's atol=0.2 at K=128 because K is smaller, giving a tighter √K·u bound); multi_einsum atol=0.2 for K=64.

## What's next

`precision=` is now wired end-to-end. Two gates remain before the scaffolding becomes fully operational:

- **`precision="dd"` (trnblas#22)**: when trnblas Phase 2 double-double GEMM kernels land, `_execute_contraction`'s `NotImplementedError` for `"dd"` becomes a call to trnblas. The routing through `_try_batched_multi_einsum`'s `return None` path is already in place — `"dd"` will fall to the per-contraction loop and reach `_execute_contraction` without any further changes.
- **`nki.collectives.allreduce` (SDK 2.30+)**: the one-line swap that replaces `_mock_allreduce` in `_execute_sharded`, completing reduce-parallel and mixed sharding on hardware. Covered in [v0.12.0](https://trnsci.dev/blog/trntensor-v0120-mixed-sharding/).

Live roadmap: [trnsci.dev/roadmap/](https://trnsci.dev/roadmap/). Suite tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## Takeaway

A precision contract that doesn't propagate through every dispatch path isn't a contract — it's a hint. The BMM kernel had supported `use_sr` since v0.11.0; `_execute_bmm` just never passed it. `multi_einsum` had no `precision=` kwarg because no one had written one. Neither gap surfaced an error; both produced silently wrong output whenever the caller's intent was stochastic rounding. v0.13.0 is three tests, one expression, and one function signature — the kind of release whose diff doesn't look like much until you trace what the dispatch layer was silently discarding.
