---
date: YYYY-MM-DD
categories: [Deep dive, trnXXX]   # trnfft | trnblas | trnrand | trnsolver | trnsparse | trntensor
comments: true
---

# trnXXX: short, specific title naming the library

<!--
Editorial brief: https://trnsci.dev/blog/AUTHOR_BRIEF/
Voice: authorless, library-as-subject. No "I", no byline. See brief.
Required sections below — keep them in this order.
-->

Lead paragraph — 2–3 sentences. One on what shipped, one on why it's interesting, one on who should care.

<!-- more -->

## The problem

What Trainium lacks or does differently from NVIDIA. Cite the cuX analog by name.

## The approach

The design. Why this design over alternatives. At least one deliberate tradeoff.

## Implementation

Real code from the repo, not pseudocode. At least one snippet showing the key NKI kernel or dispatch pattern.

```python
# excerpt from trnXXX/nki/dispatch.py
```

## What didn't work

Blind alleys. Reverted approaches. NKI compiler surprises. Numbers that disappointed. This section is required.

## Numbers

| Workload | CPU (torch) | NKI (trn1) | Notes |
|---|---|---|---|
| | | | |

Honest numbers, including ones that look bad. Explain the context when a number is surprising.

## What's next

Link the Phase 2/3/4/5 tracker issues for this library. Readers should know where the project goes from here.

- [Phase 2 — ...](https://github.com/trnsci/trnXXX/issues/NN)
- [Phase 3 — ...](https://github.com/trnsci/trnXXX/issues/NN)

## Takeaway

3–5 sentences. One idea the reader should leave with.
