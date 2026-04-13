---
date: YYYY-MM-DD
categories: [Deep dive, trnXXX]   # trnfft | trnblas | trnrand | trnsolver | trnsparse | trntensor
comments: true
---

# trnXXX: short, specific title naming the library

<!--
Editorial brief: https://trnsci.dev/blog/AUTHOR_BRIEF/
Voice: authorless, library-as-subject. No "I", no byline.
Stance: architecture-first (what the hardware affords), transparency-always
        (what didn't work, honest numbers).
Required sections below — keep them in this order.
-->

Lead paragraph — 2–3 sentences. One on what shipped, one on why it's interesting architecturally, one on who should care.

<!-- more -->

## The problem

What workload is this solving, and what's awkward about solving it on Trainium with a naive port of the CUDA approach? Cite the cuX analog by name but don't privilege its design.

## What the architecture suggests

**Required section.** What does Trainium's hardware actually afford for this problem? Which engines are load-bearing (Tensor / Vector / Scalar / GpSimd)? What tile shape, SBUF layout, PSUM accumulation, DMA pattern, or NEFF cache behavior is the natural fit?

The native-to-Trainium design, independent of what CUDA does. This section is the heart of the post.

## The approach

The design chosen. How it exploits what the previous section identified. At least one deliberate tradeoff. If the design ended up looking like the CUDA approach, say so and say why — but don't default to that framing.

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

Honest numbers, including ones that look bad. Numbers confirm or contextualize the architectural choice above — they don't justify it on their own.

## What's next

Link the Phase 2/3/4/5 tracker issues for this library. Readers should know where the project goes from here.

- [Phase 2 — ...](https://github.com/trnsci/trnXXX/issues/NN)
- [Phase 3 — ...](https://github.com/trnsci/trnXXX/issues/NN)

## Takeaway

3–5 sentences. One architectural idea the reader should leave with.
