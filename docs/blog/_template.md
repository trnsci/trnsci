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

A small Mermaid diagram of the dataflow (HBM → SBUF → engines → PSUM → HBM, or similar) is often the right move here.

## The approach

The design chosen. How it exploits what the previous section identified. At least one deliberate tradeoff. If the design ended up looking like the CUDA approach, say so and say why — but don't default to that framing.

## Implementation

Real code from the repo, not pseudocode. At least one snippet showing the key NKI kernel or dispatch pattern.

```python
# excerpt from trnXXX/nki/dispatch.py
```

## What didn't work

Blind alleys. Reverted approaches. Numbers that disappointed. This section is required.

Also belong here (when they apply): NKI compiler bugs or surprising behaviors (with SDK version), missing primitives that forced workarounds, awkward APIs, documentation gaps you had to discover empirically, unhelpful error messages (quote them so future searchers find them), and concrete suggestions for the AWS Neuron team. Professional and specific, not bitter — the goal is useful feedback for the whole ecosystem.

Candid fit-assessment also belongs here: is this workload actually well-matched to Trainium, or is the library working around a shape mismatch? Where is the silicon over-indexed for training workloads at the expense of this one? Where is it under-indexed? Name both, with specifics.

## Numbers

| Workload | CPU (torch) | NKI (trn1) | Notes |
|---|---|---|---|
| | | | |

Honest numbers, including ones that look bad. Numbers confirm or contextualize the architectural choice above — they don't justify it on their own.

A table here is almost always the right shape. Add columns for any cross-platform comparisons that exist (CPU, vintage-matched GPU, hardware-validated NKI). Bold the winning column per row only when meaningful.

## What's next

Link the Phase 2/3/4/5 tracker issues for this library. Readers should know where the project goes from here.

- [Phase 2 — ...](https://github.com/trnsci/trnXXX/issues/NN)
- [Phase 3 — ...](https://github.com/trnsci/trnXXX/issues/NN)

## Takeaway

3–5 sentences. One architectural idea the reader should leave with.
