---
hide:
  - navigation
  - toc
---

# Author brief — trnsci technical deep-dives

This page is the editorial brief for technical retrospectives on the [trnsci blog](https://trnsci.dev/blog/). It's designed to be read once and referenced against the template.

## What you're writing

A technical retrospective on a piece of work that just shipped in one of the trnsci libraries — trnfft, trnblas, trnrand, trnsolver, trnsparse, or trntensor.

**Audience:** CUDA programmers evaluating Trainium. Neuron SDK users looking for scientific computing code. Maintainers of the sibling libraries in the trnsci suite. Assume readers know Trainium exists but not what NKI is — link out for NKI, don't redefine it each post.

## Voice — authorless by default, library as subject

**Technical deep-dives are unsigned.** No byline. No "I". No named maintainer.

The library itself is the subject:

> trnfft's butterfly kernel now validates on trn1.2xlarge across 70/70 benchmark cases. Getting there required…

Not:

> I built a butterfly kernel…
> We built a butterfly kernel…

Why: the work is being done collaboratively, the authorship distinctions are arbitrary, and what matters is the technical content. A byline adds nothing and creates false personhood.

Exceptions where a byline is appropriate:

- **Editorial suite digests** — bylined to Scott Friedman as the suite director, because curatorial judgement is exercised.
- **Thinking pieces about accelerator architecture, open-source governance, ecosystem positioning** — bylined to whoever writes them, because opinion is being expressed.

When in doubt on a technical post: no byline.

## Editorial stance — architecture-first, transparency-always

**Two principles stacked.**

### Architecture first, not straight porting

The trnsci suite isn't interesting because it ports cuFFT / cuBLAS / cuRAND / cuSOLVER / cuSPARSE / cuTENSOR techniques to Trainium one-to-one. If all we have to say is "we replicated NVIDIA's approach on different silicon", that's not a post worth writing.

What's worth writing: **what does Trainium's architecture — the four programmable engines, the fixed 128-partition × 512-moving tile, the explicit SBUF/PSUM memory hierarchy, the DMA engine as a first-class resource, the NEFF cache semantics, whole-program NKI compilation — make possible that a GPU wouldn't naturally suggest?**

Concrete examples of the framing we want:

- "Jacobi rotations land on the Tensor Engine because each Givens rotation is a rank-2 matmul that matches the 128-partition tile exactly; that tile-friendliness matters more than the O(n³) FLOP count disadvantage versus Householder." Not: "we chose Jacobi because it's different."
- "BSR at 128×128 isn't a port of cuSPARSE's BSR — it's the native Trainium sparse format because each block is a Tensor Engine tile with zero gather overhead." Not: "we added BSR because cuSPARSE has it."
- "The MP2 energy kernel fuses the per-(i,j) contraction with the orbital-energy denominator division in one NKI pass because intermediates can stay SBUF-resident across a DAG-scheduled kernel — that's a pattern that doesn't translate back to per-op PyTorch cleanly." Not: "we wrote a fused kernel for speed."

The question to hold in your head while drafting: **what about this kernel reveals something about the hardware that a CUDA programmer wouldn't have reached for?** If the answer is "nothing", the post probably isn't ready.

### Transparency over polish

**Post what actually happened, including the parts that didn't work.** Vendor-marketing voice ("trnblas delivers unprecedented DF-MP2 throughput") is not useful and actively harms credibility.

Useful content we want in every technical deep-dive:

- **What was tried that didn't work.** Reverted kernels, approaches that looked plausible on paper and failed on hardware, NKI compiler behaviors that surprised. Name the blind alleys.
- **Honest benchmark numbers, including the ones that disappointed.** If the NKI path is slower than PyTorch CPU at small shapes (it often is), say so and explain why. Readers will find out anyway.
- **Tradeoffs made deliberately.** "Jacobi was chosen over Householder because X, and the cost is Y." The "cost is Y" half is the credibility half.
- **Open questions.** Follow-ups that are known, behaviors that have been observed but not fully explained.

**Benchmarks are validation, not the point.** A post full of numbers and no architectural insight is a fail state. A post with modest numbers but a clear articulation of what the hardware enabled that wasn't natural on NVIDIA is a win.

## Required structure

Use these section headings in this order. The `posts/_template.md` file has them pre-filled.

1. **Lead** (2–3 sentences) — one sentence on what shipped, one on why it's interesting *architecturally*, one on who should care.
2. **The problem** — what workload is this solving, and what's awkward about solving it on Trainium with a naive port of the CUDA approach. Cite the cuX analog by name but don't privilege its design.
3. **What the architecture suggests** — **required section.** What does Trainium's hardware (engines, tile, memory hierarchy, NEFF, DMA) actually afford for this problem? What's the native-to-Trainium design, independent of CUDA? This section is the heart of the post.
4. **The approach** — the design chosen. How it exploits what Section 3 identified. At least one tradeoff made deliberately. If the design ended up looking like the CUDA approach, say so and say why — but don't default to that framing.
5. **Implementation** — at least one code sample showing the key NKI kernel or dispatch pattern. Real code from the repo, not pseudocode.
6. **What didn't work** — blind alleys, reverted approaches, NKI compiler surprises, numbers that disappointed. Named in full. This section is required; "nothing" is almost never the right answer.
7. **Numbers** — if hardware benchmarks exist, put a table. CPU baseline + Trainium, same inputs, honest numbers including the ones that look bad. Numbers confirm or contextualize the architectural choice in Section 3 — they don't justify it on their own.
8. **What's next** — explicitly link the Phase 2/3/4/5 tracker issues for the library. Readers should know where the project goes from here.
9. **Takeaway** (3–5 sentences) — what one architectural idea should the reader leave with.

## Style rules

- Name the library in the title (e.g., "trnfft: FFT on hardware without a complex dtype"). Not a generic framing.
- Use absolute numbers with units. Not "fast", not "3x speedup". Specific microseconds, TFLOPS, basis-function counts, nnz densities.
- Link back to the suite: [trnsci.dev](https://trnsci.dev), the [roadmap](https://trnsci.dev/roadmap/), the [suite-wide tracker](https://github.com/trnsci/trnsci/issues/1). Act like part of a suite.
- No emoji unless the post is explicitly about emoji. (It isn't.)
- Apache 2.0 code samples only. If borrowed, cite.
- Length: 1,200–2,500 words. Longer splits into two posts.

## Frontmatter

Copy verbatim into the top of the post; change date / slug / categories:

```yaml
---
date: 2026-04-13
categories: [Deep dive, trnfft]   # or trnblas, trnrand, trnsolver, trnsparse, trntensor
comments: true
---
```

No `authors:` line for technical deep-dives.

Slug and filename: `docs/blog/posts/<YYYY-MM-DD>-<short-slug>.md`. Keep slugs short — `2026-04-13-fft-without-complex-dtype.md`, not the full title.

## Submitting

1. Open a PR against [trnsci/trnsci](https://github.com/trnsci/trnsci) with the post at `docs/blog/posts/<YYYY-MM-DD>-<slug>.md`.
2. Scott (as suite director) reviews for editorial consistency — structural edits, fact-checks against the repo, smoothing. Technical content isn't rewritten.
3. Expect one or two review rounds.
4. On merge, the post appears at `trnsci.dev/blog/` within minutes (on push) or by 06:00 UTC the next day (daily cron).

## Optional pre-draft pitch

For sanity-checking scope and angle before investing the writing time: open a GitHub issue on [trnsci/trnsci](https://github.com/trnsci/trnsci/issues) with label `blog-pitch`. Title + 3-sentence abstract. Lightweight.

## Questions

Open a discussion on [trnsci/trnsci](https://github.com/trnsci/trnsci/discussions) under the "Editorial" category.
