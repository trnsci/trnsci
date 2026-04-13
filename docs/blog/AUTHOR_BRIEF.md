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

## Editorial stance — transparency over polish

**Post what actually happened, including the parts that didn't work.** Vendor-marketing voice ("trnblas delivers unprecedented DF-MP2 throughput") is not useful and actively harms credibility.

Useful content we want in every technical deep-dive:

- **What was tried that didn't work.** Reverted kernels, approaches that looked plausible on paper and failed on hardware, NKI compiler behaviors that surprised. Name the blind alleys.
- **Honest benchmark numbers, including the ones that disappointed.** If the NKI path is slower than PyTorch CPU at small shapes (it often is), say so and explain why. Readers will find out anyway.
- **Tradeoffs made deliberately.** "Jacobi was chosen over Householder because X, and the cost is Y." The "cost is Y" half is the credibility half.
- **Open questions.** Follow-ups that are known, behaviors that have been observed but not fully explained.

A post that says "here's what this cost and here's what we don't know" is more useful than one that only claims wins.

## Required structure

Use these section headings in this order. The `posts/_template.md` file has them pre-filled.

1. **Lead** (2–3 sentences) — one sentence on what shipped, one on why it's interesting, one on who should care.
2. **The problem** — the technical constraint or gap. What Trainium lacks or does differently from NVIDIA. Cite the cuX analog by name.
3. **The approach** — the design. Why this design over alternatives. At least one tradeoff made deliberately.
4. **Implementation** — at least one code sample showing the key NKI kernel or dispatch pattern. Real code from the repo, not pseudocode.
5. **What didn't work** — blind alleys, reverted approaches, NKI compiler surprises, numbers that disappointed. Named in full. This section is required; "nothing" is almost never the right answer.
6. **Numbers** — if hardware benchmarks exist, put a table. CPU baseline + Trainium, same inputs, honest numbers including the ones that look bad.
7. **What's next** — explicitly link the Phase 2/3/4/5 tracker issues for the library. Readers should know where the project goes from here.
8. **Takeaway** (3–5 sentences) — what one idea should the reader leave with.

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
