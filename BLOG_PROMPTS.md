# Blog prompts for sub-project agents

Self-contained prompts to paste into each sub-project's Claude agent when you want them to draft a technical deep-dive for the [trnsci blog](https://trnsci.dev/blog/).

Each prompt is designed to be pasted verbatim — it carries all the context the agent needs (editorial brief, structure, voice, suite positioning). The agent drafts a post in the correct location (`docs/blog/posts/<date>-<slug>.md`), opens a PR against `trnsci/trnsci`, and you/Scott reviews before merge.

Editorial brief the prompts reference: [`docs/blog/AUTHOR_BRIEF.md`](docs/blog/AUTHOR_BRIEF.md).
Template file: [`docs/blog/_template.md`](docs/blog/_template.md).

**A standing editorial note for every post:** the brief's "What didn't work" section also covers two axes of candor beyond internal project decisions.

1. **Toolchain feedback** — NKI compiler bugs, missing primitives, awkward APIs, doc gaps, unhelpful error messages, and concrete suggestions for the AWS Neuron team.
2. **Fit assessment** — whether this workload is actually well-matched to Trainium's current architecture, where the silicon looks over-indexed for training workloads at the expense of the target workload, and where it looks under-indexed for patterns the workload needs.

Professional and specific, not bitter. Readers evaluating Trainium benefit more from that candor than from a post that pretends everything was smooth. See the brief for the detailed stance on both.

---

## One-time setup: pointer in each sub-project's `CLAUDE.md`

So sub-project agents can find their own prompt without the user couriering it, add this block to `CLAUDE.md` in each of the six sub-project repos (`trnfft`, `trnblas`, `trnrand`, `trnsolver`, `trnsparse`, `trntensor`). Paste once per repo — content is identical across the suite.

```markdown
## Blog posts

When asked to draft a blog post for this library for the [trnsci blog](https://trnsci.dev/blog/):

1. Read the editorial brief at [`docs/blog/AUTHOR_BRIEF.md`](https://github.com/trnsci/trnsci/blob/main/docs/blog/AUTHOR_BRIEF.md) in the umbrella repo (trnsci/trnsci). It defines voice (authorless, library-as-subject), stance (architecture-first, transparency-always), and the nine required section headings.

2. Find the prompt block for this library in [`BLOG_PROMPTS.md`](https://github.com/trnsci/trnsci/blob/main/BLOG_PROMPTS.md) at the umbrella repo root. It carries library-specific context and suggested architectural angles.

3. Draft the post following the brief. Open a PR against `trnsci/trnsci` at `docs/blog/posts/<YYYY-MM-DD>-<slug>.md`. Scott (suite director) reviews before merge.

The umbrella repo — not this one — owns the blog. Per-library retrospective posts are unsigned; library is the subject, no byline. See the brief for the full set of rules.
```

After this is in place, prompting a sub-project agent with "draft your Phase 1 blog post" (or similar) is enough — the agent fetches the brief and its own prompt block without further instruction.

---

## Ready now

### trnfft — Phase 1 retrospective

```
You maintain trnfft, a library in the trnsci scientific computing suite for
AWS Trainium (trnsci.dev). trnfft is the cuFFT-equivalent — FFT, complex
tensors, STFT, complex NN layers. In v0.8.0 you shipped hardware-validated
NKI kernels for butterfly FFT, complex GEMM, complex linear, complex
multiply, and a Kahan-compensated butterfly variant. All 70 benchmark cases
pass on trn1.2xlarge.

Please draft a technical retrospective blog post on the Phase 1 work.

Frame the post around what Trainium's architecture affords, not around
porting cuFFT. A useful spine: start from "Trainium has no complex dtype
and a fixed 128-partition Tensor Engine tile" and ask what that makes
possible (or forces). Where did the split real/imag representation,
four-real-GEMM complex multiply, and stationary-tile reuse come from? Not
from cuFFT — from the engine layout. Butterfly stages mapping to
Tensor Engine (multiply) + Vector Engine (add) in parallel isn't a
textbook radix-2 Cooley-Tukey; it's what the per-engine scheduling makes
cheap. The Kahan variant exists because FP32 PSUM has a ceiling that
long Bluestein chains bump into — and because Vector Engine can do the
compensation cheaply in the same kernel.

Candidate angle for the title: "FFT on hardware that doesn't want to be
an FFT engine" — leaning into the "the architecture suggests something
different" framing over "here's how we built an FFT."

Before writing: read https://trnsci.dev/blog/AUTHOR_BRIEF/. Key rules:

  - Authorless. Library as subject ("trnfft's butterfly kernel", not "I"
    or "we"). No byline.
  - Nine required sections in order: Lead, The problem, What the
    architecture suggests (required, the heart of the post), The
    approach, Implementation, What didn't work, Numbers, What's next,
    Takeaway.
  - "What the architecture suggests" is where the post earns its keep.
    What does the four-engine layout, PSUM accumulation, and 128-partition
    tile make natural for FFT that a CUDA warp-per-butterfly approach
    wouldn't? Write that paragraph as if cuFFT didn't exist.
  - "What didn't work" also required — NKI compiler surprises, reverted
    approaches, FP32 precision fails that motivated the Kahan variant.
  - Absolute numbers with units. Benchmarks are context, not the point.
  - 1200–2500 words.

Open the PR as docs/blog/posts/2026-04-DD-trnfft-fft-without-complex-dtype.md
(pick the date) against trnsci/trnsci. Frontmatter:

  ---
  date: 2026-04-DD
  categories: [Deep dive, trnfft]
  comments: true
  ---

Scott (suite director) will review for editorial consistency before merge.
```

### trnblas — Phase 1 retrospective

```
You maintain trnblas, a library in the trnsci scientific computing suite
for AWS Trainium (trnsci.dev). trnblas is the cuBLAS-equivalent — Level
1-3 BLAS. In v0.4.0 you shipped hardware-validated NKI kernels for GEMM
(stationary-tile reuse), SYRK, and a fused MP2 energy reduction kernel;
end-to-end density-fitted MP2 is validated against PySCF at nanohartree
tolerance on H2O, CH4, and NH3 at cc-pVDZ.

Please draft a technical retrospective blog post on the Phase 1 work.

Frame the post around what Trainium's architecture affords for GEMM-heavy
chemistry, not around porting cuBLAS. The interesting story isn't "we
made GEMM work on NKI" — it's what the Tensor Engine's systolic layout
plus SBUF-resident stationary operand plus PSUM accumulation lets you do
that per-op cuBLAS calls can't express cleanly.

Load-bearing architectural points worth exploring:

  - Stationary-tile reuse (A SBUF-resident, B streaming) is natural on
    Trainium and saves dramatic HBM bandwidth versus the per-call GEMM
    mental model CUDA programmers carry.
  - The fused MP2 energy kernel (#15) is the real architectural story:
    intermediates stay SBUF-resident across the contraction +
    denominator division + reduction, in one NKI pass. That's not a
    cuBLAS shape. CuBLAS would need three calls and three HBM round
    trips. On Trainium, NEFF-cached DAG compilation keeps the whole
    thing in one kernel. This is the kind of thing the architecture
    makes natural that a CUDA port wouldn't reach for.
  - Why batched_gemm is a hybrid (host-side loop around a real NKI GEMM)
    rather than a true 3D batched NKI kernel in Phase 1 — and what the
    Tensor Engine makes that choice about. Defer the true batched kernel
    to Phase 3 where the perf case justifies it.
  - FP32-accumulate precision story against FP64 PySCF — why
    nanohartree tolerance holds despite FP32, and what that says about
    iterative-refinement as a Phase 2 direction.

Candidate title direction: something that leads with "what the Tensor
Engine wanted us to do" rather than "how we ported cuBLAS."

Before writing: read https://trnsci.dev/blog/AUTHOR_BRIEF/. Key rules:

  - Authorless. Library as subject. No byline.
  - Nine required sections in order. "What the architecture suggests" is
    where the post earns its keep — write it as if cuBLAS didn't exist.
  - "What didn't work" is required. Candidates: the examples/df_mp2
    revert in #15, NKI partition-dim constraints that tripped you up,
    numbers that disappointed at small shapes, the hybrid batched_gemm
    tradeoff.
  - Include a benchmark table — real numbers from v0.4.0 CHANGELOG /
    benchmarks. Present them as confirming the architectural choice, not
    as the reason for it.
  - Absolute numbers with units. 1200–2500 words.

Open the PR as docs/blog/posts/2026-04-DD-trnblas-df-mp2-nanohartree.md
against trnsci/trnsci. Frontmatter:

  ---
  date: 2026-04-DD
  categories: [Deep dive, trnblas]
  comments: true
  ---

Scott (suite director) will review for editorial consistency before merge.
```

### trnsparse — Phase 1 retrospective (honest-about-perf angle)

```
You maintain trnsparse, a library in the trnsci scientific computing
suite for AWS Trainium (trnsci.dev). trnsparse is the cuSPARSE-equivalent —
CSR/COO, SpMV, SpMM, Schwarz screening. In v0.2.0 you shipped the first
hardware-validated NKI SpMM kernel in the suite, using a densify-then-GEMM
approach. You also published benchmarks honestly showing the NKI path is
slower than scipy and torch.sparse at current shapes — because v0.2.0 is
about correctness, not speed; nnz-bucketing / gather-matmul-scatter lands
in v0.3.0.

Please draft a technical retrospective blog post.

The architectural story here is especially rich because Trainium's sparse
primitive isn't CSR — it's the 128×128 Tensor Engine tile. The interesting
angle is: what does Trainium's architecture think sparse matrices should
look like?

Load-bearing architectural points worth exploring:

  - BSR at 128×128 isn't a port of cuSPARSE's BSR — it IS the native
    Trainium sparse format because each block matches a Tensor Engine
    tile with zero gather overhead. CSR/COO become input formats that
    get materialized into BSR-style tiles at dispatch time. This reframes
    the whole library.
  - The densify-then-GEMM approach in v0.2 is an admission of this: the
    Tensor Engine doesn't want sparsity at the element level, it wants
    dense tiles. v0.2 materializes the dense tile naively; v0.3's row-
    bucketing + gather-matmul-scatter is the native Trainium version.
  - The DMA engine as a first-class gather/scatter resource is what
    makes the full gather-matmul-scatter pattern pay off — and it's not
    a thing CUDA programmers think about with the same granularity.
  - Why the "slower than scipy" numbers in v0.2 aren't a failure: the
    full Neuron toolchain (compile, NEFF cache, XLA, PyTorch autograd
    bridge) is exercised end-to-end. That's what Phase 1 validates.
    Perf is what v0.3 validates, against the right architectural pattern.

Candidate title direction: "what Trainium thinks a sparse matrix is" or
similar — foregrounding that the library is on a different axis than
cuSPARSE, not a worse version of it.

Before writing: read https://trnsci.dev/blog/AUTHOR_BRIEF/. Key rules:

  - Authorless. Library as subject. No byline.
  - Nine required sections in order. "What the architecture suggests"
    is where the post earns its keep — frame around the 128×128 tile
    and the DMA engine, not around cuSPARSE CSR.
  - Include the full benchmark table from docs/benchmarks.md. Don't
    editorialize the numbers. Present them as confirming that v0.2 is
    Phase 1 (correctness) and perf belongs to the architecturally-right
    approach in v0.3.
  - "What didn't work" — any kernel-launch overhead investigation,
    initial nnz-aware attempts that didn't pan out for Phase 1, the
    decision to materialize rather than gather in v0.2.
  - 1200–2500 words.

Open the PR as docs/blog/posts/2026-04-DD-trnsparse-phase1-correctness-
over-speed.md against trnsci/trnsci. Frontmatter:

  ---
  date: 2026-04-DD
  categories: [Deep dive, trnsparse]
  comments: true
  ---

Scott (suite director) will review.
```

---

## Queued — activate when Phase 1 hardware-validates

### trnrand — Philox / Box-Muller validation story

```
(Use this prompt when trnrand Phase 1 (issue #18) lands — i.e., when the
Philox 4x32 and Box-Muller NKI kernels have been validated on trn1/trn2.)

You maintain trnrand, the cuRAND-equivalent in the trnsci suite. Phase 1
hardware validation just completed: the Philox 4x32-10 NKI kernel on the
GpSimd engine and the Box-Muller kernel on the Vector Engine are verified
against the Salmon SC'11 published test vectors and match PyTorch
reference moments.

Frame the post around what Trainium's four-engine architecture affords
for RNG, not around porting cuRAND. The interesting angle: RNG is where
Trainium's *non*-Tensor-Engine resources (GpSimd for integer multiply-
XOR, Vector Engine for Box-Muller's cos/sin/log/sqrt) become first-class,
not afterthoughts. A CUDA programmer reaches for cuRAND and gets a
single-purpose RNG; on Trainium the RNG naturally spans engines in a
way that mirrors the workload's structure (counter-based integer stage
→ transcendental stage → downstream consumer). That's worth articulating.

Load-bearing points: Philox is stateless by design, which makes
partition-axis splitting trivially correct. Box-Muller on Vector Engine
fuses with downstream consumers (e.g., noise injection into trnfft's
STFT) without going through HBM. Counter-based RNG on GpSimd is a
different architectural story than Mersenne Twister on SM.

Before writing: read https://trnsci.dev/blog/AUTHOR_BRIEF/. Usual rules —
authorless, nine sections with "What the architecture suggests" as the
heart, "What didn't work" required (candidates: integer-op gotchas on
GpSimd, host-device transfer cost benchmarks that motivated on-device
generation).

Open the PR as docs/blog/posts/<date>-trnrand-philox-on-gpsimd.md.
Frontmatter categories: [Deep dive, trnrand].
```

### trnsolver — Jacobi validation story

```
(Use this prompt when trnsolver Phase 1 (issue #26) lands — when the
Jacobi rotation NKI kernel has been validated on trn1/trn2 and at least
@pytest.mark.neuron tests for eigh are passing.)

You maintain trnsolver, the cuSOLVER-equivalent in the trnsci suite.
Phase 1 just landed: the Jacobi rotation NKI kernel on the Tensor Engine,
validated for correctness against torch.linalg.eigh on random symmetric
matrices up to n=512.

Frame around what the 128-partition Tensor Engine tile affords for
eigendecomposition, not around porting cuSOLVER. The core architectural
insight: a Givens rotation is a rank-2 matmul, and rank-2 is what the
Tensor Engine tile shape is. Householder's reflector chains don't tile;
rotations do. That asymmetry — "the hardware doesn't want Householder,
it wants Jacobi" — is a concrete case where the naive port is worse than
the architecturally-native choice even though FLOP count disagrees.

Additional architectural angles: batched-within-sweep parallelism (two
rotations on disjoint pairs commute → run them simultaneously on
different Tensor Engine slices in Phase 3), convergence check as a
Scalar Engine reduction rather than a host round-trip, eigenvector
accumulation as column updates that stay SBUF-resident.

Before writing: https://trnsci.dev/blog/AUTHOR_BRIEF/. Usual rules —
authorless, nine sections, architecture-first. "What didn't work" —
Householder scaffolding you tried first, convergence-criterion missteps,
single-rotation dispatch overhead observations.

docs/blog/posts/<date>-trnsolver-jacobi-for-trainium.md.
Categories: [Deep dive, trnsolver].
```

### trntensor — einsum / contraction planner story

```
(Use this prompt when trntensor Phase 1 (issue #27) lands — when the
matmul/batched-matmul NKI kernels are hardware-validated and the
contraction planner's dispatch paths are wired end-to-end.)

You maintain trntensor, the cuTENSOR-equivalent in the trnsci suite.
Phase 1 just landed: NKI matmul and batched-matmul kernels validated on
trn1/trn2, and the ContractionPlan object's dispatch now actually routes
to the NKI backend when shapes qualify.

Frame around what whole-program NKI compilation and SBUF-resident
intermediates afford for tensor contraction, not around porting
cuTENSOR. The core architectural story: NKI can compile a multi-step
contraction DAG into one kernel where intermediates never leave SBUF.
cuTENSOR has a plan API that hides the kernel boundary; on Trainium the
kernel boundary is what you design around. A Tucker / THC pipeline that
touches a rank-R tensor multiple times can keep R-resident across
contractions — that's a cuTENSOR *superset*, not a cuTENSOR clone.

Additional angles: the FLOPs estimator drives host-side routing among
matmul/bmm/torch.einsum/nki paths at plan time. Fused kernels aren't a
Phase 3 perf nicety — they're the natural Trainium shape for contraction,
and unfused multi-op dispatch is the perf-compromise path.

Before writing: https://trnsci.dev/blog/AUTHOR_BRIEF/. Usual rules —
authorless, nine sections, architecture-first. "What didn't work" —
candidates: the DF-MP2 pair pattern that motivated fused kernels, plans
that didn't dispatch well initially, operation-fusion attempts that
didn't pan out.

docs/blog/posts/<date>-trntensor-contraction-plans.md.
Categories: [Deep dive, trntensor].
```

---

## Editorial templates (for Scott)

### Monthly suite digest

Bylined to `scttfrdmn`. Voice: "we". Target length: 400–600 words. Published on/near the first of each month.

```markdown
---
date: 2026-05-01
authors: [scttfrdmn]
categories: [Digest]
comments: true
---

# April 2026 digest

One-sentence lead: what was the headline development of the month across
the suite.

<!-- more -->

## What landed

- [Library]: one line per shipped release or phase milestone. Link the
  CHANGELOG / GitHub release. If there was a blog post, link it.
- [Library]: ...

## What's in flight

- [Library]: one line per active piece of work, linked to the phase
  tracker issue.
- ...

## Community

conda-forge PR status, new contributors, talks, external mentions, AWS
Neuron SDK version bumps that affect us.

## Next month

One paragraph on what we expect to land. Keep it honest — underpromise.

---

Live tracker: [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).
```

Open the PR against `trnsci/trnsci` as `docs/blog/posts/<YYYY-MM-DD>-<month>-digest.md` (e.g. `2026-05-01-april-digest.md`).

---

## Thinking piece (occasional, Scott)

No template — write what you want. Length flexible. Bylined. Usual frontmatter (`authors: [scttfrdmn]`, `categories: [Editorial]`, `comments: true`).

Candidates from material already in the site / discussions:

- *"Trainium between NVIDIA SMs and Google TPUs"* — expanded version of the [positioning page](https://trnsci.dev/trainium_positioning/).
- *"What open source looks like when the stakes get real"* — the MkDocs/Material governance observation, if you want to write it publicly.
- *"A scientific Python stack that isn't NumPy + CuPy"* — forward-looking thought piece on where the suite goes next.
- *"Six libraries, one roadmap: coordinating a CUDA-style stack on AWS silicon"* — suite-level coordination lessons learned.
