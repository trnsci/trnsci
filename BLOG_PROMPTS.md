# Blog prompts for sub-project agents

Self-contained prompts to paste into each sub-project's Claude agent when you want them to draft a technical deep-dive for the [trnsci blog](https://trnsci.dev/blog/).

Each prompt is designed to be pasted verbatim — it carries all the context the agent needs (editorial brief, structure, voice, suite positioning). The agent drafts a post in the correct location (`docs/blog/posts/<date>-<slug>.md`), opens a PR against `trnsci/trnsci`, and you/Scott reviews before merge.

Editorial brief the prompts reference: [`docs/blog/AUTHOR_BRIEF.md`](docs/blog/AUTHOR_BRIEF.md).
Template file: [`docs/blog/posts/_template.md`](docs/blog/posts/_template.md).

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
Suggested angle: "FFT on hardware without a complex dtype" — how trnfft
represents complex values (split real/imag ComplexTensor), how complex GEMM
decomposes into four real GEMMs with stationary-tile reuse, how butterfly
stages map onto the Tensor Engine + Vector Engine, and the Kahan
compensated variant that restores FP32 precision for long Bluestein
chains.

Before writing: read https://trnsci.dev/blog/AUTHOR_BRIEF/ (or the local
docs/blog/AUTHOR_BRIEF.md in the trnsci/trnsci umbrella repo). Key rules:

  - Authorless. Library as subject ("trnfft's butterfly kernel", not "I"
    or "we"). No byline.
  - Use the eight required section headings from the brief, in order:
    Lead, The problem, The approach, Implementation, What didn't work,
    Numbers, What's next, Takeaway.
  - Be transparent about blind alleys. NKI compiler surprises, reverted
    approaches, numbers that disappointed — put them in "What didn't
    work". That section is required.
  - Absolute numbers with units. Microseconds, TFLOPS, rel error bounds.
  - 1200–2500 words.

Open the PR as docs/blog/posts/2026-04-DD-trnfft-fft-without-complex-dtype.md
(pick the date) against trnsci/trnsci. Use the frontmatter:

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
Suggested angle: "trnblas: matching PySCF at nanohartree on Trainium" —
the DF-MP2 code path (Cholesky → half-transform → metric contract → pair
energy), the fused MP2 energy kernel that collapses the per-(i,j) store
pattern, how batched_gemm is a hybrid (host-side loop over a real NKI
GEMM) and why that's Phase 1 by design, the FP32 precision story against
FP64 PySCF.

Before writing: read https://trnsci.dev/blog/AUTHOR_BRIEF/ (or the local
docs/blog/AUTHOR_BRIEF.md in the trnsci/trnsci umbrella repo). Key rules:

  - Authorless. Library as subject. No byline.
  - Use the eight required section headings from the brief, in order.
  - "What didn't work" is required. Candidate content: any reverted
    kernel attempts (the examples/df_mp2 revert in #15 looks relevant),
    NKI partition-dim constraints that tripped you up, numbers that
    disappointed at small shapes, the hybrid batched_gemm tradeoff.
  - Include a benchmark table. Cold/warm DF-MP2 wall time by shape,
    TFLOPS, E_MP2 values. Use real numbers from the v0.4.0 CHANGELOG and
    benchmarks.
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
about correctness, not speed; row-bucketing lands in v0.3.0.

Please draft a technical retrospective blog post. Suggested angle:
"trnsparse v0.2.0: shipping SpMM when it's slower than scipy" — the
deliberate Phase 1 tradeoff of correctness over speed, the densify-then-
GEMM design (and its cost at low nnz density), the fact that the full
Neuron toolchain is exercised end-to-end, why nnz-bucketing is a separate
phase, and why autograd validation mattered more than throughput for v0.2.

The "it's slower than scipy" framing is a feature, not a bug. It's a
useful antidote to vendor-marketing posts that only claim wins, and it
sets up why Phase 3 (nnz-bucketing) is the next interesting milestone.

Before writing: read https://trnsci.dev/blog/AUTHOR_BRIEF/ (or the local
docs/blog/AUTHOR_BRIEF.md). Key rules:

  - Authorless. Library as subject. No byline.
  - Eight required sections in order.
  - Include the full benchmark table from docs/benchmarks.md. Don't
    editorialize the numbers — show them, explain them.
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

Please draft a technical retrospective. Suggested angle: counter-based
RNG on a non-NVIDIA accelerator — why Philox over Mersenne Twister, how
integer multiply-XOR maps to GpSimd (not Tensor Engine), Box-Muller on
Vector Engine for the normal distribution, partition-axis parallel
generation that's deterministic given (counter, key).

Before writing: read https://trnsci.dev/blog/AUTHOR_BRIEF/. Usual rules —
authorless, eight sections, "What didn't work" required (candidates:
integer-op gotchas on GpSimd, host-device transfer cost benchmarks that
motivated on-device generation in the first place).

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

Please draft a technical retrospective. Suggested angle: why Jacobi for
Trainium — the asymptotic tradeoff (O(n³) per sweep with O(n) sweeps vs
Householder's better constant) and why tile-friendliness wins on fixed-
tile systolic hardware even when FLOP count is higher. Each Givens
rotation as a rank-2 matmul. Batched-within-sweep parallelism as a Phase
3 item, not Phase 1.

Before writing: https://trnsci.dev/blog/AUTHOR_BRIEF/. Usual rules.
"What didn't work" — any Householder scaffolding you tried first,
convergence-criterion missteps, single-rotation dispatch overhead
observations.

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

Please draft a technical retrospective. Suggested angle: first-class
contraction plans as an API idiom — why plan_contraction returns a
reusable object (vs cuTENSOR's more hidden plan), how the planner picks
matmul/bmm/torch.einsum/nki based on index structure, the FLOPs estimator
and what it's good for (host-side routing decisions, back-of-envelope
benchmarks).

Before writing: https://trnsci.dev/blog/AUTHOR_BRIEF/. Usual rules.
"What didn't work" — candidates: the DF-MP2 pair pattern that motivated
fused kernels, plans that didn't dispatch well initially, any operation-
fusion attempts that didn't pan out.

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
