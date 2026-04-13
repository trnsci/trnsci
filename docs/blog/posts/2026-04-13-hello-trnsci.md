---
date: 2026-04-13
categories: [Announcements]
authors: [scttfrdmn]
comments: true
---

# Hello trnsci

The [trnsci](https://trnsci.dev) scientific computing suite for AWS Trainium is public. Six libraries covering the CUDA cu\* equivalents the Neuron SDK ships without, a coordinating meta-package, full docs, seven PyPI packages, a conda-forge submission in review, and a five-phase roadmap from current alpha to generation-tuned stable. This is the first post of a blog series that will tell the project's story as it unfolds.

<!-- more -->

## What trnsci is

The Neuron SDK gives AWS Trainium a tile-level programming model (NKI) and a set of fused transformer kernels. Everything else — FFT, BLAS, RNG, solvers, sparse, tensor contractions — has to be hand-rolled. On NVIDIA these are cuFFT, cuBLAS, cuRAND, cuSOLVER, cuSPARSE, and cuTENSOR, and a scientist using CUDA reaches for them by reflex. On Trainium the equivalent stack didn't exist.

trnsci is that stack: six libraries mirroring the CUDA split, Python-first APIs with PyTorch fallback on any hardware, and NKI kernels underneath for Neuron acceleration. See the [CUDA → trnsci Rosetta stone](https://trnsci.dev/cuda_rosetta/) for the library-by-library mapping, or the [why page](https://trnsci.dev/why/) for the longer-form motivation.

Status across the suite as of today:

- [**trnfft**](https://trnsci.dev/trnfft/) and [**trnblas**](https://trnsci.dev/trnblas/) have real, hardware-validated NKI kernels. trnfft v0.8.0 passes 70/70 benchmark cases on trn1.2xlarge. trnblas matches PySCF to nanohartree tolerance on H2O / CH4 / NH3 at cc-pVDZ for density-fitted MP2.
- [**trnrand**](https://trnsci.dev/trnrand/), [**trnsolver**](https://trnsci.dev/trnsolver/), [**trnsparse**](https://trnsci.dev/trnsparse/), and [**trntensor**](https://trnsci.dev/trntensor/) have Phase 1 NKI code scaffolded but are mid-validation. Progress is tracked in [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1).

## What this blog will be

Three tracks, running in parallel, with different cadences.

**Monthly suite digests.** Short editorial posts covering what landed across the suite, what's queued, and community items (conda-forge PR status, new releases, talks, mentions). Published on or near the first of each month.

**Technical deep-dives as work ships.** When a sub-project lands non-trivial NKI work, it gets a retrospective post — what the problem was, what the design is, a code sample, benchmark numbers, and *what didn't work*. Two upcoming: trnfft on how FFT works on hardware without a complex dtype, and trnblas on the DF-MP2 GEMM path that hit nanohartree tolerance against PySCF.

**Occasional thinking pieces** about Trainium's place between NVIDIA and Google TPU, about the state of scientific Python on non-NVIDIA accelerators, about the governance questions that come up when small projects get real stakes. Lower frequency, higher shelf life.

## Editorial stance

Two things worth saying up front about what this blog is going to be.

**Architecture-first, not port-first.** The trnsci suite isn't interesting because it ports cuFFT / cuBLAS / cuRAND / cuSOLVER / cuSPARSE / cuTENSOR techniques one-to-one. What's interesting is what Trainium's specific architecture — four programmable engines (Tensor, Vector, Scalar, GpSimd), a fixed 128-partition × 512-moving tile shape, explicit SBUF / PSUM memory, DMA as a first-class resource, whole-program NKI compilation — makes natural that a GPU wouldn't suggest. Jacobi rotations on the Tensor Engine because each is a rank-2 matmul that fits the tile exactly. BSR at 128×128 as the native sparse format, not a port of cuSPARSE BSR. Fused MP2 energy kernels that keep intermediates SBUF-resident across contractions, which cuBLAS can't express as one call. Writing about these reveals something about the hardware; writing about "how we replicated the CUDA approach" reveals nothing. The [author brief](https://trnsci.dev/blog/AUTHOR_BRIEF/) for technical deep-dives requires a "What the architecture suggests" section for exactly this reason.

**Transparency over polish.** The project has already reverted kernels that looked right on paper, hit NKI compiler surprises that took days to explain, and produced benchmark numbers that were honestly disappointing on some shapes. Those stories are more useful to other people building on Trainium than "trnsci delivers unprecedented throughput" would be. The author brief requires a "What didn't work" section because we've watched too many vendor-driven engineering blogs skip that part, and the result is a public narrative that doesn't match what any real user experiences.

Benchmarks appear in every technical post, but they're context — they confirm or contextualize an architectural choice, not justify one on their own. A post with modest numbers and a clear articulation of what the hardware enabled is worth more than a post with bigger numbers and no insight.

If the blog reads like marketing, it's failing. If it reads like a maintainer talking to another maintainer over coffee, it's working.

## How the writing works

Technical posts are unsigned, with the library as the subject ("trnfft's butterfly kernel", not "I built"). The work is collaborative between Scott Friedman (suite director) and the Claude agents building each library. Authorship distinctions across that collaboration are arbitrary; what matters is the technical content. Editorial posts — these digests and thinking pieces — are bylined because curatorial or opinion voice is being exercised.

## How to follow along

- **RSS:** [`https://trnsci.dev/blog/feed_rss_created.xml`](https://trnsci.dev/blog/feed_rss_created.xml)
- **GitHub:** watch the [trnsci org](https://github.com/trnsci) or any individual sub-project
- **PyPI:** `pip install trnsci[all]` for the meta-package + all six libraries
- **Suite tracker:** [trnsci/trnsci#1](https://github.com/trnsci/trnsci/issues/1) — live checkbox view of progress across the 5-phase roadmap

Comments on this post use [giscus](https://giscus.app) and are backed by GitHub Discussions on the umbrella repo. The widget activates once the final wiring lands (it's in-flight at the time of publishing).

More soon.
