# Libraries

The suite is six independent libraries. Each has its own repository, changelog, and test suite. Their full documentation is composed into this site under the paths below.

| Library | What | Latest | Docs | Repository |
|---|---|---|---|---|
| **trnfft** | cuFFT-equivalent — FFT, complex tensors, STFT, spherical harmonic transforms (`trnfft.sht`) | [[pypi:trnfft]] | [trnsci.dev/trnfft/](/trnfft/) | [github.com/trnsci/trnfft](https://github.com/trnsci/trnfft) |
| **trnblas** | cuBLAS-equivalent — BLAS Levels 1–3, DF-MP2 energy, multi-core DF-MP2 | [[pypi:trnblas]] | [trnsci.dev/trnblas/](/trnblas/) | [github.com/trnsci/trnblas](https://github.com/trnsci/trnblas) |
| **trnrand** | cuRAND-equivalent — Threefry/Philox PRNG, Sobol (Owen-scrambled)/Halton QMC | [[pypi:trnrand]] | [trnsci.dev/trnrand/](/trnrand/) | [github.com/trnsci/trnrand](https://github.com/trnsci/trnrand) |
| **trnsolver** | cuSOLVER-equivalent — factorizations, eigh, CG/GMRES, block CG, SCF eigensolvers | [[pypi:trnsolver]] | [trnsci.dev/trnsolver/](/trnsolver/) | [github.com/trnsci/trnsolver](https://github.com/trnsci/trnsolver) |
| **trnsparse** | cuSPARSE-equivalent — BSR-128/CSR/COO, SpMV/SpMM, block-sparse attention, fused kernel-matrix operator | [[pypi:trnsparse]] | [trnsci.dev/trnsparse/](/trnsparse/) | [github.com/trnsci/trnsparse](https://github.com/trnsci/trnsparse) |
| **trntensor** | cuTENSOR-equivalent — einsum, CP/Tucker/TT, streamed DF integrals (`AuxStream`) | [[pypi:trntensor]] | [trnsci.dev/trntensor/](/trntensor/) | [github.com/trnsci/trntensor](https://github.com/trnsci/trntensor) |

The "Latest" column is filled in from PyPI when the site is built. Each library's CHANGELOG lists what changed, and its README has a "Known issues in earlier releases" table.

!!! note "How this site is built"
    The umbrella CI clones each sub-project at build time and composes their `mkdocs.yml` into this site via [mkdocs-monorepo-plugin](https://github.com/backstage/mkdocs-monorepo-plugin). Each sub-project's repository is the single source of truth for its own docs — changes there land here on the next umbrella rebuild (daily cron, or a manual dispatch via `gh workflow run "Deploy docs" --repo trnsci/trnsci`).

Looking for the CUDA mapping across these libraries? See [CUDA → trnsci](cuda_rosetta.md). For what they add up to on real workloads, see [exemplars](exemplars.md).
