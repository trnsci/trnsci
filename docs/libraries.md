# Libraries

The suite is six independent libraries. Each has its own repository, changelog, test suite, and documentation site.

| Library | What | Repo | Docs |
|---|---|---|---|
| **trnfft** | cuFFT-equivalent — FFT, complex tensors, STFT | [github.com/trnsci/trnfft](https://github.com/trnsci/trnfft) | [trnsci.dev/trnfft](https://trnsci.dev/trnfft/) |
| **trnblas** | cuBLAS-equivalent — BLAS Levels 1–3 | [github.com/trnsci/trnblas](https://github.com/trnsci/trnblas) | [trnsci.dev/trnblas](https://trnsci.dev/trnblas/) |
| **trnrand** | cuRAND-equivalent — Philox PRNG, Sobol QMC | [github.com/trnsci/trnrand](https://github.com/trnsci/trnrand) | [trnsci.dev/trnrand](https://trnsci.dev/trnrand/) |
| **trnsolver** | cuSOLVER-equivalent — factorizations, eigh | [github.com/trnsci/trnsolver](https://github.com/trnsci/trnsolver) | [trnsci.dev/trnsolver](https://trnsci.dev/trnsolver/) |
| **trnsparse** | cuSPARSE-equivalent — CSR/COO, SpMV/SpMM | [github.com/trnsci/trnsparse](https://github.com/trnsci/trnsparse) | [trnsci.dev/trnsparse](https://trnsci.dev/trnsparse/) |
| **trntensor** | cuTENSOR-equivalent — einsum, CP/Tucker | [github.com/trnsci/trntensor](https://github.com/trnsci/trntensor) | [trnsci.dev/trntensor](https://trnsci.dev/trntensor/) |

!!! note "Sub-project doc sites"
    Each sub-project deploys its own Pages site under the `trnsci.github.io/<name>` subdomain, served at `trnsci.dev/<name>`. These are built by per-project CI workflows and link back to this umbrella site.

Looking for the `trnsci` Rosetta stone that maps CUDA APIs to these libraries? See [CUDA → trnsci](cuda_rosetta.md).
