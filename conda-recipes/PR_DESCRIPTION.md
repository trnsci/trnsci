# Adding the `trnsci` scientific-computing suite (7 packages)

This PR adds seven packages forming the **trnsci** scientific-computing suite for AWS Trainium.

## Suite summary

`trnsci` is a CUDA-library-equivalent stack for AWS Trainium via the Neuron Kernel Interface (NKI). Same shape as the NVIDIA cuFFT / cuBLAS / cuRAND / cuSOLVER / cuSPARSE / cuTENSOR stack, Python-first, Apache 2.0.

| Package | NVIDIA analog | Scope |
|---|---|---|
| `trnsci` | — | Coordinating meta-package (no runtime code) |
| `trnfft` | cuFFT | FFT, complex tensors, STFT, complex NN layers |
| `trnblas` | cuBLAS | BLAS Levels 1–3, batched GEMM |
| `trnrand` | cuRAND | Philox PRNG, Sobol / Halton QMC |
| `trnsolver` | cuSOLVER | Cholesky / LU / QR, Jacobi eigh, CG / GMRES |
| `trnsparse` | cuSPARSE | CSR / COO, SpMV / SpMM, Schwarz screening |
| `trntensor` | cuTENSOR | einsum with planning, CP / Tucker decomp |

All seven are pure-Python (`noarch: python`), already published to PyPI, already have tests and CI passing on CPU. Docs: https://trnsci.dev/. Source: https://github.com/trnsci.

## Why group-submit

These are sibling packages maintained together. Each release cycle bumps them in a coordinated way; `trnsci` (meta) depends on the other six. Grouping the PR avoids seven sequential round-trips and lets reviewers see the suite as a whole.

## Checklist

Per-recipe items are satisfied in each `meta.yaml`:

- [x] Title of the PR references what is being added
- [x] All seven recipes have `noarch: python` (pure Python)
- [x] `license_family` set to `APACHE`
- [x] `license_file` points at the bundled Apache 2.0 text
- [x] `run` requirements: `python >=3.10`, `pytorch >=2.1`, `numpy >=1.24`
- [x] `test.imports` covers the package and its `nki` submodule where present
- [x] `about.home`, `about.dev_url`, `about.doc_url` all filled
- [x] `extra.recipe-maintainers` set to the single maintainer (the author)

## Neuron runtime — out of scope for conda-forge

Each recipe's description notes that NKI acceleration depends on AWS's `neuronxcc` and `torch-neuronx` packages, which AWS distributes via their own pip index (not conda-forge, not PyPI proper). This mirrors the situation for CUDA-only libraries like early-days cupy on conda-forge — the Python surface is on conda-forge, the vendor runtime is installed separately.

All recipes declare only pure-Python / PyTorch deps. On non-Neuron hardware (any developer laptop), every library runs end-to-end via its PyTorch fallback. Neuron users `pip install neuronxcc torch-neuronx` from AWS's index inside the conda env; nothing in the conda recipe needs to change.

## Maintainer

Single maintainer (myself, `@scttfrdmn`) for all seven recipes. I'm the sole maintainer of the upstream suite and the author of every dist on PyPI. Happy to add co-maintainers as the community grows.

## Linting

`conda-smithy recipe-lint recipes/trn*` passes locally for all seven. I'll re-run against the PR build matrix and respond to any additional checks.
