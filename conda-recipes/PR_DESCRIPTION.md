# Adding the `trnsci` scientific-computing suite (7 packages)

This PR adds seven packages forming the **trnsci** scientific-computing suite for AWS Trainium.

## Suite summary

`trnsci` is a CUDA-library-equivalent stack for AWS Trainium via the Neuron Kernel Interface (NKI). Same shape as the NVIDIA cuFFT / cuBLAS / cuRAND / cuSOLVER / cuSPARSE / cuTENSOR stack, Python-first, Apache 2.0.

| Package | NVIDIA analog | Scope |
|---|---|---|
| `trnsci` | — | Meta-package: depends on all six libraries (no runtime code) |
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

- [x] Title of this PR is meaningful.
- [x] Recipes use the v1 `recipe.yaml` format.
- [x] License file is packaged (`license_file: LICENSE`, Apache-2.0).
- [x] Source is from official source (the PyPI sdists).
- [x] Packages do not vendor other packages.
- [x] No static libraries (pure Python, `noarch: python`).
- [x] Build number is 0.
- [x] A tarball (`url`) rather than a repo is used.
- [x] The GitHub user listed as maintainer has confirmed in a comment.

Per recipe: `run` is `python >=${{ python_min }}` (conda-forge's global minimum; upstream supports Python >=3.10), `pytorch >=2.1`, `numpy >=1.24`. The `trnsci` meta-package instead depends on the six libraries, since conda has no extras. Tests import the package and its `nki` submodule (and `trnfft.nn`), run `pip check`, and run on both `python_min` and the latest Python.

## Neuron runtime — out of scope for conda-forge

Each recipe's description notes that NKI acceleration depends on AWS's `neuronxcc` and `torch-neuronx` packages, which AWS distributes via their own pip index (not conda-forge, not PyPI proper). This mirrors the situation for CUDA-only libraries like early-days cupy on conda-forge — the Python surface is on conda-forge, the vendor runtime is installed separately.

All recipes declare only pure-Python / PyTorch deps. On non-Neuron hardware (any developer laptop), every library runs end-to-end via its PyTorch fallback. Neuron users `pip install neuronxcc torch-neuronx` from AWS's index inside the conda env; nothing in the conda recipe needs to change.

## Maintainer

Single maintainer (myself, `@scttfrdmn`) for all seven recipes. I'm the sole maintainer of the upstream suite and the author of every dist on PyPI. Happy to add co-maintainers as the community grows.

## Linting

`conda-smithy recipe-lint --conda-forge` reports all seven "in fine form", and all seven build and pass their tests locally with `rattler-build` against the conda-forge pinning (libraries first, then `trnsci`).
