# trnsci

Scientific computing suite for AWS Trainium via NKI.

NVIDIA's CUDA ecosystem ships cuFFT, cuBLAS, cuRAND, cuSOLVER, cuSPARSE, and cuTENSOR. AWS Neuron SDK ships none of these. **trnsci fills that gap.**

## The six libraries

| Library | Analog | What it provides |
|---|---|---|
| [trnfft](trnfft/) | cuFFT | FFT (Cooley-Tukey, Bluestein), complex tensors, complex NN layers, STFT |
| [trnblas](trnblas/) | cuBLAS | Level 1-3 BLAS, batched GEMM, DF-MP2 primitives |
| [trnrand](trnrand/) | cuRAND | Philox PRNG, distributions, Sobol / Halton / LHS QMC |
| [trnsolver](trnsolver/) | cuSOLVER | Cholesky / LU / QR, Jacobi eigh, CG / GMRES |
| [trnsparse](trnsparse/) | cuSPARSE | CSR / COO, SpMV / SpMM, Schwarz integral screening |
| [trntensor](trntensor/) | cuTENSOR | einsum w/ contraction planning, CP / Tucker decompositions |

## How they compose

Each library is independent; none import another. The umbrella provides a single meta-install (`pip install trnsci[all]`) and a cross-project integration example ([DF-MP2](integration.md)) that exercises multiple libraries in one pipeline.

## Design principles

- **Match NVIDIA API shapes** — CUDA programmers can navigate trnsci without a guide.
- **PyTorch fallback on CPU** — everything works off-hardware for development and CI.
- **NKI dispatch** — `set_backend("auto"|"pytorch"|"nki")` across the suite.
- **FP32-first** — Trainium's Tensor Engine is FP32-only; chemistry workarounds (double-double, Kahan) are opt-in.

## Status

All sub-projects are **Alpha**. CPU/PyTorch paths are wired end-to-end. NKI kernels are scaffolded; on-hardware validation on trn1/trn2 is the next milestone.
