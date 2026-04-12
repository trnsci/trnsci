# trnsci

Scientific computing libraries for AWS Trainium.

NVIDIA CUDA programmers reach for **cuFFT**, **cuBLAS**, **cuRAND**, **cuSOLVER**, **cuSPARSE**, and **cuTENSOR** when they need fast numerical primitives. The AWS Neuron SDK ships none of these. `trnsci` is six libraries that fill the gap:

| trnsci | NVIDIA analog | Scope |
|---|---|---|
| [trnfft](trnfft/) | cuFFT | FFT, complex tensors, STFT, complex NN layers |
| [trnblas](trnblas/) | cuBLAS | BLAS Levels 1–3, batched GEMM |
| [trnrand](trnrand/) | cuRAND | Philox PRNG, Sobol/Halton QMC |
| [trnsolver](trnsolve/) | cuSOLVER | Cholesky/LU/QR, Jacobi eigh, CG/GMRES |
| [trnsparse](trnsparse/) | cuSPARSE | CSR/COO, SpMV, SpMM, Schwarz screening |
| [trntensor](trntensor/) | cuTENSOR | Einstein summation with planning, CP/Tucker |

## What is this for

Workloads that don't fit into a deep-learning framework but still need fast linear algebra on accelerator hardware. Signal processing, quantum chemistry, Monte Carlo, spectral methods, sparse linear systems — all of them routinely depend on cuFFT, cuBLAS, and siblings. `trnsci` brings the same primitives to Trainium, with a PyTorch-first API and optional NKI kernels underneath.

## Who is this for

- **CUDA programmers** who want the mental model they already have to map onto Trainium. See the [CUDA → trnsci Rosetta stone](cuda_rosetta.md).
- **Trainium programmers** who want a scientific-computing library stack that isn't deep-learning-specific.

## Get started

```bash
pip install trnsci[all]
```

Try the cross-library integration demo:

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
make install-dev
python examples/quantum_chemistry/df_mp2_synthetic.py --demo
```

## Status

**Alpha across the suite.** PyTorch fallback works end-to-end on any machine. NKI kernels are scaffolded; on-hardware validation on trn1 / trn2 is the next milestone.

Read more:

- [Why this exists](why.md)
- [Trainium's place between SMs and TPUs](trainium_positioning.md)
- [CUDA → trnsci library mapping](cuda_rosetta.md)
- [Cross-library integration example](workflows/integration.md)
