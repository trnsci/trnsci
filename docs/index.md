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

Phase 1 has landed across all six libraries. NKI paths run end-to-end and are exercised in CI via the NKI CPU simulator on every PR. Hardware validation on trn1 is complete for **trnfft** (butterfly FFT + complex GEMM, 70/70 benchmark cases) and **trnblas** (GEMM/SYRK + fused DF-MP2 energy, PySCF agreement to 10 µHa on H₂O / CH₄ / NH₃ at cc-pVDZ). **trnsolver**, **trnsparse**, and **trntensor** are simulator-validated with hardware runs in progress. **trnrand** is simulator-validated and blocked on a named upstream NKI integer-multiply issue ([aws-neuron-sdk#1308](https://github.com/aws-neuron/aws-neuron-sdk/issues/1308)); the PyTorch fallback is the default until that lands.

PyTorch fallback works end-to-end on any machine, with or without Neuron hardware. The [NKI validation status page](nki_validation_status.md) carries the per-library detail.

Read more:

- [Why this exists](why.md)
- [Trainium's place between SMs and TPUs](trainium_positioning.md)
- [CUDA → trnsci library mapping](cuda_rosetta.md)
- [Cross-library integration example](workflows/integration.md)

Follow the [blog](blog/index.md) for monthly suite digests and technical deep-dives from the sub-project libraries.
