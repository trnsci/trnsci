# trnsci

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-trnsci.dev-blue)](https://trnsci.dev/)

Scientific computing suite for AWS Trainium via NKI. Six focused libraries providing the cu\* equivalents the Neuron SDK doesn't ship.


## Suite

| Project | NVIDIA analog | Scope |
|---|---|---|
| [trnfft](trnfft/) | cuFFT | FFT, complex tensors, complex NN layers |
| [trnblas](trnblas/) | cuBLAS | Level 1–3 BLAS, batched GEMM |
| [trnrand](trnrand/) | cuRAND | Philox PRNG, Sobol/Halton QMC, LHS |
| [trnsolver](trnsolve/) | cuSOLVER | Cholesky/LU/QR, Jacobi eigh, CG/GMRES |
| [trnsparse](trnsparse/) | cuSPARSE | CSR/COO, SpMV/SpMM, Schwarz screening |
| [trntensor](trntensor/) | cuTENSOR | einsum with contraction planning, CP/Tucker |

## Install

```bash
# Meta-package with all six sub-projects
pip install trnsci[all]

# Individual components
pip install trnsci[fft]     # just trnfft
pip install trnsci[blas]    # just trnblas
# ... etc

# On Neuron hardware
pip install trnsci[all,neuron]
```

## Development install

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
make install-dev   # pip install -e on each sub-project + umbrella
make test-all      # run pytest across all sub-projects
```

## Cross-project example

```bash
python examples/quantum_chemistry/df_mp2_synthetic.py --demo
```

DF-MP2 energy evaluation composing `trnblas` (half-transforms), `trnsolver` (Cholesky of DF metric), and `trntensor` (einsum contraction). See also `examples/nvidia_samples/` for direct ports of canonical NVIDIA CUDA samples.

## Status

All sub-projects are **Alpha**. CPU/PyTorch fallback is functional end-to-end. NKI kernels are scaffolded across the suite; on-hardware validation is the next milestone.

## Roadmap

Five-phase plan from current alpha to generation-tuned stable release. See [`ROADMAP.md`](ROADMAP.md) or the [reader-oriented version on trnsci.dev](https://trnsci.dev/roadmap/).

## Community

- Site: [trnsci.dev](https://trnsci.dev/)
- [Contributing guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## License

Apache 2.0 — Copyright 2026 Scott Friedman
