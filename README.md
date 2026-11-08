# trnsci

[![CI](https://github.com/trnsci/trnsci/actions/workflows/test.yml/badge.svg)](https://github.com/trnsci/trnsci/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/trnsci/trnsci/graph/badge.svg)](https://codecov.io/gh/trnsci/trnsci)
[![pre-commit](https://results.pre-commit.ci/badge/github/trnsci/trnsci/main.svg)](https://results.pre-commit.ci/latest/github/trnsci/trnsci/main)
[![PyPI](https://img.shields.io/pypi/v/trnsci)](https://pypi.org/project/trnsci/)
[![Python](https://img.shields.io/pypi/pyversions/trnsci)](https://pypi.org/project/trnsci/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-trnsci.dev-blue)](https://trnsci.dev/)

Scientific computing suite for AWS Trainium via NKI. Six focused libraries providing the cu\* equivalents the Neuron SDK doesn't ship.


## Suite

| Project | NVIDIA analog | Scope | PyPI |
|---|---|---|---|
| [trnfft](https://github.com/trnsci/trnfft) | cuFFT | FFT, complex tensors, STFT, spherical harmonic transforms, complex NN layers | [![trnfft](https://img.shields.io/pypi/v/trnfft)](https://pypi.org/project/trnfft/) |
| [trnblas](https://github.com/trnsci/trnblas) | cuBLAS | BLAS Levels 1–3, batched GEMM, DF-MP2 energy | [![trnblas](https://img.shields.io/pypi/v/trnblas)](https://pypi.org/project/trnblas/) |
| [trnrand](https://github.com/trnsci/trnrand) | cuRAND | Threefry/Philox PRNG, Sobol/Halton QMC | [![trnrand](https://img.shields.io/pypi/v/trnrand)](https://pypi.org/project/trnrand/) |
| [trnsolver](https://github.com/trnsci/trnsolver) | cuSOLVER | Cholesky/LU/QR/SVD, Jacobi eigh, CG/GMRES and block CG, SCF eigensolvers | [![trnsolver](https://img.shields.io/pypi/v/trnsolver)](https://pypi.org/project/trnsolver/) |
| [trnsparse](https://github.com/trnsci/trnsparse) | cuSPARSE | BSR-128 SpMM, Schwarz screening, block-sparse attention, fused kernel-matrix operator | [![trnsparse](https://img.shields.io/pypi/v/trnsparse)](https://pypi.org/project/trnsparse/) |
| [trntensor](https://github.com/trnsci/trntensor) | cuTENSOR | Einstein summation with path planning, CP/Tucker/TT, streamed DF integrals, multi-chip sharding | [![trntensor](https://img.shields.io/pypi/v/trntensor)](https://pypi.org/project/trntensor/) |

## Install

```bash
# Meta-package with all six sub-projects
pip install "trnsci[all]"

# Individual components
pip install "trnsci[fft]"     # just trnfft
pip install "trnsci[blas]"    # just trnblas
# ... etc
```

On Trn2/Trn3, start from the Neuron SDK 2.33 Deep Learning AMI (PyTorch 2.14 environment), which already has `torch-neuronx` and NKI. On Trn1/Inf2, `pip install "trnsci[all,neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com` adds the PyTorch/XLA-lineage packages. Details and the current `trnsci[all]` minimums: [installation](https://trnsci.dev/installation/).

## Development install

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
for p in trnfft trnblas trnrand trnsparse trntensor; do
  [ -d "$p" ] || git clone "git@github.com:trnsci/$p.git"
done
[ -d trnsolve ] || git clone git@github.com:trnsci/trnsolver.git trnsolve
make install-dev   # pip install -e on each sub-project + umbrella
make test-all      # run pytest across all sub-projects
```

## Cross-project example

```bash
python examples/quantum_chemistry/df_mp2_synthetic.py --demo
python examples/quantum_chemistry/df_mp2_synthetic.py --demo --device neuron   # Trn2/Trn3
```

Closed-shell DF-MP2 energy evaluation composing `trnrand` (synthetic inputs), `trnsparse` (Schwarz screening report), `trnsolver` (Cholesky of the DF metric), `trnblas` (half-transforms) and `trntensor` (`mp2_energy`), checked against a float64 reference. `examples/quantum_chemistry/df_mp2_profile.py` profiles the same pipeline across a ladder of sizes. See also `examples/nvidia_samples/` for direct ports of canonical NVIDIA CUDA samples.

## Status

All sub-projects are **Alpha**. CPU/PyTorch fallback is functional end-to-end. All six libraries ship NKI kernels with native TorchNeuron dispatch on Trn2/Trn3 (Neuron SDK 2.33), every library runs a conformance suite against independent references, and their current releases are validated on Trn2 ([validation status](https://trnsci.dev/nki_validation_status/)). The PyTorch/XLA path on Trn1 is in maintenance. On the native device, `auto` dispatch uses an NKI kernel only where it measured faster than aten, and float64 follows one [suite policy](https://trnsci.dev/precision_and_platform_limits/): the device has no float64 arithmetic, so `auto` computes it on the host.

Workload-level results on one Trn2 chip — a 10⁶-point Gaussian process, spherical harmonic transforms, DF-RHF/MP2 to 3508 basis functions and DeepVariant `call_variants` — are on [exemplars](https://trnsci.dev/exemplars/).

## Roadmap

Five-phase plan from current alpha to generation-tuned stable release, tracked in the [GitHub milestones](https://github.com/trnsci/trnsci/milestones) and on the [project board](https://github.com/orgs/trnsci/projects/1). See the [reader-oriented version on trnsci.dev](https://trnsci.dev/roadmap/) and the phase definitions in [`ROADMAP.md`](ROADMAP.md).

## Community

- Site: [trnsci.dev](https://trnsci.dev/)
- [Contributing guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## License

Apache 2.0 — Copyright 2026 Scott Friedman.

## Disclaimer

trnsci is an **independent open-source project**. It is not sponsored by, endorsed by, or affiliated with Amazon.com, Inc., Amazon Web Services, Inc., or Annapurna Labs Ltd.

"AWS", "Amazon", "Trainium", "Inferentia", "NeuronCore", "Neuron SDK", and related identifiers are trademarks of their respective owners and are used here solely for descriptive and interoperability purposes. Use does not imply endorsement, partnership, or any other relationship.

All work, opinions, analyses, benchmark results, architectural commentary, and editorial judgments in this repository and on [trnsci.dev](https://trnsci.dev) are those of the project's contributors. They do not represent the views, positions, or commitments of Amazon, AWS, or Annapurna Labs.

Feedback directed at the Neuron SDK or Trainium hardware is good-faith ecosystem commentary from independent users. It is not privileged information, is not pre-reviewed by AWS, and should not be read as authoritative about product roadmap, behavior, or quality.

For official AWS guidance, see [aws-neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/) and the [AWS Trainium product page](https://aws.amazon.com/ai/machine-learning/trainium/).
