# Installation

## From PyPI (once published)

```bash
# All six libraries
pip install trnsci[all]

# Just the ones you need
pip install trnsci[fft]
pip install trnsci[blas,solver,tensor]

# On Neuron hardware
pip install trnsci[all,neuron]
```

## From source

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
make install-dev
```

`make install-dev` runs `pip install -e ./<pkg>[dev]` for each of the six sub-projects and then installs the umbrella in editable mode.

## Hardware compatibility

trnsci runs NKI kernels through two PyTorch integrations. Which one you get depends on the instance generation:

| Instance | PyTorch integration | Status in trnsci |
|---|---|---|
| **Trn2, Trn3** | **TorchNeuron** (native PyTorch backend, `device="neuron"`). Neuron SDK 2.33+, PyTorch 2.13/2.14, NKI 0.7 | Primary target. Validated on trn2.3xlarge with SDK 2.33.0 (public beta); see [#35](https://github.com/trnsci/trnsci/issues/35). Library-by-library native ports are in progress. |
| **Trn1, Inf2** | **PyTorch/XLA** (`torch-neuronx` 2.9). Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04) | Maintenance only. TorchNeuron does not support Trn1/Inf2, and PyTorch 2.9 is the last PyTorch/XLA release. Bug fixes only, no new kernels. |

### XLA path deprecation

AWS Neuron is replacing PyTorch/XLA with TorchNeuron. When TorchNeuron reaches general availability, trnsci will **deprecate its XLA path on Trn2/Trn3**: the libraries will emit a `DeprecationWarning` when they fall back to XLA on a Trn2+ device. On Trn1/Inf2 the XLA path stays available in maintenance mode, because no native path exists there. Any later removal will be announced separately. The decision record is on [#35](https://github.com/trnsci/trnsci/issues/35).

See each sub-project's `docs/aws_setup.md` for instance-type specifics.
