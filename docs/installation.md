# Installation

## From PyPI

```bash
# All six libraries
pip install "trnsci[all]"

# Just the ones you need
pip install "trnsci[fft]"
pip install "trnsci[blas,solver,tensor]"

# Or the libraries directly
pip install trnfft trnblas
```

Python ≥ 3.10, torch ≥ 2.1 and numpy ≥ 1.24. Without Neuron hardware every library runs its PyTorch fallback.

| Library | `trnsci[all]` minimum | Latest on PyPI |
|---|---|---|
| trnfft | [[min:trnfft]] | [[pypi:trnfft]] |
| trnblas | [[min:trnblas]] | [[pypi:trnblas]] |
| trnrand | [[min:trnrand]] | [[pypi:trnrand]] |
| trnsolver | [[min:trnsolver]] | [[pypi:trnsolver]] |
| trnsparse | [[min:trnsparse]] | [[pypi:trnsparse]] |
| trntensor | [[min:trntensor]] | [[pypi:trntensor]] |

The minimums are the umbrella repo's [`pyproject.toml`](https://github.com/trnsci/trnsci/blob/main/pyproject.toml), and the site fills both columns in when it is built. They are published from meta-package 0.2.0 on (0.1.0 carried much older ones), so `pip install -U "trnsci[all]"` brings an existing environment up to at least these releases. To move every library to its newest release instead, use `pip install -U --upgrade-strategy eager "trnsci[all]"`.

## On Neuron hardware

**Trn2 / Trn3 (TorchNeuron, the native path).** Start from the Neuron Deep Learning AMI for SDK 2.33 and its PyTorch 2.14 environment, which already has `torch-neuronx` 2.14 and NKI 0.7. Then `pip install "trnsci[all]"`. To install the Neuron packages yourself, add the `[torchneuron]` extra (`torch>=2.14`, `torch-neuronx>=2.14`, `nki>=0.6`, the same as each library's own `[torchneuron]` extra). They come from the AWS pip index, and `torch-neuronx` 2.14 needs Python ≥ 3.12:

```bash
pip install "trnsci[all,torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
# or one library
pip install "trnblas[torchneuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

**Trn1 / Inf2 (PyTorch/XLA, maintenance).** The umbrella's `[neuron]` extra names the XLA-lineage packages (`neuronxcc>=2.24`, `torch-neuronx>=2.9`), which also come from the AWS pip index:

```bash
pip install "trnsci[all,neuron]" --extra-index-url https://pip.repos.neuron.amazonaws.com
```

Each sub-project's installation page has the per-library detail, for example [trnblas](trnblas/installation.md).

## From source

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
# The sub-projects are separate repositories; clone them next to the umbrella files.
# trnsolver lives in the directory trnsolve.
for p in trnfft trnblas trnrand trnsparse trntensor; do
  [ -d "$p" ] || git clone "git@github.com:trnsci/$p.git"
done
[ -d trnsolve ] || git clone git@github.com:trnsci/trnsolver.git trnsolve
make install-dev
```

`make install-dev` runs `pip install -e ./<pkg>[dev]` for each of the six sub-projects and then installs the umbrella in editable mode.

## Hardware compatibility

trnsci runs NKI kernels through two PyTorch integrations. Which one you get depends on the instance generation:

| Instance | PyTorch integration | Status in trnsci |
|---|---|---|
| **Trn2, Trn3** | **TorchNeuron** (native PyTorch backend, `device="neuron"`). Neuron SDK 2.33+, PyTorch 2.13/2.14, NKI 0.7 | Primary target. Validated on trn2.3xlarge with SDK 2.33 (public beta); see [validation status](nki_validation_status.md#current-releases). All six libraries ship native dispatch (since 2026-10-30). |
| **Trn1, Inf2** | **PyTorch/XLA** (`torch-neuronx` 2.9). Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04) | Maintenance only. TorchNeuron does not support Trn1/Inf2, and PyTorch 2.9 is the last PyTorch/XLA release. Bug fixes only, no new kernels. |

### XLA path deprecation

AWS Neuron is replacing PyTorch/XLA with TorchNeuron. When TorchNeuron reaches general availability, trnsci will **deprecate its XLA path on Trn2/Trn3**: the libraries will emit a `DeprecationWarning` when they fall back to XLA on a Trn2+ device. On Trn1/Inf2 the XLA path stays available in maintenance mode, because no native path exists there. Any later removal will be announced separately. The decision record is on [#35](https://github.com/trnsci/trnsci/issues/35), and the steps for GA are on [#136](https://github.com/trnsci/trnsci/issues/136).

See each sub-project's `docs/aws_setup.md` for instance-type specifics.
