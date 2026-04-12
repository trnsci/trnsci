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

NKI kernels across all six sub-projects are validated against **Neuron SDK 2.24+** on the **Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)**. See each sub-project's `docs/aws_setup.md` for instance-type specifics.
