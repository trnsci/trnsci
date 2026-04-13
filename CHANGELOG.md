# Changelog

All notable changes to the trnsci umbrella package.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-04-13

Initial umbrella release.

### Added

- **Meta-package** that coordinates installation of the six trnsci sub-projects:
  `trnfft`, `trnblas`, `trnrand`, `trnsolver`, `trnsparse`, `trntensor`.
- **Optional extras**: `[fft]`, `[blas]`, `[rand]`, `[solver]`, `[sparse]`,
  `[tensor]`, `[all]`, `[neuron]`, `[dev]`. `pip install trnsci[all]` pulls
  in every sub-project.
- **Cross-project DF-MP2 integration example** at
  `examples/quantum_chemistry/df_mp2_synthetic.py`, composing `trnrand` +
  `trnsparse` + `trnsolver` + `trnblas` + `trntensor` in one pipeline.
- **NVIDIA CUDA sample ports** under `examples/nvidia_samples/`: direct ports
  of `cuFFT/2d_r2c_c2r`, `cuda-samples/batchCUBLAS`, `MC_EstimatePi{P,Q}`,
  `cuSOLVER/syevd`, `cuda-samples/conjugateGradient`, and
  `cuTENSOR/contraction` against trnsci APIs, each citing the upstream
  sample by URL.
- **Speech-enhancement workflow** at `examples/speech_enhancement/demo.py` —
  synthetic cIRM pipeline exercising trnfft's STFT, ComplexTensor, and
  complex NN layers.
- **Unified documentation site** at https://trnsci.dev/ — landing,
  `why.md` essay, `cuda_rosetta.md` (CUDA → trnsci mapping table with
  per-library deep dives), `trainium_positioning.md` (placing Trainium
  between NVIDIA SMs and Google TPUs), and workflow pages for speech
  enhancement and quantum chemistry. Sub-project docs are composed under
  `trnsci.dev/<name>/` via `mkdocs-monorepo-plugin`.
- **Canonical contribution norms**: `CONTRIBUTING.md` and
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) mirrored across all
  seven repos.
- **CI workflows** (`test.yml`, `docs.yml`, `publish.yml`) on Node.js 24
  (actions/checkout@v6, actions/setup-python@v6). `docs.yml` clones all
  six sub-projects at build time and publishes the combined site; daily
  cron + repository_dispatch keeps the site fresh.

### Notes

- All sub-projects pinned to the same neuronxcc floor (`>=2.24`) and the
  same torch-neuronx floor (`>=2.9`).
- Sub-projects remain independent repos with independent releases. The
  umbrella tracks minimum sub-project versions via the extras above.
- PyTorch fallback works end-to-end for every sub-project on any hardware.
  NKI kernels are scaffolded; on-hardware validation on trn1 / trn2 is
  the next milestone across the suite.

[0.1.0]: https://github.com/trnsci/trnsci/releases/tag/v0.1.0
