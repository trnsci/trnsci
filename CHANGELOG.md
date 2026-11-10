# Changelog

All notable changes to the trnsci umbrella package.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] — 2026-11-10

`pip install "trnsci[all]==0.2.2"` requires the two silent-correctness patch
releases validated on Trn2
([#35](https://github.com/trnsci/trnsci/issues/35#issuecomment-6761384230)):
trnsparse 0.16.1 and trntensor 0.25.2, both from the int32/int64 device audit
([#135](https://github.com/trnsci/trnsci/issues/135)).

### Changed

- Extras minimums: `trnsparse>=0.16.1` (device `from_dense` returned wrong
  indices and values above 2^24 elements) and `trntensor>=0.25.2` (integer
  einsum on the device rounded above 2^24, and int32/int64 `out=` copies of
  transposed results were rounded). The others are unchanged. The trnsci
  conda recipe's `run` minimums follow, and the trnsparse and trntensor
  recipes pin the new releases (sdist sha256 from PyPI). The trnsci recipe
  itself moves to 0.2.2 after the sdist is published.
- docs: validation-status and roadmap rows for trnsparse 0.16.1 and trntensor
  0.25.2.
- `conda-recipes/trnsci` pins trnsci 0.2.1 (sdist sha256 from PyPI), as
  the 0.2.0 recipe followed its release.

## [0.2.1] — 2026-11-10

`pip install "trnsci[all]==0.2.1"` requires the window-6 releases validated
on Trn2 ([#35](https://github.com/trnsci/trnsci/issues/35)): trnblas 0.13.0
(`gemm_fp64`), trnsolver 0.23.0 and trnsparse 0.16.0.

### Added

- `scripts/check_docs_drift.py` and the weekly `docs-drift.yml` workflow
  ([#151](https://github.com/trnsci/trnsci/issues/151)). The script reports
  `[[described:]]` site rows older than PyPI, extras minimums below a
  library's latest minor, a published meta-package that trails the repo, and
  `[torchneuron]` extras that differ from the libraries'. On drift the
  workflow opens or updates one `docs-stale` issue; it doesn't touch the docs
  build.
- The drift check also compares each `conda-recipes/*/recipe.yaml` with
  PyPI: version, sdist `sha256` and `run` requirements, the `trnsci`
  recipe's `run` minimums with `[all]` in `pyproject.toml`, and any
  `python_min` override with conda-forge's global minimum
  ([#159](https://github.com/trnsci/trnsci/issues/159)).

### Changed

- Extras minimums: `trnblas>=0.13` (0.13.0 adds `gemm_fp64`, an fp64-class
  Ozaki GEMM on the Tensor Engine) and `trnsparse>=0.16` (0.16.0 fixes two
  silent wrong results: complex V in `kernel_matmul`, and fused BSR CG at
  65,536 or more blocks), and `trnsolver>=0.23` (0.23.0: LNC-invariant
  `cg` / `gmres` / `cg_block` reductions; `solve_spd` reads `B.shape ==
  A.shape[:-1]` as a batch of vectors, as `torch.linalg.solve` does). The
  others are unchanged. The trnsci conda
  recipe's `run` minimums follow, and the trnblas, trnsolver and trnsparse
  recipes pin the new releases.
- The `trnsci` conda recipe depends on all six libraries (trnblas >=0.12.1,
  trnfft >=0.29.0, trnrand >=0.14.0, trnsolver >=0.22.1, trnsparse >=0.15.0,
  trntensor >=0.25.1). Conda has no extras, so `conda install trnsci` now
  installs the full suite, like `pip install trnsci[all]`
  ([#159](https://github.com/trnsci/trnsci/issues/159)).
- The conda recipes use the v1 `recipe.yaml` format, which
  conda-forge/staged-recipes requires for new recipes, and conda-forge's
  global `python_min` instead of a 3.10 override
  ([#159](https://github.com/trnsci/trnsci/issues/159)).

### Fixed

- `conda-recipes/` pinned trnsci 0.1.0, trnblas 0.4.0 and similarly old
  releases of the other five. They now pin trnsci 0.2.0, trnblas 0.12.1,
  trnfft 0.29.0, trnrand 0.14.0, trnsolver 0.22.1, trnsparse 0.15.0 and
  trntensor 0.25.1, with `setuptools >=77` in `host` (the SPDX `license`
  string needs it) and no `import trnsci` test, since the meta-package has
  no module ([#159](https://github.com/trnsci/trnsci/issues/159)).

## [0.2.0] — 2026-11-08

First meta-package release since 0.1.0. `pip install "trnsci[all]==0.2.0"`
now requires the releases validated on Trn2 (#35, #152).

### Changed

- Extras minimums raised to the current releases: `trnfft>=0.29`,
  `trnblas>=0.12.1`, `trnrand>=0.14`, `trnsolver>=0.22.1`, `trnsparse>=0.15`,
  `trntensor>=0.25.1` (0.1.0 required `trnfft>=0.7`, `trnblas>=0.3` and
  `>=0.1` for the rest). An existing environment upgrading to 0.2.0 now moves
  past the releases with known silent wrong results.
- Installation page and README: dropped the "PyPI meta-package is still
  0.1.0" caveat; document `trnsci[all,torchneuron]` for Trn2/Trn3.

### Fixed

- `examples/quantum_chemistry/df_mp2_synthetic.py`: the pair-energy stage
  computed only the direct term `Σ T²/Δ` (about 30% off the closed-shell
  DF-MP2 energy on the demo system). It now calls `trntensor.mp2_energy`,
  `Σ T(2T − Tᵀ)/Δ`, which matches a float64 reference to ~1e-7. This also
  removes the per-(i, j) Python loop, which synced the device once per pair.
- The metric stage built its identity on the host; it now follows `J`'s device.

### Added

- `[torchneuron]` extra for the Trn2/Trn3 native PyTorch backend
  (`torch>=2.14`, `torch-neuronx>=2.14`, `nki>=0.6`), matching each library's
  own `[torchneuron]` extra. `[neuron]` stays the Trn1/Inf2 PyTorch/XLA
  lineage.
- `scripts/warm_neuron_cache.py`: compiles Neuron NEFF caches off-box
  ([#118](https://github.com/trnsci/trnsci/issues/118)). `manifest` records
  the box toolchain, and `check` refuses on any mismatch. `capture` runs
  `pytest -m neuron` in compile-only mode on a CPU host, with parallel
  workers. `recompile` replays recorded `module.mlir` files. `pack` /
  `unpack` produce a tarball plus a JSON index, and `verify` reports the
  persistent-hit rate on the box. The pytest plugin is
  `scripts/neuron_cache_plugin.py`, and the usage notes and limits are in
  `docs/developing_kernels.md`. The `[dev]` extra now includes `xxhash`.
- `--device neuron` for the DF-MP2 example. Inputs are built on the host and
  copied to the device once.
- `examples/quantum_chemistry/df_mp2_profile.py`: an end-to-end profile across
  a size ladder. It reports per-stage time, the route each library's `auto`
  dispatch picked, and error against float64, in `cpu` / `host-auto` /
  `neuron` modes.
- Integration test pinning the example energy to the float64 closed-shell
  reference.

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

[0.2.0]: https://github.com/trnsci/trnsci/releases/tag/v0.2.0
[0.1.0]: https://github.com/trnsci/trnsci/releases/tag/v0.1.0
