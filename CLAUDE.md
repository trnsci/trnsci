# trnsci

Umbrella for the trn-* scientific computing suite for AWS Trainium via NKI.

## Layout

```
trnsci/
├── trnfft/       # cuFFT-equivalent: FFT, complex tensors
├── trnblas/      # cuBLAS-equivalent: BLAS Level 1-3
├── trnrand/      # cuRAND-equivalent: PRNG + QMC
├── trnsolve/     # cuSOLVER-equivalent: factorizations, eigh (package name: trnsolver)
├── trnsparse/    # cuSPARSE-equivalent: sparse formats + SpMM
├── trntensor/    # cuTENSOR-equivalent: einsum + decompositions
├── examples/     # Cross-project integration examples
├── tests/        # Integration tests
├── docs/         # Umbrella mkdocs site (links into per-project docs)
└── pyproject.toml  # Meta-package (trnsci[all] → all six)
```

Each sub-project has its own `CLAUDE.md`, `pyproject.toml`, `tests/`, and `.git/`. The umbrella is a coordinating layer — it does not own their code.

## Conventions across the suite

- Apache 2.0, Copyright 2026 Scott Friedman
- Python ≥3.10, torch ≥2.1, numpy ≥1.24
- Optional `[neuron]` extra pins `neuronxcc>=2.24` (NKI 2.24+ calling convention)
- NKI dispatch pattern: each project exposes `set_backend("auto"|"pytorch"|"nki")` via `<pkg>/nki/dispatch.py`
- Tests marked `@pytest.mark.neuron` for on-hardware-only tests
- `pytest -m "not neuron"` is the CPU-only CI path

## Cross-project composition

`examples/df_mp2_integrated.py` is the canonical integration demo — shows how the sub-projects compose for a realistic workload (density-fitted MP2 energy). Treat it as the reference for cross-project API usage.

## Release coordination

Sub-projects version independently. The umbrella `trnsci` package version reflects the meta-package itself, not the sub-projects. When a sub-project releases, bump the corresponding minimum in `pyproject.toml` extras.
