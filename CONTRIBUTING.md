# Contributing to trnsci

Thanks for your interest in contributing. This document covers the whole suite — the same guidelines apply to the umbrella and to each of the six sub-projects (`trnfft`, `trnblas`, `trnrand`, `trnsolver`, `trnsparse`, `trntensor`).

## Where does my change belong?

- **Scoped to a single library** — bug fix, new feature, added kernel in one sub-project → open a PR in that sub-project's repo.
- **Cross-cutting** — integration examples, umbrella docs, CUDA-rosetta content, coordinated version bumps, shared tooling → open a PR against [trnsci/trnsci](https://github.com/trnsci/trnsci).

If you aren't sure, open the PR where it's easiest and we'll move it.

## Development setup

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
# Clone sibling sub-projects if you don't already have them:
for p in trnfft trnblas trnrand trnsolver trnsparse trntensor; do
  [ -d "$p" ] || git clone "git@github.com:trnsci/$p.git"
done
make install-dev
make test-all
```

A single sub-project in isolation:

```bash
cd trnfft   # or any other
pip install -e ".[dev]"
pytest tests/ -v -m "not neuron"
```

## Pre-commit hooks

The umbrella repo uses [pre-commit](https://pre-commit.com) for lint, formatting, and hygiene. Install once after cloning:

```bash
pip install pre-commit         # or part of make install-dev
pre-commit install             # wires the git hook
pre-commit run --all-files     # sanity check before your first commit
```

Hooks run automatically on every commit. CI enforces them via `.github/workflows/pre-commit.yml` and [pre-commit.ci](https://pre-commit.ci/) auto-updates the pinned versions weekly.

Sub-project repos (trnfft, trnblas, etc.) each maintain their own `.pre-commit-config.yaml`.

## Tests

- `pytest tests/ -v -m "not neuron"` — the CPU-only suite. Must pass on every PR.
- `pytest tests/ -v -m neuron` — on-hardware suite, run from a machine with AWS credentials via `scripts/run_neuron_tests.sh`. Not part of standard CI.
- Coverage is collected on the umbrella CI and uploaded to [Codecov](https://codecov.io/gh/trnsci/trnsci).

## Conventions

- **License header** — new files get the Apache 2.0 notice and `Copyright 2026 Scott Friedman` where a notice is customary (long source files, not one-line configs).
- **Python** — ≥ 3.10; `torch` ≥ 2.1; `numpy` ≥ 1.24.
- **No emoji** in source, commits, or docs unless the feature is literally about emoji.
- **Docstrings** only where they help a reader who doesn't have the context you had when writing. Don't restate the signature.
- **No throwaway files** in the repo root — work-in-progress notes belong in PR descriptions or `docs/`.

## Commit style

Conventional-ish prefixes: `fix:`, `feat:`, `docs:`, `test:`, `bench:`, `chore:`, `refactor:`. Keep the subject line ≤ 72 characters. Longer context goes in the body.

## Pull requests

Before opening a PR:

- [ ] `pytest tests/ -v -m "not neuron"` passes
- [ ] If you changed public API or behaviour, `CHANGELOG.md` has an `[Unreleased]` entry
- [ ] If you added a public symbol, the project's `docs/api.md` lists it
- [ ] PR description says *what* changed and *why*, not just *what*

## Issues

Label each issue one of: `bug`, `feature`, `question`, `docs`, `infra`. Bugs include repro steps and expected-vs-actual output. Features describe the use case.

## Code of conduct

By participating, you agree to abide by the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). Report concerns to the contact listed there.

## License

Contributions are Apache 2.0, per the repository's [LICENSE](LICENSE). You attest you have the right to submit the code.
