# Contributing to trnsci

Thanks for your interest in contributing.

## Where does my change belong?

- Bug or feature scoped to a single library → open a PR in that sub-project's repo (`trnsci/trnfft`, `trnsci/trnblas`, etc.)
- Cross-project examples, integration tests, umbrella docs, or coordinated version bumps → this repo

## Development setup

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
# If the sub-projects aren't already sibling directories, clone them too:
for p in trnfft trnblas trnrand trnsolver trnsparse trntensor; do
  [ -d "$p" ] || git clone "git@github.com:trnsci/$p.git"
done
make install-dev
make test-all
```

## Conventions

- Apache 2.0, Copyright 2026 Scott Friedman on all new files
- Python ≥ 3.10, torch ≥ 2.1
- Mark hardware-only tests with `@pytest.mark.neuron`
- `pytest -m "not neuron"` must pass on CPU
- Match each sub-project's layout (`<pkg>/`, `tests/`, `examples/`, `docs/`, `scripts/`, `infra/terraform/`)

## Commit style

Conventional-ish prefixes (`fix:`, `feat:`, `docs:`, `bench:`). Keep subject ≤ 72 chars.

## Questions

Open a discussion or issue in [trnsci/trnsci](https://github.com/trnsci/trnsci).
