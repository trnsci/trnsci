# trnsci conda-forge submission

This directory stages the `conda-forge/staged-recipes` submission for all seven `trnsci` packages. It is not part of the runtime trnsci release — it's a workspace for the conda-forge PR.

## Layout

```
conda-recipes/
├── README.md              (this file — workspace notes)
├── PR_DESCRIPTION.md      (paste into the conda-forge staged-recipes PR body)
├── trnsci/meta.yaml
├── trnfft/meta.yaml
├── trnblas/meta.yaml
├── trnrand/meta.yaml
├── trnsolver/meta.yaml
├── trnsparse/meta.yaml
└── trntensor/meta.yaml
```

## Submission workflow

1. **Fork** https://github.com/conda-forge/staged-recipes on GitHub.

2. **Clone the fork and check out a branch:**
   ```bash
   git clone git@github.com:scttfrdmn/staged-recipes.git
   cd staged-recipes
   git checkout -b trnsci-suite
   ```

3. **Copy the recipes** into `recipes/`:
   ```bash
   for pkg in trnsci trnfft trnblas trnrand trnsolver trnsparse trntensor; do
     cp -r ~/src/trnsci/conda-recipes/$pkg recipes/$pkg
   done
   ```

4. **Lint locally** (optional but recommended):
   ```bash
   pip install conda-smithy
   conda-smithy recipe-lint recipes/trnsci recipes/trnfft recipes/trnblas \
     recipes/trnrand recipes/trnsolver recipes/trnsparse recipes/trntensor
   ```

5. **Commit, push, open PR:**
   ```bash
   git add recipes/trn*
   git commit -m "Add trnsci scientific-computing suite (7 packages)"
   git push origin trnsci-suite
   gh pr create --repo conda-forge/staged-recipes \
     --title "Add trnsci scientific-computing suite (7 packages)" \
     --body-file ~/src/trnsci/conda-recipes/PR_DESCRIPTION.md
   ```

## After the PR merges

conda-forge auto-creates one **feedstock repository per package** under `conda-forge/<name>-feedstock`. You (as maintainer) get push access.

Future releases are handled by `regro-cf-autotick-bot`: it watches PyPI, detects new versions, and opens a PR to each feedstock bumping the version + sha256. You merge, the new build ships to the conda-forge channel within hours. Zero ongoing manual work per release.

## Status

Not published: there is no `conda-forge/<name>-feedstock` and nothing in this
repo builds these recipes. The submission is
[conda-forge/staged-recipes#32941](https://github.com/conda-forge/staged-recipes/pull/32941);
its recipes are copies of these, so update it from here before review.

## Keeping the recipes current

Each `meta.yaml` pins one PyPI release: `version`, the sdist `sha256`, the
`run` requirements (the library's `dependencies`, with `torch` as
`pytorch`) and `python_min` (its `requires-python`). The weekly drift check,
`scripts/check_docs_drift.py`, compares all four with PyPI and opens a
`docs-stale` issue when a recipe falls behind. To update one:

```bash
curl -s https://pypi.org/pypi/<name>/json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["info"]["version"], [u["digests"]["sha256"] for u in d["urls"] if u["packagetype"] == "sdist"])'
```

`setuptools >=77` in `host` is deliberate: the libraries' `pyproject.toml`
uses an SPDX `license = "Apache-2.0"` string, which setuptools 68-76 rejects,
even though their `[build-system]` still says `setuptools>=68.0`.
