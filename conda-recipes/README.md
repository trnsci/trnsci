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

## Updating these recipes when PyPI versions change

Before submitting, if any sub-project has released a new PyPI version since these recipes were drafted:

```bash
python3 /tmp/fetch_sdists.py   # reprints current PyPI versions + sha256
```

Then update the affected `meta.yaml` files and re-submit.

## Current pinned versions

| Package | Version | sha256 |
|---|---|---|
| trnsci | 0.1.0 | `1c05eb4c8addd4fe230886ec76326a0811ea7597c75cbbdcb8b378be19114722` |
| trnfft | 0.7.0 | `ac4c5732b747d807848e85e82433a1e7df047355a75d87b996a6e612af41f696` |
| trnblas | 0.4.0 | `173424ac24d5807dfc9fcf0744d04a822e0e61a2e9aec5129d370e33a8578a10` |
| trnrand | 0.1.0 | `1b4ec3c37520b514d67cec59c493b93372ede6958d6bc049f4f65ee477936c8d` |
| trnsolver | 0.3.0 | `c641c3e6458d0175f179f9a1a63da8f5a03d49772f6444f786a89cec08b9ba1c` |
| trnsparse | 0.1.1 | `26d58a161efc67a10bbf3f9357aa0c4befd9d22824149e3381003f606a3bd9d5` |
| trntensor | 0.1.1 | `f5ed259d81ea5bdd47daa30ec54abda4410ede5db83017653227c5ec2bbd0314` |
