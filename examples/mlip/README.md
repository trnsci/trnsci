# MACE-style MLIP evaluator: block contracts and reference data (exemplar, trnsci/trnsci#170)

Conservative machine-learned interatomic potentials compute forces as
F = −∂E/∂positions, so every step needs the forward **and** the input
gradient of two kinds of kernel: equivariant contractions and graph
gather / scatter. This directory traces the public MACE-MP-0b2 "medium"
model on CPU, defines the two block contracts that trntensor and trnsparse
will implement, and ships captured tensors plus a checker for candidate
kernels. `check.py` and `mace_trnsci_step.py` also run the trnsci candidates on a Neuron device.

## Sources (public only)

| | |
|---|---|
| Model code | MACE, github.com/ACEsuit/mace (MIT), `mace-torch==0.3.15`; e3nn 0.4.4; ASE 3.29 |
| Checkpoint | MACE-MP-0b2 medium, `https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/mace-medium-density-agnesi-stress.model` |
| sha256 | `a90be07c8aa6623c390fcc4653d3e319c4a356f8b4a53d323f96b2012d375caf` (79 462 798 bytes) |
| Architecture | 2 interactions, 128 channels, hidden 128x0e+128x1o, spherical harmonics l ≤ 3, correlation 3, r_max 5 Å, 89 elements, ZBL pair term |

The checkpoint is downloaded at run time (`mace_setup.py`) and is never committed.

## Files

| file | what it does |
|---|---|
| `mace_setup.py` | checkpoint fetch + sha256, the three test structures, ASE atoms → MACE batch, energy + forces |
| `profile_mace.py` | PyTorch-profiler trace of energy + forces; forward ops attributed by module range, backward nodes by autograd sequence number |
| `blocks.py` | **the reference**: plain-PyTorch `conv_tp`, `symmetric_contraction`, `gather`, `segment_sum` (no e3nn / MACE import) |
| `capture.py` | captures each block's inputs, outputs, cotangents and input gradients from one energy + forces run; writes full captures and the CI fixture |
| `check.py` | checks a candidate module (or the reference) against captures, plus synthetic edge cases |
| `trnsci_candidates.py` | candidate module for `check.py`: A1/A2 from `trntensor`, B1/B2 from `trnsparse` (backend `nki` unless `MLIP_TRNSCI_BACKEND` is set) |
| `mace_trnsci_step.py` | one MACE energy + forces step with all three blocks swapped to the trnsci kernels inside MACE's own forward; checked against the unmodified model on the CPU, timed, and (`--shares`) split into per-block shares |
| `fixtures/mace_mp0b2_tiny.pt` | committed slice of the 16-atom capture (335 KiB): 8 channels per irrep, 24 edges for A1, all nodes and indices |

## Structures

| name | atoms | edges | neighbours / atom (min / mean / max) |
|---|---|---|---|
| `molecule_box_16` (2 × CH3COOH, 12 Å periodic box) | 16 | 150 | 8 / 9.4 / 13 |
| `cu111_slab_64` (Cu(111) 4×4×4, 15 Å vacuum) | 64 | 2156 | 27 / 33.7 / 42 |
| `tic001_slab_216` (TiC(001) 3×3 cells × 6 layers) | 216 | 10392 | 36 / 48.1 / 56 |

All are rattled by 0.05 Å (`seed`); `--seed 1` gives a second geometry with
different connectivity (148, 2140 and 10378 edges).

## Run

```bash
python -m venv ~/.xtmp/mlip/venv && . ~/.xtmp/mlip/venv/bin/activate
pip install "mace-torch==0.3.15" ase
python mace_setup.py --dest ~/.xtmp/mlip/ckpt          # or set MLIP_CKPT_DIR
python profile_mace.py --threads 8 --json profile.json
python capture.py --out ~/.xtmp/mlip/captures --fixture fixtures/mace_mp0b2_tiny.pt
python capture.py --out ~/.xtmp/mlip/captures --seed 1
python check.py ~/.xtmp/mlip/captures/*.pt --edge-cases            # reference vs MACE
python check.py ~/.xtmp/mlip/captures/*.pt --candidate my.kernels --device neuron
# the trnsci kernels (trntensor + trnsparse on PYTHONPATH), per block and end to end
python check.py ~/.xtmp/mlip/captures/*.pt --candidate trnsci_candidates --device neuron
python mace_trnsci_step.py --structure tic001_slab_216 --mode mace --device cpu --threads 2
python mace_trnsci_step.py --structure tic001_slab_216 --mode trnsci --device neuron --shares
```

Full captures are about 20 MiB, 170 MiB and 790 MiB per structure and are
regenerated with `capture.py`, not committed. The reference reproduces
MACE's own tensors to ≤ 3e-7 relative (fp32) for every block, output and
gradient on all six captures. `tests/test_mlip_blocks.py` runs the fixture
and the edge cases on CPU without MACE; its live re-capture test is skipped
when `mace-torch` is not installed.

## The two contracts

**A (trntensor), equivariant contractions**

- A1 `conv_tp(x, y, w, spec)`: per edge, per channel u, `out[e,u,k] = α_p · w[e,p,u] · Σ_ij C^{l1 l2 l3}_ijk · x[e,u,i] · y[e,j]`, with e3nn real Wigner-3j constants. Needs ∂/∂x (layer 1 only), ∂/∂y and ∂/∂w.
- A2 `symmetric_contraction(x, species, spec)`: per node and channel, `out[n,c,m] = Σ_ν Σ_k W_ν[z_n,k,c] · Σ_a U_ν[m,a_1..a_ν,k] · Π_j x[n,c,a_j]`, ν ≤ 3, with fixed U tensors that are 0.4–1.6% non-zero. Needs ∂/∂x.

**B (trnsparse), graph gather / segment sum**

- B1 `gather(h, idx)`: `out[e] = h[idx[e]]`. Its backward is B2 over idx.
- B2 `segment_sum(m, idx, N)`: `out[i] = Σ_{e: idx[e]=i} m[e]`. Its backward is B1. Indices are int64. Senders are sorted, receivers are not.

Shapes, time shares and the reuse / adapt / build questions are in trnsci/trnsci#170; the library work is trnsci/trnsparse#139 (B) and trnsci/trntensor#138 (A).
