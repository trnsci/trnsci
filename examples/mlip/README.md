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
| `mace_trnsci_step.py` | one MACE energy + forces step (`--batch B`: B structures in one step) with all three blocks swapped to the trnsci kernels inside MACE's own forward; each structure checked against the unmodified model on the CPU, timed, and (`--shares`) split into per-block shares |
| `profile_device_step.py` | the aten remainder of the trnsci step by MACE module on the device: each module call replayed in isolation (forward, forward + the input gradients the step needed, aten op counts), plus a whole-step op count; `--fuse` for the prototypes below, `--batch` and the bucket flags (trnsci/trnsci#178) |
| `batching.py` | B structures as one block-diagonal graph (MACE's own collation: concatenated atoms and edges, offset indices), padded to a shape bucket for (total atoms, total edges); the padding atoms are a separate discarded structure and the padding edges a `[E]` mask for `trnsparse.segment_plan(..., padding=mask)`; per-structure split and `isolate` (one structure alone at the batch's shapes) |
| `cpu_ref.py` | the CPU reference (unmodified MACE, each structure alone and unpadded, timed as `cpu_ref_s`) and the comparison against it; a missing, empty or non-finite reference raises `MissingReferenceError` instead of reporting an error of 0 |
| `accept.py` | model-level acceptance: energy invariance under translation and rotation, force (and stress) equivariance, finite-difference forces along random directions and, when the model gives a stress, finite-difference stress under random strains; baseline CPU MACE against a target on `--device`, with a paired outcome table. Every tolerance is a flag |
| `fusions.py` | fusion prototypes: `edge` (the whole per-edge chain, i.e. spherical harmonics, radial basis, ZBL per edge and both layers' radial and density MLPs, in one `torch.compile(backend="neuron")` region, with the species constants built once per neighbour list) and `dense` (e3nn `Linear`s and the one-hot `skip_tp` as GEMMs against probed dense matrices) |
| `trnsci_fused.py` | the maintained fused ops for the aten remainder (trnsci/trnsci#178): `sh` (`trntensor.spherical_harmonics`), `edge` (`trntensor.edge_chain`: radial basis + both layers' radial and density MLPs), `zbl` (the ZBL pair energy per edge, exemplar-side, with an analytic backward), `linear` and `skip` (`trntensor.equivariant_linear` for every e3nn `Linear` and the one-hot `skip_tp`); `mace_trnsci_step.py --fuse` installs them |
| `check_fused.py` | each fused op against MACE at the six exemplar captures: the output and the input gradient for the real force-step cotangent, against MACE fp32 and fp64; `--trntensor-fixture` writes trntensor's CI fixture |
| `symcon_bwd_probe.py` | per-op device timing of trntensor's PyTorch-path `symmetric_contraction` forward and backward (trnsci/trntensor#140) |
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
# the remainder by module, then the fusion prototypes and padded edge buckets
python profile_device_step.py --device neuron --json prof216.json
python profile_device_step.py --device neuron --no-profile --fuse edge,dense --seeds 0,1
# batches: B structures per step, one bucketed block-diagonal graph, split per structure
python mace_trnsci_step.py --structure tic001_slab_216 --batch 8 --backend pytorch   # CPU: split checked bitwise
python mace_trnsci_step.py --structure tic001_slab_216 --batch 8 --device neuron
python profile_device_step.py --device neuron --no-profile --fuse edge,dense --batch 8 --seeds 0,1
# model-level acceptance: CPU MACE (baseline) against the target
python accept.py                                                # CPU MACE vs CPU MACE
python accept.py --target trnsci --device neuron --backend nki --batch 8
# the fused library ops (#178 item 1): accuracy at the captures, then the step
python check_fused.py                                              # CPU, PyTorch paths
python check_fused.py --device neuron --backend nki                # the NKI kernels
python mace_trnsci_step.py --mode trnsci --device neuron --fuse all --region-backend neuron \
    --count-ops --json fused.json
python mace_trnsci_step.py --mode trnsci --device neuron --fuse all --backend nki \
    --region-backend neuron --batch 8 --json fused_b8.json
```

**Batching.** `--batch B` takes the `--structure` list cycled with seeds
`seed .. seed+B-1`. Atoms and edges are padded to buckets (multiples of
`2^(floor(log2 n) - 4)`, at most 6.25% padding; `--bucket-shift`, or a fixed
`--edge-multiple`), so a second geometry or batch of similar size reuses
every compiled shape: seeds 0-7 and 8-15 of the 216-atom slab both land in
(1792 atoms, 86016 edges). The padding atoms belong to an extra structure
whose energy is dropped; the padding edges are self-edges on them, past the
cutoff, and the plans mask them out of every segment sum. On the CPU,
`--split-check` runs each structure alone at the batch's own shapes
(`batching.isolate`) and requires the trnsci-path split to match bitwise.
Unpadded single-structure runs are not bitwise comparable on the CPU (the
SIMD tails of the per-edge elementwise kernels and the K = 8, N = 1 density
GEMV depend on the row count and offset); those are compared against
`cpu_ref.py` to fp32 rounding instead.

Full captures are about 20 MiB, 170 MiB and 790 MiB per structure and are
regenerated with `capture.py`, not committed. The reference reproduces
MACE's own tensors to ≤ 3e-7 relative (fp32) for every block, output and
gradient on all six captures. `tests/test_mlip_blocks.py` runs the fixture
and the edge cases on CPU without MACE; its live re-capture test is skipped
when `mace-torch` is not installed. `tests/test_mlip_batching.py` and
`tests/test_mlip_accept.py` cover the buckets, the reference check, the
acceptance suite (on a torch pair potential and broken variants of it) and,
with mace-torch, batched steps on the small structures.

## The two contracts

**A (trntensor), equivariant contractions**

- A1 `conv_tp(x, y, w, spec)`: per edge, per channel u, `out[e,u,k] = α_p · w[e,p,u] · Σ_ij C^{l1 l2 l3}_ijk · x[e,u,i] · y[e,j]`, with e3nn real Wigner-3j constants. Needs ∂/∂x (layer 1 only), ∂/∂y and ∂/∂w.
- A2 `symmetric_contraction(x, species, spec)`: per node and channel, `out[n,c,m] = Σ_ν Σ_k W_ν[z_n,k,c] · Σ_a U_ν[m,a_1..a_ν,k] · Π_j x[n,c,a_j]`, ν ≤ 3, with fixed U tensors that are 0.4–1.6% non-zero. Needs ∂/∂x.

**B (trnsparse), graph gather / segment sum**

- B1 `gather(h, idx)`: `out[e] = h[idx[e]]`. Its backward is B2 over idx.
- B2 `segment_sum(m, idx, N)`: `out[i] = Σ_{e: idx[e]=i} m[e]`. Its backward is B1. Indices are int64. Senders are sorted, receivers are not.

Shapes, time shares and the reuse / adapt / build questions are in trnsci/trnsci#170; the library work is trnsci/trnsparse#139 (B) and trnsci/trntensor#138 (A).
