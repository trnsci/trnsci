"""Public MACE-MP-0b2 medium checkpoint, test structures and graph batches.

Everything here uses public sources only:

- MACE (github.com/ACEsuit/mace, MIT), pinned ``mace-torch==0.3.15``;
- the MACE-MP-0b2 "medium" foundation model released by the MACE authors at
  github.com/ACEsuit/mace-mp (release ``mace_mp_0b2``);
- e3nn and ASE.

The checkpoint is downloaded at run time and never committed.

    python mace_setup.py --dest ~/.xtmp/mlip/ckpt      # fetch + verify sha256
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

MACE_VERSION = "0.3.15"
CHECKPOINT_URL = (
    "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b2/"
    "mace-medium-density-agnesi-stress.model"
)
CHECKPOINT_NAME = "mace-medium-density-agnesi-stress.model"
# recorded 2026-11-22, 79 462 798 bytes
CHECKPOINT_SHA256 = "a90be07c8aa6623c390fcc4653d3e319c4a356f8b4a53d323f96b2012d375caf"

# Structure ladder: small / medium / large, one chemistry each.
STRUCTURES = ("molecule_box_16", "cu111_slab_64", "tic001_slab_216")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_checkpoint(dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, CHECKPOINT_NAME)
    if not os.path.exists(path):
        print(f"downloading {CHECKPOINT_URL}", file=sys.stderr)
        urllib.request.urlretrieve(CHECKPOINT_URL, path)
    got = sha256(path)
    if got != CHECKPOINT_SHA256:
        raise RuntimeError(f"{path}: sha256 {got} != {CHECKPOINT_SHA256}")
    return path


def default_checkpoint() -> str:
    root = os.environ.get("MLIP_CKPT_DIR")
    if root is None:
        xtmp = os.path.expanduser("~/.xtmp")
        root = os.path.join(xtmp if os.path.isdir(xtmp) else "/tmp", "mlip", "ckpt")
    return fetch_checkpoint(root)


def load_model(path: str | None = None, dtype: str = "float32"):
    """Load the MACE-MP-0b2 medium model on CPU in eval mode."""
    import warnings

    import torch

    path = path or default_checkpoint()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(path, map_location="cpu", weights_only=False)
    model = model.to(getattr(torch, dtype)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_structure(name: str, seed: int = 0):
    """Return an ASE Atoms for one rung of the ladder, rattled (0.05 Å) so
    forces are non-zero. ``seed`` changes the rattle only (same atoms,
    slightly different connectivity)."""
    from ase import Atoms
    from ase.build import bulk, fcc111, molecule

    if name == "molecule_box_16":
        # Acetic-acid dimer (2 x CH3COOH, 16 atoms) in a 12 Å periodic box.
        a = molecule("CH3COOH")
        b = molecule("CH3COOH")
        b.rotate(180, "z")
        b.translate((0.0, 5.0, 0.4))  # closest intermolecular pair 2.3 Å
        atoms = Atoms(a + b)
        atoms.set_cell([12.0, 12.0, 12.0])
        atoms.center()
        atoms.pbc = True
    elif name == "cu111_slab_64":
        # Cu(111), 4 x 4 in-plane, 4 layers, 15 Å vacuum.
        atoms = fcc111("Cu", size=(4, 4, 4), vacuum=7.5)
        atoms.pbc = True
    elif name == "tic001_slab_216":
        # TiC(001) rocksalt slab: 3 x 3 cubic cells in-plane, 6 layers, 15 Å vacuum.
        atoms = bulk("TiC", "rocksalt", a=4.33, cubic=True).repeat((3, 3, 3))
        atoms.center(vacuum=7.5, axis=2)
        atoms.pbc = True
    elif name.startswith("tic_bulk_"):
        # Scaling rung only (not captured): TiC bulk, n x n x n cubic cells.
        n = int(name.rsplit("_", 1)[1])
        atoms = bulk("TiC", "rocksalt", a=4.33, cubic=True).repeat((n, n, n))
    else:
        raise ValueError(name)
    atoms.rattle(stdev=0.05, seed=seed)
    return atoms


def atoms_to_batch(atoms, model):
    """Same path as ``mace.calculators.MACECalculator._atoms_to_batch``:
    config -> AtomicData (matscipy neighbour list at r_max) -> batch of 1."""
    import torch
    from mace import data as mace_data
    from mace.tools import torch_geometric, utils

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = list(getattr(model, "heads", ["default"]))
    keyspec = mace_data.KeySpecification(info_keys={}, arrays_keys={})
    config = mace_data.config_from_atoms(
        atoms, key_specification=keyspec, head_name=heads[0]
    )
    data = mace_data.AtomicData.from_config(
        config, z_table=z_table, cutoff=float(model.r_max), heads=heads
    )
    loader = torch_geometric.dataloader.DataLoader([data], batch_size=1, shuffle=False)
    batch = next(iter(loader))
    dtype = next(model.parameters()).dtype
    out = batch.to_dict()
    for k, v in out.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = v.to(dtype)
    return out


def energy_forces(model, batch):
    """Energy + forces (forces = -dE/dpositions via autograd)."""
    b = dict(batch)
    b["positions"] = b["positions"].detach().clone()
    out = model(b, compute_force=True, training=False)
    return out["energy"], out["forces"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=None, help="checkpoint directory")
    args = ap.parse_args()
    path = fetch_checkpoint(args.dest) if args.dest else default_checkpoint()
    print(path, CHECKPOINT_SHA256)


if __name__ == "__main__":
    main()
