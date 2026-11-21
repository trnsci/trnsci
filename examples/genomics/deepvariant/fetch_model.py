"""Download the public DeepVariant 1.9.0 WGS model and verify checksums.

Source: the open-source DeepVariant release bucket (BSD-3-Clause), served
over HTTPS from ``storage.googleapis.com/deepvariant``. The same files ship
in the ``google/deepvariant:1.9.0`` container under ``/opt/models/wgs``.
Weights are downloaded at run time and never committed.

    python fetch_model.py --dest ~/.xtmp/dv114/model
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

BASE = "https://storage.googleapis.com/deepvariant/models/DeepVariant/1.9.0"

# path relative to BASE -> sha256 (recorded 2026-11-02)
MANIFEST = {
    "savedmodels/deepvariant.wgs.savedmodel/saved_model.pb": (
        "7ff713f03e367c9c6b363d9b6e047c22f8d85756b2b5aec2d91850e44d42de34"
    ),
    "savedmodels/deepvariant.wgs.savedmodel/fingerprint.pb": (
        "72d8b0809446de01d4e9e446fee9f86127441eaf17e90746762eba7f87650417"
    ),
    "savedmodels/deepvariant.wgs.savedmodel/example_info.json": (
        "fe39ee9145855d17229248b26eb5568fc8b17f9650d02161c6a155ee80f1f02e"
    ),
    "savedmodels/deepvariant.wgs.savedmodel/variables/variables.data-00000-of-00001": (
        "29f2862b01507cc780162384f53915e498eb6874b11ba2ff8e8caa9f72291e24"
    ),
    "savedmodels/deepvariant.wgs.savedmodel/variables/variables.index": (
        "c27e79f82290fc0712c51179032cc50ebd40420184e0a923abb680e8bca50d62"
    ),
}

# DeepVariant's own unit-test examples (r1.9 branch), real 100x221x7 pileups.
GOLDEN_BASE = (
    "https://raw.githubusercontent.com/google/deepvariant/r1.9/deepvariant/testdata"
)
GOLDEN = {
    "golden.calling_examples.tfrecord.gz": (
        "770d3fe5034fb89209a7bb872a67a5ab3b7fbd28553595bdd548c9cb500c6aab"
    ),
    "golden.calling_examples.tfrecord.gz.example_info.json": (
        "fe39ee9145855d17229248b26eb5568fc8b17f9650d02161c6a155ee80f1f02e"
    ),
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(base: str, manifest: dict[str, str], dest: str) -> None:
    for rel, want in manifest.items():
        path = os.path.join(dest, rel)
        if not (os.path.exists(path) and sha256(path) == want):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"GET {base}/{rel}", file=sys.stderr)
            urllib.request.urlretrieve(f"{base}/{rel}", path + ".part")
            os.replace(path + ".part", path)
        got = sha256(path)
        if got != want:
            raise SystemExit(f"checksum mismatch for {rel}: {got} != {want}")
        print(f"ok  {got}  {rel}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", required=True, help="directory to place the model in")
    ap.add_argument(
        "--golden", help="also fetch DeepVariant's golden calling examples here"
    )
    args = ap.parse_args()
    fetch(BASE, MANIFEST, args.dest)
    if args.golden:
        fetch(GOLDEN_BASE, GOLDEN, args.golden)


if __name__ == "__main__":
    main()
