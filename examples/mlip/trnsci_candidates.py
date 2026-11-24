"""Candidate module for ``check.py``: the trnsci kernels behind the block contracts.

- A1 ``conv_tp`` and A2 ``symmetric_contraction``: ``trntensor`` (#138).
- B1 ``gather`` and B2 ``segment_sum``: ``trnsparse`` (#139).

The backend is ``nki`` unless ``MLIP_TRNSCI_BACKEND`` says otherwise
(``pytorch`` runs the libraries' own PyTorch paths, useful on a CPU host).
Set ``TRNTENSOR_REQUIRE_NKI=1`` and ``TRNSPARSE_REQUIRE_NKI=1`` so a call
that would fall back to torch raises instead.

    python check.py captures/*.pt --candidate trnsci_candidates --device neuron
"""

from __future__ import annotations

import os

import trnsparse
import trntensor

BACKEND = os.environ.get("MLIP_TRNSCI_BACKEND", "nki")
trnsparse.set_backend(BACKEND)
trntensor.set_backend(BACKEND)

# The exemplar passes the capture's spec dicts; convert each once (keyed by
# identity, holding a reference so the id stays valid).
_SPECS: dict = {}


def _spec(spec, kind):
    if not isinstance(spec, dict):
        return spec
    hit = _SPECS.get(id(spec))
    if hit is None or hit[0] is not spec:
        hit = (spec, kind.from_dict(spec))
        _SPECS[id(spec)] = hit
    return hit[1]


def conv_tp(x, y, w, spec):
    return trntensor.conv_tp(x, y, w, _spec(spec, trntensor.ConvTPSpec))


def symmetric_contraction(x, species, spec):
    return trntensor.symmetric_contraction(
        x, species, _spec(spec, trntensor.SymmetricContractionSpec)
    )


def gather(h, idx):
    return trnsparse.gather(h, idx)


def segment_sum(m, idx, num_segments):
    return trnsparse.segment_sum(m, idx, num_segments)
