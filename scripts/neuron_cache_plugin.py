"""pytest plugin used by scripts/warm_neuron_cache.py (trnsci/trnsci#118).

Loaded with ``-p neuron_cache_plugin`` and ``scripts/`` on ``PYTHONPATH``. It
does three things:

1. Turns on ``torch_neuronx.metrics`` and, at session end, writes the
   ``CompilationCache.*`` counters to ``$TRNSCI_WARM_COUNTERS`` (JSON) and
   prints them as a ``WARM_COUNTERS`` line.
2. After each test, lists the NEFF files that appeared in
   ``$TORCH_NEURONX_NEFF_CACHE_DIR`` and appends ``{"key", "source"}`` lines
   to ``$TRNSCI_WARM_SOURCES`` (JSON lines). During ``capture`` that maps
   every NEFF to the test that produced it; during ``verify`` it names the
   tests that compiled, that is, the misses.
3. In compile-only mode (``TORCH_NEURONX_COMPILE_ONLY=1``) device reads
   return placeholder data, so it turns the numpy / torch result checks into
   no-ops and each test keeps dispatching (and compiling) past its first
   comparison. Bare ``assert`` statements are removed by running under
   ``python -O``.

The plugin imports nothing from torch or numpy at import time, so it loads
on any host.
"""

from __future__ import annotations

import json
import os
import re

_NEFF_FILE = re.compile(r"^([0-9a-f]{32})\.(neff|wrapped_neff_hlo)$")

# Result checks that become no-ops in compile-only mode.
RESULT_CHECKS = (
    (
        "numpy.testing",
        (
            "assert_allclose",
            "assert_almost_equal",
            "assert_approx_equal",
            "assert_array_almost_equal",
            "assert_array_almost_equal_nulp",
            "assert_array_equal",
            "assert_array_less",
            "assert_array_max_ulp",
            "assert_equal",
        ),
    ),
    ("torch.testing", ("assert_close",)),
)


def compile_only() -> bool:
    return os.environ.get("TORCH_NEURONX_COMPILE_ONLY") == "1"


def scan_neff_keys(cache_dir: str | os.PathLike | None) -> set[str]:
    """Keys present in a NEFF cache dir (``<k0:2>/<k2:4>/<k>.neff`` layout)."""
    keys: set[str] = set()
    if not cache_dir or not os.path.isdir(cache_dir):
        return keys
    for a in os.scandir(cache_dir):
        if not (a.is_dir() and len(a.name) == 2):
            continue
        for b in os.scandir(a.path):
            if not (b.is_dir() and len(b.name) == 2):
                continue
            for f in os.scandir(b.path):
                m = _NEFF_FILE.match(f.name)
                if m and m.group(1).startswith(a.name + b.name):
                    keys.add(m.group(1))
    return keys


def _noop(*_args, **_kwargs):
    return None


def neutralise_result_checks() -> list[tuple[object, str, object]]:
    """Replace the functions in RESULT_CHECKS with no-ops.

    Returns ``(module, name, original)`` triples so a caller can restore them.
    Modules that aren't installed are skipped.
    """
    import importlib

    patched = []
    for modname, names in RESULT_CHECKS:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name in names:
            if hasattr(mod, name):
                patched.append((mod, name, getattr(mod, name)))
                setattr(mod, name, _noop)
    return patched


def _counters() -> dict[str, int]:
    try:
        import torch_neuronx.metrics as tnm
    except Exception:
        return {}
    out = {}
    for name in tnm.get_counter_names():
        if name.startswith("CompilationCache"):
            out[name] = int(tnm.get_counter_value(name))
    return out


def _sync() -> None:
    try:
        import torch

        neuron = getattr(torch, "neuron", None)
        if neuron is not None and hasattr(neuron, "synchronize"):
            neuron.synchronize()
    except Exception:
        pass


class _State:
    seen: set[str] = set()


def pytest_configure(config):
    try:
        import torch_neuronx.metrics as tnm

        tnm.set_enabled(True)
    except Exception as e:
        print(f"neuron_cache_plugin: metrics unavailable ({e})")
    _State.seen = scan_neff_keys(os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR"))


def pytest_sessionstart(session):
    # Runs before collection, so `from numpy.testing import assert_allclose`
    # in a test module binds the no-op.
    if compile_only():
        patched = neutralise_result_checks()
        print(
            f"neuron_cache_plugin: compile-only, {len(patched)} result checks disabled"
        )


def pytest_runtest_logfinish(nodeid, location):
    sources = os.environ.get("TRNSCI_WARM_SOURCES")
    if not sources:
        return
    _sync()
    now = scan_neff_keys(os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR"))
    new = sorted(now - _State.seen)
    _State.seen = now
    if not new:
        return
    label = os.environ.get("TRNSCI_WARM_UNIT", "")
    with open(sources, "a", encoding="utf-8") as fh:
        for key in new:
            rec = {"key": key, "source": nodeid}
            if label:
                rec["project"] = label
            fh.write(json.dumps(rec, sort_keys=True) + "\n")


def pytest_unconfigure(config):
    vals = _counters()
    print("\nWARM_COUNTERS " + json.dumps(vals, sort_keys=True))
    path = os.environ.get("TRNSCI_WARM_COUNTERS")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(vals, fh, sort_keys=True)
