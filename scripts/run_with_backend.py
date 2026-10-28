#!/usr/bin/env python
"""Run a script with every installed trnsci library on one backend.

    python scripts/run_with_backend.py nki|pytorch <script.py> [args...]

Used to compare cross-library examples (e.g. examples/quantum_chemistry/
df_mp2_synthetic.py) on the NKI path against the PyTorch reference path.
"""

import importlib
import runpy
import sys

LIBS = ("trnfft", "trnblas", "trnrand", "trnsolver", "trnsparse", "trntensor")

if __name__ == "__main__":
    backend, script, *args = sys.argv[1:]
    for name in LIBS:
        try:
            lib = importlib.import_module(name)
        except ImportError:
            continue
        lib.set_backend(backend)
        print(f"[run_with_backend] {name}.set_backend({backend!r})", flush=True)
    sys.argv = [script, *args]
    runpy.run_path(script, run_name="__main__")
