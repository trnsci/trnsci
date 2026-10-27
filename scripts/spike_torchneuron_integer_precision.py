#!/usr/bin/env python
"""Spike C — Philox integer exactness on public TorchNeuron (SDK 2.33).

trnrand's Philox 4x32-10 is blocked on hardware by inexact uint32 arithmetic
(aws-neuron-sdk#1308: integer ops routed through the fp32 path, exact only to
2**24). This spike runs trnrand's real ``philox4x32_kernel`` through ``nki_op``
and compares it bit-for-bit against ``philox4x32_reference``. Small integer
probes (round-trip, 32-bit multiply, wrapping add) diagnose a failure.

Each compiler configuration runs in a fresh subprocess with the NEFF cache
disabled, so a flag change really recompiles:

  default   — SDK 2.33 defaults
  int64     — the defaults planned for the next release (2.33 notes):
              --native-int64=enabled --implicit-integer-downcast=none

Requires Trn2/Trn3 + torch 2.14 + torch-neuronx 2.14.* + trnrand installed.

    python scripts/spike_torchneuron_integer_precision.py 2>&1 | tee ~/spike_int.log
"""

import os
import subprocess
import sys

PASSES = {
    "default": "",
    "int64": "--native-int64=enabled --implicit-integer-downcast=none",
}


def orchestrate():
    """Run each pass in a fresh process. SPIKEC_CORES="8,12" runs the passes
    concurrently, one per NeuronCore (NEURON_RT_VISIBLE_CORES), on multi-chip hosts."""
    import tempfile

    cores = [c for c in os.environ.get("SPIKEC_CORES", "").split(",") if c]
    procs = []
    for i, (name, flags) in enumerate(PASSES.items()):
        env = dict(os.environ)
        env["NEURON_CC_FLAGS"] = (env.get("NEURON_CC_FLAGS", "") + " " + flags).strip()
        env["TORCH_NEURONX_NEFF_DISABLE_CACHE"] = "1"
        env["NKI_ENABLE_TRACE_CACHE"] = "0"
        env["TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR"] = tempfile.mkdtemp(
            prefix=f"spikec_{name}_"
        )
        banner = f"\n######## pass '{name}'  NEURON_CC_FLAGS='{env['NEURON_CC_FLAGS']}'"
        cmd = [sys.executable, __file__, "--pass", name]
        if cores:
            env["NEURON_RT_VISIBLE_CORES"] = cores[i]
            out = tempfile.TemporaryFile("w+")
            procs.append(
                (
                    banner + f"  core={cores[i]} ########",
                    out,
                    subprocess.Popen(
                        cmd, env=env, stdout=out, stderr=subprocess.STDOUT, text=True
                    ),
                )
            )
        else:
            print(banner + " ########", flush=True)
            subprocess.run(cmd, env=env, check=False)
    for banner, out, proc in procs:
        proc.wait()
        out.seek(0)
        print(banner, flush=True)
        print(out.read(), flush=True)


def single_pass(name):
    import nki
    import nki.isa as nisa
    import nki.language as nl
    import torch
    import torch_neuronx
    from torch_neuronx import nki_op

    from trnrand.nki.dispatch import philox4x32_kernel, philox4x32_reference

    dev = "neuron"
    results = {}

    @nki.jit
    def _roundtrip_kernel(x):
        t = nl.load(x)
        out = nl.ndarray(x.shape, dtype=nl.int32, buffer=nl.shared_hbm)
        nl.store(out, value=t)
        return out

    # Integer add/multiply via nisa.tensor_tensor: nl.add / nl.multiply on (P, 1)
    # integer tiles lower to float-only tensor_scalar_arith (trnrand#56).
    @nki.jit
    def _mul_kernel(a, b):
        # Integer tensor_tensor needs matching operand dtypes (NCC_EBIR028); int32
        # carries the same low 32 bits as uint32.
        p = nl.ndarray(a.shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(p, nl.load(a), nl.load(b), op=nl.multiply)
        out = nl.ndarray(a.shape, dtype=nl.int32, buffer=nl.shared_hbm)
        nl.store(out, value=p)
        return out

    @nki.jit
    def _add_kernel(a, b):
        s = nl.ndarray(a.shape, dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(s, nl.load(a), nl.load(b), op=nl.add)
        out = nl.ndarray(a.shape, dtype=nl.int32, buffer=nl.shared_hbm)
        nl.store(out, value=s)
        return out

    @nki_op("spikec::roundtrip", mutates_args={})
    def roundtrip(x: torch.Tensor) -> torch.Tensor:
        return _roundtrip_kernel(x)

    @nki_op("spikec::mul", mutates_args={})
    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _mul_kernel(a, b)

    @nki_op("spikec::add", mutates_args={})
    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _add_kernel(a, b)

    @nki_op("spikec::philox", mutates_args={})
    def philox(cl: torch.Tensor, kl: torch.Tensor, kh: torch.Tensor) -> torch.Tensor:
        return philox4x32_kernel(cl, kl, kh)

    M = 0xFFFFFFFF

    def to_i32(vals):
        # Store uint32 bit patterns in int32 tensors (two's complement).
        return torch.tensor(
            [v - (1 << 32) if v >= (1 << 31) else v for v in vals], dtype=torch.int32
        )

    def bits(t):
        return [int(v) & M for v in t.reshape(-1).tolist()]

    def check(label, got, want):
        bad = [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if g != w]
        results[label] = not bad
        print(f"  {label}: {'EXACT' if not bad else f'{len(bad)}/{len(want)} WRONG'}")
        for i, g, w in bad[:4]:
            print(f"    [{i}] got {g:#010x} want {w:#010x}")

    def attempt(label, run, want):
        """Run one check; a compile/runtime failure is a FAIL, not an abort."""
        try:
            got = run()
        except Exception as e:  # noqa: BLE001 - report and keep going
            results[label] = False
            msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            print(f"  {label}: ERROR {type(e).__name__}: {msg}")
            return None
        check(label, bits(got), want)
        return got

    # Probe values straddle the 2**24 boundary and include the Philox constants.
    probe = [
        1,
        0xFFFF,
        0xFFFFFF,
        0x1000000,
        0x1000001,
        0x7FFFFFFF,
        0x80000000,
        0xD2511F53,
        0xCD9E8D57,
        0x9E3779B9,
        0xBB67AE85,
        M,
    ]
    P = 128
    vals_a = (probe * (P // len(probe) + 1))[:P]
    vals_b = list(reversed(vals_a))
    a = to_i32(vals_a).reshape(P, 1)
    b = to_i32(vals_b).reshape(P, 1)

    print(f"[{name}] integer probes (P={P}, int32 storage of uint32 bit patterns):")
    attempt("roundtrip", lambda: roundtrip(a.to(dev)).cpu(), vals_a)
    attempt(
        "mul32 low bits",
        lambda: mul(a.to(dev), b.to(dev)).cpu(),
        [(x * y) & M for x, y in zip(vals_a, vals_b)],
    )
    attempt(
        "add32 wrap",
        lambda: add(a.to(dev), b.to(dev)).cpu(),
        [(x + y) & M for x, y in zip(vals_a, vals_b)],
    )

    # Real gate: trnrand Philox vs the CPU reference, counters/keys over full uint32 range.
    g = torch.Generator().manual_seed(1308)
    cl_u = torch.randint(0, 1 << 32, (P,), generator=g, dtype=torch.int64)
    kl_u = torch.randint(0, 1 << 32, (P,), generator=g, dtype=torch.int64)
    kh_u = torch.randint(0, 1 << 32, (P,), generator=g, dtype=torch.int64)
    ref = philox4x32_reference(
        torch.stack(
            [
                cl_u,
                torch.zeros_like(cl_u),
                torch.zeros_like(cl_u),
                torch.zeros_like(cl_u),
            ],
            -1,
        ),
        torch.stack([kl_u, kh_u], -1),
    )
    cl, kl, kh = (to_i32(t.tolist()).reshape(P, 1).to(dev) for t in (cl_u, kl_u, kh_u))

    print(
        f"[{name}] trnrand philox4x32_kernel vs philox4x32_reference ({P} lanes x 4 words):"
    )
    out = attempt("philox eager", lambda: philox(cl, kl, kh).cpu(), bits(ref))
    compiled = torch.compile(lambda *x: philox(*x), backend="neuron", fullgraph=True)
    attempt("philox compile", lambda: compiled(cl, kl, kh).cpu(), bits(ref))

    if out is not None:
        u = torch.tensor(bits(out), dtype=torch.float64) / 2**32
        print(
            f"  uniform mean from kernel output: {u.mean().item():.4f} "
            "(expect ~0.5; #1308 symptom was ~0.31)"
        )

    verdict = (
        "UNBLOCKED"
        if results["philox eager"] and results["philox compile"]
        else "STILL BLOCKED"
    )
    print(f"RESULT [{name}]: Philox {verdict}")
    print("  torch_neuronx", torch_neuronx.__version__)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--pass":
        single_pass(sys.argv[2])
    else:
        orchestrate()
