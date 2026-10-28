#!/usr/bin/env python
"""Spike D — does trntensor's ``precision="sr"`` still round stochastically?

NKI 0.7 dropped ``nisa.activation``'s per-instruction ``round_mode``, so
trntensor's ``use_sr`` PSUM->SBUF downcast is now a plain Scalar Engine copy
(trnsci/trntensor#41). Whether that copy rounds stochastically depends on the
runtime: AWS documents ``NEURON_RT_STOCHASTIC_ROUNDING_EN=1`` (stochastic) and
``=0`` (round-to-nearest-even) in about-neuron/arch/neuron-features/rounding-modes.

Runs a BF16 matmul (FP32 PSUM -> BF16 output) repeatedly in a fresh process per
setting and reports run-to-run variation and agreement with FP32-then-RNE. SR
shows run-to-run differences, many elements off the RNE result, and a near-zero
mean signed error; RNE is deterministic and equals the RNE reference.

    python scripts/spike_torchneuron_sr_probe.py
"""

import os
import subprocess
import sys

SETTINGS = ("0", "1")
RUNS = 8


def orchestrate():
    for v in SETTINGS:
        env = dict(os.environ, NEURON_RT_STOCHASTIC_ROUNDING_EN=v)
        print(f"\n######## NEURON_RT_STOCHASTIC_ROUNDING_EN={v} ########", flush=True)
        subprocess.run([sys.executable, __file__, "--probe"], env=env, check=False)


def probe():
    import torch
    from torch_neuronx import nki_op

    from trntensor.nki._kernels import matmul_kernel

    @nki_op("trnsci_spike::mm_sr", mutates_args={})
    def mm_sr(a: torch.Tensor, b: torch.Tensor, use_sr: bool) -> torch.Tensor:
        return matmul_kernel(a, b, use_sr=use_sr)

    torch.manual_seed(0)
    A = torch.randn(128, 128).bfloat16()
    B = torch.randn(128, 512).bfloat16()
    exact = A.double() @ B.double()
    rne = exact.float().bfloat16()  # FP32 accumulate, then round-to-nearest-even

    for use_sr in (True, False):
        outs = [
            mm_sr(A.to("neuron"), B.to("neuron"), use_sr).cpu() for _ in range(RUNS)
        ]
        stack = torch.stack([o.double() for o in outs])
        varies = (stack != stack[0]).any(dim=0).double().mean().item()
        off_rne = torch.stack([(o != rne) for o in outs]).double().mean().item()
        mean_err = (stack - exact).mean().item()
        mean_abs = (stack - exact).abs().mean().item()
        print(
            f"use_sr={use_sr!s:5} dtype={outs[0].dtype}  "
            f"run-to-run varying elems={varies:.3f}  off-RNE elems={off_rne:.3f}  "
            f"mean signed err={mean_err:+.2e}  mean |err|={mean_abs:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["--probe"]:
        probe()
    else:
        orchestrate()
