#!/usr/bin/env python
"""Spike A — the per-library migration recipe on public TorchNeuron (SDK 2.33).

Registers real, hardware-validated suite kernels as native PyTorch custom ops
via ``torch_neuronx.nki_op`` and checks they dispatch on ``device="neuron"`` in
eager and ``torch.compile(backend="neuron")``, with no torch_xla anywhere:

  A1. trnfft ``_complex_gemm_kernel_bf16`` (multi-output kernel, BF16 in / FP32 out)
  A2. trntensor ``matmul_kernel`` (non-tensor ``use_sr`` arg → compile-time constant)
  A3. autograd through an nki_op via ``register_autograd`` (the training path
      trnsparse block-sparse attention will need)

Requires: Trn2/Trn3, Python 3.12+, torch 2.14, torch-neuronx 2.14.*, and the
suite installed (``pip install -e trnfft trntensor``). Run from the umbrella root:

    python scripts/spike_torchneuron_recipe.py 2>&1 | tee ~/spike_recipe.log

Post the log to trnsci/trnsci#35.
"""

import os

# Editable installs have no .dist-info, so the NKI trace cache can't version them
# and may return stale kernels. Disable it before nki/torch_neuronx import.
os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")

import importlib.util
import time
import traceback

import torch
import torch_neuronx
from torch_neuronx import nki_op

from trnfft.nki.dispatch import _complex_gemm_kernel_bf16
from trntensor.nki._kernels import matmul_kernel

DEV = "neuron"
RESULTS = {}


def header():
    import nki
    import neuronxcc

    print("torch", torch.__version__)
    print("torch_neuronx", torch_neuronx.__version__)
    print("neuronxcc", getattr(neuronxcc, "__version__", "?"))
    print("nki", getattr(nki, "__version__", "?"))
    print("torch_xla installed:", importlib.util.find_spec("torch_xla") is not None)
    print("'neuron' dynamo backend:", "neuron" in torch._dynamo.list_backends())
    print(
        "NEURON_LOGICAL_NC_CONFIG:",
        os.environ.get("NEURON_LOGICAL_NC_CONFIG", "(unset)"),
    )


def run(name, fn):
    print(f"\n===== {name} =====")
    try:
        ok = fn()
    except Exception:
        traceback.print_exc()
        ok = False
    RESULTS[name] = "PASS" if ok else "FAIL"
    print(f"RESULT [{name}]: {RESULTS[name]}")


# ---------------------------------------------------------------- A1: trnfft
@nki_op("trnfft_spike::cgemm_bf16", mutates_args={})
def cgemm_bf16(
    ar: torch.Tensor, ai: torch.Tensor, br: torch.Tensor, bi: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _complex_gemm_kernel_bf16(ar, ai, br, bi)


def a1_trnfft_cgemm():
    torch.manual_seed(0)
    ar, ai, br, bi = (torch.randn(128, 128).bfloat16() for _ in range(4))
    ref = (ar.double() + 1j * ai.double()) @ (br.double() + 1j * bi.double())

    def rel_err(cr, ci):
        got = cr.cpu().double() + 1j * ci.cpu().double()
        return ((got - ref).abs().max() / ref.abs().max()).item()

    args = [t.to(DEV) for t in (ar, ai, br, bi)]

    t0 = time.perf_counter()
    cr, ci = cgemm_bf16(*args)
    torch_neuronx.synchronize()
    print(f"eager cold {time.perf_counter() - t0:.1f}s")
    e_eager = rel_err(cr, ci)

    compiled = torch.compile(
        lambda *a: cgemm_bf16(*a), backend="neuron", fullgraph=True
    )
    cr2, ci2 = compiled(*args)
    e_comp = rel_err(cr2, ci2)
    same = torch.equal(cr.cpu(), cr2.cpu()) and torch.equal(ci.cpu(), ci2.cpu())

    print(
        f"eager rel_err={e_eager:.2e}  compile rel_err={e_comp:.2e}  eager==compile bitwise: {same}"
    )
    # BF16 inputs with FP32 PSUM accumulate: expect ~1e-3, well under 1e-2.
    return e_eager < 1e-2 and e_comp < 1e-2


# -------------------------------------------------------------- A2: trntensor
@nki_op("trntensor_spike::matmul", mutates_args={})
def nki_matmul(a: torch.Tensor, b: torch.Tensor, use_sr: bool) -> torch.Tensor:
    return matmul_kernel(a, b, use_sr=use_sr)


def a2_trntensor_matmul():
    torch.manual_seed(1)
    A, B = torch.randn(256, 256), torch.randn(256, 512)
    ref = A.double() @ B.double()
    ok = True
    for use_sr in (False, True):
        c = nki_matmul(A.to(DEV), B.to(DEV), use_sr).cpu()
        err = ((c.double() - ref).abs().max() / ref.abs().max()).item()
        print(f"use_sr={use_sr}: rel_err={err:.2e}")
        ok &= err < 1e-2
    compiled = torch.compile(
        lambda a, b: nki_matmul(a, b, False), backend="neuron", fullgraph=True
    )
    c = compiled(A.to(DEV), B.to(DEV)).cpu()
    err = ((c.double() - ref).abs().max() / ref.abs().max()).item()
    print(f"compile: rel_err={err:.2e}")
    return ok and err < 1e-2


# ---------------------------------------------------------------- A3: autograd
@nki_op("trntensor_spike::matmul_ag", mutates_args={})
def nki_matmul_ag(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return matmul_kernel(a, b)


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    # Backward reuses the same NKI op: dA = dC @ B^T, dB = A^T @ dC.
    return (
        nki_matmul_ag(grad.contiguous(), b.t().contiguous()),
        nki_matmul_ag(a.t().contiguous(), grad.contiguous()),
    )


nki_matmul_ag.register_autograd(_backward, setup_context=_setup_context)


def a3_autograd():
    torch.manual_seed(2)
    A0, B0 = torch.randn(128, 128), torch.randn(128, 128)
    A = A0.clone().to(DEV).requires_grad_()
    B = B0.clone().to(DEV).requires_grad_()
    nki_matmul_ag(A, B).sum().backward()

    Ar, Br = A0.clone().requires_grad_(), B0.clone().requires_grad_()
    (Ar @ Br).sum().backward()
    ea = ((A.grad.cpu() - Ar.grad).abs().max() / Ar.grad.abs().max()).item()
    eb = ((B.grad.cpu() - Br.grad).abs().max() / Br.grad.abs().max()).item()
    print(f"grad rel_err  dA={ea:.2e}  dB={eb:.2e}")
    return ea < 1e-2 and eb < 1e-2


if __name__ == "__main__":
    header()
    run("A1 trnfft cgemm_bf16 eager+compile", a1_trnfft_cgemm)
    run("A2 trntensor matmul (constant arg) eager+compile", a2_trntensor_matmul)
    run("A3 autograd through nki_op", a3_autograd)
    print("\n===== SUMMARY =====")
    for k, v in RESULTS.items():
        print(f"  {v}  {k}")
