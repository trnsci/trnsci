# SDK 2.30.0 spike prompts

Neuron SDK 2.30.0 shipped May 21, 2026 with NKI 0.4.0. This file contains self-contained
prompts to run the four gating spike experiments on trn1 before the NKI-native dispatch
migration begins (trnsci/trnsci#35).

**Dispatch rule:** run each spike on a trn1.2xlarge or trn2.48xlarge instance via SSM.
Post results as comments on [trnsci/trnsci#35](https://github.com/trnsci/trnsci/issues/35).
Do not begin per-library migration until all four spike results are posted there.

---

## Spike 1 — torch.compile + @nki.jit interop

**Paste this into the agent you use to run hardware scripts:**

```
You are running a spike experiment on a Neuron SDK 2.30.0 trn1.2xlarge instance.
The goal is to verify that torch.compile(fn, backend="neuron") correctly traces and
dispatches a function that calls a @nki.jit kernel.

Write and run the following script via SSM (scripts/spike_sdk230_compile_interop.py):

    import torch
    import torch_xla.core.xla_model as xm  # old path — check if still works
    try:
        import torch_neuronx
        has_neuronx = True
    except ImportError:
        has_neuronx = False

    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def _simple_matmul_kernel(A, B, C_out):
        """Minimal 128x128 matmul kernel to test compile dispatch."""
        C = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.shared_hbm)
        a_tile = nl.load(A[0:128, 0:128])
        b_tile = nl.load(B[0:128, 0:128])
        psum = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=psum, stationary=a_tile, moving=b_tile, accumulate=True)
        c_sbuf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(src=psum, dst=c_sbuf)
        nl.store(C[0:128, 0:128], value=c_sbuf)
        return C

    # Test 1: does @nki.jit still work via the existing XLA path?
    A = torch.randn(128, 128)
    B = torch.randn(128, 128)
    print("NKI jit dispatch (existing path):", end=" ")
    try:
        result = _simple_matmul_kernel(A, B)
        print("OK, shape", result.shape)
    except Exception as e:
        print("FAILED:", e)

    # Test 2: does torch.compile with neuron backend work?
    print("torch.compile(backend='neuron'):", end=" ")
    try:
        def fn(A, B):
            return _simple_matmul_kernel(A, B)
        compiled = torch.compile(fn, backend="neuron")
        result2 = compiled(A, B)
        print("OK, shape", result2.shape)
    except Exception as e:
        print("FAILED:", e)

    # Test 3: are results numerically equivalent?
    if 'result' in dir() and 'result2' in dir():
        diff = (result.cpu() - result2.cpu()).abs().max().item()
        ref = (A @ B).abs().max().item()
        print(f"Max abs diff vs reference: {diff:.3e} (ref magnitude: {ref:.3e})")

Run this script on the trn1 instance and post the full stdout+stderr output as a comment
on https://github.com/trnsci/trnsci/issues/35 with the header "Spike 1 result: torch.compile
+ @nki.jit interop". Note the SDK version from `python -c "import neuronxcc; print(neuronxcc.__version__)"`.
```

---

## Spike 2 — nki.collectives availability and interface

**Paste this into the agent you use to run hardware scripts:**

```
You are running a spike experiment on a Neuron SDK 2.30.0 trn1.2xlarge or trn2.48xlarge
instance. The goal is to verify that nki.collectives.allreduce is callable from within a
@nki.jit program, and to document its calling convention.

Write and run the following script via SSM (scripts/spike_sdk230_collectives.py):

    import nki
    import nki.language as nl

    # Check 1: is nki.collectives importable?
    print("nki.collectives importable:", end=" ")
    try:
        import nki.collectives
        print("YES")
        print("  dir(nki.collectives):", [x for x in dir(nki.collectives) if not x.startswith('_')])
    except ImportError as e:
        print("NO:", e)

    # Check 2: allreduce signature
    print("nki.collectives.allreduce signature:", end=" ")
    try:
        import inspect
        sig = inspect.signature(nki.collectives.allreduce)
        print(sig)
    except Exception as e:
        print("FAILED:", e)

    # Check 3: simple @nki.jit kernel using allreduce
    # trntensor's _mock_allreduce uses: torch.stack(partials).sum(dim=0)
    # We need to know if the hardware primitive takes a list of tensors + op=SUM
    print("Minimal allreduce kernel:", end=" ")
    try:
        @nki.jit
        def _allreduce_test(x, y):
            """Sum two tensors across shards via allreduce."""
            partial = nl.load(x[0:128, 0:128]) + nl.load(y[0:128, 0:128])
            result = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.shared_hbm)
            nki.collectives.allreduce(partial, result, op=nki.collectives.SUM)
            return result

        import torch
        x = torch.ones(128, 128)
        y = torch.ones(128, 128)
        r = _allreduce_test(x, y)
        print("OK, result[0,0]:", r[0,0].item(), "(expected 2.0)")
    except Exception as e:
        print("FAILED:", e)
        print("  Full error:", repr(e))

Run this on both trn1 and trn2 if available. Post the full stdout+stderr as a comment on
https://github.com/trnsci/trnsci/issues/35 with the header "Spike 2 result: nki.collectives
availability". The calling convention question — does allreduce take a list[Tensor] or
individual arguments, what op constants exist — is the primary question this spike answers.
```

---

## Spike 3 — integer arithmetic precision (Philox gate / aws-neuron-sdk#1308)

**Paste this into the agent you use to run hardware scripts:**

```
You are running a spike experiment on a Neuron SDK 2.30.0 trn1.2xlarge instance.
The goal is to determine whether the 2^24 exact-integer ceiling for nl.copy and nl.multiply
on uint32 tiles has changed in SDK 2.30.0 / NKI 0.4.0.

Background: in SDK 2.29, nl.multiply on uint32 tiles routes through the float32 activation
path, which is exact only up to 2^24 ≈ 16.7M. Values above this are rounded, making
Philox 4x32-10 produce wrong output (distribution mean 0.31 vs expected 0.5). This was
filed as aws-neuron-sdk#1308 and is the gate for trnrand's Philox hardware validation.

Write and run the following script via SSM (scripts/spike_sdk230_integer_precision.py):

    import nki
    import nki.language as nl
    import nki.isa as nisa
    import torch
    import numpy as np

    @nki.jit
    def _integer_roundtrip_kernel(x_in, x_out):
        """Round-trip uint32 values through nl.copy. Tests whether values
        above 2^24 are preserved exactly."""
        tile = nl.load(x_in[0:128, 0:1])
        nl.store(x_out[0:128, 0:1], value=tile)

    @nki.jit
    def _integer_multiply_kernel(a_in, b_in, c_out):
        """Multiply two uint32 tiles. In SDK 2.29, values above 2^24
        are rounded through float32. Tests whether SDK 2.30 changes this."""
        a = nl.load(a_in[0:128, 0:1])
        b = nl.load(b_in[0:128, 0:1])
        c = nl.multiply(a, b, dtype=nl.uint32)
        nl.store(c_out[0:128, 0:1], value=c)

    # Test: input values spanning the 2^24 boundary
    test_values = [1, 256, 16777215, 16777216, 16777217, 0x7FFFFFFF, 0xD2511F53]
    x = torch.zeros(128, 1, dtype=torch.int32)
    for i, v in enumerate(test_values):
        x[i, 0] = v
    x_out = torch.zeros_like(x)

    _integer_roundtrip_kernel(x, x_out)
    print("Round-trip test (nl.copy):")
    for i, v in enumerate(test_values):
        got = x_out[i, 0].item()
        ok = "OK" if got == v else f"WRONG (got {got})"
        print(f"  {v:#010x} -> {got:#010x}: {ok}")

    # Multiply test: a=0x7FFFFFFF, b=0xD251 (expected hi=0x692B6AE8)
    a = torch.zeros(128, 1, dtype=torch.int32)
    b = torch.zeros(128, 1, dtype=torch.int32)
    c = torch.zeros(128, 1, dtype=torch.int32)
    a[0, 0] = 0x7FFFFFFF
    b[0, 0] = 0xD251
    _integer_multiply_kernel(a, b, c)
    got_lo = c[0, 0].item() & 0xFFFFFFFF
    expected_lo = (0x7FFFFFFF * 0xD251) & 0xFFFFFFFF
    print(f"\nMultiply test: 0x7FFFFFFF * 0xD251")
    print(f"  Expected low 32 bits: {expected_lo:#010x}")
    print(f"  Got:                  {got_lo:#010x}")
    print(f"  Exact: {'YES' if got_lo == expected_lo else 'NO'}")

Post the full output as a comment on https://github.com/trnsci/trnsci/issues/35 with the
header "Spike 3 result: integer arithmetic precision (Philox gate)". If round-trip and
multiply are both exact, Philox hardware validation can proceed; update aws-neuron-sdk#1308
with the result. If still broken, document which SDK 2.30 change was expected to fix it
and what the actual behavior is.
```

---

## Spike 4 — per-dispatch overhead under zero-copy

**Paste this into the agent you use to run hardware scripts:**

```
You are running a spike experiment on a Neuron SDK 2.30.0 trn1.2xlarge instance.
The goal is to measure warm-cache @nki.jit dispatch overhead with zero-copy host-device
transfers (enabled by default in SDK 2.30.0) and compare to the ~100 ms baseline measured
in SDK 2.29.

Background: in SDK 2.29, each @nki.jit call via the XLA lazy-eval path costs ~100 ms of
overhead (host->XLA transfer, XLA graph evaluation, NeuronCore enqueue, sync). This overhead
drove trnblas's batched-pair kernel design and trntensor's homogeneous batching optimization.
SDK 2.30.0 enables zero-copy host-device transfers by default. This spike measures whether
the overhead changed.

Write and run the following script via SSM (scripts/spike_sdk230_dispatch_overhead.py):

    import time
    import torch
    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def _matmul_kernel(A, B):
        C = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.shared_hbm)
        a = nl.load_transpose2d(A[0:128, 0:128])
        b = nl.load(B[0:128, 0:128])
        psum = nl.zeros((128, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=psum, stationary=a, moving=b, accumulate=True)
        c_sbuf = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(src=psum, dst=c_sbuf)
        nl.store(C[0:128, 0:128], value=c_sbuf)
        return C

    A = torch.randn(128, 128)
    B = torch.randn(128, 128)

    # Cold call (NEFF compile)
    t0 = time.perf_counter()
    _ = _matmul_kernel(A, B)
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"Cold call (compile): {cold_ms:.1f} ms")

    # Warm calls
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = _matmul_kernel(A, B)
        times.append((time.perf_counter() - t0) * 1000)

    import statistics
    print(f"Warm calls (n=20): mean={statistics.mean(times):.1f} ms, "
          f"median={statistics.median(times):.1f} ms, "
          f"min={min(times):.1f} ms, max={max(times):.1f} ms")
    print(f"SDK 2.29 baseline was ~100 ms warm. "
          f"Ratio vs baseline: {statistics.median(times)/100:.2f}x")

    # Also test with a 2048x2048 matmul to check if overhead scales with work
    A2 = torch.randn(2048, 2048)
    B2 = torch.randn(2048, 2048)
    # (adapt kernel for 2048 or use torch_neuronx trace path)

Post the full output as a comment on https://github.com/trnsci/trnsci/issues/35 with the
header "Spike 4 result: per-dispatch overhead under zero-copy". The key number is the
warm-call median vs the 100 ms SDK 2.29 baseline. If the overhead dropped to <10 ms,
the batched-pair and homogeneous-batching designs in trnblas and trntensor should be
reassessed before the NKI-native migration — the kernel granularity choices were driven
by that 100 ms floor.
```

---

## After all four spikes post results on trnsci/trnsci#35

The per-library migration order should be:

1. **trnrand** — if Spike 3 shows integer arithmetic is now exact, Philox hardware
   validation can finally close. Migration is also the simplest (no public residency API).

2. **trnfft, trnblas, trnsparse, trnsolver** — universal `_to_xla()` deletion, replace
   with `torch.compile(backend="neuron")` wrappers. See trnsci/trnsci#35 per-library
   checklist.

3. **trntensor** — most involved: `to_xla()`/`from_xla()` are public API, need
   `DeprecationWarning`; `xm.mark_step()` removal; `_mock_allreduce` replacement with
   `nki.collectives.allreduce` if Spike 2 confirms the interface.

Each library migration should be a PR to that library's own repo, then a coordinating
update to `docs/nki_validation_status.md` on the umbrella once all six are done.
