# Developing NKI kernels (suite convention)

Canonical reference for how every trnsci library (`trnblas`, `trnfft`,
`trnrand`, `trnsolver`, `trnsparse`, `trntensor`) dispatches NKI
kernels and iterates on them. Each library re-uses the same pattern
with a library-specific prefix on its env vars.

Reference implementation: [`trnblas/trnblas/nki/dispatch.py`](https://github.com/trnsci/trnblas/blob/main/trnblas/nki/dispatch.py).

## Three dispatch modes

Every sub-project supports three ways to run its NKI kernels:

| Mode | Trigger | When to use |
|------|---------|-------------|
| **PyTorch fallback** | `HAS_NKI = False` (non-Neuron host), or an `_nki_*_impl` exception gets caught | Laptops, GPUs, CI's ubuntu-latest — the default for anyone who doesn't have Neuron installed |
| **NKI hardware** | `HAS_NKI = True` + default env. Kernel runs through `torch_xla` → NEFF compile → Trainium dispatch | Real perf numbers, final validation |
| **NKI simulator** | `{LIB}_USE_SIMULATOR=1` + `HAS_NKI = True`. Kernel runs through `nki.simulate(kernel)(numpy_args)` on CPU | Fast correctness iteration during kernel design |

All three share the same kernel source: `@nki.jit`-decorated functions
inside `if HAS_NKI:` blocks.

## Simulator workflow (the big unlock)

NKI 0.3.0 Stable (Neuron SDK 2.29, April 2026) ships a CPU simulator
that runs kernels without Trainium hardware. It collapses the iteration
loop from ~8–12 min per attempt (instance start + SSM + NEFF compile)
to seconds — critical for kernel design where each new semantic
constraint costs one round-trip to discover.

trnblas's v0.4.3 silent-fallback correction plus the six #15 M1
iterations each paid ~8 minutes of AWS time. Under the simulator that
becomes under a minute total.

### How each sub-project enables it

Each library carries its own env var mirroring `{LIB}_REQUIRE_NKI`:

| Library | Env var |
|---------|---------|
| trnblas | `TRNBLAS_USE_SIMULATOR=1` |
| trnfft | `TRNFFT_USE_SIMULATOR=1` |
| trnrand | `TRNRAND_USE_SIMULATOR=1` |
| trnsolver | `TRNSOLVER_USE_SIMULATOR=1` |
| trnsparse | `TRNSPARSE_USE_SIMULATOR=1` |
| trntensor | `TRNTENSOR_USE_SIMULATOR=1` |

Each library exposes a runner: `./scripts/run_simulator_tests.sh`
(SSM → trn1 DLAMI) **and** a `nki-simulator` CI job on
`ubuntu-latest` that runs the marked suite on every push + PR.
Trnblas is the reference implementation; sister libraries adopt the
job as they land their simulator dispatch.

The GH Actions install line:

```bash
pip install -e ".[dev]"
pip install --extra-index-url https://pip.repos.neuron.amazonaws.com "nki>=0.3.0"
```

Verified: `nki 0.3.0+23928721754` wheel is 15.3 MB, installs on
`ubuntu-latest` (py 3.12) in ~3 s; full `nki-simulator` job runs in
under a minute. `torch-neuronx` is **not** needed for the simulator
path — `nki.simulate` takes NumPy directly and bypasses `torch_xla`.

### What the CI gate catches (and misses)

| Gate | Runner | Catches | Misses |
|------|--------|---------|--------|
| `test` matrix | `ubuntu-latest` | Pure-Python correctness against `torch.*` reference. | Anything NKI-kernel-specific. |
| `nki-simulator` | `ubuntu-latest` | Python trace-level kernel errors: wrong `nc_matmul` kwargs, dropped ops, shape/tile-size mismatches, PSUM→HBM dma_copy refusals. | MLIR verifier errors — simulator explicitly skips compile. Perf. |
| `neuron` (SSM) | `trn1`/`trn2` | Full NEFF compile + on-hardware execution. MLIR verification. Real perf. | Nothing. |

**On the trnblas NKI 0.3.0 migration, four of five breaking-change
errors would have surfaced on the `nki-simulator` gate.** The fifth
(partition-broadcast strictness, MLIR-level) still requires hardware.
NKI 0.3.0 has **no documented device-free NEFF compile API**:
`nki.baremetal` requires a Neuron device; `nki.simulate` explicitly
skips compile. So "simulator on ubuntu-latest + hardware on SSM" is
the full gate set — there is no CPU-side compile-check layer to add
between them.

### Simulator limitations

From [`nki.simulate` API docs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.simulate.html):

- **No compile.** NKI compiler errors won't surface — kernels that
  simulate clean can still fail the NEFF compile on hardware.
- **Meta-programming mismatch.** The simulator accepts arbitrary
  Python; the compiler enforces a restricted subset.
- **Memory model is loose.** SBUF / PSUM capacity overflows aren't
  detected; kernels that simulate clean can still OOM the SBUF at
  runtime.
- **No parallelism / latency.** Multi-engine pipelining (TE + VE +
  Scalar + GPSIMD running concurrently) isn't modelled. Simulator
  output gives no perf signal.
- **Not-implemented simulator APIs:** `nki.collectives`,
  `local_gather`, `nc_stream_shuffle` with `mask=255`, `nc_matmul_mx`,
  `quantize_mx`.

**Use the simulator for correctness + constraint iteration. Use
hardware for perf numbers and final sign-off.**

## Dispatch pattern (boilerplate)

The pattern each `_nki_*_impl` in any sub-project should follow:

```python
def _nki_foo_impl(A, B):
    if not HAS_NKI:
        raise RuntimeError("NKI not available")
    # ... shape padding, dtype coercion as needed ...
    try:
        A_feed = A.contiguous()
        B_feed = B.contiguous()
        if _use_simulator():
            # CPU path — NumPy inputs, bypasses torch_xla.
            out_np = nki.simulate(_foo_kernel)(
                A_feed.cpu().numpy(), B_feed.cpu().numpy()
            )
            result = torch.from_numpy(np.asarray(out_np)).to(A.device)
        else:
            # Hardware path — torch_xla bridge + @nki.jit dispatch.
            (a, b), orig_device = _to_xla(A_feed, B_feed)
            c = _foo_kernel(a, b)
            result = c.to(orig_device)
        return result  # slice back to original shape if padded
    except Exception as exc:
        if _REQUIRE_NKI:
            raise
        _warn_fallback(exc)
        return _torch_reference(A, B)
```

### Helpers already defined in every dispatch module

- `_to_xla(*tensors) -> (list, device)` — XLA device setup.
- `_REQUIRE_NKI` — env var gate that turns the try/except `Exception`
  into a re-raise.
- `_use_nki()` — `auto / pytorch / nki` backend selection.
- `_warn_fallback(exc)` — warn-once categorized warning on silent
  fallback (catches PATH / plugin misconfigurations).

Add to each library:

- `_USE_SIMULATOR` (module constant) from `{LIB}_USE_SIMULATOR` env.
- `_use_simulator()` — returns `_USE_SIMULATOR and HAS_NKI`.

## Namespace: `nki.*` is canonical

NKI 0.3.0 promotes `nki.*` as the official namespace. The legacy
`neuronxcc.nki.*` shim still works in 2.29 but is deprecated. Every
sub-project imports from `nki.*` exclusively:

```python
import nki
import nki.isa as nisa
import nki.language as nl
```

Minimum dependency: `nki>=0.3.0` (Neuron SDK 2.29+) in each library's
`[neuron]` extra.

## NKI 0.3.0 migration reference

Breaking changes trnblas navigated on the migration pass. Every
sub-project with NKI kernels needs to handle these:

| Area | Before (NKI Beta 2) | After (NKI 0.3.0) |
|------|---------------------|-------------------|
| Namespace | `neuronxcc.nki.*` | `nki.*` |
| `nc_matmul` | `psum[...] += nisa.nc_matmul(a, b)` returning tile | `nisa.nc_matmul(dst=psum, stationary=a, moving=b, accumulate=True)` writes in-place |
| Kwargs | Positional accepted | `stationary` / `moving` / `dst` all required keywords |
| PSUM → HBM via nl.store | `c_sbuf = nl.copy(psum, ...); nl.store(hbm, value=c_sbuf)` | Must explicitly allocate SBUF + `nisa.tensor_copy(src=psum, dst=sbuf_tile)` first |
| Tensor-tensor divide | `nl.divide(a, b)` | Dropped. Use `nl.multiply(a, nl.reciprocal(b))` |
| Tensor-tensor broadcast | Loose broadcasting | Partition dims must match. `(1, 1) ⊕ (P_TILE, 1)` rejected by MLIR verifier |

See the [NKI 0.3.0 migration guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/migration/nki-0-3-0-update-guide.html)
for the complete list. Trnblas records each change it encountered in its
CHANGELOG; sister projects should do the same.

## Design discipline

Kernels in the trnsci suite should **exploit Trainium architecture**
rather than port cuBLAS equivalents. Before designing a new kernel,
state which of these features it uses:

- **Multi-engine pipelining** — Tensor + Vector + Scalar + GPSIMD
  engines run concurrently.
- **Explicit SBUF hierarchy** — keep operands resident across many
  ops; avoid HBM round-trips.
- **Persistent operands** — load once, reuse across many ops.
- **PSUM accumulator** — internal `accumulate=True` path on
  `nc_matmul`.
- **Fused non-matmul reductions** — patterns with no cuBLAS
  equivalent.

If the answer is "this is the NKI version of the cuBLAS call," the
framing is wrong — rethink what the kernel should be doing.

## Suite coordination

trnblas is the first library to land this pattern (Apr 14, 2026).
Mirror adoption issues tracked in each sister repo. The suite-wide
coordination issue lives in the umbrella.
