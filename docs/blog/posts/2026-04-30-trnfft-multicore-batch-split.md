---
date: 2026-04-30
categories: [Deep dive, trnfft]
comments: true
---

# trnfft: the NeuronCore is the unit of parallelism

v0.20 ships multi-NeuronCore batch-split FFT. B transforms dispatched across C NeuronCores
gives B/C per core — no shared state, no communication, linear scaling up to B = C. The
architecture made this obvious. Getting the dispatch infrastructure right was less so.

<!-- more -->

## The problem

`precision="bf16"` (v0.17) delivers ~1.2 ms per transform at N=256 on trn1. For a single
spectrogram or a one-shot filter bank that is fine. For workloads that run thousands of
transforms — STFT over a multi-second audio stream, spectral PDE solvers, RF signal
pipelines — single-core throughput caps the pipeline. trn1 has 16 NeuronCores. Until v0.20
they were all idle except one.

The question: can a batch of B transforms run B/C transforms per core with C NeuronCores?

## What the architecture suggests

A GPU spreads work within a kernel via threads and warps; NeuronCores are independent
processors with **separate** SBUF, PSUM, and Tensor Engines. There is no shared scratchpad
between NeuronCores. The natural parallelism is data-parallel, not instruction-parallel —
dispatch an independent transform to each core, collect results.

The Tensor Engine and NEFF cache make this more interesting than it sounds. On first call,
`torch_neuronx.trace` compiles the FFT kernel to a NEFF artifact. Subsequent calls on the
same shape hit the cache and skip compilation. `torch_neuronx.DataParallel` wraps one
compiled model and replicates it across C cores. The compilation cost is paid once per
`(N, inverse, num_cores)` triple; after warmup the per-call overhead is dispatch latency
only.

This is the framing cuFFT's `cufftPlanMany` works in: allocate once, execute many times.
On Trainium the plan is a NEFF artifact, and `DataParallel` is the "plan many" layer.

## The approach

```python
class _FFTModule(torch.nn.Module):
    def forward(self, real, imag):
        y = fft_core(ComplexTensor(real, imag), inverse=self.inverse)
        return y.real, y.imag

# Compile once
traced = torch.jit.trace(module, [sample_real, sample_imag])
neuron_model = torch_neuronx.trace(traced, [sample_real, sample_imag])
dp_model = torch_neuronx.DataParallel(neuron_model)
_dp_model_cache[(n, inverse, num_cores)] = dp_model
```

The batch split itself is three lines:

```python
real_shards = x.real.chunk(num_cores, dim=0)
imag_shards = x.imag.chunk(num_cores, dim=0)
results = [dp_model(r, i) for r, i in zip(real_shards, imag_shards, strict=True)]
```

On CPU (no `torch_neuronx`): the same chunk-and-dispatch loop runs sequentially. The
output is identical; there's no hardware parallelism, but the code path is tested and the
architecture is clear.

## What didn't work

**The `import torch_neuronx` problem.** The initial implementation used a try/except to
check availability:

```python
try:
    import torch_neuronx
    results = _neuron_dp_dispatch(...)
except ImportError:
    results = [...]
```

Ruff's F401 rule flagged `torch_neuronx` as "imported but unused" — because from ruff's
static analysis, `torch_neuronx` isn't referenced by name after the import (the dispatch
goes through `_neuron_dp_dispatch`). The fix: `importlib.util.find_spec("torch_neuronx")`
at module load time, then branch on `HAS_TORCH_NEURONX` instead of a runtime try/except.
The error message was accurate; the fix was non-obvious.

**zip() strictness.** B905 requires an explicit `strict=` argument on all `zip()` calls.
`strict=True` is semantically correct here (real_shards and imag_shards always have the
same length from the same `chunk` call), but ruff won't infer that. Added `strict=True`
throughout.

**First-call compilation cost.** On hardware, `torch_neuronx.trace` takes 30–90 s per
`(N, inverse, num_cores)` key. This is a property of the Neuron compiler, not of the
dispatch design — it compiles a full NEFF artifact and writes it to the NEFF cache on disk.
After warmup (or after the NEFF cache warms across runs), subsequent calls take milliseconds.
The model cache in `_dp_model_cache` prevents re-compilation within a session; the NEFF
disk cache persists across sessions. The first call per shape per instance type is slow.
This needs a note in the API docs; it is not a bug, but it surprises users who expect
`multi_core_fft` to be faster than `fft` on the first call.

**Upstream ask (Neuron SDK):** A `torch_neuronx.is_available()` function analogous to
`torch.cuda.is_available()` would be cleaner than `importlib.util.find_spec`. The
recommended pattern in the current SDK docs is to catch `ImportError`, which triggers
static analysis warnings in lint-enforced codebases.

## Numbers

Hardware validation pending. The batch-split architecture is CPU-verified: 16 sizes and
core counts, roundtrip tested, shape contracts confirmed. Theoretical throughput for a
batched workload with B=16 transforms at N=256 across C=16 NeuronCores: ~1.2 ms total
(≈ single-core per-transform latency, all 16 in parallel). The actual number depends on
dispatch overhead, NEFF cache state, and batch balance — hardware measurement is the next
step.

**Fit assessment:** multi-NeuronCore batch split is well-indexed for workloads with large
batch dimensions (STFT, spectral solvers) where each transform is independent. It is
poorly indexed for single-transform workloads, which is what trn1's training-first design
is optimised for — a single N=256 FFT leaves 15 of 16 NeuronCores completely idle. The
stage-parallel path (v0.21, row-column decomposition) addresses single-transform scaling
for large N.

## What's next

- **Hardware benchmark:** `set_multicore(True, num_cores=16)` on trn1 with batch sizes
  {1, 4, 8, 16, 32}. Measure wall-clock throughput and per-core utilisation.
- **Stage parallelism (v0.21):** single-transform multi-core via row-column FFT
  decomposition. N = n1×n2: column DFTs on C cores, twiddle multiply, row DFTs on C cores.
  The twiddle is the "inter-core exchange" and runs in FP32 on the host — no NKI
  inter-core primitive needed. See the [v0.21 CHANGELOG](https://github.com/trnsci/trnfft).
- **ozaki_hq precision characterisation:** hardware measurement of 2-level Ozaki rel
  error (theory: ~2e-9 at N=64) via `scripts/run_precision_characterization.sh`.

Full roadmap: [trnsci/trnsci](https://github.com/trnsci/trnsci/issues).

## Takeaway

The NeuronCore is the unit of data parallelism on Trainium, not the thread or warp. The
NEFF cache is the "plan" abstraction — compile once, dispatch many. `torch_neuronx.DataParallel`
wraps both into a model-parallel interface that looks like any other PyTorch `DataParallel`,
except the first call compiles a NEFF artifact rather than allocating a CUDA kernel. The
throughput model is simple: B transforms, C cores, B/C per core, linear scaling up to B = C.
What's left is the single-transform case, where the row-column decomposition replaces the
missing inter-core allreduce with a host-side twiddle multiply.
