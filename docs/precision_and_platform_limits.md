# Precision and platform limits

The Trn2 compute engines do their arithmetic in fp32, and PSUM accumulates in fp32. There is no float64 datapath. A float64 tensor can live on the `neuron` device, but arithmetic on it is fp32. This page states what the suite does about that, the platform limits the suite has measured and guards against, and where its accuracy targets come from.

Everything here was measured on a trn2.3xlarge with public Neuron SDK 2.33 (torch-neuronx 2.14.3, NKI 0.7.0, neuronx-cc 2.28), LNC=2 unless stated. Upstream behaviour may change in later SDKs. The suite tracks each upstream issue it depends on in [trnsci#135](https://github.com/trnsci/trnsci/issues/135), and drops a guard once a fix ships in a public SDK release.

## The suite float64 policy {#fp64-policy}

A caller who asks for float64 either gets float64 accuracy or is told, once and loudly, that they didn't. The rule is the same in all six libraries ([decision](https://github.com/trnsci/trnsci/issues/35#issuecomment-6583096371), [refinement](https://github.com/trnsci/trnsci/issues/35#issuecomment-6583607270)). `<LIB>` is the upper-case package name: `TRNFFT`, `TRNBLAS`, `TRNRAND`, `TRNSOLVER`, `TRNSPARSE` or `TRNTENSOR`.

| setting | float64 on the host | float64 already on the `neuron` device |
|---|---|---|
| `set_backend("auto")` (default) | host torch at float64 accuracy, no kernel | copied to the host, computed at float64, returned to the device as float64, one `UserWarning` per op |
| `auto` + `<LIB>_FP64_ON_DEVICE=1` | unchanged | device aten at fp32 precision, no warning (an explicit opt-in for speed) |
| `set_backend("nki")` | cast to fp32 for the kernel, one `UserWarning` per op, result returned as float64 | same |
| `<LIB>_REQUIRE_NKI=1` | `NotImplementedError` | `NotImplementedError`, before any copy, and ahead of `FP64_ON_DEVICE` |

In every path the result dtype matches the pytorch backend. The cost is deliberate: device-resident float64 under `auto` pays a host round trip and host compute on every op, so a caller who wants device speed passes float32 or opts in.

Underneath the policy is a dtype guard at the NKI kernel boundary. Each library has a per-kernel dtype table, checked in every launch route (native, XLA, simulator), and a float64, integer, bool or complex tensor that a kernel doesn't take raises `TypeError` before the kernel runs. Before the guard, float64 reaching fp32-only kernels was the root cause of five of the nine device bugs in the [window-3 audit](https://github.com/trnsci/trnsci/issues/35#issuecomment-6570060466).

The guard and policy PRs: [trnblas#91](https://github.com/trnsci/trnblas/pull/91), [trnfft#139](https://github.com/trnsci/trnfft/pull/139), [trnrand#106](https://github.com/trnsci/trnrand/pull/106), [trnsolver#90](https://github.com/trnsci/trnsolver/pull/90), [trnsparse#87](https://github.com/trnsci/trnsparse/pull/87), [trntensor#93](https://github.com/trnsci/trntensor/pull/93). The policy is released in trnblas 0.11.0, trnfft 0.28.0, trnrand 0.14.0, trnsolver 0.22.0, trnsparse 0.14.0 and trntensor 0.24.0, and every current release has it. The narrative is in [the suite's silent wrong answers](blog/posts/2026-11-04-suite-silent-wrong-answers.md).

## Known Trn2 platform limits

### Device float64 is fp32 arithmetic

Device aten computes float64 at fp32 precision *and range*: values below about 1e-38 underflow, as in fp32. A [device-side check](https://github.com/trnsci/trntensor/pull/93#issuecomment-6610999978) found where the line falls. Data movement keeps its bits: `.to`, `.cpu()`, `clone`, `.contiguous()` of a slice or a transpose, and `A[5]` are exact. **`torch.cat`, `index_select`, fancy indexing (`A[idx]`) and even `x * 1.0` round to fp32.** The suite infers this from measurements; the Neuron docs don't state it. It is why the float64 policy routes device float64 to the host by default, and why tests that compare device float64 at float64 tolerance fail by design (they are the largest "platform" category in every conformance run on [#35](https://github.com/trnsci/trnsci/issues/35)).

### DMA spans of 4 GiB: rows of 32 MiB or more

On Trn2 a device operand or result is unsafe when `min(rows, 128) × row_bytes ≥ 2³²`, that is, rows of 32 MiB or more across 64 or more rows. Partitions 64–127 then read or write memory 4 GiB too low ([aws-neuron-sdk#1457](https://github.com/aws-neuron/aws-neuron-sdk/issues/1457), the abort form [#1448](https://github.com/aws-neuron/aws-neuron-sdk/issues/1448)). The same family shows up in three ways:

| op | what happens | upstream | suite guard |
|---|---|---|---|
| matmul, `bmm`, column-slice copy, dim-1 `cat` with such rows | **silently wrong**, or an NRT 1203 DMA abort | [aws-neuron-sdk#1457](https://github.com/aws-neuron/aws-neuron-sdk/issues/1457) | trnblas 0.12.1 and trntensor 0.25.1 row-block every such product into blocks of at most 64 rows ([trnblas#112](https://github.com/trnsci/trnblas/issues/112), [trntensor#123](https://github.com/trnsci/trntensor/issues/123)) |
| `index_select(0, idx)` with an output of 2³² bytes or more, at LNC=1 | wrong data, or the 30 s runtime timeout | [aws-neuron-sdk#1459](https://github.com/aws-neuron/aws-neuron-sdk/issues/1459) | trntensor 0.24.0 gathers rows in blocks of at most 2 GiB ([trntensor#107](https://github.com/trnsci/trntensor/pull/107)) |
| one elementwise op on a single tensor of 4 GiB or more | DMA abort (loud in every case measured) | reported on [aws-neuron-sdk#1457](https://github.com/aws-neuron/aws-neuron-sdk/issues/1457) | audit open, [trnsci#145](https://github.com/trnsci/trnsci/issues/145); streamed chemistry keeps every chunk ≤ 2 GiB |

Still open: the `df_mp2_energy` reduce-scatter ([trnblas#116](https://github.com/trnsci/trnblas/issues/116)) and the trnsparse CG operand ([trnsparse#121](https://github.com/trnsci/trnsparse/issues/121)). The narrative, including a published data point that was right only by luck, is in [rows of 32 MiB](blog/posts/2026-11-08-dma-32mib-rows.md).

### `torch.fft` doesn't run on the device

The `neuron` device doesn't support complex dtypes natively. `torch.fft.irfft` raises "complex dtypes are not supported natively on Neuron", and the rest of `torch.fft` falls back to the CPU at best ([trnfft#82](https://github.com/trnsci/trnfft/issues/82), [trnfft#157](https://github.com/trnsci/trnfft/pull/157#issuecomment-6617354400)). trnfft computes every transform as pairs of real tensors, which is why it is the only FFT on the device and why the [SHT exemplar](exemplars.md#sht) keeps all complex arithmetic off it.

### Complex linear algebra is unsupported

Complex `cholesky`, `eigh`, `svd`, `solve`, `qr` and `lu`, and complex `mv`, `triu` and `tril`, raise on the device. These are loud, and they are the second-largest platform category in the conformance runs ([trnsolver#98](https://github.com/trnsci/trnsolver/issues/98) tracks complex support). One related upstream bug is silent: device matmul ignores PyTorch's lazy conjugate bit, so `A.mH @ B` returns `A.T @ B` ([aws-neuron-sdk#1455](https://github.com/aws-neuron/aws-neuron-sdk/issues/1455)). The [plane-wave exemplar](exemplars.md#plane-wave) starts at the Γ point because there every subspace matrix is real.

### Other measured limits

- **Dispatch floor:** about 0.02 ms per aten call; an `nki_op` call was about 0.10 ms before the suite's cached kernel handles ([aws-neuron-sdk#1456](https://github.com/aws-neuron/aws-neuron-sdk/issues/1456)).
- **Memory and the host link:** 96 GiB of HBM per trn2.3xlarge chip, 24 GiB per logical core at LNC=2; about 13–14 GB/s host to device.
- **Compilation:** every distinct static shape is a separate compiled program, and large cold compiles take minutes. NEFFs compiled off-box are reused ([trnsci#118](https://github.com/trnsci/trnsci/issues/118), [the cache post](blog/posts/2026-11-05-neuron-cache-off-box.md)).

## Where accuracy targets come from

**Start from the fp32 floor.** The fp32 unit roundoff is u = 2⁻²⁴ ≈ 6e-8. A sum of n terms with random rounding errs like √n·u, and with biased rounding like n·u. A tolerance tighter than the floor can't be met by any fp32 route, however correct.

**Check against an independent reference above the floor.** Comparing two fp32 routes with each other measures their difference, not either one's error: it fails when both are right and round differently, and passes when both share a bug. So suite gates compare each route with a float64 reference that shares no code with it, which is also how the conformance suites work.

**The MP2 gate is 1e-6 Eh.** For DF-MP2 on the fp32 device route the gate is |E_MP2(device) − E_MP2(fp64 reference)| ≤ 1e-6 Eh, with the relative error recorded. The reference is a full fp64 run, or an fp64 subset check when a full run isn't practical ([the revised gate](https://github.com/trnsci/trntensor/issues/122#issuecomment-6714981609)). 1e-6 Eh is about 1,600× below chemical accuracy (1 kcal/mol ≈ 1.6e-3 Eh). The earlier 1e-7 Eh gate asked for 8.7e-9 relative agreement at |E_corr| = 11.46 Eh, below u, and failed a device route that the fp64 check showed to be as good as the host's ([the fp32-floor post](blog/posts/2026-11-08-dfmp2-fp32-floor.md)). 1e-7 Eh and below is assigned to device fp64, the error-compensated GEMM of [trnblas#22](https://github.com/trnsci/trnblas/issues/22) under the [epic trnsci#129](https://github.com/trnsci/trnsci/issues/129).

Other exemplars set their targets the same way: the SHT is checked within 1e-6 of fp64 ducc0, and the GP posterior mean within 1e-4 relative of exact fp64 scipy at n = 2×10⁴ ([exemplars](exemplars.md)).

**Stochastic rounding is not the way below the floor.** On NKI 0.7 the rounding mode is a process-wide runtime setting, `NEURON_RT_STOCHASTIC_ROUNDING_EN`, set at process start, not a per-instruction choice ([AWS: rounding modes](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/rounding-modes.html)). With it on, every downcast in the process rounds stochastically and results aren't bitwise reproducible. SR removes bias from long accumulations in a narrow format such as BF16. It doesn't make any single fp32 operation more precise; that takes more bits per operation, which is what trnblas#22 is for.
