# trnsci

Scientific computing libraries for AWS Trainium — the reference implementation for the post-FP64 era.

NVIDIA CUDA programmers reach for **cuFFT**, **cuBLAS**, **cuRAND**, **cuSOLVER**, **cuSPARSE**, and **cuTENSOR** when they need fast numerical primitives. The AWS Neuron SDK ships none of these. `trnsci` is six libraries that fill the gap:

| trnsci | NVIDIA analog | Scope |
|---|---|---|
| [trnfft](trnfft/) | cuFFT | FFT, complex tensors, STFT, complex NN layers |
| [trnblas](trnblas/) | cuBLAS | BLAS Levels 1–3, batched GEMM, DF-MP2 energy |
| [trnrand](trnrand/) | cuRAND | Philox/Threefry PRNG, Sobol/Halton QMC |
| [trnsolver](trnsolve/) | cuSOLVER | Cholesky/LU/QR/SVD, Jacobi eigh, CG/GMRES with iterative refinement |
| [trnsparse](trnsparse/) | cuSPARSE | BSR-128 SpMM, Schwarz screening, block-sparse attention |
| [trntensor](trntensor/) | cuTENSOR | Einstein summation with path planning, CP/Tucker/TT, multi-chip sharding |

## What is this for

Two things at once.

**Practically:** workloads that depend on the CUDA cu\* libraries — signal processing, quantum chemistry, Monte Carlo, sparse linear systems, tensor contractions — and need to run on Trainium without per-project rewrites.

**Architecturally:** every major silicon roadmap from 2022 onward has converged on tensor-native cores with FP32 accumulate on BF16/FP8/FP4 inputs and FP64 either halved, emulated, or absent. On H100 the FP64:BF16 throughput ratio is already 1:14.8; on B200 it is 1:61; cuBLAS in CUDA 13 now emulates FP64 DGEMM via the Ozaki scheme on FP8 tensor cores. Trainium was designed without an FP64 legacy to protect — it codified this architecture in 2020. trnsci is the library that treats the mixed-precision numerical-analysis literature (Carson–Higham iterative refinement, Ozaki-scheme FP64 emulation, stochastic rounding, randomized NLA) as the default API rather than an exotic overlay on an FP64-centric BLAS stack.

See [why trnsci exists](why.md) for the full argument.

## Why Trainium specifically

The post-FP64 argument applies to every accelerator vendor. What makes Trainium the right target for a library built on that argument is that every feature the mixed-precision numerical-analysis community has spent a decade asking for is already present — and exposed to the programmer:

- **PSUM is a named, addressable FP32 accumulator.** The 128×128 systolic array always writes its K-reduction result into PSUM in FP32, and every other engine (Vector, Scalar, GpSimd) can read and write PSUM. On NVIDIA tensor cores, the FP32 accumulator is hidden inside the WMMA abstraction. On Trainium it is named, sized, and programmable. This is what makes error-free transformations at systolic scale practical rather than theoretical.
- **Stochastic rounding is in the ISA.** Per-instruction, controllable via `nisa.activation(..., round_mode="stochastic")`. Connolly–Higham–Mary (2021) proved SR replaces the worst-case K·u Krylov error bound with a mean-zero √K·u probabilistic bound — which is what makes BF16 iterative solvers converge rather than stagnate. On NVIDIA, SR requires a third-party library or a CUDA kernel. On Trainium it is one argument.
- **Determinism is structural.** The compiler statically schedules all instructions and DMAs. Every kernel produces bitwise-identical output given a fixed seed — not as a ReproBLAS-style overlay, but as a default.
- **The compiler is open-source.** NKI's MLIR backend (Apache 2.0) is public. Formal error bounds require inspecting the reduction order; on Trainium it is documented and stable.
- **Trainium3 adds per-block microscaling (MXFP8/MXFP4).** These are the hardware realization of the per-block precision assumptions that mixed-precision HODLR, BLR-LU, and progressive-precision multigrid have been writing theorems about for years.

No other shipping accelerator — H100, B200, MI300X, TPU Trillium, Gaudi 3 — exposes all five of these simultaneously with an open-source compiler. NVIDIA has PSUM-equivalent accumulators (hidden), no public SR ISA, and a proprietary compiler. AMD has none of the five. Google has none. Trainium is not missing FP64; it is the architecture the post-FP64 literature has been designing algorithms for. trnsci is the library that makes that concrete.

## Who is this for

- **CUDA programmers** migrating scientific workloads to Trainium. The [CUDA → trnsci Rosetta stone](cuda_rosetta.md) maps cu\* symbols to their trnsci counterparts.
- **HPC and computational-science researchers** who need certified accuracy at stated forward-error tolerances rather than a precision knob and a prayer.
- **Trainium programmers** who want a scientific-computing library stack that exploits the hardware's architectural strengths — PSUM as a free FP32 accumulator, stochastic rounding in the ISA, deterministic dataflow — rather than working around them.

## Get started

```bash
pip install trnsci[all]
```

Try the cross-library integration demo (composes trnblas, trnsolver, trntensor, and trnsparse into an end-to-end DF-MP2 energy evaluation, validated against PySCF to nanohartree tolerance):

```bash
git clone git@github.com:trnsci/trnsci.git
cd trnsci
make install-dev
python examples/quantum_chemistry/df_mp2_synthetic.py --demo
```

## Status

Phase 1 has landed across all six libraries. NKI paths run end-to-end and are exercised in CI via the NKI CPU simulator on every PR. Hardware validation on trn1 is complete for **trnfft** (butterfly FFT + complex GEMM, 70/70 benchmark cases) and **trnblas** (GEMM/SYRK + fused DF-MP2 energy, PySCF agreement to 10 µHa on H₂O / CH₄ / NH₃ at cc-pVDZ). **trnsolver**, **trnsparse**, and **trntensor** are simulator-validated with hardware runs in progress. **trnrand** ships a hardware-validated Threefry4×32 path; Philox is blocked on a named upstream integer-multiply issue ([aws-neuron-sdk#1308](https://github.com/aws-neuron/aws-neuron-sdk/issues/1308)).

PyTorch fallback works end-to-end on any machine, with or without Neuron hardware. The [NKI validation status page](nki_validation_status.md) carries the per-library detail.

Read more:

- [Why this exists](why.md) — the post-FP64 thesis
- [Trainium as a numerical-computing substrate](trainium_positioning.md) — PSUM, SR, determinism, competitive context
- [CUDA → trnsci library mapping](cuda_rosetta.md)
- [Roadmap](roadmap.md)
- [Cross-library integration example](workflows/integration.md)

Follow the [blog](blog/index.md) for technical deep-dives from the sub-project libraries.
