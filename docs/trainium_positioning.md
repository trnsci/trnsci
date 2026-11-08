# Trainium as a numerical-computing substrate

## The competitive context

Every major silicon roadmap from 2022 onward has converged on the same architecture: a tensor-native core with FP32 accumulate on BF16/FP8/FP4 inputs, FP64 either halved, emulated, or absent. The numbers tell the story clearly:

| Chip | FP64 tensor | BF16 tensor | Ratio |
|---|---:|---:|---:|
| H100 SXM5 | 67 TFLOPS | 989 TFLOPS | 1:14.8 |
| B200 | ~30 TFLOPS | 2,250 TFLOPS | 1:61 |
| B300 Ultra | <1 TFLOPS | ~5,000 TFLOPS | <1:5000 |
| Trn2 (per chip) | none | 667 TFLOPS | — |
| Trn3 (per chip) | none | 671 TFLOPS (2,517 MXFP8) | — |

cuBLAS in CUDA 13 now emulates FP64 DGEMM via the Ozaki scheme on FP8 tensor cores (NVIDIA Developer Blog, January 2026). AMD halved FP64 matrix throughput from MI300X to MI355X. Google TPU has never exposed FP64. Intel Gaudi 3, Cerebras WSE-3, SambaNova SN40L — none.

Trainium was designed without an FP64 legacy to protect. It codified the post-FP64 architecture in 2020, roughly five years before the rest of the industry caught up. This is the context in which trnsci's algorithms make sense: not as workarounds for a missing FP64 path, but as the correct algorithms for hardware the industry is converging on.

The HPL-MxP benchmark — which scores the fastest achievable HPC performance using hardware-native precision with certified FP64-accuracy output — delivers 4–24× over HPL-FP64 on every top-10 system. Jack Dongarra's ISC 2026 keynote stated the conclusion plainly: *"The most plausible path to effective zettascale is not brute-force FP64, but certified mixed-precision algorithms."*

---

## What Trainium actually is

A Trainium2 chip contains 8 NeuronCore-v3 cores. Each NeuronCore is a heterogeneous compute engine, not a GPU SM:

**TensorEngine** — a 128×128 systolic array consuming a stationary matrix and a moving matrix from SBUF, always accumulating into PSUM. Delivers 158 TFLOPS FP8 / 79 TFLOPS BF16 / 20 TFLOPS FP32 dense per core; AWS lists the 8-core chip at 1,299 FP8 / 667 BF16 / 181 FP32 TFLOPS. Supported dtypes: FP8 (E4M3/E5M2), BF16, FP16, TF32, FP32, INT8.

**VectorEngine** — elementwise/SIMD for axpy, LayerNorm, pooling, reductions, transcendentals with broadcast/scan semantics. On NC-v3, a new performance mode shares a memory bus with GpSimdE for 2–4× uplift on selected BF16/FP16 ops; NC-v3 also allows VectorE and ScalarE to access PSUM in parallel (a restriction lifted from NC-v2).

**ScalarEngine** — pointwise non-linear ops (GELU, SIGMOID, EXP, bias+scale).

**GpSimdEngine** — 8 fully-programmable 512-bit general-purpose vector cores running straight-line C/C++ via Neuron Custom C++ Operators. Each has its own integrated DMA engine on NC-v3. This is the most under-appreciated Trainium feature: it lets the suite offload control flow, rank-revealing logic, adaptive-precision decisions, sparse index computation, and RNG to a Turing-complete engine without leaving the chip or blocking the TensorEngine. trnrand's NKI generator is Threefry4×32-20, with a Philox 4×32-10 kernel alongside it. Neither runs only on GpSimdE: the 32-bit integer adds are pinned to GpSimdE, which on NKI 0.7 does native integer arithmetic on all-int32/uint32 operands, while the xor and rotate steps are bitwise ops on VectorE.

**PSUM** — 2 MiB dedicated FP32 accumulation buffer, the exclusive write target of the systolic array, addressable by VectorE/ScalarE/GpSimdE for post-processing. SBUF is 128 partitions on both generations: 24 MiB (192 KiB per partition) on NC-v2 (Trn1), and 28 MiB (224 KiB per partition) on NC-v3 (Trn2). The compiler keeps part of each partition for itself, so NKI's allocator on Trn2 reports 212,472 bytes (about 207 KiB) per partition.

!!! note "Sources (checked 2026-11-08, Neuron SDK 2.33)"
    Per-core figures, SBUF and PSUM are from AWS's [Trainium2 architecture guide for NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html) and the [NeuronCore-v3](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html) page. The NC-v2 SBUF figure is from the [Trainium/Inferentia2 guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html). The per-chip figures are from the [Trainium2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) and [Trainium3](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html) pages. GpSimd integer arithmetic is described at [`nki.isa.tensor_tensor`](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_tensor.html). The allocator capacity was measured on Trn2 in [trnsparse#111](https://github.com/trnsci/trnsparse/pull/111). Earlier versions of this page gave 79 TFLOPS BF16 per chip (it is per core), SBUF as "192 or 224 KiB" per partition without the generation, and said trnrand's Philox kernel targeted GpSimdE directly.

**Trainium3** (re:Invent 2025, 3nm process): 2.52 PFLOPs FP8 per chip, 144 GB HBM3e at 4.9 TB/s, MXFP8 and MXFP4 microscaled formats per OCP specification, NeuronSwitch-v1 all-to-all fabric for 144-chip UltraServers at 20.7 TB shared HBM3e and 362 MXFP8 PFLOPs. The MXFP formats are the hardware realization of the per-block precision assumptions that mixed-precision HODLR, BLR-LU, and progressive-precision multigrid have been assuming in theory.

---

## Four architectural principles for scientific computing

### 1. PSUM is a free FP32 accumulator

PSUM is wider than SBUF (FP32 vs BF16/FP8), exclusively written by the systolic array (deterministic order), and addressable by every other engine. This makes PSUM the natural target for error-free transformations at systolic scale:

- After a BF16 matmul writes C = A⊗B into PSUM at FP32, VectorE can compute the BF16 split C_hi = fl(C) and the residual C_lo = C − C_hi in PSUM — an in-place error-free split — while the next TensorEngine matmul proceeds on the next tile. The Ogita–Rump–Oishi Dot2 construction becomes the default rather than an expensive overlay.
- For iterative refinement (Carson–Higham), PSUM *is* the high-precision residual buffer: compute r = b − Ax with A and x in BF16, accumulate in FP32 in PSUM, downcast with SR. The inner loop can live entirely on-chip.
- trnblas Phase 2 target (trnblas#22): a `nc_matmul_compensated` kernel delivering FP32-accuracy output from BF16 inputs at roughly 2× the work of naive BF16 matmul, exploiting PSUM as the hidden FP32 accumulator.

### 2. Engine concurrency is free adaptivity

The four engines run independently and overlap. This lets trnsci amortize adaptive logic:

- **Adaptive Ozaki splitting.** TensorEngine runs the k-th Ozaki split-product; simultaneously VectorE estimates the residual norm to decide whether to stop. The adaptive decision has effectively zero marginal cost because it runs on the idle engine.
- **GMRES-IR inner orthogonalization.** The Arnoldi Gram–Schmidt sweep runs on VectorE while TensorEngine runs the next Krylov matvec. trnsolver's cg/gmres paths have the scaffolding; exploitable once Phase 2's compensated-dot primitives land.
- **Randomized sketching.** For Hutch++ trace estimation, the m/3 Hutchinson queries run on VectorE while TensorEngine builds the low-rank projector Q.

!!! warning "Correction (2026-11-08)"
    Engine overlap is not unconditional. On Trn2 the Vector and GpSimd engines were measured not to overlap, against the architecture guide ([aws-neuron-sdk#1449](https://github.com/aws-neuron/aws-neuron-sdk/issues/1449)), and the GP exemplar's fused kernel reached 89% of a fp32 ceiling computed assuming perfect Tensor/Scalar overlap ([the GP post](blog/posts/2026-11-04-gp-million-points.md)). The Tensor → PSUM → Scalar chain does overlap; measure before counting on any other pair.

### 3. Stochastic rounding is in the ISA

Per the Neuron rounding-modes documentation: from NeuronCore-v2 onward, stochastic rounding is programmable per-instruction in NKI/NISA and globally via `NEURON_RT_STOCHASTIC_ROUNDING_EN=1`. Connolly, Higham, and Mary (SIAM J. Sci. Comput., 2021) proved SR rounding errors are mean-zero — which replaces Wilkinson's worst-case n·u inner-product error bound with a √n·u probabilistic bound. This is not a nice-to-have; it is a correctness requirement for BF16 Krylov on long sequences. For BF16 (u ≈ 2⁻⁸), any dot product of length n ≥ ~300 should use SR. Trainium makes this free; most other hardware requires CPU fallback or significant overhead to achieve it.

!!! warning "Correction (2026-11-08)"
    Stochastic rounding is no longer per-instruction in NKI. NKI 0.7 (Neuron SDK 2.33) removed `round_mode` from `nisa.activation`, and the mode is set for the whole process by `NEURON_RT_STOCHASTIC_ROUNDING_EN` when the runtime initializes ([trntensor#41](https://github.com/trnsci/trntensor/issues/41)). With it on, every downcast in the process, in every library, rounds stochastically. SR still removes bias from long BF16 accumulations. It does nothing for an fp32 pipeline whose errors are already random, which is the suite's main case ([the fp32-floor post](blog/posts/2026-11-08-dfmp2-fp32-floor.md)).

### 4. Determinism is structural

The TensorEngine has a fixed reduction order; the compiler statically schedules all instructions and DMAs. Bitwise reproducibility is a structural property of trnsci kernels given a fixed seed — not a ReproBLAS-style overlay with 4–10× overhead. The only non-determinism is seeded SR, which is reproducible across runs.

!!! warning "Correction (2026-11-08)"
    Two qualifications. SR with `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` is not seeded or reproducible run to run, so it is the exception to this principle, not part of it. And the guarantee holds for the suite's own kernels (trnrand and trnsparse are bitwise identical across 1–4 cores, and DeepVariant `call_variants` gives byte-identical VCFs run to run), but device aten reductions can differ between LNC=1 and LNC=2: the GP posterior mean moved by 4.5e-4 ([trnsolver#128](https://github.com/trnsci/trnsolver/issues/128)).

---

## Trainium between SM and TPU

Architecturally, Trainium sits between NVIDIA's SIMT model and Google's TPU compiler-mediated model:

| Characteristic | NVIDIA SM | Google TPU | Trainium NKI |
|---|---|---|---|
| Programming model | Per-thread SIMT | Per-tensor XLA | Per-tile, per-engine Python |
| Tile size | Warp (32 lanes) + WMMA | Large systolic slice | 128 × (up to 512) fixed |
| Memory management | Shared memory (explicit) | Compiler-managed | SBUF/PSUM explicit; 128 partitions |
| Control flow | Full (GPU branches) | Graph-static | Static in `affine_range`; GpSimd for irregular |
| Precision control | Via cuBLAS/cuFFT APIs | Via XLA ops | Per-instruction in NKI |
| Stochastic rounding | No (vendor SR is black-box) | No | ISA-level, per-instruction |
| Accumulator | Hidden in WMMA | Compiler-managed | Named PSUM, addressable by all engines |

Same workload expressed in each model:

| Primitive | NVIDIA SM | Google TPU | Trainium NKI |
|---|---|---|---|
| GEMM tile | Warp-level MMA (WMMA / cutlass) | HLO DotGeneral on MXU | `nisa.nc_matmul` stationary + moving into PSUM |
| Compensated matmul | cuBLAS + separate kernel | Not standard | TensorEngine matmul → PSUM split via VectorE |
| FFT butterfly stage | Per-thread complex multiply-add | XLA fused reduction + permute | TE multiply, VE add, SBUF swap |
| Jacobi rotation | Warp updates two rows | HLO scatter | TE matmul pair + VE reduction to find max |
| Stochastic rounding | N/A or separate kernel | N/A | `nisa.activation(..., round_mode="stochastic")` |
| Integer bit logic (RNG) | CUDA thread, shared mem | N/A (JAX) | NKI integer ops: adds on GpSimdE, bitwise on VectorE |

---

## What fits and what doesn't

| Area | Fit | Notes |
|---|---|---|
| Dense LU / QR / Cholesky / SVD | ✅ Clean | Flagship; DF-MP2 validated PySCF to nanohartree |
| Krylov (CG, GMRES, BiCGStab) | ✅ Clean with SR | SR required for BF16 convergence at n ≫ 300 |
| Block-sparse at 128×128 (BSR) | ✅ Clean | Native tile match; CSR is interop only |
| Hierarchical-matrix BLR/HODLR | ✅ Clean | Phase 2+ opportunity; mixed-precision bounds proved |
| FFT at small N | Partial | DFT-as-GEMM maps onto the Tensor Engine, but on Trn2 per-launch cost dominates standalone small transforms (see the correction below) |
| FFT at large N | Partial | Error accumulates over log₂(N) stages; compensated butterfly + iterative refinement under development |
| Randomized NLA (RSVD, Hutch++) | ✅ Clean | Only need approximate arithmetic; Phase 3 flagship |
| Monte Carlo / QMC | ✅ Clean | SR-tolerant by construction |
| Tensor contractions / einsum | ✅ Clean | DF-MP2 validated end-to-end |
| Eigensolvers (Jacobi, Householder) | ✅ Clean | Jacobi is a natural NKI kernel (each rotation = 2-row matmul) |
| Mixed-precision multigrid | ✅ Clean | Phase 3 opportunity; progressive-precision framework fits |
| Direct sparse solvers (multifrontal) | Partial | Frontal-matrix updates fit; symbolic analysis on GpSimdE |
| ODE integrators (defect correction) | Partial | Stiff linear/semilinear fits; long nonlinear trajectories do not |
| Long MD trajectories | ❌ Doesn't fit | 10⁸–10¹² steps accumulate roundoff; FP32 is marginal |
| Classical CCSD(T) | ❌ Partial | Near-cancellation in correlated sums needs compensated summation everywhere; gated on trnblas#22 |
| Unstructured CSR SpMV | ❌ Doesn't fit | Irregular access hostile to systolic arrays; use CPU fallback |

!!! warning "Correction (2026-11-08)"
    Two rows are overtaken by Trn2 measurements. "DF-MP2 validated PySCF to nanohartree" does not hold for the fp32 device route: E_MP2 agrees with PySCF to about 1e-7–1e-5 Eh at about 500 basis functions, and the suite's gate is 1e-6 Eh against an fp64 reference, since 1e-7 Eh is below the fp32 floor at these energies ([the fp32-floor post](blog/posts/2026-11-08-dfmp2-fp32-floor.md)). Nanohartree needs device fp64 ([trnsci#129](https://github.com/trnsci/trnsci/issues/129)). The small-N FFT row read "✅ Clean, DFT-as-GEMM up to 14× at N ≤ 256". The 14× had no recorded source and predates the native Trn2 path, so it was removed on 2026-11-08. On the native Trn2 device per-launch cost dominates small transforms, and a multithreaded CPU beat standalone small FFTs ([launch-cost post](blog/posts/2026-10-27-trnfft-launch-cost.md)). Large N did better than this table says: the six-step FFT beat a 2-thread CPU by 1.5–11× from 2²⁰ to 2²⁴ on device-resident input. Current status: [exemplars](exemplars.md) and [precision and platform limits](precision_and_platform_limits.md).

---

## Scale-out

Trn1: 2 NeuronCore-v2 per chip, 2D torus across 16 chips per instance (800 Gbps EFAv2).

Trn2: 8 NeuronCore-v3 per chip, 1 TB/s chip-to-chip, 2D torus within instance. Trn2 UltraServer: 64 chips, 512 NeuronCore-v3, 6 TB shared HBM, 185 TB/s aggregate, 12.8 Tbps EFAv3. 83.2 PFLOPS FP8 dense per UltraServer.

Trn3: NeuronSwitch-v1 all-to-all fabric, 144 chips per UltraServer, 20.7 TB HBM3e, 362 MXFP8 PFLOPs. Dedicated collective-compute engines run in parallel with core compute — overlapping residual-estimation matvecs with the next Ozaki split.

The scale-out topology is embarrassingly parallel for Monte Carlo, randomized NLA (RSVD with independent Gaussian sketches per chip), and ensemble PDE workflows. The trnsci Phase 4 roadmap targets these workloads: sharded tensor abstractions, collective ops, and dispatch glue that makes multi-chip operation transparent from the user's perspective.
