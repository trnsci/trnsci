# Trainium as a numerical-computing substrate

## The competitive context

Every major silicon roadmap from 2022 onward has converged on the same architecture: a tensor-native core with FP32 accumulate on BF16/FP8/FP4 inputs, FP64 either halved, emulated, or absent. The numbers tell the story clearly:

| Chip | FP64 tensor | BF16 tensor | Ratio |
|---|---:|---:|---:|
| H100 SXM5 | 67 TFLOPS | 989 TFLOPS | 1:14.8 |
| B200 | ~30 TFLOPS | 2,250 TFLOPS | 1:61 |
| B300 Ultra | <1 TFLOPS | ~5,000 TFLOPS | <1:5000 |
| Trn2 (per chip) | none | 79 TFLOPS BF16 | — |
| Trn3 (per chip) | none | 2,520 TFLOPS FP8 | — |

cuBLAS in CUDA 13 now emulates FP64 DGEMM via the Ozaki scheme on FP8 tensor cores (NVIDIA Developer Blog, January 2026). AMD halved FP64 matrix throughput from MI300X to MI355X. Google TPU has never exposed FP64. Intel Gaudi 3, Cerebras WSE-3, SambaNova SN40L — none.

Trainium was designed without an FP64 legacy to protect. It codified the post-FP64 architecture in 2020, roughly five years before the rest of the industry caught up. This is the context in which trnsci's algorithms make sense: not as workarounds for a missing FP64 path, but as the correct algorithms for hardware the industry is converging on.

The HPL-MxP benchmark — which scores the fastest achievable HPC performance using hardware-native precision with certified FP64-accuracy output — delivers 4–24× over HPL-FP64 on every top-10 system. Jack Dongarra's ISC 2026 keynote stated the conclusion plainly: *"The most plausible path to effective zettascale is not brute-force FP64, but certified mixed-precision algorithms."*

---

## What Trainium actually is

A Trainium2 chip contains 8 NeuronCore-v3 cores. Each NeuronCore is a heterogeneous compute engine, not a GPU SM:

**TensorEngine** — a 128×128 systolic array consuming a stationary matrix and a moving matrix from SBUF, always accumulating into PSUM. Delivers 158 TFLOPS FP8 / 79 TFLOPS BF16 / 20 TFLOPS FP32 per core. Supported dtypes: FP8 (E4M3/E5M2), BF16, FP16, TF32, FP32, INT8.

**VectorEngine** — elementwise/SIMD for axpy, LayerNorm, pooling, reductions, transcendentals with broadcast/scan semantics. On NC-v3, a new performance mode shares a memory bus with GpSimdE for 2–4× uplift on selected BF16/FP16 ops; NC-v3 also allows VectorE and ScalarE to access PSUM in parallel (a restriction lifted from NC-v2).

**ScalarEngine** — pointwise non-linear ops (GELU, SIGMOID, EXP, bias+scale).

**GpSimdEngine** — 8 fully-programmable 512-bit general-purpose vector cores running straight-line C/C++ via Neuron Custom C++ Operators. Each has its own integrated DMA engine on NC-v3. This is the most under-appreciated Trainium feature: it lets the suite offload control flow, rank-revealing logic, adaptive-precision decisions, sparse index computation, and RNG to a Turing-complete engine without leaving the chip or blocking the TensorEngine. trnrand's Philox kernel targets GpSimdE directly.

**PSUM** — 2 MiB dedicated FP32 accumulation buffer, the exclusive write target of the systolic array, addressable by VectorE/ScalarE/GpSimdE for post-processing. SBUF is 24 MiB on NC-v2, 28 MiB on NC-v3, organized as 128 partitions × (192 or 224) KiB.

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

### 3. Stochastic rounding is in the ISA

Per the Neuron rounding-modes documentation: from NeuronCore-v2 onward, stochastic rounding is programmable per-instruction in NKI/NISA and globally via `NEURON_RT_STOCHASTIC_ROUNDING_EN=1`. Connolly, Higham, and Mary (SIAM J. Sci. Comput., 2021) proved SR rounding errors are mean-zero — which replaces Wilkinson's worst-case n·u inner-product error bound with a √n·u probabilistic bound. This is not a nice-to-have; it is a correctness requirement for BF16 Krylov on long sequences. For BF16 (u ≈ 2⁻⁸), any dot product of length n ≥ ~300 should use SR. Trainium makes this free; most other hardware requires CPU fallback or significant overhead to achieve it.

### 4. Determinism is structural

The TensorEngine has a fixed reduction order; the compiler statically schedules all instructions and DMAs. Bitwise reproducibility is a structural property of trnsci kernels given a fixed seed — not a ReproBLAS-style overlay with 4–10× overhead. The only non-determinism is seeded SR, which is reproducible across runs.

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
| Integer bit logic (RNG) | CUDA thread, shared mem | N/A (JAX) | GpSimdE straight-line C |

---

## What fits and what doesn't

| Area | Fit | Notes |
|---|---|---|
| Dense LU / QR / Cholesky / SVD | ✅ Clean | Flagship; DF-MP2 validated PySCF to nanohartree |
| Krylov (CG, GMRES, BiCGStab) | ✅ Clean with SR | SR required for BF16 convergence at n ≫ 300 |
| Block-sparse at 128×128 (BSR) | ✅ Clean | Native tile match; CSR is interop only |
| Hierarchical-matrix BLR/HODLR | ✅ Clean | Phase 2+ opportunity; mixed-precision bounds proved |
| FFT at small N | ✅ Clean | DFT-as-GEMM up to 14× at N ≤ 256 |
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

---

## Scale-out

Trn1: 2 NeuronCore-v2 per chip, 2D torus across 16 chips per instance (800 Gbps EFAv2).

Trn2: 8 NeuronCore-v3 per chip, 1 TB/s chip-to-chip, 2D torus within instance. Trn2 UltraServer: 64 chips, 512 NeuronCore-v3, 6 TB shared HBM, 185 TB/s aggregate, 12.8 Tbps EFAv3. 83.2 PFLOPS FP8 dense per UltraServer.

Trn3: NeuronSwitch-v1 all-to-all fabric, 144 chips per UltraServer, 20.7 TB HBM3e, 362 MXFP8 PFLOPs. Dedicated collective-compute engines run in parallel with core compute — overlapping residual-estimation matvecs with the next Ozaki split.

The scale-out topology is embarrassingly parallel for Monte Carlo, randomized NLA (RSVD with independent Gaussian sketches per chip), and ensemble PDE workflows. The trnsci Phase 4 roadmap targets these workloads: sharded tensor abstractions, collective ops, and dispatch glue that makes multi-chip operation transparent from the user's perspective.
