# Why trnsci exists

## The practical reason

A programmer targeting an NVIDIA GPU for a scientific workload reaches, by reflex, for the CUDA library ecosystem: cuFFT for transforms, cuBLAS for dense linear algebra, cuRAND for reproducible random draws, cuSOLVER for factorizations and eigendecomposition, cuSPARSE for sparse-matrix operations, cuTENSOR for general tensor contractions. Each library is decades of accumulated numerical care — stable algorithms, careful precision handling, tuned kernels.

On AWS Trainium, there is no equivalent. The Neuron SDK ships NKI, a tile-level programming model for the NeuronCore engines, plus fused kernels for transformer training. That covers one workload very well. It does not cover what a computational scientist encounters most days: a spectral method, a density-fit correlation energy, a Monte Carlo integration, a sparse solver, a tensor decomposition.

trnsci fills that gap: six libraries mirroring the CUDA cu\* split, Python-first APIs with PyTorch fallback on any hardware, and NKI kernels underneath for Neuron acceleration. See the [CUDA → trnsci Rosetta stone](cuda_rosetta.md) for the symbol-by-symbol mapping.

## The deeper reason

The practical reason is just software coverage. The deeper reason is architectural timing.

Every major silicon roadmap from 2022 onward has converged on the same design: a **tensor-native core with FP32 accumulate on BF16/FP8/FP4 inputs, FP64 either halved, emulated, or absent**. On H100, FP64 tensor throughput is already 1:14.8 versus BF16. On B200 it is 1:61. On B300 Ultra, NVIDIA's own datasheets show FP64 tensor throughput under 1 TFLOPS; cuBLAS in CUDA 13 now emulates FP64 DGEMM via the Ozaki scheme on FP8 tensor cores. AMD halved FP64 matrix throughput from MI300X to MI355X in the same generation that added FP4/FP6. Google TPU has never exposed FP64. Intel Gaudi 3, Cerebras WSE-3, SambaNova SN40L — none.

Trainium was designed without an FP64 legacy to protect. It is not late to this trend. It codified it in 2020, roughly five years before the industry caught up. Trainium3 (re:Invent 2025, 3nm, 2.52 PFLOPs FP8 per chip) extends this with MXFP8/MXFP4 microscaled formats — which are not incidental to scientific computing: per-block scale factors are exactly the structure that mixed-precision iterative refinement, Ozaki decomposition, and hierarchical-precision block methods have been assuming in theory for years.

The numerical-analysis community has been building the framework for this hardware for over a decade. Buttari and Dongarra (2007) showed mixed-precision iterative refinement delivers FP64 accuracy at FP32+FP16 cost. Carson and Higham (2017–2018) generalized this to three-precision GMRES-IR with convergence to κ∞(A) ≲ 10⁷. Halko, Martinsson, and Tropp (2011) built randomized SVD on the observation that algorithms only need approximate arithmetic where statistical error dominates anyway. Connolly, Higham, and Mary proved stochastic rounding makes BF16 Krylov solvers converge to their theoretical floor where round-to-nearest stagnates. Ozaki, Ogita, and colleagues showed how to recover FP64 accuracy from FP8/BF16 tensor units via slice decomposition — which is now in cuBLAS.

What was missing was a **coherent library** treating these ideas as the default API rather than exotic overlays on an FP64-centric LAPACK stack. trnsci is that library.

## What Trainium actually exposes

Trainium2's NeuronCore-v3 contains:

- A **128×128 systolic array** that accumulates BF16/FP8 matmul products **into a dedicated FP32 PSUM buffer** — the free wider accumulator the Carson–Higham framework has been asking accelerators to expose. Unlike NVIDIA tensor cores, PSUM is named, sized, and addressable by every other engine.
- **Per-instruction stochastic rounding in the ISA** — not a runtime flag, a per-instruction argument in NKI. This turns BF16 Krylov convergence from aspirational to structural.
- **Deterministic static scheduling** — the compiler schedules all instructions and DMAs; hardware semaphores synchronize engines. Bitwise reproducibility is a structural guarantee, not a ReproBLAS-style overlay.
- **Four independently-operating engines** (Tensor, Vector, Scalar, GpSimd) that run concurrently — adaptive precision decisions (the Ozaki stopping criterion, GMRES-IR residual norm) can run on VectorE/GpSimdE while the TensorEngine runs the next matmul.
- **NKI** — an open-source (Apache 2.0) MLIR-based Python DSL that exposes the full hardware surface, including `nisa.nc_matmul`, `nisa.activation` with rounding-mode control, direct SBUF/PSUM allocation, and GpSimd programming for arbitrary integer logic.

Trainium3 adds 144 GB HBM3e at 4.9 TB/s, MXFP8/MXFP4 per-block microscaling, and a NeuronSwitch-v1 all-to-all fabric for 144-chip UltraServers. The MXFP formats are the hardware realization of the per-block precision assumptions that mixed-precision HODLR, BLR-LU, and progressive-precision multigrid have been writing theorems about.

## What fits and what doesn't

The thesis is not that all scientific computing belongs on Trainium. The honest verdicts:

**Fits cleanly:** dense linear algebra (LU, QR, Cholesky, SVD, eigensolvers), Krylov solvers (CG, GMRES, BiCGStab) with SR-enabled BF16 matvec, block-sparse at 128×128 (natural match to the systolic partition dimension), FFT via DFT-as-GEMM at small N and Stockham decompositions at larger N, Monte Carlo and QMC (SR-tolerant by construction), tensor contractions for quantum chemistry (DF-MP2 end-to-end matches PySCF to nanohartree), randomized NLA (RSVD, Hutch++ — only need approximate arithmetic where statistical error dominates).

**Partial fit:** general sparse SpMV (CSR is hostile to systolic arrays; BSR at 128×128 and hierarchical-matrix compression work well for structured sparsity), FFT at large N (precision accumulates across log₂(N) butterfly stages; compensated butterfly and iterative FFT refinement are the engineering response), classical CCSD(T) (correlated methods accumulate cancellation error that requires compensated summation everywhere).

**Doesn't fit:** long-trajectory molecular dynamics (10⁸–10¹² time steps accumulate roundoff that FP32 doesn't solve and BF16 makes worse), fully unstructured sparse SpMV (the PyTorch CPU fallback is the right choice there).

trnsci says this explicitly in its documentation rather than pretending universal applicability. The areas that fit are a substantial fraction of scientific computing's FLOPs — dense LA, Krylov, randomized, MC, PDE discretization, tensor contractions — and they fit very well.

## The API contract

The long-term API direction is not a precision knob. It is a **solve-to-target-error contract**:

```python
x, info = trnsolver.solve(A, b, target_forward_error=1e-10)
```

trnsci selects factorization precision, residual precision, Ozaki split count, IR iteration count, and SR policy automatically. The `info` struct returns achieved forward error, the theorems invoked to bound it, and the precision trajectory used. This is the operational realization of Higham's "accuracy as a resource" thesis and what serious HPC users actually want — LAPACK's `?GESV` is silent on achieved forward error and returns garbage on ill-conditioned inputs. trnsolver's existing `iterative_refinement=True` flag and trntensor's `precision=` kwarg are the in-place scaffolding; the unified `target_forward_error` contract across the suite is the Phase 2/3 API deliverable.

## What this is not

- **Not a replacement for the Neuron SDK's transformer kernels.** LLM training has production-grade kernels already shipping; trnsci targets the orthogonal space.
- **Not a deep-learning framework.** PyTorch or JAX for that.
- **Not a claim of faster FFTs than cuFFT on N=2²⁰.** Trainium is sized for large sustained GEMMs; the per-op comparisons depend heavily on shape and context. The goal is certified accuracy at stated forward-error tolerances, not peak TFLOPS.
- **Not AWS-affiliated.** trnsci is an independent open-source project under Apache 2.0. The benchmarks and architectural takes are third-party observations, not AWS product claims. See the [disclaimer](disclaimer.md).
