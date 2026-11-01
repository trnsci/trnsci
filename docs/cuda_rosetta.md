# CUDA → trnsci Rosetta stone

A reference map between NVIDIA CUDA numerical libraries and their `trnsci` equivalents. If you're porting a CUDA codebase to Trainium, start here.

One framing note before the symbol table: the CUDA cu\* libraries were designed for a hardware generation where FP64 was the native precision. Trainium was designed for the generation after that — BF16/FP8 tensor units with FP32 accumulate in addressable PSUM, stochastic rounding in the ISA, and no FP64 path to protect. This means the porting story is not purely mechanical symbol replacement. For workloads that relied on cuBLAS DGEMM's native FP64, the trnsci equivalent is GMRES-IR with BF16 factorization and FP32 residual via PSUM — which delivers the same Carson–Higham accuracy guarantee at lower cost. For workloads that used cuRAND for reproducible MC, trnrand's seeded Philox/Threefry paths preserve that contract. The table below maps symbols; [why trnsci exists](why.md) explains when the mapping is direct and when the algorithm changes.

## Library mapping

| CUDA | `trnsci` | Scope | Notes |
|---|---|---|---|
| [cuFFT](https://docs.nvidia.com/cuda/cufft/) | [trnfft](https://github.com/trnsci/trnfft) | FFT, complex tensors, STFT | No complex dtype — split real/imag; DFT-as-GEMM fast path at small N |
| [cuBLAS](https://docs.nvidia.com/cuda/cublas/) | [trnblas](https://github.com/trnsci/trnblas) | BLAS Levels 1–3, batched GEMM | BF16+FP32-accum (PSUM); FP64 accuracy via GMRES-IR or Ozaki (Phase 2) |
| [cuRAND](https://docs.nvidia.com/cuda/curand/) | [trnrand](https://github.com/trnsci/trnrand) | Philox/Threefry PRNG, Sobol/Halton QMC | Counter-based, stateless; GpSimd engine target |
| [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/) | [trnsolver](https://github.com/trnsci/trnsolver) | Factorizations, eigendecomposition, Krylov | Jacobi `eigh` (tile-native); GMRES-IR for solve to κ ≲ 10⁷ |
| [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) | [trnsparse](https://github.com/trnsci/trnsparse) | Sparse formats, SpMV/SpMM | BSR-128 is the native compute format; CSR is interop |
| [cuTENSOR](https://docs.nvidia.com/cuda/cutensor/) | [trntensor](https://github.com/trnsci/trntensor) | Einstein summation with planning, decompositions | Fused multi-contraction kernels; PSUM-resident intermediates |

## Per-library detail

### cuFFT → trnfft

cuFFT exposes `cufftPlan*` + `cufftExec*` for 1D/2D/3D complex and real transforms, with plan caching for repeated shapes. `trnfft` mirrors this:

```python
import trnfft
x = torch.randn(1024)
X = trnfft.fft(x)          # like cufftExecC2C, forward
y = trnfft.ifft(X).real    # like cufftExecC2C, inverse
R = trnfft.rfft(x)         # like cufftExecR2C
S = trnfft.stft(x, n_fft=256, hop_length=128)
```

**Semantic differences.**

- Trainium has no `complex64` dtype. `trnfft.ComplexTensor` stores real and imaginary parts as paired real tensors. Arithmetic is decomposed: complex multiply is 4 real multiplies + 2 adds.
- Non-power-of-two sizes use Bluestein's chirp-z transform, padded to a power of two. FP32 accuracy degrades above ~N=500 through the 3-FFT Bluestein chain. Use FP64 on CPU (`x.double()`) for higher precision; Trainium itself is FP32-only.
- Plan caching is keyed by `(size, inverse)`. Plans are cheap to create but not free — reuse them when calling in a loop.

### cuBLAS → trnblas

cuBLAS is a two-tier API: the classic BLAS (`cublasSgemm` etc.) and `cublasLt` for batched / strided / mixed-precision paths. `trnblas` offers the classic surface plus batched GEMM:

```python
C = trnblas.gemm(1.0, A, B)                    # like cublasSgemm
Cb = trnblas.batched_gemm(1.0, A_batch, B_batch)  # like cublasSgemmBatched
X = trnblas.trsm(1.0, L, B, uplo="lower", trans=True)   # like cublasStrsm
```

**Semantic differences.**

- FP32-only. Trainium's Tensor Engine does not support FP64 natively. Chemistry workloads that need FP64 accuracy have to use double-double arithmetic (two FP32 values), which is documented but not yet implemented in trnblas.
- Level-1 and Level-2 (dot, axpy, gemv) are provided for API completeness but don't get NKI kernels. The Tensor Engine would be wasted on vector-only ops. Level-3 is where NKI acceleration lives.
- Tile shapes are fixed: 128 (partition) × 512 (moving). Matrix dimensions are padded implicitly.
- On the native `neuron` device, `auto` sends plain dense GEMMs to aten, which measured 2.4–3.2× faster than the NKI GEMM (trnblas 0.9.0). The NKI wins are the fused kernels, such as the DF-MP2 pair energy (`pair_energy_sym`, 1.8× vs aten).

### cuRAND → trnrand

cuRAND provides two families: pseudo-random (Philox, XORWOW, MRG32k3a) and quasi-random (Sobol, scrambled Sobol). `trnrand` mirrors this:

```python
g = trnrand.manual_seed(42)
x = trnrand.normal(1024, 1024, generator=g)      # like curandGenerateNormal
q = trnrand.sobol(d=8, n=4096)                    # like curandGenerateQuasiSobol
lhs = trnrand.latin_hypercube(d=4, n=1024)        # extra — not in cuRAND
```

**Semantic differences.**

- On the NKI path, `Generator` draws from Threefry4×32-20, counter-based like Philox, which makes it easy to place per-tile counters in parallel on the GpSimd engine. Streams are bit-identical across kernel widths, launch plans and cores. A Philox 4×32 kernel ships too.
- Box-Muller is used for the normal distribution, with an NKI Box-Muller kernel on the Vector Engine (cos, sin, log, sqrt).
- Halton loses quality above ~20 dimensions. Use Sobol for `d > 10`.

### cuSOLVER → trnsolver

cuSOLVER has a dense API (`cusolverDnSpotrf`, `cusolverDnSsyevd`, `cusolverDnSgesvdj`) and a sparse API. `trnsolver` currently covers the dense surface plus iterative Krylov methods:

```python
L = trnsolver.cholesky(A)                        # like cusolverDnSpotrf
w, V = trnsolver.eigh(A)                          # like cusolverDnSsyevd
w, V = trnsolver.eigh_generalized(F, S)
x, info = trnsolver.cg(A, b)                      # iterative
```

**Semantic differences.**

- `eigh` offers two NKI paths. The Householder tridiagonalization + QR kernel is single-tile (n ≤ 128); above that the default cuts over to aten / host LAPACK. `eigh(method="jacobi")` is a batched Jacobi kernel for FP32 n ≤ 128: each rotation is a fixed-size matmul on the Tensor Engine, which maps cleanly to NKI tiles, and it is the better fit for thousands of small problems. Against a fair (thread-split) CPU baseline the margin is small: 1.13× on one core at n = 128, B = 1024, at most 1.25× on 4 cores vs 8 CPU threads (trnsolver 0.19.0).
- `inv_sqrt_spd` uses eigendecomposition; `inv_sqrt_spd_ns` is the Newton-Schulz iteration (`X_{k+1} = 0.5 X_k (3I − A X_k²)`), all GEMMs, which maps better to Trainium.

### cuSPARSE → trnsparse

cuSPARSE handles sparse formats (CSR, CSC, COO, BSR) and sparse-BLAS kernels. `trnsparse` offers BSR (128×128 blocks, the native compute format), CSR / COO + SpMV / SpMM, block-sparse attention, BSR Krylov solvers, plus domain-specific screening:

```python
A = trnsparse.CSRMatrix.from_dense(dense)
y = trnsparse.spmv(A, x)                          # like cusparseSpMV
Y = trnsparse.spmm(A, X)                          # like cusparseSpMM
Q = trnsparse.schwarz_bounds(shell_pair_integrals)
mask = trnsparse.screen_quartets(Q, threshold=1e-10)
```

**Semantic differences.**

- BSR at 128×128 is where the Tensor Engine pays: on one Trn2 core the static-structure BSR SpMM kernel beats dense aten up to ~82–98% block density (trnsparse 0.10.0). Other block sizes run on torch.
- CSR SpMM uses a gather-matmul-scatter pattern: DMA gathers non-zero columns into a dense SBUF tile, Tensor Engine multiplies against the RHS, DMA scatters back. Efficiency depends on the nnz distribution per row.
- Row-variable sparsity patterns need bucketing. Uniform nnz maps cleanly to fixed tiles; highly variable nnz is currently penalized.

### cuTENSOR → trntensor

cuTENSOR is NVIDIA's general tensor-contraction library — it handles arbitrary einsum expressions by selecting contraction paths and dispatching to optimized kernels. `trntensor` offers the same shape:

```python
C = trntensor.einsum("ijk,klm->ijlm", A, B)       # like cutensorContract
plan = trntensor.plan_contraction("ijk,klm->ijlm", A, B)
flops = trntensor.estimate_flops("ijk,klm->ijlm", A, B)
factors = trntensor.cp_decompose(X, rank=8)
core, facs = trntensor.tucker_decompose(X, ranks=(4, 4, 4))
```

**Semantic differences.**

- The planner picks `matmul`, `bmm`, `torch.einsum`, or a multi-step `path` based on the subscript pattern. 2D-over-single-index → `matmul`; batched → `bmm`; complex multi-index → `einsum`. On the native `neuron` device the NKI matmul / bmm kernels are deprecated and these plans run on aten (trntensor 0.19.0); the NKI kernels are the fused chemistry contractions (`mp2_energy`, `ao_to_mo_transform`, `df_exchange`, `df_coulomb`).
- Contraction ordering for 3+ operands uses a greedy path search (like `opt_einsum`).
- CP and Tucker decompositions use alternating least squares on top of `trnblas`-style GEMM primitives. They can be fed back through `einsum` at evaluation time, so a Tucker-compressed tensor can be contracted without materializing.

## Reverse direction

The mapping goes both ways. A few idioms from `trnsci` are arguably cleaner than their CUDA counterparts — see `examples/reverse_port_note.md` for patterns a CUDA programmer might borrow back.
