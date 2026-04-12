# Why trnsci exists

## The gap

A programmer targeting an NVIDIA GPU for a scientific workload reaches, by reflex, for the CUDA library ecosystem. cuFFT handles the transforms. cuBLAS does the dense linear algebra. cuRAND gives reproducible random draws. cuSOLVER factors and diagonalizes. cuSPARSE handles the sparse-matrix path. cuTENSOR does general tensor contractions. Each library is decades of accumulated numerical care — stable algorithms, careful precision handling, tuned kernels.

On AWS Trainium, there is no equivalent.

The Neuron SDK ships NKI, a tile-level programming model for the NeuronCore engines, plus a set of fused kernels targeting transformer training. That covers one workload — LLM training — very well. It does not cover the workloads a computational scientist encounters most days: a spectral method, a density-fit correlation energy, a Monte Carlo integration, a sparse solver, a tensor decomposition.

If you want to run any of these on Trainium today, you hand-roll. The cost is high enough that most people don't bother — they move the workload to an NVIDIA instance even when Trainium has the cost or availability advantage.

## Why fill it

Trainium has economics that favour dense, fixed-shape scientific kernels. The systolic Tensor Engine is very well suited to GEMM-heavy workloads — which is to say, most of scientific computing's hot path. Per-FLOP and per-dollar, Trainium is competitive. The gap between that economic case and the observable reality — everyone running LLMs, nobody running eigensolvers — is almost entirely a software-coverage problem.

A library stack closes that gap. Once `trnblas.gemm` does what `cublasSgemm` does, the existing literature of quantum chemistry, computational fluid dynamics, and signal processing becomes portable to Trainium without per-project rewrites.

## Why this shape

The six-library split mirrors the CUDA stack on purpose. The community around NVIDIA's ecosystem has spent a decade internalizing the cuFFT/cuBLAS/cuSOLVER/cuSPARSE taxonomy. Rebuilding that taxonomy on Trainium means a programmer coming from CUDA already knows where to look. The [Rosetta-stone mapping](cuda_rosetta.md) is the primary documentation artefact — every cu\* symbol should have a visible `trn*` analog.

The other reason is composition. A real scientific workload touches four or five of these libraries in a single run. A density-fitted MP2 correlation energy needs Cholesky (solver), dense GEMMs (BLAS), einsum contractions (tensor), integral screening (sparse), and sometimes stochastic trace estimators (RNG). Owning all six in one suite means the cross-library composition is our test case, not an afterthought. See `examples/quantum_chemistry/df_mp2_synthetic.py` for the concrete pattern.

## Non-goals

- **Not a replacement for the Neuron SDK's transformer kernels.** LLM training is a separate specialty with production-grade kernels already shipping.
- **Not a deep-learning framework.** Use PyTorch-XLA or `torch_neuronx` for that.
- **Not a benchmark competition with CUDA.** We are not claiming faster FFTs than cuFFT. We are claiming *any* FFT, plus a consistent path to NKI-accelerated FFT once the kernels land.
- **Not scope creep into arbitrary numerical libraries.** The six-library boundary is drawn deliberately and matches the CUDA stack; new libraries join the suite only when there is a durable analog in the CUDA ecosystem.

## Status

Alpha. Every public API is wired through a PyTorch fallback, so the whole suite runs on a laptop for development. NKI kernels exist as scaffolds — the dispatch layer is in place and picks `nki` over `pytorch` when Neuron hardware is present, but the kernels themselves currently forward to PyTorch. Hardware validation on trn1 / trn2 is the near-term milestone.
