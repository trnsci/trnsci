# Trainium between SMs and TPUs

Accelerator architectures have historically sat on a spectrum. At one end, NVIDIA GPUs — many small cores, fine-grained SIMT, warp-level programming. At the other, Google TPUs — one very large systolic array, coarse-grained, compiler-mediated via XLA. Trainium sits in the middle of this spectrum, with NKI as the programming model that exposes both halves.

This page is a reference frame, not a comparison. We are not ranking architectures — we are placing Trainium on the axis so that a programmer coming from either side knows what to expect.

## NVIDIA SMs — fine-grained, drifting coarser

A classic CUDA program is a grid of thread blocks, each thread block a dense bundle of 32-lane warps running SIMT. The mental model is per-thread: you write one thread's view of the kernel and the hardware multiplexes many. Shared memory is explicit; synchronization is explicit; tiling is handled by the programmer or by cuBLAS / cuFFT underneath.

Recent generations have drifted in the coarse direction. Hopper added the Tensor Memory Accelerator and asynchronous tile moves; Blackwell pushes further into tile-granularity APIs. CUDA Graphs capture whole pipelines. `cuBLASLt` and `cutlass` expose tile-sized abstractions above the per-thread view.

The direction of travel is toward larger, more compiler-scheduled units of work — but the baseline model is still "many cores doing fine-grained things".

## Google TPU — coarse, drifting finer

A TPU is closer to a single very large systolic array. The programming model is XLA: you describe the computation as a dataflow graph, the compiler lays out a schedule, and the hardware executes at tile granularity. You don't write per-thread code; you write per-tensor code.

This favours fixed-shape, predictable workloads. XLA needs static shapes to schedule well; dynamic control flow costs recompilation.

The recent drift is in the opposite direction of NVIDIA's. Pallas — a JAX-level programming model — gives per-tile control over memory hierarchy and compute, bringing TPU programming closer to the kernel-level mental model CUDA programmers are used to. The hardware hasn't changed, but the programmer's view of it has.

## Trainium — tile-sized systolic plus programmable engines

A Trainium NeuronCore has a systolic Tensor Engine — TPU-like — plus three other engines: a Vector Engine for element-wise / reduction operations, a Scalar Engine for small-count arithmetic, and a GpSimd engine for RNG and bit manipulation. NKI exposes all four through a single Python-embedded DSL.

The tile shapes are fixed: 128 on the partition axis, up to 512 on the moving axis. That's TPU-like in being systolic-array-sized. But within a tile, NKI gives you per-engine control — SBUF loads explicit, PSUM accumulation explicit, engine dispatch explicit. That's NVIDIA-like in granularity.

The result is a programming model that looks coarse at the macro level (you think in tiles) and fine at the micro level (you schedule engines). It is neither pure SIMT nor pure systolic dataflow. It sits in between.

## Same workload, three idioms

A few representative primitives, expressed in each model:

| Primitive | NVIDIA SM | Google TPU | Trainium NKI |
|---|---|---|---|
| GEMM tile | Warp-level MMA (wmma / cutlass) | HLO DotGeneral on MXU | `nisa.nc_matmul` stationary + moving |
| Butterfly stage of FFT | Per-thread complex multiply-add | XLA fused reduction + permute | Per-engine: TE multiply, VE add, SBUF swap |
| Jacobi rotation | Warp of threads updates two rows | HLO scatter with permutation | TE matmul pair + VE reduction to find max |
| SpMV | Warp per row, shared indices | Challenging — sparsity hostile to XLA | DMA gather, TE matmul dense tile, DMA scatter |

None of these is a declaration that one architecture is better. The takeaway is that the idioms *translate*. A CUDA programmer reading `trnsci` source can see the mapping. A TPU programmer can see the scheduled tile structure.

## Implication for the suite

The design of each `trnsci` library reflects this middle position:

- The **API surface** is PyTorch-first, so users don't see the tiling at all most of the time. This is TPU-style — declarative.
- The **NKI kernels underneath** are hand-written per-engine. This is NVIDIA-style — explicit kernel authorship.
- The **algorithm choices** (Jacobi eigh, gather-matmul-scatter SpMM, Bluestein for non-power-of-2 FFT) favour ones that map cleanly to the fixed tile shape. This is TPU-pragmatic.

The project's bet is that this middle position is productive for scientific computing in particular — the workloads are dense and fixed-shape enough to enjoy systolic acceleration, but varied enough to need per-engine control.
