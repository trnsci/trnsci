# Reverse port note — trnsci idioms a CUDA programmer might borrow

The Rosetta stone is bidirectional. Most traffic is CUDA → `trnsci` (someone porting a CUDA workload to Trainium), but a few idioms from the `trnsci` side are cleaner than the CUDA equivalent and worth noting in the other direction. This file documents patterns a CUDA programmer could borrow — as design inspiration, not as a runnable port.

## Complex arithmetic without a complex dtype

Trainium has no `complex64` dtype. `trnfft` handles this by storing real and imaginary parts as paired real tensors in a `ComplexTensor` wrapper, and decomposing complex multiplication into four real multiplications and two additions:

```
(a + bi)(c + di) = (ac − bd) + (ad + bc)i
```

In `trnblas.nki.dispatch`, a complex GEMM is four real GEMMs on paired real tensors, with PSUM accumulation folding the sign of the `bd` term. The pseudocode:

```python
# Inputs: A = A_re + i A_im, B = B_re + i B_im
# Output: C = C_re + i C_im

C_re = nki.matmul(A_re, B_re) - nki.matmul(A_im, B_im)  # both in PSUM
C_im = nki.matmul(A_re, B_im) + nki.matmul(A_im, B_re)
```

What's interesting here is the **stationary-tile reuse** optimization: `A_re` and `A_im` are each loaded once into SBUF and reused across both output matmuls. This gives two SBUF loads per complex operand instead of four (the naive pattern).

### The CUDA analog

CUDA has native `cuComplex` / `float2`, so the four-real-GEMM decomposition isn't forced. But on architectures without hardware complex support (or when implementing a complex GEMM in Tensor Cores, which operate on real matrices), the same stationary-tile reuse pattern applies. A `cutlass`-based complex GEMM could borrow the "load each real operand once, dispatch two matmuls, accumulate in PSUM-equivalent registers" layout directly.

## Gather-matmul-scatter for SpMM

`trnsparse.spmm` uses a three-step pattern for sparse-dense matrix multiplication:

1. DMA engine gathers non-zero columns of `A` into a dense SBUF tile.
2. Tensor Engine does a dense GEMM against the corresponding rows of `B`.
3. DMA engine scatters the result back to the output rows.

This is similar to how `cuSPARSE` handles SpMM internally on newer architectures, but it's expressed more explicitly in NKI because the DMA engine is a first-class resource. A CUDA programmer writing a custom SpMM in Tensor Cores could structure it the same way — with `cp.async` as the gather, the Tensor Core MMA as the middle step, and another `cp.async` as the scatter.

## Jacobi eigh for fixed-tile hardware

`trnsolver.eigh` uses Jacobi rotations instead of the Householder + QR path that `cuSOLVER` uses internally. Jacobi is O(n³) per sweep with O(n) sweeps, so it loses asymptotically to Householder + QR. But:

- Each rotation is a fixed-size matmul on a 2-row block — maps cleanly to a 128-partition tile.
- Rotations on disjoint pairs commute, so a sweep is embarrassingly parallel.
- No growing reflector chains to manage.

For fixed-tile systolic hardware (Trainium, TPU, Tensor Cores used at tile granularity), Jacobi is often competitive even though the FLOP count is higher. A CUDA programmer writing an eigendecomposition directly against Tensor Cores might find Jacobi the more natural fit than the cuSOLVER-internal path.

## Contraction planning as a first-class object

`trntensor.plan_contraction(expr, *operands)` returns a `ContractionPlan` with a `.dispatch` choice (matmul / bmm / einsum / nki), a FLOPs estimate, and any reshape/transpose preamble. The plan is reusable — you pay the analysis cost once and re-execute against new operand tensors with the same shape.

`cuTENSOR` has a similar concept (`cutensorContractionPlan_t`), but it's often hidden behind convenience wrappers. Surfacing the plan as a user-visible Python object makes it easier to introspect dispatch choices, compare alternatives, and use FLOPs estimates to decide whether to distribute a contraction across devices. A Python binding to `cuTENSOR` could adopt the same API shape.

## Takeaway

These aren't performance claims. `cuFFT` beats `trnfft` on every axis that matters for a modern NVIDIA GPU today, and probably will continue to. The interesting direction is the **idiom** — fixed-tile-friendly algorithms, explicit engine scheduling, first-class contraction plans. These patterns are migrating into CUDA (Hopper TMA, Blackwell tile APIs, CUDA Graphs), and `trnsci` is one place to see them written in Python against a hardware target that forces them from the start.
