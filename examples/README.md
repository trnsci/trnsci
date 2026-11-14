# trnsci examples

Four kinds of examples live here:

## [`nvidia_samples/`](nvidia_samples/)

Python ports of canonical NVIDIA CUDA samples. Each script cites the upstream sample by URL and implements the same algorithm against `trnsci` APIs. Use these if you already know the CUDA library and want to see the equivalent `trnsci` call pattern.

See [`nvidia_samples/README.md`](nvidia_samples/README.md) for the mapping table.

## [`quantum_chemistry/`](quantum_chemistry/)

Density-fitted MP2 plumbing demo composing `trnrand` + `trnsparse` + `trnsolver` + `trnblas` + `trntensor` end-to-end, plus `df_mp2_profile.py`, a per-stage profile of the same pipeline across a ladder of sizes. This is the cross-library integration test — the single workload that touches the most libraries in the suite.

## [`gp_regression/`](gp_regression/)

Exact Gaussian-process regression, matrix-free (trnsci/trnsci#109): the posterior mean at n up to 10⁶ with the fused kernel operator from `trnsparse` (`KernelOperator`, K never formed) and block CG from `trnsolver` (`cg_block`, one operator launch per iteration for all right-hand sides, pivoted-Cholesky preconditioner). Validated against exact fp64 scikit-learn / scipy and against gpytorch; `bench.py --device` gives the time-to-solution table against three CPU baselines.

## [`speech_enhancement/`](speech_enhancement/)

Synthetic complex-ratio-mask demo exercising `trnfft` STFT + complex NN layers + iSTFT. Shows the full signal-processing pipeline for a complex-valued workload.

## [`genomics/deepvariant/`](genomics/deepvariant/)

Application exemplar for trnsci/trnsci#114: DeepVariant 1.9 `call_variants` (InceptionV3-class CNN inference on pileup tensors) rebuilt in PyTorch from the public WGS model, checked against TensorFlow on CPU, and run through a batched probe on `cpu` or a TorchNeuron `neuron` device. It uses none of the six libraries.

## Reverse direction

[`reverse_port_note.md`](reverse_port_note.md) — a short note on `trnsci` idioms a CUDA programmer might borrow in the other direction (stationary-tile complex GEMM, gather-matmul-scatter SpMM, Jacobi eigh, first-class contraction plans).
