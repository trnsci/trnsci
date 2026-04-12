# trnsci examples

Three kinds of examples live here:

## [`nvidia_samples/`](nvidia_samples/)

Python ports of canonical NVIDIA CUDA samples. Each script cites the upstream sample by URL and implements the same algorithm against `trnsci` APIs. Use these if you already know the CUDA library and want to see the equivalent `trnsci` call pattern.

See [`nvidia_samples/README.md`](nvidia_samples/README.md) for the mapping table.

## [`quantum_chemistry/`](quantum_chemistry/)

Density-fitted MP2 plumbing demo composing `trnrand` + `trnsolver` + `trnblas` + `trntensor` end-to-end. This is the cross-library integration test — the single workload that touches the most libraries in the suite.

## [`speech_enhancement/`](speech_enhancement/)

Synthetic complex-ratio-mask demo exercising `trnfft` STFT + complex NN layers + iSTFT. Shows the full signal-processing pipeline for a complex-valued workload.

## Reverse direction

[`reverse_port_note.md`](reverse_port_note.md) — a short note on `trnsci` idioms a CUDA programmer might borrow in the other direction (stationary-tile complex GEMM, gather-matmul-scatter SpMM, Jacobi eigh, first-class contraction plans).
