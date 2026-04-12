# Workflow: speech enhancement

One of the workloads that motivates `trnfft` is **speech enhancement** — taking a noisy waveform and recovering a cleaner version of the underlying speech. The standard pipeline is:

1. **STFT** the noisy waveform to a complex spectrogram.
2. **Estimate a complex mask** from the spectrogram using a small neural network.
3. **Apply the mask** (complex multiplication) to the input spectrogram.
4. **Inverse STFT** back to the time domain.

The mask is typically a *complex ratio mask* (cIRM) — it has both magnitude and phase components, and the network's task is to predict both. This is where Trainium's lack of native complex arithmetic matters: every step of the pipeline operates on complex values, and without `trnfft.ComplexTensor` the programmer ends up implementing complex multiply-and-add by hand everywhere.

## What `trnfft` covers

- `trnfft.stft`, `trnfft.istft` — windowed short-time Fourier transforms
- `trnfft.ComplexTensor` — split real/imag storage, full arithmetic
- `trnfft.nn.ComplexLinear`, `ComplexConv1d`, `ComplexBatchNorm1d`, `ComplexModReLU` — complex neural network layers for mask estimation

Together these are enough to express the full cIRM pipeline in PyTorch syntax. On CPU, it uses `torch.stft` and ordinary tensor ops; on Trainium with NKI enabled, it dispatches to the butterfly FFT and complex GEMM kernels.

## Why it matters for `trnsci`

Speech enhancement is an excellent test case for `trnfft` because:

- The shapes are regular — fixed sample rate, fixed window size, fixed hop length. That's exactly the workload pattern the Neuron compiler handles best.
- It exercises every part of the library: transforms, complex arithmetic, complex neural-network layers, and plan caching.
- It's a real deployed workload (not just a benchmark), so regressions are detectable through end-to-end speech quality metrics (PESQ, STOI).

## Example

A minimal, synthetic-signal version of the pipeline is in [`examples/speech_enhancement/`](https://github.com/trnsci/trnsci/tree/main/examples/speech_enhancement). It constructs a synthetic clean signal + noise, runs STFT → complex mask network → iSTFT, and trains the mask for a handful of steps. No real speech data is used — this is a plumbing demo for the API surface. Production cIRM training needs a real dataset (noisy/clean pairs) and a longer schedule.

## Related CUDA samples

The closest upstream NVIDIA sample is `Samples/4_CUDA_Libraries/simpleCUFFT` in the [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) repository. `simpleCUFFT` demonstrates a 1D complex FFT against a reference implementation — `trnfft` covers the same ground in `examples/nvidia_samples/simple_cufft_2d_r2c.py`. The speech-enhancement demo is the composition of `simpleCUFFT` + complex-valued neural network layers (which have no direct cuFFT analog; they would be written against cuBLAS in a CUDA implementation).
