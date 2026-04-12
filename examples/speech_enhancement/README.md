# Example workflow: speech enhancement

Synthetic, end-to-end plumbing demo for a complex-ratio-mask (cIRM) speech-enhancement pipeline. Exercises `trnfft` end-to-end: STFT, complex neural network layers, iSTFT.

See [`../../docs/workflows/speech_enhancement.md`](../../docs/workflows/speech_enhancement.md) for workflow context.

## `demo.py`

Builds a synthetic clean + noise signal, runs it through a minimal complex-ratio-mask network, and inverts back to the time domain. Trains the mask for a handful of steps to verify that loss decreases through the complex-valued pipeline.

| Stage | API |
|---|---|
| STFT (noisy waveform → spectrogram) | `trnfft.stft` |
| Complex mask estimation | `trnfft.ComplexTensor`, `trnfft.nn.ComplexLinear`, `trnfft.nn.ComplexModReLU` |
| Apply mask (complex multiply) | `ComplexTensor.__mul__` |
| iSTFT (spectrogram → waveform) | `trnfft.istft` (or `torch.istft` fallback) |

```bash
python examples/speech_enhancement/demo.py --demo
```

The signal is synthetic — this is a **plumbing** demo that verifies the API surface composes end-to-end. Production cIRM training needs a real noisy/clean-pair dataset and a much longer training schedule.
