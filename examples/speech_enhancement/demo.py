"""Synthetic complex-ratio-mask (cIRM) speech-enhancement plumbing demo.

Shows how trnfft's STFT + ComplexTensor + complex NN layers compose into an
end-to-end speech-enhancement pipeline. The signal is synthetic — a mix of
tones at known frequencies plus white noise — so no real dataset is required
to exercise the API surface.

Run:

    python examples/speech_enhancement/demo.py --demo
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import trnfft
from trnfft import ComplexTensor
from trnfft.nn import ComplexLinear, ComplexModReLU


def synthesize(*, sr: int, duration: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (noisy, clean) signals of shape (samples,).

    clean = sum of a few sinusoids; noisy = clean + white noise. The network's
    task is to estimate a complex ratio mask that recovers clean from noisy in
    the STFT domain.
    """
    torch.manual_seed(seed)
    n = int(sr * duration)
    t = torch.arange(n, dtype=torch.float32) / sr
    clean = (
        0.4 * torch.sin(2 * torch.pi * 220.0 * t)
        + 0.3 * torch.sin(2 * torch.pi * 440.0 * t)
        + 0.2 * torch.sin(2 * torch.pi * 880.0 * t)
    )
    noise = 0.3 * torch.randn(n)
    noisy = clean + noise
    return noisy, clean


class CIRMNet(nn.Module):
    """Tiny complex-valued MLP that predicts a cIRM from a noisy spectrogram."""

    def __init__(self, freq_bins: int, hidden: int = 64):
        super().__init__()
        self.fc1 = ComplexLinear(freq_bins, hidden)
        self.act = ComplexModReLU(hidden)
        self.fc2 = ComplexLinear(hidden, freq_bins)

    def forward(self, spec: ComplexTensor) -> ComplexTensor:
        """spec: ComplexTensor of shape (frames, freq_bins). Returns mask of the same shape."""
        h = self.fc1(spec)
        h = self.act(h)
        return self.fc2(h)


def train_one_step(
    net: CIRMNet, noisy_spec: ComplexTensor, clean_spec: ComplexTensor, opt: torch.optim.Optimizer
) -> float:
    """One gradient step. Loss is MSE on (real, imag) of the masked spectrogram vs clean."""
    mask = net(noisy_spec)
    predicted = noisy_spec * mask
    loss = (
        (predicted.real - clean_spec.real).pow(2).mean()
        + (predicted.imag - clean_spec.imag).pow(2).mean()
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=1.0, help="seconds of synthetic audio")
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    noisy, clean = synthesize(sr=args.sr, duration=args.duration, seed=args.seed)

    # Transform to the STFT domain
    noisy_spec = trnfft.stft(noisy, n_fft=args.n_fft, hop_length=args.hop)
    clean_spec = trnfft.stft(clean, n_fft=args.n_fft, hop_length=args.hop)

    # Spec shape is (freq_bins, frames); the network wants (frames, freq_bins).
    def as_frames_first(spec: ComplexTensor) -> ComplexTensor:
        return ComplexTensor(spec.real.T, spec.imag.T)

    noisy_frames = as_frames_first(noisy_spec)
    clean_frames = as_frames_first(clean_spec)
    freq_bins = noisy_frames.real.shape[1]

    net = CIRMNet(freq_bins=freq_bins, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    print(f"sr={args.sr} duration={args.duration}s n_fft={args.n_fft} hop={args.hop}")
    print(f"spec shape: {tuple(noisy_frames.real.shape)} (frames, freq_bins)")
    print()

    initial_loss = None
    for step in range(args.steps):
        loss = train_one_step(net, noisy_frames, clean_frames, opt)
        if step == 0:
            initial_loss = loss
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"  step {step:>4}: loss={loss:.6f}")

    print()
    print(f"initial loss: {initial_loss:.6f}")
    print(f"final loss:   {loss:.6f}")
    print(f"status:       {'OK (loss decreased)' if loss < initial_loss else 'FAIL (no improvement)'}")


if __name__ == "__main__":
    main()
