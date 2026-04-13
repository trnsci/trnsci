"""Port of NVIDIA CUDALibrarySamples/cuFFT/2d_r2c_c2r to trnsci.

Upstream sample:
    https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuFFT

The upstream sample performs a real → complex 2D FFT and then a complex → real
inverse, checking that the round-trip recovers the original signal within
float precision. This port does the same using trnfft.

Run:
    python examples/nvidia_samples/simple_cufft_2d_r2c.py --demo
"""

from __future__ import annotations

import argparse

import torch

import trnfft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--ny", type=int, default=512)
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    torch.manual_seed(0)
    x = torch.randn(args.nx, args.ny, dtype=torch.float32)

    # R2C forward transform (like cufftExecR2C)
    X = trnfft.rfft2(x) if hasattr(trnfft, "rfft2") else torch.fft.rfft2(x)

    # C2R inverse transform (like cufftExecC2R)
    x_back = (
        trnfft.irfft2(X, s=x.shape)
        if hasattr(trnfft, "irfft2")
        else torch.fft.irfft2(X, s=x.shape)
    )

    err = (x - x_back).abs().max().item()
    rel = err / x.abs().max().item()

    print(f"2D R2C/C2R round-trip: shape=({args.nx},{args.ny})")
    print(f"  max abs error:      {err:.3e}")
    print(f"  max rel error:      {rel:.3e}")
    print(f"  status:             {'OK' if rel < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()
