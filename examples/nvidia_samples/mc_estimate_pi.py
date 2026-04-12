"""Port of NVIDIA cuda-samples MC_EstimatePiP and MC_EstimatePiQ to trnsci.

Upstream samples:
    https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/MC_EstimatePiP
    https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/MC_EstimatePiQ

The upstream samples estimate pi by Monte Carlo integration of the unit circle
within a 2x2 square. MC_EstimatePiP uses pseudo-random Philox; MC_EstimatePiQ
uses quasi-random Sobol. This port does both using trnrand and shows how the
QMC sequence converges faster than the PRNG for this integration.

Run:
    python examples/nvidia_samples/mc_estimate_pi.py --demo
"""

from __future__ import annotations

import argparse
import math

import torch

import trnrand


def mc_pi_pseudorandom(n: int, seed: int) -> float:
    """Philox-based MC estimate of pi. Analog of MC_EstimatePiP."""
    g = trnrand.manual_seed(seed)
    x = trnrand.uniform(n, generator=g)
    y = trnrand.uniform(n, generator=g)
    inside = ((x * x + y * y) <= 1.0).sum().item()
    return 4.0 * inside / n


def qmc_pi_sobol(n: int) -> float:
    """Sobol-based QMC estimate of pi. Analog of MC_EstimatePiQ."""
    s = trnrand.sobol(d=2, n=n)
    x, y = s[:, 0], s[:, 1]
    inside = ((x * x + y * y) <= 1.0).sum().item()
    return 4.0 * inside / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        return

    print(f"{'N':>10} {'MC (Philox)':>14} {'err':>10} {'QMC (Sobol)':>14} {'err':>10}")
    for n in (1_000, 10_000, 100_000, 1_000_000):
        mc = mc_pi_pseudorandom(n, args.seed)
        qmc = qmc_pi_sobol(n)
        print(
            f"{n:>10d} {mc:>14.6f} {abs(mc - math.pi):>10.2e} "
            f"{qmc:>14.6f} {abs(qmc - math.pi):>10.2e}"
        )


if __name__ == "__main__":
    main()
