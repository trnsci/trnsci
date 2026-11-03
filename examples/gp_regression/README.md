# Exact Gaussian-process regression, matrix-free

Milestone 1 of [trnsci/trnsci#109](https://github.com/trnsci/trnsci/issues/109):
the posterior mean of an exact GP (RBF kernel, Gaussian noise, fixed
hyperparameters) at n up to 10⁶, d = 3, without forming the n × n kernel
matrix and without a low-rank or inducing-point approximation of the model.

| piece | library |
|---|---|
| `KernelOperator`: (K + σ²I)·V in one fused launch, K never formed | trnsparse (`kernel_matmul`, trnsci/trnsparse#93) |
| block CG over [y, z₁ … z₃₂], one operator application per iteration | trnsolver (`cg_block`, trnsci/trnsolver#97) |
| pivoted-Cholesky preconditioner from the kernel's diagonal and rows | trnsolver (`pivoted_lowrank_preconditioner`) |
| fp64 iterative refinement where an fp64 host operator is affordable | trnsolver (`cg_block_ir`) |

The 32 random-sign probe solves ride along in the same launches; they are
what the log-marginal-likelihood estimate of milestone 3 needs.

Requires trnsparse ≥ 0.14 and trnsolver ≥ 0.22, the first releases with
`KernelOperator` and `cg_block` (`pip install "trnsci[sparse,solver]"`).

## Files

- `gp_regression.py`: synthetic field with known hyperparameters (random
  Fourier features of the RBF kernel) and `posterior_mean(...)`. CLI:
  `python gp_regression.py --n 1000000 --device`.
- `validate.py`: against exact fp64 scikit-learn and `scipy.linalg.cho_solve`
  at n = 2×10⁴ (requirement: relative error ≤ 1e-4), and gpytorch's CG exact
  GP on the CPU at n = 10⁵.
- `bench.py`: time-to-solution with `--device`, against the three CPU
  baselines: this algorithm on 2 threads, on all vCPUs, and gpytorch.
  CPU cells above `--cpu-full-max` are projections from the measured CPU
  rate, marked `~`.

## Precision

Distances and kernel entries are fp32 on the device, with inputs centred on
the host. CG iterates in fp32; the solver reports its true residual, and
the example checks an fp64 residual on sampled rows (all rows for n ≤ 2048).
`--refine` adds fp64 iterative refinement against an fp64 host operator.
