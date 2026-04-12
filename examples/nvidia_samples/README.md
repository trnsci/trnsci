# NVIDIA sample ports

Each script in this directory is a Python port of a specific NVIDIA CUDA sample to `trnsci` APIs. The goal is one-to-one correspondence: same algorithm, same problem size, same reference check — different library surface.

| This port | Upstream sample | Upstream URL |
|---|---|---|
| [`simple_cufft_2d_r2c.py`](simple_cufft_2d_r2c.py) | `CUDALibrarySamples/cuFFT/2d_r2c_c2r` | https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuFFT |
| [`batch_cublas.py`](batch_cublas.py) | `cuda-samples/Samples/4_CUDA_Libraries/batchCUBLAS` | https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/batchCUBLAS |
| [`mc_estimate_pi.py`](mc_estimate_pi.py) | `cuda-samples/Samples/5_Domain_Specific/MC_EstimatePiP` and `MC_EstimatePiQ` | https://github.com/NVIDIA/cuda-samples/tree/master/Samples/5_Domain_Specific/MC_EstimatePiP |
| [`cusolver_syevd.py`](cusolver_syevd.py) | `CUDALibrarySamples/cuSOLVER/syevd` | https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSOLVER/syevd |
| [`conjugate_gradient.py`](conjugate_gradient.py) | `cuda-samples/Samples/4_CUDA_Libraries/conjugateGradient` | https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/conjugateGradient |
| [`cutensor_contraction.py`](cutensor_contraction.py) | `CUDALibrarySamples/cuTENSOR/contraction` | https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR/contraction |

Each port:

- Runs on CPU via the PyTorch fallback without any special setup.
- Prints a numerical check against a reference (either `torch` / `numpy` or an analytic value).
- Accepts `--demo` to run with the upstream sample's default parameters.

See [`../reverse_port_note.md`](../reverse_port_note.md) for patterns that go the other direction — `trnsci` idioms that a CUDA programmer might borrow back.
