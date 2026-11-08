# Exemplars

The suite's libraries are judged by whole workloads on one Trainium chip, not by per-kernel speedups. Each exemplar below is an application that composes the libraries end to end, validated against an established CPU code, with its own issue, a results log on GitHub and, where one exists, a blog post. The candidates and the ranking rubric are in [trnsci#112](https://github.com/trnsci/trnsci/issues/112).

The rubric comes from the first large-basis SCF result. A workload fits Trainium when:

1. a large operand stays resident in HBM and is reused for many iterations;
2. the hot loop is Tensor Engine-shaped contractions, accumulated in fp32 PSUM;
3. fp32 is enough for the iterate, and the fp64 check is cheap;
4. it needs few, large launches, so the per-call dispatch floor is noise.

**How to read the numbers.** All device runs are on a trn2.3xlarge (one Trn2 chip, four logical cores at LNC=2) with public Neuron SDK 2.33. A "2-thread" CPU baseline is the same algorithm on the same box with the host pinned to 2 vCPUs. It is like-for-like but flatters the device by roughly the host's vCPU count over 2, so each exemplar also reports all-vCPU and reference-code baselines where they were measured. GPU figures appear only when measured; none have been yet. The precision rules behind every accuracy figure are on [precision and platform limits](precision_and_platform_limits.md).

| Exemplar | Libraries | State | Headline |
|---|---|---|---|
| [GP regression](#gp) | trnsparse, trnsolver | milestone 1 done | n = 10⁶ posterior mean in 675 s on one chip |
| [Spherical harmonic transform](#sht) | trnfft | milestone 1 done (scalar SHT) | forward 8.5–28× fp64 ducc0 on the same box |
| [Large-basis DF-RHF/MP2](#large-basis) | trnsolver, trntensor, trnblas | ladder to 3508 AOs done | 0.33 s per SCF iteration at 1658 AOs, against 28.7 s on 12 host vCPUs |
| [DeepVariant `call_variants`](#deepvariant) | none of the six (PyTorch on the native device) | chr20 slice done | fp32 calls identical to CPU DeepVariant; 609 examples/s on one core |
| [Plane-wave DFT](#plane-wave) | trnfft, trnsolver, trnblas | planned | — |

## Exact Gaussian-process regression at n = 10⁶ {#gp}

Exact GP regression needs solves with K + σ²I, and at n = 10⁶ K is 4 TB in fp32, so the exemplar never forms it. trnsparse's `KernelOperator` computes each 128 × 512 distance tile as one Tensor Engine matmul into PSUM, applies `exp` on the Scalar Engine straight from PSUM, and multiplies by V in place. trnsolver's `cg_block` solves 33 right-hand sides per operator launch (the data plus 32 probes for a later log-determinant), with a pivoted-Cholesky preconditioner. There is no low-rank or inducing-point approximation of the model.

**Headline:** the n = 10⁶ (d = 3, RBF) posterior mean converges in **675 s** on one chip with 4 logical cores: 50 CG iterations at 7.27 s per operator apply ([window-5 run](https://github.com/trnsci/trnsci/issues/109#issuecomment-6708645851), trnsparse 0.15.0). The blog post's 813 s predates trnsparse 0.15.0's kernel changes. At n = 2×10⁴, where an exact reference exists, the device posterior mean agrees with fp64 scipy and scikit-learn to 4.5e-5. At 10⁶ the fp64 check covers sampled rows only (residual 1.0e-3).

Open caveats: the all-vCPU and gpytorch baselines at n = 10⁶ were not run; CG reductions differ between LNC=1 and LNC=2 ([trnsolver#128](https://github.com/trnsci/trnsolver/issues/128)); the preconditioner build is host-fp64-bound ([trnsci#129](https://github.com/trnsci/trnsci/issues/129)). n = 10⁷ needs a trn2.48xlarge ([trnsparse#116](https://github.com/trnsci/trnsparse/issues/116)).

Issue [trnsci#109](https://github.com/trnsci/trnsci/issues/109) · post [a million-point Gaussian process on one Trn2 chip](blog/posts/2026-11-04-gp-million-points.md) · code [`examples/gp_regression/`](https://github.com/trnsci/trnsci/tree/main/examples/gp_regression)

## Spherical harmonic transform {#sht}

`trnfft.sht` is a forward and inverse scalar SHT (L = 511 and 1023, up to 1024 fields), the kernel of spectral-transform weather and climate models. The longitude stage is trnfft's real FFT; the latitude stage is a per-m GEMM against Legendre tables that stay stationary in SBUF while every field streams through the Tensor Engine into fp32 PSUM. All-zero polar tiles are skipped through a static mask, and the table is real, so no complex arithmetic reaches the device.

**Headline:** the forward transform is **8.5–28× faster** than fp64 ducc0 on the same box. At L = 511, F = 1024 it takes 297 ms, against 7.2 s for ducc0 on 2 threads and 3.2 s on all 12 vCPUs ([results](https://github.com/trnsci/trnfft/pull/152#issuecomment-6613524387)). Error against fp64 ducc0 is about 1.7e-7. The inverse was 26× slower than the forward in the blog post's measurements; trnfft 0.29.0 removed the `torch.flip` behind that, and the inverse at F = 1024 is now 280 ms, below the forward ([CHANGELOG](https://github.com/trnsci/trnfft/blob/main/CHANGELOG.md)). The honest qualifier: a padded plain-PyTorch latitude stage gets within 1.2–2.2× of the NKI kernel at large batch and beats it at one field. What plain PyTorch can't do on the device is the FFT.

Next: the vector SHT, then the shallow-water model and ensembles, which don't exist yet.

Issue [trnsci#111](https://github.com/trnsci/trnsci/issues/111) · post [spherical harmonic transforms, and how close plain PyTorch gets](blog/posts/2026-11-04-trnfft-spherical-harmonics.md) · code [`trnfft.sht`](trnfft/api/sht.md) and [`scripts/bench_sht.py`](https://github.com/trnsci/trnfft/blob/main/scripts/bench_sht.py)

## Large-basis DF-RHF and DF-MP2 to 3508 AOs {#large-basis}

Density-fitted RHF plus MP2 runs on one trn2.3xlarge from about 500 to 3508 basis functions, including systems whose fp32 integrals (up to 207 GB) exceed the chip's 96 GiB of HBM and the host's RAM. trntensor's `AuxStream` keeps what fits resident and streams the rest over the aux index; trnsolver's drivers run the SCF loop in fp32 with a noise-aware convergence test and evaluate the energy in float64. Each rung was bound by a different resource: Tensor Engine compute while resident, the host link, then the disk. Since window 5 the 3508-AO rung (alkane60 / cc-pVTZ) runs SCF, AO→MO and MP2 on one logical core with no host driver patches.

**Headline:** at 1658 AOs (C40H82 / aug-cc-pVDZ, integrals resident) an SCF iteration takes **0.33 s**, against 150 s for the same fp32 algorithm on 2 pinned host threads and 28.7 s on all 12 vCPUs. E_HF is within 1.7e-5 Eh of PySCF on every rung PySCF could run. At 3508 AOs, J/K takes 212–235 s per iteration, bound by the instance NVMe. There the device MP2 is about 1e-8 Eh from fp64 on the checked subset of pairs, inside the suite's 1e-6 Eh gate. The longest stage is now the host's fp64 canonical Fock, 2255 s ([trnsci#129](https://github.com/trnsci/trnsci/issues/129)).

Issue [trnsci#115](https://github.com/trnsci/trnsci/issues/115) ([window-5 summary](https://github.com/trnsci/trnsci/issues/115#issuecomment-6711939379)) · posts [DF-RHF and DF-MP2 from 500 to 3508 AOs](blog/posts/2026-11-05-dfmp2-ladder-3508.md) and [3508-AO DF-MP2 at the fp32 floor](blog/posts/2026-11-08-dfmp2-fp32-floor.md) · code trnsolver [`examples/scf_df_rhf.py`](https://github.com/trnsci/trnsolver/blob/main/examples/scf_df_rhf.py), [`examples/scf_df_mp2.py`](https://github.com/trnsci/trnsolver/blob/main/examples/scf_df_mp2.py), [`examples/df_store.py`](https://github.com/trnsci/trnsolver/blob/main/examples/df_store.py) and [`scripts/bench_scf_ladder.py`](https://github.com/trnsci/trnsolver/blob/main/scripts/bench_scf_ladder.py)

## DeepVariant `call_variants` {#deepvariant}

DeepVariant's variant-calling CNN (release 1.9, WGS model) rebuilt in PyTorch from the public weights, running on one NeuronCore. It needs almost nothing from the six libraries; it is here as an application test of the native device. The device consumes the byte-identical examples that CPU `make_examples` produces, so input drift can't hide inference error. **`make_examples` and `postprocess_variants` still run on the CPU**, unchanged. The test set is HG002 chr20:10–20 Mb, 25,729 examples.

**Results parity:** in fp32 the genotype is identical to CPU DeepVariant on **100% of 23,991 VCF records**, with no GQ or FILTER differences. Class probabilities are within 2.0e-6 of TensorFlow on the CPU, and two device runs give byte-identical VCFs ([results](https://github.com/trnsci/trnsci/issues/114#issuecomment-6614686915), [probabilities](https://github.com/trnsci/trnsci/issues/114#issuecomment-6615855576)). bf16 changes 4 genotypes (99.95% identical) and is not adopted.

**Throughput:** one NeuronCore runs **609 examples/s** in fp32 (1006 in bf16), against 207.5 for PyTorch on all 12 host vCPUs and 49.8 on 2 threads ([CPU baseline](https://github.com/trnsci/trnsci/issues/114#issuecomment-6617385771)). Batches of 256 or more don't compile for this graph on one core. The pipeline bound is `make_examples`, at 84 examples/s on 2 vCPUs, or about 53 vCPU-hours per genome. One NeuronCore keeps up with about 15 vCPUs of it, and the split-pipeline cost model is [trnsci#134](https://github.com/trnsci/trnsci/issues/134).

Issue [trnsci#114](https://github.com/trnsci/trnsci/issues/114) · no blog post yet · code in the open PR [trnsci#116](https://github.com/trnsci/trnsci/pull/116) (`examples/genomics/deepvariant/`, not merged)

## Plane-wave DFT (planned) {#plane-wave}

Plane-wave pseudopotential DFT for condensed matter, the materials-science sibling of the DF-RHF exemplar: a Γ-point, norm-conserving LDA SCF for a 512-atom silicon supercell on one logical core, then 1000 atoms, then ensembles on a trn2.48xlarge. At a few hundred atoms the step is dominated by dense contractions over the plane-wave index, which are Tensor Engine-shaped, and at Γ every subspace matrix is real, which sidesteps the device's lack of complex linear algebra.

**Status:** no code and no headline number. Milestone 1 ([trnsci#133](https://github.com/trnsci/trnsci/issues/133)) is a CPU SCF matching DFTK and Quantum ESPRESSO on Si-8 and Si-64 to 1e-5 Eh/atom per term, then Si-64 on the device, which waits on batched 3-D FFTs ([trnfft#147](https://github.com/trnsci/trnfft/issues/147)).

Issue [trnsci#110](https://github.com/trnsci/trnsci/issues/110)
