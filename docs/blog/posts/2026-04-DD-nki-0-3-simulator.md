---
date: 2026-04-DD
authors: [scttfrdmn]
categories: [Ecosystem]
comments: true
draft: true
---

# The dev loop just got a lot shorter

Until this week, working on an NKI kernel inside any trnsci library looked roughly like this: edit the kernel, push to a branch, wait for the GitHub Actions runner to start a trn1.2xlarge instance over SSM, wait for user-data to finish, wait for the NEFF compile, finally run `pytest -m neuron`, read the result, stop the instance. Eight to twelve minutes per iteration in the best case. Longer when anything went sideways.

AWS's Neuron SDK 2.29 shipped a month ago with NKI 0.3.0 Stable. The change that matters most isn't in the SDK itself — it's in the fact that the full stack now installs cleanly on `ubuntu-latest` GitHub runners via AWS's pip index, and the CPU simulator exposed by `nki.simulate(kernel)(numpy_args)` runs the same kernels device-free. The dev loop dropped from minutes to seconds, and the cost-of-iteration barrier for contributing to a trnsci library just collapsed.

<!-- more -->

## What shipped

Three things landed together this week:

1. **The `nki.*` namespace is canonical.** The old `neuronxcc.nki.*` shim still works in 2.29 but is deprecated. Migration is mechanical — import rewrites, a few breaking signatures (`nc_matmul` keyword/in-place form, `nl.copy` semantics, dropped `nl.divide` paths), stricter partition-dim broadcasting.

2. **`nki.simulate(kernel)(numpy_args)` runs on CPU.** No device, no compile, no NEFF cache. It's a Python trace of the kernel logic, which means it catches the failures that show up at the Python layer — bad kwargs, shape mismatches, dropped operations, broadcast errors. MLIR verifier errors and actual on-hardware numerical behavior still need a real NeuronCore, so the simulator doesn't replace hardware CI — it's a fast gate in front of it.

3. **The `nki` and `neuronx-cc` wheels install on `ubuntu-latest`.** We'd assumed this wasn't possible — that the toolchain required an AMI with the Neuron driver in place — and marked it out of scope on the coordination issue. A test run on 2026-04-14 showed the wheels install cleanly from `https://pip.repos.neuron.amazonaws.com` in about three seconds. That's what unlocks the CI gate.

Put those three together and the shape of a trnsci CI pipeline becomes two layers: a fast `nki-simulator` job on `ubuntu-latest` that runs on every PR, and the existing hardware-on-SSM gate that runs on release or on demand. Between them, most classes of regression catch on the first layer.

## What it doesn't solve

We want to be specific about this — the simulator is a correctness gate at the Python layer, not a complete verifier.

- **MLIR verifier errors still need hardware.** If a kernel is syntactically valid Python but generates invalid MLIR, the simulator won't catch it. Real compile happens on device.
- **Numerical behavior differences.** FP32 accumulation on the Tensor Engine is not bit-identical to NumPy on CPU. The simulator validates *structure*, not *final numbers*. Hardware CI still owns the numerical check.
- **Performance is not modelled.** The simulator is a correctness tool. Latency, PSUM pressure, SBUF occupancy, DMA overlap — none of these come from `nki.simulate`.
- **Device-free NEFF compile.** NKI 0.3.0 doesn't expose one. `nki.baremetal` requires a device; `nki.simulate` skips compile entirely. Until AWS ships a standalone cross-compile entry point, the full CI gate is "simulator on `ubuntu-latest`" + "hardware on SSM" with nothing in between.

So the honest framing: the dev loop is dramatically faster, the coverage is good, but we're not hardware-free. Libraries that require per-kernel hardware validation still do.

## What this means for the suite

Adoption is underway across the six libraries this week. trnblas shipped the reference implementation (commits [`f24993b`](https://github.com/trnsci/trnblas/commit/f24993b) for the dispatch pattern, [`77eeb82`](https://github.com/trnsci/trnblas/commit/77eeb82) for the CI job). The other five — trnfft, trnrand, trnsolver, trnsparse, trntensor — are tracking their adoption in sister issues under the [suite coordination issue](https://github.com/trnsci/trnsci/issues/5).

Once adoption is complete across all six, a contributor can make NKI kernel changes to any library without any AWS account, any trn1 instance, or any SDK install on their laptop. The `[dev]` extras pull in the simulator; `pytest -m nki_simulator` runs locally; a PR opens a simulator-gated run on GitHub Actions. That's the threshold where NKI kernel contribution goes from "expert-only" to "anyone who knows Python and a little linear algebra."

## A thank-you and a small ask

This landed because the AWS Neuron team took the simulator seriously as a first-class feature rather than a debug tool, and because they published the `nki` wheel to the pip index in a form that works on generic Linux runners. Both of those were real decisions that made this week possible.

The small ask: a device-free NEFF compile entry point would let us add a third gate (shape/MLIR verification on `ubuntu-latest` with no device), filling the one gap in the current pipeline. If that's on the roadmap, it's worth calling out. If it isn't, that's a concrete request.

---

*Draft note: this post is sitting behind a `draft: true` frontmatter flag and on the `blog/nki-0-3-simulator-milestone` branch. Flip `draft` to `false` and merge when adoption lands across all five sister libraries. Replace the `YYYY-MM-DD` in the date / filename with the actual publish date.*
