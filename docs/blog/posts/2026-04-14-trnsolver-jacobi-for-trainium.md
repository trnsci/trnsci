---
date: 2026-04-14
categories: [Deep dive, trnsolver]
comments: true
---

# trnsolver: Jacobi for Trainium — when the hardware inverts the algorithm choice

Symmetric `eigh` on Trainium wants **Jacobi**, not Householder-QR — even though Householder has the better asymptotic FLOP count and is what every numerical-linear-algebra textbook reaches for. The inversion doesn't sit in the arithmetic; it sits in the 128-partition Tensor Engine tile and NKI 0.3.0's per-traced-graph compile model. This post is how trnsolver's Phase 1 got there, what the hardware kept telling it, and which blind alleys it walked first.

<!-- more -->

## The problem

Symmetric eigendecomposition is the hot loop inside `FC = SCε` — the SCF step in quantum chemistry — and the first place trnsolver has to own the NKI backend end-to-end. Trainium ships no LAPACK. The naive port would grab a Householder-based symmetric eigensolver from scipy or from a textbook, wire it through NKI kernels, tune for tile sizes, and ship it. That matches cuSOLVER's shape: Householder tridiagonalization followed by implicit-shift QR on the tridiagonal.

The problem with that path isn't that it's slow. Every architecturally-tempting move the textbook implies runs into friction with NKI 0.3.0 at the kernel-dispatch layer. The rest of this post is what trnsolver learned trying.

## What the architecture suggests

Start from the 128×128 partition tile, set aside all existing algorithms, and ask: what dense-eigendecomposition moves does this hardware actually make cheap?

The Tensor Engine is a systolic array with a stationary operand and a streaming operand. It's built to reuse one matrix across many applications of another. Its natural operation is a GEMM with at least one operand resident in SBUF across many invocations. Its partition dim tops out at 128 (`PMAX`), so anything expressing parallelism on that axis has to fit that budget. PSUM accumulates in FP32 regardless of input precision — mixed precision is free at the matmul boundary.

Two eigendecomposition primitives look attractive against that shape:

1. **Householder reflector** `I − 2 v vᵀ`. The update `A ← A − 2 v (vᵀA)` is a rank-1 outer product. Stationary v, streaming A. PSUM accumulates the sum. This looks like exactly what the Tensor Engine was built for.

2. **Givens rotation** on a disjoint pair of rows/columns `(p, q)`. The update is a 2×2 rotation matrix applied to two n-element rows. In a batched Jacobi sweep, n/2 rotations on disjoint pairs commute; they can apply in parallel. A single sweep round maps to one NKI kernel call that rotates n/2 pairs at strided positions.

Naively, Householder wins: rank-1 matches the systolic array's dataflow, and asymptotic FLOPs for symmetric eigh via Householder-QR beat Jacobi's O(n³) per sweep × O(n) sweeps.

The inversion happens at the kernel-boundary layer. Householder-QR for a symmetric matrix is inherently **serial**: each reflector `v_k` is computed from `A`'s state after `v_{k-1}` has been applied. The scalar quantities — the Householder norm, β = 2/(vᵀv), γ for the rank-2 update — are tiny values that feed back into big tensor operations. On NKI 0.3.0 that scalar dataflow has to traverse the kernel boundary, and every such traversal is a new traced graph for the XLA layer underneath NKI. Every new traced graph is a new NEFF compile.

Jacobi's sweep round, by contrast, is **data-parallel at the granularity of a single kernel call**. Compute all n/2 rotation angles from `D`'s current state in one pass. Apply all n/2 rotations in a second kernel call. No per-rotation scalar dataflow across the kernel boundary. The Brent–Luk round-robin schedule fixes the pairing for each of the n-1 rounds per sweep; those permutations are pure combinatorics — int64 tables computed once on the host and reused forever.

The 128-partition tile doesn't care about rank-1 vs rank-2 FLOP arithmetic at this scale. It cares about **dispatch granularity**. Jacobi fits the dispatch budget. Householder, applied step by step, doesn't.

## The approach

trnsolver's Phase 1 `eigh` is batched-sweep parallel Jacobi driven from the host:

- **Brent–Luk round schedule** computed once on the host (`trnsolver/_brent_luk.py`). For even matrix size n, this returns `(n-1, n)` int64 permutations. Applying permutation `perms[r]` to D's rows and columns lands the pair set for round r at strided positions `(0,1), (2,3), …, (n-2, n-1)`.

- **`rotate_pairs_kernel`** in `trnsolver/nki/dispatch.py` — one NKI kernel that rotates n/2 disjoint row pairs at strided positions. Signature `(even, odd, c, s) → (new_even, new_odd)`, with `even, odd : (n/2, n)` and `c, s : (n/2, 1)`. Partition dim is n/2, free dim is n. For n ≤ 256 the whole thing fits a single partition tile; NKI's compile graph is stable per `(n/2, n, dtype)`.

- **Host driver** `_jacobi_eigh_nki` in `trnsolver/eigen.py`. Per round: permute D and V into strided-pair layout, compute per-pair (c, s) from the current D diagonals and off-diagonals via a host helper (`_rotation_angles_strided`), call the kernel three times (D rows, D columns, V columns), apply a small diagonal-block fixup where row-and-column rotations double-touch the 2×2 block on the diagonal.

- **Simulator-first development**. `nki.simulate(rotate_pairs_kernel)(numpy_args)` runs the kernel on CPU with no NEFF compile and no XLA tracing. Seconds per iteration on `ubuntu-latest` CI — the inner loop for kernel design.

## Implementation

The kernel is short enough to inline:

```python
@nki.jit
def rotate_pairs_kernel(even, odd, c, s):
    """Rotate two stacked tiles by a per-row Givens rotation.

    even, odd : (half, n)  — the two rows of each pair
    c, s      : (half, 1)  — per-pair cosine, sine
    returns new_even, new_odd — both (half, n)
    """
    half, n = even.shape
    new_even = nl.ndarray((half, n), dtype=even.dtype, buffer=nl.shared_hbm)
    new_odd  = nl.ndarray((half, n), dtype=even.dtype, buffer=nl.shared_hbm)

    e = nl.load(even[0:half, 0:n])
    o = nl.load(odd[0:half, 0:n])
    c_tile = nl.load(c[0:half, 0:1])
    s_tile = nl.load(s[0:half, 0:1])
    neg_s  = nl.negative(s_tile)

    ne = nl.add(nl.multiply(e, c_tile), nl.multiply(o, neg_s))
    no = nl.add(nl.multiply(e, s_tile), nl.multiply(o, c_tile))

    nl.store(new_even[0:half, 0:n], value=ne)
    nl.store(new_odd[0:half, 0:n],  value=no)
    return new_even, new_odd
```

Two loads, four Vector-Engine element-wise ops, two stores. `(c, s)` broadcasts across the free dim. The Vector Engine is doing all the math in Phase 1 — the Tensor Engine reformulation where the 2×2 rotation block is stationary for a `nisa.nc_matmul` is the obvious next lever, and the one that finally justifies calling this "a rank-2 matmul" at the hardware layer. Phase 1's simpler form validates correctness first.

The host driver's inner loop:

```python
for r in range(n - 1):
    perm = perms[r]                              # precomputed int64
    D = D[perm][:, perm]                         # row+col permute
    V = V[:, perm]

    cs = _rotation_angles_strided(D)             # (n/2, 2)
    c_col = cs[:, 0:1].contiguous()
    s_col = cs[:, 1:2].contiguous()

    D_even, D_odd = D[idx_p, :], D[idx_q, :]
    D_even, D_odd = rotate_pairs_kernel(D_even, D_odd, c_col, s_col)
    # store back, repeat for D columns and V columns
```

`_rotation_angles_strided` is a host function: five element-wise torch ops on the strided pair diagonals. It could live in the kernel, but keeping it on the host removes a would-be scalar-intermediate dataflow from the NKI-side traced graph. Same reasoning that sank Householder.

## What didn't work

Householder-QR was the first path attempted. The simulator-side implementation landed cleanly and passed 18 correctness tests against `torch.linalg.eigh` at rtol=1e-3 for n ∈ {8, 16, 32, 64, 128}. On `ubuntu-latest`, the full suite ran in 2.67 seconds.

The same code running against real trn1 hardware logged **303 separate NEFF compile workdirs** inside `/tmp/ubuntu/neuroncc_compile_workdir/` during a single test run before the run was cancelled. The traced XLA graph changed on every outer-loop iteration because `A_work` had a different computation history per Householder step — it had been mutated by the previous rank-2 update — and NKI compiles per traced graph, not per kernel signature. The simulator didn't surface this: `nki.simulate(kernel)(np_args)` runs the kernel logic directly on numpy, with no XLA layer and no trace cache, so the host-integration failure mode was invisible until hardware.

That recompile behavior is a hard architectural statement from the compiler. A solver that wants to live inside a Python loop has to present a fixed traced graph to NKI. Householder's scalar-intermediate dataflow fights that requirement head-on. Jacobi's batched-sweep form doesn't.

A handful of smaller items from the iteration:

- **Single-rotation kernel dispatch.** Before the batched-round design landed, an earlier kernel took `(p, q, c, s)` as scalar arguments and applied one Givens rotation per call. Same per-call-recompile pathology as Householder — the slicing indices `A[p:p+1, :]` changed per call, so every rotation was a fresh graph. A full Jacobi sweep of n²/2 rotations, each needing its own compile, is worse than unviable. The Brent–Luk permutation is the fix: indices become strided constants `(0,1), (2,3), …`, and the permutation is applied on the host.

- **Newton–Schulz convergence criterion.** The matrix-square-root-inverse `inv_sqrt_spd_ns` that shipped alongside eigh uses a coupled `Y, Z` iteration. The first convergence test asserted `‖Y − I‖_F < tol`, which is wrong — `Y` converges to `(A/s)^(1/2)`, not to `I`. The correct test is `‖Y·Z − I‖_F < tol`. Twenty minutes of confusion before the identity clicked.

- **NKI 0.3.0 migration specifics.** The `neuronxcc.nki.*` shim was dropped in favor of the top-level `nki` package. `nl.store(dest, expr)` needs the keyword form `nl.store(dest, value=expr)`. Python unary minus on an `InstTile` raises `TypeError`; use `nl.negative(t)`. Output tensors need explicit `buffer=nl.shared_hbm`. None of these are surprising once documented, but the failure surfaces are at kernel-compile time on hardware — only partly at simulator time.

- **Dynamic path composition inside the SSM test runner.** The runner shell-quotes a bash-in-JSON payload handed to `aws ssm send-command`. A first attempt used `NEURON_VENV=\\\$(ls -d /opt/aws_neuronx_venv_pytorch_* …)` for deferred expansion; that rendered as `\$` in the JSON, which isn't a valid JSON escape, and the command failed parameter validation before it reached the instance. The current runner hardcodes the venv path. Not a kernel story, but a toolchain-integration tax that shows up whenever automation crosses the shell/JSON/remote-shell boundary.

## Numbers

Simulator parity, `nki-simulator` CI job on `ubuntu-latest`, NKI 0.3.0 / neuronxcc 2.24.5133 from the AWS pip index:

Eigenvalue relative error vs `torch.linalg.eigh`, reconstruction error `‖V·diag(w)·Vᵀ − A‖`, and orthonormality `‖VᵀV − I‖` all hold at **rtol=1e-3** across n ∈ {8, 16, 32, 64, 128} on random symmetric inputs. Full simulator suite — 18 tests including the cases above plus the tridiag-specific correctness tests carried through the Householder pivot-and-back work — runs in **2.67 seconds wall-clock** per CI run.

Hardware `@pytest.mark.neuron` numbers up to n=512 are the next gate. The simulator isn't a substitute for hardware on perf: it skips NEFF compile, tile-capacity checks, and PSUM/SBUF latency modelling. What's measured so far is the math, not the wall-clock-on-silicon.

The disappointing number, to the extent Phase 1 has one: **per-sweep kernel-dispatch count is 3**, not 1 (D rows, D columns, V columns). That's 3× the dispatch overhead of a "one kernel per round" design. Collapsing those three calls into one kernel, plus reformulating the rotation as a stationary 2×2-matmul on the Tensor Engine, is the Phase 3 perf headline.

## What's next

The trnsolver phase trackers:

- [Phase 2 — iterative refinement + Newton-Schulz variants](https://github.com/trnsci/trnsolver/issues/27). FP32 Jacobi eigendecomposition isn't accurate enough for nanohartree-level chemistry; iterative refinement on top of the Jacobi result, using trnblas's double-double GEMM, restores precision where the motivating workloads demand it. Gated by trnblas Phase 2.

- [Phase 3 — preconditioned iterative solvers + NEFF cache reuse](https://github.com/trnsci/trnsolver/issues/28). Jacobi preconditioner for CG already landed; IC0 and SSOR are the next primitives. NEFF cache reuse in examples and benchmarks pays back across the whole suite.

- [Phase 4 — parallel Jacobi sweeps across NeuronCores](https://github.com/trnsci/trnsolver/issues/29). The disjoint rotations of a sweep round commute; they shard trivially across NeuronCores. Large-n SCF eigh is Phase 4's headline.

- [Phase 5 — trn2 rotation-block tuning](https://github.com/trnsci/trnsolver/issues/30). Trn2's larger SBUF changes the rotation-block tile size. Generation-specific kernels live here.

## Takeaway

The 128-partition Tensor Engine tile's preferences invert at the kernel-dispatch layer. Householder's rank-1 outer products look like the right hardware-native move until the serial dataflow between reflectors collides with NKI's per-traced-graph compile model. Jacobi's batched-sweep form — n/2 rotations commuting on disjoint pairs, mapping to one kernel dispatch per round — trades asymptotic FLOP count for kernel-boundary efficiency, and wins on Trainium.

The concrete lesson for anyone building a dense-linear-algebra library on NKI 0.3.0: the simulator validates kernel math; only hardware validates host-integration shape. A kernel that compiles per call on the simulator will compile per call on hardware too, and on hardware the compile cost is an order of magnitude higher. Design for a stable traced graph from the first commit, not as a later optimization.
