#!/usr/bin/env python
"""Spike B4 — multi-rank ``nki.collectives`` on one trn2.3xlarge (SDK 2.33, NKI 0.7.0).

Goal: find out whether an ``@nki.jit`` kernel that calls ``nki.collectives``
(``all_reduce``, ``all_gather``, ``reduce_scatter``) runs correctly across 1, 2
and 4 logical NeuronCores under TorchNeuron. This has never worked for trnsci
(May 2026: a single-rank attempt died with a tracing error), so every case is
isolated in its own cold multi-process job with a hard timeout.

Usage (from the umbrella root; the script is its own launcher and worker):

    python scripts/spike_collectives_multirank.py --world 1,2,4 2>&1 | tee ~/spike_b4.log
    python scripts/spike_collectives_multirank.py --world 2 --cases nki_all_reduce
    python scripts/spike_collectives_multirank.py --world 1,2,4 --cpu     # Mac pre-flight

Post the log (plus the per-rank logs directory it prints) to trnsci/trnsci#35.

Cases (each one = one fresh N-process job):
  torch_collectives   baseline: ``dist.all_reduce`` + ``dist.all_gather_into_tensor``
                      of a neuron tensor through the ``"neuron"`` c10d backend.
  nki_all_reduce      NKI kernel, ``ncc.all_reduce`` op=nl.add and op=nl.maximum,
                      src/dst on ``shared_hbm`` (the torch-neuronx in-tree pattern).
  nki_all_reduce_private  same, op=nl.add, src/dst on ``private_hbm``
                      (the nki-library pattern; mandatory when LNC=1).
  nki_all_gather      NKI ``ncc.all_gather`` along dim 0.
  nki_reduce_scatter  NKI ``ncc.reduce_scatter`` (op=nl.add) along dim 0.

Inputs are small exact integers stored in fp32, rank-dependent and
position-dependent, so every rank can compute the exact expected result and
the check is bit-exact (``torch.equal``), not a tolerance.

What was learned about launching ranks (public sources, read 2026-10-27)
-----------------------------------------------------------------------
* TorchNeuron ships a c10d backend named ``"neuron"``: ``import torch_neuronx``
  registers it, user code calls ``dist.init_process_group("neuron")``.
  https://github.com/aws-neuron/torch-neuronx/blob/main/docs/AI-distributed.md
  https://github.com/aws-neuron/torch-neuronx/blob/main/torch_neuronx/distributed/backend.py
* The runtime must NOT be initialized before ``init_process_group`` (backend.py
  asserts it), i.e. no neuron tensor may be created before the PG exists. One
  device per process (asserts ``_vnc_count() == 1``). The backend picks
  ``NEURON_RT_ROOT_COMM_ID`` itself (rank 0 chooses a free port, shares it via
  the c10d store) and calls an NRT barrier at PG creation.
* Core pinning: with ``LOCAL_RANK``/``LOCAL_WORLD_SIZE`` set, ``backend.py``
  ``resolve_visible_cores`` keeps a single-id ``NEURON_RT_VISIBLE_CORES`` as is,
  or indexes a list/range by local rank. AWS's own multi-rank NKI collectives
  test sets ``NEURON_RT_VISIBLE_CORES=str(core_start + local_rank)`` in each
  worker before ``init_process_group("neuron")``, then
  ``torch.neuron.set_device(rank)`` and uses ``device=f"neuron:{rank}"``; its
  driver STRIPS ``RANK/LOCAL_RANK/WORLD_SIZE/LOCAL_WORLD_SIZE/MASTER_*``,
  ``NEURON_RT_VISIBLE_CORES``, ``NEURON_RT_NUM_CORES``, ``NEURON_RT_ROOT_COMM_ID``
  from the child env ("dev pods pin a single core, but a collective torchrun
  needs nproc x LNC cores") and sets ``NEURON_LOGICAL_NC_CONFIG=2`` by default.
  Ranks are launched with ``python -m torch.distributed.run
  --nproc_per_node=N --master-addr=127.0.0.1 --master-port=P``.
  https://github.com/aws-neuron/torch-neuronx/blob/main/tests/vertical/test_dynamic_collectives.py
  This script reproduces exactly that env (RANK, LOCAL_RANK, WORLD_SIZE,
  LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT) with its own ``subprocess``
  launcher instead of torchrun, so it can hard-kill a hung job and keep going.
* Invoking a collectives kernel from torch: torch-neuronx's own in-tree
  collective kernels (``_all_reduce_max_hbm``, ``_all_to_all_v_hbm``) are called
  eagerly inside a live ``"neuron"`` PG as
  ``wrap_nki(kernel)[(int(get_logical_neuron_cores()),)](tensor, ReplicaGroup(...))``;
  ``wrap_nki`` treats a ``ReplicaGroup`` argument as a compile-time constant.
  https://github.com/aws-neuron/torch-neuronx/blob/main/torch_neuronx/distributed/ops/all_to_all_v.py
  https://github.com/aws-neuron/torch-neuronx/blob/main/torch_neuronx/python_ops/nki_kernels/all_to_all_v.py
  https://github.com/aws-neuron/torch-neuronx/blob/main/torch_neuronx/nki_hop.py
* Kernel-side rules (AWS comments in nki-library + torch-neuronx kernels):
  collectives cannot read/write kernel IO tensors -> ``dma_copy`` input into an
  internal HBM ``src``, collective ``src -> dst``, ``dma_copy`` ``dst`` to the
  returned ``shared_hbm`` output; ``name=`` is REQUIRED on collective src/dst
  (else NCC_IBIR440 DRAM allocation failure). At LNC=1 srcs must be
  ``private_hbm``; at LNC=2 shared or private both allowed (all_reduce docstring).
  Every rank must compile an identical NEFF (same shapes on every rank).
  https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/collectives/collectives.py
  https://github.com/aws-neuron/aws-neuron-sdk/blob/master/nki/api/nki/collectives/__init__.py
* AWS's torch_xla-era sample (nki-samples allgather_matmul_ring) notes that
  world_size x LNC combinations "may fail if the replica group does not map to
  a valid CPI ring topology"; world=4 at LNC=2 on trn2.3xlarge is untested by us.

Knobs (env): SPIKE_JOB_TIMEOUT (s, default 300) hard-kills a job;
SPIKE_TOTAL_BUDGET (s, default 2100) skips remaining jobs; SPIKE_PEER_GRACE
(s, default 60) kills surviving ranks after one rank dies; SPIKE_P/SPIKE_F tile
shape (default 128x512); SPIKE_REPS warm reps (default 5); SPIKE_GRID =
lnc (default, ``wrap_nki(k)[(lnc,)]``) | none (``wrap_nki(k)``) | direct (``k(x, rg)``);
SPIKE_CORE_START first core id (default 0); SPIKE_LOGDIR per-rank logs
(default ./spike_b4_logs); SPIKE_LOG_TAIL lines of each rank log echoed (default 40).
"""

import os

os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")

import argparse
import json
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
from datetime import timedelta

ALL_CASES = (
    "torch_collectives",
    "nki_all_reduce",
    "nki_all_reduce_private",
    "nki_all_gather",
    "nki_reduce_scatter",
)
_WORKER_FLAG = "SPIKE_B4_WORKER"
_CPU = "--cpu" in sys.argv or os.environ.get("SPIKE_DEVICE", "") == "cpu"
_IS_WORKER = os.environ.get(_WORKER_FLAG) == "1"
if _IS_WORKER and not _CPU:
    # Pin this rank to its own logical core BEFORE torch / torch_neuronx import
    # (AWS test_dynamic_collectives.py pattern; the runtime reads it at init).
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(
        int(os.environ.get("SPIKE_CORE_START", "0")) + int(os.environ["LOCAL_RANK"])
    )

P = int(os.environ.get("SPIKE_P", "128"))
F = int(os.environ.get("SPIKE_F", "512"))
REPS = int(os.environ.get("SPIKE_REPS", "5"))


# ------------------------------------------------------------ expected values
# Pure torch on CPU; used by every rank and exercised by the --cpu dry run.
def make_input(rank, rows=None, cols=None):
    """Exact small integers in fp32: position pattern + 1000 * (rank + 1)."""
    import torch

    rows = P if rows is None else rows
    cols = F if cols is None else cols
    idx = torch.arange(rows * cols, dtype=torch.int64).reshape(rows, cols)
    base = ((idx * 7 + 3) % 251).to(torch.float32)
    return base + 1000.0 * (rank + 1)


def expected_sum(world, rows=None, cols=None):
    import torch

    return torch.stack([make_input(r, rows, cols) for r in range(world)]).sum(0)


def expected_max(world, rows=None, cols=None):
    import torch

    return torch.stack([make_input(r, rows, cols) for r in range(world)]).amax(0)


def expected_gather(world):
    import torch

    return torch.cat([make_input(r) for r in range(world)], dim=0)


def expected_reduce_scatter(world, rank):
    # every rank contributes a (world*P, F) tensor; rank r keeps rows [r*P, (r+1)*P)
    full = expected_sum(world, rows=world * P)
    return full[rank * P : (rank + 1) * P]


# ------------------------------------------------------------ NKI kernels
# Defined only inside a neuron worker: the launcher must never import nki or
# initialize the Neuron runtime (it would hold a core the ranks need).
if _IS_WORKER and not _CPU:
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def b4_all_reduce_add_shared(x, replica_group):
        src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="b4_ar_src")
        dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="b4_ar_dst")
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=x)
        ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=replica_group, op=nl.add)
        nisa.dma_copy(dst=out, src=dst)
        return out

    @nki.jit
    def b4_all_reduce_max_shared(x, replica_group):
        src = nl.ndarray(
            x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="b4_armax_src"
        )
        dst = nl.ndarray(
            x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="b4_armax_dst"
        )
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=x)
        ncc.all_reduce(
            srcs=[src], dsts=[dst], replica_group=replica_group, op=nl.maximum
        )
        nisa.dma_copy(dst=out, src=dst)
        return out

    @nki.jit
    def b4_all_reduce_add_private(x, replica_group):
        src = nl.ndarray(
            x.shape, dtype=x.dtype, buffer=nl.private_hbm, name="b4_arp_src"
        )
        dst = nl.ndarray(
            x.shape, dtype=x.dtype, buffer=nl.private_hbm, name="b4_arp_dst"
        )
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=x)
        ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=replica_group, op=nl.add)
        nisa.dma_copy(dst=out, src=dst)
        return out

    @nki.jit
    def b4_all_gather_shared(x, replica_group, num_ranks):
        rows, cols = x.shape
        src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="b4_ag_src")
        dst = nl.ndarray(
            (rows * num_ranks, cols),
            dtype=x.dtype,
            buffer=nl.shared_hbm,
            name="b4_ag_dst",
        )
        out = nl.ndarray((rows * num_ranks, cols), dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=x)
        ncc.all_gather(
            srcs=[src], dsts=[dst], replica_group=replica_group, collective_dim=0
        )
        nisa.dma_copy(dst=out, src=dst)
        return out

    @nki.jit
    def b4_reduce_scatter_shared(x, replica_group, num_ranks):
        rows, cols = x.shape
        src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="b4_rs_src")
        dst = nl.ndarray(
            (rows // num_ranks, cols),
            dtype=x.dtype,
            buffer=nl.shared_hbm,
            name="b4_rs_dst",
        )
        out = nl.ndarray((rows // num_ranks, cols), dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=x)
        ncc.reduce_scatter(
            srcs=[src],
            dsts=[dst],
            replica_group=replica_group,
            collective_dim=0,
            op=nl.add,
        )
        nisa.dma_copy(dst=out, src=dst)
        return out


# ------------------------------------------------------------ worker
class Worker:
    def __init__(self):
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.case = os.environ["SPIKE_B4_CASE"]
        self.result_file = os.environ["SPIKE_B4_RESULT"]
        self.results = []  # list of dicts: sub, status, detail, cold_ms, warm_ms
        self.dist = None
        self.dev = "cpu"

    def log(self, msg):
        print(f"[r{self.rank}/{self.world} {self.case}] {msg}", flush=True)

    def record(self, sub, status, detail="", cold_ms=None, warm_ms=None):
        self.results.append(
            dict(
                sub=sub, status=status, detail=detail, cold_ms=cold_ms, warm_ms=warm_ms
            )
        )
        self.log(f"{sub}: {status} {detail}")
        self.flush()

    def flush(self):
        with open(self.result_file, "w") as fh:
            json.dump(
                dict(
                    rank=self.rank,
                    world=self.world,
                    case=self.case,
                    results=self.results,
                ),
                fh,
            )

    def sync(self):
        if not _CPU:
            import torch_neuronx

            torch_neuronx.synchronize()

    def timed(self, fn):
        """(result, cold_ms, warm_median_ms). Cold includes NEFF compile/load."""
        t0 = time.perf_counter()
        out = fn()
        self.sync()
        cold = (time.perf_counter() - t0) * 1e3
        warm = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            out = fn()
            self.sync()
            warm.append((time.perf_counter() - t0) * 1e3)
        return out, cold, (statistics.median(warm) if warm else None)

    def check(self, sub, fn, expected):
        import torch

        try:
            out, cold, warm = self.timed(fn)
            got = out.detach().to("cpu")
            if tuple(got.shape) != tuple(expected.shape):
                self.record(
                    sub,
                    "FAIL",
                    f"shape {tuple(got.shape)} != {tuple(expected.shape)}",
                    cold,
                    warm,
                )
                return
            if torch.equal(got, expected):
                tag = (
                    " (cpu stand-in for NKI)" if _CPU and sub.startswith("ncc.") else ""
                )
                self.record(sub, "PASS", "bit-exact" + tag, cold, warm)
            else:
                diff = (got.double() - expected.double()).abs()
                nbad = int((diff > 0).sum())
                self.record(
                    sub,
                    "FAIL",
                    f"max_abs_err={diff.max().item():.6g} n_mismatch={nbad}/{got.numel()} "
                    f"got[0,:4]={got.flatten()[:4].tolist()} exp[0,:4]={expected.flatten()[:4].tolist()}",
                    cold,
                    warm,
                )
        except Exception as e:
            traceback.print_exc()
            first = (str(e).strip().splitlines() or [type(e).__name__])[0]
            self.record(sub, "ERROR", f"{type(e).__name__}: {first[:300]}")

    def setup(self):
        import torch
        import torch.distributed as dist

        self.dist = dist
        if _CPU:
            dist.init_process_group("gloo", timeout=timedelta(seconds=120))
            self.dev = "cpu"
        else:
            import torch_neuronx  # registers the "neuron" c10d backend

            dist.init_process_group("neuron", timeout=timedelta(seconds=120))
            torch.neuron.set_device(self.rank)
            self.dev = f"neuron:{self.rank}"
            if self.rank == 0:
                from torch_neuronx.utils import get_logical_neuron_cores

                self.log(
                    f"torch={torch.__version__} torch_neuronx={torch_neuronx.__version__} "
                    f"nki={getattr(nki, '__version__', '?')} lnc={get_logical_neuron_cores()}"
                )
            self.log(
                "env NEURON_RT_VISIBLE_CORES=%s NEURON_LOGICAL_NC_CONFIG=%s NEURON_RT_ROOT_COMM_ID=%s"
                % tuple(
                    os.environ.get(k, "(unset)")
                    for k in (
                        "NEURON_RT_VISIBLE_CORES",
                        "NEURON_LOGICAL_NC_CONFIG",
                        "NEURON_RT_ROOT_COMM_ID",
                    )
                )
            )
        assert dist.get_rank() == self.rank and dist.get_world_size() == self.world
        self.log(f"process group up on {self.dev} backend={dist.get_backend()}")

    # -- NKI invocation helpers
    def call_nki(self, kernel, *args):
        mode = os.environ.get("SPIKE_GRID", "lnc")
        if mode == "direct":
            return kernel(*args)
        from torch_neuronx import wrap_nki

        w = wrap_nki(kernel)
        if mode == "lnc":
            from torch_neuronx.utils import get_logical_neuron_cores

            w = w[(int(get_logical_neuron_cores()),)]
        return w(*args)

    def replica_group(self):
        from nki.collectives import ReplicaGroup

        return ReplicaGroup([list(range(self.world))])

    # -- CPU stand-ins: same expected-value checks, torch.distributed (gloo) instead of NKI
    def _cpu_all_reduce(self, x, op):
        y = x.clone()
        self.dist.all_reduce(y, op=op)
        return y

    def _cpu_all_gather(self, x):
        import torch

        parts = [torch.empty_like(x) for _ in range(self.world)]
        self.dist.all_gather(parts, x)
        return torch.cat(parts, dim=0)

    def _cpu_reduce_scatter(self, x):
        # gloo has no reduce_scatter: all_reduce then slice this rank's chunk
        y = self._cpu_all_reduce(x, self.dist.ReduceOp.SUM)
        return y[self.rank * P : (self.rank + 1) * P].clone()

    # -- cases
    def case_torch_collectives(self):
        import torch

        dist = self.dist
        x = make_input(self.rank).to(self.dev)

        def ar():
            y = x.clone()
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
            return y

        self.check("dist.all_reduce_sum", ar, expected_sum(self.world))

        def ag():
            out = torch.empty((self.world * P, F), dtype=x.dtype, device=self.dev)
            dist.all_gather_into_tensor(out, x)
            return out

        self.check("dist.all_gather_into_tensor", ag, expected_gather(self.world))

    def case_nki_all_reduce(self):
        dist = self.dist
        x = make_input(self.rank).to(self.dev)
        if _CPU:
            add = lambda: self._cpu_all_reduce(x, dist.ReduceOp.SUM)  # noqa: E731
            mx = lambda: self._cpu_all_reduce(x, dist.ReduceOp.MAX)  # noqa: E731
        else:
            rg = self.replica_group()
            add = lambda: self.call_nki(b4_all_reduce_add_shared, x, rg)  # noqa: E731
            mx = lambda: self.call_nki(b4_all_reduce_max_shared, x, rg)  # noqa: E731
        self.check("ncc.all_reduce[add,shared_hbm]", add, expected_sum(self.world))
        self.check("ncc.all_reduce[max,shared_hbm]", mx, expected_max(self.world))

    def case_nki_all_reduce_private(self):
        x = make_input(self.rank).to(self.dev)
        if _CPU:
            fn = lambda: self._cpu_all_reduce(x, self.dist.ReduceOp.SUM)  # noqa: E731
        else:
            rg = self.replica_group()
            fn = lambda: self.call_nki(b4_all_reduce_add_private, x, rg)  # noqa: E731
        self.check("ncc.all_reduce[add,private_hbm]", fn, expected_sum(self.world))

    def case_nki_all_gather(self):
        x = make_input(self.rank).to(self.dev)
        if _CPU:
            fn = lambda: self._cpu_all_gather(x)  # noqa: E731
        else:
            rg = self.replica_group()
            fn = lambda: self.call_nki(b4_all_gather_shared, x, rg, self.world)  # noqa: E731
        self.check("ncc.all_gather[dim0,shared_hbm]", fn, expected_gather(self.world))

    def case_nki_reduce_scatter(self):
        x = make_input(self.rank, rows=self.world * P).to(self.dev)
        if _CPU:
            fn = lambda: self._cpu_reduce_scatter(x)  # noqa: E731
        else:
            rg = self.replica_group()
            fn = lambda: self.call_nki(b4_reduce_scatter_shared, x, rg, self.world)  # noqa: E731
        self.check(
            "ncc.reduce_scatter[add,dim0,shared_hbm]",
            fn,
            expected_reduce_scatter(self.world, self.rank),
        )

    def run(self):
        self.flush()
        try:
            self.setup()
        except Exception as e:
            traceback.print_exc()
            first = (str(e).strip().splitlines() or [type(e).__name__])[0]
            self.record("setup", "ERROR", f"{type(e).__name__}: {first[:300]}")
            return 1
        # Pre-flight hook for the launcher's timeout path: SPIKE_B4_TEST_HANG=<rank>.
        if os.environ.get("SPIKE_B4_TEST_HANG") == str(self.rank):
            time.sleep(10**6)
        getattr(self, f"case_{self.case}")()
        try:
            self.dist.destroy_process_group()
        except Exception as e:  # teardown problems are logged, not fatal
            self.log(f"destroy_process_group: {type(e).__name__}: {e}")
        return 0


# ------------------------------------------------------------ launcher
_STRIP_CHILD_ENV = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "NEURON_RT_VISIBLE_CORES",
    "NEURON_RT_NUM_CORES",
    "NEURON_RT_ROOT_COMM_ID",
)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill(procs):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def run_job(world, case, logdir, job_timeout, peer_grace, tail):
    """Spawn `world` ranks for one case; return (per-rank results, job note)."""
    port = free_port()
    env_base = {k: v for k, v in os.environ.items() if k not in _STRIP_CHILD_ENV}
    if not _CPU:
        env_base.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
    else:
        env_base["SPIKE_DEVICE"] = "cpu"
    env_base[_WORKER_FLAG] = "1"
    env_base["SPIKE_B4_CASE"] = case
    env_base.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world),
        LOCAL_WORLD_SIZE=str(world),
    )
    procs, logs, rfiles = [], [], []
    t_start = time.time()
    for r in range(world):
        env = dict(env_base, RANK=str(r), LOCAL_RANK=str(r))
        rfile = os.path.join(logdir, f"w{world}_{case}_r{r}.json")
        lfile = os.path.join(logdir, f"w{world}_{case}_r{r}.log")
        if os.path.exists(rfile):
            os.remove(rfile)
        env["SPIKE_B4_RESULT"] = rfile
        fh = open(lfile, "w")
        cmd = [sys.executable, os.path.abspath(__file__)] + (["--cpu"] if _CPU else [])
        procs.append(
            subprocess.Popen(
                cmd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )
        fh.close()
        logs.append(lfile)
        rfiles.append(rfile)

    deadline = t_start + job_timeout
    note = ""
    while True:
        rcs = [p.poll() for p in procs]
        if all(rc is not None for rc in rcs):
            break
        now = time.time()
        if any(rc not in (None, 0) for rc in rcs) and deadline > now + peer_grace:
            deadline = now + peer_grace
            note = f"a rank exited rc={[rc for rc in rcs if rc not in (None, 0)][0]}; peers given {peer_grace}s"
        if now > deadline:
            hung = [r for r, rc in enumerate(rcs) if rc is None]
            note = (
                note + "; " if note else ""
            ) + f"TIMEOUT after {now - t_start:.0f}s, killed ranks {hung}"
            _kill(procs)
            break
        time.sleep(0.5)
    elapsed = time.time() - t_start

    per_rank = []
    for r, (p, lfile, rfile) in enumerate(zip(procs, logs, rfiles)):
        data = None
        if os.path.exists(rfile):
            try:
                with open(rfile) as fh:
                    data = json.load(fh)
            except Exception:
                data = None
        per_rank.append(dict(rank=r, rc=p.returncode, data=data, log=lfile))
        with open(lfile, errors="replace") as fh:
            lines = fh.read().splitlines()
        print(
            f"--- w{world} {case} rank {r} rc={p.returncode} log tail ({len(lines)} lines total: {lfile})"
        )
        for line in lines[-tail:]:
            print(f"  [r{r}] {line}")
    return per_rank, note, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--world", default="1,2,4", help="comma list of world sizes (default 1,2,4)"
    )
    ap.add_argument("--cases", default=",".join(ALL_CASES), help="comma list of cases")
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="gloo on CPU; NKI replaced by torch stand-ins",
    )
    ap.add_argument(
        "--keep-going-after-baseline-timeout",
        action="store_true",
        help="run NKI cases for a world size even if its torch_collectives baseline hung",
    )
    args = ap.parse_args()
    worlds = [int(w) for w in args.world.split(",") if w]
    cases = [c for c in args.cases.split(",") if c]
    bad = [c for c in cases if c not in ALL_CASES]
    if bad:
        sys.exit(f"unknown case(s) {bad}; choose from {ALL_CASES}")

    job_timeout = float(os.environ.get("SPIKE_JOB_TIMEOUT", "300"))
    budget = float(os.environ.get("SPIKE_TOTAL_BUDGET", "2100"))
    peer_grace = float(os.environ.get("SPIKE_PEER_GRACE", "60"))
    tail = int(os.environ.get("SPIKE_LOG_TAIL", "40"))
    logdir = os.path.abspath(os.environ.get("SPIKE_LOGDIR", "spike_b4_logs"))
    os.makedirs(logdir, exist_ok=True)

    print(
        f"spike B4 nki.collectives multi-rank  device={'cpu (gloo, NKI stand-ins)' if _CPU else 'neuron'}"
    )
    print(
        f"worlds={worlds} cases={cases} tile=({P},{F}) fp32 reps={REPS} grid={os.environ.get('SPIKE_GRID', 'lnc')}"
    )
    print(f"job_timeout={job_timeout:.0f}s budget={budget:.0f}s logdir={logdir}")
    print(
        f"parent env NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES', '(unset)')} (stripped for ranks)"
    )
    print(
        f"python={sys.executable}  utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
    )

    t0 = time.time()
    summary = []  # (name, status, detail)
    for world in worlds:
        baseline_hung = False
        for case in cases:
            name = f"w{world}/{case}"
            if time.time() - t0 > budget:
                summary.append((name, "SKIP", "total budget exhausted"))
                print(f"RESULT [{name}]: SKIP total budget exhausted")
                continue
            if (
                baseline_hung
                and case != "torch_collectives"
                and not args.keep_going_after_baseline_timeout
            ):
                summary.append(
                    (
                        name,
                        "SKIP",
                        "torch_collectives baseline timed out at this world size",
                    )
                )
                print(f"RESULT [{name}]: SKIP baseline timed out")
                continue
            print(f"\n===== {name} =====", flush=True)
            per_rank, note, elapsed = run_job(
                world, case, logdir, job_timeout, peer_grace, tail
            )
            if case == "torch_collectives" and "TIMEOUT" in note:
                baseline_hung = True
            # collect sub-results across ranks
            subs = []
            for pr in per_rank:
                for res in (pr["data"] or {}).get("results", []):
                    if res["sub"] not in subs:
                        subs.append(res["sub"])
            if not subs:
                subs = ["(no result)"]
            for sub in subs:
                statuses = []
                for pr in per_rank:
                    res = next(
                        (
                            x
                            for x in (pr["data"] or {}).get("results", [])
                            if x["sub"] == sub
                        ),
                        None,
                    )
                    if res is None:
                        st = "ERROR"
                        det = f"no result (rc={pr['rc']}{'; ' + note if note else ''})"
                        cold = warm = None
                    else:
                        st, det, cold, warm = (
                            res["status"],
                            res["detail"],
                            res["cold_ms"],
                            res["warm_ms"],
                        )
                    statuses.append(st)
                    tm = ""
                    if cold is not None:
                        tm = f" cold={cold:.1f}ms" + (
                            f" warm_med={warm:.3f}ms" if warm is not None else ""
                        )
                    print(f"RESULT [{name}/{sub}/r{pr['rank']}]: {st}{tm} {det}")
                agg = (
                    "PASS"
                    if all(s == "PASS" for s in statuses)
                    else ("FAIL" if "FAIL" in statuses else "ERROR")
                )
                detail = f"ranks={statuses} job={elapsed:.0f}s" + (
                    f" {note}" if note else ""
                )
                print(f"RESULT [{name}/{sub}]: {agg} {detail}")
                summary.append((f"{name}/{sub}", agg, detail))

    print("\n===== SUMMARY =====")
    for name, st, det in summary:
        print(f"{st:5s} {name}  {det}")
    print(f"total {time.time() - t0:.0f}s; per-rank logs in {logdir}")
    return 0


if __name__ == "__main__":
    if _IS_WORKER:
        sys.exit(Worker().run())
    sys.exit(main())
