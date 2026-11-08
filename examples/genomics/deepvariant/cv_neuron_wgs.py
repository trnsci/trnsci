"""Whole-genome DeepVariant 1.9 ``call_variants`` on Trainium2, one process per
NeuronCore (trnsci/trnsci#114, window 6).

Each worker owns a disjoint subset of the ``make_examples`` shards (shard *i*
goes to worker ``i % num_workers``) and writes one stock
``CallVariantsOutput`` TFRecord (GZIP) per input shard::

    OUTDIR/cvo-neuron-000NN-of-00064.tfrecord.gz

so ``postprocess_variants --infile OUTDIR/cvo-neuron@64.tfrecord.gz`` reads
the full set. A shard is written to ``OUTDIR/partial/`` and renamed into
place only when complete, with a marker in ``OUTDIR/done/``; a rerun skips
completed shards (resumable per shard).

Process layout, per worker:

* the **main process** owns the device. It builds and compiles the model and
  runs every forward pass. It never imports torch in a thread other than its
  main thread, so nothing traces NKI from a thread;
* a **decode pool** (``--decode-procs``, spawned, optionally pinned with
  ``--decode-cpus``) reads whole shards, inflates them and decodes the
  ``tf.Example`` records with the TF-free reader in ``dv_io`` into fixed-shape
  uint8 batches in a shared-memory ring of ``--slots`` slots;
* a **writer thread** (pure Python, no torch) applies DeepVariant's
  ``round_gls`` and serializes the CVO records.

Run one worker per NeuronCore, each with ``NEURON_RT_VISIBLE_CORES=k`` (see
``run_wgs_neuron.sh``)::

    NEURON_RT_VISIBLE_CORES=2 python cv_neuron_wgs.py --worker 2 --num-workers 4 \\
        --examples /nvme/dv/tmp/me.tfrecord@64.gz --weights dv19_wgs.pt \\
        --outdir /nvme/dv/out/fp32 --device neuron --check-example-info

CPU smoke run (no device)::

    python cv_neuron_wgs.py --examples 'tmp/me.tfrecord@64.gz' --shards 0,1 \\
        --limit 512 --device cpu --weights dv19_wgs.pt --outdir out/cpu

``--summarize OUTDIR`` aggregates the per-worker reports into the instance
totals.

CVO format (``deepvariant/protos/deepvariant.proto`` at r1.9): ``variant``
(1) and ``alt_allele_indices`` (2) are the example's ``variant/encoded`` and
``alt_allele_indices/encoded`` bytes, ``genotype_probabilities`` (3) is
``round_gls(p, 10)`` as packed doubles, and ``debug_info`` (4) is absent, as
in stock ``call_variants`` without ``--include_debug_info``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import platform
import queue
import re
import signal
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dv_io import (  # noqa: E402
    TFRecordWriter,
    encode_cvo,
    example_fields,
    read_tfrecords,
    round_gls,
)

# DeepVariant 1.9.0 WGS model (its savedmodel example_info.json).
MODEL_SHAPE = (100, 221, 7)
MODEL_CHANNELS = [1, 2, 3, 4, 5, 6, 19]
MODEL_VERSION = "1.9.0"

CVO_NAME = "cvo-neuron"

# --------------------------------------------------------------------------
# Shards and example_info
# --------------------------------------------------------------------------


def input_shards(pattern: str) -> list[str]:
    """``me.tfrecord@64.gz`` -> the 64 paths, in shard order; globs also work."""
    m = re.match(r"^(.*)@(\d+)(.*)$", pattern)
    if m:
        n = int(m.group(2))
        return [f"{m.group(1)}-{i:05d}-of-{n:05d}{m.group(3)}" for i in range(n)]
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths


def cvo_path(outdir: str, shard: int, nshards: int, sub: str = "") -> str:
    d = os.path.join(outdir, sub) if sub else outdir
    return os.path.join(d, f"{CVO_NAME}-{shard:05d}-of-{nshards:05d}.tfrecord.gz")


def example_info_path(shard_path: str) -> str:
    """make_examples writes ``<shard>.example_info.json`` next to each shard."""
    return shard_path + ".example_info.json"


def check_example_info(paths: list[str], require_all: bool = True) -> dict:
    """Validate each shard's example_info.json (shape, channels, version)
    against the model, and the first record's ``image/shape`` and image size.

    Raises SystemExit with every mismatch listed. Returns a summary dict.
    """
    from dv_io import parse_tf_example

    errors, checked, missing = [], 0, []
    infos = set()
    for p in paths:
        ip = example_info_path(p)
        if not os.path.exists(ip):
            missing.append(ip)
            continue
        with open(ip) as f:
            info = json.load(f)
        infos.add(json.dumps(info, sort_keys=True))
        shape = [int(v) for v in info.get("shape", [])]
        chans = [int(v) for v in info.get("channels", [])]
        if shape != list(MODEL_SHAPE):
            errors.append(f"{ip}: shape {shape} != model {list(MODEL_SHAPE)}")
        if chans != MODEL_CHANNELS:
            errors.append(f"{ip}: channels {chans} != model {MODEL_CHANNELS}")
        if str(info.get("version", MODEL_VERSION)) != MODEL_VERSION:
            errors.append(f"{ip}: version {info.get('version')} != {MODEL_VERSION}")
        checked += 1
    if missing and require_all:
        errors.append(
            f"missing example_info for {len(missing)} shard(s): {missing[:3]}"
        )
    # First record of the first present shard: image/shape and byte count.
    first = next((p for p in paths if os.path.exists(p)), None)
    if first is None:
        errors.append("no input shard present")
    else:
        rec = next(iter(read_tfrecords(first, check_crc=True)), None)
        if rec is None:
            errors.append(f"{first}: empty shard")
        else:
            feats = parse_tf_example(rec)
            ishape = [int(v) for v in feats.get("image/shape", [])]
            if ishape and ishape != list(MODEL_SHAPE):
                errors.append(f"{first}: image/shape {ishape} != {list(MODEL_SHAPE)}")
            n = len(feats["image/encoded"][0])
            if n != int(np.prod(MODEL_SHAPE)):
                errors.append(
                    f"{first}: image/encoded {n} B != {int(np.prod(MODEL_SHAPE))}"
                )
            for k in ("variant/encoded", "alt_allele_indices/encoded"):
                if k not in feats:
                    errors.append(f"{first}: missing {k}")
    if errors:
        raise SystemExit("example_info check FAILED:\n  " + "\n  ".join(errors))
    out = {
        "example_info_checked": checked,
        "example_info_missing": len(missing),
        "example_info": [json.loads(s) for s in sorted(infos)],
    }
    print(f"example_info OK: {checked} shard(s), {sorted(infos)}", flush=True)
    return out


# --------------------------------------------------------------------------
# Decode pool (spawned processes; no torch)
# --------------------------------------------------------------------------


def _attach_shm(name: str):
    from multiprocessing import shared_memory

    # Spawned children share the parent's resource tracker, so the attach-time
    # registration is a no-op and the owner (main process) unlinks the segment.
    return shared_memory.SharedMemory(name=name)


def _get(q, ppid: int):
    """Blocking get that gives up if the worker (our parent) has died, so a
    killed worker never leaves decoders blocked on its queues."""
    while True:
        try:
            return q.get(timeout=5)
        except queue.Empty:
            if os.getppid() != ppid:
                raise SystemExit("decoder: parent worker exited") from None


def decode_worker(
    wid: int,
    shm_name: str,
    nslots: int,
    batch: int,
    paths: list[str],
    limit: int,
    task_q,
    free_q,
    ready_q,
    cpus: list[int] | None,
) -> None:
    """Decode shards from ``task_q`` into shared-memory slots.

    Messages on ``ready_q`` (FIFO per decoder, so a shard's batches arrive in
    order and before its ``eof``):
      ("batch", shard, seq, slot, n, variants, alts)
      ("eof", shard, n_records, decode_s, wait_s)
      ("error", shard, text)
    """
    if cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    ppid = os.getppid()
    shm = _attach_shm(shm_name)
    ring = flat = None
    try:
        ring = np.ndarray((nslots, batch) + MODEL_SHAPE, np.uint8, buffer=shm.buf)
        flat = ring.reshape(nslots, batch, -1)
        nbytes = flat.shape[2]
        while True:
            shard = _get(task_q, ppid)
            if shard is None:
                return
            t0 = time.perf_counter()
            wait = 0.0
            n_rec = seq = 0
            slot, row, variants, alts = -1, 0, [], []
            try:
                for rec in read_tfrecords(paths[shard]):
                    if limit and n_rec >= limit:
                        break
                    if slot < 0:
                        tw = time.perf_counter()
                        slot = _get(free_q, ppid)
                        wait += time.perf_counter() - tw
                    img, var, alt = example_fields(rec)
                    if len(img) != nbytes:
                        raise ValueError(
                            f"record {n_rec}: image {len(img)} B != {nbytes} B"
                        )
                    flat[slot, row] = np.frombuffer(img, np.uint8)
                    variants.append(var)
                    alts.append(alt)
                    row += 1
                    n_rec += 1
                    if row == batch:
                        ready_q.put(("batch", shard, seq, slot, row, variants, alts))
                        seq += 1
                        slot, row, variants, alts = -1, 0, [], []
                if row:
                    flat[slot, row:] = 0  # fixed shape: zero-pad the tail
                    ready_q.put(("batch", shard, seq, slot, row, variants, alts))
                ready_q.put(
                    ("eof", shard, n_rec, time.perf_counter() - t0 - wait, wait)
                )
            except Exception as e:  # report and keep the worker from hanging
                import traceback

                ready_q.put(
                    ("error", shard, f"decoder {wid}: {e!r}\n{traceback.format_exc()}")
                )
                return
    finally:
        del ring, flat
        shm.close()


# --------------------------------------------------------------------------
# Writer thread (no torch)
# --------------------------------------------------------------------------


class ShardWriter(threading.Thread):
    """Serializes CVOs per shard; renames each shard into place when complete."""

    def __init__(self, outdir: str, nshards: int, meta: dict):
        super().__init__(daemon=True, name="cvo-writer")
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.outdir, self.nshards, self.meta = outdir, nshards, meta
        self.writers: dict[int, tuple[TFRecordWriter, list]] = {}
        self.error: BaseException | None = None
        self.records = 0
        self.busy_s = 0.0
        self.class_counts = np.zeros(3, np.int64)

    def run(self):
        try:
            while True:
                item = self.q.get()
                if item is None:
                    return
                t0 = time.perf_counter()
                kind = item[0]
                if kind == "rec":
                    _, shard, probs, variants, alts = item
                    w, st = self.writers.get(shard) or self._open(shard)
                    for p, v, a in zip(probs, variants, alts):
                        w.write(encode_cvo(v, a, round_gls(p)))
                    st[0] += len(variants)
                    self.records += len(variants)
                    self.class_counts += np.bincount(probs.argmax(1), minlength=3)
                elif kind == "close":
                    _, shard, n_expected, extra = item
                    w, st = self.writers.pop(shard, None) or self._open(shard)
                    w.close()
                    if st[0] != n_expected:
                        raise RuntimeError(
                            f"shard {shard}: wrote {st[0]} records, decoder read {n_expected}"
                        )
                    final = cvo_path(self.outdir, shard, self.nshards)
                    os.replace(
                        cvo_path(self.outdir, shard, self.nshards, "partial"), final
                    )
                    marker = dict(self.meta, shard=shard, records=st[0], **extra)
                    tmp = os.path.join(self.outdir, "done", f".{shard:05d}.json")
                    with open(tmp, "w") as f:
                        json.dump(marker, f)
                    os.replace(
                        tmp, os.path.join(self.outdir, "done", f"{shard:05d}.json")
                    )
                self.busy_s += time.perf_counter() - t0
        except BaseException as e:  # surfaced by the main loop
            self.error = e

    def _open(self, shard):
        w = TFRecordWriter(cvo_path(self.outdir, shard, self.nshards, "partial"))
        self.writers[shard] = (w, [0])
        return self.writers[shard]


# --------------------------------------------------------------------------
# Host monitor (psutil; no torch)
# --------------------------------------------------------------------------


class HostMonitor(threading.Thread):
    """Samples system-wide CPU busy % and this worker's process-tree CPU."""

    def __init__(self, interval: float = 2.0):
        super().__init__(daemon=True, name="host-monitor")
        self.interval = interval
        self.samples: list[float] = []
        self.stop = threading.Event()
        try:
            import psutil

            self.psutil = psutil
            psutil.cpu_percent(None)
        except ImportError:  # pragma: no cover - /proc/stat fallback (Linux)
            self.psutil = None
        self._last = self._proc_stat()

    @staticmethod
    def _proc_stat():
        try:
            with open("/proc/stat") as f:
                v = [int(x) for x in f.readline().split()[1:]]
            return sum(v) - v[3] - v[4], sum(v)  # busy (minus idle, iowait), total
        except (OSError, ValueError, IndexError):
            return None

    def _busy_pct(self) -> float | None:
        if self.psutil is not None:
            return self.psutil.cpu_percent(None)
        cur = self._proc_stat()
        if cur is None or self._last is None:
            return None
        (b0, t0), (b1, t1) = self._last, cur
        self._last = cur
        return 100.0 * (b1 - b0) / max(1, t1 - t0)

    def run(self):
        while not self.stop.wait(self.interval):
            v = self._busy_pct()
            if v is not None:
                self.samples.append(v)

    def tree_cpu_s(self) -> float:
        """CPU seconds of this process and its live children (decoders)."""
        if self.psutil is None:
            return float("nan")
        me = self.psutil.Process()
        tot = 0.0
        for p in [me] + me.children(recursive=True):
            try:
                t = p.cpu_times()
                tot += t.user + t.system
            except self.psutil.Error:
                pass
        return tot

    def summary(self) -> dict:
        s = self.samples
        return {
            "host_cpu_busy_mean_pct": float(np.mean(s)) if s else None,
            "host_cpu_busy_p95_pct": float(np.percentile(s, 95)) if s else None,
            "host_cpu_busy_max_pct": float(np.max(s)) if s else None,
            "host_cpu_samples": len(s),
            "host_vcpus": os.cpu_count(),
        }


# --------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------


def setup_device(args):
    import torch

    if args.device.startswith("neuron"):
        flags = os.environ.get("NEURON_CC_FLAGS", "")
        if "--auto-cast" not in flags:  # as probe.py: no silent downcasts
            flags = (flags + " --auto-cast=none").strip()
            os.environ["NEURON_CC_FLAGS"] = flags
        import torch_neuronx  # noqa: F401  (registers the "neuron" device)

        print(
            f"torch {torch.__version__}  torch_neuronx {getattr(torch_neuronx, '__version__', '?')}  "
            f"NEURON_CC_FLAGS={flags!r}  "
            f"NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES', '')!r}",
            flush=True,
        )
    else:
        if args.threads:
            torch.set_num_threads(args.threads)
        print(
            f"torch {torch.__version__} cpu threads={torch.get_num_threads()}",
            flush=True,
        )
    return torch.device(args.device)


def device_memory_peak(device) -> dict:
    """Best effort: peak device memory from the torch device module, if any.
    ``run_wgs_neuron.sh`` also samples ``neuron-monitor`` for the whole chip."""
    import torch

    if device.type == "cpu":
        return {}
    mod = getattr(torch, device.type, None)
    out = {}
    for fn in ("max_memory_allocated", "max_memory_reserved"):
        f = getattr(mod, fn, None)
        if callable(f):
            try:
                out[f"device_{fn}_bytes"] = int(f())
            except Exception as e:  # noqa: BLE001
                out[f"device_{fn}_error"] = repr(e)[:200]
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


def _qsize(q) -> str:
    try:
        return str(q.qsize())
    except NotImplementedError:  # macOS
        return "?"


def parse_cpus(s: str | None) -> list[int] | None:
    if not s:
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def select_shards(args, nshards: int) -> list[int]:
    if args.shards:
        shards = sorted({int(s) for s in args.shards.split(",") if s != ""})
    else:
        shards = list(range(nshards))
    return [s for s in shards if s % args.num_workers == args.worker]


def shard_done(outdir: str, shard: int, nshards: int, meta: dict) -> bool:
    mpath = os.path.join(outdir, "done", f"{shard:05d}.json")
    if not (os.path.exists(mpath) and os.path.exists(cvo_path(outdir, shard, nshards))):
        return False
    try:
        with open(mpath) as f:
            m = json.load(f)
    except (OSError, ValueError):
        return False
    return all(
        m.get(k) == meta[k] for k in ("limit", "dtype", "weights_sha256", "input")
    )


def run_worker(args) -> dict:
    paths = input_shards(args.examples)
    nshards = len(paths)
    shards = select_shards(args, nshards)
    for sub in ("", "partial", "done", "reports"):
        os.makedirs(os.path.join(args.outdir, sub), exist_ok=True)

    report: dict = {
        "worker": args.worker,
        "num_workers": args.num_workers,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "mode": args.mode,
        "host_cast": args.host_cast,
        "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS", ""),
        "limit": args.limit,
        "decode_procs": args.decode_procs,
        "slots": args.slots,
        "host": platform.node(),
        "neuron_rt_visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES", ""),
    }
    if args.check_example_info:
        report.update(check_example_info([paths[s] for s in shards] or paths[:1]))

    weights_sha = sha256_file(args.weights)
    meta = {
        "limit": args.limit,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "device": args.device,
        "weights_sha256": weights_sha,
        "input": os.path.basename(args.examples),
    }
    todo = [
        s for s in shards if args.force or not shard_done(args.outdir, s, nshards, meta)
    ]
    missing = [paths[s] for s in todo if not os.path.exists(paths[s])]
    if missing:
        raise SystemExit(f"missing input shard(s): {missing[:4]}")
    report.update(
        shards=shards, shards_todo=todo, shards_skipped=len(shards) - len(todo)
    )
    print(
        f"worker {args.worker}/{args.num_workers}: {len(shards)} shard(s), "
        f"{len(shards) - len(todo)} already done, {len(todo)} to run",
        flush=True,
    )
    if not todo:
        return report

    main_cpus = parse_cpus(args.main_cpus)
    if main_cpus and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, main_cpus)

    # Start the decode pool before torch/torch_neuronx are imported (spawn).
    ctx = mp.get_context("spawn")
    from multiprocessing import shared_memory

    bs = args.batch_size
    slot_bytes = bs * int(np.prod(MODEL_SHAPE))
    shm = shared_memory.SharedMemory(create=True, size=args.slots * slot_bytes)
    ring = np.ndarray((args.slots, bs) + MODEL_SHAPE, np.uint8, buffer=shm.buf)
    task_q, free_q, ready_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    for s in range(args.slots):
        free_q.put(s)
    for s in todo:
        task_q.put(s)
    for _ in range(args.decode_procs):
        task_q.put(None)
    dcpus = parse_cpus(args.decode_cpus)
    decoders = [
        ctx.Process(
            target=decode_worker,
            args=(
                i,
                shm.name,
                args.slots,
                bs,
                paths,
                args.limit,
                task_q,
                free_q,
                ready_q,
                dcpus,
            ),
            daemon=True,
            name=f"dv-decode-{args.worker}-{i}",
        )
        for i in range(args.decode_procs)
    ]
    for p in decoders:
        p.start()

    mon = HostMonitor(args.monitor_interval)
    mon.start()
    writer = ShardWriter(args.outdir, nshards, meta)
    writer.start()

    t_start = time.time()
    p0 = time.perf_counter()
    try:
        device = setup_device(args)
        report["neuron_cc_flags"] = os.environ.get("NEURON_CC_FLAGS", "")
        import torch

        from dv_inception import build_model

        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        model = build_model(args.weights, backbone_dtype=dtype).to(device)
        fwd = model
        if args.mode == "compile":
            backend = "neuron" if device.type == "neuron" else "inductor"
            fwd = torch.compile(
                model,
                backend=backend,
                dynamic=False,
                fullgraph=not args.allow_graph_breaks,
            )
        report["torch"] = torch.__version__
        report["model_load_s"] = time.perf_counter() - p0

        n_ex = n_batches = 0
        wait_s = compute_s = 0.0
        first_batch_s = None
        steady_t0 = None
        steady_ex = 0
        open_shards = set(todo)
        decode_stats = []
        last_log = time.perf_counter()
        last_ex = 0
        cpu0 = mon.tree_cpu_s()

        with torch.inference_mode():
            while open_shards:
                tw = time.perf_counter()
                while True:
                    try:
                        msg = ready_q.get(timeout=5)
                        break
                    except queue.Empty:
                        dead = [p for p in decoders if p.exitcode not in (None, 0)]
                        if dead:
                            raise RuntimeError(
                                f"decoder died: {[p.exitcode for p in dead]}"
                            )
                        if writer.error:
                            raise writer.error
                        if time.perf_counter() - tw > args.input_timeout:
                            raise RuntimeError(f"no input for {args.input_timeout}s")
                waited = time.perf_counter() - tw
                if steady_t0 is not None:
                    wait_s += waited
                kind = msg[0]
                if kind == "error":
                    raise RuntimeError(msg[2])
                if kind == "eof":
                    _, shard, n_rec, dec_s, dec_wait = msg
                    writer.q.put(("close", shard, n_rec, {"decode_s": dec_s}))
                    decode_stats.append((n_rec, dec_s, dec_wait))
                    open_shards.discard(shard)
                    print(
                        f"[w{args.worker}] shard {shard} decoded: {n_rec} ex, "
                        f"{n_rec / max(dec_s, 1e-9):,.0f} ex/s/decoder (excl. slot waits {dec_wait:.0f}s)",
                        flush=True,
                    )
                    continue
                _, shard, _seq, slot, n, variants, alts = msg
                tc = time.perf_counter()
                xt = torch.from_numpy(ring[slot])
                if args.host_cast:
                    xt = xt.float()
                xt = xt.to(device)
                p, _lg = fwd(xt)
                probs = p.to("cpu").numpy()[:n].copy()  # D2H; input consumed
                free_q.put(slot)
                dt = time.perf_counter() - tc
                if first_batch_s is None:
                    first_batch_s = dt
                    steady_t0 = time.perf_counter()
                    print(
                        f"[w{args.worker}] first batch (compile + run) {dt:.1f}s",
                        flush=True,
                    )
                else:
                    compute_s += dt
                    steady_ex += n
                n_ex += n
                n_batches += 1
                writer.q.put(("rec", shard, probs, variants, alts))
                if writer.error:
                    raise writer.error
                now = time.perf_counter()
                if now - last_log >= args.log_every:
                    el = now - (steady_t0 or now)
                    busy = mon.samples[-1] if mon.samples else float("nan")
                    print(
                        f"[w{args.worker}] {n_ex:,} ex  {(n_ex - last_ex) / (now - last_log):,.0f} ex/s now  "
                        f"{steady_ex / max(el, 1e-9):,.0f} ex/s steady  "
                        f"input-wait {100 * wait_s / max(el, 1e-9):.1f}%  host-busy {busy:.0f}%  "
                        f"queue {_qsize(ready_q)}  "
                        f"shards left {len(open_shards)}",
                        flush=True,
                    )
                    last_log, last_ex = now, n_ex
        steady_s = time.perf_counter() - steady_t0 if steady_t0 else 0.0
        writer.q.put(None)
        writer.join()
        if writer.error:
            raise writer.error
        cpu1 = mon.tree_cpu_s()
        wall = time.perf_counter() - p0
        dec_n = sum(d[0] for d in decode_stats)
        dec_s = sum(d[1] for d in decode_stats)
        report.update(
            {
                "examples": n_ex,
                "batches": n_batches,
                "records_written": writer.records,
                "class_counts": writer.class_counts.tolist(),
                "first_batch_s": first_batch_s,
                "steady_s": steady_s,
                "wall_s": wall,
                "examples_per_s_wall": n_ex / wall if wall else None,
                "examples_per_s_steady": steady_ex / steady_s if steady_s else None,
                "device_compute_s": compute_s,
                "input_wait_s": wait_s,
                # The device idles while the main loop waits on the decode
                # pool; the rest of the non-compute time is H2D/D2H/host glue.
                "device_idle_input_frac": wait_s / steady_s if steady_s else None,
                "device_busy_frac_upper": compute_s / steady_s if steady_s else None,
                "decode_ex_per_s_per_proc": dec_n / dec_s if dec_s else None,
                "writer_busy_s": writer.busy_s,
                "worker_tree_cpu_s": cpu1 - cpu0,
                "worker_vcpus_used_mean": (cpu1 - cpu0) / wall if wall else None,
                "t_start_unix": t_start,
                "t_end_unix": time.time(),
            }
        )
        report.update(device_memory_peak(device))
        try:
            import resource

            report["main_peak_rss_mb"] = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
        except Exception:  # noqa: BLE001
            pass
    finally:
        mon.stop.set()
        for p in decoders:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        del ring
        try:
            shm.close()
        except BufferError:  # a tensor still views the ring; the OS frees it at exit
            pass
        shm.unlink()
    report.update(mon.summary())
    return report


def summarize(outdir: str) -> dict:
    reps = []
    # One report per worker invocation (worker-K.<start>.json); a resumed run
    # leaves several. Throughput is from each worker's latest invocation that
    # did work; the record total is from the per-shard markers.
    latest: dict[int, dict] = {}
    n_inv = 0
    for p in sorted(glob.glob(os.path.join(outdir, "reports", "worker-*.json"))):
        with open(p) as f:
            r = json.load(f)
        if not r.get("examples"):
            continue
        n_inv += 1
        w = r["worker"]
        if w not in latest or r["t_start_unix"] > latest[w]["t_start_unix"]:
            latest[w] = r
    reps = [latest[w] for w in sorted(latest)]
    if not reps:
        raise SystemExit(f"no worker reports with examples under {outdir}/reports")
    records = 0
    for p in glob.glob(os.path.join(outdir, "done", "[0-9]*.json")):
        with open(p) as f:
            records += json.load(f).get("records", 0)
    t0 = min(r["t_start_unix"] for r in reps)
    t1 = max(r["t_end_unix"] for r in reps)
    n = sum(r["examples"] for r in reps)
    done = glob.glob(os.path.join(outdir, "done", "[0-9]*.json"))
    out = {
        "workers": len(reps),
        "invocations": n_inv,
        "examples": n,
        "records_in_done_shards": records,
        "instance_wall_s": t1 - t0,
        "instance_examples_per_s": n / (t1 - t0),
        "sum_steady_examples_per_s": sum(r["examples_per_s_steady"] or 0 for r in reps),
        "per_worker": [
            {
                k: r.get(k)
                for k in (
                    "worker",
                    "neuron_rt_visible_cores",
                    "examples",
                    "wall_s",
                    "first_batch_s",
                    "examples_per_s_steady",
                    "device_idle_input_frac",
                    "decode_ex_per_s_per_proc",
                    "worker_vcpus_used_mean",
                    "host_cpu_busy_mean_pct",
                    "host_cpu_busy_max_pct",
                    "device_max_memory_allocated_bytes",
                )
            }
            for r in reps
        ],
        "max_device_idle_input_frac": max(
            r["device_idle_input_frac"] or 0 for r in reps
        ),
        "host_cpu_busy_mean_pct": float(
            np.mean([r["host_cpu_busy_mean_pct"] or 0 for r in reps])
        ),
        "host_cpu_busy_max_pct": max(r["host_cpu_busy_max_pct"] or 0 for r in reps),
        "shards_done": len(done),
    }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="DeepVariant 1.9 call_variants, one process per NeuronCore",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--examples", help="make_examples shards, e.g. DIR/me.tfrecord@64.gz"
    )
    ap.add_argument("--weights", help="dv19_wgs.pt (convert_weights.py)")
    ap.add_argument("--outdir")
    ap.add_argument("--worker", type=int, default=0, help="this worker's index k")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--shards", help="comma list of shard indices (default: all)")
    ap.add_argument("--device", default="neuron", help="neuron | cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument(
        "--batch-size", type=int, default=0, help="0 = 32 for fp32, 64 for bf16"
    )
    ap.add_argument("--mode", choices=["compile", "eager"], default=None,
                    help="default: compile on neuron, eager on cpu")  # fmt: skip
    ap.add_argument("--host-cast", action="store_true",
                    help="ship fp32 (cast on host) instead of uint8 (probe.py fallback)")  # fmt: skip
    ap.add_argument("--allow-graph-breaks", action="store_true",
                    help="torch.compile without fullgraph (probe.py fallback)")  # fmt: skip
    ap.add_argument(
        "--threads", type=int, default=0, help="CPU intra-op threads (cpu device)"
    )
    ap.add_argument("--decode-procs", type=int, default=2)
    ap.add_argument("--decode-cpus", help="pin decoders to these CPUs, e.g. 4,5,10,11")
    ap.add_argument("--main-cpus", help="pin the device process to these CPUs")
    ap.add_argument("--slots", type=int, default=12, help="shared-memory batch slots")
    ap.add_argument(
        "--limit", type=int, default=0, help="max examples per shard (smoke runs)"
    )
    ap.add_argument("--check-example-info", action="store_true")
    ap.add_argument(
        "--check-only", action="store_true", help="example_info check, then exit"
    )
    ap.add_argument("--force", action="store_true", help="rerun shards already done")
    ap.add_argument("--log-every", type=float, default=30.0)
    ap.add_argument("--monitor-interval", type=float, default=2.0)
    ap.add_argument("--input-timeout", type=float, default=900.0)
    ap.add_argument("--report", help="default OUTDIR/reports/worker-K.<start>.json")
    ap.add_argument(
        "--summarize", metavar="OUTDIR", help="aggregate worker reports and exit"
    )
    args = ap.parse_args(argv)

    if args.summarize:
        s = summarize(args.summarize)
        print(json.dumps(s, indent=1))
        with open(os.path.join(args.summarize, "reports", "summary.json"), "w") as f:
            json.dump(s, f, indent=1)
        return
    if not args.examples:
        ap.error("--examples is required")
    if args.check_only:
        paths = input_shards(args.examples)
        check_example_info(paths)
        return
    if not (args.weights and args.outdir):
        ap.error("--weights and --outdir are required")
    if not 0 <= args.worker < args.num_workers:
        ap.error("need 0 <= --worker < --num-workers")
    if not args.batch_size:
        args.batch_size = 64 if args.dtype == "bf16" else 32
    if args.mode is None:
        args.mode = "compile" if args.device.startswith("neuron") else "eager"

    # timeout(1) sends SIGTERM: exit through the finally blocks (decoders
    # joined, shared memory unlinked) instead of dying in place.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    t0 = int(time.time())
    report = run_worker(args)
    rp = args.report or os.path.join(
        args.outdir, "reports", f"worker-{args.worker}.{t0}.json"
    )
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=1)
    keys = (
        "examples", "wall_s", "first_batch_s", "examples_per_s_steady",
        "device_idle_input_frac", "decode_ex_per_s_per_proc", "host_cpu_busy_mean_pct",
    )  # fmt: skip
    print(
        f"[w{args.worker}] done: " + json.dumps({k: report.get(k) for k in keys}),
        flush=True,
    )


if __name__ == "__main__":
    main()
