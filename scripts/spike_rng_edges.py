#!/usr/bin/env python
"""Spike B10 — trnrand edge-case sweep on Trn2 after #56 (bitcast), #58 (stream order), #59 (Sobol).

Bit-exact checks of trnrand's NKI kernels (via the ``trnrand.nki.dispatch`` host
wrappers and the public API) against the CPU references:

  philox.*     Philox4x32-10 kernel vs ``philox4x32_reference``: counters near
               2^32-1 (wrap), 2^31 and 2^24 boundaries, keys 0 / 2^31 / 2^32-1,
               >=64 seeds.
  threefry.*   raw Threefry4x32-20 kernel vs ``threefry4x32_reference`` with
               every counter/key word at edge values (the kernel emits the low
               24 bits of each word / 2^24); the ``threefry_uniform_nki``
               wrapper over many seeds and counter_offsets near the 2^24 c1
               wrap; the public ``uniform``/``normal`` generator path.
  stream.*     ``threefry_stream_{uniform,normal}`` (one launch = 32 tiles) vs the
               per-tile wrappers, multi-launch, several counter offsets (incl.
               the 2^24 wrap); ``GeneratorProgram.stream_into`` vs per-tile.
  sobol.*      ``sobol_nki`` and public ``sobol`` vs ``sobol_reference`` and
               ``torch.quasirandom.SobolEngine(scramble=False)`` (top 24 bits)
               for dims 1..10 and start indices {0,1,255,256,1023,4095}.
  halton.*     ``halton_nki`` and public ``halton`` vs a float64 CPU radical
               inverse, at scale (default 2^16 points, also near the 2^22 limit).

Run on the box (one NeuronCore, pinned by the harness):

    python scripts/spike_rng_edges.py 2>&1 | tee ~/spike_rng_edges.log

CPU pre-flight (``--cpu`` or ``SPIKE_DEVICE=cpu``): the device kernels are
replaced by CPU emulators built from the references, so the dry run checks the
harness, the reference compositions and SobolEngine agreement, not the device.
Public-API cases that only differ on device are SKIPped on CPU.

Env knobs: ``SPIKE_RNG_SEEDS`` (default 64), ``SPIKE_SOBOL_N`` (1024),
``SPIKE_HALTON_N`` (65536), ``SPIKE_STREAM_N`` (2*16384+777),
``SPIKE_RNG_CASES`` (comma-separated prefix filter, e.g. ``philox,sobol``).

Output: ``RESULT [<case>]: PASS|FAIL|ERROR|SKIP ...`` lines and a ``SUMMARY``
block. Post the log to trnsci/trnrand (#56/#58/#59) and trnsci/trnsci#35.
"""

import os

os.environ.setdefault("NKI_ENABLE_TRACE_CACHE", "0")

import argparse
import math
import sys
import time
import traceback

import numpy as np
import torch

M32 = 0xFFFFFFFF
M24 = 0xFFFFFF
LANES = 128
TILES = 32  # _PROGRAM_TILES
PER_LAUNCH = TILES * LANES * 4  # 16384

N_SEEDS = int(os.environ.get("SPIKE_RNG_SEEDS", "64"))
SOBOL_N = int(os.environ.get("SPIKE_SOBOL_N", "1024"))
HALTON_N = int(os.environ.get("SPIKE_HALTON_N", "65536"))
STREAM_N = int(os.environ.get("SPIKE_STREAM_N", str(2 * PER_LAUNCH + 777)))
CASE_FILTER = [s for s in os.environ.get("SPIKE_RNG_CASES", "").split(",") if s]

RESULTS = {}
CPU = False
d = None  # trnrand.nki.dispatch
trnrand = None


def first_line(exc):
    msg = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {msg[0] if msg else ''}"[:300]


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


CASES = []


def run_case(name, fn):
    if CASE_FILTER and not any(name.startswith(p) for p in CASE_FILTER):
        return
    print(f"\n===== {name} =====", flush=True)
    t0 = time.perf_counter()
    try:
        r = fn()
        verdict, info = r if isinstance(r, tuple) else (("PASS" if r else "FAIL"), "")
    except Exception as e:
        traceback.print_exc(limit=4)
        verdict, info = "ERROR", first_line(e)
    RESULTS[name] = verdict
    print(
        f"RESULT [{name}]: {verdict} {info} ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )


# ------------------------------------------------------------------ helpers
def u32_to_i32_tensor(a):
    """uint32 numpy -> int32 torch tensor holding the same bits."""
    return torch.from_numpy(
        np.ascontiguousarray(np.asarray(a, dtype=np.uint32)).view(np.int32)
    )


def bits_f32(t):
    return t.detach().cpu().float().contiguous().view(torch.int32)


def cmp_f32(got, exp, label=""):
    """Bit-exact float32 compare. Returns (verdict, info)."""
    got = got.detach().cpu().float().reshape(-1)
    exp = exp.detach().cpu().float().reshape(-1)
    if got.shape != exp.shape:
        return "FAIL", f"{label} shape {tuple(got.shape)} != {tuple(exp.shape)}"
    neq = bits_f32(got) != bits_f32(exp)
    n = int(neq.sum())
    md = float((got.double() - exp.double()).abs().max()) if got.numel() else 0.0
    info = f"{label} n={got.numel()} mismatches={n} max_abs_diff={md:.3e}"
    if n:
        i = int(torch.nonzero(neq)[0])
        info += f" first@{i}: got={got[i].item()!r} exp={exp[i].item()!r}"
    return ("PASS" if n == 0 else "FAIL"), info


def cmp_u32(got, exp, label=""):
    got = np.asarray(got, dtype=np.uint64).reshape(-1)
    exp = np.asarray(exp, dtype=np.uint64).reshape(-1)
    if got.shape != exp.shape:
        return "FAIL", f"{label} shape {got.shape} != {exp.shape}"
    neq = got != exp
    n = int(neq.sum())
    info = f"{label} words={got.size} mismatches={n}"
    if n:
        i = int(np.nonzero(neq)[0][0])
        info += f" first@{i}: got={int(got[i]):#010x} exp={int(exp[i]):#010x}"
    return ("PASS" if n == 0 else "FAIL"), info


def combine(*results):
    verdicts = [r[0] for r in results]
    v = "PASS" if all(x == "PASS" for x in verdicts) else "FAIL"
    return v, " | ".join(r[1] for r in results)


# ------------------------------------------------------------------ references
def philox_ref(c_lo, k_lo, k_hi):
    n = len(c_lo)
    ctr = torch.zeros(n, 4, dtype=torch.int64)
    ctr[:, 0] = torch.from_numpy(np.asarray(c_lo, dtype=np.int64))
    key = torch.stack(
        [
            torch.from_numpy(np.asarray(k_lo, dtype=np.int64)),
            torch.from_numpy(np.asarray(k_hi, dtype=np.int64)),
        ],
        dim=-1,
    )
    return (d.philox4x32_reference(ctr, key) & M32).numpy().astype(np.uint64)  # (n,4)


def threefry_ref_words(c, k):
    """c, k: (n,4) int-like arrays of uint32 values -> (n,4) uint32 words (as int64)."""
    ct = torch.from_numpy(np.asarray(c, dtype=np.int64))
    kt = torch.from_numpy(np.asarray(k, dtype=np.int64))
    return d.threefry4x32_reference(ct, kt) & M32


def threefry_ref_f32(c, k):
    """Kernel convention: low 24 bits of each word / 2^24, float32. (n,4)."""
    w = threefry_ref_words(c, k)
    return ((w & M24).double() / float(1 << 24)).float()


def threefry_wrapper_ref(n, seed, off):
    """Reference for threefry_uniform_nki(n, seed, off) / per-tile layout."""
    n_words = (n + 3) // 4
    w = np.arange(n_words, dtype=np.int64)
    c = np.zeros((n_words, 4), dtype=np.int64)
    c[:, 0] = w % LANES
    c[:, 1] = (w // LANES + off) & M24
    k = np.zeros((n_words, 4), dtype=np.int64)
    k[:, 0] = seed & M24
    k[:, 1] = (seed >> 24) & M24
    return threefry_ref_f32(c, k).reshape(-1)[:n]


def sobol_engine_top24(n, dims, start):
    eng = torch.quasirandom.SobolEngine(dimension=dims, scramble=False)
    if start:
        eng.fast_forward(start)
    x = eng.draw(n, dtype=torch.float64)
    return (torch.floor(x * (1 << 24)) / (1 << 24)).float()


def sobol_ref(n, dims, start):
    if hasattr(d, "sobol_reference"):
        return d.sobol_reference(n, dims, start)
    return sobol_engine_top24(n, dims, start)


HALTON_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


def halton_ref64(n, dims, start):
    first = max(start, 1)
    out = np.zeros((n, dims), dtype=np.float64)
    for j in range(dims):
        p = HALTON_PRIMES[j]
        idx = np.arange(first, first + n, dtype=np.int64)
        r = np.zeros(n, dtype=np.float64)
        f = 1.0
        while (idx > 0).any():
            f /= p
            r += f * (idx % p)
            idx //= p
        out[:, j] = r
    return torch.from_numpy(out)


# ------------------------------------------------------------------ device/emulated kernels
class Kernels:
    """Device kernels (neuron) or CPU emulators built from the references (--cpu)."""

    def philox(self, c_lo, k_lo, k_hi):
        if CPU:
            return philox_ref(c_lo, k_lo, k_hi)
        out = d.philox4x32_nki(
            u32_to_i32_tensor(c_lo), u32_to_i32_tensor(k_lo), u32_to_i32_tensor(k_hi)
        )
        return (
            out.cpu()
            .to(torch.int32)
            .numpy()
            .view(np.uint32)
            .astype(np.uint64)
            .reshape(-1, 4)
        )

    def threefry_raw(self, c, k):
        """c, k: (n,4) uint32 arrays -> (n,4) float32 from the raw threefry4x32 kernel."""
        if CPU:
            return threefry_ref_f32(c, k)
        c = np.asarray(c, dtype=np.uint32)
        k = np.asarray(k, dtype=np.uint32)
        outs = []
        for s in range(0, c.shape[0], LANES):
            cols = [c[s : s + LANES, j].reshape(-1, 1) for j in range(4)]
            cols += [k[s : s + LANES, j].reshape(-1, 1) for j in range(4)]
            t = d._launch(
                d.threefry4x32_kernel,
                "threefry4x32",
                *(u32_to_i32_tensor(a) for a in cols),
            )
            outs.append(t.cpu().float().reshape(-1, 4))
        return torch.cat(outs)

    def threefry_uniform(self, n, seed, off):
        if CPU:
            return threefry_wrapper_ref(n, seed, off)
        return d.threefry_uniform_nki(n, seed=seed, counter_offset=off).cpu()

    def threefry_normal(self, n, seed, off):
        if CPU:  # emulate: Box-Muller on the reference uniforms (pairs)
            u = threefry_wrapper_ref(((n + 3) // 4) * 4, seed, off)
            return d.box_muller_cpu(u)[:n]
        return d.threefry_normal_nki(n, seed=seed, counter_offset=off).cpu()

    def stream_uniform(self, n, seed, off):
        if CPU:
            return threefry_wrapper_ref(n, seed, off)
        return d.threefry_stream_uniform(n, seed=seed, counter_offset=off).cpu()

    def stream_normal(self, n, seed, off):
        if CPU:
            return self.threefry_normal(n, seed, off)
        return d.threefry_stream_normal(n, seed=seed, counter_offset=off).cpu()

    def sobol(self, n, dims, start):
        if CPU:
            return sobol_ref(n, dims, start)
        return d.sobol_nki(n, dims, seed=0, start_index=start).cpu()

    def halton(self, n, dims, start):
        if CPU:
            return halton_ref64(n, dims, start).float()
        return d.halton_nki(n, dims, start_index=start).cpu()


K = Kernels()

EDGE32 = [
    0,
    1,
    0xFF,
    0x100,
    0xFFFF,
    0x10000,
    0xFFFFFF,
    0x1000000,
    0x7FFFFFFF,
    0x80000000,
    0x80000001,
    0xFFFFFFFE,
    0xFFFFFFFF,
]
KEY_EDGES = [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xDEADBEEF]


def rng_np(seed):
    return np.random.default_rng(seed)


# ------------------------------------------------------------------ philox
@case("philox.edge_counters_x_keys")
def _():
    c, kl, kh = [], [], []
    for cv in EDGE32:
        for a in KEY_EDGES:
            for b in KEY_EDGES:
                c.append(cv)
                kl.append(a)
                kh.append(b)
    return cmp_u32(K.philox(c, kl, kh), philox_ref(c, kl, kh), f"lanes={len(c)}")


@case("philox.counter_wrap_runs")
def _():
    # contiguous counter runs straddling 2^32 (wraps to 0 mod 2^32), 2^31 and 2^24
    c = []
    for centre in (1 << 32, 1 << 31, 1 << 24, 1 << 16):
        c += [(centre + i) & M32 for i in range(-64, 64)]
    kl = [0x9E3779B9] * len(c)
    kh = [0xFFFFFFFF] * len(c)
    return cmp_u32(K.philox(c, kl, kh), philox_ref(c, kl, kh), f"lanes={len(c)}")


@case("philox.many_seeds")
def _():
    r = rng_np(56)
    seeds = [0, 1, (1 << 31), (1 << 32) - 1, (1 << 32), (1 << 63) - 1, (1 << 64) - 1]
    seeds += [int(x) for x in r.integers(0, 1 << 63, size=max(0, N_SEEDS - len(seeds)))]
    c, kl, kh = [], [], []
    for s in seeds:
        base = int(r.integers(0, 1 << 32))
        c += [(base + i) & M32 for i in range(LANES)]
        kl += [s & M32] * LANES
        kh += [(s >> 32) & M32] * LANES
    return cmp_u32(
        K.philox(c, kl, kh), philox_ref(c, kl, kh), f"seeds={len(seeds)} lanes={len(c)}"
    )


@case("philox.uniform_cpu_consistency")
def _():
    # philox_uniform_cpu (the documented oracle) == reference words * 2^-32 for big offsets
    res = []
    for off in (0, (1 << 32) - 3, (1 << 31) - 1):
        u = d.philox_uniform_cpu(
            64, seed=0xFFFFFFFF_00000001, counter_offset=off, dtype=torch.float64
        )
        c = [(off + i) for i in range(16)]
        w = philox_ref([x & M32 for x in c], [1] * 16, [0xFFFFFFFF] * 16).reshape(-1)
        exp = torch.from_numpy(w.astype(np.float64) / 2**32)
        # philox_uniform_cpu does not mask the counter: offsets past 2^32 are masked by the ref
        res.append(cmp_f32(u.float(), exp.float(), f"off={off:#x}"))
    return combine(*res)


# ------------------------------------------------------------------ threefry raw
def _threefry_lanes(full32):
    r = rng_np(57 if full32 else 24)
    hi = 1 << 32 if full32 else 1 << 24
    edges = EDGE32 if full32 else [v for v in EDGE32 if v < (1 << 24)]
    kedges = KEY_EDGES if full32 else [v for v in KEY_EDGES if v < (1 << 24)] + [M24]
    c, k = [], []
    # each counter word at each edge value, others random
    for j in range(4):
        for v in edges:
            cc = [int(x) for x in r.integers(0, hi, 4)]
            cc[j] = v
            c.append(cc)
            k.append([int(x) for x in r.integers(0, hi, 4)])
    # each key word at each key edge, counters random
    for j in range(4):
        for v in kedges:
            kk = [int(x) for x in r.integers(0, hi, 4)]
            kk[j] = v
            k.append(kk)
            c.append([int(x) for x in r.integers(0, hi, 4)])
    # all-equal words
    for v in edges:
        c.append([v] * 4)
        k.append([v] * 4)
    # (c0, c1) carry boundary: c0 = max, c1 = x  and  c0 = 0, c1 = x+1
    top = hi - 1
    for x in (0, 1, 0x7FFFFF, top - 1):
        c.append([top, x, 0, 0])
        k.append([0, 0, 0, 0])
        c.append([0, x + 1, 0, 0])
        k.append([0, 0, 0, 0])
    # many seeds: 128 sequential lanes each
    for _s in range(N_SEEDS):
        kk = [int(x) for x in r.integers(0, hi, 4)]
        base = int(r.integers(0, hi))
        for i in range(LANES):
            c.append([(base + i) % hi, 0, 0, 0])
            k.append(kk)
    return np.array(c, dtype=np.int64), np.array(k, dtype=np.int64)


@case("threefry.raw_inputs_lt_2^24")
def _():
    c, k = _threefry_lanes(full32=False)
    return cmp_f32(
        K.threefry_raw(c, k),
        threefry_ref_f32(c, k),
        f"lanes={len(c)} (kernel contract)",
    )


@case("threefry.raw_inputs_full32")
def _():
    # beyond the documented < 2^24 input contract: exercises the #56 bitcast path
    c, k = _threefry_lanes(full32=True)
    return cmp_f32(
        K.threefry_raw(c, k),
        threefry_ref_f32(c, k),
        f"lanes={len(c)} (beyond contract)",
    )


@case("threefry.uniform_wrapper_many_seeds")
def _():
    r = rng_np(1)
    seeds = [0, 1, M24, 1 << 24, (1 << 31), M32, (1 << 48) - 1]
    seeds += [int(x) for x in r.integers(0, 1 << 48, size=max(0, N_SEEDS - len(seeds)))]
    n = 4 * LANES * 2 + 36  # 2 full tiles + a partial tile
    bad, first = 0, None
    for s in seeds:
        v, info = cmp_f32(
            K.threefry_uniform(n, s, 0), threefry_wrapper_ref(n, s, 0), f"seed={s:#x}"
        )
        if v != "PASS":
            bad += 1
            first = first or info
    return (
        "PASS" if bad == 0 else "FAIL"
    ), f"seeds={len(seeds)} n={n} failing={bad} {first or ''}"


@case("threefry.uniform_wrapper_counter_offsets")
def _():
    n = 4 * LANES * 3
    res = []
    for off in (0, 1, 1000, M24 - 2, M24 - 1, M24, 1 << 24):
        res.append(
            cmp_f32(
                K.threefry_uniform(n, 0xABCDEF, off),
                threefry_wrapper_ref(n, 0xABCDEF, off),
                f"off={off:#x}",
            )
        )
    return combine(*res)


@case("threefry.public_uniform_normal_generator")
def _():
    if CPU:
        return "SKIP", "public NKI path only on device"
    from trnrand.distributions import _nki_seed

    n = 10000
    res = []
    for s in (0, 7, 123456789):
        g, ge = trnrand.Generator(seed=s), trnrand.Generator(seed=s)
        for call in range(2):
            got = trnrand.uniform(n, generator=g)
            seed24 = _nki_seed(ge)
            off = ge._chip_counter_offset(n)
            ge._advance_by_elements(n)
            res.append(
                cmp_f32(
                    got,
                    threefry_wrapper_ref(n, seed24, off),
                    f"uniform s={s} call={call}",
                )
            )
        g, ge = trnrand.Generator(seed=s), trnrand.Generator(seed=s)
        got = trnrand.normal(n, generator=g)
        seed24 = _nki_seed(ge)
        res.append(
            cmp_f32(got, K.threefry_normal(n, seed24, 0), f"normal-vs-wrapper s={s}")
        )
    return combine(*res)


# ------------------------------------------------------------------ streaming vs per-tile
STREAM_OFFSETS = (0, 1, 31, 32, 33, 1000, M24 - 40, M24)


@case("stream.uniform_vs_per_tile")
def _():
    res = []
    for seed in (0, 12345, (1 << 24) + 7):
        for off in STREAM_OFFSETS:
            s = K.stream_uniform(STREAM_N, seed, off)
            p = K.threefry_uniform(STREAM_N, seed, off)
            v1 = cmp_f32(s, p, f"s={seed:#x} off={off:#x} stream-vs-tile")
            v2 = cmp_f32(s, threefry_wrapper_ref(STREAM_N, seed, off), "stream-vs-ref")
            res.append(combine(v1, v2))
    bad = [r[1] for r in res if r[0] != "PASS"]
    return (
        "PASS" if not bad else "FAIL"
    ), f"cases={len(res)} n={STREAM_N} failing={len(bad)} {bad[0] if bad else ''}"


@case("stream.normal_vs_per_tile")
def _():
    res = []
    for seed in (0, 12345, (1 << 24) + 7):
        for off in STREAM_OFFSETS:
            res.append(
                cmp_f32(
                    K.stream_normal(STREAM_N, seed, off),
                    K.threefry_normal(STREAM_N, seed, off),
                    f"s={seed:#x} off={off:#x}",
                )
            )
    bad = [r[1] for r in res if r[0] != "PASS"]
    return (
        "PASS" if not bad else "FAIL"
    ), f"cases={len(res)} n={STREAM_N} failing={len(bad)} {bad[0] if bad else ''}"


@case("stream.generator_program_multi_launch")
def _():
    if CPU:
        return "SKIP", "GeneratorProgram NKI path only with neuronxcc"
    res = []
    nu, nz = STREAM_N, PER_LAUNCH + 5
    for seed, advance in ((42, 0), (42, 512 * 1000), (99, 512 * (M24 - 40))):
        g = trnrand.Generator(seed=seed)
        if advance:
            g.advance(advance)
        prog = g.new_program().uniform(nu, out="u").normal(nz, out="z").build()
        ctr = g._counter
        for call in range(2):
            buf_u, buf_z = torch.empty(nu), torch.empty(nz)
            prog.stream_into({"u": buf_u, "z": buf_z})
            off_u = ctr
            ctr = (ctr + math.ceil(nu / PER_LAUNCH) * TILES) & M24
            off_z = ctr
            ctr = (ctr + math.ceil(nz / PER_LAUNCH) * TILES) & M24
            res.append(
                cmp_f32(
                    buf_u,
                    K.threefry_uniform(nu, seed, off_u),
                    f"s={seed} adv={advance} call={call} u off={off_u:#x}",
                )
            )
            res.append(
                cmp_f32(buf_z, K.threefry_normal(nz, seed, off_z), f"z off={off_z:#x}")
            )
    bad = [r[1] for r in res if r[0] != "PASS"]
    return (
        "PASS" if not bad else "FAIL"
    ), f"checks={len(res)} failing={len(bad)} {bad[0] if bad else ''}"


# ------------------------------------------------------------------ sobol
SOBOL_STARTS = (0, 1, 255, 256, 1023, 4095)


@case("sobol.nki_vs_reference_and_engine")
def _():
    rows, bad = 0, []
    for dims in range(1, 11):
        for start in SOBOL_STARTS:
            got = K.sobol(SOBOL_N, dims, start)
            v1 = cmp_f32(
                got, sobol_ref(SOBOL_N, dims, start), f"d={dims} start={start} vs-ref"
            )
            v2 = cmp_f32(
                got, sobol_engine_top24(SOBOL_N, dims, start), "vs-SobolEngine-top24"
            )
            rows += 1
            for v in (v1, v2):
                if v[0] != "PASS":
                    bad.append(f"d={dims} start={start}: {v[1]}")
    for b in bad[:12]:
        print("   ", b)
    return (
        "PASS" if not bad else "FAIL"
    ), f"configs={rows} n={SOBOL_N} failing={len(bad)}"


@case("sobol.reference_vs_engine_cpu")
def _():
    res = [
        cmp_f32(sobol_ref(4096, 10, s), sobol_engine_top24(4096, 10, s), f"start={s}")
        for s in SOBOL_STARTS
    ]
    return combine(*res)


@case("sobol.public_unscrambled")
def _():
    res = []
    for dims in range(1, 11):
        got = trnrand.sobol(SOBOL_N, dims, scramble=False)
        if CPU:  # CPU path = SobolEngine float32 draw: compare within one 2^-24 step
            md = float(
                (got.double() - sobol_engine_top24(SOBOL_N, dims, 0).double())
                .abs()
                .max()
            )
            res.append(
                ("PASS" if md <= 2**-23 else "FAIL", f"d={dims} max_abs_diff={md:.2e}")
            )
        else:
            res.append(cmp_f32(got, sobol_engine_top24(SOBOL_N, dims, 0), f"d={dims}"))
    bad = [r[1] for r in res if r[0] != "PASS"]
    return (
        "PASS" if not bad else "FAIL"
    ), f"dims=1..10 failing={len(bad)} {bad[0] if bad else ''}"


# ------------------------------------------------------------------ halton
HALTON_TOL = 4e-6


def _halton_check(got, ref64, label):
    got = got.detach().cpu().float()
    md = float((got.double() - ref64).abs().max())
    exact = float((bits_f32(got) == bits_f32(ref64.float())).double().mean())
    in_range = bool(((got > 0) & (got < 1)).all())
    ok = md <= HALTON_TOL and in_range
    return (
        "PASS" if ok else "FAIL"
    ), f"{label} max_abs_diff={md:.2e} exact_f32_frac={exact:.4f} in(0,1)={in_range}"


@case("halton.nki_at_scale")
def _():
    res = []
    limit = 1 << 22
    for dims in (10, 3):
        for start in (0, 1, 1000, limit - HALTON_N - 1):
            res.append(
                _halton_check(
                    K.halton(HALTON_N, dims, start),
                    halton_ref64(HALTON_N, dims, start),
                    f"d={dims} start={start}",
                )
            )
    return combine(*res)


@case("halton.public")
def _():
    n = min(HALTON_N, 16384)
    return _halton_check(trnrand.halton(n, 10), halton_ref64(n, 10, 0), f"n={n} d=10")


# ------------------------------------------------------------------ main
def main():
    global CPU, d, trnrand
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    CPU = args.cpu or os.environ.get("SPIKE_DEVICE", "neuron").lower() == "cpu"
    if not CPU:
        try:
            import torch_neuronx

            print("torch_neuronx", getattr(torch_neuronx, "__version__", "?"))
        except Exception as e:
            print("torch_neuronx import failed:", first_line(e))
    import trnrand as _t
    from trnrand.nki import dispatch as _d

    trnrand, d = _t, _d
    trnrand.set_backend("pytorch" if CPU else "nki")
    print("torch", torch.__version__, "trnrand", getattr(trnrand, "__version__", "?"))
    print(
        "device",
        "cpu (emulated kernels)" if CPU else "neuron",
        "backend",
        trnrand.get_backend(),
    )
    print(
        "HAS_NKI",
        d.HAS_NKI,
        "dispatch_path",
        getattr(d, "_get_dispatch_path", lambda: "?")(),
    )
    print(
        "NEURON_RT_VISIBLE_CORES", os.environ.get("NEURON_RT_VISIBLE_CORES", "(unset)")
    )
    print(f"seeds={N_SEEDS} sobol_n={SOBOL_N} halton_n={HALTON_N} stream_n={STREAM_N}")
    t0 = time.time()
    for name, fn in CASES:
        run_case(name, fn)
    print("\n========== SUMMARY ==========")
    for k, v in RESULTS.items():
        print(f"  {v:5}  {k}")
    counts = {
        s: sum(1 for v in RESULTS.values() if v == s)
        for s in ("PASS", "FAIL", "ERROR", "SKIP")
    }
    print(f"  totals: {counts} of {len(RESULTS)}  device={'cpu' if CPU else 'neuron'}")
    print(f"elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
