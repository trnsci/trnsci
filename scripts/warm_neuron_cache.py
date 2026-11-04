#!/usr/bin/env python
"""Warm Neuron NEFF compile caches off-box (trnsci/trnsci#118).

On SDK 2.33 (native TorchNeuron) a compiled NEFF is keyed only by the
StableHLO program and the compiler configuration:

    key = xxh3_128_hex(module.mlir || 0xFF ||
                       f"{target}_{lnc}_{opt}_{compiler_args}_v{neuronxcc}" [+ "_wlo"])

and stored at ``$TORCH_NEURONX_NEFF_CACHE_DIR/<k[0:2]>/<k[2:4]>/<k>.neff``.
No host, path, PID or time goes into it, so NEFFs compiled on a CPU host with
the same toolchain are persistent-cache hits on the Trainium box.

Subcommands:

    manifest   on the Neuron box: record the toolchain identity and compiler
               configuration (plus a probe compile) to a JSON manifest
    check      compare the local toolchain with a manifest; exit 1 on mismatch
    capture    on a CPU host: run each project's ``pytest -m neuron`` in
               compile-only mode and collect the NEFFs into a cache dir
    recompile  compile recorded ``module.mlir`` files with the manifest's flags
               (for data-dependent shapes compile-only can't reach)
    pack       tar a cache dir plus a JSON index (key, size, sha256, source)
    unpack     install a packed archive into one or more NEFF cache dirs
    verify     on the box: run a selection with the CompilationCache counters
               and report the persistent-hit rate

Typical flow:

    # box, during a window (one probe compile, ~2 s)
    python scripts/warm_neuron_cache.py manifest -o trn2-lnc2.json
    # CPU host booted from the same DLAMI, before the next window
    python scripts/warm_neuron_cache.py check --manifest trn2-lnc2.json
    python scripts/warm_neuron_cache.py capture --manifest trn2-lnc2.json \\
        --out ~/warm --projects trnblas,trnsolver --workers 16
    python scripts/warm_neuron_cache.py pack ~/warm
    # box
    python scripts/warm_neuron_cache.py unpack neff-<tag>.tar.zst --into /tmp/neff_cache
    python scripts/warm_neuron_cache.py verify --projects trnblas,trnsolver

The toolchain has to match exactly: the neuronx-cc version is in the key, and
the HLO bytes depend on the torch-neuronx build. A mismatch can't produce a
wrong answer; it only misses, and the box silently compiles as before.
``capture`` and ``recompile`` therefore refuse to run until ``check`` passes.
See docs/developing_kernels.md for the full list of requirements and blockers.

Moving archives between hosts (S3 or otherwise) is out of scope; ``pack``
writes a local file.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
PLUGIN = "neuron_cache_plugin"

MANIFEST_FORMAT = "trnsci-neuron-toolchain-manifest"
INDEX_FORMAT = "trnsci-neff-index"
FORMAT_VERSION = 1

# Fixed eager-op compiler arguments on SDK 2.33 (torch-neuronx 2.14.3).
# NEURON_CC_FLAGS and int64-policy flags are appended to this.
BASE_COMPILER_ARGS = (
    "--model-type transformer --auto-cast=none --internal-disable-fma-on-ios"
)
DEFAULT_OPT = "-O1"  # eager; torch.compile uses -O2 and different flags
FAKE_INSTANCE_TYPE = {"trn1": "trn1.32xlarge", "trn2": "trn2.48xlarge"}

# Probe compile: the (37,211)@(211,53) matmul from #118. On the window-4
# toolchain (neuronx-cc 2.28.7631.0+8f7b0cf5, trn2, LNC=2, -O1) it keyed to
# 360afa1c88f3a4a256dda985c5843023.
PROBE_SHAPE = (37, 211, 53)

# umbrella directory name for each project (trnsolver lives in trnsolve/)
PROJECTS = {
    "trnfft": "trnfft",
    "trnblas": "trnblas",
    "trnrand": "trnrand",
    "trnsolver": "trnsolve",
    "trnsparse": "trnsparse",
    "trntensor": "trntensor",
}
PROJECT_ALIASES = {"trnsolve": "trnsolver"}

# Environment variables that change the NEFF key (or the HLO bytes).
CONFIG_ENV = (
    "NEURON_PLATFORM_TARGET_OVERRIDE",
    "NEURON_LOGICAL_NC_CONFIG",
    "NEURON_COMPILER_OPT_LEVEL",
    "NEURON_CC_FLAGS",
    "TORCH_NEURONX_NEFF_KEY_FROM_HLO",
    "TORCH_NEURONX_NEFF_DISABLE_CACHE",
)
CONFIG_ENV_PREFIXES = ("TORCH_NEURONX_INT64_",)
# Toolchain fields that must match exactly, and ones reported for information.
STRICT_TOOLCHAIN = ("neuronxcc", "neuronxcc_key_version", "torch_neuronx", "nki")
INFO_TOOLCHAIN = ("torch", "python", "platform")
STRICT_CONFIG = ("target", "lnc", "opt", "compiler_args")

# capture workers merge into one output dir; serialise that step
_MERGE_LOCK = threading.Lock()

_KEY_RE = re.compile(r"^[0-9a-f]{32}$")
_MEMBER_RE = re.compile(
    r"^neff/([0-9a-f]{2})/([0-9a-f]{2})/([0-9a-f]{32})\.(neff|wrapped_neff_hlo)$"
)
_CACHE_SUFFIXES = (".neff", ".wrapped_neff_hlo")


class WarmError(RuntimeError):
    """A user-facing failure: printed without a traceback, exit code 2."""


def _log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Key derivation and cache layout
# ---------------------------------------------------------------------------


def key_config(
    *,
    target: str,
    lnc: str | int,
    opt: str,
    compiler_args: str,
    compiler_version: str,
    wlo: bool = False,
) -> str:
    """The configuration string hashed after the HLO (CompilationCacheKey.cpp)."""
    cfg = f"{target}_{lnc}_{opt}_{compiler_args}_v{compiler_version}"
    return cfg + "_wlo" if wlo else cfg


def neff_key(mlir: bytes, **config) -> str:
    """Persistent NEFF cache key for a StableHLO module (``module.mlir`` bytes).

    ``config`` takes the keyword arguments of :func:`key_config`.
    """
    try:
        import xxhash
    except ImportError as e:  # pragma: no cover - dependency message
        raise WarmError(
            "key derivation needs the 'xxhash' package (pip install xxhash)"
        ) from e
    return xxhash.xxh3_128_hexdigest(mlir + b"\xff" + key_config(**config).encode())


def neff_relpath(key: str, kind: str = "neff") -> str:
    if not _KEY_RE.match(key):
        raise ValueError(f"not a 32-hex NEFF key: {key!r}")
    return f"{key[0:2]}/{key[2:4]}/{key}.{kind}"


def iter_cache(cache_dir: str | os.PathLike):
    """Yield ``(key, kind, relpath, abspath)`` for each entry of a NEFF cache dir."""
    root = Path(cache_dir)
    if not root.is_dir():
        return
    for a in sorted(root.iterdir()):
        if not (a.is_dir() and len(a.name) == 2):
            continue
        for b in sorted(a.iterdir()):
            if not (b.is_dir() and len(b.name) == 2):
                continue
            for f in sorted(b.iterdir()):
                for suffix in _CACHE_SUFFIXES:
                    if f.name.endswith(suffix):
                        key = f.name[: -len(suffix)]
                        if (
                            _KEY_RE.match(key)
                            and key.startswith(a.name + b.name)
                            and f.is_file()
                        ):
                            yield key, suffix[1:], f"{a.name}/{b.name}/{f.name}", f
                        break


def cache_keys(cache_dir) -> set[str]:
    return {key for key, _kind, _rel, _p in iter_cache(cache_dir)}


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def merge_cache(src, dst) -> set[str]:
    """Move entries of NEFF cache ``src`` into ``dst``; return keys new to ``dst``.

    Entries already in ``dst`` are left alone: the cache is content-addressed,
    and NEFFs for one key differ only in ``debug_info*`` members.
    """
    new = set()
    for key, _kind, rel, path in iter_cache(src):
        dest = Path(dst) / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".tmp-{os.getpid()}-{dest.name}")
        shutil.copyfile(path, tmp)
        os.replace(tmp, dest)
        new.add(key)
    return new


def parse_compiler_txt(text: str) -> dict:
    """Parse the neuronx-cc command line in a preserved workdir's compiler.txt.

    Returns ``target``, ``lnc``, ``opt``, ``framework`` and ``compiler_args``
    (the remaining flags, in order, as the key's ``compiler_args`` string).
    """
    line = next((ln for ln in text.splitlines() if "compile" in ln), text).strip()
    tokens = shlex.split(line)
    if "compile" in tokens:
        tokens = tokens[tokens.index("compile") + 1 :]
    out = {"target": None, "lnc": None, "opt": None, "framework": None}
    rest: list[str] = []
    valued = {
        "--target": "target",
        "--lnc": "lnc",
        "--framework": "framework",
        "--output": None,
    }
    i = 0
    seen_input = False
    while i < len(tokens):
        tok = tokens[i]
        name, eq, val = tok.partition("=")
        if name in valued:
            if not eq:
                i += 1
                val = tokens[i] if i < len(tokens) else None
            if valued[name]:
                out[valued[name]] = val
        elif re.fullmatch(r"-O\d", tok):
            out["opt"] = tok
        elif not tok.startswith("-") and not seen_input:
            seen_input = True  # the input module path
        else:
            rest.append(tok)
        i += 1
    out["compiler_args"] = " ".join(rest)
    return out


# ---------------------------------------------------------------------------
# Toolchain identity, manifest, comparison
# ---------------------------------------------------------------------------

_IDENTITY_SNIPPET = r"""
import json, os, platform, sys
def dist(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None
tc = {"python": sys.version.split()[0], "platform": platform.machine() + "-" + platform.system()}
try:
    import neuronxcc
    tc["neuronxcc"] = getattr(neuronxcc, "__version__", None) or dist("neuronx-cc")
except Exception:
    tc["neuronxcc"] = None
for mod, dists, key in (("torch", ("torch",), "torch"),
                        ("torch_neuronx", ("torch-neuronx",), "torch_neuronx"),
                        ("nki", ("nki", "neuronx-nki"), "nki")):
    if os.environ.get("WARM_IDENTITY_SKIP_" + key.upper()):
        tc[key] = None
        continue
    try:
        m = __import__(mod)
        v = getattr(m, "__version__", None)
    except Exception:
        v = None
    if v is None:
        v = next((d for d in map(dist, dists) if d), None)
    tc[key] = str(v) if v is not None else None
# import torch_neuronx sets NEURONXCC_VERSION, the string that goes into the key
tc["neuronxcc_key_version"] = os.environ.get("NEURONXCC_VERSION")
print("WARM_IDENTITY " + json.dumps(tc, sort_keys=True))
"""

_PROBE_SNIPPET = r"""
import os, sys
import torch
import torch_neuronx  # noqa: F401
m, k, n = (int(x) for x in sys.argv[1:4])
a = torch.empty(m, k, device="neuron")
b = torch.empty(k, n, device="neuron")
c = a @ b
if os.environ.get("TORCH_NEURONX_COMPILE_ONLY") == "1":
    neuron = getattr(torch, "neuron", None)
    if neuron is not None and hasattr(neuron, "synchronize"):
        neuron.synchronize()
else:
    c.cpu()
print("WARM_PROBE_DONE")
"""


def _run_snippet(python: str, snippet: str, args=(), env=None, timeout=900) -> str:
    with tempfile.TemporaryDirectory(prefix="warm-snippet-") as cwd:
        # cwd is an empty dir, so sys.path[0] ('') can't shadow anything
        proc = subprocess.run(
            [python, "-c", snippet, *map(str, args)],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
        raise WarmError(
            f"{python} snippet failed (exit {proc.returncode}):\n  " + "\n  ".join(tail)
        )
    return proc.stdout


def collect_identity(
    python: str = sys.executable, *, skip_torch_neuronx: bool = False
) -> dict:
    env = dict(os.environ)
    # import only; compile-only mode keeps a CPU host from looking for a device
    env.setdefault("TORCH_NEURONX_COMPILE_ONLY", "1")
    if skip_torch_neuronx:
        env["WARM_IDENTITY_SKIP_TORCH_NEURONX"] = "1"
    out = _run_snippet(python, _IDENTITY_SNIPPET, env=env, timeout=600)
    for line in out.splitlines():
        if line.startswith("WARM_IDENTITY "):
            return json.loads(line[len("WARM_IDENTITY ") :])
    raise WarmError("identity probe printed no result")


def config_env(env) -> dict:
    """The key-relevant environment variables that are set in ``env``."""
    out = {}
    for name, val in env.items():
        if name in CONFIG_ENV or name.startswith(CONFIG_ENV_PREFIXES):
            out[name] = val
    return dict(sorted(out.items()))


def config_from_env(env, target=None, lnc=None) -> dict:
    """Resolve target/LNC/opt/compiler_args from the environment alone.

    Used when no probe compile is available; ``compiler_args`` is then the
    documented SDK 2.33 base string plus NEURON_CC_FLAGS, not an observation.
    """
    target = target or env.get("NEURON_PLATFORM_TARGET_OVERRIDE")
    lnc = lnc or env.get("NEURON_LOGICAL_NC_CONFIG")
    if not target or not lnc:
        raise WarmError(
            "target/LNC unknown: run with the probe, or pass --target and --lnc"
        )
    opt = env.get("NEURON_COMPILER_OPT_LEVEL") or DEFAULT_OPT
    if not opt.startswith("-O"):
        opt = f"-O{opt.lstrip('O')}"
    args = BASE_COMPILER_ARGS
    if env.get("NEURON_CC_FLAGS"):
        args = f"{args} {env['NEURON_CC_FLAGS'].strip()}"
    return {
        "target": target,
        "lnc": str(lnc),
        "opt": opt,
        "compiler_args": args,
        "source": "env",
    }


def find_module_dirs(root) -> list[Path]:
    """Directories under ``root`` that hold a preserved ``module.mlir``."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.rglob("module.mlir") if p.is_file())


def _read(p: Path) -> str | None:
    return p.read_text(errors="replace") if p.is_file() else None


def describe_module_dir(d: Path, compiler_version: str | None) -> dict:
    mlir = (d / "module.mlir").read_bytes()
    comp = _read(d / "compiler.txt")
    rec = {
        "mlir": mlir.decode("utf-8", errors="replace"),
        "mlir_sha256": hashlib.sha256(mlir).hexdigest(),
        "compiler_txt": comp,
        "metadata": _read(d / "metadata.txt"),
        "computed_key": None,
    }
    if comp:
        parsed = parse_compiler_txt(comp)
        rec["parsed"] = parsed
        if compiler_version and all(parsed[k] for k in ("target", "lnc", "opt")):
            try:
                rec["computed_key"] = neff_key(
                    mlir,
                    target=parsed["target"],
                    lnc=parsed["lnc"],
                    opt=parsed["opt"],
                    compiler_args=parsed["compiler_args"],
                    compiler_version=compiler_version,
                )
            except WarmError:
                pass
    return rec


def run_probe(
    python: str, env: dict, compiler_version: str | None, shape=PROBE_SHAPE
) -> dict:
    """Compile the probe matmul with private cache dirs; report its keys.

    ``env`` decides the mode: with TORCH_NEURONX_COMPILE_ONLY=1 it runs on a
    CPU host, otherwise it needs a NeuronCore. Cache dirs are overridden.
    """
    with tempfile.TemporaryDirectory(prefix="warm-probe-") as tmp:
        t = Path(tmp)
        env = dict(env)
        env.update(
            TORCH_NEURONX_NEFF_CACHE_DIR=str(t / "neff"),
            TORCH_NEURONX_HLO_CACHE_DIR=str(t / "hlo"),
            TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=str(t / "local"),
            TORCH_NEURONX_DEBUG_DIR=str(t / "debug"),
            TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS="1",
            NKI_ENABLE_TRACE_CACHE="0",
        )
        env.pop("TORCH_NEURONX_NEFF_DISABLE_CACHE", None)
        _run_snippet(python, _PROBE_SNIPPET, args=shape, env=env)
        keys = sorted(cache_keys(t / "neff"))
        modules = [
            describe_module_dir(d, compiler_version)
            for d in find_module_dirs(t / "debug")
        ]
    computed = sorted(m["computed_key"] for m in modules if m["computed_key"])
    return {
        "shape": list(shape),
        "keys": keys,
        "modules": modules,
        "key_formula_verified": bool(keys) and computed == keys,
    }


def build_manifest(
    identity: dict,
    env,
    probe: dict | None,
    *,
    target=None,
    lnc=None,
    instance_type=None,
) -> dict:
    """Assemble a manifest from an identity dict, the box env and a probe."""
    config = None
    if probe:
        parsed = next(
            (m.get("parsed") for m in probe["modules"] if m.get("parsed")), None
        )
        if parsed and all(parsed[k] for k in ("target", "lnc", "opt")):
            config = {k: parsed[k] for k in STRICT_CONFIG}
            config["source"] = "probe"
    if config is None:
        config = config_from_env(env, target=target, lnc=lnc)
    if target and target != config["target"] or lnc and str(lnc) != config["lnc"]:
        raise WarmError(
            f"--target/--lnc disagree with the probe compile ({config['target']}, lnc {config['lnc']})"
        )
    config["instance_type"] = instance_type or FAKE_INSTANCE_TYPE.get(config["target"])
    toolchain = dict(identity)
    if not toolchain.get("neuronxcc_key_version"):
        toolchain["neuronxcc_key_version"] = toolchain.get("neuronxcc")
    return {
        "format": MANIFEST_FORMAT,
        "version": FORMAT_VERSION,
        "created": _now(),
        "toolchain": toolchain,
        "config": config,
        "env": config_env(env),
        "probe": probe,
    }


def load_manifest(path) -> dict:
    try:
        m = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise WarmError(f"cannot read manifest {path}: {e}") from e
    if m.get("format") != MANIFEST_FORMAT:
        raise WarmError(f"{path} is not a {MANIFEST_FORMAT} file")
    if m.get("version") != FORMAT_VERSION:
        raise WarmError(
            f"{path}: manifest version {m.get('version')} unsupported (want {FORMAT_VERSION})"
        )
    return m


def manifest_tag(manifest: dict) -> str:
    """``neuronxcc-<v>_torch-neuronx-<v>_<target>-lnc<N><opt>``, filename-safe."""
    tc, cfg = manifest["toolchain"], manifest["config"]
    tag = (
        f"neuronxcc-{tc.get('neuronxcc')}_torch-neuronx-{tc.get('torch_neuronx')}"
        f"_{cfg['target']}-lnc{cfg['lnc']}{cfg['opt']}"
    )
    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


def compiler_version(manifest: dict) -> str:
    tc = manifest["toolchain"]
    v = tc.get("neuronxcc_key_version") or tc.get("neuronxcc")
    if not v:
        raise WarmError("manifest has no neuronx-cc version")
    return v


@dataclass
class Finding:
    level: str  # "FAIL" | "INFO" | "OK"
    field: str
    expected: object
    found: object
    note: str = ""

    def line(self) -> str:
        if self.level == "OK":
            return f"OK    {self.field:<34} {self.found}"
        msg = f"{self.level:<5} {self.field:<34} manifest={self.expected!r} local={self.found!r}"
        return msg + (f"  ({self.note})" if self.note else "")


def compare_toolchain(
    manifest: dict, identity: dict, env, *, fields=STRICT_TOOLCHAIN
) -> list[Finding]:
    """Compare a manifest with a local identity and environment.

    Strict toolchain fields must be equal. A key-relevant environment variable
    that is set locally must equal the manifest's value (unset is fine: the
    tool exports the manifest's values itself). Other fields are INFO.
    """
    out = []
    tc = manifest["toolchain"]
    for f in fields:
        exp, got = tc.get(f), identity.get(f)
        if f == "neuronxcc_key_version" and got is None:
            continue  # only known after `import torch_neuronx`
        if exp == got:
            out.append(Finding("OK", f"toolchain.{f}", exp, got))
        else:
            out.append(
                Finding(
                    "FAIL", f"toolchain.{f}", exp, got, "toolchain must match exactly"
                )
            )
    for f in INFO_TOOLCHAIN:
        if tc.get(f) != identity.get(f):
            out.append(
                Finding(
                    "INFO",
                    f"toolchain.{f}",
                    tc.get(f),
                    identity.get(f),
                    "not part of the NEFF key",
                )
            )
    menv, lenv = manifest.get("env", {}), config_env(env)
    exported = {
        "NEURON_PLATFORM_TARGET_OVERRIDE": manifest["config"]["target"],
        "NEURON_LOGICAL_NC_CONFIG": str(manifest["config"]["lnc"]),
    }
    for name in sorted(set(menv) | set(lenv)):
        exp = menv.get(name, exported.get(name))
        got = lenv.get(name)
        if got is None or got == exp:
            continue
        out.append(
            Finding("FAIL", f"env.{name}", exp, got, "unset it or match the box")
        )
    return out


def compare_probe(manifest: dict, probe: dict) -> Finding:
    exp = (manifest.get("probe") or {}).get("keys")
    got = probe.get("keys")
    if exp == got:
        return Finding("OK", "probe.keys", exp, got)
    return Finding(
        "FAIL",
        "probe.keys",
        exp,
        got,
        "same probe program, different key: HLO bytes (torch-neuronx build) or compiler config differ",
    )


def verify_manifest_probe(manifest: dict) -> list[str]:
    """Recompute the manifest's probe keys from its embedded module text.

    Returns a list of problems (empty when every module reproduces a key the
    box wrote). Needs no Neuron software, only xxhash.
    """
    probe = manifest.get("probe")
    if not probe:
        return ["manifest has no probe"]
    problems = []
    version = compiler_version(manifest)
    keys = set(probe.get("keys", []))
    for i, mod in enumerate(probe.get("modules", [])):
        if not mod.get("compiler_txt"):
            problems.append(f"module {i}: no compiler.txt")
            continue
        p = parse_compiler_txt(mod["compiler_txt"])
        k = neff_key(
            mod["mlir"].encode(),
            target=p["target"],
            lnc=p["lnc"],
            opt=p["opt"],
            compiler_args=p["compiler_args"],
            compiler_version=version,
        )
        if k not in keys:
            problems.append(f"module {i}: computed key {k} not among the box's keys")
    return problems


def compile_only_env(
    manifest: dict, base_env, workdir, *, preserve: bool = False
) -> dict:
    """Environment for a device-free compile-only run matching ``manifest``."""
    cfg = manifest["config"]
    workdir = Path(workdir)
    env = {
        k: v
        for k, v in base_env.items()
        if not (k in CONFIG_ENV or k.startswith(CONFIG_ENV_PREFIXES))
    }
    env.update(manifest.get("env", {}))
    env.pop("TORCH_NEURONX_NEFF_DISABLE_CACHE", None)
    env.pop("TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS", None)
    env.update(
        TORCH_NEURONX_COMPILE_ONLY="1",
        NEURON_PLATFORM_TARGET_OVERRIDE=cfg["target"],
        NEURON_LOGICAL_NC_CONFIG=str(cfg["lnc"]),
        NKI_ENABLE_TRACE_CACHE="0",
        TORCH_NEURONX_NEFF_CACHE_DIR=str(workdir / "neff"),
        TORCH_NEURONX_HLO_CACHE_DIR=str(workdir / "hlo"),
        TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=str(workdir / "local"),
        TORCH_NEURONX_DEBUG_DIR=str(workdir / "debug"),
    )
    if cfg.get("instance_type"):
        env["NEURON_RT_FAKE_INSTANCE_TYPE"] = cfg["instance_type"]
    if preserve:
        env["TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS"] = "1"
    return env


def _with_plugin_path(env: dict) -> dict:
    env = dict(env)
    parts = [str(SCRIPTS_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_check(
    manifest: dict, python: str, *, probe: bool, compiler_only: bool = False
) -> list[Finding]:
    identity = collect_identity(python, skip_torch_neuronx=compiler_only)
    fields = ("neuronxcc",) if compiler_only else STRICT_TOOLCHAIN
    findings = compare_toolchain(manifest, identity, os.environ, fields=fields)
    if probe and not compiler_only:
        if not manifest.get("probe"):
            findings.append(
                Finding("INFO", "probe", None, None, "manifest has no probe; skipped")
            )
        elif identity.get("torch_neuronx") is None:
            findings.append(
                Finding(
                    "FAIL",
                    "probe",
                    "torch_neuronx",
                    None,
                    "torch-neuronx not importable",
                )
            )
        else:
            with tempfile.TemporaryDirectory(prefix="warm-check-") as tmp:
                env = compile_only_env(manifest, os.environ, tmp)
            local = run_probe(python, env, compiler_version(manifest))
            findings.append(compare_probe(manifest, local))
    return findings


def _require_match(
    manifest: dict,
    python: str,
    *,
    force: bool,
    probe: bool = True,
    compiler_only=False,
    tag="check",
):
    findings = run_check(manifest, python, probe=probe, compiler_only=compiler_only)
    for f in findings:
        if f.level != "OK":
            _log(tag, f.line())
    bad = [f for f in findings if f.level == "FAIL"]
    if bad and not force:
        raise WarmError(
            f"{len(bad)} toolchain mismatch(es) with the manifest; the output would only "
            "produce cache misses. Fix the toolchain, or pass --force to run anyway."
        )
    if bad:
        _log(tag, "--force: continuing despite mismatches; expect misses on the box")
    else:
        _log(tag, "toolchain matches the manifest")


# ---------------------------------------------------------------------------
# Index, pack, unpack
# ---------------------------------------------------------------------------


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sources(path) -> dict[str, list[str]]:
    """``sources.jsonl`` -> {key: sorted unique source labels}."""
    out: dict[str, set[str]] = {}
    if not path or not Path(path).is_file():
        return {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        label = rec["source"]
        if rec.get("project"):
            label = f"{rec['project']}:{label}"
        out.setdefault(rec["key"], set()).add(label)
    return {k: sorted(v) for k, v in out.items()}


def build_index(
    cache_dir, *, sources: dict | None = None, manifest: dict | None = None
) -> dict:
    sources = sources or {}
    entries = []
    for key, kind, rel, path in iter_cache(cache_dir):
        entries.append(
            {
                "key": key,
                "kind": kind,
                "path": f"neff/{rel}",
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
                "sources": sources.get(key, []),
            }
        )
    return {
        "format": INDEX_FORMAT,
        "version": FORMAT_VERSION,
        "created": _now(),
        "manifest_tag": manifest_tag(manifest) if manifest else None,
        "manifest": manifest,
        "totals": {
            "files": len(entries),
            "keys": len({e["key"] for e in entries}),
            "bytes": sum(e["size"] for e in entries),
        },
        "entries": entries,
    }


def validate_index(index: dict) -> None:
    if index.get("format") != INDEX_FORMAT or index.get("version") != FORMAT_VERSION:
        raise WarmError("not a trnsci NEFF index (format/version)")
    seen = set()
    for e in index["entries"]:
        m = _MEMBER_RE.match(e["path"])
        if (
            not m
            or m.group(3) != e["key"]
            or m.group(1) + m.group(2) != e["key"][:4]
            or m.group(4) != e["kind"]
        ):
            raise WarmError(f"bad index entry path {e['path']!r}")
        if e["path"] in seen:
            raise WarmError(f"duplicate index entry {e['path']!r}")
        seen.add(e["path"])


def _resolve_compression(compression: str, out: Path | None) -> str:
    if compression == "auto":
        if out is not None:
            for suffix, comp in (
                (".tar.zst", "zst"),
                (".tar.gz", "gz"),
                (".tgz", "gz"),
                (".tar", "none"),
            ):
                if out.name.endswith(suffix):
                    return comp
        return "zst" if shutil.which("zstd") else "gz"
    return compression


def _tarinfo(name: str, size: int) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size, ti.mode, ti.mtime = size, 0o644, 0
    return ti


def resolve_capture_dir(path) -> tuple[Path, Path | None, Path | None]:
    """Accept a capture/recompile output dir or a bare NEFF cache dir.

    Returns ``(cache_dir, sources_jsonl or None, manifest_json or None)``.
    """
    p = Path(path)
    if (p / "neff").is_dir() and not cache_keys(p):
        src, man = p / "sources.jsonl", p / "manifest.json"
        return (
            p / "neff",
            src if src.is_file() else None,
            man if man.is_file() else None,
        )
    return p, None, None


def pack(
    cache_dir, out=None, *, manifest=None, sources=None, compression="auto"
) -> tuple[Path, dict]:
    """Write ``cache_dir`` as a tar (index.json first) plus a sidecar index."""
    index = build_index(cache_dir, sources=sources, manifest=manifest)
    if not index["entries"]:
        raise WarmError(f"no NEFFs in {cache_dir}")
    out = Path(out) if out else None
    comp = _resolve_compression(compression, out)
    if comp == "zst" and not shutil.which("zstd"):
        raise WarmError(
            "zstd compression needs the zstd binary on PATH (or use --compression gz)"
        )
    ext = {"zst": ".tar.zst", "gz": ".tar.gz", "none": ".tar"}[comp]
    if out is None:
        tag = index["manifest_tag"] or "neff-" + _dt.datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        out = Path.cwd() / (
            f"neff-{tag}{ext}" if index["manifest_tag"] else f"{tag}{ext}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(index, indent=1, sort_keys=True).encode()
    tar_path = out if comp != "zst" else out.with_name(out.name + ".tmp.tar")
    mode = "w:gz" if comp == "gz" else "w"
    kwargs = {"compresslevel": 6} if comp == "gz" else {}
    try:
        with tarfile.open(tar_path, mode, format=tarfile.PAX_FORMAT, **kwargs) as tf:
            tf.addfile(_tarinfo("index.json", len(blob)), io.BytesIO(blob))
            root = Path(cache_dir)
            for e in index["entries"]:
                p = root / e["path"][len("neff/") :]
                with open(p, "rb") as fh:
                    tf.addfile(_tarinfo(e["path"], e["size"]), fh)
        if comp == "zst":
            subprocess.run(
                ["zstd", "-q", "-f", "-T0", str(tar_path), "-o", str(out)], check=True
            )
    finally:
        if comp == "zst":
            tar_path.unlink(missing_ok=True)
    Path(f"{out}.index.json").write_bytes(blob)
    return out, index


def _open_archive(path: Path):
    """Return ``(tarfile, cleanup)`` reading ``path`` as a stream."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == b"\x28\xb5\x2f\xfd":
        if not shutil.which("zstd"):
            raise WarmError("archive is zstd-compressed; install zstd")
        proc = subprocess.Popen(
            ["zstd", "-q", "-dc", str(path)], stdout=subprocess.PIPE
        )
        tf = tarfile.open(fileobj=proc.stdout, mode="r|")

        def cleanup():
            # member sizes and sha256 are checked against the index, so a
            # zstd failure surfaces there; just make sure the child is gone
            tf.close()
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()

        return tf, cleanup
    tf = tarfile.open(path, mode="r|*")
    return tf, tf.close


def read_index(path) -> dict:
    """Read the index of a packed archive (sidecar if present, else member)."""
    side = Path(f"{path}.index.json")
    if side.is_file():
        index = json.loads(side.read_text())
    else:
        tf, cleanup = _open_archive(Path(path))
        try:
            first = tf.next()
            if first is None or first.name != "index.json":
                raise WarmError(f"{path}: first member is not index.json")
            index = json.loads(tf.extractfile(first).read())
        finally:
            cleanup()
    validate_index(index)
    return index


def unpack(archive, into, *, overwrite: bool = False) -> dict:
    """Install the NEFFs of ``archive`` into each cache dir in ``into``.

    Only members named like ``neff/<aa>/<bb>/<key>.neff`` that are listed in
    the embedded index and match its size and sha256 are written; nothing is
    extracted by path. Existing files are kept unless ``overwrite``.
    """
    into = [Path(d) for d in into]
    stats = {"installed": 0, "skipped": 0, "bytes": 0, "index": None}
    tf, cleanup = _open_archive(Path(archive))
    try:
        first = tf.next()
        if first is None or first.name != "index.json" or not first.isfile():
            raise WarmError(f"{archive}: first member is not index.json")
        index = json.loads(tf.extractfile(first).read())
        validate_index(index)
        stats["index"] = index
        expected = {e["path"]: e for e in index["entries"]}
        seen = set()
        # tf.next(), not `for m in tf`: iterating restarts at index.json
        while (m := tf.next()) is not None:
            if m.name not in expected or not m.isfile():
                raise WarmError(f"{archive}: unexpected member {m.name!r}")
            e = expected[m.name]
            data = tf.extractfile(m).read()
            if (
                len(data) != e["size"]
                or hashlib.sha256(data).hexdigest() != e["sha256"]
            ):
                raise WarmError(
                    f"{archive}: {m.name} does not match its index entry (corrupt archive?)"
                )
            seen.add(m.name)
            rel = m.name[len("neff/") :]
            for d in into:
                dest = d / rel
                if dest.exists() and not overwrite:
                    stats["skipped"] += 1
                    continue
                _atomic_write(dest, data)
                stats["installed"] += 1
                stats["bytes"] += len(data)
        missing = set(expected) - seen
        if missing:
            raise WarmError(
                f"{archive}: {len(missing)} indexed member(s) missing (truncated archive?)"
            )
    finally:
        cleanup()
    return stats


# ---------------------------------------------------------------------------
# Counters (verify)
# ---------------------------------------------------------------------------


def _counter(counters: dict, suffix: str) -> int:
    return sum(v for k, v in counters.items() if k.endswith(suffix))


def aggregate_counters(runs: list[dict]) -> dict:
    """Sum CompilationCache counters over runs and compute the hit rate.

    hit_rate = PersistentHits / (PersistentHits + TotalCompilations); None
    when nothing was looked up.
    """
    total: dict[str, int] = {}
    for r in runs:
        for k, v in r.items():
            total[k] = total.get(k, 0) + int(v)
    hits = _counter(total, "PersistentHits")
    comps = _counter(total, "TotalCompilations")
    rate = hits / (hits + comps) if hits + comps else None
    return {
        "counters": dict(sorted(total.items())),
        "persistent_hits": hits,
        "compilations": comps,
        "hit_rate": rate,
    }


# ---------------------------------------------------------------------------
# Running pytest (capture, verify)
# ---------------------------------------------------------------------------


@dataclass
class Unit:
    project: str
    cwd: Path
    targets: list[str]
    label: str


@dataclass
class UnitResult:
    unit: Unit
    exit_code: int | None
    seconds: float
    new_keys: set[str] = field(default_factory=set)
    produced: int = 0
    counters: dict = field(default_factory=dict)
    log: str = ""
    timed_out: bool = False


def parse_projects(spec: str) -> list[str]:
    if spec in ("", "all"):
        return list(PROJECTS)
    out = []
    for name in spec.split(","):
        name = PROJECT_ALIASES.get(name.strip(), name.strip())
        if name not in PROJECTS:
            raise WarmError(
                f"unknown project {name!r}; choose from {', '.join(PROJECTS)}"
            )
        out.append(name)
    return out


_NEURON_MARK = re.compile(r"mark\.neuron\b")


def discover_units(
    src_root, projects, *, by: str = "file", prefilter: bool = True
) -> list[Unit]:
    """One pytest invocation per test file (``by="file"``) or per project."""
    units = []
    for proj in projects:
        pdir = Path(src_root) / PROJECTS[proj]
        tdir = pdir / "tests"
        if not tdir.is_dir():
            raise WarmError(
                f"{proj}: no tests/ under {pdir} (set --src to the umbrella checkout)"
            )
        if by == "project":
            units.append(Unit(proj, pdir, ["tests"], proj))
            continue
        for f in sorted(tdir.rglob("test_*.py")):
            if prefilter and not _NEURON_MARK.search(f.read_text(errors="replace")):
                continue
            rel = f.relative_to(pdir).as_posix()
            units.append(Unit(proj, pdir, [rel], f"{proj}/{rel}"))
    return units


_P_FLAG: dict[str, bool] = {}


def _python_flags(python: str) -> list[str]:
    """``-P`` (don't put the cwd on sys.path) where the interpreter has it."""
    if python not in _P_FLAG:
        out = subprocess.run(
            [python, "-c", "import sys; print(sys.version_info >= (3, 11))"],
            capture_output=True,
            text=True,
        )
        _P_FLAG[python] = out.stdout.strip() == "True"
    return ["-P"] if _P_FLAG[python] else []


def pytest_command(
    python: str, targets, *, marker: str, optimize: bool, extra=()
) -> list[str]:
    """The pytest invocation for one unit.

    ``optimize`` (compile-only capture) runs ``python -O`` *and*
    ``--assert=plain``: pytest's assertion rewriting turns test-module asserts
    into explicit raises that survive ``-O``, so without it a test still stops
    at its first comparison against placeholder data.
    """
    cmd = [python, *(["-O"] if optimize else []), *_python_flags(python)]
    cmd += ["-m", "pytest", "-p", PLUGIN, "-p", "no:cacheprovider", "-q"]
    if optimize:
        cmd.append("--assert=plain")
    return [*cmd, "-m", marker, *targets, *extra]


def _run(cmd, *, cwd, env, timeout, log_path: Path) -> tuple[int | None, bool]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        log.write("$ " + shlex.join(cmd) + f"\n# cwd={cwd}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            # kill the whole session: neuronx-cc runs as grandchildren
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return None, True


def _load_counters(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def run_capture_unit(unit: Unit, manifest: dict, out: Path, args) -> UnitResult:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", unit.label)
    work = out / "work" / safe
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = _with_plugin_path(
        compile_only_env(manifest, os.environ, work, preserve=args.keep_mlir)
    )
    env.update(
        TRNSCI_WARM_SOURCES=str(work / "sources.jsonl"),
        TRNSCI_WARM_COUNTERS=str(work / "counters.json"),
        TRNSCI_WARM_UNIT=unit.project,
    )
    cmd = pytest_command(
        args.python,
        unit.targets,
        marker=args.marker,
        optimize=True,
        extra=args.pytest_args,
    )
    log = out / "logs" / f"{safe}.log"
    t0 = time.monotonic()
    code, timed_out = _run(
        cmd, cwd=unit.cwd, env=env, timeout=args.timeout, log_path=log
    )
    res = UnitResult(
        unit, code, time.monotonic() - t0, log=str(log), timed_out=timed_out
    )
    res.produced = len(cache_keys(work / "neff"))
    res.counters = _load_counters(work / "counters.json")
    with _MERGE_LOCK:
        res.new_keys = merge_cache(work / "neff", out / "neff")
        src = work / "sources.jsonl"
        if src.is_file():
            with open(out / "sources.jsonl", "a") as fh:
                fh.write(src.read_text())
    if args.keep_mlir:
        for i, d in enumerate(find_module_dirs(work / "debug")):
            dest = out / "mlir" / safe / f"{i:04d}"
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("module.mlir", "compiler.txt", "metadata.txt"):
                if (d / name).is_file():
                    shutil.copy2(d / name, dest / name)
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return res


def _unit_status(r: UnitResult) -> str:
    if r.timed_out:
        return "TIMEOUT"
    # 0 pass, 1 test failures (expected on placeholder data), 5 nothing collected
    return {
        0: "ok",
        1: "ok (tests failed on placeholder data)",
        5: "no neuron tests",
    }.get(r.exit_code, "ERROR")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_manifest(args) -> int:
    identity = collect_identity(args.python)
    if not identity.get("neuronxcc"):
        raise WarmError(
            "neuronx-cc is not importable here; run `manifest` in the box's Neuron venv"
        )
    probe = None
    if not args.no_probe:
        if identity.get("torch_neuronx") is None:
            raise WarmError(
                "torch-neuronx is not importable; pass --no-probe with --target/--lnc"
            )
        _log("manifest", "probe compile on the device (one small matmul)")
        version = identity.get("neuronxcc_key_version") or identity["neuronxcc"]
        probe = run_probe(args.python, dict(os.environ), version)
        _log(
            "manifest",
            f"probe keys {probe['keys']} formula verified={probe['key_formula_verified']}",
        )
    m = build_manifest(
        identity,
        os.environ,
        probe,
        target=args.target,
        lnc=args.lnc,
        instance_type=args.instance_type,
    )
    Path(args.output).write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    _log("manifest", f"wrote {args.output} ({manifest_tag(m)})")
    return 0


def cmd_check(args) -> int:
    m = load_manifest(args.manifest)
    if m.get("probe"):
        try:
            problems = verify_manifest_probe(m)
        except WarmError as e:
            problems = [str(e)]
        for p in problems:
            _log("check", f"INFO  manifest probe: {p}")
    findings = run_check(
        m, args.python, probe=not args.no_probe, compiler_only=args.compiler_only
    )
    for f in findings:
        print(f.line())
    bad = [f for f in findings if f.level == "FAIL"]
    if bad:
        print(
            f"\n{len(bad)} mismatch(es). NEFFs built here would only be cache misses on the box. "
            "The warming host needs the box's exact neuronx-cc version and torch-neuronx build "
            "(boot the same DLAMI on a CPU instance)."
        )
        return 1
    print(f"\nmatch: {manifest_tag(m)}")
    return 0


def _out_manifest(out: Path, m: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def cmd_capture(args) -> int:
    m = load_manifest(args.manifest)
    _require_match(
        m, args.python, force=args.force, probe=not args.no_probe, tag="capture"
    )
    projects = parse_projects(args.projects)
    units = discover_units(
        args.src, projects, by=args.by, prefilter=not args.no_prefilter
    )
    if not units:
        raise WarmError("no test files with neuron tests found")
    out = Path(args.out)
    _out_manifest(out, m)
    _log(
        "capture",
        f"{len(units)} unit(s), {args.workers} worker(s), cache {out / 'neff'}",
    )
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_capture_unit, u, m, out, args): u for u in units}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            _log(
                "capture",
                f"{i}/{len(units)} {r.unit.label}: {_unit_status(r)}, {r.produced} NEFF(s), "
                f"{len(r.new_keys)} new, {r.seconds:.1f}s",
            )
    summary = {
        "created": _now(),
        "manifest_tag": manifest_tag(m),
        "units": [
            {
                "label": r.unit.label,
                "exit_code": r.exit_code,
                "status": _unit_status(r),
                "seconds": round(r.seconds, 2),
                "neffs": r.produced,
                "new": len(r.new_keys),
                "counters": r.counters,
                "log": r.log,
            }
            for r in sorted(results, key=lambda r: r.unit.label)
        ],
        "total_neffs": len(cache_keys(out / "neff")),
    }
    (out / "capture.json").write_text(json.dumps(summary, indent=2) + "\n")
    errors = [r for r in results if _unit_status(r) in ("ERROR", "TIMEOUT")]
    _log(
        "capture",
        f"{summary['total_neffs']} NEFF(s) in {out / 'neff'}; summary {out / 'capture.json'}",
    )
    for r in errors:
        _log("capture", f"{_unit_status(r)}: {r.unit.label} (see {r.log})")
    return 1 if errors and args.strict else 0


def _neuronx_cc(args) -> str:
    if args.neuronx_cc:
        return args.neuronx_cc
    cand = Path(args.python).parent / "neuronx-cc"
    found = str(cand) if cand.exists() else shutil.which("neuronx-cc")
    if not found:
        raise WarmError("neuronx-cc not found; pass --neuronx-cc")
    return found


def plan_recompile(module_dir: Path, manifest: dict) -> dict:
    """Decide flags and key for one recorded module dir.

    Uses the module's own compiler.txt when present (it carries per-op extras
    such as int64-policy flags) but only if its target/LNC/opt equal the
    manifest's; otherwise the manifest's configuration.
    """
    cfg = manifest["config"]
    mlir = (module_dir / "module.mlir").read_bytes()
    flags = {
        "target": cfg["target"],
        "lnc": str(cfg["lnc"]),
        "opt": cfg["opt"],
        "compiler_args": cfg["compiler_args"],
    }
    origin = "manifest"
    comp = _read(module_dir / "compiler.txt")
    if comp:
        p = parse_compiler_txt(comp)
        if (p["target"], p["lnc"], p["opt"]) != (
            flags["target"],
            flags["lnc"],
            flags["opt"],
        ):
            return {
                "skip": f"recorded for {p['target']}/lnc{p['lnc']}/{p['opt']}, manifest is "
                f"{flags['target']}/lnc{flags['lnc']}/{flags['opt']}"
            }
        flags["compiler_args"] = p["compiler_args"]
        origin = "compiler.txt"
    key = neff_key(mlir, compiler_version=compiler_version(manifest), **flags)
    return {"key": key, "flags": flags, "origin": origin, "mlir": mlir}


def recompile_one(
    module_dir: Path, plan: dict, cc: str, out: Path, timeout: float
) -> tuple[str, str]:
    f = plan["flags"]
    with tempfile.TemporaryDirectory(prefix="warm-cc-", dir=out) as tmp:
        t = Path(tmp)
        (t / "module.mlir").write_bytes(plan["mlir"])
        cmd = [
            cc,
            "compile",
            "module.mlir",
            "--framework",
            "XLA",
            "--target",
            f["target"],
            "--lnc",
            f["lnc"],
        ]
        cmd += [f["opt"], *shlex.split(f["compiler_args"]), "--output", "module.neff"]
        code, timed_out = _run(
            cmd, cwd=t, env=dict(os.environ), timeout=timeout, log_path=t / "log.txt"
        )
        if timed_out or code != 0 or not (t / "module.neff").is_file():
            log = (t / "log.txt").read_text(errors="replace").strip().splitlines()[-5:]
            return "failed", "timeout" if timed_out else f"exit {code}: " + " | ".join(
                log
            )
        dest = out / "neff" / neff_relpath(plan["key"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(t / "module.neff", dest)
    return "compiled", ""


def cmd_recompile(args) -> int:
    m = load_manifest(args.manifest)
    _require_match(
        m,
        args.python,
        force=args.force,
        probe=False,
        compiler_only=True,
        tag="recompile",
    )
    cc = _neuronx_cc(args)
    out = Path(args.out)
    _out_manifest(out, m)
    corpus = Path(args.corpus)
    dirs = find_module_dirs(corpus)
    if not dirs:
        raise WarmError(f"no module.mlir under {corpus}")
    have = cache_keys(out / "neff")
    jobs, counts = [], {"compiled": 0, "cached": 0, "skipped": 0, "failed": 0}
    for d in dirs:
        plan = plan_recompile(d, m)
        rel = d.relative_to(corpus).as_posix() or "."
        if "skip" in plan:
            counts["skipped"] += 1
            _log("recompile", f"skip {rel}: {plan['skip']}")
            continue
        if plan["key"] in have:
            counts["cached"] += 1
            continue
        have.add(plan["key"])
        jobs.append((d, rel, plan))
    _log(
        "recompile",
        f"{len(dirs)} module(s): {len(jobs)} to compile, {counts['cached']} already in the cache",
    )
    with (
        cf.ThreadPoolExecutor(max_workers=args.workers) as ex,
        open(out / "sources.jsonl", "a") as src,
    ):
        futs = {
            ex.submit(recompile_one, d, plan, cc, out, args.timeout): (rel, plan)
            for d, rel, plan in jobs
        }
        for fut in cf.as_completed(futs):
            rel, plan = futs[fut]
            status, why = fut.result()
            counts[status] += 1
            if status == "compiled":
                src.write(
                    json.dumps(
                        {"key": plan["key"], "source": f"module:{rel}"}, sort_keys=True
                    )
                    + "\n"
                )
            _log(
                "recompile",
                f"{status} {rel} -> {plan['key']} [{plan['origin']}] {why}".rstrip(),
            )
    _log(
        "recompile",
        ", ".join(f"{k} {v}" for k, v in counts.items()) + f"; cache {out / 'neff'}",
    )
    return 1 if counts["failed"] else 0


def cmd_pack(args) -> int:
    cache, sources_path, manifest_path = resolve_capture_dir(args.cache)
    if args.sources:
        sources_path = Path(args.sources)
    if args.manifest:
        manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path) if manifest_path else None
    out, index = pack(
        cache,
        args.output,
        manifest=manifest,
        sources=load_sources(sources_path),
        compression=args.compression,
    )
    t = index["totals"]
    _log(
        "pack",
        f"{out} ({t['keys']} keys, {t['files']} files, {t['bytes'] / 1e6:.1f} MB); index {out}.index.json",
    )
    if manifest is None:
        _log(
            "pack",
            "warning: no manifest embedded; `unpack --check` won't be able to compare toolchains",
        )
    return 0


def cmd_unpack(args) -> int:
    if args.check:
        index = read_index(args.archive)
        if not index.get("manifest"):
            raise WarmError("archive has no embedded manifest to check against")
        _require_match(
            index["manifest"], args.python, force=False, probe=False, tag="unpack"
        )
    stats = unpack(args.archive, args.into, overwrite=args.overwrite)
    tag = stats["index"].get("manifest_tag")
    _log(
        "unpack",
        f"installed {stats['installed']} file(s) ({stats['bytes'] / 1e6:.1f} MB), kept {stats['skipped']} "
        f"existing, into {', '.join(map(str, args.into))}"
        + (f"; built for {tag}" if tag else ""),
    )
    return 0


def cmd_index(args) -> int:
    index = read_index(args.archive)
    if args.json:
        print(json.dumps(index, indent=2, sort_keys=True))
        return 0
    t = index["totals"]
    print(
        f"{args.archive}: {t['keys']} keys, {t['files']} files, {t['bytes'] / 1e6:.1f} MB, {index['manifest_tag']}"
    )
    for e in index["entries"]:
        print(f"  {e['key']}  {e['size']:>10}  {', '.join(e['sources']) or '-'}")
    return 0


def cmd_verify(args) -> int:
    if args.manifest:
        _require_match(
            load_manifest(args.manifest),
            args.python,
            force=args.force,
            probe=False,
            tag="verify",
        )
    cache = (
        args.cache_dir
        or os.environ.get("TORCH_NEURONX_NEFF_CACHE_DIR")
        or "/tmp/neff_cache"
    )
    projects = parse_projects(args.projects)
    rows, runs = [], []
    with tempfile.TemporaryDirectory(prefix="warm-verify-") as tmp:
        for proj in projects:
            pdir = Path(args.src) / PROJECTS[proj]
            if not (pdir / "tests").is_dir():
                raise WarmError(f"{proj}: no tests/ under {pdir}")
            env = _with_plugin_path(dict(os.environ))
            env.pop("TORCH_NEURONX_COMPILE_ONLY", None)
            env.update(
                TORCH_NEURONX_NEFF_CACHE_DIR=cache,
                TRNSCI_WARM_COUNTERS=str(Path(tmp) / f"{proj}.counters.json"),
                TRNSCI_WARM_SOURCES=str(Path(tmp) / f"{proj}.sources.jsonl"),
                TRNSCI_WARM_UNIT=proj,
            )
            if args.record_mlir:
                env["TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS"] = "1"
                env["TORCH_NEURONX_DEBUG_DIR"] = str(Path(args.record_mlir) / proj)
            cmd = pytest_command(
                args.python, args.tests or ["tests"], marker=args.marker, optimize=False
            )
            cmd += args.pytest_args
            log = Path(args.logs or tmp) / f"verify-{proj}.log"
            _log("verify", f"{proj}: {shlex.join(cmd)}")
            t0 = time.monotonic()
            code, timed_out = _run(
                cmd, cwd=pdir, env=env, timeout=args.timeout, log_path=log
            )
            counters = _load_counters(Path(env["TRNSCI_WARM_COUNTERS"]))
            agg = aggregate_counters([counters])
            missed = sorted(
                {
                    s
                    for v in load_sources(env["TRNSCI_WARM_SOURCES"]).values()
                    for s in v
                }
            )
            rows.append(
                {
                    "project": proj,
                    "exit_code": code,
                    "timed_out": timed_out,
                    "seconds": round(time.monotonic() - t0, 2),
                    **agg,
                    "compiled_in": missed,
                    "log": str(log) if args.logs else None,
                }
            )
            runs.append(counters)
    total = aggregate_counters(runs)
    print(f"\n{'project':<10} {'exit':>4} {'hits':>6} {'compiles':>8} {'hit rate':>8}")
    for r in rows:
        rate = "-" if r["hit_rate"] is None else f"{r['hit_rate']:.1%}"
        print(
            f"{r['project']:<10} {str(r['exit_code']):>4} {r['persistent_hits']:>6} {r['compilations']:>8} {rate:>8}"
        )
        for s in r["compiled_in"][:20]:
            print(f"    compiled in {s}")
    rate = "-" if total["hit_rate"] is None else f"{total['hit_rate']:.1%}"
    print(
        f"{'total':<10} {'':>4} {total['persistent_hits']:>6} {total['compilations']:>8} {rate:>8}"
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps({"cache": cache, "projects": rows, "total": total}, indent=2)
            + "\n"
        )
    if args.min_hit_rate is not None and (total["hit_rate"] or 0.0) < args.min_hit_rate:
        print(f"hit rate below --min-hit-rate {args.min_hit_rate}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    cpus = os.cpu_count() or 4
    p = argparse.ArgumentParser(
        prog="warm_neuron_cache.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Neuron venv interpreter (default: this one)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("manifest", help="record the box toolchain and compiler config")
    s.add_argument("-o", "--output", required=True)
    s.add_argument(
        "--no-probe",
        action="store_true",
        help="skip the probe compile (needs --target/--lnc or env)",
    )
    s.add_argument("--target", help="e.g. trn2 (default: from the probe compile)")
    s.add_argument(
        "--lnc", help="logical NeuronCore config, e.g. 2 (default: from the probe)"
    )
    s.add_argument(
        "--instance-type", help="for NEURON_RT_FAKE_INSTANCE_TYPE (default: per target)"
    )
    s.set_defaults(func=cmd_manifest)

    s = sub.add_parser("check", help="compare the local toolchain to a manifest")
    s.add_argument("--manifest", required=True)
    s.add_argument(
        "--no-probe", action="store_true", help="skip the compile-only probe comparison"
    )
    s.add_argument(
        "--compiler-only",
        action="store_true",
        help="check neuronx-cc only (recompile hosts)",
    )
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("capture", help="compile-only pytest runs on a CPU host")
    s.add_argument("--manifest", required=True)
    s.add_argument(
        "--out",
        required=True,
        help="output dir: neff/, sources.jsonl, logs/, capture.json",
    )
    s.add_argument("--projects", default="all", help="comma list (default: all six)")
    s.add_argument(
        "--src", default=str(REPO_ROOT), help="dir holding the sub-project checkouts"
    )
    s.add_argument("--workers", type=int, default=max(1, cpus // 4))
    s.add_argument(
        "--by",
        choices=("file", "project"),
        default="file",
        help="pytest invocation granularity",
    )
    s.add_argument("--marker", default="neuron")
    s.add_argument(
        "--no-prefilter",
        action="store_true",
        help="run files without a `mark.neuron` too",
    )
    s.add_argument("--timeout", type=float, default=3600, help="seconds per unit")
    s.add_argument(
        "--keep-mlir",
        action="store_true",
        help="also save module.mlir corpus to OUT/mlir/",
    )
    s.add_argument("--keep-work", action="store_true")
    s.add_argument(
        "--no-probe", action="store_true", help="skip the probe in the pre-run check"
    )
    s.add_argument(
        "--force", action="store_true", help="run despite toolchain mismatches"
    )
    s.add_argument(
        "--strict", action="store_true", help="exit 1 if any unit errored or timed out"
    )
    s.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="passed to pytest (last)",
    )
    s.set_defaults(func=cmd_capture)

    s = sub.add_parser("recompile", help="compile recorded module.mlir files")
    s.add_argument("corpus", help="dir searched recursively for module.mlir")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=max(1, cpus // 4))
    s.add_argument(
        "--neuronx-cc", help="compiler binary (default: next to --python, else PATH)"
    )
    s.add_argument("--timeout", type=float, default=3600, help="seconds per module")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_recompile)

    s = sub.add_parser("pack", help="tar a cache dir plus a JSON index")
    s.add_argument(
        "cache", help="capture/recompile --out dir, or a bare NEFF cache dir"
    )
    s.add_argument("-o", "--output", help="default: ./neff-<manifest tag>.tar.zst")
    s.add_argument("--manifest", help="default: <cache>/manifest.json if present")
    s.add_argument("--sources", help="default: <cache>/sources.jsonl if present")
    s.add_argument(
        "--compression", choices=("auto", "zst", "gz", "none"), default="auto"
    )
    s.set_defaults(func=cmd_pack)

    s = sub.add_parser("unpack", help="install a packed archive into NEFF cache dirs")
    s.add_argument("archive")
    s.add_argument(
        "--into",
        action="append",
        required=True,
        help="cache dir (repeatable, e.g. one per lane)",
    )
    s.add_argument("--overwrite", action="store_true")
    s.add_argument(
        "--check", action="store_true", help="refuse unless the local toolchain matches"
    )
    s.set_defaults(func=cmd_unpack)

    s = sub.add_parser("index", help="print a packed archive's index")
    s.add_argument("archive")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser(
        "verify", help="run on the box and report the persistent-hit rate"
    )
    s.add_argument("--projects", default="all")
    s.add_argument("--src", default=str(REPO_ROOT))
    s.add_argument(
        "--tests",
        nargs="+",
        help="test paths relative to each project (default: tests)",
    )
    s.add_argument(
        "--cache-dir", help="default: $TORCH_NEURONX_NEFF_CACHE_DIR or /tmp/neff_cache"
    )
    s.add_argument("--manifest", help="also check the toolchain first")
    s.add_argument("--marker", default="neuron")
    s.add_argument("--timeout", type=float, default=7200)
    s.add_argument(
        "--record-mlir", help="preserve module.mlir of every miss under this dir"
    )
    s.add_argument("--logs", help="keep pytest logs in this dir")
    s.add_argument("--report", help="write a JSON report")
    s.add_argument("--min-hit-rate", type=float, help="exit 1 below this rate (0-1)")
    s.add_argument("--force", action="store_true")
    s.add_argument("--pytest-args", nargs=argparse.REMAINDER, default=[])
    s.set_defaults(func=cmd_verify)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except WarmError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
