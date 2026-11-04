"""CPU tests for the pure parts of scripts/warm_neuron_cache.py (trnsci/trnsci#118).

The end-to-end path (manifest + verify on a Trainium box, capture on a CPU
host with the box's toolchain) needs hardware and isn't covered here.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import neuron_cache_plugin as plugin  # noqa: E402
import warm_neuron_cache as w  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "neuron_cache"
# Window-4 Trn2 box (#118): neuronx-cc version string and eager config string.
BOX_VERSION = "2.28.7631.0+8f7b0cf5"
BOX_CONFIG = (
    "trn2_2_-O1_--model-type transformer --auto-cast=none "
    "--internal-disable-fma-on-ios_v2.28.7631.0+8f7b0cf5"
)
BOX_COMPILER_TXT = (
    "neuronx-cc compile module.mlir --framework XLA --target trn2 --lnc 2 -O1 "
    "--model-type transformer --auto-cast=none --internal-disable-fma-on-ios "
    "--output module.neff"
)
# Key of the fixture module under the box config (pinned; see the fixture).
FIXTURE_KEY = "6b782329afe89c5921caa74e82985609"


@pytest.fixture
def manifest():
    return json.loads((FIXTURES / "manifest_trn2_lnc2.json").read_text())


@pytest.fixture
def identity(manifest):
    return dict(manifest["toolchain"])


def _box_flags():
    return dict(
        target="trn2",
        lnc=2,
        opt="-O1",
        compiler_args="--model-type transformer --auto-cast=none --internal-disable-fma-on-ios",
        compiler_version=BOX_VERSION,
    )


# --------------------------------------------------------------------------- keys


def test_key_config_matches_box_string():
    assert w.key_config(**_box_flags()) == BOX_CONFIG
    assert w.key_config(**_box_flags(), wlo=True) == BOX_CONFIG + "_wlo"


def test_neff_key_fixture():
    xxhash = pytest.importorskip("xxhash")
    mlir = (FIXTURES / "mm_37x211x53.mlir").read_bytes()
    key = w.neff_key(mlir, **_box_flags())
    assert key == FIXTURE_KEY
    # byte layout: module bytes, one 0xFF separator, config string
    assert key == xxhash.xxh3_128_hexdigest(mlir + b"\xff" + BOX_CONFIG.encode())


@pytest.mark.parametrize(
    "change",
    [
        {"lnc": 1},
        {"opt": "-O2"},
        {"target": "trn1"},
        {"compiler_version": "2.28.7631.0"},
        {"compiler_args": "--model-type transformer --auto-cast=none"},
        {"wlo": True},
    ],
)
def test_neff_key_depends_on_config(change):
    pytest.importorskip("xxhash")
    mlir = (FIXTURES / "mm_37x211x53.mlir").read_bytes()
    assert w.neff_key(mlir, **{**_box_flags(), **change}) != FIXTURE_KEY


def test_neff_key_depends_on_every_mlir_byte():
    pytest.importorskip("xxhash")
    mlir = (FIXTURES / "mm_37x211x53.mlir").read_bytes()
    assert w.neff_key(mlir.rstrip(b"\n"), **_box_flags()) != FIXTURE_KEY


def test_neff_relpath():
    assert w.neff_relpath(FIXTURE_KEY) == f"6b/78/{FIXTURE_KEY}.neff"
    assert w.neff_relpath(FIXTURE_KEY, "wrapped_neff_hlo").endswith(".wrapped_neff_hlo")
    with pytest.raises(ValueError):
        w.neff_relpath("../../etc/passwd")


def test_parse_compiler_txt_box_line():
    p = w.parse_compiler_txt(BOX_COMPILER_TXT + "\n")
    assert p == {
        "target": "trn2",
        "lnc": "2",
        "opt": "-O1",
        "framework": "XLA",
        "compiler_args": "--model-type transformer --auto-cast=none --internal-disable-fma-on-ios",
    }


def test_parse_compiler_txt_equals_form_and_extra_flags():
    p = w.parse_compiler_txt(
        "/opt/venv/bin/neuronx-cc compile /tmp/x/module.mlir --framework=XLA --target=trn2 "
        "--lnc=1 -O2 --model-type transformer --enable-foo --output=/tmp/x/module.neff"
    )
    assert (p["target"], p["lnc"], p["opt"]) == ("trn2", "1", "-O2")
    assert p["compiler_args"] == "--model-type transformer --enable-foo"


def test_manifest_probe_reproduces_its_keys(manifest):
    pytest.importorskip("xxhash")
    assert w.verify_manifest_probe(manifest) == []
    broken = copy.deepcopy(manifest)
    broken["probe"]["modules"][0]["mlir"] += " "
    assert len(w.verify_manifest_probe(broken)) == 1


# --------------------------------------------------------------------------- manifest


def test_fixture_manifest_loads(tmp_path, manifest):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest))
    m = w.load_manifest(p)
    assert m["config"]["source"] == "probe"
    assert m["config"]["instance_type"] == "trn2.48xlarge"
    assert w.manifest_tag(m) == (
        "neuronxcc-2.28.7631.0_8f7b0cf5_torch-neuronx-2.14.3.0.1_b1b6c66_trn2-lnc2-O1"
    )
    p.write_text(json.dumps({**manifest, "version": 99}))
    with pytest.raises(w.WarmError):
        w.load_manifest(p)


def test_build_manifest_from_env_without_probe(identity):
    env = {"NEURON_PLATFORM_TARGET_OVERRIDE": "trn2", "NEURON_LOGICAL_NC_CONFIG": "1"}
    env["NEURON_CC_FLAGS"] = "--enable-x "
    env["TORCH_NEURONX_INT64_POLICY"] = "cast"
    env["PATH"] = "/usr/bin"
    m = w.build_manifest(identity, env, None)
    assert m["config"] == {
        "target": "trn2",
        "lnc": "1",
        "opt": "-O1",
        "compiler_args": w.BASE_COMPILER_ARGS + " --enable-x",
        "source": "env",
        "instance_type": "trn2.48xlarge",
    }
    assert "PATH" not in m["env"]
    assert m["env"]["TORCH_NEURONX_INT64_POLICY"] == "cast"
    with pytest.raises(w.WarmError):
        w.build_manifest(identity, {}, None)


def test_build_manifest_rejects_conflicting_target(identity, manifest):
    with pytest.raises(w.WarmError):
        w.build_manifest(identity, {}, manifest["probe"], lnc=1)


# --------------------------------------------------------------------------- compare


def _fails(findings):
    return sorted(f.field for f in findings if f.level == "FAIL")


def test_compare_identical(manifest, identity):
    findings = w.compare_toolchain(manifest, identity, {})
    assert _fails(findings) == []
    assert all(f.level == "OK" for f in findings)


@pytest.mark.parametrize(
    "name", ["neuronxcc", "torch_neuronx", "nki", "neuronxcc_key_version"]
)
def test_compare_strict_toolchain(manifest, identity, name):
    identity[name] = "0.0.0"
    assert _fails(w.compare_toolchain(manifest, identity, {})) == [f"toolchain.{name}"]


def test_compare_missing_package_fails(manifest, identity):
    identity["torch_neuronx"] = None
    assert _fails(w.compare_toolchain(manifest, identity, {})) == [
        "toolchain.torch_neuronx"
    ]


def test_compare_info_only_fields(manifest, identity):
    identity["torch"] = "2.14.0+cpu"  # #118: the Mac had +cpu and still hit
    identity["python"] = "3.11.9"
    findings = w.compare_toolchain(manifest, identity, {})
    assert _fails(findings) == []
    assert {f.field for f in findings if f.level == "INFO"} == {
        "toolchain.torch",
        "toolchain.python",
    }


def test_compare_compiler_only(manifest, identity):
    identity["torch_neuronx"] = None
    identity["neuronxcc_key_version"] = None
    findings = w.compare_toolchain(manifest, identity, {}, fields=("neuronxcc",))
    assert _fails(findings) == []


def test_compare_env(manifest, identity):
    # matching or unset is fine; anything set differently fails
    ok = {"NEURON_LOGICAL_NC_CONFIG": "2", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2"}
    assert _fails(w.compare_toolchain(manifest, identity, ok)) == []
    bad = {
        "NEURON_LOGICAL_NC_CONFIG": "1",
        "NEURON_CC_FLAGS": "--foo",
        "TORCH_NEURONX_INT64_POLICY": "x",
    }
    assert _fails(w.compare_toolchain(manifest, identity, bad)) == [
        "env.NEURON_CC_FLAGS",
        "env.NEURON_LOGICAL_NC_CONFIG",
        "env.TORCH_NEURONX_INT64_POLICY",
    ]


def test_compare_env_recorded_on_box(manifest, identity):
    manifest["env"] = {"NEURON_CC_FLAGS": "--foo"}
    assert _fails(w.compare_toolchain(manifest, identity, {})) == []
    assert (
        _fails(w.compare_toolchain(manifest, identity, {"NEURON_CC_FLAGS": "--foo"}))
        == []
    )
    assert _fails(
        w.compare_toolchain(manifest, identity, {"NEURON_CC_FLAGS": "--bar"})
    ) == ["env.NEURON_CC_FLAGS"]


def test_compare_probe(manifest):
    assert w.compare_probe(manifest, {"keys": [FIXTURE_KEY]}).level == "OK"
    assert w.compare_probe(manifest, {"keys": ["0" * 32]}).level == "FAIL"


def test_compile_only_env(manifest, tmp_path):
    manifest["env"] = {"NEURON_CC_FLAGS": "--foo"}
    base = {
        "PATH": "/usr/bin",
        "NEURON_LOGICAL_NC_CONFIG": "1",
        "TORCH_NEURONX_NEFF_DISABLE_CACHE": "1",
        "TORCH_NEURONX_INT64_POLICY": "local",
    }
    env = w.compile_only_env(manifest, base, tmp_path, preserve=True)
    assert env["PATH"] == "/usr/bin"
    assert env["TORCH_NEURONX_COMPILE_ONLY"] == "1"
    assert env["NEURON_PLATFORM_TARGET_OVERRIDE"] == "trn2"
    assert env["NEURON_LOGICAL_NC_CONFIG"] == "2"
    assert env["NEURON_RT_FAKE_INSTANCE_TYPE"] == "trn2.48xlarge"
    assert env["NKI_ENABLE_TRACE_CACHE"] == "0"
    assert env["NEURON_CC_FLAGS"] == "--foo"
    assert env["TORCH_NEURONX_NEFF_CACHE_DIR"] == str(tmp_path / "neff")
    assert env["TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS"] == "1"
    assert "TORCH_NEURONX_NEFF_DISABLE_CACHE" not in env
    assert "TORCH_NEURONX_INT64_POLICY" not in env  # not set on the box
    assert "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS" not in w.compile_only_env(
        manifest, base, tmp_path
    )


# --------------------------------------------------------------------------- cache, index, pack


def _fake_cache(root: Path, n=3, wrapped=1) -> dict[str, bytes]:
    files = {}
    for i in range(n):
        key = hashlib.md5(f"neff{i}".encode()).hexdigest()  # any 32-hex
        rel = w.neff_relpath(key)
        files[rel] = os.urandom(1000 + 37 * i)
        if i < wrapped:
            files[w.neff_relpath(key, "wrapped_neff_hlo")] = b"hlo" * (i + 5)
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    # noise that isn't a cache entry
    (root / "README").write_text("x")
    (root / "ab").mkdir(exist_ok=True)
    (root / "ab" / "stray.neff").write_bytes(b"x")
    return files


def test_iter_cache_and_merge(tmp_path):
    files = _fake_cache(tmp_path / "a")
    entries = list(w.iter_cache(tmp_path / "a"))
    assert sorted(rel for _k, _kind, rel, _p in entries) == sorted(files)
    assert plugin.scan_neff_keys(tmp_path / "a") == w.cache_keys(tmp_path / "a")
    assert len(w.cache_keys(tmp_path / "a")) == 3
    dst = tmp_path / "b"
    # a key with only a .neff, already present in dst
    first = next(
        r for r in sorted(files) if r.replace(".neff", ".wrapped_neff_hlo") not in files
    )
    (dst / first).parent.mkdir(parents=True)
    (dst / first).write_bytes(b"existing")
    new = w.merge_cache(tmp_path / "a", dst)
    assert len(new) == 2
    assert (dst / first).read_bytes() == b"existing"
    assert sorted(rel for _k, _kind, rel, _p in w.iter_cache(dst)) == sorted(files)


def test_load_sources(tmp_path):
    p = tmp_path / "sources.jsonl"
    recs = [
        {"key": "a" * 32, "source": "tests/test_x.py::test_1", "project": "trnblas"},
        {"key": "a" * 32, "source": "tests/test_x.py::test_1", "project": "trnblas"},
        {"key": "a" * 32, "source": "module:run1/0001"},
        {"key": "b" * 32, "source": "tests/test_y.py::test_2", "project": "trnfft"},
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs) + "\n")
    assert w.load_sources(p) == {
        "a" * 32: ["module:run1/0001", "trnblas:tests/test_x.py::test_1"],
        "b" * 32: ["trnfft:tests/test_y.py::test_2"],
    }
    assert w.load_sources(tmp_path / "missing.jsonl") == {}


def test_index_format(tmp_path, manifest):
    files = _fake_cache(tmp_path)
    key0 = sorted(files)[0].split("/")[-1].split(".")[0]
    idx = w.build_index(tmp_path, sources={key0: ["trnblas:t::a"]}, manifest=manifest)
    assert idx["format"] == "trnsci-neff-index" and idx["version"] == 1
    assert idx["manifest_tag"] == w.manifest_tag(manifest)
    assert idx["manifest"] == manifest
    assert idx["totals"] == {
        "files": len(files),
        "keys": 3,
        "bytes": sum(len(v) for v in files.values()),
    }
    assert [e["path"] for e in idx["entries"]] == sorted(f"neff/{r}" for r in files)
    for e in idx["entries"]:
        assert set(e) == {"key", "kind", "path", "size", "sha256", "sources"}
        data = files[e["path"][5:]]
        assert e["size"] == len(data)
        assert e["sha256"] == hashlib.sha256(data).hexdigest()
        assert e["kind"] in ("neff", "wrapped_neff_hlo")
        assert e["sources"] == (["trnblas:t::a"] if e["key"] == key0 else [])
    w.validate_index(idx)
    json.dumps(idx)  # serialisable
    bad = copy.deepcopy(idx)
    bad["entries"][0]["path"] = "neff/../../x.neff"
    with pytest.raises(w.WarmError):
        w.validate_index(bad)


COMPRESSIONS = ["gz", "none"] + (["zst"] if shutil.which("zstd") else [])


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_pack_unpack_round_trip(tmp_path, manifest, compression):
    cache = tmp_path / "cache"
    files = _fake_cache(cache)
    out, idx = w.pack(
        cache,
        tmp_path / "out" / f"warm-{compression}.bin",
        manifest=manifest,
        compression=compression,
    )
    assert out.is_file()
    side = json.loads(Path(f"{out}.index.json").read_text())
    assert side == idx
    assert w.read_index(out) == idx
    Path(f"{out}.index.json").unlink()
    assert w.read_index(out) == idx  # from the embedded member
    with tarfile.open(out) if compression != "zst" else _zst_tar(out) as tf:
        names = tf.getnames()
    assert names[0] == "index.json"
    assert sorted(names[1:]) == sorted(f"neff/{r}" for r in files)

    lanes = [tmp_path / "lane0", tmp_path / "lane1"]
    stats = w.unpack(out, lanes)
    assert stats["installed"] == 2 * len(files) and stats["skipped"] == 0
    for lane in lanes:
        got = {
            rel: (lane / rel).read_bytes() for _k, _kind, rel, _p in w.iter_cache(lane)
        }
        assert got == files
        assert not (lane / "README").exists()
    # second unpack keeps what's there
    assert w.unpack(out, lanes)["skipped"] == 2 * len(files)
    assert w.unpack(out, lanes, overwrite=True)["installed"] == 2 * len(files)


def _zst_tar(path):
    import subprocess

    data = subprocess.run(
        ["zstd", "-dc", str(path)], capture_output=True, check=True
    ).stdout
    return tarfile.open(fileobj=io.BytesIO(data))


def test_pack_default_name_and_capture_dir(tmp_path, manifest, monkeypatch):
    out_dir = tmp_path / "capture"
    _fake_cache(out_dir / "neff")
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    key = sorted(w.cache_keys(out_dir / "neff"))[0]
    (out_dir / "sources.jsonl").write_text(
        json.dumps({"key": key, "source": "tests/t.py::a", "project": "trnblas"}) + "\n"
    )
    cache, src, man = w.resolve_capture_dir(out_dir)
    assert (cache, src, man) == (
        out_dir / "neff",
        out_dir / "sources.jsonl",
        out_dir / "manifest.json",
    )
    monkeypatch.chdir(tmp_path)
    assert w.main(["pack", str(out_dir), "--compression", "gz"]) == 0
    archive = tmp_path / f"neff-{w.manifest_tag(manifest)}.tar.gz"
    idx = w.read_index(archive)
    assert [e["sources"] for e in idx["entries"] if e["key"] == key][0] == [
        "trnblas:tests/t.py::a"
    ]
    assert w.main(["index", str(archive)]) == 0
    assert w.main(["unpack", str(archive), "--into", str(tmp_path / "lane")]) == 0
    assert w.cache_keys(tmp_path / "lane") == w.cache_keys(out_dir / "neff")


def test_pack_empty_cache(tmp_path):
    with pytest.raises(w.WarmError):
        w.pack(tmp_path, tmp_path / "x.tar")


def _tamper(src: Path, dst: Path, edit):
    """Rewrite an uncompressed archive, letting ``edit`` change members."""
    with tarfile.open(src) as tf_in, tarfile.open(dst, "w") as tf_out:
        members = [(m, tf_in.extractfile(m).read()) for m in tf_in.getmembers()]
        for m, data in edit(members):
            m.size = len(data)
            tf_out.addfile(m, io.BytesIO(data))


@pytest.fixture
def plain_archive(tmp_path, manifest):
    _fake_cache(tmp_path / "cache")
    out, _ = w.pack(
        tmp_path / "cache", tmp_path / "a.tar", manifest=manifest, compression="none"
    )
    Path(f"{out}.index.json").unlink()
    return out


def test_unpack_rejects_corrupt_member(tmp_path, plain_archive):
    def flip(members):
        m, data = members[1]
        members[1] = (m, bytes([data[0] ^ 1]) + data[1:])
        return members

    bad = tmp_path / "bad.tar"
    _tamper(plain_archive, bad, flip)
    with pytest.raises(w.WarmError, match="does not match"):
        w.unpack(bad, [tmp_path / "lane"])


def test_unpack_rejects_path_traversal(tmp_path, plain_archive):
    def evil(members):
        m = tarfile.TarInfo("../evil.neff")
        return members + [(m, b"x")]

    bad = tmp_path / "evil.tar"
    _tamper(plain_archive, bad, evil)
    with pytest.raises(w.WarmError, match="unexpected member"):
        w.unpack(bad, [tmp_path / "lane"])
    assert not (tmp_path / "evil.neff").exists()


def test_unpack_rejects_truncated(tmp_path, plain_archive):
    bad = tmp_path / "short.tar"
    _tamper(plain_archive, bad, lambda members: members[:-1])
    with pytest.raises(w.WarmError, match="missing"):
        w.unpack(bad, [tmp_path / "lane"])


def test_unpack_requires_index_first(tmp_path, plain_archive):
    bad = tmp_path / "noindex.tar"
    _tamper(plain_archive, bad, lambda members: members[1:] + members[:1])
    with pytest.raises(w.WarmError, match="index.json"):
        w.unpack(bad, [tmp_path / "lane"])


# --------------------------------------------------------------------------- counters, units


def test_aggregate_counters():
    runs = [
        {
            "CompilationCache.PersistentHits": 14,
            "CompilationCache.TotalCompilations": 0,
        },
        {"CompilationCache.PersistentHits": 6, "CompilationCache.TotalCompilations": 4},
        {},
    ]
    agg = w.aggregate_counters(runs)
    assert agg["persistent_hits"] == 20 and agg["compilations"] == 4
    assert agg["hit_rate"] == pytest.approx(20 / 24)
    assert agg["counters"]["CompilationCache.PersistentHits"] == 20
    assert w.aggregate_counters([{}])["hit_rate"] is None


def test_parse_projects():
    assert w.parse_projects("all") == list(w.PROJECTS)
    assert w.parse_projects("trnblas,trnsolve") == ["trnblas", "trnsolver"]
    with pytest.raises(w.WarmError):
        w.parse_projects("trncuda")


def test_discover_units(tmp_path):
    t = tmp_path / "trnsolve" / "tests"
    t.mkdir(parents=True)
    (t / "test_eigh.py").write_text(
        "import pytest\n@pytest.mark.neuron\ndef test_a(): pass\n"
    )
    (t / "test_cpu.py").write_text("def test_b(): pass\n")
    (t / "helpers.py").write_text("pytest.mark.neuron\n")
    units = w.discover_units(tmp_path, ["trnsolver"])
    assert [(u.label, u.targets) for u in units] == [
        ("trnsolver/tests/test_eigh.py", ["tests/test_eigh.py"])
    ]
    assert len(w.discover_units(tmp_path, ["trnsolver"], prefilter=False)) == 2
    assert [
        u.targets for u in w.discover_units(tmp_path, ["trnsolver"], by="project")
    ] == [["tests"]]
    with pytest.raises(w.WarmError):
        w.discover_units(tmp_path, ["trnblas"])


def test_plan_recompile(tmp_path, manifest):
    pytest.importorskip("xxhash")
    d = tmp_path / "rec" / "0001"
    d.mkdir(parents=True)
    shutil.copy(FIXTURES / "mm_37x211x53.mlir", d / "module.mlir")
    plan = w.plan_recompile(d, manifest)
    assert plan["key"] == FIXTURE_KEY and plan["origin"] == "manifest"
    (d / "compiler.txt").write_text(BOX_COMPILER_TXT)
    plan = w.plan_recompile(d, manifest)
    assert plan["key"] == FIXTURE_KEY and plan["origin"] == "compiler.txt"
    (d / "compiler.txt").write_text(BOX_COMPILER_TXT.replace("--lnc 2", "--lnc 1"))
    assert "skip" in w.plan_recompile(d, manifest)


def test_recompile_with_stub_compiler(tmp_path, manifest):
    """Drives `recompile` end to end with a stand-in for neuronx-cc."""
    pytest.importorskip("xxhash")
    corpus = tmp_path / "corpus"
    for i, text in enumerate([b"", b"// second module\n"]):
        d = corpus / f"{i:04d}"
        d.mkdir(parents=True)
        (d / "module.mlir").write_bytes(
            (FIXTURES / "mm_37x211x53.mlir").read_bytes() + text
        )
    stub = tmp_path / "neuronx-cc"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "a = sys.argv\n"
        "assert a[1:4] == ['compile', 'module.mlir', '--framework']\n"
        "out = a[a.index('--output') + 1]\n"
        "open(out, 'wb').write(b'NEFF' + open('module.mlir', 'rb').read())\n"
    )
    stub.chmod(0o755)
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(manifest))
    out = tmp_path / "out"
    argv = ["recompile", str(corpus), "--manifest", str(mpath), "--out", str(out)]
    argv += ["--neuronx-cc", str(stub), "--force", "--workers", "2"]
    assert w.main(argv) == 0
    keys = w.cache_keys(out / "neff")
    assert FIXTURE_KEY in keys and len(keys) == 2
    neff = (out / "neff" / w.neff_relpath(FIXTURE_KEY)).read_bytes()
    assert neff == b"NEFF" + (FIXTURES / "mm_37x211x53.mlir").read_bytes()
    assert sorted(
        s for v in w.load_sources(out / "sources.jsonl").values() for s in v
    ) == [
        "module:0000",
        "module:0001",
    ]
    # rerun: everything already cached, nothing recompiled
    assert w.main(argv) == 0
    assert len(w.load_sources(out / "sources.jsonl")) == 2


# --------------------------------------------------------------------------- plugin


def test_plugin_neutralise_result_checks(monkeypatch):
    np = pytest.importorskip("numpy")
    import numpy.testing as npt

    original = npt.assert_allclose
    patched = plugin.neutralise_result_checks()
    try:
        assert any(name == "assert_allclose" for _m, name, _o in patched)
        npt.assert_allclose(np.zeros(2), np.ones(2))  # no-op now
    finally:
        for mod, name, orig in patched:
            setattr(mod, name, orig)
    assert npt.assert_allclose is original


def test_plugin_records_new_keys(tmp_path, monkeypatch):
    cache = tmp_path / "neff"
    _fake_cache(cache, n=1, wrapped=0)
    sources = tmp_path / "sources.jsonl"
    monkeypatch.setenv("TORCH_NEURONX_NEFF_CACHE_DIR", str(cache))
    monkeypatch.setenv("TRNSCI_WARM_SOURCES", str(sources))
    monkeypatch.setenv("TRNSCI_WARM_UNIT", "trnblas")
    plugin.pytest_configure(None)
    plugin.pytest_runtest_logfinish("tests/t.py::a", None)
    assert not sources.exists()  # nothing new
    key = "f" * 32
    (cache / "ff" / "ff").mkdir(parents=True)
    (cache / "ff" / "ff" / f"{key}.neff").write_bytes(b"n")
    plugin.pytest_runtest_logfinish("tests/t.py::b", None)
    assert [json.loads(x) for x in sources.read_text().splitlines()] == [
        {"key": key, "project": "trnblas", "source": "tests/t.py::b"}
    ]


def test_capture_with_stub_project(tmp_path, manifest, monkeypatch):
    """Runs `capture` over a fake project whose neuron test writes a NEFF.

    Exercises the runner (compile-only env, plugin attribution, per-unit
    caches merged into one, summary), not torch-neuronx itself.
    """
    tests = tmp_path / "src" / "trnblas" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_fake.py").write_text(
        "import os, pytest\n"
        "from pathlib import Path\n\n"
        "@pytest.mark.neuron\n"
        "@pytest.mark.parametrize('i', [1, 2])\n"
        "def test_compile(i):\n"
        "    assert os.environ['TORCH_NEURONX_COMPILE_ONLY'] == '1'\n"
        "    assert os.environ['NEURON_LOGICAL_NC_CONFIG'] == '2'\n"
        "    key = f'{i:032x}'\n"
        "    p = Path(os.environ['TORCH_NEURONX_NEFF_CACHE_DIR'], key[:2], key[2:4], key + '.neff')\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.write_bytes(b'neff')\n"
        "    assert False  # placeholder data: stripped under python -O\n"
    )
    (tests / "test_cpu_only.py").write_text("def test_x(): pass\n")
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(manifest))
    out = tmp_path / "out"
    monkeypatch.delenv("PYTHONPATH", raising=False)
    argv = [
        "capture",
        "--manifest",
        str(mpath),
        "--out",
        str(out),
        "--projects",
        "trnblas",
    ]
    argv += ["--src", str(tmp_path / "src"), "--force", "--no-probe", "--strict"]
    assert w.main(argv) == 0
    summary = json.loads((out / "capture.json").read_text())
    assert [(u["label"], u["exit_code"], u["neffs"]) for u in summary["units"]] == [
        ("trnblas/tests/test_fake.py", 0, 2)
    ]
    assert w.cache_keys(out / "neff") == {f"{1:032x}", f"{2:032x}"}
    assert w.load_sources(out / "sources.jsonl") == {
        f"{1:032x}": ["trnblas:tests/test_fake.py::test_compile[1]"],
        f"{2:032x}": ["trnblas:tests/test_fake.py::test_compile[2]"],
    }
    assert not (out / "work").exists() or not any((out / "work").iterdir())
    assert json.loads((out / "manifest.json").read_text()) == manifest
