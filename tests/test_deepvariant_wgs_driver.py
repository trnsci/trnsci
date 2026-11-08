"""CPU tests for the whole-genome call_variants driver (cv_neuron_wgs.py).

Synthetic make_examples shards, a randomly initialised model, the CPU
device: decode pool -> model -> per-shard CVO files, resume, and the
example_info check. No weights or data downloads needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

DV_DIR = (
    Path(__file__).resolve().parent.parent / "examples" / "genomics" / "deepvariant"
)
sys.path.insert(0, str(DV_DIR))

torch = pytest.importorskip("torch")

SHAPE = (100, 221, 7)
INFO = {"version": "1.9.0", "shape": list(SHAPE), "channels": [1, 2, 3, 4, 5, 6, 19]}


def _pb(fno: int, payload) -> bytes:
    from dv_io import _enc_varint

    if isinstance(payload, int):
        return _enc_varint(fno << 3) + _enc_varint(payload)
    return _enc_varint(fno << 3 | 2) + _enc_varint(len(payload)) + payload


def _tf_example(feats: dict) -> bytes:
    """Encode a tf.train.Example from {name: bytes | list[int]}."""
    from dv_io import _enc_varint

    entries = b""
    for k, v in feats.items():
        if isinstance(v, bytes):
            feat = _pb(1, _pb(1, v))  # Feature.bytes_list.value
        else:
            packed = b"".join(_enc_varint(x) for x in v)
            feat = _pb(3, _pb(1, packed))  # Feature.int64_list (packed)
        entries += _pb(1, _pb(1, k.encode()) + _pb(2, feat))
    return _pb(1, entries)


def _variant(i: int) -> bytes:
    return (
        _pb(14, b"chr1")
        + _pb(16, 1000 + i)
        + _pb(13, 1001 + i)
        + _pb(6, b"A")
        + _pb(7, b"G")
    )


def _write_shards(d: Path, nshards: int, per_shard: int, info=INFO) -> str:
    from dv_io import TFRecordWriter, synthetic_pileups

    imgs = synthetic_pileups(nshards * per_shard, SHAPE, seed=3)
    for s in range(nshards):
        p = d / f"me.tfrecord-{s:05d}-of-{nshards:05d}.gz"
        with TFRecordWriter(str(p)) as w:
            for j in range(per_shard):
                i = s * per_shard + j
                w.write(
                    _tf_example(
                        {
                            "locus": b"chr1:%d" % i,
                            "image/encoded": imgs[i].tobytes(),
                            "variant/encoded": _variant(i),
                            "alt_allele_indices/encoded": _pb(1, 0),
                            "image/shape": list(SHAPE),
                            "variant_type": [1],
                        }
                    )
                )
        (d / (p.name + ".example_info.json")).write_text(json.dumps(info))
    return str(d / f"me.tfrecord@{nshards}.gz")


@pytest.fixture(scope="module")
def weights(tmp_path_factory) -> str:
    from dv_inception import build_model

    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("w") / "random.pt"
    torch.save(build_model().state_dict(), p)
    return str(p)


def test_example_fields_matches_decode_example():
    from dv_io import decode_example, example_fields

    img = np.arange(np.prod(SHAPE), dtype=np.uint64).astype(np.uint8).reshape(SHAPE)
    buf = _tf_example(
        {
            "locus": b"chr1:5",
            "image/encoded": img.tobytes(),
            "variant/encoded": _variant(5),
            "alt_allele_indices/encoded": _pb(1, 1),
            "image/shape": list(SHAPE),
        }
    )
    ex = decode_example(buf, SHAPE)
    im, var, alt = example_fields(buf)
    assert bytes(im) == ex.image.tobytes() == img.tobytes()
    assert var == ex.variant == _variant(5)
    assert alt == ex.alt_allele_indices == _pb(1, 1)
    with pytest.raises(ValueError, match="variant/encoded"):
        example_fields(_tf_example({"image/encoded": b"x"}))


def test_shard_assignment_is_disjoint_and_complete():
    import argparse

    from cv_neuron_wgs import select_shards

    seen = []
    for k in range(4):
        a = argparse.Namespace(shards=None, worker=k, num_workers=4)
        seen += select_shards(a, 64)
    assert sorted(seen) == list(range(64))


def test_check_example_info_rejects_wrong_channels(tmp_path):
    from cv_neuron_wgs import check_example_info, input_shards

    bad = dict(INFO, channels=[1, 2, 3, 4, 5, 6, 7])
    pat = _write_shards(tmp_path, 1, 1, info=bad)
    with pytest.raises(SystemExit, match="channels"):
        check_example_info(input_shards(pat))


def test_driver_cpu_end_to_end_and_resume(tmp_path, weights):
    from cv_neuron_wgs import main
    from dv_io import decode_cvo, read_tfrecords, record_key

    data, out = tmp_path / "in", tmp_path / "out"
    data.mkdir()
    pat = _write_shards(data, nshards=3, per_shard=21)
    common = [
        "--examples", pat, "--weights", weights, "--outdir", str(out),
        "--device", "cpu", "--batch-size", "8", "--decode-procs", "2", "--slots", "4",
        "--check-example-info", "--threads", "2",
    ]  # fmt: skip
    main(common + ["--shards", "0,2", "--limit", "20"])
    files = sorted(p.name for p in out.glob("cvo-neuron-*.tfrecord.gz"))
    assert files == [
        "cvo-neuron-00000-of-00003.tfrecord.gz",
        "cvo-neuron-00002-of-00003.tfrecord.gz",
    ]
    recs = list(read_tfrecords(str(out / files[1]), check_crc=True))
    assert len(recs) == 20  # --limit is per shard
    want = [record_key(_variant(42 + j), _pb(1, 0)) for j in range(20)]
    got = [decode_cvo(r) for r in recs]
    assert [k for k, _ in got] == want  # input order kept
    for _, p in got:
        assert p.shape == (3,) and abs(p.sum() - 1) < 1e-6

    # Resume: same settings skip the done shards and run only shard 1.
    main(common + ["--limit", "20"])
    reports = [
        json.loads(p.read_text())
        for p in sorted((out / "reports").glob("worker-0.*.json"))
    ]
    last = max(reports, key=lambda r: r["t_start_unix"])
    assert last["shards_skipped"] == 2 and last["shards_todo"] == [1]
    # A different --limit is a different run: everything is redone.
    main(common + ["--shards", "1", "--limit", "0"])
    assert (
        len(list(read_tfrecords(str(out / "cvo-neuron-00001-of-00003.tfrecord.gz"))))
        == 21
    )
    marker = json.loads((out / "done" / "00001.json").read_text())
    assert marker["records"] == 21 and marker["limit"] == 0
    assert not list((out / "partial").iterdir())

    main(["--summarize", str(out)])
    s = json.loads((out / "reports" / "summary.json").read_text())
    assert s["records_in_done_shards"] == 20 + 21 + 20
