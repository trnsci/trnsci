"""CPU smoke tests for the DeepVariant call_variants exemplar (no weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

DV_DIR = (
    Path(__file__).resolve().parent.parent / "examples" / "genomics" / "deepvariant"
)
sys.path.insert(0, str(DV_DIR))

torch = pytest.importorskip("torch")


def test_crc32c_known_vector():
    from dv_io import crc32c

    # RFC 3720 test vector: 32 bytes of zeros.
    assert crc32c(b"\x00" * 32) == 0x8A9136AA
    assert crc32c(b"123456789") == 0xE3069283


def test_tfrecord_cvo_roundtrip(tmp_path):
    from dv_io import TFRecordWriter, decode_cvo, encode_cvo, read_tfrecords, round_gls

    probs = round_gls(np.array([0.1, 0.7, 0.2], np.float32))
    path = str(tmp_path / "cvo-00000-of-00001.tfrecord.gz")
    with TFRecordWriter(path) as w:
        w.write(encode_cvo(b"\x08\x01", b"\x08\x00", probs))
    recs = list(read_tfrecords(path, check_crc=True))
    assert len(recs) == 1
    _, got = decode_cvo(recs[0])
    np.testing.assert_allclose(got, [float(p) for p in probs])
    assert abs(got.sum() - 1) < 1e-6


def test_model_shapes_and_mapping_order():
    from dv_inception import INPUT_SHAPE, build_model
    from dv_io import synthetic_pileups

    model = build_model()
    units = model.conv_units()
    assert len(units) == 94  # Keras conv2d .. conv2d_93
    assert units[0].conv.weight.shape == (32, 7, 3, 3)
    assert units[93].conv.weight.shape == (192, 2048, 1, 1)
    x = torch.from_numpy(synthetic_pileups(2, INPUT_SHAPE, seed=1))
    with torch.inference_mode():
        p, logits = model(x)
    assert p.shape == (2, 3) and logits.shape == (2, 3)
    torch.testing.assert_close(p.sum(1), torch.ones(2), atol=1e-6, rtol=0)
