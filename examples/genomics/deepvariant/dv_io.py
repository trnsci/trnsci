"""TensorFlow-free I/O for DeepVariant examples and call_variants outputs.

* TFRecord framing (read and write, plain or GZIP), with masked CRC32C.
* A minimal ``tf.train.Example`` decoder: ``make_examples`` writes
  ``image/encoded`` (raw uint8 HWC bytes), ``image/shape``,
  ``variant/encoded`` (a serialized ``nucleus.genomics.v1.Variant``) and
  ``alt_allele_indices/encoded``.
* A ``CallVariantsOutput`` encoder. Fields 1 and 2 are copied byte for byte
  from the example; field 3 is the packed ``repeated double``
  ``genotype_probabilities``. ``postprocess_variants`` reads these files.
* DeepVariant's ``round_gls`` (precision 10), reproduced with numpy float32
  scalars so the stored doubles match what ``call_variants`` stores.
* A synthetic pileup generator for dry runs without ``make_examples``.

Nothing here imports TensorFlow or protobuf.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import io
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# CRC32C (Castagnoli), as used by TFRecord framing.
# --------------------------------------------------------------------------

try:  # optional fast path
    import google_crc32c as _gcrc

    def crc32c(data: bytes) -> int:
        return _gcrc.value(data)

except ImportError:  # pragma: no cover - exercised only without the wheel
    _TABLE = []
    for _i in range(256):
        _c = _i
        for _ in range(8):
            _c = (_c >> 1) ^ 0x82F63B78 if _c & 1 else _c >> 1
        _TABLE.append(_c)

    def crc32c(data: bytes) -> int:
        c = 0xFFFFFFFF
        t = _TABLE
        for b in data:
            c = t[(c ^ b) & 0xFF] ^ (c >> 8)
        return c ^ 0xFFFFFFFF


def masked_crc(data: bytes) -> int:
    c = crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


# --------------------------------------------------------------------------
# TFRecord framing
# --------------------------------------------------------------------------


def expand_sharded(pattern: str) -> list[str]:
    """Expand ``name@N`` / ``name-?????-of-N`` / globs into sorted paths."""
    m = re.match(r"^(.*)@(\d+)(.*)$", pattern)
    if m:
        n = int(m.group(2))
        return [f"{m.group(1)}-{i:05d}-of-{n:05d}{m.group(3)}" for i in range(n)]
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths


def _open(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def read_tfrecords(path: str, check_crc: bool = False) -> Iterator[bytes]:
    with _open(path, "rb") as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) != 12:
                raise OSError(f"truncated TFRecord header in {path}")
            (n,) = struct.unpack("<Q", head[:8])
            data = f.read(n)
            tail = f.read(4)
            if len(data) != n or len(tail) != 4:
                raise OSError(f"truncated TFRecord in {path}")
            if check_crc:
                if struct.unpack("<I", head[8:])[0] != masked_crc(head[:8]):
                    raise OSError("TFRecord length CRC mismatch")
                if struct.unpack("<I", tail)[0] != masked_crc(data):
                    raise OSError("TFRecord data CRC mismatch")
            yield data


class TFRecordWriter:
    def __init__(self, path: str):
        self._f = _open(path, "wb")

    def write(self, data: bytes) -> None:
        head = struct.pack("<Q", len(data))
        self._f.write(head)
        self._f.write(struct.pack("<I", masked_crc(head)))
        self._f.write(data)
        self._f.write(struct.pack("<I", masked_crc(data)))

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# Minimal protobuf wire-format helpers
# --------------------------------------------------------------------------


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = 0
    out = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7


def _enc_varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _fields(buf: bytes) -> Iterator[tuple[int, int, object]]:
    """Yield (field_number, wire_type, value) for one message."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v = buf[i : i + 8]
            i += 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v = buf[i : i + ln]
            i += ln
        elif wt == 5:
            v = buf[i : i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        yield fno, wt, v


def parse_tf_example(buf: bytes) -> dict[str, list]:
    """tf.train.Example -> {name: [values]} (bytes or int64 lists)."""
    out: dict[str, list] = {}
    for fno, _, features in _fields(buf):
        if fno != 1:
            continue
        for fno2, _, entry in _fields(features):
            if fno2 != 1:
                continue
            key, feat = None, b""
            for f3, _, v in _fields(entry):
                if f3 == 1:
                    key = bytes(v).decode()
                elif f3 == 2:
                    feat = v
            vals: list = []
            for kind, _, lst in _fields(feat):
                if kind == 1:  # BytesList
                    vals = [bytes(v) for f, _, v in _fields(lst) if f == 1]
                elif kind == 3:  # Int64List (packed or not)
                    for f, wt, v in _fields(lst):
                        if f != 1:
                            continue
                        if wt == 2:
                            j = 0
                            while j < len(v):
                                x, j = _varint(v, j)
                                vals.append(x - (1 << 64) if x >= 1 << 63 else x)
                        else:
                            vals.append(v)
                elif kind == 2:  # FloatList
                    for f, wt, v in _fields(lst):
                        if f == 1 and wt == 2:
                            vals.extend(np.frombuffer(v, "<f4").tolist())
            out[key] = vals
    return out


@dataclass
class Example:
    image: np.ndarray  # (H, W, C) uint8
    variant: bytes  # serialized nucleus Variant
    alt_allele_indices: bytes  # serialized CallVariantsOutput.AltAlleleIndices

    @property
    def key(self) -> str:
        """Stable id for matching records across runs."""
        return hashlib.sha1(self.variant + b"|" + self.alt_allele_indices).hexdigest()[
            :16
        ]


def decode_example(buf: bytes, shape: tuple[int, int, int] | None = None) -> Example:
    feats = parse_tf_example(buf)
    if shape is None and "image/shape" in feats:
        shape = tuple(int(v) for v in feats["image/shape"])
    img = np.frombuffer(feats["image/encoded"][0], np.uint8)
    if shape is not None:
        img = img.reshape(shape)
    return Example(
        image=img,
        variant=feats.get("variant/encoded", [b""])[0],
        alt_allele_indices=feats.get("alt_allele_indices/encoded", [b""])[0],
    )


def iter_examples(pattern: str, shape=None) -> Iterator[Example]:
    for path in expand_sharded(pattern):
        for rec in read_tfrecords(path):
            yield decode_example(rec, shape)


# --------------------------------------------------------------------------
# CallVariantsOutput
# --------------------------------------------------------------------------

GL_PRECISION = 10  # call_variants._GL_PRECISION


def round_gls(gls, precision: int | None = GL_PRECISION) -> list:
    """DeepVariant's ``call_variants.round_gls`` on float32 scalars.

    ``call_variants`` iterates a float32 numpy row, so each element (and
    the sums) are ``np.float32``; we keep that arithmetic so the doubles we
    store match the reference implementation's.
    """
    gls = [np.float32(g) for g in gls]
    if abs(sum(gls) - 1) > 1e-6:
        raise ValueError(f"genotype likelihoods do not sum to one: {gls}")
    if precision is None:
        return gls
    min_ix = int(np.argmin(gls))
    rounded = [round(g, precision) for g in gls]
    rounded[min_ix] = max(
        0.0, round(1 - sum(rounded[:min_ix] + rounded[min_ix + 1 :]), precision)
    )
    return rounded


def encode_cvo(variant: bytes, alt_allele_indices: bytes, probs) -> bytes:
    """Serialize a CallVariantsOutput (fields 1, 2, 3; no debug_info)."""
    out = io.BytesIO()
    if variant:
        out.write(b"\x0a" + _enc_varint(len(variant)) + variant)
    if alt_allele_indices:
        out.write(b"\x12" + _enc_varint(len(alt_allele_indices)) + alt_allele_indices)
    dbl = struct.pack(f"<{len(probs)}d", *(float(p) for p in probs))
    out.write(b"\x1a" + _enc_varint(len(dbl)) + dbl)
    return out.getvalue()


def decode_cvo(buf: bytes) -> tuple[str, np.ndarray]:
    """CallVariantsOutput -> (key, genotype_probabilities float64)."""
    variant = alt = b""
    probs: list[float] = []
    for fno, wt, v in _fields(buf):
        if fno == 1:
            variant = bytes(v)
        elif fno == 2:
            alt = bytes(v)
        elif fno == 3:
            if wt == 2:
                probs.extend(struct.unpack(f"<{len(v) // 8}d", v))
            else:
                probs.append(struct.unpack("<d", v)[0])
    key = hashlib.sha1(variant + b"|" + alt).hexdigest()[:16]
    return key, np.asarray(probs, np.float64)


# --------------------------------------------------------------------------
# Synthetic pileups (dry runs only; not biologically meaningful)
# --------------------------------------------------------------------------


def synthetic_pileups(n: int, shape=(100, 221, 7), seed: int = 0) -> np.ndarray:
    """Pileup-like uint8 tensors with DeepVariant's channel value alphabets.

    Rows 0-4 are the reference band, rows 5.. are reads of random length
    with a few mismatch columns. Channels follow the 1.9 WGS order
    (base, base quality, mapping quality, strand, read-supports-variant,
    base-differs-from-ref, insert size). Use for shape, throughput and
    numerical-agreement checks, never for accuracy claims.
    """
    rng = np.random.default_rng(seed)
    h, w, c = shape
    out = np.zeros((n, h, w, c), np.uint8)
    bases = np.array([250, 180, 100, 30], np.uint8)  # A G T C
    for k in range(n):
        ref = bases[rng.integers(0, 4, w)]
        img = out[k]
        img[:5, :, 0] = ref
        img[:5, :, 1] = 254
        img[:5, :, 2] = 254
        img[:5, :, 3] = 254
        img[:5, :, 4] = 254
        img[:5, :, 5] = 50
        depth = int(rng.integers(10, h - 5))
        alt_frac = rng.choice([0.0, 0.5, 1.0])
        var_col = w // 2
        alt_base = bases[(np.where(bases == ref[var_col])[0][0] + 1) % 4]
        for r in range(5, 5 + depth):
            s = int(rng.integers(0, w // 2))
            e = int(rng.integers(w // 2 + 1, w + 1))
            row = img[r]
            row[s:e, 0] = ref[s:e]
            supports = rng.random() < alt_frac
            if supports:
                row[var_col, 0] = alt_base
            noise = rng.random(e - s) < 0.01
            row[s:e, 0][noise] = bases[rng.integers(0, 4, int(noise.sum()))]
            row[s:e, 1] = np.clip(rng.normal(200, 30, e - s), 0, 254).astype(np.uint8)
            row[s:e, 2] = np.uint8(rng.choice([254, 200, 120], p=[0.9, 0.07, 0.03]))
            row[s:e, 3] = np.uint8(70 if rng.random() < 0.5 else 240)
            row[s:e, 4] = np.uint8(254 if supports else 152)
            row[s:e, 5] = np.where(row[s:e, 0] != ref[s:e], 254, 50).astype(np.uint8)
            row[s:e, 6] = np.uint8(rng.integers(100, 160))
    return out
