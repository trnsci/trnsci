"""Compare two call_variants runs, or two DeepVariant VCFs.

Probabilities: each side is a probe ``.npz`` (``--out-probs``) or a
CallVariantsOutput TFRecord pattern (e.g. CPU DeepVariant's
``call_variants_output@N.tfrecord.gz``). Records are matched by a hash of
the serialized variant + alt-allele indices.

    python compare.py probs cpu_cvo.tfrecord.gz neuron_fp32.npz --tol 1e-5

VCFs: record-level GT / FILTER / GQ comparison keyed on CHROM, POS, REF, ALT.

    python compare.py vcf cpu.vcf.gz neuron.vcf.gz
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dv_io import decode_cvo, expand_sharded, read_tfrecords  # noqa: E402


def load_probs(src: str) -> dict[str, np.ndarray]:
    if src.endswith(".npz"):
        z = np.load(src)
        return {str(k): p.astype(np.float64) for k, p in zip(z["keys"], z["probs"])}
    out = {}
    for path in expand_sharded(src):
        for rec in read_tfrecords(path):
            k, p = decode_cvo(rec)
            out[k] = p
    return out


def cmd_probs(a: str, b: str, tol: float, show: int) -> int:
    pa, pb = load_probs(a), load_probs(b)
    common = sorted(set(pa) & set(pb))
    print(
        f"A={len(pa)}  B={len(pb)}  matched={len(common)}  only-A={len(pa) - len(common)}  "
        f"only-B={len(pb) - len(common)}"
    )
    if not common:
        return 1
    A = np.stack([pa[k] for k in common])
    B = np.stack([pb[k] for k in common])
    d = np.abs(A - B).max(1)
    ga, gb = A.argmax(1), B.argmax(1)
    disc = np.nonzero(ga != gb)[0]
    srt = np.sort(A, 1)
    margin = srt[:, -1] - srt[:, -2]
    print(
        f"max|dp|={d.max():.3e}  mean={d.mean():.3e}  p99={np.quantile(d, 0.99):.3e}  "
        f"over tol {tol:g}: {(d > tol).sum()}"
    )
    print(
        f"argmax identical: {len(common) - len(disc)}/{len(common)} "
        f"({100 * (1 - len(disc) / len(common)):.4f}%)"
    )
    for j in disc[:show]:
        print(
            f"  {common[j]}  A={np.round(A[j], 6)}  B={np.round(B[j], 6)}  A-margin={margin[j]:.3e}"
        )

    # A GQ-like view of the shift: -10 log10(1 - p_called), capped like DV's GQ.
    def gq(P):
        return np.minimum(
            99, np.floor(-10 * np.log10(np.clip(1 - P.max(1), 1e-10, 1)))
        ).astype(int)

    dgq = gq(B) - gq(A)
    print(
        "GQ-like delta histogram (B - A):", dict(sorted(Counter(dgq.tolist()).items()))
    )
    return 0 if (d.max() <= tol and not len(disc)) else 2


def _read_vcf(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    recs = {}
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            t = line.rstrip("\n").split("\t")
            fmt = t[8].split(":")
            smp = dict(zip(fmt, t[9].split(":")))
            recs[(t[0], int(t[1]), t[3], t[4])] = (
                smp.get("GT", "."),
                t[6],
                smp.get("GQ", "."),
            )
    return recs


def cmd_vcf(a: str, b: str, show: int) -> int:
    va, vb = _read_vcf(a), _read_vcf(b)
    keys = sorted(set(va) | set(vb))
    gt = filt = gq = only = 0
    gq_hist: Counter = Counter()
    lines = []
    for k in keys:
        if k not in va or k not in vb:
            only += 1
            lines.append(f"  only-{'B' if k not in va else 'A'} {k}")
            continue
        (ga, fa, qa), (gb, fb, qb) = va[k], vb[k]
        if ga != gb:
            gt += 1
            lines.append(f"  GT {k}: {ga} -> {gb} (FILTER {fa}->{fb}, GQ {qa}->{qb})")
        elif fa != fb:
            filt += 1
            lines.append(f"  FILTER {k}: {fa} -> {fb} (GQ {qa}->{qb})")
        if qa != qb:
            gq += 1
            try:
                gq_hist[int(qb) - int(qa)] += 1
            except ValueError:
                pass
    n = len(keys)
    print(
        f"records A={len(va)} B={len(vb)} union={n}  only-one-side={only}  GT diffs={gt}  "
        f"FILTER-only diffs={filt}  GQ diffs={gq}"
    )
    if n:
        print(f"GT identical on {100 * (n - gt - only) / n:.4f}% of records")
    print("GQ delta histogram (B - A):", dict(sorted(gq_hist.items())))
    for line in lines[:show]:
        print(line)
    return 0 if not (gt or only) else 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probs")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--show", type=int, default=50)
    v = sub.add_parser("vcf")
    v.add_argument("a")
    v.add_argument("b")
    v.add_argument("--show", type=int, default=200)
    args = ap.parse_args()
    if args.cmd == "probs":
        sys.exit(cmd_probs(args.a, args.b, args.tol, args.show))
    sys.exit(cmd_vcf(args.a, args.b, args.show))


if __name__ == "__main__":
    main()
