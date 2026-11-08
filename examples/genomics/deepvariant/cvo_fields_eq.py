"""CVO format gate: every test CallVariantsOutput equals the reference record
for the same site in every field except ``genotype_probabilities``.

Runs inside the DeepVariant 1.9.0 image with its runfiles on ``PYTHONPATH``
(see ``run_wgs_neuron.sh``'s ``dvpy``), so records are parsed by DeepVariant's
own protos. The Variant is compared after a deterministic re-serialization,
which sorts the call ``info`` map (``make_examples`` and ``call_variants``
write DP/AD/VAF in different orders, trnsci/trnsci#114).

    python3 cvo_fields_eq.py 'cpu/cvo-*.tfrecord.gz' 'neuron/cvo-neuron-*.tfrecord.gz'

Prints one JSON line. ``format_ok`` is true when every test record matched a
reference record, all non-probability fields are equal, each record has three
probabilities summing to 1 within 1e-6, and neither side carries
``debug_info``.
"""

import collections
import glob
import json
import sys

from deepvariant.protos import deepvariant_pb2
from third_party.nucleus.io import tfrecord


def read(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        comp = "GZIP" if p.endswith(".gz") else None
        for r in tfrecord.read_tfrecords(
            p, proto=deepvariant_pb2.CallVariantsOutput, compression_type=comp
        ):
            v = r.variant
            key = (
                v.reference_name,
                v.start,
                v.end,
                tuple(v.alternate_bases),
                tuple(r.alt_allele_indices.indices),
            )
            out.setdefault(key, []).append(r)
    return out


def without_probs(r):
    c = deepvariant_pb2.CallVariantsOutput()
    c.CopyFrom(r)
    del c.genotype_probabilities[:]
    return c.SerializeToString(deterministic=True)


def main():
    ref, test = read(sys.argv[1]), read(sys.argv[2])
    c = collections.Counter()
    for key, recs in test.items():
        for r in recs:
            c["test_records"] += 1
            c["test_debug_info"] += r.HasField("debug_info")
            c["three_probs"] += len(r.genotype_probabilities) == 3
            c["sum_to_1"] += abs(sum(r.genotype_probabilities) - 1) < 1e-6
            if key not in ref:
                c["unmatched"] += 1
                continue
            c["matched"] += 1
            c["ref_debug_info"] += ref[key][0].HasField("debug_info")
            c["fields_equal"] += without_probs(r) == without_probs(ref[key][0])
    n = c["test_records"]
    out = dict(c, ref_records=sum(len(v) for v in ref.values()))
    out["format_ok"] = bool(
        n
        and c["matched"] == n
        and c["fields_equal"] == n
        and c["three_probs"] == n
        and c["sum_to_1"] == n
        and c["test_debug_info"] == 0
        and c["ref_debug_info"] == 0
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
