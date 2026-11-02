"""Batched DeepVariant 1.9 ``call_variants`` inference probe (CPU or Neuron).

Inputs (one of):
  --examples PATTERN   make_examples TFRecords (``name@N``, globs, ``.gz`` ok)
  --npy FILE           uint8 cache written by ``--cache`` (decode excluded from timing)
  --synthetic N        generated pileups, for dry runs

Outputs:
  --out-probs FILE.npz  keys, probabilities, logits (for compare.py)
  --out-cvo FILE        CallVariantsOutput TFRecord (GZIP if it ends in .gz)
                        that DeepVariant's postprocess_variants consumes
  --report FILE.json    timings and examples/s

Device paths:
  --device cpu
  --device neuron[:k]   TorchNeuron (Neuron SDK 2.33, torch 2.14, Trn2/Trn3).
                        ``--mode compile`` = torch.compile(backend="neuron") at a
                        fixed batch (last batch zero-padded, so one compile);
                        ``--mode eager`` runs op by op.
  --dtype fp32|bf16     bf16 casts the convolutional backbone only; the pooled
                        features, Dense head and softmax stay fp32.

For fp32 the probe puts ``--auto-cast=none`` in NEURON_CC_FLAGS (unless the
caller already set an --auto-cast flag) so the compiler does not downcast
matmuls/convolutions to bf16 behind our back.

    python probe.py --weights dv19_wgs.pt --synthetic 2048 --device cpu
    python probe.py --weights dv19_wgs.pt --npy chr20.npy --device neuron \
        --mode compile --batch-size 1024 --out-probs neuron_fp32.npz
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dv_io import (  # noqa: E402
    TFRecordWriter,
    encode_cvo,
    expand_sharded,
    iter_examples,
    round_gls,
    synthetic_pileups,
)


def load_inputs(args):
    """-> (images uint8 [N,H,W,C], keys, variants, alt_allele_indices)."""
    from dv_inception import INPUT_SHAPE

    if args.synthetic:
        imgs = synthetic_pileups(args.synthetic, INPUT_SHAPE, seed=args.seed)
        keys = np.array([f"syn{i}" for i in range(len(imgs))])
        if args.cache:
            np.save(args.cache, imgs)
            np.savez(
                args.cache[: -len(".npy")] + ".meta.npz",
                keys=keys,
                variants=np.array([""] * len(imgs)),
                alts=np.array([""] * len(imgs)),
            )
            print(f"cached {args.cache} (+ .meta.npz)")
        return imgs, keys, None, None
    if args.npy:
        imgs = np.load(args.npy, mmap_mode="r")
        side = np.load(args.npy[: -len(".npy")] + ".meta.npz", allow_pickle=False)
        return imgs, side["keys"], side["variants"], side["alts"]
    t0 = time.perf_counter()
    imgs, keys, variants, alts = [], [], [], []
    for ex in iter_examples(args.examples, INPUT_SHAPE):
        imgs.append(ex.image)
        keys.append(ex.key)
        variants.append(ex.variant)
        alts.append(ex.alt_allele_indices)
        if args.max_examples and len(imgs) >= args.max_examples:
            break
    imgs = np.stack(imgs)
    dt = time.perf_counter() - t0
    print(
        f"decoded {len(imgs)} examples from {len(expand_sharded(args.examples))} file(s) in {dt:.1f}s "
        f"({len(imgs) / dt:.0f} ex/s host decode)"
    )
    variants = np.array(variants, dtype=object)
    alts = np.array(alts, dtype=object)
    if args.cache:
        np.save(args.cache, imgs)
        np.savez(
            args.cache[: -len(".npy")] + ".meta.npz",
            keys=np.array(keys),
            variants=np.array([v.hex() for v in variants]),
            alts=np.array([a.hex() for a in alts]),
        )
        print(f"cached {args.cache} (+ .meta.npz)")
    return imgs, np.array(keys), variants, alts


def _as_bytes(x) -> bytes:
    return x if isinstance(x, bytes) else bytes.fromhex(str(x))


def setup_device(args):
    import torch

    if args.device.startswith("neuron"):
        flags = os.environ.get("NEURON_CC_FLAGS", "")
        if "--auto-cast" not in flags:
            flags = (flags + " --auto-cast=none").strip()
            os.environ["NEURON_CC_FLAGS"] = flags
        import torch_neuronx  # noqa: F401  (registers the "neuron" device)

        print(
            f"torch_neuronx {getattr(torch_neuronx, '__version__', '?')}  NEURON_CC_FLAGS={flags!r}  "
            f"NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES', '')!r}"
        )
    else:
        if args.threads:
            torch.set_num_threads(args.threads)
        print(f"cpu threads={torch.get_num_threads()}  {platform.machine()}")
    return torch.device(args.device)


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepVariant call_variants probe")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--examples")
    src.add_argument("--npy")
    src.add_argument("--synthetic", type=int)
    ap.add_argument("--weights", required=True, help=".pt from convert_weights.py")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--mode", choices=["eager", "compile"], default="eager")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--allow-graph-breaks",
        action="store_true",
        help="torch.compile without fullgraph",
    )
    ap.add_argument(
        "--host-cast",
        action="store_true",
        help="ship fp32 (cast on host) instead of uint8",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="CPU intra-op threads (0 = torch default)",
    )
    ap.add_argument(
        "--warmup", type=int, default=2, help="untimed batches before the timed pass"
    )
    ap.add_argument(
        "--repeat", type=int, default=1, help="timed passes over the inputs"
    )
    ap.add_argument(
        "--shard", default="0/1", help="i/n: process every n-th batch starting at i"
    )
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", help="write decoded/generated inputs to this .npy")
    ap.add_argument("--out-probs")
    ap.add_argument("--out-cvo")
    ap.add_argument("--report")
    args = ap.parse_args()
    if args.cache and not args.cache.endswith(".npy"):
        ap.error("--cache must end in .npy")

    imgs, keys, variants, alts = load_inputs(args)
    device = setup_device(args)

    import torch

    from dv_inception import build_model

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = build_model(args.weights, backbone_dtype=dtype).to(device)
    fwd = model
    if args.mode == "compile":
        backend = "neuron" if device.type == "neuron" else "inductor"
        fwd = torch.compile(
            model, backend=backend, dynamic=False, fullgraph=not args.allow_graph_breaks
        )

    n = len(imgs)
    bs = args.batch_size
    si, sn = (int(v) for v in args.shard.split("/"))
    batches = [(s, min(s + bs, n)) for s in range(0, n, bs)][si::sn]
    rows = (
        np.concatenate([np.arange(s, e) for s, e in batches])
        if batches
        else np.zeros(0, int)
    )

    def run_batch(s, e):
        x = np.asarray(imgs[s:e])
        if e - s < bs:  # fixed shape -> one compiled graph
            x = np.concatenate([x, np.zeros((bs - (e - s),) + x.shape[1:], np.uint8)])
        if args.host_cast:
            x = x.astype(np.float32)
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        p, lg = fwd(xt)
        return p.cpu()[: e - s].numpy(), lg.cpu()[: e - s].numpy()

    report = {
        "device": args.device,
        "dtype": args.dtype,
        "mode": args.mode,
        "batch_size": bs,
        "n_examples": int(len(rows)),
        "torch": torch.__version__,
        "host_cast": args.host_cast,
        "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS", ""),
        "host": platform.node(),
    }
    with torch.inference_mode():
        t0 = time.perf_counter()
        run_batch(*batches[0])
        report["first_batch_s"] = time.perf_counter() - t0  # includes any compile
        print(f"first batch (compile + run): {report['first_batch_s']:.1f}s")
        for b in batches[1 : 1 + max(0, args.warmup - 1)]:
            run_batch(*b)

        probs = np.zeros((len(rows), 3), np.float32)
        logits = np.zeros((len(rows), 3), np.float32)
        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            off = 0
            for s, e in batches:
                p, lg = run_batch(s, e)
                probs[off : off + e - s] = p
                logits[off : off + e - s] = lg
                off += e - s
            times.append(time.perf_counter() - t0)
    best = min(times)
    report["timed_pass_s"] = times
    report["examples_per_s"] = len(rows) / best
    print(
        f"{len(rows)} examples  best pass {best:.3f}s  -> {len(rows) / best:,.1f} examples/s "
        f"({args.device}, {args.dtype}, {args.mode}, batch {bs})"
    )
    sums = probs.astype(np.float64).sum(1)
    report["max_abs_prob_sum_minus_1"] = float(np.abs(sums - 1).max())
    report["class_counts"] = np.bincount(probs.argmax(1), minlength=3).tolist()
    print(
        f"class counts {report['class_counts']}  max|sum-1|={report['max_abs_prob_sum_minus_1']:.2e}"
    )

    if args.out_probs:
        np.savez(args.out_probs, keys=keys[rows], probs=probs, logits=logits)
        print(f"wrote {args.out_probs}")
    if args.out_cvo:
        if variants is None or not len(variants) or not str(variants[0]):
            raise SystemExit("--out-cvo needs real examples (variant records)")
        with TFRecordWriter(args.out_cvo) as w:
            for j, r in enumerate(rows):
                w.write(
                    encode_cvo(
                        _as_bytes(variants[r]), _as_bytes(alts[r]), round_gls(probs[j])
                    )
                )
        print(f"wrote {args.out_cvo}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
