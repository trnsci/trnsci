"""Convert the DeepVariant 1.9 WGS SavedModel to a PyTorch state dict, and
check the PyTorch model against TensorFlow on CPU.

Needs TensorFlow (CPU) on the conversion host only; inference does not.

    python fetch_model.py --dest $M --golden $G
    python convert_weights.py \
        --savedmodel $M/savedmodels/deepvariant.wgs.savedmodel \
        --out $M/dv19_wgs.pt \
        --verify-examples "$G/golden.calling_examples.tfrecord.gz" \
        --verify-synthetic 256 --save-reference $M/tf_reference.npz

Mapping: Keras ``conv2d_i/kernel`` (kh, kw, cin, cout) -> unit i
``conv.weight`` (cout, cin, kh, kw); ``batch_normalization_i/{beta,
moving_mean, moving_variance}`` -> unit i ``bn.{bias, running_mean,
running_var}`` (gamma is 1: Keras ``scale=False``); ``classification/
{kernel, bias}`` -> ``classification.{weight (transposed), bias}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dv_inception import INPUT_SHAPE, build_model  # noqa: E402
from dv_io import iter_examples, synthetic_pileups  # noqa: E402


def _idx(name: str, stem: str) -> int | None:
    m = re.fullmatch(rf"{stem}(?:_(\d+))?/[a-z_]+:0", name)
    if not m:
        return None
    return int(m.group(1) or 0)


def convert(savedmodel: str) -> tuple[dict, object]:
    import tensorflow as tf

    loaded = tf.saved_model.load(savedmodel)
    var = {v.name: v.numpy() for v in loaded.variables}
    model = build_model()
    units = model.conv_units()
    sd = model.state_dict()
    name_of = {id(m): n for n, m in model.named_modules()}
    used = set()
    for name, arr in var.items():
        if name.startswith("classification/"):
            continue
        i = _idx(name, "conv2d")
        if i is not None:
            p = name_of[id(units[i])] + ".conv.weight"
            t = torch.from_numpy(np.ascontiguousarray(arr.transpose(3, 2, 0, 1)))
        else:
            i = _idx(name, "batch_normalization")
            if i is None:
                raise KeyError(f"unmapped variable {name}")
            leaf = {
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var",
            }[name.split("/")[1][:-2]]
            p = name_of[id(units[i])] + ".bn." + leaf
            t = torch.from_numpy(arr.copy())
        if tuple(sd[p].shape) != tuple(t.shape):
            raise ValueError(f"{name} {tuple(t.shape)} -> {p} {tuple(sd[p].shape)}")
        sd[p] = t
        used.add(p)
    sd["classification.weight"] = torch.from_numpy(
        var["classification/kernel:0"].T.copy()
    )
    sd["classification.bias"] = torch.from_numpy(var["classification/bias:0"].copy())
    used |= {"classification.weight", "classification.bias"}
    for unit in units:  # gamma fixed at 1
        sd[name_of[id(unit)] + ".bn.weight"] = torch.ones_like(unit.bn.weight)
        used.add(name_of[id(unit)] + ".bn.weight")
    missing = [k for k in sd if k not in used and not k.endswith("num_batches_tracked")]
    if missing:
        raise KeyError(f"state-dict entries not filled: {missing[:5]}")
    print(f"mapped {len(var)} TF variables onto {len(units)} ConvBN units + head")
    return sd, loaded


def tf_predict(loaded, images_u8: np.ndarray, batch: int = 64) -> np.ndarray:
    import tensorflow as tf

    fn = loaded.signatures["serving_default"]
    outs = []
    for s in range(0, len(images_u8), batch):
        x = tf.cast(tf.constant(images_u8[s : s + batch]), tf.float32)
        x = tf.math.divide(tf.subtract(x, 128.0), 128.0)  # dv_utils.preprocess_images
        outs.append(fn(x)["classification"].numpy())
    return np.concatenate(outs)


@torch.no_grad()
def torch_predict(model, images_u8: np.ndarray, batch: int = 64) -> np.ndarray:
    outs = []
    for s in range(0, len(images_u8), batch):
        p, _ = model(torch.from_numpy(images_u8[s : s + batch]))
        outs.append(p.numpy())
    return np.concatenate(outs)


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepVariant 1.9 SavedModel -> PyTorch")
    ap.add_argument("--savedmodel", required=True)
    ap.add_argument("--out", required=True, help="output .pt state dict")
    ap.add_argument(
        "--verify-examples", help="TFRecord pattern of make_examples output"
    )
    ap.add_argument(
        "--max-examples", type=int, default=0, help="cap on real examples (0 = all)"
    )
    ap.add_argument(
        "--verify-synthetic", type=int, default=0, help="number of synthetic pileups"
    )
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument(
        "--save-reference", help="write TF probabilities + inputs' keys to this .npz"
    )
    args = ap.parse_args()

    info_path = os.path.join(args.savedmodel, "example_info.json")
    info = json.load(open(info_path))
    if tuple(info["shape"]) != INPUT_SHAPE:
        raise SystemExit(
            f"model expects {info['shape']}, module built for {INPUT_SHAPE}"
        )
    print(f"example_info.json: {info}")

    sd, loaded = convert(args.savedmodel)
    torch.save(sd, args.out)
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"wrote {args.out} sha256={digest}")

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = build_model(args.out)
    sets = {}
    if args.verify_examples:
        exs = []
        for ex in iter_examples(args.verify_examples, INPUT_SHAPE):
            exs.append(ex)
            if args.max_examples and len(exs) >= args.max_examples:
                break
        sets["real"] = (np.stack([e.image for e in exs]), [e.key for e in exs])
    if args.verify_synthetic:
        syn = synthetic_pileups(args.verify_synthetic)
        sets["synthetic"] = (syn, [f"syn{i}" for i in range(len(syn))])

    ok = True
    ref = {}
    for label, (imgs, keys) in sets.items():
        p_tf = tf_predict(loaded, imgs)
        p_pt = torch_predict(model, imgs)
        d = np.abs(p_tf.astype(np.float64) - p_pt.astype(np.float64))
        same_argmax = int((p_tf.argmax(1) == p_pt.argmax(1)).sum())
        hist = np.bincount(p_tf.argmax(1), minlength=3).tolist()
        print(
            f"[{label}] n={len(imgs)} max|dp|={d.max():.3e} mean|dp|={d.mean():.3e} "
            f"argmax identical {same_argmax}/{len(imgs)} TF class counts={hist}"
        )
        ok &= bool(d.max() <= args.tol and same_argmax == len(imgs))
        ref[f"{label}_tf_probs"] = p_tf
        ref[f"{label}_pt_probs"] = p_pt
        ref[f"{label}_keys"] = np.array(keys)
    if args.save_reference and ref:
        np.savez(args.save_reference, **ref)
        print(f"wrote {args.save_reference}")
    if sets:
        print("PASS" if ok else "FAIL", f"(tol {args.tol:g})")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
