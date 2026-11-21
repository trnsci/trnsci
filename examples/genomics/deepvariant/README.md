# DeepVariant `call_variants` on Trainium2 (exemplar, trnsci/trnsci#114)

DeepVariant's `call_variants` stage is CNN inference over pileup tensors,
which makes it the one clean Tensor Engine fit in a germline pipeline
(assessment in trnsci/trnsci#113). This directory holds the CPU-side
tooling: get the public model, rebuild it in PyTorch, check it against
TensorFlow, and run a batched inference probe on `cpu` or a TorchNeuron
`neuron` device. `make_examples` and `postprocess_variants` stay on the host,
unchanged, and the device reads the same examples bytes CPU `make_examples`
writes.

It uses none of the six trnsci libraries; it is an application exemplar.

## Model

| | |
|---|---|
| Release | DeepVariant 1.9.0, WGS model (BSD-3-Clause) |
| Source | `https://storage.googleapis.com/deepvariant/models/DeepVariant/1.9.0/savedmodels/deepvariant.wgs.savedmodel/` (same files as `/opt/models/wgs` in `google/deepvariant:1.9.0`) |
| Checksums | sha256 for every file in `fetch_model.py::MANIFEST` |
| Input | `(100, 221, 7)` uint8, channels `[1, 2, 3, 4, 5, 6, 19]` (from the model's `example_info.json`) |
| Graph | Keras InceptionV3 backbone (`include_top=False`, `pooling="avg"`), 94 conv+BN units, `Dense(3, softmax)` head named `classification` |
| Preprocess | `(x - 128) / 128`, applied inside the PyTorch module so uint8 crosses the host-device link |

Weights are downloaded at run time and are never committed.

## Files

| file | what it does |
|---|---|
| `fetch_model.py` | downloads the SavedModel (and optionally DeepVariant's golden test examples) and verifies sha256 |
| `dv_inception.py` | the classifier rebuilt natively in PyTorch; `ConvBN` units are created in Keras order so unit *i* takes `conv2d_i` / `batch_normalization_i` |
| `convert_weights.py` | SavedModel variables -> PyTorch state dict, then the TF-vs-PyTorch probability check (needs TensorFlow; conversion host only) |
| `dv_io.py` | TF-free TFRecord read/write (GZIP, CRC32C), `tf.Example` decoding, `CallVariantsOutput` encoding, DeepVariant's `round_gls`, synthetic pileups |
| `probe.py` | batched inference on `--device cpu\|neuron`, fp32 or bf16 backbone, eager or `torch.compile`; writes probabilities and/or a `CallVariantsOutput` TFRecord for `postprocess_variants`; reports examples/s |
| `compare.py` | `probs`: two runs (probe `.npz` or DeepVariant CVO TFRecords) matched per record; `vcf`: record-level GT / FILTER / GQ diff |
| `cv_neuron_wgs.py` | whole-genome `call_variants`: one process per NeuronCore, each with a pinned decode pool over its share of the `make_examples` shards; writes one stock `CallVariantsOutput` file per input shard, resumable per shard; logs examples/s, host busy and input-wait (device idle) |
| `run_wgs_neuron.sh` | the window run: S3 sync, example_info check, compile warm-up, 4 workers under the chip lock, CVO gates, `postprocess_variants`, VCF/gVCF compares |
| `cvo_fields_eq.py` | CVO format gate (runs in the DeepVariant image): every non-probability field equals the reference CVO's |

## CPU conversion check

```bash
pip install "tensorflow==2.16.*" torch numpy     # conversion host only
M=$HOME/dv/model; G=$HOME/dv/golden
python fetch_model.py --dest $M --golden $G
python convert_weights.py --savedmodel $M/savedmodels/deepvariant.wgs.savedmodel \
    --out $M/dv19_wgs.pt --verify-examples "$G/golden.calling_examples.tfrecord.gz" \
    --verify-synthetic 256
```

Result on an arm64 macOS host (TensorFlow 2.16.2, PyTorch 2.15.0):

| input set | n | max \|Δp\| | mean \|Δp\| | identical argmax |
|---|---|---|---|---|
| DeepVariant r1.9 `golden.calling_examples` (real pileups) | 84 | 1.0e-6 | 5.6e-8 | 84/84 |
| synthetic pileups | 256 | 6.0e-7 | 3.1e-8 | 256/256 |

Both are inside the #114 conversion bar (≤ 1e-5, identical argmax). The
`CallVariantsOutput` encoder re-encodes all 84 records of DeepVariant's
`golden.postprocess_single_site_input` byte for byte, and TensorFlow's
`TFRecordDataset` reads the probe's GZIP output with CRC checks on.

The chr20 check on real `make_examples` output runs on an x86_64 host,
because the DeepVariant container is published for `linux/amd64` only.

## Probe

```bash
# dry run, no make_examples needed
python probe.py --weights $M/dv19_wgs.pt --synthetic 2048 --batch-size 256

# real examples: decode once to a uint8 cache, so decode is out of the timing
python probe.py --weights $M/dv19_wgs.pt --examples "make_examples.tfrecord@4.gz" \
    --cache ex.npy --out-probs cpu.npz \
    --out-cvo out/call_variants_output-00000-of-00001.tfrecord.gz

# Trn2, TorchNeuron (SDK 2.33, torch 2.14), one fixed-shape compile
python probe.py --weights $M/dv19_wgs.pt --npy ex.npy --device neuron \
    --mode compile --batch-size 1024 --dtype fp32 --out-probs neuron_fp32.npz
python compare.py probs cpu.npz neuron_fp32.npz --tol 1e-5
```

`postprocess_variants --infile out/call_variants_output@1.tfrecord.gz`
then builds the VCF exactly as it does after CPU `call_variants`.

Notes:

- **Fixed shape.** The last batch is zero-padded to `--batch-size`, so a run
  compiles one graph. Padded rows are dropped before writing.
- **fp32 means fp32.** On `neuron` the probe adds `--auto-cast=none` to
  `NEURON_CC_FLAGS` unless the caller already set an `--auto-cast` flag.
- **bf16** casts the convolutional backbone only. The pooled features, head
  and softmax stay fp32, so the probabilities still sum to 1 within
  `round_gls`'s 1e-6 check. bf16 is reported separately and adopted only if
  it meets the same criteria.
- **Small model.** DeepVariant 1.9 WGS by default sends easy candidates to a
  small model inside `make_examples`, and those never reach `call_variants`.
  For the CNN identity and throughput runs, generate examples with
  `--disable_small_model` so every candidate goes through the CNN.
- **Golden CVOs are not a model reference.** DeepVariant's
  `golden.postprocess_single_site_input` fixture does not line up with the
  1.9.0 WGS model on the golden examples: only 20 of 84 records share a
  variant key, and their probabilities differ by up to 0.19. Compare against
  a CPU `call_variants` run of the 1.9.0 container instead.

## Whole genome (`cv_neuron_wgs.py`)

`call_variants` for a whole genome's `make_examples` output (`me.tfrecord@64.gz`)
on a Trn2 chip, one worker process per NeuronCore:

```bash
for k in 0 1 2 3; do
  NEURON_RT_VISIBLE_CORES=$k timeout 1740 python cv_neuron_wgs.py --worker $k --num-workers 4 \
      --examples in/tmp/me.tfrecord@64.gz --weights $M/dv19_wgs.pt --outdir out/fp32 \
      --device neuron --check-example-info > w$k.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
python cv_neuron_wgs.py --summarize out/fp32
postprocess_variants --infile out/fp32/cvo-neuron@64.tfrecord.gz ...   # stock DV 1.9.0
```

- **Shards.** Worker *k* takes the shards *i* with *i* mod 4 = *k*, and writes
  `cvo-neuron-000ii-of-00064.tfrecord.gz`. A shard is renamed into place, and
  marked in `done/`, only once it is complete. A rerun skips completed shards
  and redoes partial ones.
- **Host side.** Decoding runs in spawned processes (TF-free: `dv_io.example_fields`),
  into a shared-memory ring of fixed-shape uint8 batches. The device
  process only copies a slot to the device and runs the model, so torch and
  NKI are touched from one thread of one process per core.
- **CVO format.** `variant` and `alt_allele_indices` are the example's bytes;
  `genotype_probabilities` is `round_gls(p, 10)`. There is no `debug_info`, as in stock
  `call_variants` without `--include_debug_info`.
- **Smoke run.** `--limit N` caps examples per shard and `--shards` picks shards. Both run on `--device cpu`.

`run_wgs_neuron.sh` wraps the whole window run (stages `fetch tools check warm cv
cvocmp post cmp summary`). See the header of that script for its environment.
