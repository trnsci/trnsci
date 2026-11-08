#!/usr/bin/env bash
# Whole-genome DeepVariant 1.9 call_variants on one Trn2 chip, then stock
# postprocess_variants and the aws-realign gates (trnsci/trnsci#114, window 6;
# aws-realign#126). Inputs: aws-realign's NA12878-on-CHM13 make_examples split
# (2.5M big-model examples in 64 shards), staged in the window bucket.
#
#   run_wgs_neuron.sh [STAGE ...]      default: fetch tools check warm cv cvocmp post cmp
#
# Stages (each is resumable; rerun a stage to pick up where it stopped):
#   fetch   s3 sync inputs + reference to $BASE (skips what is already there), sha256 checks
#   tools   pull the DeepVariant image; extract its runfiles for the proto-based gates
#   check   example_info.json of every shard vs the model (shape, channels, version)
#   warm    one worker, one core, 32 examples: populates the compile cache (LOCK=shared|exclusive|none)
#   cv      $NW workers, worker k on NeuronCore k, under the whole-chip lock; per-shard resumable
#   cvocmp  pre-postprocess gates vs the CPU call_variants CVOs: aws-realign cvo_cmp.py
#           (max |dp|, argmax), cvo_set_eq.py (parsed multiset), cvo_fields_eq.py (CVO format)
#   post    postprocess_variants (DV 1.9.0 image, aws-realign's flags) -> VCF + gVCF
#   cmp     aws-realign vcf_cmp.py vs the CPU-pipeline VCF; VCF and gVCF record-hash compare
#   summary per-worker and instance throughput, host busy, device idle, device memory
#
# The device stages never run anything on a core other than their own: worker k
# gets NEURON_RT_VISIBLE_CORES=k, and each worker is one process (no threads touch
# the device). Workers run under `timeout`; the script waits on their PIDs only.
#
# Environment (defaults in brackets):
#   BASE [/nvme/dvwgs]  local NVMe root (format/mount needs sudo; see DEVICE.md)
#   DTYPE [fp32]  BATCH [32 fp32, 64 bf16]  NW [4]  LIMIT [0 = all]  (LIMIT > 0: smoke run)
#   OUT [$BASE/out/$DTYPE, or $BASE/out/$DTYPE-limit$LIMIT]  PY [/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python]
#   WEIGHTS [$BASE/model/dv19_wgs.pt]  REALIGN_HACK [$BASE/aws-realign/hack]
#   S3_RUN [s3://trnsci-window-446c8388/dv-wgs/20261106T181212Z]  S3_REF [.../dv-wgs/ref]
#   AWS_REGION [sa-east-1]  DV_IMAGE [google/deepvariant:1.9.0]  DOCKER [sudo docker]
#   LOCK [exclusive]  LOCK_FILE [/home/ubuntu/chip.lock]  WANT_X [/home/ubuntu/chip.want_x]
#   EXCL_BUDGET_S [1800]  whole-chip budget: the lock is held for at most this long
#   MAIN_CPUS_k [k,k+6]  DECODE_CPUS [4,5,10,11]  DECODE_PROCS [2]  (physical core j = CPUs j, j+6)
#   DEVICE [neuron]  SHARDS [all]  (DEVICE=cpu SHARDS=0,1 LOCK=none: CPU dry run of the orchestration)
#   WARM_CORE [0]  WARM_CPUS [0,6]  core and CPUs for the warm stage (use CHIP_LOCK.md's under LOCK=shared)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-/nvme/dvwgs}
DTYPE=${DTYPE:-fp32}
if [[ $DTYPE == bf16 ]]; then BATCH=${BATCH:-64}; else BATCH=${BATCH:-32}; fi
NW=${NW:-4}
LIMIT=${LIMIT:-0}
if ((LIMIT > 0)); then OUT=${OUT:-$BASE/out/$DTYPE-limit$LIMIT}; else OUT=${OUT:-$BASE/out/$DTYPE}; fi
PY=${PY:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python}
export PATH=$PATH:/opt/aws/neuron/bin   # neuron-monitor, neuron-ls
WEIGHTS=${WEIGHTS:-$BASE/model/dv19_wgs.pt}
WEIGHTS_SHA256=9a1402ee75fce373c243aff4ed9a4fd43f73b41050f91bd1610030b186a89ca1
REALIGN_HACK=${REALIGN_HACK:-$BASE/aws-realign/hack}
S3_RUN=${S3_RUN:-s3://trnsci-window-446c8388/dv-wgs/20261106T181212Z}
S3_REF=${S3_REF:-s3://trnsci-window-446c8388/dv-wgs/ref}
export AWS_REGION=${AWS_REGION:-sa-east-1}
DV_IMAGE=${DV_IMAGE:-google/deepvariant:1.9.0}
read -r -a DOCKER <<< "${DOCKER:-sudo docker}"
LOCK=${LOCK:-exclusive}
LOCK_FILE=${LOCK_FILE:-/home/ubuntu/chip.lock}
WANT_X=${WANT_X:-/home/ubuntu/chip.want_x}
EXCL_BUDGET_S=${EXCL_BUDGET_S:-1800}
DECODE_CPUS=${DECODE_CPUS:-4,5,10,11}
DECODE_PROCS=${DECODE_PROCS:-2}
DEVICE=${DEVICE:-neuron}
SHARDS=${SHARDS:-}
SAMPLE=NA12878
NSH=64

IN=$BASE/in            # mirror of $S3_RUN (NA12878.vcf.gz, NA12878.g.vcf.gz, tmp/...)
T=$IN/tmp
R=$BASE/ref
EXAMPLES=$T/me.tfrecord@$NSH.gz
CV=("$PY" "$HERE/cv_neuron_wgs.py")
mkdir -p "$BASE"/{tmp,neff,hlo,logs} "$OUT"/reports
export TORCH_NEURONX_NEFF_CACHE_DIR=$BASE/neff TORCH_NEURONX_HLO_CACHE_DIR=$BASE/hlo
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT/run.log" >&2; }
stamp() { date -u +%s; }

# ---- device lock (CHIP_LOCK.md protocol) ---------------------------------
# lock_acquire: exclusive -> set want_x (only if not already set; remove only
# a flag we created) and take flock -x on fd 9; shared -> wait for want_x to
# clear and take flock -s; none -> no-op. Children that must not keep the
# lock (neuron-monitor) are started with 9>&-.
CREATED_WANT_X=0
LOCK_HELD=0
lock_acquire() {
	case $LOCK in
	none) return 0 ;;
	exclusive)
		if [[ ! -e $WANT_X ]]; then touch "$WANT_X" && CREATED_WANT_X=1; fi
		exec 9> "$LOCK_FILE"
		log "waiting for exclusive chip lock $LOCK_FILE"
		flock -x 9
		;;
	shared)
		until [[ ! -e $WANT_X ]]; do sleep 20; done
		exec 9> "$LOCK_FILE"
		flock -s 9
		;;
	*) echo "LOCK must be exclusive|shared|none" >&2; exit 2 ;;
	esac
	LOCK_HELD=1
	log "chip lock ($LOCK) held"
}
lock_release() {
	((LOCK_HELD)) || return 0
	LOCK_HELD=0
	flock -u 9 || true
	exec 9>&-
	if ((CREATED_WANT_X)); then rm -f "$WANT_X"; CREATED_WANT_X=0; fi
	log "chip lock released"
}
trap 'lock_release' EXIT

# ---- stages ---------------------------------------------------------------

stage_fetch() {
	mkdir -p "$IN" "$R" "$BASE/model"
	log "fetch: $S3_RUN -> $IN"
	aws s3 sync --only-show-errors "$S3_RUN/" "$IN/"
	aws s3 sync --only-show-errors --exclude '*' --include 'chm13.fa' --include 'chm13.fa.fai' \
		--include 'SHA256SUMS' "$S3_REF/" "$R/"
	(cd "$R" && grep -E '  chm13\.fa(\.fai)?$' SHA256SUMS | sha256sum -c -) | tee -a "$OUT/run.log"
	local n_me n_info n_small n_gvcf n_cpu
	n_me=$(find "$T" -name "me.tfrecord-*-of-000$NSH.gz" | wc -l)
	n_info=$(find "$T" -name "me.tfrecord-*-of-000$NSH.gz.example_info.json" | wc -l)
	n_small=$(find "$T" -name "me_call_variant_outputs.tfrecord-*-of-000$NSH.gz" | wc -l)
	n_gvcf=$(find "$T" -name "gvcf.tfrecord-*-of-000$NSH.gz" | wc -l)
	n_cpu=$(find "$T" -name 'cvo-*-of-00004.tfrecord.gz' | wc -l)
	log "inputs: me=$n_me example_info=$n_info small_cvo=$n_small gvcf=$n_gvcf cpu_cvo=$n_cpu $(du -sh "$IN" | cut -f1)"
	if ((n_me != NSH || n_info != NSH || n_small != NSH || n_gvcf != NSH || n_cpu != 4)) \
		|| [[ ! -s $IN/$SAMPLE.vcf.gz || ! -s $IN/$SAMPLE.g.vcf.gz ]]; then
		log "fetch: inputs incomplete (the bucket copy may still be in flight); rerun fetch"
		return 1
	fi
	[[ -s $WEIGHTS ]] || { log "missing $WEIGHTS (scp it from the Mac, DEVICE.md step 0)"; return 1; }
	echo "$WEIGHTS_SHA256  $WEIGHTS" | sha256sum -c - | tee -a "$OUT/run.log"
}

RF=$BASE/dv-runfiles/runfiles
dvpy() { "${DOCKER[@]}" run --rm -u 0:0 -v "$BASE:$BASE" -v "$HERE:$HERE" -v "$REALIGN_HACK:$REALIGN_HACK" \
	-w "$BASE" -e "RUNFILES_DIR=$RF" -e "PYTHONPATH=$RF/com_google_deepvariant:$RF" --entrypoint python3 "$DV_IMAGE" "$@"; }

stage_tools() {
	for f in vcf_cmp.py cvo_cmp.py cvo_set_eq.py unzip_runfiles.py; do
		[[ -f $REALIGN_HACK/ie/$f ]] || { log "missing $REALIGN_HACK/ie/$f (aws-realign checkout)"; return 1; }
	done
	"${DOCKER[@]}" image inspect "$DV_IMAGE" > /dev/null 2>&1 || "${DOCKER[@]}" pull -q "$DV_IMAGE"
	"${DOCKER[@]}" image inspect "$DV_IMAGE" --format '{{index .RepoDigests 0}}' | tee "$OUT/dv_image.txt"
	if [[ ! -d $RF ]]; then
		"${DOCKER[@]}" run --rm -u 0:0 -v "$BASE:$BASE" -v "$REALIGN_HACK:$REALIGN_HACK" "$DV_IMAGE" \
			python3 "$REALIGN_HACK/ie/unzip_runfiles.py" /opt/deepvariant/bin/postprocess_variants.zip "$BASE/dv-runfiles"
	fi
	"$PY" -c "import torch, torch_neuronx, numpy; print('torch', torch.__version__, 'torch_neuronx', torch_neuronx.__version__)" \
		| tee -a "$OUT/run.log"
	"$PY" -c "import psutil" 2> /dev/null || log "psutil: not in $PY; host busy falls back to /proc/stat"
	"$PY" -c "import google_crc32c" 2> /dev/null || log "google_crc32c: not in $PY; pure-Python CRC32C for the (small) CVO records"
	# TensorFlow is not needed (TF-free reader in dv_io.py); record whether the venv has it.
	"$PY" -c "import tensorflow as tf; print('tensorflow', tf.__version__)" 2>/dev/null | tee -a "$OUT/run.log" \
		|| log "tensorflow: not in $PY (not needed)"
}

stage_check() {
	"${CV[@]}" --examples "$EXAMPLES" --check-only 2>&1 | tee -a "$OUT/run.log"
}

# One worker: index k of nw, NeuronCore, main CPUs, outdir, timeout seconds, extra args...
launch_worker() {
	local k=$1 nw=$2 core=$3 main_cpus=$4 od=$5 to=$6; shift 6
	mkdir -p "$BASE/tmp/w$core" "$od/reports"
	NEURON_RT_VISIBLE_CORES=$core TMPDIR=$BASE/tmp/w$core \
		timeout -k 30 "$to" "${CV[@]}" --examples "$EXAMPLES" --weights "$WEIGHTS" --outdir "$od" \
		--worker "$k" --num-workers "$nw" --device "$DEVICE" --dtype "$DTYPE" --batch-size "$BATCH" \
		--decode-procs "$DECODE_PROCS" --decode-cpus "$DECODE_CPUS" --main-cpus "$main_cpus" "$@" \
		> "$od/reports/worker-$k.log" 2>&1 9>&- &
	PIDS+=($!)
}

wait_pids() {  # wait on each PID explicitly; return non-zero if any failed
	local rc=0 i r
	for i in "${!PIDS[@]}"; do
		r=0; wait "${PIDS[$i]}" || r=$?
		if ((r)); then log "worker $i (pid ${PIDS[$i]}) exited $r$( ((r == 124)) && echo ' (timeout)')"; rc=1; fi
	done
	PIDS=()
	return $rc
}

MON_PID=
monitor_start() {  # neuron-monitor: device memory and NeuronCore utilization, 1 s period
	command -v neuron-monitor > /dev/null || { log "neuron-monitor not found; no device memory sampling"; return 0; }
	cat > "$OUT/neuron-monitor.json" << 'EOF'
{"period": "1s",
 "neuron_runtimes": [{"tag_filter": ".*", "metrics": [{"type": "neuroncore_counters"}, {"type": "memory_used"}]}],
 "system_metrics": [{"type": "vcpu_usage"}, {"type": "memory_info"}]}
EOF
	neuron-monitor -c "$OUT/neuron-monitor.json" > "$OUT/neuron-monitor.jsonl" 2> "$OUT/neuron-monitor.err" 9>&- &
	MON_PID=$!
}
monitor_stop() {
	[[ -n $MON_PID ]] || return 0
	kill "$MON_PID" 2> /dev/null || true
	wait "$MON_PID" 2> /dev/null || true
	MON_PID=
}

stage_warm() {
	# Compile once into the shared NEFF/HLO cache so the timed workers load it
	# (fp32 b32: ~200 s cold; bf16 b64: ~900 s cold, window 4).
	local od=$BASE/out/warm-$DTYPE-b$BATCH s
	lock_acquire
	s=$(stamp)
	PIDS=()
	launch_worker 0 1 "${WARM_CORE:-0}" "${WARM_CPUS:-0,6}" "$od" 2400 --shards 0 --limit 32 --force
	wait_pids || { lock_release; return 1; }
	lock_release
	log "warm: $(grep -h 'first batch' "$od/reports/worker-0.log") (lock held $(($(stamp) - s)) s)"
}

stage_cv() {
	local s to k
	lock_acquire
	s=$(stamp)
	monitor_start
	log "cv: $NW workers, $DTYPE batch $BATCH, limit $LIMIT, budget ${EXCL_BUDGET_S}s -> $OUT"
	to=$((EXCL_BUDGET_S - 60))
	PIDS=()
	local extra=(--check-example-info)
	((LIMIT > 0)) && extra+=(--limit "$LIMIT")
	[[ -n $SHARDS ]] && extra+=(--shards "$SHARDS")
	local mc
	for ((k = 0; k < NW; k++)); do
		mc="MAIN_CPUS_$k"
		launch_worker "$k" "$NW" "$k" "${!mc:-$k,$((k + 6))}" "$OUT" "$to" "${extra[@]}"
	done
	local rc=0
	wait_pids || rc=1
	monitor_stop
	lock_release
	log "cv: workers finished rc=$rc, lock held $(($(stamp) - s)) s"
	for ((k = 0; k < NW; k++)); do grep -h '\] done:' "$OUT/reports/worker-$k.log" | tee -a "$OUT/run.log" || true; done
	local nd want=$NSH
	[[ -n $SHARDS ]] && want=$(tr ',' '\n' <<< "$SHARDS" | grep -c .)
	nd=$(find "$OUT/done" -name '[0-9]*.json' | wc -l)
	log "cv: $nd/$want shards done"
	((rc == 0 && nd == want)) || { log "cv incomplete; rerun 'cv' to resume the remaining shards"; return 1; }
}

stage_cvocmp() {
	# Pre-postprocess: device CVOs vs stock TF call_variants on CPU (c8a), same examples.
	local sub=() partial=0
	((LIMIT > 0)) || [[ -n $SHARDS ]] && partial=1
	((partial)) && sub=(--subset)
	dvpy "$REALIGN_HACK/ie/cvo_cmp.py" "$T/cvo-*-of-00004.tfrecord.gz" "$OUT/cvo-neuron-*-of-000$NSH.tfrecord.gz" \
		"neuron-$DTYPE:cpu" "${sub[@]}" 2> "$OUT/cvo_cmp.err" | grep '^{' | tee "$OUT/cvo_cmp.json"
	dvpy "$HERE/cvo_fields_eq.py" "$T/cvo-*-of-00004.tfrecord.gz" "$OUT/cvo-neuron-*-of-000$NSH.tfrecord.gz" \
		2> "$OUT/cvo_fields_eq.err" | grep '^{' | tee "$OUT/cvo_fields_eq.json"
	if ((!partial)); then
		dvpy "$REALIGN_HACK/ie/cvo_set_eq.py" "$T/cvo-*-of-00004.tfrecord.gz" "$OUT/cvo-neuron-*-of-000$NSH.tfrecord.gz" \
			2> "$OUT/cvo_set_eq.err" | grep '^gate' | tee "$OUT/cvo_set_eq.txt"
	fi
}

stage_post() {
	local vcf=$OUT/$SAMPLE.neuron-$DTYPE.vcf.gz gvcf=$OUT/$SAMPLE.neuron-$DTYPE.g.vcf.gz s
	if [[ -s $vcf && -s $gvcf && -e $OUT/post.done ]]; then log "post: done already"; return 0; fi
	if ((LIMIT > 0)) || [[ -n $SHARDS ]]; then
		log "post: skipped for LIMIT=$LIMIT SHARDS=$SHARDS (partial CVOs; postprocess needs every shard)"
		return 0
	fi
	s=$(stamp)
	"${DOCKER[@]}" run --rm -u 0:0 -v "$BASE:$BASE" -w "$T" "$DV_IMAGE" /opt/deepvariant/bin/postprocess_variants \
		--ref "$R/chm13.fa" --infile "$OUT/cvo-neuron@$NSH.tfrecord.gz" --outfile "$vcf" --cpus "$(nproc)" \
		--small_model_cvo_records "$T/me_call_variant_outputs.tfrecord@$NSH.gz" \
		--nonvariant_site_tfrecord_path "$T/gvcf.tfrecord@$NSH.gz" \
		--gvcf_outfile "$gvcf" > "$OUT/postprocess.log" 2>&1
	touch "$OUT/post.done"
	log "post: wall_s=$(($(stamp) - s)) cpus=$(nproc)"
}

stage_cmp() {
	local vcf=$OUT/$SAMPLE.neuron-$DTYPE.vcf.gz gvcf=$OUT/$SAMPLE.neuron-$DTYPE.g.vcf.gz
	python3 "$REALIGN_HACK/ie/vcf_cmp.py" "$IN/$SAMPLE.vcf.gz" "$vcf" "neuron-$DTYPE-vs-cpu" | tee "$OUT/vcf-cmp.json"
	recs() { zcat "$1" | grep -v '^#' | sha256sum | cut -d' ' -f1; }
	local a b
	a=$(recs "$IN/$SAMPLE.vcf.gz"); b=$(recs "$vcf")
	if [[ $a == "$b" ]]; then echo "VCF records IDENTICAL to CPU ($a)"; else echo "VCF records DIFFER from CPU"; fi | tee "$OUT/gate.txt"
	a=$(recs "$IN/$SAMPLE.g.vcf.gz"); b=$(recs "$gvcf")
	if [[ $a == "$b" ]]; then echo "gVCF records IDENTICAL to CPU ($a): tier 0" | tee -a "$OUT/gate.txt"
	else
		echo "gVCF records DIFFER from CPU: $(diff <(zcat "$IN/$SAMPLE.g.vcf.gz" | grep -v '^#') <(zcat "$gvcf" | grep -v '^#') | grep -c '^[<>]') lines" \
			| tee -a "$OUT/gate.txt"
	fi
}

stage_summary() {
	"${CV[@]}" --summarize "$OUT" | tee "$OUT/reports/summary.txt"
	[[ -s $OUT/neuron-monitor.jsonl ]] || return 0
	python3 - "$OUT/neuron-monitor.jsonl" << 'EOF' | tee "$OUT/reports/neuron-monitor-summary.json"
import json, sys
peak_dev = 0; util = {}; n = 0
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    n += 1
    dev = 0
    for rt in d.get("neuron_runtime_data") or []:
        rep = rt.get("report", {})
        mu = rep.get("memory_used", {}).get("neuron_runtime_used_bytes", {})
        dev += int(mu.get("neuron_device", 0) or 0)
        for core, c in (rep.get("neuroncore_counters", {}).get("neuroncores_in_use") or {}).items():
            util.setdefault(core, []).append(float(c.get("neuroncore_utilization", 0) or 0))
    peak_dev = max(peak_dev, dev)
busy = {k: v for k, v in util.items() if any(x > 1 for x in v)}
print(json.dumps({
    "samples": n,
    "peak_device_memory_gib": round(peak_dev / 2**30, 3),
    "neuroncore_util_mean_pct_while_active": {k: round(sum(x for x in v if x > 1) / max(1, sum(1 for x in v if x > 1)), 1) for k, v in sorted(busy.items())},
}))
EOF
}

STAGES=("$@")
((${#STAGES[@]})) || STAGES=(fetch tools check warm cv cvocmp post cmp summary)
for st in "${STAGES[@]}"; do
	declare -F "stage_$st" > /dev/null || { echo "unknown stage $st" >&2; exit 2; }
	log "== stage $st (DTYPE=$DTYPE BATCH=$BATCH LIMIT=$LIMIT OUT=$OUT)"
	"stage_$st"
done
