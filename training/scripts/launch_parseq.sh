#!/bin/bash
# Pretrain, then fine-tune, beside the resident VLLM process. Do not kill it.
set -euo pipefail
PY="${PY:-/home/znsoft/miniconda3/envs/pytorch/bin/python}"
ROOT="${ROOT:-/home/rapidstamp_data/src/RapidStamp}"
DATA="${DATA:-/home/rapidstamp_data}"
LOG="$DATA/logs/parseq_reader.log"
PID="$DATA/logs/parseq_reader.pid"
mkdir -p "$DATA/logs" "$DATA/checkpoints/parseq_reader"
cd "$ROOT"
export PYTHONPATH="$ROOT/python/src:$ROOT/training${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
nohup "$PY" -u training/scripts/train_parseq.py \
  --names "$DATA/synth/names.txt" \
  --blocked "$DATA/synth/blocked.txt" \
  --kept "$DATA/ring_v2/kept.jsonl" \
  --crops "$DATA/web_seals_cn_all/crops" \
  --dict "$DATA/pretrained/ppocr_keys_v1.txt" \
  --fonts "$DATA/pretrained/fonts" \
  --manifest "$ROOT/training/manifests/user_seals_split.jsonl" \
  --gold "$DATA/gold/seals_to_label" \
  --out "$DATA/checkpoints/parseq_reader" \
  --samples "${SAMPLES:-2000000}" \
  --batch "${BATCH:-32}" \
  --workers "${WORKERS:-6}" \
  --backbone "${BACKBONE:-vit_small_patch16_384}" \
  --no-checkpointing \
  --memory-fraction "${MEMORY_FRACTION:-0.105}" \
  > "$LOG" 2>&1 &
echo $! | tee "$PID"
echo "log $LOG"
