#!/usr/bin/env bash
# Train the direct seal reader. Does not stop other GPU jobs.
set -euo pipefail
DATA=/home/rapidstamp_data
REPO=${REPO:-$DATA/src/RapidStamp}
PY=${PY:-/home/znsoft/miniconda3/envs/pytorch/bin/python}
OUT=$DATA/checkpoints/seal_reader
LOG=$DATA/logs/seal_reader.train.log
mkdir -p "$OUT" "$DATA/logs"
export PYTHONPATH="$REPO/python/src:$REPO/training${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
nohup "$PY" training/scripts/train_direct_reader.py \
  --kept "$DATA/ring_v2/kept.jsonl" \
  --crops "$DATA/web_seals_cn_all/crops" \
  --dict "$DATA/pretrained/ppocr_keys_v1.txt" \
  --font "$DATA/pretrained/fonts/NotoSansCJKsc-Regular.otf" \
  --out "$OUT" \
  --epochs 40 \
  --batch 16 \
  > "$LOG" 2>&1 &
echo $! | tee "$DATA/logs/seal_reader.pid"
echo "pid $(cat "$DATA/logs/seal_reader.pid") log $LOG"
