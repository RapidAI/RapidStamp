#!/usr/bin/env bash
# Zero-shot eval, then a nohup fine-tune of PP-OCRv4 on ring strips.
# Reads and writes only under /home/rapidstamp_data. Does not stop other GPU jobs.
set -euo pipefail

DATA=/home/rapidstamp_data
REPO=${REPO:-$DATA/src/RapidStamp}
PADDLE=${PADDLE:-$DATA/src/PaddleOCR}
VENV=${VENV:-$DATA/venvs/paddleocr}
CFG=${CFG:-$REPO/training/configs/ch_PP-OCRv4_rec_seal.yml}
LOG=$DATA/logs/ring_baseline.train.log
ZERO=$DATA/logs/ring_baseline.zeroshot.log

mkdir -p "$DATA/logs" "$DATA/checkpoints/ring_baseline"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$REPO/python/src:$REPO/training${PYTHONPATH:+:$PYTHONPATH}"
export FLAGS_allocator_strategy=auto_growth
export FLAGS_initial_gpu_memory_in_mb=256
export FLAGS_reallocate_gpu_memory_in_mb=128
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Paddle 2.6 (CUDA 11.8) needs cuDNN 8. The PyTorch env on this host only has cuDNN 9.
CUDNN8=${CUDNN8:-/home/znsoft/miniconda3/envs/lxswarp/lib/python3.10/site-packages/torch/lib}
mkdir -p "$VENV/cudnn"
if [ -f "$CUDNN8/libcudnn.so.8" ] && [ ! -e "$VENV/cudnn/libcudnn.so" ]; then
  ln -s "$CUDNN8/libcudnn.so.8" "$VENV/cudnn/libcudnn.so"
fi
export LD_LIBRARY_PATH="$VENV/cudnn:$CUDNN8${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

INIT="$PADDLE/ppocr/data/imaug/__init__.py"
if ! grep -q "SealStripAug" "$INIT"; then
  python - "$INIT" << 'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = "from .rec_img_aug import ("
insert = "from rapidstamp_train.seal_aug import SealStripAug\n"
if "SealStripAug" not in text:
    text = text.replace(needle, insert + needle, 1)
    path.write_text(text, encoding="utf-8")
PY
fi

cd "$PADDLE"
python - << 'PY'
import rapidstamp_train.seal_aug  # noqa: F401
from ppocr.data.imaug import SealStripAug, create_operators
ops = create_operators([{"SealStripAug": {"jpeg_prob": 0.0, "roll_prob": 0.0}}], None)
assert ops and isinstance(ops[0], SealStripAug)
print("seal aug registered", flush=True)
PY

echo "zero-shot eval -> $ZERO"
python tools/eval.py -c "$CFG" \
  -o Global.pretrained_model="$DATA/pretrained/ch_PP-OCRv4_rec_train/student" \
     Global.checkpoints= \
  > "$ZERO" 2>&1
tail -n 40 "$ZERO"

echo "train -> $LOG"
nohup python tools/train.py -c "$CFG" > "$LOG" 2>&1 &
echo $! | tee "$DATA/logs/ring_baseline.pid"
echo "pid $(cat "$DATA/logs/ring_baseline.pid")"
