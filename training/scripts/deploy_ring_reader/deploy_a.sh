#!/bin/bash
# Deploy A + length routing (2026-11-01). Idempotent-ish; never deletes.
set -euo pipefail
D=/home/rapidstamp_data
PY=/home/znsoft/miniconda3/envs/pytorch/bin/python
export PYTHONPATH=$D/src/RapidStamp/python/src:$D/src/RapidStamp/training
B=$D/backup/prod_before_A_20261101
R=$D/deploy/ring_reader
REL=$R/releases/20261101_A_route
OLD=$R/releases/20261101_prod_S512
mkdir -p $B/checkpoint $REL $OLD
# 1) backup current production (copy, never move)
cp -n $D/checkpoints/parseq_reader_large/finetune/{best.pt,parseq_reader.onnx,eval.json} $B/checkpoint/
cp -n $D/synth_names/names_for_vl1.txt $D/gold_rescreen/eval_gold_v2_cli.py $B/ 2>/dev/null || true
cp -n $D/pretrained/ppocr_keys_v1.txt $B/
(cd $B && sha256sum checkpoint/* > SHA256SUMS)
# 2) export logprob ONNX
cd $D/src/RapidStamp/training/scripts
[ -f $REL/reader_main.onnx ] || $PY export_ring_reader.py --ckpt $D/checkpoints/parseq_reader_large_clean_csd/finetune/best.pt --out $REL/reader_main.onnx
[ -f $REL/reader_long.onnx ] || $PY export_ring_reader.py --ckpt $D/checkpoints/parseq_reader_large/finetune/best.pt --out $REL/reader_long.onnx
# 3) charset + names + route.json
$PY - <<'PY'
import json
from pathlib import Path
from rapidstamp_train.direct_aug import load_charset
D=Path("/home/rapidstamp_data"); REL=D/"deploy/ring_reader/releases/20261101_A_route"; OLD=D/"deploy/ring_reader/releases/20261101_prod_S512"
cs=load_charset(D/"pretrained/ppocr_keys_v1.txt")
txt="\n".join(cs.chars)+"\n"
rows=[json.loads(l) for l in open(D/"gold_v2/labels.jsonl",encoding="utf-8")]
blocked={t.split("#")[0].strip() for r in rows for t in r["texts"]}
names=[t.strip() for t in open(D/"synth_names/names_for_vl1.txt",encoding="utf-8") if t.strip() and t.strip() not in blocked]
for d in (REL,OLD):
    (d/"charset.txt").write_text(txt,encoding="utf-8")
    (d/"names_lm.txt").write_text("\n".join(names)+"\n",encoding="utf-8")
base={"charset":"charset.txt","names":"names_lm.txt","alpha":0.4,"tau":4.0,"max_flips":3,"image_size":512}
(REL/"route.json").write_text(json.dumps({"main":"reader_main.onnx","long":"reader_long.onnx","long_threshold":16,**base,
  "note":"main=A parseq_reader_large_clean_csd best.pt; long=production parseq_reader_large best.pt (step 325024); use long when main lex text >= long_threshold chars"},ensure_ascii=False,indent=1),encoding="utf-8")
(OLD/"route.json").write_text(json.dumps({"main":"reader_main.onnx","long_threshold":0,**base,
  "note":"rollback: production S@512 parseq_reader_large best.pt alone, lex alpha0.4 tau4"},ensure_ascii=False,indent=1),encoding="utf-8")
# CSD eval lexicon (blocks gold + CSD test names, as csd_eval.py)
norm=lambda t:"".join((t or "").split())
gn={norm(t.split("#")[0]) for r in rows for t in r["texts"]}
csd={json.loads(l)["text"] for l in open(D/"external/csd/preds_csd_cand_20261031.jsonl",encoding="utf-8")}
bl=gn|csd
(D/"deploy/ring_reader/eval").mkdir(parents=True,exist_ok=True)
(D/"deploy/ring_reader/eval/names_lm_csd_eval.txt").write_text("\n".join(t.strip() for t in open(D/"synth_names/names_for_vl1.txt",encoding="utf-8") if t.strip() and norm(t.strip()) not in bl)+"\n",encoding="utf-8")
print("names",len(names),"chars",len(cs.chars))
PY
[ -f $OLD/reader_main.onnx ] || ln $REL/reader_long.onnx $OLD/reader_main.onnx
ln -sfn releases/20261101_A_route $R/current
ls -la $REL $OLD $R
