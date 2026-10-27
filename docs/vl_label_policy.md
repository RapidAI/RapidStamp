# VL-1.6 labeling policy

Measured and merged 2026-10-27. The model is PaddleOCR-VL-1.6 only (`pipeline_version="v1.6"`, prompt `Seal Recognition:`, layout detection off). Version 1.5 was not used. The ring is the first line. Spaces inside that line count.

## Rule

Good-quality seal crops keep the VL first line when it looks like a place or organization name. A line passes when all of these hold:

- length 2–25, no internal space, no template placeholder
- at least four CJK characters and no weird characters
- it ends with an organization or place suffix such as 公司, 局, 委员会, 大学, 银行, 医院, 人民政府, 中心, 院, 所
- at least two CJK characters sit in front of that suffix
- it is not a broken ending (`有限公`, `股份限公司` without 有限)

Poor quality, or a first line that fails the check, falls back to Aliyun stamp OCR. Poor quality means min side under 64, grayscale standard deviation under 18, Laplacian variance under 80, or ink span under 0.55. Crops with no ink, a solid blob, heavy non-seal color, a bad aspect, or an inner box are skipped. They are not sent to Aliyun.

Existing labels win on perceptual hash (Hamming distance 6 or less). Aliyun rows already in `labels.jsonl` were not rewritten. HTTP 464 is not a label. Each new row stores `label_source` (`vl` or `aliyun_fallback`) and `validation_reason`. The audit is `/home/rapidstamp_data/third/vl_policy/decisions.jsonl`.

## What was labeled

Batch6 crops that the old 0.75 filter left unlabeled, plus the 150 detector rejects already read by VL. Older batch rejects were not labeled. Batch6's 1,037 filtered crops were already Aliyun-labeled and were left alone.

| Item | Count |
| --- | --- |
| Candidates considered | 2,563 |
| Skipped before a model call | 1,121 |
| VL first line accepted | 636 (612 batch6, 24 earlier rejects) |
| Aliyun fallback accepted into labels | 540 (500 batch6, 40 earlier rejects) |
| Aliyun status 200 / 464 | 541 / 265 |
| labels.jsonl | 25,139 → 26,315 |
| kept.jsonl | 20,211 → 21,214 |
| Kept train / val added | 966 / 37 |
| Kept from VL | 612 (train 589, val 23) |
| Kept from Aliyun fallback | 391 (train 377, val 14) |

Key2 went from 5,286 to 6,092 (806 calls). Key1 stayed at 19,483. Kept drops: length_short 91, template 41, val_text 28, length_long 11, gold_text 1, charset 1. One Aliyun 200 body was empty and was not merged.

Backups: `labels.jsonl.bak-before-vlpolicy-20261027`, `kept.jsonl.bak-before-vlpolicy-20261027`. Do not run this merge again.

## Fine-tune

Three epochs from `checkpoints/parseq_reader_mix2/finetune/best.pt` into `checkpoints/parseq_reader_vl1/`. Real probability 0.55. Names `synth_names/names_for_vl1.txt` (19,999; none matched a new val or gold string). Step target `int(3 * 20549 / 0.55)` = 112,085. The run stopped at seen 112,096. Startup: train 20,549, val 665, gold 63.

Selected weights are `checkpoints/parseq_reader_vl1/finetune/best.pt` at step 100,000: val 313/665, 1−NED 0.84470; gold 25/63, 1−NED 0.91142 (verified 25/57, 0.92416; pending 0/6, 0.79042). `last.pt` at step 112,096 was not selected (val 309/665, gold 1−NED 0.91036). The ONNX is 126,972,402 bytes, reloaded from `best.pt`. Gold exact stayed 25/63. Gold 1−NED 0.91142 is above mix2's 0.90795. Quote `best.pt`.
