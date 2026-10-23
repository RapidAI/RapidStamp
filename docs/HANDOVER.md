# RapidStamp ring-text handover

Updated 2026-10-23T07:19:18Z. This file is the operational handover. `docs/data.md` and `docs/research.md` are the Chinese measured notes; `docs/sdk.md` is the published detection plus class-routed CTC / PP-OCRv5 SDK. Those three are not this recognizer.

Update this file after every meaningful step, measured result, or change to data or models. Commit it on `main`, then copy `docs/*.md` to `/home/rapidstamp_data/docs/` on the training machine. Do not put passwords, the SSH password, or Aliyun APPCODEs in this file or in git.

## Goal and deliverables

Read the curved ring text of a Chinese seal from the whole crop, without unwrapping. The bar is the Aliyun stamp-OCR draft on the 63 user-proofread gold crops: **55/63 exact, 1−NED 0.979**. The published class-routed pipeline scores **0/63, 1−NED 0.050** on the same crops. The new model has to stay ONNX-exportable for a later C++ path. It is a separate recognizer from `models/seal_rec.onnx` and from `seal_rec_ppocr.onnx`.

Still open: finish synthetic pretraining, fine-tune on the filtered real crops, and report exact match and 1−NED on val and on the 63 gold crops next to the Aliyun numbers. Do not train a second head for center text or the bottom code until ring text is near that bar.

Headline metric, from `training/rapidstamp_train/ring_text.py` `one_ned`: mean over samples of `1 − edit_distance / max(len(pred), len(gold))`. Both strings empty scores 1. Spaces count. Do not strip. This is not Paddle `RecMetric` (`ignore_space=True`, and it keeps only CJK and alphanumerics).

## Access and hard rules

- Training host: `home.rapidai.tech`, SSH port 44, user `znsoft`. The password is held by the user and is already in the operator environment as `SSHPASS`. Use `sshpass -e`. Do not write it down. Connections reset; retry.
- Read and write only `/home/rapidstamp_data`. `/home` is `/dev/sdb`, about 1.8T, 699G used, 1.1T free, 41% at the last check.
- `/home/mayong` is forbidden. `ls` shows a root-owned directory dated 2025-05-19. `df` of that path reports the same `/dev/sdb` as `/home`, so it is not a separate live data disk. Older notes in `docs/data.md` tell you to put large files there. Those notes are historical. Do not read it, write it, or try to recover it.
- GPU: RTX 6000 Ada, 49140 MiB. Resident VLLM is PID 17019 (`VLLM::EngineCore`, about 42880 MiB, up since before this work). Do not kill it. Do not start a second trainer. The live reader is PID 52706 (see the pretraining section). Check `ps -p <pid>`. A `pgrep` that matches an `ssh bash -c` line does not mean the trainer is alive.
- The server tree `/home/rapidstamp_data/src/RapidStamp` is a partial copy, not a git checkout. Sync code with tar over SSH. Do not `git pull` there.
- Do not commit the SSH password, an Aliyun APPCODE, real seal images, gold crops, name lists, fonts, or checkpoints.
- Do not delete `incoming_v2`, the gold crops, or the ring-baseline checkpoint.

## Server layout

Inventory taken 2026-10-23T05:37:44Z. Sizes are `du -sh`. File counts are recursive.

| Path | Size | What is there |
| --- | --- | --- |
| `/home/rapidstamp_data/backup` | 651M plus the incoming full backup | `gold_db18.zip` (16,362,139 bytes), `web_seals_cn_all.zip` (385,220,370), `web_seals_delta_v2.zip` (263,497,928), and `gold_seals/seals_to_label/` (63 png + `labels.csv`, 64 files). `full_v1/` is a second backup that started landing on 2026-10-23; see below. The 651M figure is from before those parts. |
| `/home/rapidstamp_data/checkpoints` | 6.7G | `parseq_reader/`, `ring_baseline/` (75 files), `ring_v2/` (partial, killed), `seal_reader/`. |
| `/home/rapidstamp_data/gold` | 20M | `seals_to_label/` (63 png + `labels.csv`), `baseline_strips/` (63 strips), `baseline_report/gold_report.json`. |
| `/home/rapidstamp_data/incoming_v2` | 503M | `delta.zip` (263,497,928) and `part_00.zip` … `part_11.zip` (parts 00–10 are 23,068,672 each, part 11 is 9,742,536). Keep them. |
| `/home/rapidstamp_data/logs` | 1.1M | 32 log and pid files. Live log is `parseq_reader.log`, pid file `parseq_reader.pid`. |
| `/home/rapidstamp_data/pretrained` | 264M | `ppocr_keys_v1.txt` (6,623 lines, 26,250 bytes), `ch_PP-OCRv4_rec_train/` plus its tar, `ch_PP-OCRv4_rec_infer/` plus its tar, `fonts/`. |
| `/home/rapidstamp_data/ring_baseline` | 496M | `strips/train` 10,907, `strips/val` 348, `kept.jsonl` 11,255 rows, `train_list.txt`, `val_list.txt`, `meta.json`, `sample_sheet.jpg`. |
| `/home/rapidstamp_data/ring_v2` | 589M | `strips/train` 10,839, `strips/val` 344, `kept.jsonl` 11,183 rows, lists, `meta.json`, `sample_sheet.jpg`. |
| `/home/rapidstamp_data/src` | 417M | `PaddleOCR/` and `RapidStamp/` (partial runtime tree: `python/` and `training/` only). |
| `/home/rapidstamp_data/synth` | 267M | Name corpus, uniform strings, blocked strings, `stats.json`, `sample_sheet.jpg`, and `src/` (public name lists). |
| `/home/rapidstamp_data/venvs` | 3.7G | `paddleocr` only (Paddle 2.6.2, cu118). |
| `/home/rapidstamp_data/web_seals_cn_all` | 1016M | `crops/` 15,738 png, `labels.jsonl` (11,440,722 bytes), `labels_v1.jsonl` (6,615,523 bytes, 9,085 rows), `web_seals_cn_all.zip` (385,220,370), `SHA256SUMS.txt`. |

There is no `docs/` on the server until this handover is mirrored. PyTorch is not in the data-disk venv. The reader uses `/home/znsoft/miniconda3/envs/pytorch/bin/python` (3.11.11, torch 2.8.0+cu129, timm 1.0.30). Do not pip-install Paddle into that env.

Fonts under `pretrained/fonts/`: `NotoSansCJKsc-Regular.otf` (Hei, 16,437,364), `NotoSerifCJKsc-Regular.otf` (Song, 24,543,080), `LXGWWenKaiGBLite-Regular.ttf` (Kai, 14,191,220), `zhuque/ZhuqueFangsong-Regular.ttf` (仿宋, 8,824,084) and `zhuque.zip`.

`web_seals_cn_all/SHA256SUMS.txt` records `347286fe8a4fab9f9c8f3c568d9c31efc07376149def009bafbdace1948f7789` for the zip, with a `/workspace/...` path from the machine that packed it.

### Full box backup, in progress

A second copy of the collection box is arriving as five plain tar archives, split into raw parts whose names end in `.zip` only so the attachment filter accepts them. Do not unzip a part. Manifest at `/home/rapidstamp_data/backup/full_v1/MANIFEST.txt` (2026-10-23 13:33:20 UTC+8). Checksums at `full_v1/SHA256SUMS.txt`. Parts land in `full_v1/parts/`. When the last part of an archive is in, `cat` them in order to `full_v1/<archive>.tar`, check the tar sha256, and extract into `/home/rapidstamp_data/raw_box_backup/`. APPCODEs in the source were already replaced with the placeholder `${ALIYUN_APPCODE}`.

| Archive | Parts | Bytes | Files | Received |
| --- | --- | --- | --- | --- |
| `a1_meta.tar` | 20 | 442,204,160 | 1,124 | complete. Tar sha256 `f2556116d5aa80cdd48af47887a72a40274fff0d398de67620c711b0da25f0dc`. Extracted. |
| `a2_cn_crops.tar` | 60 | 1,361,745,920 | 26,265 | complete. Tar sha256 `b708e5980449559949c136bc1d60127b00d6bbcbb66978b1fcd1218a191fd379`. Extracted into the existing `web_seals_v2_raw/` tree with no file collisions. |
| `a3_foreign_crops.tar` | 26 | 591,646,720 | 4,787 | 000–021, sha256 checked on the server. |
| `a4_cn_rejected.tar` | 13 | 277,166,080 | 4,350 | none |
| `a5_foreign_rejected.tar` | 36 | 813,629,440 | 3,637 | none |

`a1_meta` is labels, manifests, urls, logs, scripts, the first Aliyun scrape (998 images), misc csvs, `gold.zip`, `seals_to_label.zip`, `seal_det.onnx`, and combine scripts. `a2` is CN crops not already in `web_seals_cn_all` (b2 3,899, b3 4,034, b4 8,025, b5 6,864, b6 3,443). `a3` is 4,787 foreign crops. `a4` is 4,350 rejected CN batch-1 crops. `a5` is 3,637 rejected foreign crops. `a1_meta.tar` is `/home/rapidstamp_data/backup/full_v1/a1_meta.tar` (442,204,160 bytes). `a2_cn_crops.tar` is `/home/rapidstamp_data/backup/full_v1/a2_cn_crops.tar` (1,361,745,920 bytes, 26,271 tar members, 26,265 files). Both are extracted under `/home/rapidstamp_data/raw_box_backup/` (27,389 files and 18 directories after both archives). a1 top level: `misc/` (13 files, including `gold.zip` and `seals_to_label.zip`, each 16,362,139 bytes, plus `aliyun_calls.txt` and `aliyun_calls_key2.txt`, which were not printed), `models/seal_det.onnx` (3,671,258 bytes), `web_seals_aliyun_firstscrape/` (998 files under `raw/`, plus `downloaded.json`, `manifest.jsonl`, `urls.json`), `web_seals_cn_all_scripts/` (3 files), `web_seals_v2_foreign/` (14 files), and the scripts, manifests, and urls already in `web_seals_v2_raw/`. a2 added crops into that same `web_seals_v2_raw/` tree and did not overwrite an existing file. Batch directories now hold `batch2` 3,915, `batch3` 4,047, `batch4` 8,038, `batch5` 6,881, and `batch6` 3,460 files; the extra files beyond the manifest snapshot (3,899 / 4,034 / 8,025 / 6,864 / 3,443) are the metadata a1 already had in those directories. The trainer was left running through both copies and extracts.

## Datasets

### `web_seals_cn_all`

15,738 Aliyun-labeled crops. `labels.jsonl` keys: `batch`, `det_class`, `det_conf`, `file`, `other_texts`, `phash`, `probs` (`ring`, `other`), `query`, `raw`, `ring_text`, `roi` (`height`, `left`, `top`, `width`), `source_url`, `status`. Status 200: 14,667. Status 464: 1,071. `labels_v1.jsonl` has the same keys and is an earlier 9,085-row slice. The Aliyun payload has an ROI and no angle.

`ring_text` / `other_texts` / `probs` come from the Aliyun stamp OCR API. `det_class` and `det_conf` come from the YOLOX-Nano crop. `phash`, `query`, and `source_url` come from the scraper. These strings are drafts, not user proofreading.

### Filtered sets

Filter in `training/rapidstamp_train/ring_text.py`: status 200, non-empty ring text, ring probability ≥ 0.90, not a template, class in `round_official` / `oval` / `invoice` / `finance` / `notary` / `company_latin`, length 2–25, every character in `ppocr_keys_v1` or a space. Val is `sha256(basename)[:8]` as int, mod 1000, less than 30. Gold crops are excluded. Bare `测试` is not banned. The template regex covers 某某, two or more of `X`/`x`/`×`, 演示章, 测试专用, 样章, sample, 示例章, 双击, 替换代用, 可以替换, 代用名, 公司名称, 此处可输, 请在这里, 在这里输入, 输入您的, 输入单位, 编辑文字, 测试样品, 测试效果, 免费设计测试.

Baseline filter (`ring_baseline/meta.json`): label-keep 11,262, then 7 out-of-dict rows dropped, dedupe removed nothing, train 10,907 / val 348. Drop reasons: empty 1,413, status 1,071, template 970, prob 878, class 77, length 67. Median gap 0°, so most strips start at 140°. Wide gap (≥ 40°) on 631 rows.

Extended filter (`ring_v2/meta.json`): label-keep 11,190, template 1,042, before-dedupe 11,183, deduped 0, train 10,839 / val 344. Other reasons match the baseline (empty 1,413, status 1,071, prob 878, class 77, length 67, dict 7). This `kept.jsonl` is what the direct reader and the current pretraining use. Its keys add unwrap geometry: `angle` (always null; Aliyun has no angle), `confidence`, `file_md5`, `gap_deg`, `prob`, `r_in`, `r_out`, `roi`, `source`, `span_deg`, `split`, `start_deg`, `strip`, `strip_h`, `strip_w`, `text`, `unwrap_start`, `wide_gap`. Sources: gap 10,776, default 181, center 86, star 115, code 25. Median span 300°, median gap 80°, median strip width 960.

Baseline `kept.jsonl` keys are the shorter set: `det_class`, `file`, `file_md5`, `gap_deg`, `prob`, `r_in`, `r_out`, `split`, `start_deg`, `strip`, `text`, `wide_gap`.

### Gold

`training/manifests/user_seals_split.jsonl` is the proofread text. 229 rows: train 156, test 63, duplicate 10. `text_status`: unverified 166 (the train and duplicate rows), `user_verified` 57, `user_verified_pending_recheck` 6. The six pending files are the crops numbered 014, 027, 031, 039, 040, 041 (`234327ee…_0`, `5a789c66…_1`, `6b186b3a…_1`, `8b4459b4…_1`, `8bb68f5c…_0`, `8bfe6df4…_0`). Score pending separately. Train rows have no transcription. Do not change the 156 train rows or the 10 duplicate rows.

A test seal has `bbox`, `center`, `cls`, `code`, `cols`, `layout`, `name`, `ring`, `rows`, `text`. `ring` and `text` are the ring string, `center` is the center text, `code` is the bottom code. `labels.csv` in the image folder has columns 编号, 文件名, 环形主文, 中心文字, 底部编码, and the last three are empty. Do not score from that CSV.

Images on the server: `/home/rapidstamp_data/gold/seals_to_label/NNN_<manifest-basename>.png` (63 png). Backup zip and an unpacked copy are under `backup/`. `match_gold_image` returns a path only when exactly one filename equals the manifest basename or ends with it. Pass `--gold` as `.../gold/seals_to_label`. Passing `.../gold` matches both `seals_to_label` and `baseline_strips` and reports 63 missing.

### Synthetic corpus

Built by `training/scripts/build_name_corpus.py`. Not in git.

| File | Lines | Role |
| --- | --- | --- |
| `synth/names.txt` | 1,535,269 | Unique training names. Leak check against blocked strings was 0. |
| `synth/uniform.txt` | 39,738 | Six strings per charset character, so rare characters appear. |
| `synth/blocked.txt` | 433 | Val ring text plus gold ring, center, and code. Never used as a training name. |
| `synth/stats.json` | — | Counts below. |
| `synth/sample_sheet.jpg` | 264,660 bytes, 1120×1264 | Eight synthetic tiles (first four upright) then real gold crops. |

`stats.json`: places 6,213 (from `modood` `pca-code.json`). Company reservoir 1,000,000 kept from 4,803,499 accepted (wainshine Company-Names-Corpus, Apache-2.0; file `synth/src/company/`, about 4.81M lines, header skipped). Org reservoir 250,000 from 1,023,249 (`synth/src/org/`). Brands 80,000 from 267,281 (`shorter.txt`). Generated companies 120,000. Generated agencies 80,000. Train-split ring strings 8,728 (deduped; the CJK-ratio rule does not apply to these). Charset characters 6,624. Header lines such as `By@萌名NameMoe` are skipped. Out-of-vocabulary and blocked overlaps were dropped (company charset drops 4,600, blocked 44; org charset 1,083, blocked 54; brands charset 115, blocked 1).

Sources of names: public company, organization, and short-brand lists, generated `place + brand + industry + suffix` and `place + agency`, and ring text from the **train** split only. Val and gold strings are blocked.

## Data collection pipeline

The pipeline that produced `web_seals_cn_all` runs on a **separate machine**. It is not in this git repo, and this training host only has the labeled output.

For each query it scrapes Bing, Baidu, and 360 image search, crops seals with the published YOLOX-Nano detector, filters, dedupes by perceptual hash, and labels the crop with the Aliyun stamp OCR API. The APPCODE is read from the environment variable `ALIYUN_APPCODE`. Two keys exist. They are held by the user and the bot. This repo does not store the codes. A locked call counter and a cap are enforced on the collector box. The numeric cap is not stored in this repo.

`training/scripts/collect_web_seals.py` is an older, smaller scraper. On 2026-10-16 it kept 401 images (360: 350, Bing: 51, Baidu: 0). Its docstring still points at `/home/mayong`. It is not the 15,738-crop Aliyun pipeline.

## Experiments

| Run | Val exact / 1−NED | Gold exact / 1−NED | Notes |
| --- | --- | --- | --- |
| Aliyun ring draft | — | 55/63, 0.979 | The bar. Same 63 proofread strings. |
| Published class-routed CTC | — | 0/63, 0.050 (verified 0/57, 0.053) | 2026-10-17. Every crop detected as `round_official`, so all 63 took CTC. Mean IoU 0.959. About 199 ms/crop. No training in that pass. |
| Unwrap baseline, zero-shot PP-OCRv4 | 8/348, 0.138 | — | Empty predictions 203. Scored on the old strips. |
| Unwrap baseline, 20 epochs | 16/348, 0.224 | 0/63, 0.24173 (verified 0/57, 0.23913; pending 0/6, 0.26641) | Paddle RecMetric best acc 0.083, norm_edit_dis 0.405 at epoch 20. That acc drops spaces. Empty predictions 44. Gold strips were the old full-circle unwrap, 48×640, resized to width 320. `gold_report.json` has top-level `images_found` 0 because `--images` was not passed; the inner `missing` is 0, so all 63 joined. |
| Arc-only unwrap (v2) | not finished | not run | Data prep finished. Training was killed around epoch 6 of 60. Checkpoints under `checkpoints/ring_v2` are partial. Do not resume and do not quote them as a result. |
| ViT-Tiny direct reader, epoch 33 of 40 | 9/344, 0.31392 | 0/63, 0.18681 (verified 0/57, 0.19851; pending 0/6, 0.07564) | Best by val exact. Predictions are mostly generic shells. One val near-miss: 吉南省人力资源和社会保障厅 vs 吉林省人力资源和社会保障行. |
| PARSeq ViT-S pretrain, in progress | see below | see below | Not a result yet. |

Pretraining snapshot, job still running, log tail at 2026-10-23T07:19:18Z (PID 52706): seen 749,024, loss 4.558, 110.4 img/s. Eval so far (1−NED). Latest val exact is 75/344 at 725,024, up from 68/344 at 675,008, so this row wins `(exact, 1−NED)`. Gold exact is 1/63, the third time (also at 625,024 and 700,000).

| Seen | Val 1−NED (n=344) | Gold 1−NED (n=63) | Verified (57) | Pending (6) |
| --- | --- | --- | --- | --- |
| 25,024 | 0.05468 | 0.03358 | 0.03711 | 0.00000 |
| 50,016 | 0.11895 | 0.14898 | 0.15744 | 0.06859 |
| 75,008 | 0.14275 | 0.14012 | 0.14940 | 0.05193 |
| 100,000 | 0.14770 | 0.10530 | 0.11396 | 0.02302 |
| 125,024 | 0.15642 | 0.14399 | 0.15001 | 0.08683 |
| 150,016 | 0.17131 | 0.15716 | 0.17058 | 0.02961 |
| 175,008 | 0.22195 | 0.18961 | 0.20008 | 0.09012 |
| 200,000 | 0.29209 | 0.22540 | 0.23453 | 0.13873 |
| 225,024 | 0.31187 | 0.22866 | 0.24159 | 0.10583 |
| 250,016 | 0.36724 | 0.25206 | 0.27077 | 0.07423 |
| 275,008 | 0.37067 | 0.26332 | 0.27761 | 0.12756 |
| 300,000 | 0.42490 | 0.27651 | 0.28944 | 0.15370 |
| 325,024 | 0.47119 | 0.28823 | 0.29877 | 0.18808 |
| 350,016 | 0.50338 | 0.35322 | 0.35556 | 0.33100 |
| 375,008 | 0.52798 | 0.35030 | 0.35463 | 0.30922 |
| 400,000 | 0.54466 | 0.40570 | 0.41377 | 0.32902 |
| 425,024 | 0.56698 | 0.44048 | 0.44968 | 0.35307 |
| 450,016 | 0.59657 | 0.46049 | 0.46429 | 0.42436 |
| 475,008 | 0.60992 | 0.44776 | 0.44761 | 0.44920 |
| 500,000 | 0.62499 | 0.50167 | 0.50924 | 0.42976 |
| 525,024 | 0.61392 | 0.38780 | 0.39359 | 0.33282 |
| 550,016 | 0.64059 | 0.49132 | 0.50129 | 0.39656 |
| 575,008 | 0.65926 | 0.50657 | 0.50583 | 0.51367 |
| 600,000 | 0.64814 | 0.50251 | 0.50786 | 0.45174 |
| 625,024 | 0.68069 | 0.55406 | 0.56736 | 0.42778 |
| 650,016 | 0.69367 | 0.55997 | 0.57290 | 0.43710 |
| 675,008 | 0.71306 | 0.59534 | 0.60888 | 0.46666 |
| 700,000 | 0.72041 | 0.58637 | 0.59806 | 0.47540 |
| 725,024 | 0.72596 | 0.62392 | 0.64079 | 0.46364 |

`eval.json` on disk matches the 725,024 row. Val exact is 75/344, so this row wins `(exact, 1−NED)` over 68/344 at 675,008. Gold exact is 1/63. Gold 1−NED rose to 0.62392. Gold exact hits so far are at 625,024, 700,000, and 725,024. Predictions are still mostly generic shells outside those gold hits. Loss started near 10 (two-pass CE; `log(6627) ≈ 8.8`) and is falling. These rows are early. The zero-shot number is the eval written when the 2,000,000-sample stage ends.

## Checkpoints and ONNX

Charset for both whole-crop readers: `ppocr_keys_v1.txt` (6,623 lines) plus a trailing space. `Charset` in `direct_aug.py`: PAD=0, BOS=1, EOS=2, character ids start at 3, vocab 6,627, `MAX_CHARS=25`, so the target length is 26. Decode stops at EOS or PAD. For C++, the character at line `i` (0-based) has id `i+3`.

ImageNet mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`. Input is RGB, square, NCHW, already normalized.

| Artifact | Bytes | I/O |
| --- | --- | --- |
| `checkpoints/ring_baseline/best_accuracy.pdparams` | 92,071,446 | Paddle training weights. Keep. |
| `checkpoints/ring_baseline/inference/` | — | Exported Paddle inference model. |
| `checkpoints/ring_baseline/ring_rec.onnx` | 10,815,395 | Input `x`, height 48 (training resize is 3×48×320). PP-OCRv4 CTC. |
| `checkpoints/ring_v2/best_accuracy.pdparams` and `latest.*` | 92,071,446 each | Killed around epoch 6. Not a model to deploy. `best_model/` holds `model.pdparams` and an inference trio. |
| `checkpoints/seal_reader/best.pt` | 32,045,116 | Epoch 33. Also `last.pt` (same size) and `eval.json`. |
| `checkpoints/seal_reader/seal_reader.onnx` | 32,231,644 | Input `image` float NCHW, dynamic batch, 3×224×224. Output `tokens` int64 `(N, 26)`. Opset 17, `dynamo=False`. |
| `checkpoints/parseq_reader/pretrain/best.pt` and `last.pt` | 126,668,855 each | Updating during the run. `eval.json` is the latest eval, not necessarily the final zero-shot. |
| `checkpoints/parseq_reader/pretrain/parseq_reader.onnx` | not written yet | Export runs at the end of the stage: input `image` 3×384×384, output `tokens` int64 `(N, 26)`, dynamic batch, opset 17, `dynamo=False`. A failed export is printed and does not kill training. Fine-tune writes `checkpoints/parseq_reader/finetune/parseq_reader.onnx`. |

Direct reader: timm `vit_tiny_patch16_224` (embed 192, 3 heads) plus a 2-layer `TransformerDecoder` over 26 learned queries. Not autoregressive. An autoregressive head collapsed to always emitting 中国共产主义青年团. Do not revive it.

Current reader (`parseq_reader.py` `ParseqReader`): timm `vit_small_patch16_384`, ImageNet init, width 384, 6 heads, no grad checkpointing. Decoder is 2 layers, `norm_first`, GELU, dropout 0.1, feed-forward 1536, token embedding plus a learned position. `tokens()` is the ONNX graph: blank context, argmax, second decode with that context, argmax. `HybridEncoder` (five stride-2 convs to 12×12 tokens, 6 encoder layers, name `hybrid384`) exists and was not selected.

## How to train, evaluate, and export

Runtime tree on the server, after a tar sync of this repo:

```bash
export PYTHONPATH=/home/rapidstamp_data/src/RapidStamp/python/src:/home/rapidstamp_data/src/RapidStamp/training
PY=/home/znsoft/miniconda3/envs/pytorch/bin/python
```

The live job was started with `training/scripts/launch_parseq.sh`. It nohups `train_parseq.py` with `--samples 2000000`, `--batch 32`, `--workers 6`, `--backbone vit_small_patch16_384`, `--no-checkpointing`, `--memory-fraction 0.105`, names and blocked files under `synth/`, kept rows `ring_v2/kept.jsonl`, crops `web_seals_cn_all/crops`, gold dir `gold/seals_to_label`. Log: `/home/rapidstamp_data/logs/parseq_reader.log`. Pid file: `logs/parseq_reader.pid`.

Pretrain defaults in `train_parseq.py`: 2,000,000 samples, real probability 0.12, uniform probability 0.10, the rest drawn from `names.txt`. AdamW, encoder lr 1e-4, other lr 3e-4, weight decay 0.01, warmup 4,000 images then cosine, grad clip 1.0, AMP. Loss is CE (`ignore_index=0`) on a blank context plus 0.5 × CE on a second pass where 20% of content positions are replaced with random ids. Eval every 25,000 images, save every 20,000. Checkpoint choice is val `(exact, 1−NED)`, not gold. `best.pt` / `last.pt` store `model`, `step`, `val`, `gold`, `vocab_size`, `max_len`, `backbone`, `image_size`, `pretrained`.

There is no resume-from-step flag. `--init` loads a state dict at build time. `--finetune-only` loads `pretrain/best.pt` when `--init` is unset. `--probe-steps N` runs N optimizer steps and returns; do not point a probe at the live output directory. `torch.load` does not pass `weights_only=False`. If a future torch refuses the checkpoint, add that flag. Do not restart the live job to change it.

Fine-tune is chained unless `--skip-finetune`. It reloads pretrain `best.pt`, then runs `int(12 * n_real / 0.60)` steps. With 10,839 real train rows that is 216,780 steps. Real probability 0.60, uniform still 0.10, encoder lr 2e-5, other lr 1e-4. Real crops get `augment_seal` (full 360° rotation, perspective, blur, noise, JPEG, fade, occlusion, overlaid text). Synthetic samples are rendered on the fly.

Renderer (`seal_render.py`): round, oval, and invoice seals; double ring; star; horizontal invoice line and a tax-id distractor; bottom code or a bottom phrase; Song, Fangsong, Hei, and Kai; stroke, ink texture, fade, double print, paper, overlaid document text. The label is the ring string only. Training always spins the seal. About 90% red ink, 10% blue. Missing glyphs fall through the font list. Upright samples were readable (radial text, tops outward, gap at the bottom).

Paddle baseline, only if you must reproduce it, uses `/home/rapidstamp_data/venvs/paddleocr` and `training/configs/ch_PP-OCRv4_rec_seal.yml`. Do not launch `ch_PP-OCRv4_rec_seal_v2.yml`.

Evaluate the finished direct reader:

```bash
$PY training/scripts/eval_direct_reader.py \
  --ckpt /home/rapidstamp_data/checkpoints/seal_reader/best.pt \
  --dict /home/rapidstamp_data/pretrained/ppocr_keys_v1.txt \
  --kept /home/rapidstamp_data/ring_v2/kept.jsonl \
  --crops /home/rapidstamp_data/web_seals_cn_all/crops \
  --manifest training/manifests/user_seals_split.jsonl \
  --gold /home/rapidstamp_data/gold/seals_to_label \
  --out /tmp/seal_reader_eval.json
```

The docstring example that passes `--gold .../gold` is wrong; use `seals_to_label`. Export that reader with `training/scripts/export_direct_reader.py --ckpt .../best.pt --out .../seal_reader.onnx`. The PARSeq export is the in-process `_export` in `train_parseq.py`. After it writes an ONNX, check it with onnxruntime: input `image`, output `tokens`.

Watch the live job without touching it:

```bash
ps -p $(cat /home/rapidstamp_data/logs/parseq_reader.pid) -o pid,etime,cmd
tail -n 3 /home/rapidstamp_data/logs/parseq_reader.log
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

At about 111 img/s, 2,000,000 samples is roughly five hours from the start (about 05:26 UTC on 2026-10-23), then about half an hour of fine-tune. That is a rate observation, not a schedule. Memory fraction 0.105 keeps the trainer near 4500 MiB so VLLM can stay.

DataLoader workers use spawn. At process shutdown you may see `Exception ignored` and `worker killed by signal: Aborted`, and the traceback mentions `popen_fork.py` even though the context is spawn. That has happened after a clean exit. A worker dying mid-run will take the main process down. Do not restart the live job because of the shutdown message.

## Decisions

- The user stopped the unwrap-plus-CTC track. The long v2 train was killed. The baseline checkpoint stays, and its gold number above is the reference. Arc-only unwrap code is on `main` (`ring_text.py`) and on the server, and it must not be trained again. Aliyun rows have no angle, and the full-circle strips mostly started at the default 140° because the ink gap was under 40°.
- The recognizer reads the whole crop. A direct reader was chosen because unwrapping was dropped, and the model still has to export to ONNX for C++.
- The first direct reader (ViT-Tiny, 224, learned queries) was trained to completion and does not approach Aliyun. Errors are generic company and bureau shells, not a single repeated string. That is why the autoregressive head was abandoned and why scale moved to synthetic pretraining.
- Synthetic pretraining, then real fine-tuning, is the track the user chose after that result. The corpus blocks val and gold strings. The renderer is a new one (`seal_render.py`); it does not import the older closed-charset synthesizer.
- ViT-S/16 at 384 was chosen over the hybrid CNN encoder because it has ImageNet weights and, once workers are warm, runs at about 111 img/s in about 4500 MiB beside VLLM. Short probes at 4–27 img/s were startup (weight download and loading 1.5M names), not the steady rate. Do not retune the batch from a probe.
- Gold is reported at every eval and is not used to pick checkpoints.

## Open issues and next steps

1. Leave PID 52706 running through 2,000,000 samples. It chains fine-tune by itself. Do not launch a second trainer.
2. When pretrain `eval.json` is from the end of the stage, that is the zero-shot report (val and gold, verified and pending) against Aliyun 55/63 and 0.979. Record it here and in `docs/research.md` / `docs/data.md`. The sample sheet is `/home/rapidstamp_data/synth/sample_sheet.jpg`.
3. When fine-tune finishes, record its val and gold numbers the same way. If the in-process ONNX export failed, export again and check the file with onnxruntime. Write the byte size and the path here.
4. The 25k/50k/75k rows above are not the pretrain result.
5. `docs/data.md` still describes the 2026-10-16 `/home/mayong` layout after the warning at the top. Do not follow those paths.
6. Center text and the bottom code stay out of scope until ring text is near the Aliyun bar.
7. `torch.load` may need `weights_only=False` on a later PyTorch. There is still no mid-pretrain resume.
8. After each of the events above, update this file, commit it on `main`, and refresh `/home/rapidstamp_data/docs/*.md`.

Code on `main` for this track starts at `fb7b45d` (synthetic pretraining). Earlier direct-reader commits are `85cc0c7`, `edb34ed`, `d8cd585`, and the score note `e347e6e`.
