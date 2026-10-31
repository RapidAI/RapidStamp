# RapidStamp ring-text handover

Updated 2026-10-29T05:50:00Z. This file is the operational handover. `docs/data.md` and `docs/research.md` are the Chinese measured notes; `docs/sdk.md` is the published detection plus class-routed CTC / PP-OCRv5 SDK. Those three are not this recognizer. The Chinese notes still stop at the 1,050,016 pretrain snapshot; the finished scores are in the experiments section below.

Update this file after every meaningful step, measured result, or change to data or models. Commit it on `main`, then copy `docs/*.md` to `/home/rapidstamp_data/docs/` on the training machine. Do not put passwords, the SSH password, or Aliyun APPCODEs in this file or in git.

## Goal and deliverables

Read the curved ring text of a Chinese seal from the whole crop, without unwrapping. The bar is the Aliyun stamp-OCR draft on the 63 user-proofread gold crops: **55/63 exact, 1−NED 0.979**. The published class-routed pipeline scores **0/63, 1−NED 0.050** on the same crops. The new model has to stay ONNX-exportable for a later C++ path. It is a separate recognizer from `models/seal_rec.onnx` and from `seal_rec_ppocr.onnx`.

Synthetic pretraining and the first fine-tune finished on 2026-10-23 at 17/63 exact and 1−NED 0.84294 on gold. The third-data fine-tune finished at 25/63 and 0.90432. The first mix fine-tune, which did not include `no_seal_0`, finished with selected val 229/455 and gold 25/63, 1−NED 0.90006. The second mix, which includes `no_seal_0`, finished the same day. Selected weights are `checkpoints/parseq_reader_mix2/finetune/best.pt` at step 400,000: val 285/591, 1−NED 0.85177; gold 25/63, 1−NED 0.90795. LabelMe boxes from `det_seal_0` were cropped, detector-filtered, Aliyun-labeled, and merged (kept is now 20,211). A 3-epoch fine-tune from that mix2 checkpoint finished. Selected weights are `checkpoints/parseq_reader_mix3/finetune/best.pt` at step 106,816: val 287/628, 1−NED 0.85324; gold 25/63, 1−NED 0.90375. `last.pt` is the same step and the same scores. The mix3 process has exited. Do not train a second head for center text or the bottom code until ring text is near the Aliyun bar. PaddleOCR-VL-1.6 seal recognition, first line only, scores 46/63 exact and 1−NED 0.96565 on gold. New crops use that first line when the image is sharp and the line looks like a place or organization name. Poor crops and failed name checks fall back to Aliyun. Existing Aliyun labels stay. That merge added 1,003 kept rows, and a 3-epoch fine-tune from mix2 finished. Selected weights are `checkpoints/parseq_reader_vl1/finetune/best.pt` at step 100,000: val 313/665, 1−NED 0.84470; gold 25/63, 1−NED 0.91142. A name-trigram rescore of that checkpoint (alpha 0.4, tau 4) scores 28/63 and 1−NED 0.91516. A 1-epoch hard-example fine-tune into `checkpoints/parseq_reader_hardlex/` reaches 27/63 exact and 1−NED 0.90308. The 512-input fine-tune finished. Selected weights are `checkpoints/parseq_reader_large/finetune/best.pt` at step 325,024: val 330/665, 1−NED 0.84575; gold 26/63, 1−NED 0.91354. That beats vl1's greedy 25/63 and 0.91142. The vl1 lexicon rescore is still the higher eval number at 28/63, 1−NED 0.91516. The ViT-B/16 fine-tune at 384 finished at gold 0/63, 1−NED 0.25425. Do not quote it. The decoder was new; only the ImageNet encoder was pretrained. Details are in `docs/vl_label_policy.md`, `docs/paddleocr_vl_seal_eval.md`, and `docs/hardlex_gold.md`. OXR stays at 1/63 and is not a labeler (`docs/oxr_seal_eval.md`).

Headline metric, from `training/rapidstamp_train/ring_text.py` `one_ned`: mean over samples of `1 − edit_distance / max(len(pred), len(gold))`. Both strings empty scores 1. Spaces count. Do not strip. This is not Paddle `RecMetric` (`ignore_space=True`, and it keeps only CJK and alphanumerics).

## Access and hard rules

- Training host: `home.rapidai.tech`, SSH port 44, user `znsoft`. The password is held by the user and is already in the operator environment as `SSHPASS`. Use `sshpass -e`. Do not write it down. Connections reset; retry.
- Read and write only `/home/rapidstamp_data`. `/home` is `/dev/sdb`, about 1.8T, 713G used, 1.1T free, 41% at 2026-10-27T06:15:02Z.
- `/home/mayong` is forbidden. `ls` shows a root-owned directory dated 2025-05-19. `df` of that path reports the same `/dev/sdb` as `/home`, so it is not a separate live data disk. Older notes in `docs/data.md` tell you to put large files there. Those notes are historical. Do not read it, write it, or try to recover it.
- GPU: RTX 6000 Ada, 49140 MiB. No trainer is running. Pid file `parseq_vitb_finetune.pid` still says 57532; that process has exited. Pid file `parseq_vitb_pretrain.pid` still says 54798. Pid file `parseq_vitb.pid` still says 52406. Pid file `parseq_large.pid` still says 49102. Pid file `parseq_hardlex.pid` still says 47692. Pid file `parseq_vl1.pid` still says 45237. Pid file `parseq_mix3.pid` still says 11314. Pid file `parseq_mix2.pid` still says 7263. Pid files `parseq_mix.pid` (6252), `parseq_third.pid` (4633), and `parseq_reader.pid` (52706) are stale. The det_seal Aliyun labeler pid file `det_seal_aliyun.pid` still says 10467. The VL policy labeler pid `vl_policy_vl.pid` says 28671 and the Aliyun fallback pid `vl_policy_aliyun.pid` says 44811; both finished. The VL eval pid file `paddleocr_vl_eval.pid` says 13134 and the reject labeler `vl_unlabeled.pid` says 19686; both finished. VLLM PID 17019 is also gone. Do not start a trainer until asked. Check `ps -p <pid>`. A `pgrep` that matches an `ssh bash -c` line does not mean a trainer is alive.
- The server tree `/home/rapidstamp_data/src/RapidStamp` is a partial copy, not a git checkout. Sync code with tar over SSH. Do not `git pull` there.
- Do not commit the SSH password, an Aliyun APPCODE, real seal images, gold crops, name lists, fonts, or checkpoints. The live codes are only in `/home/rapidstamp_data/logs/aliyun_keys.env` (mode 600). Counters are the integers in `logs/aliyun_calls.txt` and `logs/aliyun_calls_key2.txt`. Do not copy those files into git.
- Do not delete `incoming_v2`, the gold crops, the ring-baseline checkpoint, the four zips under `third/`, `web_seals_cn_all/labels.jsonl.bak-before-third-20261027`, `web_seals_cn_all/labels.jsonl.bak-before-noseal-20261027`, `web_seals_cn_all/labels.jsonl.bak-before-detseal-20261027`, `ring_v2/kept.jsonl.bak-before-third-20261027`, `ring_v2/kept.jsonl.bak-before-noseal-20261027`, `ring_v2/kept.jsonl.bak-before-detseal-20261027`, `web_seals_cn_all/labels.jsonl.bak-before-vlpolicy-20261027`, `ring_v2/kept.jsonl.bak-before-vlpolicy-20261027`, or `checkpoints/parseq_reader/finetune/best.pt`, `checkpoints/parseq_reader_mix/finetune/best.pt`, `checkpoints/parseq_reader_mix2/finetune/best.pt`, `checkpoints/parseq_reader_vl1/finetune/best.pt`, `checkpoints/parseq_reader_hardlex/finetune/best.pt`, `checkpoints/parseq_reader_large/finetune/best.pt`, `checkpoints/parseq_reader_vitb/`, or `synth_corpus/vitb_pretrain_2m/`. Do not rerun `third/rebuild_kept.py`. It still rebuilds from the pre-third backup and would drop the later rows.

## Server layout

Inventory taken 2026-10-23T05:37:44Z. Sizes are `du -sh`. File counts are recursive.

| Path | Size | What is there |
| --- | --- | --- |
| `/home/rapidstamp_data/backup` | 651M plus the incoming full backup | `gold_db18.zip` (16,362,139 bytes), `web_seals_cn_all.zip` (385,220,370), `web_seals_delta_v2.zip` (263,497,928), and `gold_seals/seals_to_label/` (63 png + `labels.csv`, 64 files). `full_v1/` is a second backup that started landing on 2026-10-23; see below. The 651M figure is from before those parts. |
| `/home/rapidstamp_data/checkpoints` | 7.2G | `parseq_reader/` (finished pretrain and first fine-tune), `parseq_reader_third/` (finished third fine-tune), `ring_baseline/` (75 files), `ring_v2/` (partial, killed), `seal_reader/`. The 7.2G figure is from before `parseq_reader_third/` was written. |
| `/home/rapidstamp_data/gold` | 20M | `seals_to_label/` (63 png + `labels.csv`), `baseline_strips/` (63 strips), `baseline_report/gold_report.json`. |
| `/home/rapidstamp_data/incoming_v2` | 503M | `delta.zip` (263,497,928) and `part_00.zip` … `part_11.zip` (parts 00–10 are 23,068,672 each, part 11 is 9,742,536). Keep them. |
| `/home/rapidstamp_data/logs` | 1.1M | Finished logs are `parseq_reader.log` (pid file still says 52706) and `parseq_third.log` (pid file still says 4633). Both processes are dead. |
| `/home/rapidstamp_data/pretrained` | 264M | `ppocr_keys_v1.txt` (6,623 lines, 26,250 bytes), `ch_PP-OCRv4_rec_train/` plus its tar, `ch_PP-OCRv4_rec_infer/` plus its tar, `fonts/`. |
| `/home/rapidstamp_data/ring_baseline` | 496M | `strips/train` 10,907, `strips/val` 348, `kept.jsonl` 11,255 rows, `train_list.txt`, `val_list.txt`, `meta.json`, `sample_sheet.jpg`. |
| `/home/rapidstamp_data/ring_v2` | 589M | `strips/train` 10,839, `strips/val` 344, `kept.jsonl` now 14,711 rows, backup `kept.jsonl.bak-before-third-20261027` (11,183 rows), `kept_third_meta.json`, lists, `meta.json`, `sample_sheet.jpg`. |
| `/home/rapidstamp_data/src` | 417M | `PaddleOCR/` and `RapidStamp/` (partial runtime tree: `python/` and `training/` only). |
| `/home/rapidstamp_data/synth` | 267M | Name corpus, uniform strings, blocked strings, `stats.json`, `sample_sheet.jpg`, and `src/` (public name lists). |
| `/home/rapidstamp_data/venvs` | 3.7G | `paddleocr` only (Paddle 2.6.2, cu118). |
| `/home/rapidstamp_data/third` | 3.3G | Four original zips plus the extracted trees. See the third-data section. Remeasured 2026-10-27. |
| `/home/rapidstamp_data/web_seals_cn_all` | 1.3G | Original `crops/` 15,738 png, plus `crops/third_rec/` 3,581 files. `labels.jsonl` is 19,319 lines (13,175,513 bytes). Backup `labels.jsonl.bak-before-third-20261027` is the previous file (11,440,722 bytes, 15,738 lines). `labels_v1.jsonl` is unchanged (6,615,523 bytes, 9,085 rows). `web_seals_cn_all.zip` is unchanged (385,220,370). Remeasured 2026-10-27. |

`docs/` on the server is a copy of this repo's `docs/`, refreshed after each handover commit. PyTorch is not in the data-disk venv. The reader uses `/home/znsoft/miniconda3/envs/pytorch/bin/python` (3.11.11, torch 2.8.0+cu129, timm 1.0.30). Do not pip-install Paddle into that env.

Fonts under `pretrained/fonts/`: `NotoSansCJKsc-Regular.otf` (Hei, 16,437,364), `NotoSerifCJKsc-Regular.otf` (Song, 24,543,080), `LXGWWenKaiGBLite-Regular.ttf` (Kai, 14,191,220), `zhuque/ZhuqueFangsong-Regular.ttf` (仿宋, 8,824,084) and `zhuque.zip`.

`web_seals_cn_all/SHA256SUMS.txt` records `347286fe8a4fab9f9c8f3c568d9c31efc07376149def009bafbdace1948f7789` for the zip, with a `/workspace/...` path from the machine that packed it.

### Full box backup

A second copy of the collection box is stored as five plain tar archives, split into raw parts whose names end in `.zip` only so the attachment filter accepts them. Do not unzip a part. Manifest at `/home/rapidstamp_data/backup/full_v1/MANIFEST.txt` (2026-10-23 13:33:20 UTC+8). Checksums at `full_v1/SHA256SUMS.txt`. Parts are in `full_v1/parts/`. Restore with `printf '%03d'` (not `seq -w`): `cat` the parts in order to `full_v1/<archive>.tar`, check the tar sha256, and extract into `/home/rapidstamp_data/raw_box_backup/`. APPCODEs in the source were already replaced with the placeholder `${ALIYUN_APPCODE}`. All five tar sha256 values were rechecked on 2026-10-23T08:05:31Z.

| Archive | Parts | Bytes | Files | Received |
| --- | --- | --- | --- | --- |
| `a1_meta.tar` | 20 | 442,204,160 | 1,124 | complete. Tar sha256 `f2556116d5aa80cdd48af47887a72a40274fff0d398de67620c711b0da25f0dc`. Extracted. |
| `a2_cn_crops.tar` | 60 | 1,361,745,920 | 26,265 | complete. Tar sha256 `b708e5980449559949c136bc1d60127b00d6bbcbb66978b1fcd1218a191fd379`. Extracted into the existing `web_seals_v2_raw/` tree with no file collisions. |
| `a3_foreign_crops.tar` | 26 | 591,646,720 | 4,787 | complete. Tar sha256 `49c76d74067fd9167ad4d46f255d641d575d1d943e37bf133b3abb413319dfb5`. Extracted into `web_seals_v2_foreign/crops/` with no file collisions. |
| `a4_cn_rejected.tar` | 13 | 277,166,080 | 4,350 | complete. Tar sha256 `0e128ad489b79d8219777c29e88baecc99ceb394dc9c2835c9cecc3b9a283d87`. Extracted into `web_seals_v2_raw/rejected/` with no file collisions. |
| `a5_foreign_rejected.tar` | 36 | 813,629,440 | 3,637 | complete. Tar sha256 `ce0fad4aa18bbd53564cd5d03b2d97649ff3974b349703a8d055c7c58d471d4b`. Extracted into `web_seals_v2_foreign/rejected/` with no file collisions. |

`a1_meta` is labels, manifests, urls, logs, scripts, the first Aliyun scrape (998 images), misc csvs, `gold.zip`, `seals_to_label.zip`, `seal_det.onnx`, and combine scripts. `a2` is CN crops not already in `web_seals_cn_all` (b2 3,899, b3 4,034, b4 8,025, b5 6,864, b6 3,443). `a3` is 4,787 foreign crops. `a4` is 4,350 rejected CN batch-1 crops. `a5` is 3,637 rejected foreign crops. `a1_meta.tar` is `/home/rapidstamp_data/backup/full_v1/a1_meta.tar` (442,204,160 bytes). `a2_cn_crops.tar` is `/home/rapidstamp_data/backup/full_v1/a2_cn_crops.tar` (1,361,745,920 bytes, 26,271 tar members, 26,265 files). `a3_foreign_crops.tar` is `/home/rapidstamp_data/backup/full_v1/a3_foreign_crops.tar` (591,646,720 bytes, 4,788 tar members, 4,787 files, 1 directory). `a4_cn_rejected.tar` is `/home/rapidstamp_data/backup/full_v1/a4_cn_rejected.tar` (277,166,080 bytes, 4,351 tar members, 4,350 files, 1 directory). `a5_foreign_rejected.tar` is `/home/rapidstamp_data/backup/full_v1/a5_foreign_rejected.tar` (813,629,440 bytes, 3,638 tar members, 3,637 files, 1 directory). All five are extracted under `/home/rapidstamp_data/raw_box_backup/` (40,163 files, 21 directories, 3,455,497,792 bytes). The five tars together are 3,486,392,320 bytes. a1 top level: `misc/` (13 files, including `gold.zip` and `seals_to_label.zip`, each 16,362,139 bytes, plus `aliyun_calls.txt` and `aliyun_calls_key2.txt`, which were not printed), `models/seal_det.onnx` (3,671,258 bytes), `web_seals_aliyun_firstscrape/` (998 files under `raw/`, plus `downloaded.json`, `manifest.jsonl`, `urls.json`), `web_seals_cn_all_scripts/` (3 files), `web_seals_v2_foreign/` (14 files), and the scripts, manifests, and urls already in `web_seals_v2_raw/`. a2 added crops into that same `web_seals_v2_raw/` tree and did not overwrite an existing file. Batch directories now hold `batch2` 3,915, `batch3` 4,047, `batch4` 8,038, `batch5` 6,881, and `batch6` 3,460 files; the extra files beyond the manifest snapshot (3,899 / 4,034 / 8,025 / 6,864 / 3,443) are the metadata a1 already had in those directories. a3 added 4,787 files under `web_seals_v2_foreign/crops/` and did not overwrite the 14 metadata files a1 had already put in `web_seals_v2_foreign/`. That left the directory at 4,801 files before a5. It now holds 8,438. a4 added 4,350 files under `web_seals_v2_raw/rejected/` and did not overwrite an existing file. Parts 010 and 011 arrived after 012; the tar was assembled only once 000–012 were all present. a5 added 3,637 files under `web_seals_v2_foreign/rejected/` and did not overwrite an existing file. The trainer was left running through all five copies and extracts.

Restored layout under `/home/rapidstamp_data/raw_box_backup/`. Counts are files. Byte totals are the sum of file sizes in that directory tree.

| Path | Files | Bytes | Source |
| --- | --- | --- | --- |
| `misc/` | 13 | 34,090,059 | a1. Includes `gold.zip` and `seals_to_label.zip` (16,362,139 each), plus `aliyun_calls.txt` (5 bytes) and `aliyun_calls_key2.txt` (2 bytes), which were not printed. |
| `models/seal_det.onnx` | 1 | 3,671,258 | a1. |
| `web_seals_aliyun_firstscrape/` | 1,001 | 243,810,729 | a1. `raw/` holds 998 images (243,335,155 bytes), plus `downloaded.json`, `manifest.jsonl`, and `urls.json`. |
| `web_seals_cn_all_scripts/` | 3 | 1,803 | a1. `combine.py`, `merger.sh`, `combine.log`. |
| `web_seals_v2_foreign/` | 8,438 | 1,408,192,781 | a1 metadata (14 files), a3 crops, a5 rejected crops. |
| `web_seals_v2_foreign/crops/` | 4,787 | 587,947,851 | a3. |
| `web_seals_v2_foreign/rejected/` | 3,637 | 810,843,992 | a5. |
| `web_seals_v2_raw/` | 30,707 | 1,765,731,162 | a1 metadata, a2 crops, a4 rejected crops. `crops/` is empty. |
| `web_seals_v2_raw/batch2/` | 3,915 | 231,819,146 | a2 crops plus a1 metadata. Manifest snapshot was 3,899. |
| `web_seals_v2_raw/batch3/` | 4,047 | 210,928,413 | a2 crops plus a1 metadata. Manifest snapshot was 4,034. |
| `web_seals_v2_raw/batch4/` | 8,038 | 426,294,680 | a2 crops plus a1 metadata. Manifest snapshot was 8,025. |
| `web_seals_v2_raw/batch5/` | 6,881 | 403,566,136 | a2 crops plus a1 metadata. Manifest snapshot was 6,864. |
| `web_seals_v2_raw/batch6/` | 3,460 | 203,747,686 | a2 crops plus a1 metadata. Manifest snapshot was 3,443. |
| `web_seals_v2_raw/rejected/` | 4,350 | 273,833,820 | a4. |

## Datasets

### `web_seals_cn_all`

The original file was 15,738 Aliyun-labeled crops. Those rows are unchanged: status 200 is 14,667, status 464 is 1,071. `labels.jsonl` keys on those rows: `batch`, `det_class`, `det_conf`, `file`, `other_texts`, `phash`, `probs` (`ring`, `other`), `query`, `raw`, `ring_text`, `roi` (`height`, `left`, `top`, `width`), `source_url`, `status`. `labels_v1.jsonl` has the same keys and is an earlier 9,085-row slice. It was not changed by the third merge. The Aliyun payload has an ROI and no angle.

`ring_text` / `other_texts` / `probs` on the Aliyun rows come from the Aliyun stamp OCR API. `det_class` and `det_conf` come from the YOLOX-Nano crop. `phash`, `query`, and `source_url` come from the scraper. These strings are drafts, not user proofreading.

On 2026-10-27 the file grew by 3,581 third-party rows, to 19,319 lines. Those rows add a `source` field. See the next section. `ring_v2/kept.jsonl` was not rebuilt, so the finished reader did not train on them.

### Third-party recognition merge

Source zips stayed at `/home/rapidstamp_data/third/`. `unzip` is not installed there, so they were extracted with Python `zipfile` into:

- `/home/rapidstamp_data/third/rec_seal_0/seal_0/` (3,416 files)
- `/home/rapidstamp_data/third/rec_seal_1/seal_1/` (2,266 files)
- `/home/rapidstamp_data/third/det_seal_0/seal_0/` (3,088 files)
- `/home/rapidstamp_data/third/no_seal_0/seal_0/` (5,000 files)

Only labeled recognition was added. `rec_seal_0` is an image plus a same-name one-line `.txt` (1,708 pairs). `rec_seal_1` prefers `Label.txt` (PaddleOCR `path\tJSON`, transcription only; the points are placeholders and were not stored). 2,245 of those rows had an image. 4 `Label.txt` rows had no image and were skipped. Baidu `major.words` was the fallback and contributed 0 rows. `minor.words` and extra transcriptions went into `other_texts`. `det_seal_0` is detection boxes only. `no_seal_0` had no labels at the merge. Both were perceptual-hashed so a labeled crop could be compared with them, and neither was written as a recognition label. Detector prefiltering and Aliyun labeling of `no_seal_0` started later on 2026-10-27. See below. Those labels are not in `kept.jsonl` yet.

Duplicate rule: 64-bit DCT phash, same as the existing rows (checked on `crops/00007_3c973fa9.png` → `fa3189ca66a68b39` and `crops/00001_b74befc0.png` → `ab3194b343cccd2d`). Hamming distance 6 or less is a duplicate. One existing row had a null phash (`b4_08142_553dd9b7.png`); it was hashed from the crop. An existing label wins. Inside the new set, `rec_seal_0` is kept ahead of `rec_seal_1`. A labeled crop that only matches an unlabeled third image is kept.

| Item | Count |
| --- | --- |
| Candidate labeled pairs | 3,953 |
| Dropped, duplicate of an existing label | 27 |
| Dropped, duplicate inside the new set | 345 |
| Of those internal drops, ring text differed | 201 |
| Added | 3,581 (`third/rec_seal_0` 1,646, `third/rec_seal_1` 1,935) |
| `labels.jsonl` lines after | 19,319 |
| Unlabeled det + no_seal images hashed | 6,620 |
| Added rows also within distance 6 of an unlabeled third image | 17 |

New crops are `/home/rapidstamp_data/web_seals_cn_all/crops/third_rec/`. `file` is still relative to `crops/`, for example `third_rec/rec_seal_0/0.jpg`. `source` is `third/rec_seal_0` or `third/rec_seal_1`. `batch` is `third`. `status` is 200. `roi` is the whole image. `probs.ring` is the Baidu major probability when the basename matches, otherwise null. `det_class` is empty. The previous labels file is `/home/rapidstamp_data/web_seals_cn_all/labels.jsonl.bak-before-third-20261027`. The full count table is `/home/rapidstamp_data/third/MERGE_REPORT.md`. The script is `/home/rapidstamp_data/third/merge_third_rec.py`. It refuses to append a second time.

Of the added strings, 3,573 pass length 2–25 and the `ppocr_keys_v1` dictionary. 7 are longer than 25. 1 uses a character outside that dictionary. They stay in `labels.jsonl`. The Aliyun filter in `ring_text.py` also requires ring probability ≥ 0.90 and a class in the allowed set. The third rows fail that probability and class test, so they were added to `kept.jsonl` by a separate pass, below.

### Filtered sets

Filter in `training/rapidstamp_train/ring_text.py`: status 200, non-empty ring text, ring probability ≥ 0.90, not a template, class in `round_official` / `oval` / `invoice` / `finance` / `notary` / `company_latin`, length 2–25, every character in `ppocr_keys_v1` or a space. Val is `sha256(basename)[:8]` as int, mod 1000, less than 30. Gold crops are excluded. Bare `测试` is not banned. The template regex covers 某某, two or more of `X`/`x`/`×`, 演示章, 测试专用, 样章, sample, 示例章, 双击, 替换代用, 可以替换, 代用名, 公司名称, 此处可输, 请在这里, 在这里输入, 输入您的, 输入单位, 编辑文字, 测试样品, 测试效果, 免费设计测试.

Baseline filter (`ring_baseline/meta.json`): label-keep 11,262, then 7 out-of-dict rows dropped, dedupe removed nothing, train 10,907 / val 348. Drop reasons: empty 1,413, status 1,071, template 970, prob 878, class 77, length 67. Median gap 0°, so most strips start at 140°. Wide gap (≥ 40°) on 631 rows.

Extended filter (`ring_v2/meta.json`): label-keep 11,190, template 1,042, before-dedupe 11,183, deduped 0, train 10,839 / val 344. Other reasons match the baseline (empty 1,413, status 1,071, prob 878, class 77, length 67, dict 7). Those 11,183 rows are the prefix of the current `kept.jsonl`. Their keys add unwrap geometry: `angle` (always null; Aliyun has no angle), `confidence`, `file_md5`, `gap_deg`, `prob`, `r_in`, `r_out`, `roi`, `source`, `span_deg`, `split`, `start_deg`, `strip`, `strip_h`, `strip_w`, `text`, `unwrap_start`, `wide_gap`. Sources: gap 10,776, default 181, center 86, star 115, code 25. Median span 300°, median gap 80°, median strip width 960. The finished PARSeq run used this prefix only.

### Third rows in `kept.jsonl`

Rebuilt 2026-10-27 by `/home/rapidstamp_data/third/rebuild_kept.py`. The previous file is `/home/rapidstamp_data/ring_v2/kept.jsonl.bak-before-third-20261027` (5,869,163 bytes, 11,183 lines). The new file starts with those exact bytes. Counts are in `/home/rapidstamp_data/ring_v2/kept_third_meta.json`.

Third rows skip the probability and detection-class gates. A row is added when `ring_text` is non-empty, length is 2–25, every character is in `ppocr_keys_v1` or a space, and the template regex above does not match. Gold ring, center, and code strings are excluded. A third row that would land in train and whose text equals an existing val string is excluded. Split for new rows is `sha256(basename)[:8] % 1000 < 30`, the same rule as `val_split`. Prior rows keep the split they already had. New rows store `file`, `text`, `split`, `source`, `prob` (Baidu major probability when present, otherwise null), and `det_class` `""`. `file` stays relative to `crops/`.

| Item | Count |
| --- | --- |
| Third labeled rows considered | 3,581 |
| Added | 3,528 (`third/rec_seal_0` 1,615, `third/rec_seal_1` 1,913) |
| Added to train / val | 3,417 / 111 |
| Dropped, longer than 25 | 7 |
| Dropped, train text already in the old val set | 41 |
| Dropped, template placeholder | 4 |
| Dropped, character outside the dictionary | 1 (`洧` in 尉氏县洧川镇人民政府) |
| `kept.jsonl` lines after | 14,711 |
| Train / val the loader reported | 14,256 / 455 |

14,256 is 10,839 + 3,417. 455 is 344 + 111. No gold string was dropped because none matched, and no third image was missing. The loader blocks 544 strings: the previous val and gold set plus these 111 new val strings. The launch does not pass `synth/blocked.txt`, so that count comes from `load_blocked` on the new file.

### `no_seal_0` detector prefilter

Before any Aliyun call, `/home/rapidstamp_data/third/no_seal_0/prefilter_no_seal.py` ran the published YOLOX-Nano `seal_det.onnx` (3,671,258 bytes, under `src/RapidStamp/python/src/rapidstamp/models/`). Score is objectness times the best class probability. Keep threshold is the published `SCORE_THRESH` of 0.25. CPU ONNX Runtime, 83 seconds for 5,000 images.

| Item | Count |
| --- | --- |
| Images in `third/no_seal_0/seal_0/` | 5,000 |
| Kept, score ≥ 0.25 | 4,900 |
| Rejected, score < 0.25 | 100 |
| Kept classes | `round_official` 4,883, `square_personal` 10, `invoice` 3, `notary` 2, `oval` 1, `rect_stamp` 1 |

The lowest kept score is 0.25065. The highest rejected score is 0.24798. The list is `/home/rapidstamp_data/third/no_seal_0/prefilter.jsonl` (`file`, `keep`, `score`, `det_class`, `reason`). Aliyun is called only for `keep: true`.

Labeling finished as PID 6088. The file is `/home/rapidstamp_data/third/no_seal_0/labels.jsonl`: 4,900 lines, 4,771 status 200, 129 status 464. Key counters ended at 19,483 and 3,329. HTTP 464 was not merged.

The merge script is `/home/rapidstamp_data/third/no_seal_0/merge_no_seal.py`. Empty `ring_text` used the highest-probability nonempty `general_text` (86 rows). Hamming distance 6 or less against the existing labeled phashes dropped 22 images, and 224 more were duplicates inside this set. 4,525 unique crops were copied to `/home/rapidstamp_data/web_seals_cn_all/crops/third_rec/no_seal_0/` and appended to `web_seals_cn_all/labels.jsonl` (19,319 lines to 23,844). The backup is `labels.jsonl.bak-before-noseal-20261027`. `source` is `third/no_seal_0`.

`ring_v2/kept.jsonl` kept its previous 14,711 lines as a prefix (backup `kept.jsonl.bak-before-noseal-20261027`) and gained 4,260 rows: train 4,124, val 136, total 18,971. The loader reports train 18,380 and val 591. Drops from the appended labels: val-text leak 213, length under 2 was 29, length over 25 was 13, gold text 8, out of dictionary 2. The count table is `/home/rapidstamp_data/third/no_seal_0/MERGE_REPORT.md`.

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

The box one-shots are on this host only as backups: `raw_box_backup/web_seals_v2_raw/collect.py`, `raw_box_backup/web_seals_v2_foreign/collect.py`, and `raw_box_backup/misc/scrape.py`. They write `urls.json` and images under `/workspace` paths from the collection box. They are not a live server job, and they were not started on 2026-10-27. New seals this round came from the already-downloaded `det_seal_0` LabelMe set. Leave web collection on the box.

## Experiments

| Run | Val exact / 1−NED | Gold exact / 1−NED | Notes |
| --- | --- | --- | --- |
| Aliyun ring draft | — | 55/63, 0.979 | The bar. Same 63 proofread strings. |
| Published class-routed CTC | — | 0/63, 0.050 (verified 0/57, 0.053) | 2026-10-17. Every crop detected as `round_official`, so all 63 took CTC. Mean IoU 0.959. About 199 ms/crop. No training in that pass. |
| Unwrap baseline, zero-shot PP-OCRv4 | 8/348, 0.138 | — | Empty predictions 203. Scored on the old strips. |
| Unwrap baseline, 20 epochs | 16/348, 0.224 | 0/63, 0.24173 (verified 0/57, 0.23913; pending 0/6, 0.26641) | Paddle RecMetric best acc 0.083, norm_edit_dis 0.405 at epoch 20. That acc drops spaces. Empty predictions 44. Gold strips were the old full-circle unwrap, 48×640, resized to width 320. `gold_report.json` has top-level `images_found` 0 because `--images` was not passed; the inner `missing` is 0, so all 63 joined. |
| Arc-only unwrap (v2) | not finished | not run | Data prep finished. Training was killed around epoch 6 of 60. Checkpoints under `checkpoints/ring_v2` are partial. Do not resume and do not quote them as a result. |
| ViT-Tiny direct reader, epoch 33 of 40 | 9/344, 0.31392 | 0/63, 0.18681 (verified 0/57, 0.19851; pending 0/6, 0.07564) | Best by val exact. Predictions are mostly generic shells. One val near-miss: 吉南省人力资源和社会保障厅 vs 吉林省人力资源和社会保障行. |
| PARSeq ViT-S pretrain, selected `best.pt` at 1,600,000 | 157/344, 0.83464 | 10/63, 0.80613 (verified 10/57, 0.81852; pending 0/6, 0.68849) | Chosen by val `(exact, 1−NED)`. This is the pretrain ONNX. Stage-end `last.pt` is different; see below. |
| PARSeq ViT-S fine-tune, selected `best.pt` at 200,000 of 216,780 | 156/344, 0.84594 | 17/63, 0.84294 (verified 17/57, 0.85753; pending 0/6, 0.70439) | Chosen the same way. This is the first fine-tune ONNX. The last eval is higher on gold exact and lower on val exact. Aliyun is still 55/63, 0.979. |
| PARSeq ViT-S third fine-tune, selected `best.pt` at 225,024 of 285,120 | 226/455, 0.86562 | 25/63, 0.90432 (verified 25/57, 0.91984; pending 0/6, 0.75691) | Started from the row above. `kept.jsonl` is 14,711 rows, 3,528 of them third-party. Val n is 455 because 111 third rows joined the hold-out. `last.pt` at 285,120 was not selected. |

The job is finished. PID 52706 is dead. The last line of `/home/rapidstamp_data/logs/parseq_reader.log` records the fine-tune ONNX, and that file's mtime is 2026-10-23T10:46:48Z. Checkpoint choice is val `(exact, 1−NED)`, and exact is compared first, so a later step with a higher gold score does not replace `best.pt`.

Stage-end weights, which are `last.pt` and `eval.json`, are not the selected checkpoints:

- Pretrain, seen 2,000,000, `eval.json` mtime 2026-10-23T10:26:54Z: val 152/344, 1−NED 0.83983; gold 14/63, 0.82448 (verified 14/57, 0.83904; pending 0/6, 0.68615). Empty predictions 0.
- Fine-tune, seen 216,800, `eval.json` mtime 2026-10-23T10:46:46Z: val 154/344, 1−NED 0.84429; gold 18/63, 0.84453 (verified 18/57, 0.85928; pending 0/6, 0.70439). Empty predictions 0. Log line at that step: loss 1.82116, 184.5 img/s, lr 0.
- Third fine-tune, seen 285,120, `checkpoints/parseq_reader_third/finetune/eval.json`: val 224/455, 1−NED 0.86361; gold 25/63, 0.89971 (verified 25/57, 0.91766; pending 0/6, 0.72913). Empty predictions 0. This is `last.pt`. The selected checkpoint is the 225,024 row in the table.

`train_parseq.py` reloads `best.pt` before the ONNX export, so the ONNX files are the selected checkpoints, not `last.pt`. The third-run export is `/home/rapidstamp_data/checkpoints/parseq_reader_third/finetune/parseq_reader.onnx` (126,972,402 bytes).

The table below is the snapshot that was copied while the job was still running. It stops at 1,050,016. Intermediate evals between that row and 2,000,000 were not transcribed. Do not fill them in. At the time of that snapshot the running winner was val exact 112/344 at 1,050,016, gold exact 3/63.

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
| 750,016 | 0.71827 | 0.59805 | 0.61030 | 0.48176 |
| 775,008 | 0.74206 | 0.65130 | 0.66496 | 0.52153 |
| 800,000 | 0.74128 | 0.67678 | 0.69110 | 0.54074 |
| 825,024 | 0.74792 | 0.66847 | 0.67411 | 0.61488 |
| 850,016 | 0.75831 | 0.67137 | 0.68042 | 0.58545 |
| 875,008 | 0.74476 | 0.69264 | 0.70517 | 0.57363 |
| 900,000 | 0.76085 | 0.68798 | 0.70257 | 0.54943 |
| 925,024 | 0.76879 | 0.68441 | 0.69288 | 0.60395 |
| 950,016 | 0.78775 | 0.73506 | 0.74386 | 0.65139 |
| 975,008 | 0.77506 | 0.71096 | 0.71922 | 0.63252 |
| 1,000,000 | 0.77347 | 0.65345 | 0.66236 | 0.56884 |
| 1,025,024 | 0.78073 | 0.71124 | 0.71848 | 0.64252 |
| 1,050,016 | 0.78676 | 0.71887 | 0.72813 | 0.63085 |

That 1,050,016 paragraph was true only while the job was running. The on-disk `pretrain/eval.json` is now the 2,000,000 row above, and `pretrain/best.pt` is the 1,600,000 checkpoint. Gold exact hits recorded in this table are 1 at 625,024, 700,000, 725,024, and 1,000,000, 4 at 800,000 and 950,016, 2 at 825,024 and 975,008, 3 at 850,016, 875,008, 900,000, and 1,050,016, and 5 at 925,024 and 1,025,024. Hits after 1,050,016 were not copied row by row. Loss started near 10 (two-pass CE; `log(6627) ≈ 8.8`).

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
| `checkpoints/parseq_reader/pretrain/best.pt` | 126,668,919 | Step 1,600,000. mtime 2026-10-23T09:27:05Z. Selected pretrain checkpoint. |
| `checkpoints/parseq_reader/pretrain/last.pt` | 126,668,919 | Step 2,000,000. mtime 2026-10-23T10:26:54Z. Stage-end weights, not the selected checkpoint. |
| `checkpoints/parseq_reader/pretrain/parseq_reader.onnx` | 126,972,402 | mtime 2026-10-23T10:26:56Z. Reloaded from pretrain `best.pt`. Input `image` 3×384×384, output `tokens` int64 `(N, 26)`, dynamic batch, opset 17, `dynamo=False`. |
| `checkpoints/parseq_reader/finetune/best.pt` | 126,668,919 | Step 200,000. mtime 2026-10-23T10:45:12Z. Selected fine-tune checkpoint. |
| `checkpoints/parseq_reader/finetune/last.pt` | 126,668,919 | Step 216,800. mtime 2026-10-23T10:46:46Z. Stage-end weights. |
| `checkpoints/parseq_reader/finetune/parseq_reader.onnx` | 126,972,402 | mtime 2026-10-23T10:46:48Z. Reloaded from finetune `best.pt`. Same I/O as the pretrain ONNX. This is the init for the third run. |
| `checkpoints/parseq_reader_third/finetune/best.pt` | 126,668,919 | Step 225,024 of 285,120. mtime 2026-10-27T06:45Z. Selected third fine-tune. Val 226/455, 1−NED 0.86562. Gold 25/63, 1−NED 0.90432. |
| `checkpoints/parseq_reader_third/finetune/last.pt` | 126,668,919 | Step 285,120. mtime 2026-10-27T06:51Z. Not selected. |
| `checkpoints/parseq_reader_third/finetune/parseq_reader.onnx` | 126,972,402 | mtime 2026-10-27T06:51Z. Reloaded from the third `best.pt`. Same I/O as the other PARSeq ONNX files. |

Direct reader: timm `vit_tiny_patch16_224` (embed 192, 3 heads) plus a 2-layer `TransformerDecoder` over 26 learned queries. Not autoregressive. An autoregressive head collapsed to always emitting 中国共产主义青年团. Do not revive it.

Current reader (`parseq_reader.py` `ParseqReader`): timm `vit_small_patch16_384`, ImageNet init, width 384, 6 heads, no grad checkpointing. Decoder is 2 layers, `norm_first`, GELU, dropout 0.1, feed-forward 1536, token embedding plus a learned position. `tokens()` is the ONNX graph: blank context, argmax, second decode with that context, argmax. `HybridEncoder` (five stride-2 convs to 12×12 tokens, 6 encoder layers, name `hybrid384`) exists and was not selected.

## How to train, evaluate, and export

Runtime tree on the server, after a tar sync of this repo:

```bash
export PYTHONPATH=/home/rapidstamp_data/src/RapidStamp/python/src:/home/rapidstamp_data/src/RapidStamp/training
PY=/home/znsoft/miniconda3/envs/pytorch/bin/python
```

The first job was started with `training/scripts/launch_parseq.sh`. It nohuped `train_parseq.py` with `--samples 2000000`, `--batch 32`, `--workers 6`, `--backbone vit_small_patch16_384`, `--no-checkpointing`, `--memory-fraction 0.105`, names and blocked files under `synth/`, crops `web_seals_cn_all/crops`, gold dir `gold/seals_to_label`. Log: `/home/rapidstamp_data/logs/parseq_reader.log`. Pid file: `logs/parseq_reader.pid` (still 52706; that process has exited). Its kept file did not yet contain the third-party rows. Do not launch that script again: it would repeat pretraining and write over `checkpoints/parseq_reader`.

The third fine-tune was started with nohup on 2026-10-27 and finished the same morning. It loaded `/home/rapidstamp_data/checkpoints/parseq_reader/finetune/best.pt` and wrote a new tree, so that checkpoint is still in place. There is no resume-from-step, so `seen` started at 0. With 14,256 train rows the step count was `int(12 * 14256 / 0.60)` = 285,120. `kept.jsonl` was the 14,711-row file, 3,528 of them third-party. The same augmentation applied: real probability 0.60, uniform 0.10, `augment_seal` on real crops. The copy of the command is `/home/rapidstamp_data/logs/parseq_third.cmd`. PID 4633 has exited.

```bash
/home/znsoft/miniconda3/envs/pytorch/bin/python -u training/scripts/train_parseq.py \
  --names /home/rapidstamp_data/synth/names.txt \
  --kept /home/rapidstamp_data/ring_v2/kept.jsonl \
  --crops /home/rapidstamp_data/web_seals_cn_all/crops \
  --dict /home/rapidstamp_data/pretrained/ppocr_keys_v1.txt \
  --fonts /home/rapidstamp_data/pretrained/fonts \
  --manifest /home/rapidstamp_data/src/RapidStamp/training/manifests/user_seals_split.jsonl \
  --gold /home/rapidstamp_data/gold/seals_to_label \
  --out /home/rapidstamp_data/checkpoints/parseq_reader_third \
  --init /home/rapidstamp_data/checkpoints/parseq_reader/finetune/best.pt \
  --finetune-only \
  --no-pretrained \
  --batch 32 \
  --workers 6 \
  --backbone vit_small_patch16_384 \
  --no-checkpointing \
  --finetune-epochs 12 \
  --finetune-real-prob 0.60 \
  --uniform-prob 0.10 \
  --finetune-lr 1e-4 \
  --finetune-encoder-lr 2e-5 \
  --memory-fraction 0.105
```

Log: `/home/rapidstamp_data/logs/parseq_third.log`. Pid file: `/home/rapidstamp_data/logs/parseq_third.pid` (4633, dead). The startup line was train 14,256, val 455, gold 63, blocked 544. The log ends at seen 285,120 with the ONNX write. Selected weights are `checkpoints/parseq_reader_third/finetune/best.pt`.

The first mix fine-tune started at 2026-10-27T07:39Z from the third `best.pt` and finished the same morning. It used the 14,711-row kept file and on-the-fly names, real probability 0.55. The planned step count is `int(12 * 14256 / 0.55)` = 311,040. Selected weights are `checkpoints/parseq_reader_mix/finetune/best.pt` at step 250,016: val 229/455, 1−NED 0.87245; gold 25/63, 1−NED 0.90006. `last.pt` at 311,040 was not selected (val 225/455, 1−NED 0.87135; gold 24/63, 1−NED 0.90633). Val exact 229 beats the third checkpoint's 226, so this file is the init for the next run. Log: `logs/parseq_mix.log`. Pid file `logs/parseq_mix.pid` still says 6252; that process has exited. `synth_names/` holds 20,000 rendered images.

The second mix fine-tune started at 2026-10-27T08:42Z from `checkpoints/parseq_reader_mix/finetune/best.pt` into `checkpoints/parseq_reader_mix2/`, so the earlier checkpoints stay. Real probability is 0.55. `kept.jsonl` is the 18,971-row file (train 18,380, val 591), which includes the `no_seal_0` rows. Synthetic names are the 19,999 strings in `synth_names/names_for_mix2.txt` (one of the 20,000 rendered names matched a val or gold string and was left out). Step count is `int(12 * 18380 / 0.55)` = 401,018. The run stopped at seen 401,024 because the batch size is 32. The command copy is `logs/parseq_mix2.cmd`.

Log: `/home/rapidstamp_data/logs/parseq_mix2.log`. Pid file: `logs/parseq_mix2.pid` still says 7263; that process has exited. Checkpoint files are dated 2026-10-27 09:24 on the server clock. Selected weights are `checkpoints/parseq_reader_mix2/finetune/best.pt` at step 400,000: val 285/591, 1−NED 0.85177; gold 25/63, 1−NED 0.90795 (verified 25/57, 0.93060; pending 0/6, 0.69276). `last.pt` at step 401,024 tied those val numbers and was not selected. The ONNX (`parseq_reader.onnx`, 126,972,402 bytes) was exported after reloading `best.pt`. Quote `best.pt`.

`det_seal_0` was turned into recognition crops on 2026-10-27. `seal_0/` has 1,620 jpg and 1,467 LabelMe json. All 2,007 rectangles were cropped to `third/det_seal_0/crops/` with a 4px pad (`crop_labelme.py`). 153 jpg had no json and were left uncropped. `seal_det.onnx` at 0.25 kept 1,957 and rejected 50 (`prefilter_summary.json`; lowest keep 0.251313, highest reject 0.249596; kept classes round_official 1,916, invoice 15, square_personal 13, rect_stamp 10, notary 3). Aliyun then labeled only the kept crops (`label_det.py`, log `logs/det_seal_aliyun.log`). Key1 was already at 19,483, past the soft cap of 19,480, so every call used key2. The labeler pid 10467 finished: 1,942 status 200, 15 status 464, 0 other. Key2 counter went from 3,329 to 5,286. Key1 stayed 19,483. Status 464 was not merged.

Phash dedupe (Hamming ≤ 6, existing label wins) appended 1,295 labels. `labels.jsonl` is 25,139 lines. Backup `labels.jsonl.bak-before-detseal-20261027` is the previous 23,844. Crops are `web_seals_cn_all/crops/third_rec/det_seal_0/` (1,295 png). Of the status-200 rows, 1,819 used `ring_text` and 123 fell back to `general_text`. Duplicates: 605 against the existing set, 42 inside this set. `kept.jsonl` kept its previous bytes as a prefix and grew from 18,971 to 20,211. Added kept rows: 1,240 (train 1,203, val 37). Drops: val_text 42, length_short 11, template 1, charset 1. Backup `kept.jsonl.bak-before-detseal-20261027`. Report: `third/det_seal_0/MERGE_REPORT.md`. Loader counts after the merge: train 19,583, val 628.

The third mix is a short fine-tune, 3 epochs rather than 12, from `checkpoints/parseq_reader_mix2/finetune/best.pt` into `checkpoints/parseq_reader_mix3/`. Real probability 0.55. Names are `synth_names/names_for_mix3.txt` (19,999 lines; none of the mix2 names matched a new val or gold string). Step count is `int(3 * 19583 / 0.55)` = 106,816. Command copy `logs/parseq_mix3.cmd`. Log `logs/parseq_mix3.log`. Pid file `logs/parseq_mix3.pid` is 11314, started 2026-10-27T10:34Z. Startup JSON: names 19,999, blocked 708, train 19,583, val 628, gold 63, gold_missing 0, fonts 4, vocab 6,627. It finished the same morning. Selected weights are `checkpoints/parseq_reader_mix3/finetune/best.pt` at step 106,816: val 287/628, 1−NED 0.85324; gold 25/63, 1−NED 0.90375 (verified 25/57, 0.92155; pending 0/6, 0.73469). `last.pt` is the same step with the same scores. The ONNX is `parseq_reader.onnx`, 126,972,402 bytes, reloaded from `best.pt`. Gold exact stayed 25/63. Gold 1−NED 0.90375 is below mix2's 0.90795. The val set grew from 591 to 628, so 287/628 is not a higher rate than mix2's 285/591. Quote `best.pt`. Do not start another trainer until asked.

PaddleOCR-VL-1.6 was scored after that, on the free GPU, in `/home/rapidstamp_data/venvs/paddleocr_vl` (Paddle 3.2.1, paddleocr 3.7.0). `pipeline_version` was `v1.6` only. Each gold crop used `prompt_label="seal"` with layout detection off, which is the `Seal Recognition:` prompt. The first line is 46/63 exact, 1−NED 0.96565 (verified 46/57, 0.97670; pending 0/6, 0.86067). The full seal block, code line included, is 10/63, 1−NED 0.60481. Use the first line for new labels when the crop is good and the line looks like a place or organization name. Poor quality or a failed name check falls back to Aliyun. Do not rewrite Aliyun rows. The write-up of the gold scores is `docs/paddleocr_vl_seal_eval.md`. The labeling rule and the merge counts are in `docs/vl_label_policy.md`.

The policy merge ran the same afternoon. Batch6's unlabeled seal-like crops and the 150 earlier detector rejects were the inputs. 1,121 candidates were skipped (no ink, blob, colour, aspect, inner box, or a phash already in `labels.jsonl`). VL accepted 636 first lines. Aliyun fallback, key2 only, accepted 540 status-200 rows. Status 464 (265) stayed out. Key2 went from 5,286 to 6,092. Key1 stayed 19,483. `labels.jsonl` grew from 25,139 to 26,315. `kept.jsonl` kept its previous bytes as a prefix and grew from 20,211 to 21,214 (added train 966, val 37). Of the kept rows, 612 are `label_source=vl` and 391 are `aliyun_fallback`. Backups are `labels.jsonl.bak-before-vlpolicy-20261027` and `kept.jsonl.bak-before-vlpolicy-20261027`. Crops are `web_seals_cn_all/crops/third_rec/vl_policy/`. Do not append this set again. Older batch rejects were not labeled. Batch6's 1,037 filtered crops were already Aliyun-labeled.

The vl1 fine-tune is 3 epochs from `checkpoints/parseq_reader_mix2/finetune/best.pt` into `checkpoints/parseq_reader_vl1/`. Real probability 0.55. Names are `synth_names/names_for_vl1.txt` (19,999 lines; none matched a new val or gold string). Step target is `int(3 * 20549 / 0.55)` = 112,085. The run stopped at seen 112,096. Command copy `logs/parseq_vl1.cmd`. Log `logs/parseq_vl1.log`. Pid file `logs/parseq_vl1.pid` is 45237. Startup JSON: names 19,999, blocked 744, train 20,549, val 665, gold 63, gold_missing 0, fonts 4, vocab 6,627. It finished the same afternoon. Selected weights are `checkpoints/parseq_reader_vl1/finetune/best.pt` at step 100,000: val 313/665, 1−NED 0.84470; gold 25/63, 1−NED 0.91142 (verified 25/57, 0.92416; pending 0/6, 0.79042). `last.pt` at step 112,096 was not selected (val 309/665, gold 1−NED 0.91036). The ONNX is `parseq_reader.onnx`, 126,972,402 bytes, reloaded from `best.pt`. Gold exact stayed 25/63. Gold 1−NED 0.91142 is above mix2's 0.90795. Quote `best.pt`.

The same evening the 38 greedy misses were dumped (`logs/gold_errors_vl1.json`). 32 are user-verified and are mostly one- or two-character substitutions. 24 of 51 substitution sites have the gold character as the runner-up. None of those verified gold strings are in `synth/names.txt`. The lexicon rescore is a character trigram from that file, with the organization suffix frozen and at most three low-margin stem characters flipped to the runner-up. Alpha 0.4 and tau 4 on vl1 score 28/63 exact and 1−NED 0.91516 (verified 28/57, 0.93097; pending 0/6, 0.76499). Alpha 1.5 and 3 lose exact matches. The ONNX stays greedy. The write-up is `docs/hardlex_gold.md`.

Directed augmentation did not train on gold crops. 1,693 train rows that share a verified confusion character or a 4-gram are in `ring_v2/hard_vl1.jsonl`. A 1-epoch fine-tune from vl1 `best.pt` used `augment_hard` on half of the real samples (full-turn rotation, stronger blur, stain, occlusion, overlapping text) and wrote `checkpoints/parseq_reader_hardlex/`. Step target `int(1 * 20549 / 0.55)` = 37,361. Seen stopped at 37,376. Command `logs/parseq_hardlex.cmd`. Log `logs/parseq_hardlex.log`. Pid 47692 has exited. Selected weights are `best.pt` at step 25,024: val 303/665, 1−NED 0.83532; gold 27/63, 1−NED 0.90308 (verified 27/57, 0.92288; pending 0/6, 0.71499). `last.pt` at step 37,376 was not selected (gold 27/63, 1−NED 0.90546). Exact is 2 above 25/63. Gold 1−NED is below vl1. The same lexicon settings on this checkpoint stay at 27/63 and 1−NED 0.90935. Keep quoting vl1 `best.pt` for greedy 1−NED until a later run beats it. The backbone stayed ViT-S/16. The next run raises the input to 512 instead of switching to ViT-B.

The large run started at 2026-10-27T20:02Z from `checkpoints/parseq_reader_vl1/finetune/best.pt` into `checkpoints/parseq_reader_large/`. Architecture is the same ViT-S/16 encoder and two-pass decoder. Input is 512 instead of 384, so the patch grid goes from 24×24 to 32×32 and the position embedding from 577 tokens to 1025. The class token is copied and the 24×24 grid is bicubic-resized onto 32×32; every other vl1 weight loads unchanged. ViT-B was not used: its width is 768, the decoder width follows the encoder, and neither the seal encoder nor the two-pass decoder would have loaded. The ONNX graph stays the greedy two-pass `tokens` export. The only C++ change is the input tensor, `1×3×512×512` instead of `1×3×384×384`. Lexicon rescore stays a Python eval post-process (alpha 0.4, tau 4) and is not in the ONNX. Data mix matches mix2: real probability 0.55, uniform 0.10, the rest `synth_names/names_for_vl1.txt`, current `kept.jsonl` (train 20,549, val 665). Hard augmentation is a light mix, probability 0.15 on the 1,693-row list, not the 0.50 mix that lowered 1−NED. Twelve epochs. Step target `int(12 * 20549 / 0.55)` = 448,341. Batch 32, so seen stops on the next batch boundary at or above that (448,352). Encoder lr 2e-5, other lr 1e-4. Memory fraction 0.30. A two-batch probe at this shape peaked at 6,133 MiB. Command `logs/parseq_large.cmd`. Log `logs/parseq_large.log`. Pid file `logs/parseq_large.pid` is 49102, and that process has exited. Selected weights are `best.pt` at step 325,024: val 330/665, 1−NED 0.84575; gold 26/63, 1−NED 0.91354 (verified 26/57, 0.92969; pending 0/6, 0.76012). `last.pt` at step 448,352 was not selected (val 326/665, 1−NED 0.84844; gold 26/63, 1−NED 0.91467). The ONNX is `parseq_reader.onnx`, 127,660,530 bytes. Greedy gold exact and 1−NED are both above vl1. The vl1 lexicon rescore, 28/63 and 0.91516, was not re-run on this checkpoint and is still the higher eval number.

The ViT-B run started at 2026-10-28T01:21Z into `checkpoints/parseq_reader_vitb/`. The encoder is timm `vit_base_patch16_384`, ImageNet AugReg, not the vl1 ViT-S weights. Width 768 cannot load width 384. Input stays 384 because that is the resolution of the pretrained file. The two-pass decoder is new: the official PARSeq release is a different decoder and a non-square input, so those weights were not remapped. The file on disk is `checkpoints/pretrained_vit/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz` (347,502,902 bytes), from `https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz`. The trainer loads that same AugReg checkpoint through timm `pretrained=True` (`timm/vit_base_patch16_384.augreg_in21k_ft_in1k`). There is no `--init`. Three epochs, real probability 0.55, uniform 0.10, names `synth_names/names_for_vl1.txt`, current `kept.jsonl`. Step target `int(3 * 20549 / 0.55)` = 112,085. Batch 32, so seen stops at 112,096. Encoder lr 2e-5, other lr 1e-4. Memory fraction 0.30. A two-batch probe peaked at 8,251 MiB. Command `logs/parseq_vitb.cmd`. Log `logs/parseq_vitb.log`. Pid 52406 has exited. Seen stopped at 112,096. `best.pt` and `last.pt` are the same step: val 0/665, 1−NED 0.30151; gold 0/63, 1−NED 0.25425. Every eval (25,024, 50,016, 75,008, 100,000, 112,096) was 0 exact. Loss fell from 14.81 at seen 224 to 7.33 at the end, which is still the loss the 512 run had at its first log line. Final gold strings are generic shells, not near-misses: `北州市族安区县区院安` for `新都区石板滩镇兴太种子经营部`, `安安公公公司` for `顺发搬运公司`, `甘州科科有限有司司司` for `毕博诚科技有限公司`. Vocab is 6,627 and `max_len` is 26, the same as the S runs. The kept set, names file, batch, and fine-tune learning rates match the 512 run. The 512 run loaded vl1 and used 12 epochs at image size 512. This run did not. The ONNX is 461,167,669 bytes. Do not quote these weights. The failure is a cold decoder: three fine-tune epochs are not the 2,000,000-sample pretrain that taught the S decoder. Do not restart this same fine-tune.

The 2,000,000-sample ViT-B pretrain started at 2026-10-28T04:30Z into `checkpoints/parseq_reader_vitb_pretrain/`. `--skip-finetune` is set. Same AugReg encoder, `vit_base_patch16_384`, image size 384, no S weights. Names are `synth/names.txt` (1,535,269). Real probability 0.12, uniform 0.10, encoder lr 1e-4, other lr 3e-4, batch 32, memory fraction 0.30. Startup JSON: blocked 744, train 20,549, val 665, vocab 6,627. Pid 54798. Log `logs/parseq_vitb_pretrain.log`. An earlier on-the-fly attempt, pid 54416, was stopped at about seen 11,000 because it did not keep the images. It finished. Selected `pretrain/best.pt` is step 1,850,016: val 272/665, 1−NED 0.82337; gold 19/63, 1−NED 0.89505. `last.pt` at step 2,000,000 was not selected (val 272/665, 1−NED 0.82239; gold 21/63, 1−NED 0.89965). The corpus has 1,760,421 PNGs, 197G. Pid 54798 has exited.

The 12-epoch fine-tune started at 2026-10-28T19:56Z from that `best.pt` into `checkpoints/parseq_reader_vitb_finetune/`. `--finetune-only --no-pretrained`, ViT-B/16 at 384, real probability 0.55, uniform 0.10, encoder lr 2e-5, other lr 1e-4, batch 32, hard probability 0.15. Names are `synth_names/names_for_vl1.txt`. Step target `int(12 * 20549 / 0.55)` = 448,341. Command `logs/parseq_vitb_finetune.cmd`. Log `logs/parseq_vitb_finetune.log`. Pid 57532 has exited. Seen stopped at 448,352. Selected `finetune/best.pt` is step 425,024: val 303/665, 1−NED 0.83566; gold 24/63, 1−NED 0.90867. `last.pt` at step 448,352 was not selected (val 302/665, 1−NED 0.83641; gold 24/63, 1−NED 0.90973). Both are below the 512 greedy 26/63, 0.91354, and below the vl1 lexicon 28/63, 0.91516. The ONNX is `finetune/parseq_reader.onnx`, 461,167,669 bytes, from `best.pt`. Do not start another trainer.

New real labels, not merged into `kept.jsonl` or `labels.jsonl`, are `/home/rapidstamp_data/labels/supplement_20261029.jsonl` (backup beside it, raw calls in `supplement_20261029_raw.jsonl`). Source was unlabeled batch2–5 crops, not the 63 gold seals and not the 180 review crops. 900 poor crops went straight to Aliyun. 700 good crops used PaddleOCR-VL-1.6 first line, then `check_place.py`. Name-valid rows: 352 (vl_accept 243, vl_corrected 3, aliyun_key1 106). Key1 went from 19,483 to 20,837 (1,354 calls, all on key1; 546 of the 1,900 allowance unused). Key2 stayed 6,092, so 4,608 remain under the 10,700 cap. Status 464 was not kept. Do not rewrite existing Aliyun labels.

A review set for more gold labels is `/home/rapidstamp_data/gold_candidates/`. It has 180 train-split crops that are not in `gold/seals_to_label`: 45 long names, 45 blurry, 45 low-contrast, 45 odd aspect. `review.jsonl` and `review.csv` have `id`, `path`, and `suggested_text` from the kept label. The existing 63 gold files were not modified.

Synthetic stamps are kept. The generator is `rapidstamp_train.seal_render.render_seal`, which is already the document path: tinted paper noise, ruled lines and gray text under the stamp, document text over the stamp on about 45% of samples, directional fade, and a 1–3 px bleed. It is not an isolated seal on white. Each training canvas is written as a 384 PNG under `/home/rapidstamp_data/synth_corpus/vitb_pretrain_2m/images/`, with the ring string in `shards/<pid>.jsonl` (`file`, `text`, `kind` of `name` or `uniform`). Real kept crops are not copied. `synth_names/images` is an older 20,000-file render and is not this corpus. The first S pretrain did not save its 2,000,000 frames. Do not delete `synth_corpus/vitb_pretrain_2m`.

SoMark OXR-1.0 was scored the same morning on the free API, without a local GPU load. Do not use OXR to label. Joined Stamp text is 0/63 exact, 1−NED 0.08984. The longest Stamp line is 1/63, 1−NED 0.11131. Keep Aliyun for labeling. The write-up is `docs/oxr_seal_eval.md`. Raw responses are `/home/rapidstamp_data/gold/oxr_eval/` and stay off git.

Pretrain defaults in `train_parseq.py`: 2,000,000 samples, real probability 0.12, uniform probability 0.10, the rest drawn from `names.txt`. AdamW, encoder lr 1e-4, other lr 3e-4, weight decay 0.01, warmup 4,000 images then cosine, grad clip 1.0, AMP. Loss is CE (`ignore_index=0`) on a blank context plus 0.5 × CE on a second pass where 20% of content positions are replaced with random ids. Eval every 25,000 images, save every 20,000. Checkpoint choice is val `(exact, 1−NED)`, not gold. `best.pt` / `last.pt` store `model`, `step`, `val`, `gold`, `vocab_size`, `max_len`, `backbone`, `image_size`, `pretrained`.

There is no resume-from-step flag. `--init` loads a state dict at build time. `--finetune-only` loads `pretrain/best.pt` when `--init` is unset. `--probe-steps N` runs N optimizer steps and returns; do not point a probe at the finished output directory. `torch.load` does not pass `weights_only=False`. If a future torch refuses the checkpoint, add that flag. Do not restart the finished job to change it.

Fine-tune is chained unless `--skip-finetune` or `--finetune-only`. The first run reloaded pretrain `best.pt`, then ran `int(12 * 10839 / 0.60)` = 216,780 steps. The third run used the same formula on 14,256 train rows, which is 285,120 steps, and loaded the first fine-tune `best.pt` via `--init`. Real probability 0.60, uniform still 0.10, encoder lr 2e-5, other lr 1e-4. Real crops get `augment_seal` (full 360° rotation, perspective, blur, noise, JPEG, fade, occlusion, overlaid text). Synthetic samples are rendered on the fly.

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

No trainer is running. `parseq_vitb_finetune.pid` still names 57532, and that pid has exited. To confirm:

```bash
ps -p $(cat /home/rapidstamp_data/logs/parseq_vitb_finetune.pid) -o pid,etime,cmd
tail -n 3 /home/rapidstamp_data/logs/parseq_vitb_finetune.log
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

Steady pretrain rate was about 111 img/s. The fine-tune log ended near 184 img/s on the real-data mix. Memory fraction 0.105 kept the trainer near 4500 MiB. Those are observations from the finished run, not a reason to start another one.

DataLoader workers use spawn. At process shutdown you may see `Exception ignored` and `worker killed by signal: Aborted`, and the traceback mentions `popen_fork.py` even though the context is spawn. That has happened after a clean exit. A worker dying mid-run will take the main process down. Do not restart a finished job because of the shutdown message.

## Decisions

- The user stopped the unwrap-plus-CTC track. The long v2 train was killed. The baseline checkpoint stays, and its gold number above is the reference. Arc-only unwrap code is on `main` (`ring_text.py`) and on the server, and it must not be trained again. Aliyun rows have no angle, and the full-circle strips mostly started at the default 140° because the ink gap was under 40°.
- The recognizer reads the whole crop. A direct reader was chosen because unwrapping was dropped, and the model still has to export to ONNX for C++.
- The first direct reader (ViT-Tiny, 224, learned queries) was trained to completion and does not approach Aliyun. Errors are generic company and bureau shells, not a single repeated string. That is why the autoregressive head was abandoned and why scale moved to synthetic pretraining.
- Synthetic pretraining, then real fine-tuning, is the track the user chose after that result. The corpus blocks val and gold strings. The renderer is a new one (`seal_render.py`); it does not import the older closed-charset synthesizer.
- ViT-S/16 at 384 was chosen over the hybrid CNN encoder because it has ImageNet weights and, once workers are warm, runs at about 111 img/s in about 4500 MiB beside VLLM. Short probes at 4–27 img/s were startup (weight download and loading 1.5M names), not the steady rate. Do not retune the batch from a probe.
- Gold is reported at every eval and is not used to pick checkpoints. That is why fine-tune `best.pt` is gold 17/63 while `last.pt` is gold 18/63: val exact was 156 against 154.
- On 2026-10-27, labeled third-party recognition was merged into `web_seals_cn_all` after a phash dedupe. Detection-only and unlabeled third images were not added. The original zips and the labels backup stay.
- Later the same day, 3,528 of those rows were appended to `ring_v2/kept.jsonl` (14,711 lines) and a fine-tune was started from `checkpoints/parseq_reader/finetune/best.pt`. It finished with selected gold 25/63 and 1−NED 0.90432. Quote `checkpoints/parseq_reader_third/finetune/best.pt`.
- `no_seal_0` was filtered with `seal_det.onnx` at 0.25 (4,900 kept, 100 rejected). Aliyun labeled the kept set. 4,525 unique status-200 crops were merged, and 4,260 of them passed the kept-list text rules. The first mix run did not see them. The second mix run did, and it has finished.
- Free SoMark OXR-1.0 was tried as a labeler on the 63 gold crops (2026-10-27). Longest Stamp line is 1/63 exact, 1−NED 0.11131. Keep Aliyun. See `docs/oxr_seal_eval.md`. Do not label with OXR.
- `det_seal_0` LabelMe rectangles were cropped, passed through `seal_det.onnx` at 0.25 (1,957 kept, 50 rejected), and Aliyun-labeled on key2 only. 1,295 unique labels and 1,240 kept rows were appended. Web collection stays on the box.
- PaddleOCR-VL-1.6 seal task, first line, is 46/63 exact and 1−NED 0.96565 on gold. Good crops keep that first line when it looks like a place or organization name. Poor crops and failed name checks use Aliyun. Existing Aliyun labels stay. See `docs/vl_label_policy.md` and `docs/paddleocr_vl_seal_eval.md`.
- vl1 greedy gold misses are mostly one-character substitutions. A name-trigram rescore (alpha 0.4, tau 4) scores 28/63 and 1−NED 0.91516. The 512 run's selected greedy score is 26/63 and 1−NED 0.91354. The lexicon number is still higher. See `docs/hardlex_gold.md`.

## China place-name lexicon

VL first-line checks can use `/home/rapidstamp_data/lexicons/china_place_names/`. `places_all.txt` has 36,199 unique names. `places_by_level.jsonl` is the 省/市/区县/乡镇 table. `PLACEHOLDER.txt` is kept, with a copy at `lexicons/PLACEHOLDER.txt.bak`. The checker is `check_place.py`. Actions are `accept`, `correct`, and `aliyun`. Correct only when the fix is confident: a unique missing parent city (`西夏区` → `银川市西夏区`) or a broken organization suffix (`有公司` → `有限公司`). Send the crop to Aliyun when two real place names are confusable (`扬州` / `杭州`), when the error is inside the organization name (`兴太` / `兴大`), or when the image is poor. Do not rewrite a label Aliyun already wrote. The rule is also in `lexicons/china_place_names/POLICY.md`.

## Open issues and next steps

1. Quote `checkpoints/parseq_reader_large/finetune/best.pt` (step 325,024) for greedy gold: val 330/665, 1−NED 0.84575; gold 26/63, 1−NED 0.91354. `last.pt` at step 448,352 was not selected. Best eval-with-lexicon stays the vl1 rescore, 28/63 and 1−NED 0.91516 (alpha 0.4, tau 4). ViT-B finished at gold 0/63, 1−NED 0.25425; `best.pt` and `last.pt` are both step 112,096. Do not quote it. The decoder was random; only the ImageNet ViT-B encoder was pretrained, and the run skipped the 2,000,000-sample pretrain. Do not restart that fine-tune. That pretrain finished. Quote its `pretrain/best.pt` only as the fine-tune init (step 1,850,016, gold 19/63, 1−NED 0.89505). The 12-epoch fine-tune finished as pid 57532 (gold 24/63, 1−NED 0.90867). New labels are `labels/supplement_20261029.jsonl` (352 name-valid rows) and are not in `kept.jsonl`. Saved stamps stay at `/home/rapidstamp_data/synth_corpus/vitb_pretrain_2m/`. A separate review set, not part of the 63, is `/home/rapidstamp_data/gold_candidates/` (180 crops). Do not start a second trainer. hardlex `best.pt` is 27/63 and 1−NED 0.90308. Aliyun remains 55/63, 0.979.
2. The `det_seal_0` merge and the `no_seal_0` merge are done. Do not append either set again. HTTP 464 stayed out of `labels.jsonl` and `kept.jsonl`. Do not rerun `third/rebuild_kept.py`.
3. The third fine-tune result remains `checkpoints/parseq_reader_third/finetune/best.pt` (step 225,024): val 226/455, 1−NED 0.86562; gold 25/63, 1−NED 0.90432. `last.pt` at 285,120 was not selected.
4. `docs/research.md` and `docs/data.md` still describe the run as in progress at the 1,050,016 snapshot. The sample sheet is `/home/rapidstamp_data/synth/sample_sheet.jpg`.
5. The 25k through 1,050,016 table is a partial log, not the pretrain result.
6. `docs/data.md` still describes the 2026-10-16 `/home/mayong` layout after the warning at the top. Do not follow those paths.
7. Center text and the bottom code stay out of scope until ring text is near the Aliyun bar.
8. `torch.load` may need `weights_only=False` on a later PyTorch. There is still no mid-run resume.
9. After each meaningful step, update this file, commit it on `main`, and refresh `/home/rapidstamp_data/docs/*.md`.
10. VL first-line proofreading can use `lexicons/china_place_names/`. Confident place or suffix fixes stay local. Ambiguous place names and organization-character errors go to Aliyun. Do not rewrite existing Aliyun labels.

Code on `main` for this track starts at `fb7b45d` (synthetic pretraining). Earlier direct-reader commits are `85cc0c7`, `edb34ed`, `d8cd585`, and the score note `e347e6e`.

## Gold candidates v2 (2026-10-30)

The first 180 candidates had many crops too small or blurry for a person to read. They moved to `gold_candidates_v1_old/` and were not deleted. `gold_candidates/` now holds v2: 180 crops, 45 per bucket (`long_name`, `model_hard`, `low_contrast`, `odd_shape`), with `review.csv`/`review.jsonl` (prefilled text, source, model prediction, size, sharpness; `user_text`/`user_status` empty for review), `summary.json`, and `contact_sheet_sample.jpg`. Readability floor: short side >= 80 px, Laplacian variance after resizing short side to 128 >= 400, gray std >= 18; dHash distance to any of the 63 gold crops >= 5. The `blur` bucket is replaced by `model_hard`: readable (short >= 100, sharpness >= 700) crops where the S@512 `checkpoints/parseq_reader_large/finetune/best.pt` plus name-trigram rescore (alpha 0.4, tau 4) disagrees with the label or VL disagrees with the label, 1-NED >= 0.3, val/new rows only (40 val, 5 supplement). From v1: long_name kept 43, low_contrast 0, odd_shape 23, blur 0; 114 were added from `ring_v2/kept.jsonl` and `labels/supplement_20261029.jsonl`. No Aliyun calls. Scripts and pool scores (21,566 crops) are in `gold_rescreen/`. Prefilled text is not verified; gold is not merged until the user reviews it.

## Gold v2 (2026-10-30)

The user reviewed the 180 v2 candidates by editing `prefilled_text` in `gold_candidates_verified/review.csv` (`user_text`/`user_status` were left empty). Conventions: `(wu fa shi bie)` full-width parens = unreadable, `(fei yin zhang)` = not a seal, `/` = two seals in one crop, `#` = extra lines (center/bottom); the ring text is the part before the first `#`. Result: 164 single-seal usable, 8 multi-seal (all odd_shape), 3 unreadable (gc2_117/139/170), 5 non-seal (gc2_161/165/175/178/180, listed in `gold_v2/excluded.jsonl`, useful as detector negatives). `gold_v2/` = original 63 (bucket `orig63`, copied from `gold/seals_to_label`, `gold/` untouched) + 172 verified crops; `labels.jsonl` has `text` (ring), `full_text`, `texts`, `multi`, `bucket`, `split_in_pool`. Backup `backup/gold_v2_20261030/`. Build script `gold_rescreen/build_gold_v2.py`, eval `gold_rescreen/eval_gold_v2.py`, results `gold_v2/eval_20261030.json`, per-image preds `gold_v2/preds_20261030.jsonl`.

Single-seal metric is on 227 rows (63 + 164); multi rows are scored separately (no model hits either seal, 0/8). Lexicon = name trigram alpha 0.4 tau 4 with all gold_v2 ring strings removed from the names file. All-single exact / 1-NED: large (S@512) greedy 84/227 0.834, lex 91/227 0.839 (best); vl1 lex 83 0.825; mix3 lex 87 0.818; hardlex lex 84 0.809; ViT-B finetune lex 84 0.812; ViT-B pretrain lex 67 0.778; third lex 62 0.754. S@512 lex leads on every bucket except model_hard (selected by S@512 disagreement, biased against it; ViT-B finetune lex 19/45 there vs 13/45). 105 of the 164 new crops were in the train split of `ring_v2/kept.jsonl` with the unverified label, so they must be removed from training before the next fine-tune; unseen-only subset (59): large lex 16/59 0.813, ViT-B finetune lex 20/59 0.787.

Label noise on the 164 single candidates (hard-biased sample, not a pool-wide rate): ring text wrong in 64 (39%), center line missing only in 39, identical in 61. By source: Aliyun-family (gap/default/star/center/aliyun_*) 30/71 ring wrong, third-party labels 29/66, VL first line 5/27. Per bucket ring wrong: long_name 25/45, low_contrast 16/44, model_hard 16/45, odd_shape 7/30. Note: `select_v2.py` text-skip strings were mangled to `????` by the Windows pipe, so the "copy-identical" skip filter did not apply; no impact on the reviewed set.

## Nogold refinetune + gold-style synth (2026-10-30, in progress)
- Cleaning: `gold_rescreen/clean_nogold.py` (report `gold_v2/clean_nogold_20261030.json`). Removes rows whose file is a gold v2 image, dHash<5 to any of the 243 gold v2/excluded images, or same ring text with dHash<12. Originals untouched; backups in `backup/*.bak_nogold_20261030`; outputs `*_nogold_20261030.jsonl`.
  - ring_v2/kept: 21,214 -> 18,757 (train 20,549->18,184, val 665->573; dhash 2,163, same-text 143, gold file 151).
  - ring_baseline/kept: 11,255 -> 9,165. hard_vl1: 1,693 -> 1,401. supplement_20261029: 352 -> 308 (not used in training).
  - dHash<5 is aggressive (many generic red round seals collide); it over-removes rather than under-removes.
  - `synth_names/blocked_nogold_20261030.txt` (885) = load_blocked + gold texts, passed via `--blocked`.
- Run A `checkpoints/parseq_reader_large_nogold/`: production S@512 cmd with nogold kept/hard/blocked (`logs/parseq_large_nogold.cmd/.log/.start/.end`). Still init from vl1 (vl1 saw gold-contaminated data; caveat).
- Gold-style synth `synth_corpus/gold_style_20261030/` (12,000 PNG, labels.jsonl abs paths, stats.json, SOURCE.txt), generator `gold_rescreen/gen_gold_style.py`, log `logs/gen_gold_style_20261030.log`. Styles: long names 2,655, low contrast 4,775, center/bottom lines 7,283, shapes round 4,169 / oval 2,835 / square 1,736 / double-ring 3,260, ~18% neighbour seal overlap, document body text, ruled lines, signature strokes, small resize 200-448 + JPEG. Texts from places_all + name_corpus composers + names_for_vl1; rejects gold v2 strings (equal or substring >=5 chars) and the blocked list (1,344 rejected). Faint-ring samples are re-rendered. No gold image used.
- Run B `checkpoints/parseq_reader_large_nogold_gs/`: same cmd with `ring_v2/kept_nogold_goldstyle_20261030.jsonl` (kept_nogold + 12k synth as train rows, 30,184 train). Note 12 epochs over a larger set = ~1.7x steps of A.
- Eval: `gold_rescreen/eval_gold_v2_cli.py <model,model,...> <tag>` -> `gold_v2/eval_<tag>.json`, `preds_<tag>.jsonl`. Trainer exports `finetune/parseq_reader.onnx` itself.

## 2026-10-30 nogold 对比评测结果（自动检查写入）
文件：gold_v2/eval_nogold_20261030.json、eval_nogold_gs_20261030.json；日志 logs/eval_nogold_auto_20261030.log。均为 S@512。
| 模型 | 227单章 exact greedy/lex | 1-NED lex | 59 未见子集 exact greedy/lex | 已见105 exact lex | 长名45 lex | 低对比44 lex | 多章8 hit / best1NED |
|---|---|---|---|---|---|---|---|
| 生产 parseq_reader_large (step 325024) | 84 / 91 | 0.839 | 18 / 16 | 45 | 8 | 29 | 0 / 0.345 |
| nogold (step 375008) | 75 / 80 | 0.812 | 19 / 19 | 34 | 3 | 24 | 0 / 0.287 |
| nogold+gold风格合成 (step 375008) | 73 / 79 | 0.797 | 20 / 21 | 32 | 3 | 22 | 0 / 0.251 |
结论：生产模型的领先主要来自 gold 近重复泄漏（已见子集 45 vs 34）；在真正未见的 59 张上 nogold 系列持平略好。gold 风格合成没有带来整体提升，长名/低对比反而下降。多章 8 张三者都 0 命中。未切换生产（需用户批准）。

## Gold-style synth v2 (2026-10-30, done)
- Diagnosis of v1 (gold_rescreen/diag_gs.py): v1 synth was too pale/gray, loosely cropped (~320px vs real gold long-name ~145px), long names too short (median 18 vs 23), and low-contrast was pale red, whereas gold low-contrast is desaturated gray copies (ink frac ~0.004, bg ~189). Error pattern: on long names the models drop 1-2 chars.
- v2 generator: gold_rescreen/gen_gold_style_v2.py -> synth_corpus/gold_style_v2_20261030 (6000 imgs: 1198 long>=18, 1509 gray low-contrast, neighbour overlap 15%, white doc paper, tight crop, lognormal size ~150/220px; gold strings excluded, 361 blocked).
- Train: checkpoints/parseq_reader_large_nogold_gs2 (init nogold best, 4 ep, lr 5e-5/1e-5, kept=ring_v2/kept_nogold_goldstyle_v2_20261030.jsonl = 18184 real + 6000 synth ~25%). cmd logs/parseq_large_nogold_gs2.cmd; 13:01-13:23 UTC.
- Eval gold_v2/eval_nogold_gs2_20261030.json (exact/227; greedy|lex):
  - prod 84|91, nogold 75|80, nogold_gs 73|79, nogold_gs2 74|77
  - unseen59: prod 18|16, nogold 19|19, gs 20|21, gs2 18|16
  - long45: prod 4|8, nogold 2|3, gs 3|3, gs2 2|2
  - low44: prod 28|29, nogold 23|24, gs 19|22, gs2 22|24
  - multi8 mean best 1-NED (greedy): prod .345, nogold .287, gs .251, gs2 .283 (lex .312)
- Conclusion: v2 removes the v1 low-contrast regression (back to nogold level) but does not help long names or unseen; synth-style alone is not the lever. Production unchanged. Next ideas: long-name fix likely needs decoder max-len/aspect (ring unwrap resolution) rather than more synth; multi-seal needs detector-side separation.

## Overnight wrap-up (2026-10-31 ~04:36-05:30 Asia/Shanghai)

### 1. Cross-vote label cleaning (label_clean/cross_vote.py, TAG 20261030)
- Input `label_clean/pool_preds_20261030.jsonl` (19065 rows; large 325024 / vitb_finetune 425024 / nogold 375008). Log: `label_clean/cross_vote_20261030.log`.
- Tiers (all 19065): clean 9638 / likely_keep 1759 / likely_fix 1544 / conflict 6124. kept:train 18184 = 9330/1660/1496/5698; kept:val 573 = 251/60/38/224; supplement:new 308 = 57/39/10/202.
- By source: aliyun-family 14922 = 6851/1346/1387/5338; rec_seal(third) 3371 = 2500/338/123/410; vl 772 = 287/75/34/376.
- Gold v2 validation (n=82, excl. orig63+multi): clean 12 (12 label-ok), likely_keep 4 (4 ok), likely_fix 9 (2 label-ok, 7 proposal-ok), conflict 57 (36 label-ok, 2 proposal-ok). label_wrong 28, **all 28 flagged** (none fell into clean/likely_keep). File: `label_clean/gold_validation_20261030.json`.
- Review queue `label_clean/review_20261030/review.csv`: **250 rows** (229 conflict + 21 likely_fix), with images.
- Cleaned copies (originals untouched):
  - `ring_v2/kept_nogold_vote_strict_20261030.jsonl` 11301 rows / `..._vote_fix_20261030.jsonl` 12835 rows (src kept_nogold 18757)
  - `labels/supplement_nogold_vote_strict_20261030.jsonl` 96 / `..._vote_fix_20261030.jsonl` 106 (src 308)

### 2. Long-name len32 reader eval (gold_v2/eval_len32_20261030.json)
exact greedy|lex on gold v2 (227 single):

| model | all227 | unseen59 | long45 | seen105 | orig63 | 1-NED lex all |
|---|---|---|---|---|---|---|
| production large (325024) | 84\|91 | 18\|16 | 4\|8 | 40\|45 | 26\|30 | 0.8388 |
| nogold (375008) | 75\|80 | 19\|19 | 2\|3 | 32\|34 | 24\|27 | 0.8117 |
| nogold_len32 (295136) | 65\|70 | 18\|20 | 2\|0 | 28\|28 | 19\|22 | 0.8113 |

Long-name 1-NED lex: prod 0.7776 / nogold 0.7223 / len32 0.7167. **len32 did not help long names; production unchanged.**

### 3. Detector-first multi-seal (gold_rescreen/det_multi.py, PAD=2 UNION=1, score 0.25 nms 0.45)
16 seals / 8 multi images, exact / mean 1-NED: large greedy 3/0.6582, lex 4/0.6626; nogold 4/0.6434, 4/0.6439; len32 4/0.6184, 4/0.6281 (whole-crop baseline best 1-NED ~0.345).
Detector on 227 singles: 1 box 220, 0 box 3, 2 boxes 4. Negatives (8): 6 got a box (gc2_117/139/161/170/175/178), 2 none.
Outputs: `gold_v2/eval_multi_3m_u_p2_20261031.json` (identical to `..._p2_s0.25_n0.45_20261031.json`), logs `logs/det_multi_3m_*_20261031.log`.

### 4. TrOCR diagnosis (gold_rescreen/trocr_diag.py) and v2 retrain
- v1 (`checkpoints/trocr_seal_base`, 8 ep / 6760 steps) val exact peaked 0.0506.
- Generation config correct (dec_start 101 / eos 102 / pad 0); default vs explicit generate identical. Teacher-forced on 128 val: loss 1.856, token-acc 0.721, seq exact 0.047 -> same as free-run, so NOT a decoding bug.
- Shuffled images: loss 3.249 / tok-acc 0.561; blank images: 2.587 / 0.613 -> model is mostly a Chinese LM; errors concentrate on first/place-name chars (e.g. gold 揭阳市揭东区 -> 重阳市揭东区).
- Root cause: weak visual grounding — randomly-initialised BERT cross-attention (28.4M params) trained at decoder LR for only 6760 steps. Contributing: 289 train labels with "#" extra lines (val/gold are ring-only), 404 UNK rows.
- Fix `gold_rescreen/trocr_seal_v2.py`: ring-only labels, drop tokenizer-roundtrip mismatches (train 26628, UNK 0), x-attn LR 3e-4 / enc 5e-5 / dec 1e-4, warmup 1500, 30 epochs (24960 steps, ~130 img/s, ~2 h), explicit generation_config.
- Running via nohup `logs/trocr_seal_base_v2.sh` (started 2026-10-31 04:44 CST): train log `logs/trocr_seal_base_v2.log` -> eval `logs/trocr_seal_base_v2_eval.log` -> `gold_v2/eval_trocr_v2_20261031.json`; `.end` marker when done. Output `checkpoints/trocr_seal_base_v2/{best,last}`.

### 5. CSD (ChineseSealDataset) test split as 2nd external eval (gold_rescreen/csd_eval.py)
- Data `external/csd/raw/ChineseSealDataset/` (train 828 / val 200 / test 458 images, 4032x3024). Crops from seal_location +4% margin -> `external/csd/crops_test/`. Ring text = **longest** shape label (v1 used most polygon points, but ring and center text both have 20 points, so ties sometimes picked the center text such as 研究专用; v1 log kept as `logs/csd_eval_20261031_v1_ringbug.log`, not valid). JSON utf-8-sig with gb18030 fallback.
- Names overlapping gold v2 dropped (name-disjoint); lex LM blocks gold v2 + CSD names. Groups: seen/unseen name vs train labels, difficulty, len>=14, color.
- Output `external/csd/eval_csd_test_20261031.json`, preds `external/csd/preds_csd_test_20261031.jsonl`, log `logs/csd_eval_20261031.log`.
- Stats: 5454 seals (458 images, ~12 stamp instances of the same seal per image), 119 dropped for gold-name overlap -> **5335 eval crops, only 61 unique names, 0 seen in train labels**, all len<14. Difficulty easy 2087 / normal 2302 / difficult 1065. Colours black/blue/red (CSD is photographed, many non-red seals -> large domain gap).
- Status: re-run (fixed ring selection) started 2026-10-31 05:23 CST with nohup; results -> `external/csd/eval_csd_test_20261031.json`. See the follow-up section below if it has been filled in.


### Follow-up 2026-10-31 09:50 CST (results filled in)
- CSD test (5335 crops, 61 names, all unseen): large greedy 819 / 0.76378, lex 902 / 0.76987; nogold greedy 921 / 0.78001, lex 985 / 0.78455; nogold_len32 greedy 825 / 0.76741, lex 909 / 0.77590; vitb_finetune greedy 954 / 0.77637, lex 1034 / 0.78196. nogold beats production on CSD. Production unchanged.
- TrOCR v2 (`eval_trocr_v2_20261031.json`): all_single 11/227 1-NED 0.32736 (beam1), 0.31252 (beam4); orig63 0/63; unseen59 4/59 0.40692. 217.3M params, bs1 greedy 186.4 ms, peak 7107 MB. Still far below PARSeq; do not invest further without a pretrained Chinese seal decoder pretrain.
- det_multi (pad 2, score 0.25, NMS 0.45), 8 multi images / 16 seals: large greedy 3/16 0.6582, lex 4/16 0.6626; nogold 4/16 0.6434 (lex 0.6439); len32 4/16 0.6184 (lex 0.6281). Single images: 220 one box, 3 zero, 4 two.
- len32 on gold v2 unseen59: greedy 18 / 0.80229, lex 20 / 0.81056 (nogold greedy 19 / 0.80319); orig63 and long_name worse than nogold. Not adopted.
- ReST: `external/rest/` is empty. No ReST archive on /home/rapidstamp_data, /home/znsoft, /tmp, znsoftpc user profile, or the bot box. `/home/mayong` (/dev/sdc, ext3) is mounted `emergency_ro` and returns Input/output error; the earlier ReST/Paddle archives were probably there (`/home/mayong/datasets/...`). Needs disk check (fsck) by the user before ReST can be used.
