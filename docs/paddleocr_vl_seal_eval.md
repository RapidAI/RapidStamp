# PaddleOCR-VL-1.6 on the 63 gold seals

Measured 2026-10-27. The model is PaddleOCR-VL-1.6 only (`pipeline_version="v1.6"`, package pipeline name `PaddleOCR-VL-1.6`, weights `PaddleOCR-VL-1.6-0.9B`). Version 1.5 was not run.

The seal task sends each gold crop with layout detection off and `prompt_label="seal"`. That sets `use_seal_recognition=True` and uses the prompt `Seal Recognition:`. The returned block label is `seal`. Inference was about 0.5 s per crop on the RTX 6000 Ada after the weights were loaded. mix3 had already exited, so the GPU was free.

## Scores

Gold strings are `seals[0].ring`. Spaces count. Nothing inside a line was stripped.

The seal block is usually the ring on the first line and a code, or a second phrase, on the following lines. Aliyun's 55/63 number is the ring field, so the comparable string is the first line. The full block, newlines included, is reported as well.

| Reader | Exact | 1−NED | Verified (57) | Pending recheck (6) |
| --- | --- | --- | --- | --- |
| PaddleOCR-VL-1.6, full seal block | 10/63 | 0.60481 | 10/57, 0.58182 | 0/6, 0.82317 |
| PaddleOCR-VL-1.6, first line (ring) | **46/63** | **0.96565** | 46/57, 0.97670 | 0/6, 0.86067 |
| PARSeq mix3 `best.pt` (step 106,816) | 25/63 | 0.90375 | 25/57, 0.92155 | 0/6, 0.73469 |
| PARSeq mix2 `best.pt` | 25/63 | 0.90795 | 25/57, 0.93060 | 0/6, 0.69276 |
| Aliyun stamp-OCR draft | 55/63 | 0.979 | — | — |

The 17 first-line misses are a one-character swap, a dropped character, or a short dropped phrase (`太`/`大`, `有限`/`有`, `扬`/`杭`, a missing `昇`, a dropped `银川市`). Pending rows 014, 027, 031, 039, 040, and 041 are 0/6. Several of those gold strings are themselves incomplete, so that split stays separate from the 46/57 verified score.

Example of the full block versus the ring line, image 007: first line `北京市门头沟区发展和改革委员会` matches gold, and the block continues with `1101090210750`. Joining both lines is not an exact ring match.

## Recommendation

Use PaddleOCR-VL-1.6 for new unlabeled seal crops. Take the first line as the ring and keep later lines as other text. The ring line is 46/63 exact and 1−NED 0.96565, which clears the 45/63 and 0.95 bar. Aliyun is still ahead on exact match (55/63, 0.979), so do not replace labels Aliyun already wrote.

The 100 `no_seal_0` detector rejects and the 50 `det_seal_0` detector rejects are labeled in `/home/rapidstamp_data/third/vl_unlabeled/labels.jsonl` (150 lines: 139 rings of length 2–25, 1 empty, 8 length 1, 2 longer than 25). They are not in `kept.jsonl`. Batch6's filtered set is already Aliyun-labeled (1,037 of 1,037). The other 2,406 batch6 crops failed the ink, confidence, or shape filter and were not sent to VL.

Raw responses are `/home/rapidstamp_data/gold/paddleocr_vl_eval/`. The runner is `/home/rapidstamp_data/venvs/paddleocr_vl` (Paddle 3.2.1 GPU, paddleocr 3.7.0). It is separate from the training env and from the Paddle 2.6.2 env.
