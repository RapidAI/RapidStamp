# OXR-1.0 on the 63 gold seals

Measured 2026-10-27. SoMark OXR-1.0 (Optical Everything Recognition) was called on the 63 proofread gold crops. The question was whether its Stamp text is accurate enough to label new seals. The labeling bar used here is gold exact at least 45/63, or 1−NED at least 0.95. OXR is below that bar. Keep Aliyun for unlabeled seals.

## How it was called

OXR is not installed on the training machine. A local run needs an NVIDIA GPU with about 24 GB and a vLLM plugin, which would have sat on the same RTX 6000 Ada as the mix fine-tune, so this eval used the free HTTP API only.

| Item | Value |
| --- | --- |
| API | `POST https://free-oxr.somark.cn/api/v1/parse/async`, then `parse/async_check` |
| Auth | `api_key=dummy` (the free API does not check the key) |
| Output | `output_formats=json` |
| Images | `/home/rapidstamp_data/gold/seals_to_label/` (63 png, 94 KB–416 KB) |
| Labels | `seals[0].ring` in `training/manifests/user_seals_split.jsonl` (63 `split=test` rows). Spaces count. Nothing was stripped. |
| Pace | One task at a time. A second submit while a task is in progress returns code 1133. About 2 seconds between finished calls. |

`https://free-oxr.somark.ai/api/v1/parse/sync` returned Cloudflare HTTP 524 after 130 seconds. Sync on the `.cn` mirror then sat with no bytes for 120 seconds and left that client slot busy. Async on `free-oxr.somark.cn` is what completed. All 63 batch calls returned `status=SUCCESS`. Server-side `processing_time_ms` had median 230, mean 376, max 5,723. The queue made the wall clock about 58 minutes (09:09Z–10:08Z).

Raw JSON is under `/home/rapidstamp_data/gold/oxr_eval/raw/`. Per-image scores are `gold/oxr_eval/scored.json`. Those files stay off git.

## Scores

Headline 1−NED is the mean of `1 − edit_distance / max(len(pred), len(gold))`. Both strings empty scores 1. Spaces and newlines count.

Stamp text is the Stamp blocks' plain `content` joined in block order, with no extra separator. A Stamp whose content is an image markdown link counts as empty. The longest Stamp line is the longest newline-separated line inside those Stamp blocks. That second rule is the one a labeler could apply without looking at the gold string: several responses put the ring on one line and a seal code on the next. An oracle that picks the Stamp string closest to gold does not change the exact count.

| Reader | Exact | 1−NED | Verified (57) | Pending recheck (6) |
| --- | --- | --- | --- | --- |
| OXR, Stamp text joined | 0/63 | 0.08984 | 0/57, 0.09930 | 0/6, 0.00000 |
| OXR, longest Stamp line | 1/63 | 0.11131 | 1/57, 0.12303 | 0/6, 0.00000 |
| OXR, every plain-text block joined | 0/63 | 0.13576 | 0/57, 0.13183 | 0/6, 0.17314 |
| PARSeq mix2 `best.pt` (step 400,000) | 25/63 | 0.90795 | 25/57, 0.93060 | 0/6, 0.69276 |
| PARSeq first mix `best.pt` | 25/63 | 0.90006 | — | — |
| Aliyun stamp-OCR draft (historical) | 55/63 | 0.979 | — | — |

Pending rows are 014, 027, 031, 039, 040, and 041. None of them produced Stamp text. The published class-routed CTC reader remains 0/63, 1−NED 0.050, and is not a labeling candidate.

Block types across the 63 responses: Picture 51, Stamp 50, Text 34, Title 2. A Stamp element appeared on 24 images, and 23 of those had some plain Stamp text. 17 images came back as Picture only. One image (052) came back with an empty block list. 21 images had Text or Title and no Stamp text. One Stamp line matched gold exactly (007, `北京市门头沟区发展和改革委员会`). The same response also contained the fragments `大沟区发展和改` and `京门区`, and the full block continues with a newline and `1101090210750`, so joining the Stamp field misses the exact match.

## What the errors look like

Tight crops are often a Picture with no text. Image 001 (`新都区石板滩镇兴太种子经营部`) is the whole response as one Picture. Image 006 (`北京市门头沟区生态环境局`) is the same.

When OXR does emit a Stamp line, characters still drop or swap. Image 016 gold `宁夏昇禾源智能化信息科技有限公司` came back as `宁夏禾源智能化信息科技有限公司` (1−NED 0.938). Image 015, the same company on the other crop, came back as `宁夏草原智能化信息科技股份有限公司` (昇禾源 became 草原, 有限 became 股份). Image 012 gold `扬州开源土地规划设计有限公司` came back as `州开源土地规划设计有` plus the extra fragments `乙方：`, `三`, and `签章`.

Other images received Text that is not the ring. Image 002 gold is `湖北尚一特酒店管理有限公司`. The two Text blocks were `承诺按合同约定的条件、时` and `按照监理人指示开工，工具`. There was no Stamp block. Those strings are not a usable label.

Image 003 was submitted twice, about two minutes apart. The first Text string and the scored batch string differed. The scored row is the batch call. The free API is not stable on a repeat.

## Recommendation

Keep Aliyun for labeling unlabeled seals. OXR's best non-oracle Stamp rule is 1/63 exact and 1−NED 0.11131. That is under the 45/63 and 0.95 bar, under the current reader (25/63, 0.90795), and under the historical Aliyun draft (55/63, 0.979).

Do not add an OXR batch-label pipeline. Do not mass-relabel `no_seal_0` or the rest of the unlabeled pool with this API. The free service is one task at a time, queued, and the Stamp field is missing or wrong on almost every gold crop.

A later self-hosted OXR would still have to beat Aliyun on these 63 crops before it replaces the labeler. That run needs its own GPU time, after the reader work, and it is not justified by these API scores.
