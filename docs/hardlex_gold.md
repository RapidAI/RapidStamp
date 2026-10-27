# Gold errors, directed augmentation, and name-lexicon decode

Measured 2026-10-27 on `checkpoints/parseq_reader_vl1/finetune/best.pt` (step 100,000). That checkpoint has the better gold 1−NED of vl1 and mix2: greedy **25/63 exact, 1−NED 0.91142** (verified 25/57, 0.92416; pending 0/6, 0.79042). Spaces count. Nothing was stripped.

## What the 38 misses look like

32 of the misses are user-verified. Six are pending recheck, and several of those gold strings are incomplete, so they were not used to build the hard set.

| Kind | Count |
| --- | --- |
| One substitution | 19 |
| Two substitutions | 10 |
| Three substitutions | 3 |
| Four or more substitutions | 2 |
| One extra character | 1 |
| Mixed insert and substitute | 2 |

The verified misses are same-length character swaps in the name stem (`升`/`开`, `杭`/`扬`, `划`/`创`, `北`/`江`, `华`/`毕`). On 51 substitution sites, the gold character was the runner-up 24 times and inside the top 5 34 times. None of the 32 verified gold strings are in `synth/names.txt` (1,387,239 suffix-shaped names). A nearest-name snap cannot recover them.

## Lexicon rescore

`training/rapidstamp_train/lexicon_decode.py` builds a character trigram from `synth/names.txt` and leaves the organization suffix frozen. It may flip up to three stem characters whose top-two margin is under `tau`, and only toward the runner-up. Gold strings are excluded from the trigram counts. The ONNX export stays greedy.

| Decode | Exact | 1−NED | Verified | Pending |
| --- | --- | --- | --- | --- |
| vl1 greedy | 25/63 | 0.91142 | 25/57, 0.92416 | 0/6, 0.79042 |
| vl1, alpha 0.4, tau 4 | **28/63** | **0.91516** | 28/57, 0.93097 | 0/6, 0.76499 |

Alpha 0.4 and tau 4 tied tau 8 and beat the other grid points. Higher alpha (1.5 and 3) lost exact matches. The three fixes are image 012 `升`→`开`, 046 `划`→`创`, and 062 `北`→`江`. Some other strings moved and stayed wrong. Raw grid: `/home/rapidstamp_data/logs/lexicon_vl1.json`.

## Directed augmentation

Hard rows are train crops only. Verified confusion characters and 4-grams of the verified gold strings selected 1,693 train rows (`ring_v2/hard_vl1.jsonl`). Gold crops were not copied. Pending gold strings were not used. Half of the real samples in a 1-epoch fine-tune used `augment_hard`: full-turn rotation, stronger blur, stain, occlusion, and overlapping text. Init was vl1 `best.pt`. Output is `checkpoints/parseq_reader_hardlex/`. Step target `int(1 * 20549 / 0.55)` = 37,361. The run stopped at seen 37,376. Pid 47692 has exited.

Selected weights are `best.pt` at step 25,024: val 303/665, 1−NED 0.83532; gold **27/63**, 1−NED **0.90308** (verified 27/57, 0.92288; pending 0/6, 0.71499). `last.pt` at step 37,376 was not selected (val 300/665; gold 27/63, 1−NED 0.90546). Exact is 2 above the prior 25/63. Gold 1−NED is below vl1's 0.91142, and val exact fell from 313 to 303. The same lexicon settings on this checkpoint stay at 27/63 and raise 1−NED only to 0.90935.

Keep vl1 `best.pt` for greedy 1−NED (25/63). Use the alpha 0.4 / tau 4 rescore on that checkpoint when the extra exact matches matter (28/63, 0.91516). A later run, `checkpoints/parseq_reader_large/`, keeps this ViT-S and raises the input to 512. Quote these two numbers until that run beats them.
