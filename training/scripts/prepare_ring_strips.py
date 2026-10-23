"""Filter web-seal labels and write unwrapped ring strips.

Gold test crops from the user-seal manifest are excluded. The validation
split is a stable 3% of what remains, by the sha256 of the file name.

    python training/scripts/prepare_ring_strips.py \
        --labels /home/rapidstamp_data/web_seals_cn_all/labels.jsonl \
        --crops /home/rapidstamp_data/web_seals_cn_all/crops \
        --manifest training/manifests/user_seals_split.jsonl \
        --dict /home/rapidstamp_data/pretrained/ppocr_keys_v1.txt \
        --out /home/rapidstamp_data/ring_baseline
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.ring_text import (  # noqa: E402
    dedupe_by_md5,
    file_digests,
    is_gold,
    label_reason,
    load_charset,
    load_gold,
    load_rgb,
    pixel_md5,
    ring_label,
    ring_probability,
    strip_name,
    unsupported_chars,
    unwrap_ring,
    val_split,
)


def _examples(bucket: dict[str, list[str]], reason: str, text: str) -> None:
    items = bucket.setdefault(reason, [])
    if len(items) < 8 and text not in items:
        items.append(text)


def save_sample_sheet(samples: list[dict], dest: Path, columns: int = 4) -> None:
    crop_side = 128
    strip_h = 36
    cell_w = 168
    header = 16
    pad = 6
    rows = max(1, (len(samples) + columns - 1) // columns)
    cell_h = header + crop_side + strip_h + pad * 3
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for index, sample in enumerate(samples):
        col = index % columns
        row = index // columns
        x = col * cell_w + pad
        y = row * cell_h + pad
        crop = Image.fromarray(sample["crop"]).resize((crop_side, crop_side), Image.BILINEAR)
        strip = Image.fromarray(sample["strip"]).resize((cell_w - 2 * pad, strip_h), Image.BILINEAR)
        title = sample["text"][:14]
        draw.text((x, y), title, fill=(20, 20, 20))
        sheet.paste(crop, (x, y + header))
        sheet.paste(strip, (x, y + header + crop_side + pad))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gold = load_gold(args.manifest)
    charset = load_charset(args.dict)
    reasons: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    rows = []
    with args.labels.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            reason = label_reason(row)
            if reason is None:
                reasons["label_keep"] += 1
            else:
                reasons[reason] += 1
                _examples(examples, reason, ring_label(row) or str(row.get("file")))

    pending = []
    for index, row in enumerate(rows):
        if label_reason(row) is not None:
            continue
        name = Path(str(row["file"])).name
        path = args.crops / name
        if not path.is_file():
            reasons["missing"] += 1
            continue
        file_md5, file_sha = file_digests(path)
        rgb = load_rgb(path)
        pixels = pixel_md5(rgb)
        if is_gold(name, file_sha, pixels, gold):
            reasons["gold"] += 1
            _examples(examples, "gold", name)
            continue
        text = ring_label(row)
        missing = unsupported_chars(text, charset)
        if missing:
            reasons["dict"] += 1
            _examples(examples, "dict", text)
            continue
        pending.append(
            {
                "file": name,
                "text": text,
                "prob": float(ring_probability(row) or 0.0),
                "file_md5": file_md5,
                "det_class": row.get("det_class"),
                "path": str(path),
            }
        )
        if index and index % 2000 == 0:
            print(f"hashed {index}/{len(rows)}", flush=True)

    reasons["before_dedupe"] = len(pending)
    kept = dedupe_by_md5(pending)
    reasons["deduped"] = reasons["before_dedupe"] - len(kept)

    out = args.out
    for split in ("train", "val"):
        (out / "strips" / split).mkdir(parents=True, exist_ok=True)
    lists = {"train": [], "val": []}
    kept_rows = []
    sheet_pool = []
    gaps = []
    radii = []
    for index, row in enumerate(kept):
        rgb = load_rgb(Path(row["path"]))
        try:
            unwound = unwrap_ring(rgb)
        except Exception as exc:
            reasons["unwrap_error"] += 1
            _examples(examples, "unwrap_error", f"{row['file']}: {exc}")
            continue
        split = val_split(row["file"])
        name = strip_name(row["file"], bool(unwound["wide_gap"]))
        dest = out / "strips" / split / name
        Image.fromarray(unwound["strip"]).save(dest)
        lists[split].append(f"{dest}\t{row['text']}")
        record = {
            "file": row["file"],
            "strip": name,
            "split": split,
            "text": row["text"],
            "prob": row["prob"],
            "det_class": row["det_class"],
            "file_md5": row["file_md5"],
            "start_deg": unwound["start_deg"],
            "gap_deg": unwound["gap_deg"],
            "r_in": unwound["r_in"],
            "r_out": unwound["r_out"],
            "wide_gap": unwound["wide_gap"],
        }
        kept_rows.append(record)
        gaps.append(unwound["gap_deg"])
        radii.append(unwound["r_in"])
        if len(sheet_pool) < 64:
            sheet_pool.append({"crop": rgb, "strip": unwound["strip"], "text": row["text"], "wide": unwound["wide_gap"]})
        if index and index % 500 == 0:
            print(f"unwrapped {index}/{len(kept)}", flush=True)

    for split, lines in lists.items():
        text = "\n".join(lines)
        (out / f"{split}_list.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
    (out / "kept.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept_rows),
        encoding="utf-8",
    )

    rng = random.Random(args.seed)
    wide = [item for item in sheet_pool if item["wide"]]
    narrow = [item for item in sheet_pool if not item["wide"]]
    rng.shuffle(wide)
    rng.shuffle(narrow)
    chosen = (wide[:8] + narrow[:8])[:16]
    if len(chosen) < 16:
        chosen_ids = {id(item) for item in chosen}
        rest = [item for item in sheet_pool if id(item) not in chosen_ids]
        rng.shuffle(rest)
        chosen.extend(rest[: 16 - len(chosen)])
    sheet_path = out / "sample_sheet.jpg"
    if chosen:
        save_sample_sheet(chosen, sheet_path)

    gap_arr = np.asarray(gaps, dtype=np.float32) if gaps else np.zeros(0, dtype=np.float32)
    rad_arr = np.asarray(radii, dtype=np.float32) if radii else np.zeros(0, dtype=np.float32)
    meta = {
        "rows": len(rows),
        "reasons": dict(reasons),
        "examples": examples,
        "gold_test_crops": len(gold["pixel_md5"]),
        "train": len(lists["train"]),
        "val": len(lists["val"]),
        "wide_gap": int(sum(1 for row in kept_rows if row["wide_gap"])),
        "gap_deg_p50": float(np.median(gap_arr)) if gap_arr.size else None,
        "r_in_p50": float(np.median(rad_arr)) if rad_arr.size else None,
        "sample_sheet": str(sheet_path) if chosen else "",
        "dict": len(charset),
        "min_prob": 0.90,
        "max_chars": 25,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": meta["train"], "val": meta["val"], "reasons": meta["reasons"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
