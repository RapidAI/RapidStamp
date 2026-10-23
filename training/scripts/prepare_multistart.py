"""Unwrap a few start angles for strips whose gap is not obvious.

Training still uses one strip per crop. This list is only for inference:
each copy is decoded, and pick_best_start.py keeps the highest CTC score.

    python training/scripts/prepare_multistart.py \
        --kept /home/rapidstamp_data/ring_v2/kept.jsonl \
        --crops /home/rapidstamp_data/web_seals_cn_all/crops \
        --out /home/rapidstamp_data/ring_v2/multistart
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.ring_text import inference_starts, load_rgb, unwrap_ring  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kept", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    dest_dir = args.out / "strips"
    dest_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    rows = 0
    copies = 0
    with args.kept.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.split and row.get("split") != args.split:
                continue
            path = args.crops / row["file"]
            if not path.is_file():
                continue
            rgb = load_rgb(path)
            unwound = unwrap_ring(rgb, roi=row.get("roi"), angle=row.get("angle"))
            starts = inference_starts(
                unwound["start_deg"],
                unwound["confidence"],
                unwound["gap_deg"],
                source=unwound["source"],
            )
            stem = Path(row["strip"]).stem
            rows += 1
            for start in starts:
                copy = unwrap_ring(rgb, roi=row.get("roi"), angle=float(start))
                name = f"{stem}__s{start:.0f}.png"
                Image.fromarray(copy["strip"]).save(dest_dir / name)
                lines.append(f"{dest_dir / name}\t{row['text']}")
                copies += 1
            if rows % 50 == 0:
                print(f"multistart {rows} crops, {copies} strips", flush=True)
    text = "\n".join(lines)
    list_path = args.out / "list.txt"
    list_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    print(json.dumps({"crops": rows, "strips": copies, "list": str(list_path)}), flush=True)


if __name__ == "__main__":
    main()
