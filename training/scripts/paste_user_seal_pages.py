"""Paste the user-seal training crops onto pages at several scales.

The 63-image test split is not read. Full-frame copies are the case the
published detector misses. Images stay on the training machine.

    python training/scripts/paste_user_seal_pages.py \
        --src /home/mayong/rapidstamp/real/user_seals \
        --out /home/mayong/rapidstamp/real/user_pages
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.user_seals import PASTE_SCALES, pasted_page  # noqa: E402


def build(src: Path, out: Path, page: int) -> dict:
    label_dir = src / "det" / "train" / "labels"
    image_dir = src / "det" / "train" / "images"
    dest_images = out / "det" / "train" / "images"
    dest_labels = out / "det" / "train" / "labels"
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    written = 0
    for label_path in sorted(label_dir.glob("*.json")):
        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.is_file():
            continue
        meta = json.loads(label_path.read_text(encoding="utf-8"))
        seals = meta.get("seals") or []
        if not seals:
            continue
        crop = np.asarray(Image.open(image_path).convert("RGB"))
        rng = np.random.default_rng(int(label_path.stem[:8], 16))
        for ordinal, scale in enumerate(PASTE_SCALES):
            _placed, box = pasted_page(crop, seals[0]["bbox"], scale, page=page, origin=(0, 0))
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            left = int(rng.integers(0, max(1, page - box_w + 1)))
            top = int(rng.integers(0, max(1, page - box_h + 1)))
            placed, box = pasted_page(crop, seals[0]["bbox"], scale, page=page, origin=(left, top))
            stem = f"{label_path.stem}_s{ordinal}"
            Image.fromarray(placed, "RGB").save(dest_images / f"{stem}.jpg", quality=90)
            row = {
                "width": int(placed.shape[1]),
                "height": int(placed.shape[0]),
                "source": "user_seals_paste",
                "kind": "isolated_seal",
                "file": meta.get("file", ""),
                "scale": scale,
                "seals": [
                    {
                        "cls": int(seals[0]["cls"]),
                        "name": seals[0]["name"],
                        "bbox": [int(v) for v in box],
                        "text": "",
                        "layout": seals[0].get("layout", "circle"),
                        "rows": 1,
                        "cols": 1,
                    }
                ],
            }
            (dest_labels / f"{stem}.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
            written += 1
    stats = {"pages": written, "scales": list(PASTE_SCALES)}
    print(json.dumps(stats), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--page", type=int, default=640)
    args = parser.parse_args()
    build(args.src, args.out, args.page)


if __name__ == "__main__":
    main()
