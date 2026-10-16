"""Read seal crops directly, without the page detector.

Each image is one round seal. The published recognizer unwraps it at eight
starts. Draft text is not gold until a person checks it.

    python training/scripts/read_seal_crops.py \
        --data /home/mayong/rapidstamp/real/user_seals_test \
        --models /home/mayong/rapidstamp/models_pub \
        --out /home/mayong/rapidstamp/logs/user-seal-drafts.jsonl
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

from rapidstamp.pipeline import RapidStamp  # noqa: E402


def read_tree(data: Path, models: Path, out: Path, strips: Path | None) -> None:
    engine = RapidStamp(models)
    label_dir = data / "det" / "val" / "labels"
    image_dir = data / "det" / "val" / "images"
    rows = []
    if strips is not None:
        strips.mkdir(parents=True, exist_ok=True)
    for label_path in sorted(label_dir.glob("*.json")):
        meta = json.loads(label_path.read_text(encoding="utf-8"))
        rgb = np.asarray(Image.open(image_dir / f"{label_path.stem}.jpg").convert("RGB"))
        seal = meta["seals"][0]
        x1, y1, x2, y2 = (int(v) for v in seal["bbox"])
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            crop = rgb
        text, score = engine._read(crop, "circle")
        row = {
            "stem": label_path.stem,
            "file": meta.get("file", ""),
            "draft": text,
            "score": round(float(score), 4),
            "gold": "",
            "checked": False,
        }
        rows.append(row)
        if strips is not None and text:
            # The winning strip is the one whose decoded text was chosen.
            # Save the crop too, so a later check is not only the ring text.
            Image.fromarray(crop, "RGB").save(strips / f"{label_path.stem}.jpg", quality=90)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strips", type=Path, default=None)
    args = parser.parse_args()
    read_tree(args.data, args.models, args.out, args.strips)


if __name__ == "__main__":
    main()
