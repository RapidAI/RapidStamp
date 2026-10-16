"""Convert an ICDAR 2023 ReST tree into RapidStamp detection labels.

ReST is distributed from the Robust Reading Competition after registration.
This script does not download it. Point ``--src`` at a directory of images
with sibling ``gt_*.json`` (or same-stem json) polygon annotations. The
output is a detection split whose text polygons are converted to upright
boxes. Recognition polygons are not unwrapped here; use them as a fine-tune
source after a manual license check.

Expected ReST json (the competition portal is slightly inconsistent)::

    {"gt_1": [{"points": [[x, y], ...], "transcription": "text"}]}
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _items(payload: dict) -> list[dict]:
    if isinstance(payload, list):
        return payload
    items = []
    for value in payload.values():
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict) and "points" in value:
            items.append(value)
    return items


def _box(points: list) -> list[int]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1]


def convert(src: Path, dst: Path) -> int:
    images = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        images.extend(src.rglob(pattern))
    count = 0
    out_images = dst / "images"
    out_labels = dst / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images):
        candidates = [
            image_path.with_suffix(".json"),
            image_path.with_name(f"gt_{image_path.stem}.json"),
            image_path.parent.parent / "json" / f"gt_{image_path.stem}.json",
        ]
        label_path = next((path for path in candidates if path.is_file()), None)
        if label_path is None:
            continue
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        seals = []
        for item in _items(payload):
            points = item.get("points") or item.get("polygon")
            text = item.get("transcription") or item.get("text") or ""
            if not points:
                continue
            seals.append(
                {
                    "cls": 0,
                    "name": "round_official",
                    "bbox": _box(points),
                    "text": text,
                    "center_text": "",
                    "layout": "circle",
                    "rows": 1,
                    "cols": 1,
                    "note": "ReST annotates title text, not the seal instance. The box is the text polygon's AABB.",
                }
            )
        stem = f"{count:06d}"
        shutil.copy2(image_path, out_images / f"{stem}{image_path.suffix.lower()}")
        (out_labels / f"{stem}.json").write_text(
            json.dumps({"width": None, "height": None, "seals": seals, "source": str(image_path)}, ensure_ascii=False),
            encoding="utf-8",
        )
        count += 1
    print(f"converted {count} ReST images into {dst}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
