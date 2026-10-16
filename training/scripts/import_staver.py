"""Convert StaVer pixel masks into seal bounding boxes.

StaVer (Micenková & van Beusekom, ICDAR 2011) is a 400-page invoice scan set
with stamp masks. It has no SPDX license that this repository could verify;
do not redistribute the scans. Pass ``--scans`` and ``--masks`` only after
you have obtained the data yourself.

Without scipy the script emits one box around all foreground pixels. With
scipy it splits connected components so a page can contain several stamps.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    try:
        from scipy import ndimage
    except ImportError:
        return [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)]
    labels, count = ndimage.label(mask > 0)
    boxes = []
    for index in range(1, count + 1):
        cy, cx = np.where(labels == index)
        if len(cx) < 30:
            continue
        boxes.append((int(cx.min()), int(cy.min()), int(cx.max()) + 1, int(cy.max()) + 1))
    return boxes


def convert(scans: Path, masks: Path, dst: Path) -> int:
    out_images = dst / "images"
    out_labels = dst / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        [path for pattern in ("*.png", "*.jpg", "*.tif", "*.tiff") for path in scans.glob(pattern)]
    )
    count = 0
    for image_path in paths:
        mask_path = None
        for candidate in masks.glob(image_path.stem + ".*"):
            mask_path = candidate
            break
        if mask_path is None:
            continue
        mask = np.asarray(Image.open(mask_path).convert("L"))
        boxes = _components(mask)
        seals = [
            {
                "cls": 7,
                "name": "rect_stamp",
                "bbox": [x1, y1, x2, y2],
                "text": "",
                "center_text": "",
                "layout": "hline",
                "rows": 1,
                "cols": 1,
                "note": "StaVer has no transcription. Box only.",
            }
            for x1, y1, x2, y2 in boxes
        ]
        stem = f"{count:06d}"
        shutil.copy2(image_path, out_images / f"{stem}{image_path.suffix.lower()}")
        (out_labels / f"{stem}.json").write_text(
            json.dumps({"seals": seals, "source": image_path.name}, ensure_ascii=False),
            encoding="utf-8",
        )
        count += 1
    print(f"converted {count} StaVer pages into {dst}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()
    convert(args.scans, args.masks, args.dst)


if __name__ == "__main__":
    main()
