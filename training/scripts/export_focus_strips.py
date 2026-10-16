"""Symlink recognition strips for classes other than round official seals.

Paddle pages are almost all Chinese circles. Repeating these strips keeps
square, finance, invoice, and Latin seals in the training mix.

    python training/scripts/export_focus_strips.py \
        --src /home/mayong/rapidstamp/synth \
        --out /home/mayong/rapidstamp/focus_strips
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

FOCUS = {
    "oval",
    "square_personal",
    "finance",
    "invoice",
    "company_latin",
    "notary",
    "rect_stamp",
}


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def export_focus(src: Path, out: Path) -> dict:
    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        rows = []
        label_dir = src / "det" / split / "labels"
        rec_dir = src / "rec" / split / "images"
        if not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.json")):
            meta = json.loads(label_path.read_text(encoding="utf-8"))
            for ordinal, seal in enumerate(meta.get("seals") or []):
                if seal.get("name") not in FOCUS:
                    continue
                name = f"{label_path.stem}_{ordinal}.png"
                image = rec_dir / name
                if not image.is_file():
                    continue
                text = str(seal.get("text") or "")
                if not text:
                    continue
                _symlink(image, out / "rec" / split / "images" / name)
                rows.append(f"{name}\t{text}")
                counts[split] += 1
        tsv = out / "rec" / split / "labels.tsv"
        tsv.parent.mkdir(parents=True, exist_ok=True)
        tsv.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(json.dumps(counts), flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export_focus(args.src, args.out)


if __name__ == "__main__":
    main()
