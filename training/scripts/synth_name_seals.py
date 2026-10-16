"""Square name-seal strips from the open charset, including yin and yang cuts.

    python training/scripts/synth_name_seals.py \
        --charset models/charset.txt \
        --out /home/mayong/rapidstamp/name_seals \
        --train 12000 --val 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.fonts_util import find_cjk_font, find_latin_font  # noqa: E402
from rapidstamp_train.realdata import read_charset_file  # noqa: E402
from rapidstamp_train.synthesize import FontBank, render_name_seal, sample_name, strip_from_crop  # noqa: E402


def _drawable(chars: list[str], font_path: str) -> list[str]:
    font = ImageFont.truetype(font_path, 40)
    kept = []
    for ch in chars:
        if ord(ch) < 128:
            continue
        bbox = font.getbbox(ch)
        if bbox[2] - bbox[0] > 2 and bbox[3] - bbox[1] > 2:
            kept.append(ch)
    return kept


def _write_split(out: Path, split: str, count: int, chars: list[str], seed: int) -> int:
    if count <= 0:
        return 0
    fonts = FontBank(find_cjk_font(), find_latin_font())
    rng = np.random.default_rng(seed)
    image_dir = out / "rec" / split / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(count):
        text, grid_rows, grid_cols = sample_name(rng, chars)
        rgba = render_name_seal(rng, fonts, text, grid_rows, grid_cols)
        rgb = np.asarray(Image.fromarray(rgba, "RGBA").convert("RGB"))
        strip = strip_from_crop(rgb, "grid", grid_rows, grid_cols, 0.0)
        name = f"{index:06d}.png"
        Image.fromarray(strip, "RGB").save(image_dir / name)
        rows.append(f"{name}\t{text}")
        if index % 500 == 0:
            print(f"{split} {index}/{count}", flush=True)
    tsv = out / "rec" / split / "labels.tsv"
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--charset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=12000)
    parser.add_argument("--val", type=int, default=400)
    args = parser.parse_args()
    chars = _drawable(read_charset_file(args.charset), find_cjk_font())
    if len(chars) < 100:
        raise SystemExit(f"only {len(chars)} drawable CJK characters")
    print(f"drawable {len(chars)}", flush=True)
    train_n = _write_split(args.out, "train", args.train, chars, 11)
    val_n = _write_split(args.out, "val", args.val, chars, 19)
    print(f"name seals train {train_n} val {val_n}", flush=True)


if __name__ == "__main__":
    main()
