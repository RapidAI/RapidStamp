"""Contact sheet of synthetic seals, with real crops on the lower rows.

    python training/scripts/render_seal_sheet.py \
        --fonts /home/rapidstamp_data/pretrained/fonts \
        --names /home/rapidstamp_data/synth/names.txt \
        --real /home/rapidstamp_data/gold/seals_to_label \
        --out /home/rapidstamp_data/synth/sample_sheet.jpg
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.direct_aug import pad_square  # noqa: E402
from rapidstamp_train.seal_render import FontBank, list_fonts, render_seal  # noqa: E402


def _tile(image: Image.Image, caption: str, side: int, font) -> Image.Image:
    canvas = Image.new("RGB", (side, side + 36), (255, 255, 255))
    thumb = pad_square(image).resize((side, side), Image.BILINEAR)
    canvas.paste(thumb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, side + 4), caption[:18], fill=(0, 0, 0), font=font)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    fonts = FontBank(list_fonts(args.fonts))
    names = [line.strip() for line in args.names.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    picked = [names[rng.randrange(len(names))] for _ in range(8)]
    picked[0] = "杭州市西湖区市场监督管理局" if len("杭州市西湖区市场监督管理局") <= 25 else picked[0]
    tiles = []
    caption_font = fonts.font(0, 16)
    for index, text in enumerate(picked):
        # The top row stays upright so the ring layout can be compared with real crops.
        image = render_seal(text, random.Random(rng.randrange(1 << 30)), fonts, spin=index >= 4)
        tiles.append(_tile(image, text, 280, caption_font))
    real_paths = sorted(path for path in args.real.glob("*.png"))[:8]
    for path in real_paths:
        with Image.open(path) as image:
            tiles.append(_tile(image.convert("RGB"), "real " + path.name[:14], 280, caption_font))
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 280, rows * 316), (240, 240, 240))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 280, (index // columns) * 316))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out, quality=85)
    print(f"wrote {args.out} {sheet.size}", flush=True)


if __name__ == "__main__":
    main()
