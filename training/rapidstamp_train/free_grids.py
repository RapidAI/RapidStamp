"""Square name-seal strips drawn from the whole CJK charset.

The closed surname list taught the recognizer to swap one personal name for
another. These grids use the same cell unroll as inference, with characters
sampled from the recognizer charset.
"""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rapidstamp.geometry import grid_unroll
from rapidstamp.labels import REC_H, REC_W
from rapidstamp_train.corpus import contains_cjk, personal_name
from rapidstamp_train.fonts_util import find_cjk_font, find_latin_font
from rapidstamp_train.realdata import read_charset_file
from rapidstamp_train.synthesize import (
    FontBank,
    _blit,
    _compose,
    _degrade_strip,
    _export_pythonpath,
    _render_glyph,
    _rotate_rgba,
    _tight_bbox,
)

_LAYOUTS = ((2, 1), (3, 1), (4, 1), (2, 2))
_FONTS: FontBank | None = None


def sample_grid(rng: np.random.Generator, cjk: list[str]) -> tuple[str, int, int]:
    if rng.random() < 0.2:
        return personal_name(rng)
    rows, cols = _LAYOUTS[int(rng.integers(0, len(_LAYOUTS)))]
    count = rows * cols
    text = "".join(str(ch) for ch in rng.choice(cjk, size=count))
    return text, rows, cols


def render_grid_seal(rng: np.random.Generator, fonts: FontBank, text: str, rows: int, cols: int) -> np.ndarray:
    side = int(rng.integers(180, 280))
    canvas = Image.new("L", (side, side), 0)
    draw = ImageDraw.Draw(canvas)
    margin = int(side * 0.06)
    border = max(4, side // 28)
    draw.rectangle([margin, margin, side - margin, side - margin], outline=255, width=border)
    inner = margin + border + int(side * 0.03)
    cell_w = (side - 2 * inner) / cols
    cell_h = (side - 2 * inner) / rows
    font_size = int(min(cell_w, cell_h) * 0.72)
    for row in range(rows):
        for col in range(cols):
            ch = text[row * cols + col]
            glyph = _render_glyph(fonts.for_char(ch, font_size), ch, scale_x=0.72 if contains_cjk(ch) else 1.0)
            cx = inner + (col + 0.5) * cell_w
            cy = inner + (row + 0.5) * cell_h
            canvas = _blit(canvas, glyph, cx, cy)
    ink = (int(rng.integers(165, 210)), int(rng.integers(16, 45)), int(rng.integers(16, 40)))
    return _compose(canvas, ink, rng, round_shape=False, margin=margin)


def _init_worker(cjk_font: str, latin_font: str) -> None:
    global _FONTS
    _FONTS = FontBank(cjk_font, latin_font)


def _one(task: dict) -> dict:
    try:
        return _render_task(task)
    except Exception as exc:  # noqa: BLE001 - keep the pool alive
        return {"error": f"{task['index']}: {exc}"}


def _render_task(task: dict) -> dict:
    fonts = _FONTS
    if fonts is None:
        _init_worker(task["cjk_font"], task["latin_font"])
        fonts = _FONTS
    assert fonts is not None
    rng = np.random.default_rng(task["seed"])
    text, rows, cols = sample_grid(rng, task["cjk"])
    rgba = render_grid_seal(rng, fonts, text, rows, cols)
    rotation = float(rng.uniform(-5.0, 5.0))
    rotated = _rotate_rgba(rgba, rotation)
    box = _tight_bbox(rotated[..., 3])
    if box is None:
        return {"error": f"{task['index']}: empty seal"}
    x1, y1, x2, y2 = box
    crop = rotated[y1:y2, x1:x2, :3]
    strip = grid_unroll(crop, rows, cols, REC_H, REC_W)
    strip = _degrade_strip(strip, rng, mild=task["mild"])
    name = f"{task['index']:06d}.png"
    Image.fromarray(strip, "RGB").save(Path(task["rec_dir"]) / name)
    return {"name": name, "text": text}


def generate(args: argparse.Namespace) -> dict:
    _export_pythonpath()
    charset = read_charset_file(Path(args.charset))
    cjk = [ch for ch in charset if contains_cjk(ch)]
    if len(cjk) < 16:
        raise ValueError("charset does not have enough CJK characters for name seals")
    cjk_font = args.cjk_font or find_cjk_font()
    latin_font = args.latin_font or find_latin_font()
    out = Path(args.out)
    stats = {"rec_train": 0, "rec_val": 0, "errors": []}
    for split, count, seed0, mild in (
        ("train", args.train, args.seed, False),
        ("val", args.val, args.seed + 1_000_000, True),
    ):
        rec_dir = out / "rec" / split / "images"
        rec_dir.mkdir(parents=True, exist_ok=True)
        tasks = [
            {
                "index": i,
                "seed": seed0 + i,
                "rec_dir": str(rec_dir),
                "cjk_font": cjk_font,
                "latin_font": latin_font,
                "cjk": cjk,
                "mild": mild,
            }
            for i in range(count)
        ]
        workers = 1 if args.workers <= 1 or count < 4 else min(args.workers, count)
        if workers == 1:
            _init_worker(cjk_font, latin_font)
            results = [_one(task) for task in tasks]
        else:
            try:
                pool_context = get_context("fork")
            except ValueError:
                pool_context = get_context("spawn")
            with pool_context.Pool(workers, initializer=_init_worker, initargs=(cjk_font, latin_font)) as pool:
                results = pool.map(_one, tasks, chunksize=max(1, count // (workers * 4)))
        lines = []
        for result in results:
            if "error" in result:
                stats["errors"].append(result["error"])
                continue
            lines.append(f"{result['name']}\t{result['text']}")
        tsv = out / "rec" / split / "labels.tsv"
        tsv.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[f"rec_{split}"] = len(lines)
        print(f"{split}: rec={len(lines)} errors={len(stats['errors'])}", flush=True)
    (out / "charset.txt").write_text(Path(args.charset).read_text(encoding="utf-8"), encoding="utf-8")
    stats["charset"] = len(charset)
    print(stats, flush=True)
    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize free name-seal grids")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--charset", type=Path, required=True)
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--val", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cjk-font", type=str, default=None)
    parser.add_argument("--latin-font", type=str, default=None)
    return parser


def main() -> None:
    generate(build_argparser().parse_args())


if __name__ == "__main__":
    main()
