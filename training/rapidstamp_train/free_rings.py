"""Ring strips whose text is drawn from the whole charset, not the template list.

The closed company-name list taught the recognizer to swap the two-character
middle for another word in that list. These strips keep the same polar layout
and use a milder horizontal squeeze, so the glyphs stay readable.
"""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rapidstamp.geometry import corrected_unwrap_start, pad_to_square, polar_unwrap
from rapidstamp.labels import REC_H, REC_W, UNWRAP_START_DEG
from rapidstamp_train.corpus import BOTTOM_PHRASES, COMPANY_SUFFIX, contains_cjk
from rapidstamp_train.fonts_util import find_cjk_font, find_latin_font
from rapidstamp_train.realdata import read_charset_file
from rapidstamp_train.synthesize import (
    FontBank,
    _compose,
    _degrade_strip,
    _draw_arc,
    _export_pythonpath,
    _rotate_rgba,
    _tight_bbox,
)

_FONTS: FontBank | None = None


def _pools(charset: list[str]) -> tuple[list[str], list[str], list[str]]:
    cjk = [ch for ch in charset if contains_cjk(ch)]
    latin = [ch for ch in charset if ch.isascii() and ch.isalpha()]
    digits = [ch for ch in charset if ch.isdigit()]
    if not cjk or not latin or not digits:
        raise ValueError("charset is missing CJK, Latin, or digits")
    return cjk, latin, digits


def sample_ring_text(
    rng: np.random.Generator,
    cjk: list[str],
    latin: list[str],
    digits: list[str],
) -> tuple[str, str]:
    """Return ``(arc, bottom)``. The label is their concatenation."""
    roll = float(rng.random())
    bottom = ""
    if roll < 0.55:
        count = int(rng.integers(4, 14))
        text = "".join(str(ch) for ch in rng.choice(cjk, size=count))
        if rng.random() < 0.35:
            if rng.random() < 0.55:
                bottom = BOTTOM_PHRASES[int(rng.integers(0, len(BOTTOM_PHRASES)))]
            else:
                bottom = "".join(str(ch) for ch in rng.choice(digits, size=13))
    elif roll < 0.72:
        count = int(rng.integers(2, 5))
        prefix = "".join(str(ch) for ch in rng.choice(cjk, size=count))
        text = prefix + COMPANY_SUFFIX[int(rng.integers(0, len(COMPANY_SUFFIX)))]
    elif roll < 0.86:
        count = int(rng.integers(5, 14))
        text = "".join(str(ch) for ch in rng.choice(latin, size=count))
    else:
        count = int(rng.integers(8, 16))
        text = "".join(str(ch) for ch in rng.choice(digits, size=count))
    return text, bottom


def render_free_seal(
    rng: np.random.Generator,
    fonts: FontBank,
    text: str,
    bottom: str,
    elliptic: bool,
) -> np.ndarray:
    """Upright RGBA seal. CJK glyphs are squeezed less than the template renderer."""
    if elliptic:
        width = int(rng.integers(340, 460))
        height = int(rng.integers(230, 320))
    else:
        side = int(rng.integers(300, 420))
        width = height = side
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    rad_x = width * 0.46
    rad_y = height * 0.46
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    border = max(3, int(min(width, height) / 70))
    draw.ellipse([cx - rad_x, cy - rad_y, cx + rad_x, cy + rad_y], outline=255, width=border)
    inner = 0.78
    draw.ellipse(
        [cx - rad_x * inner, cy - rad_y * inner, cx + rad_x * inner, cy + rad_y * inner],
        outline=255,
        width=max(2, border - 1),
    )
    cjk = contains_cjk(text)
    span = float(np.clip(len(text) * (18.0 if cjk else 14.0), 150, 280))
    phi0 = max(155.0, 200.0 - span * 0.12)
    if bottom:
        span = min(span, 230.0)
        phi0 = max(phi0, 160.0)
    font_size = int(np.clip(min(rad_x, rad_y) * (0.22 if len(text) > 10 else 0.26), 16, 40))
    _draw_arc(canvas, text, fonts, cx, cy, rad_x, rad_y, phi0, span, 0.88, font_size, cjk_scale=0.88)
    if bottom:
        bspan = float(np.clip(len(bottom) * (7.2 if bottom.isdigit() else 11.0), 40, 100))
        bsize = int(np.clip(font_size * (0.72 if not bottom.isdigit() else 0.62), 14, 32))
        _draw_arc(
            canvas,
            bottom,
            fonts,
            cx,
            cy,
            rad_x,
            rad_y,
            90.0 - bspan / 2.0,
            bspan,
            0.88,
            bsize,
            cjk_scale=0.88,
        )
    ink = (int(rng.integers(165, 210)), int(rng.integers(16, 45)), int(rng.integers(16, 40)))
    return _compose(canvas, ink, rng, round_shape=True, margin=int(min(width, height) * 0.02))


def _init_worker(cjk_font: str, latin_font: str) -> None:
    global _FONTS
    _FONTS = FontBank(cjk_font, latin_font)


def _one(task: dict) -> dict:
    try:
        return _render_task(task)
    except Exception as exc:  # noqa: BLE001 - keep the pool going and report the index
        return {"error": f"{task['index']}: {exc}"}


def _render_task(task: dict) -> dict:
    fonts = _FONTS
    if fonts is None:
        _init_worker(task["cjk_font"], task["latin_font"])
        fonts = _FONTS
    assert fonts is not None
    rng = np.random.default_rng(task["seed"])
    text, bottom = sample_ring_text(rng, task["cjk"], task["latin"], task["digits"])
    elliptic = bool(rng.random() < 0.28)
    rgba = render_free_seal(rng, fonts, text, bottom, elliptic)
    rotation = float(rng.uniform(-4.0, 4.0) if elliptic else rng.uniform(0.0, 360.0))
    rotated = _rotate_rgba(rgba, rotation)
    box = _tight_bbox(rotated[..., 3])
    if box is None:
        return {"error": f"{task['index']}: empty seal"}
    x1, y1, x2, y2 = box
    crop = rotated[y1:y2, x1:x2, :3]
    start = corrected_unwrap_start(UNWRAP_START_DEG, rotation) + float(rng.uniform(-12.0, 12.0))
    source = crop if elliptic else pad_to_square(crop, pad_value=255)
    strip = polar_unwrap(source, out_w=REC_W, out_h=REC_H, start_deg=start)
    strip = _degrade_strip(strip, rng, mild=task["mild"])
    name = f"{task['index']:06d}.png"
    Image.fromarray(strip, "RGB").save(Path(task["rec_dir"]) / name)
    return {"name": name, "text": text + bottom}


def generate(args: argparse.Namespace) -> dict:
    _export_pythonpath()
    charset_path = Path(args.charset)
    charset = read_charset_file(charset_path)
    cjk, latin, digits = _pools(charset)
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
                "latin": latin,
                "digits": digits,
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
    (out / "charset.txt").write_text(charset_path.read_text(encoding="utf-8"), encoding="utf-8")
    stats["charset"] = len(charset)
    print(stats, flush=True)
    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize free-ring recognition strips")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--charset", type=Path, required=True)
    parser.add_argument("--train", type=int, default=16000)
    parser.add_argument("--val", type=int, default=800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cjk-font", type=str, default=None)
    parser.add_argument("--latin-font", type=str, default=None)
    return parser


def main() -> None:
    generate(build_argparser().parse_args())


if __name__ == "__main__":
    main()
