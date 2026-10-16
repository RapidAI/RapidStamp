"""Synthetic round seals.

The generator draws a circular border and a short title along an arc, then
composites it onto a paper background. Labels are the tight axis-aligned box
and the title string. Every character is in ``rapidstamp.charset.CHARSET``.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from rapidstamp.charset import CHARSET
from rapidstamp.geometry import (
    BORDER_FRAC,
    CANONICAL_START,
    TEXT_EDGE,
    TEXT_FRACTION,
    TEXT_INNER_FRAC,
    TEXT_OUTER_FRAC,
)
from rapidstamp.titles import BRANDS, CITIES, FIXED, SUFFIXES

COLORS = [(186, 28, 32), (168, 22, 28), (28, 58, 150), (110, 28, 88)]


def font_file() -> Path:
    env = os.environ.get("RAPIDSTAMP_FONT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "assets" / "fonts" / "cjk-subset.ttf"


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_file()), size)


def sample_text(rng: np.random.Generator) -> str:
    if int(rng.integers(0, 5)) == 0:
        return FIXED[int(rng.integers(0, len(FIXED)))]
    city = CITIES[int(rng.integers(0, len(CITIES)))]
    brand = BRANDS[int(rng.integers(0, len(BRANDS)))]
    suffix = SUFFIXES[int(rng.integers(0, len(SUFFIXES)))]
    text = city + brand + suffix
    missing = [ch for ch in text if ch not in CHARSET]
    if missing:
        raise RuntimeError(f"text {text} left the charset: {missing}")
    return text


@dataclass
class Page:
    image: np.ndarray
    boxes: list[list[float]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    starts: list[float] = field(default_factory=list)


def _glyph(ch: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    probe = ImageDraw.Draw(Image.new("L", (4, 4)))
    bbox = probe.textbbox((0, 0), ch, font=font)
    width = max(1, bbox[2] - bbox[0] + 2)
    height = max(1, bbox[3] - bbox[1] + 2)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((1 - bbox[0], 1 - bbox[1]), ch, font=font, fill=(255, 255, 255, 255))
    return img


def _paste(base: Image.Image, overlay: Image.Image, x: int, y: int) -> None:
    bw, bh = base.size
    ow, oh = overlay.size
    x0, y0 = int(x), int(y)
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(ow, bw - x0), min(oh, bh - y0)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return
    base.alpha_composite(overlay.crop((src_x0, src_y0, src_x1, src_y1)), (dst_x0, dst_y0))


def _draw_seal(radius: int, text: str, rotation: float) -> tuple[Image.Image, float]:
    side = int(math.ceil(radius * 2 + 8))
    sprite = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    cx = cy = side / 2.0
    stroke = max(2, int(round(radius * BORDER_FRAC)))
    path_r = radius - stroke / 2.0
    draw = ImageDraw.Draw(sprite)
    draw.ellipse(
        [cx - path_r, cy - path_r, cx + path_r, cy + path_r],
        outline=(255, 255, 255, 255),
        width=stroke,
    )
    band = (TEXT_OUTER_FRAC - TEXT_INNER_FRAC) * radius
    font = _font(max(12, int(band * 0.92)))
    r_text = radius * (TEXT_INNER_FRAC + TEXT_OUTER_FRAC) / 2.0
    arc = 2.0 * math.pi * TEXT_FRACTION
    start = CANONICAL_START + rotation
    # Keep characters away from the arc edges so a few degrees of gap-alignment
    # error does not slice the first or last glyph.
    edge = TEXT_EDGE
    usable = arc - 2.0 * edge
    # Visual top of the glyph should point outward, which is ``angle``.
    # PIL rotates counter-clockwise; image y grows downward.
    for i, ch in enumerate(text):
        angle = start + edge + (i + 0.5) * usable / len(text)
        glyph = _glyph(ch, font)
        deg = -math.degrees(angle + math.pi / 2.0)
        rotated = glyph.rotate(deg, expand=True, resample=Image.BICUBIC)
        px = cx + math.cos(angle) * r_text
        py = cy + math.sin(angle) * r_text
        _paste(sprite, rotated, int(round(px - rotated.width / 2)), int(round(py - rotated.height / 2)))
    return sprite, start


def _composite(page: np.ndarray, sprite: Image.Image, cx: float, cy: float, color: tuple[int, int, int], alpha_scale: float) -> None:
    rgba = np.asarray(sprite).astype(np.float32)
    sh, sw = rgba.shape[:2]
    x0 = int(round(cx - sw / 2.0))
    y0 = int(round(cy - sh / 2.0))
    h, w = page.shape[:2]
    src_x0, src_y0 = max(0, -x0), max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(sw, w - x0), min(sh, h - y0)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return
    crop = rgba[src_y0:src_y1, src_x0:src_x1]
    alpha = (crop[:, :, 3:4] / 255.0) * alpha_scale
    ink = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    region = page[dst_y0 : dst_y0 + crop.shape[0], dst_x0 : dst_x0 + crop.shape[1]].astype(np.float32)
    blended = region * (1.0 - alpha) + ink * alpha
    page[dst_y0 : dst_y0 + crop.shape[0], dst_x0 : dst_x0 + crop.shape[1]] = np.clip(blended, 0, 255)


def render_page(
    rng: np.random.Generator,
    n_seals: int = 1,
    augment: bool = True,
    size: int = 256,
    rotation: float | None = None,
) -> Page:
    base = float(rng.normal(236, 8))
    paper = rng.normal(base, 4.0, size=(size, size, 3)).astype(np.float32)
    paper = np.clip(paper, 160, 255)
    boxes: list[list[float]] = []
    texts: list[str] = []
    starts: list[float] = []
    centers: list[tuple[float, float, float]] = []
    for _ in range(n_seals):
        radius = float(rng.uniform(68, 100))
        placed = False
        cx = cy = 0.0
        for _try in range(20):
            cx = float(rng.uniform(radius + 2, size - radius - 2))
            cy = float(rng.uniform(radius + 2, size - radius - 2))
            if all(math.hypot(cx - ox, cy - oy) > radius + orad + 6 for ox, oy, orad in centers):
                placed = True
                break
        if not placed:
            continue
        text = sample_text(rng)
        rot = float(rng.uniform(0, 2 * math.pi)) if rotation is None else float(rotation)
        sprite, start = _draw_seal(int(round(radius)), text, rot)
        color = COLORS[int(rng.integers(0, len(COLORS)))]
        alpha = float(rng.uniform(0.72, 1.0)) if augment else 1.0
        if augment and float(rng.random()) < 0.3:
            arr = np.array(sprite)
            mask = arr[:, :, 3] > 0
            drop = rng.random(mask.shape) < 0.04
            arr[:, :, 3][mask & drop] = 0
            sprite = Image.fromarray(arr, mode="RGBA")
        _composite(paper, sprite, cx, cy, color, alpha)
        boxes.append([cx - radius, cy - radius, radius * 2, radius * 2])
        texts.append(text)
        starts.append(start)
        centers.append((cx, cy, radius))
    image = np.clip(paper, 0, 255).astype(np.uint8)
    if augment and n_seals >= 0:
        pil = Image.fromarray(image, mode="RGB")
        if float(rng.random()) < 0.7:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 0.8))))
        if float(rng.random()) < 0.5:
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=int(rng.integers(55, 95)))
            pil = Image.open(buf).convert("RGB")
        image = np.asarray(pil).astype(np.uint8)
        noise = rng.normal(0, float(rng.uniform(0, 3)), size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Page(image=image, boxes=boxes, texts=texts, starts=starts)
