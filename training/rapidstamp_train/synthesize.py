"""Synthesize Chinese and foreign seals on document pages.

The renderer draws ring text in the same angular convention as
``rapidstamp.geometry.polar_unwrap``, so a strip cut from the pasted seal is
what the recognizer trains on and what inference searches over.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from rapidstamp.geometry import (
    corrected_unwrap_start,
    grid_unroll,
    hline_band,
    pad_to_square,
    polar_unwrap,
)
from rapidstamp.labels import (
    CLASS_NAMES,
    REC_H,
    REC_W,
    UNWRAP_START_DEG,
    class_id,
)
from rapidstamp_train.corpus import (
    BOTTOM_PHRASES,
    NOTARY_EN,
    RECT_WORDS,
    all_chars,
    chinese_company,
    chinese_notary,
    contains_cjk,
    latin_company,
    personal_name,
    seal_code,
    tax_id,
)
from rapidstamp_train.fonts_util import find_cjk_font, find_latin_font

CHARSET = all_chars()
CHAR_OK = set(CHARSET)


@dataclass
class SealInstance:
    cls_name: str
    text: str
    layout: str
    rows: int
    cols: int
    rgba: np.ndarray
    rotation_ccw: float


class FontBank:
    def __init__(self, cjk_path: str, latin_path: str) -> None:
        self.cjk_path = cjk_path
        self.latin_path = latin_path
        self._cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get(self, kind: str, size: int) -> ImageFont.FreeTypeFont:
        size = int(max(8, min(180, size)))
        key = (kind, size)
        if key not in self._cache:
            path = self.cjk_path if kind == "cjk" else self.latin_path
            self._cache[key] = ImageFont.truetype(path, size=size)
        return self._cache[key]

    def for_char(self, ch: str, size: int) -> ImageFont.FreeTypeFont:
        return self.get("cjk" if contains_cjk(ch) else "latin", size)


def _assert_text(text: str) -> None:
    missing = [ch for ch in text if ch not in CHAR_OK]
    if missing:
        raise ValueError(f"text {text!r} has chars outside charset: {missing}")


def _render_glyph(font: ImageFont.FreeTypeFont, ch: str, scale_x: float = 1.0) -> Image.Image:
    bbox = font.getbbox(ch)
    width = max(1, bbox[2] - bbox[0] + 4)
    height = max(1, bbox[3] - bbox[1] + 4)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.text((2 - bbox[0], 2 - bbox[1]), ch, font=font, fill=255)
    arr = np.asarray(image)
    ys, xs = np.where(arr > 8)
    if len(xs) == 0:
        return image
    pad = 1
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(arr.shape[1], int(xs.max()) + 1 + pad)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(arr.shape[0], int(ys.max()) + 1 + pad)
    image = image.crop((x1, y1, x2, y2))
    # Seal faces are narrower than a square gothic. This is our own draw path;
    # it is not copied from an external generator.
    if scale_x < 0.99:
        image = image.resize(
            (max(1, int(round(image.width * scale_x))), image.height),
            Image.Resampling.BILINEAR,
        )
    return image


def _blit(dst: Image.Image, glyph: Image.Image, cx: float, cy: float) -> Image.Image:
    """Max-composite ``glyph`` onto ``dst`` without copying the whole canvas."""
    x = int(round(cx - glyph.width / 2.0))
    y = int(round(cy - glyph.height / 2.0))
    gx1 = max(0, -x)
    gy1 = max(0, -y)
    gx2 = min(glyph.width, dst.width - x)
    gy2 = min(glyph.height, dst.height - y)
    if gx2 <= gx1 or gy2 <= gy1:
        return dst
    piece = glyph.crop((gx1, gy1, gx2, gy2))
    dx, dy = x + gx1, y + gy1
    region = dst.crop((dx, dy, dx + piece.width, dy + piece.height))
    dst.paste(ImageChops.lighter(region, piece), (dx, dy))
    return dst


def _draw_arc(
    mask: Image.Image,
    text: str,
    fonts: FontBank,
    cx: float,
    cy: float,
    radius_x: float,
    radius_y: float,
    phi0: float,
    span: float,
    radial: float,
    font_size: int,
    cjk_scale: float = 0.70,
) -> None:
    if not text:
        return
    count = len(text)
    for index, ch in enumerate(text):
        phi = phi0 + (index + 0.5) / count * span
        rad = math.radians(phi)
        x = cx + radius_x * radial * math.cos(rad)
        y = cy + radius_y * radial * math.sin(rad)
        scale_x = cjk_scale if contains_cjk(ch) else 1.0
        glyph = _render_glyph(fonts.for_char(ch, font_size), ch, scale_x=scale_x)
        glyph = glyph.rotate(-(phi + 90.0), expand=True, resample=Image.Resampling.BICUBIC)
        _blit(mask, glyph, x, y)


def _star_points(cx: float, cy: float, outer: float, inner: float) -> list[tuple[float, float]]:
    points = []
    # Start at the top. Image angle -90 degrees, which is phi = 270 in our clockwise system,
    # but a standard math CCW star looks upright if the first vertex points up (-y).
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        ang = -math.pi / 2 + i * math.pi / 5
        points.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
    return points


def _ink_color(rng: np.random.Generator, kind: str) -> tuple[int, int, int]:
    if kind == "rect_stamp":
        palette = [(30, 70, 170), (20, 20, 20), (170, 30, 30), (110, 30, 130)]
    elif kind in {"company_latin", "notary"}:
        palette = [(170, 28, 32), (25, 55, 150), (20, 20, 20), (90, 20, 120)]
    else:
        palette = [(rng.integers(165, 210), rng.integers(16, 45), rng.integers(16, 40))]
        if rng.random() < 0.08:
            palette.append((30, 55, 150))
    rgb = palette[int(rng.integers(0, len(palette)))]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _colorize(mask: Image.Image, rgb: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(mask).astype(np.float32) / 255.0
    speckle = rng.random(arr.shape, dtype=np.float32)
    arr = np.where(speckle < 0.035, arr * 0.15, arr)
    blurred = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "L")
    blurred = blurred.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.25, 0.7))))
    alpha = np.asarray(blurred).astype(np.float32) / 255.0
    alpha = np.clip(alpha * float(rng.uniform(0.82, 0.98)), 0.0, 1.0)
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return rgba


def _paper_fill(width: int, height: int, box: list[float], round_shape: bool) -> Image.Image:
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fill = (246, 242, 234, 235)
    if round_shape:
        draw.ellipse(box, fill=fill)
    else:
        draw.rounded_rectangle(box, radius=max(4, int(min(width, height) * 0.04)), fill=fill)
    return layer


def make_seal(rng: np.random.Generator, fonts: FontBank, cls_name: str | None = None) -> dict:
    """Render one upright seal. ``text`` is the ring or stamp transcription."""
    if cls_name is None:
        cls_name = CLASS_NAMES[int(rng.choice(len(CLASS_NAMES), p=_CLASS_PROB))]
    layout = {
        "round_official": "circle",
        "oval": "ellipse",
        "square_personal": "grid",
        "finance": "ellipse",
        "invoice": "ellipse",
        "company_latin": "circle",
        "notary": "circle",
        "rect_stamp": "hline",
    }[cls_name]
    rows, cols = 1, 1
    center_text = ""
    bottom = ""

    if cls_name == "square_personal":
        text, rows, cols = personal_name(rng)
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
        for r in range(rows):
            for c in range(cols):
                ch = text[r * cols + c]
                glyph = _render_glyph(fonts.for_char(ch, font_size), ch, scale_x=0.72)
                cx = inner + (c + 0.5) * cell_w
                cy = inner + (r + 0.5) * cell_h
                canvas = _blit(canvas, glyph, cx, cy)
        _assert_text(text)
        rgba = _compose(canvas, _ink_color(rng, cls_name), rng, round_shape=False, margin=margin)
        return _pack(cls_name, text, layout, rows, cols, rgba, center_text)

    if cls_name == "rect_stamp":
        if rng.random() < 0.55:
            text = RECT_WORDS[int(rng.integers(0, len(RECT_WORDS)))]
        else:
            text = latin_company(rng)
        _assert_text(text)
        font_size = int(rng.integers(28, 42))
        font = fonts.for_char(text[0], font_size)
        probe = ImageDraw.Draw(Image.new("L", (8, 8)))
        bbox = probe.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = int(font_size * 0.7), int(font_size * 0.55)
        width, height = tw + pad_x * 2, th + pad_y * 2
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        border = max(3, font_size // 10)
        draw.rounded_rectangle([2, 2, width - 3, height - 3], radius=6, outline=255, width=border)
        glyph_line = Image.new("L", (tw + 4, th + 4), 0)
        ImageDraw.Draw(glyph_line).text((2 - bbox[0], 2 - bbox[1]), text, font=font, fill=255)
        canvas = _blit(canvas, glyph_line, width / 2, height / 2)
        rgba = _colorize(canvas, _ink_color(rng, cls_name), rng)
        return _pack(cls_name, text, layout, 1, 1, rgba, center_text)

    elliptic = layout == "ellipse"
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
    outer = [cx - rad_x, cy - rad_y, cx + rad_x, cy + rad_y]
    border = max(3, int(min(width, height) / 70))
    draw.ellipse(outer, outline=255, width=border)
    inner_scale = 0.78
    draw.ellipse(
        [
            cx - rad_x * inner_scale,
            cy - rad_y * inner_scale,
            cx + rad_x * inner_scale,
            cy + rad_y * inner_scale,
        ],
        outline=255,
        width=max(2, border - 1),
    )

    if cls_name == "company_latin":
        text = latin_company(rng)
    elif cls_name == "notary":
        text = chinese_notary(rng) if rng.random() < 0.55 else NOTARY_EN[int(rng.integers(0, len(NOTARY_EN)))]
    else:
        text = chinese_company(rng)

    if cls_name == "finance":
        bottom = "财务专用章"
    elif cls_name == "invoice":
        bottom = "发票专用章"
        center_text = tax_id(rng)
    elif cls_name == "oval" and rng.random() < 0.7:
        bottom = "合同专用章"
    elif cls_name == "round_official" and rng.random() < 0.70:
        bottom = seal_code(rng)

    _assert_text(text)
    _assert_text(bottom)
    _assert_text(center_text)

    company_span = float(np.clip(len(text) * (15.5 if contains_cjk(text) else 13.0), 130, 250))
    company_phi0 = 200.0 - company_span * 0.15
    # Keep the company arc on the upper half and clear of the bottom phrase.
    if company_phi0 < 155:
        company_phi0 = 155.0
    font_size = int(min(rad_x, rad_y) * (0.30 if len(text) < 8 else 0.24))
    font_size = int(np.clip(font_size, 16, 48))
    _draw_arc(
        canvas,
        text,
        fonts,
        cx,
        cy,
        rad_x,
        rad_y,
        company_phi0,
        company_span,
        0.88,
        font_size,
    )
    if bottom:
        span = float(np.clip(len(bottom) * (7.2 if bottom.isdigit() else 11.0), 40, 100))
        phi0 = 90.0 - span / 2.0
        bsize = int(np.clip(font_size * (0.72 if not bottom.isdigit() else 0.62), 14, 36))
        _draw_arc(canvas, bottom, fonts, cx, cy, rad_x, rad_y, phi0, span, 0.88, bsize)
    if cls_name == "round_official":
        star_r = min(rad_x, rad_y) * 0.22
        draw = ImageDraw.Draw(canvas)
        draw.polygon(_star_points(cx, cy, star_r, star_r * 0.42), fill=255)
    if center_text:
        csize = int(min(rad_x, rad_y) * 0.13)
        font = fonts.get("latin", csize)
        probe = ImageDraw.Draw(Image.new("L", (4, 4)))
        bbox = probe.textbbox((0, 0), center_text, font=font)
        glyph = Image.new("L", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4), 0)
        ImageDraw.Draw(glyph).text((2 - bbox[0], 2 - bbox[1]), center_text, font=font, fill=255)
        canvas = _blit(canvas, glyph, cx, cy)

    full_text = text + bottom
    _assert_text(full_text)
    rgba = _compose(canvas, _ink_color(rng, cls_name), rng, round_shape=True, margin=int(min(width, height) * 0.02))
    packed = _pack(cls_name, full_text, layout, rows, cols, rgba, center_text)
    return packed


def _compose(mask: Image.Image, rgb: tuple[int, int, int], rng: np.random.Generator, round_shape: bool, margin: int) -> np.ndarray:
    ink = _colorize(mask, rgb, rng)
    width, height = mask.size
    box = [margin, margin, width - margin, height - margin]
    paper = _paper_fill(width, height, box, round_shape=round_shape)
    base = paper.convert("RGBA")
    base.alpha_composite(Image.fromarray(ink, "RGBA"))
    return np.asarray(base)


def _pack(cls_name, text, layout, rows, cols, rgba, center_text) -> dict:
    return {
        "cls_name": cls_name,
        "text": text,
        "center_text": center_text,
        "layout": layout,
        "rows": rows,
        "cols": cols,
        "rgba": rgba,
    }


_CLASS_PROB = np.array([0.18, 0.10, 0.12, 0.12, 0.12, 0.12, 0.10, 0.14], dtype=np.float64)


def _tight_bbox(alpha: np.ndarray, thresh: int = 12) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(alpha > thresh)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _rotate_rgba(rgba: np.ndarray, angle_ccw: float) -> np.ndarray:
    if abs(angle_ccw) < 0.2:
        return rgba
    image = Image.fromarray(rgba, "RGBA").rotate(
        angle_ccw,
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    return np.asarray(image)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image, "RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def _degrade_strip(image: np.ndarray, rng: np.random.Generator, mild: bool) -> np.ndarray:
    im = Image.fromarray(image, "RGB")
    if not mild and rng.random() < 0.55:
        im = im.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.0, 0.7))))
    arr = np.asarray(im)
    quality = int(rng.integers(82, 93) if mild else rng.integers(60, 95))
    if rng.random() < (0.5 if mild else 0.85):
        arr = _jpeg(arr, quality)
    if not mild and rng.random() < 0.8:
        factor = float(rng.uniform(0.78, 1.18))
        arr = np.clip(arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return arr


def strip_from_crop(
    crop: np.ndarray,
    layout: str,
    rows: int,
    cols: int,
    start_deg: float,
) -> np.ndarray:
    if layout == "circle":
        square = pad_to_square(crop, pad_value=255)
        return polar_unwrap(square, out_w=REC_W, out_h=REC_H, start_deg=start_deg)
    if layout == "ellipse":
        return polar_unwrap(crop, out_w=REC_W, out_h=REC_H, start_deg=start_deg)
    if layout == "grid":
        return grid_unroll(crop, rows, cols)
    return hline_band(crop)


def _page_background(rng: np.random.Generator, width: int, height: int, latin: ImageFont.FreeTypeFont) -> np.ndarray:
    base = float(rng.uniform(232, 250))
    paper = np.full((height, width, 3), base, np.float32)
    paper += rng.normal(0.0, 4.0, paper.shape)
    image = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    ink = (int(rng.integers(150, 190)),) * 3
    if rng.random() < 0.7:
        step = int(rng.integers(28, 46))
        for y in range(40, height - 20, step):
            draw.line([(30, y), (width - 30, y)], fill=ink, width=1)
    for _ in range(int(rng.integers(4, 10))):
        y = int(rng.integers(20, height - 20))
        label = "INV " + "".join(str(int(rng.integers(0, 10))) for _ in range(6))
        draw.text((int(rng.integers(20, max(21, width // 2))), y), label, font=latin, fill=ink)
    if rng.random() < 0.35:
        # A red ruling line, not a seal, so the detector has a simple distractor.
        y = int(rng.integers(30, height - 30))
        draw.line([(20, y), (width - 20, y)], fill=(180, 40, 40), width=2)
    return np.asarray(image)


def _place_seals(page: Image.Image, seals: list[dict], rng: np.random.Generator) -> list[dict]:
    width, height = page.size
    placed = []
    boxes: list[tuple[int, int, int, int]] = []
    for seal in seals:
        rgba = seal["rgba"]
        rotation = _rotation_for(seal["layout"], rng)
        rotated = _rotate_rgba(rgba, rotation)
        alpha = rotated[..., 3]
        local = _tight_bbox(alpha)
        if local is None:
            continue
        x1, y1, x2, y2 = local
        sprite = rotated[y1:y2, x1:x2]
        sh, sw = sprite.shape[:2]
        if sw + 8 >= width or sh + 8 >= height:
            continue
        ok = False
        left = top = 0
        for _ in range(40):
            left = int(rng.integers(4, width - sw - 3))
            top = int(rng.integers(4, height - sh - 3))
            cand = (left, top, left + sw, top + sh)
            if all(_iou(cand, prev) < 0.05 for prev in boxes):
                ok = True
                break
        if not ok:
            continue
        layer = Image.fromarray(sprite, "RGBA")
        page.alpha_composite(layer, (left, top))
        boxes.append((left, top, left + sw, top + sh))
        placed.append(
            {
                **{k: seal[k] for k in ("cls_name", "text", "center_text", "layout", "rows", "cols")},
                "bbox": [left, top, left + sw, top + sh],
                "rotation_ccw": rotation,
                "upright_rgba": rgba,
            }
        )
    return placed


def _rotation_for(layout: str, rng: np.random.Generator) -> float:
    if layout == "circle":
        return float(rng.uniform(0.0, 360.0))
    if layout == "ellipse":
        return float(rng.uniform(-4.0, 4.0))
    return float(rng.uniform(-5.0, 5.0))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / float(area_a + area_b - inter + 1e-6)


def _crop_jitter(bbox: list[int], width: int, height: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    grow = float(rng.uniform(0.0, 0.08))
    shift_x = float(rng.uniform(-0.03, 0.03)) * bw
    shift_y = float(rng.uniform(-0.03, 0.03)) * bh
    nx1 = int(round(x1 - bw * grow / 2 + shift_x))
    ny1 = int(round(y1 - bh * grow / 2 + shift_y))
    nx2 = int(round(x2 + bw * grow / 2 + shift_x))
    ny2 = int(round(y2 + bh * grow / 2 + shift_y))
    nx1 = max(0, min(width - 2, nx1))
    ny1 = max(0, min(height - 2, ny1))
    nx2 = max(nx1 + 2, min(width, nx2))
    ny2 = max(ny1 + 2, min(height, ny2))
    return nx1, ny1, nx2, ny2


def _plain_strip(text: str, fonts: FontBank, rng: np.random.Generator) -> np.ndarray:
    cjk = contains_cjk(text)
    size = int(rng.integers(26, 40))
    font = fonts.get("cjk" if cjk else "latin", size)
    probe = ImageDraw.Draw(Image.new("L", (4, 4)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tw, th = max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
    image = Image.new("RGB", (tw + 16, th + 12), (248, 244, 236))
    color = (180, 30, 30) if rng.random() < 0.7 else (20, 40, 140)
    ImageDraw.Draw(image).text((8 - bbox[0], 6 - bbox[1]), text, font=font, fill=color)
    return np.asarray(image.resize((REC_W, REC_H), Image.Resampling.BILINEAR))


def _save_page(task: dict) -> dict:
    rng = np.random.default_rng(task["seed"])
    fonts = FontBank(task["cjk_font"], task["latin_font"])
    latin_small = fonts.get("latin", 18)
    width = int(rng.integers(780, 1100))
    height = int(rng.integers(900, 1300))
    background = _page_background(rng, width, height, latin_small)
    page = Image.fromarray(background, "RGB").convert("RGBA")
    n_seals = 0 if rng.random() < 0.08 else int(rng.choice([1, 2, 3], p=[0.62, 0.30, 0.08]))
    seals = [make_seal(rng, fonts) for _ in range(n_seals)]
    placed = _place_seals(page, seals, rng)
    rgb = _jpeg(np.asarray(page.convert("RGB")), int(rng.integers(72, 95)))
    image_name = f"{task['index']:06d}.jpg"
    image_path = Path(task["image_dir"]) / image_name
    Image.fromarray(rgb, "RGB").save(image_path, quality=90)
    labels = []
    rec_items = []
    for ordinal, seal in enumerate(placed):
        labels.append(
            {
                "cls": class_id(seal["cls_name"]),
                "name": seal["cls_name"],
                "bbox": seal["bbox"],
                "text": seal["text"],
                "center_text": seal["center_text"],
                "layout": seal["layout"],
                "rows": seal["rows"],
                "cols": seal["cols"],
            }
        )
        x1, y1, x2, y2 = _crop_jitter(seal["bbox"], rgb.shape[1], rgb.shape[0], rng)
        crop = rgb[y1:y2, x1:x2]
        start = corrected_unwrap_start(UNWRAP_START_DEG, seal["rotation_ccw"])
        start += float(rng.uniform(-12.0, 12.0))
        strip = strip_from_crop(crop, seal["layout"], seal["rows"], seal["cols"], start)
        strip = _degrade_strip(strip, rng, mild=task["mild"])
        rec_name = f"{task['index']:06d}_{ordinal}.png"
        Image.fromarray(strip, "RGB").save(Path(task["rec_dir"]) / rec_name)
        rec_items.append((rec_name, seal["text"]))
        if seal["center_text"]:
            upright = seal["upright_rgba"]
            alpha = upright[..., 3]
            box = _tight_bbox(alpha)
            if box is not None:
                ux1, uy1, ux2, uy2 = box
                uh = uy2 - uy1
                uw = ux2 - ux1
                cy1 = uy1 + int(uh * 0.38)
                cy2 = uy1 + int(uh * 0.66)
                cx1 = ux1 + int(uw * 0.18)
                cx2 = ux1 + int(uw * 0.82)
                center = upright[cy1:cy2, cx1:cx2, :3]
                if center.size:
                    center_name = f"{task['index']:06d}_{ordinal}c.png"
                    center_img = np.asarray(
                        Image.fromarray(center, "RGB").resize((REC_W, REC_H), Image.Resampling.BILINEAR)
                    )
                    Image.fromarray(_degrade_strip(center_img, rng, task["mild"]), "RGB").save(
                        Path(task["rec_dir"]) / center_name
                    )
                    rec_items.append((center_name, seal["center_text"]))
    if task["plain"] > 0:
        for k in range(task["plain"]):
            roll = rng.random()
            if roll < 0.4:
                extra = chinese_company(rng)
            elif roll < 0.55:
                extra = BOTTOM_PHRASES[int(rng.integers(0, len(BOTTOM_PHRASES)))]
            elif roll < 0.75:
                extra = latin_company(rng)
            elif roll < 0.9:
                extra = tax_id(rng)
            else:
                extra = RECT_WORDS[int(rng.integers(0, len(RECT_WORDS)))]
            _assert_text(extra)
            plain = _degrade_strip(_plain_strip(extra, fonts, rng), rng, task["mild"])
            pname = f"{task['index']:06d}_p{k}.png"
            Image.fromarray(plain, "RGB").save(Path(task["rec_dir"]) / pname)
            rec_items.append((pname, extra))
    meta = {"width": int(rgb.shape[1]), "height": int(rgb.shape[0]), "seals": labels}
    (Path(task["label_dir"]) / f"{task['index']:06d}.json").write_text(
        json.dumps(meta, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"rec": rec_items}


def _worker(task: dict) -> dict:
    try:
        return _save_page(task)
    except Exception as exc:  # noqa: BLE001 - report the sample and keep the pool alive
        return {"error": f"{task['index']}: {exc}"}


def _export_pythonpath() -> None:
    repo = Path(__file__).resolve().parents[2]
    entries = [str(repo / "python" / "src"), str(repo / "training")]
    current = os.environ.get("PYTHONPATH", "")
    if current:
        entries.append(current)
    os.environ["PYTHONPATH"] = os.pathsep.join(entries)


def generate(args: argparse.Namespace) -> dict:
    _export_pythonpath()
    out = Path(args.out)
    cjk = args.cjk_font or find_cjk_font()
    latin = args.latin_font or find_latin_font()
    stats = {"det_train": 0, "det_val": 0, "rec_train": 0, "rec_val": 0, "errors": []}
    for split, count, seed0, mild in (
        ("train", args.det_train, args.seed, False),
        ("val", args.det_val, args.seed + 1_000_000, True),
    ):
        image_dir = out / "det" / split / "images"
        label_dir = out / "det" / split / "labels"
        rec_dir = out / "rec" / split / "images"
        for directory in (image_dir, label_dir, rec_dir):
            directory.mkdir(parents=True, exist_ok=True)
        plain = 1 if split == "train" else 0
        tasks = [
            {
                "index": i,
                "seed": seed0 + i,
                "image_dir": str(image_dir),
                "label_dir": str(label_dir),
                "rec_dir": str(rec_dir),
                "cjk_font": cjk,
                "latin_font": latin,
                "mild": mild,
                "plain": plain,
            }
            for i in range(count)
        ]
        rec_lines: list[str] = []
        workers = 1 if args.workers <= 1 or count < 4 else min(args.workers, count)
        if workers == 1:
            results = [_worker(task) for task in tasks]
        else:
            try:
                pool_context = get_context("fork")
            except ValueError:
                pool_context = get_context("spawn")
            with pool_context.Pool(workers) as pool:
                results = pool.map(_worker, tasks, chunksize=max(1, count // (workers * 4)))
        for result in results:
            if "error" in result:
                stats["errors"].append(result["error"])
                continue
            for name, text in result["rec"]:
                rec_lines.append(f"{name}\t{text}")
        (out / "rec" / split / "labels.tsv").write_text("\n".join(rec_lines) + ("\n" if rec_lines else ""), encoding="utf-8")
        stats[f"det_{split}"] = count
        stats[f"rec_{split}"] = len(rec_lines)
        print(f"{split}: pages={count} rec={len(rec_lines)} errors={len(stats['errors'])}", flush=True)
    charset_path = out / "charset.txt"
    charset_path.write_text("\n".join(CHARSET) + "\n", encoding="utf-8")
    (out / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    from rapidstamp.labels import write_preprocess

    write_preprocess(str(out / "preprocess.txt"))
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize seal detection and recognition data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--det-train", type=int, default=12000)
    parser.add_argument("--det-val", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cjk-font", type=str, default="")
    parser.add_argument("--latin-font", type=str, default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    generate(args)


if __name__ == "__main__":
    main()
