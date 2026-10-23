"""Draw a whole seal crop whose ring text is the training label.

Round, oval, and invoice layouts share one polar convention: 0° is +x,
clockwise, y down, so the top of the seal is 270°. Ring characters sit on
that arc with their tops pointing outward. Center text, the star, the bottom
code, and the horizontal line are distractors and are not part of the label.
"""

from __future__ import annotations

import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from rapidstamp_train.direct_aug import pad_square, perspective_coeffs

BOTTOM_PHRASES = ("发票专用章", "财务专用章", "合同专用章", "业务专用章")
DOC_TEXT = "合同编号签订日期甲方乙方金额页码附注签章处复印件"


def ring_angles(length: int, rng: random.Random) -> tuple[float, float]:
    """Arc centered on the top, with the gap centered on the bottom."""
    per = rng.uniform(13.0, 18.5)
    span = max(120.0, min(330.0, length * per))
    gap = 360.0 - span
    phi0 = 90.0 + gap / 2.0
    return phi0, span


def _covers(angle: float, phi0: float, span: float) -> bool:
    return (angle - phi0) % 360.0 < span


class FontBank:
    """Song, Fangsong, Hei, and Kai files. A missing glyph falls through."""

    def __init__(self, paths: list[Path]):
        self.paths = [str(path) for path in paths if Path(path).is_file()]
        if not self.paths:
            raise FileNotFoundError("no seal fonts")
        self._font: dict[tuple[int, int], ImageFont.FreeTypeFont] = {}
        self._miss: set[tuple[int, str]] = set()

    def __len__(self) -> int:
        return len(self.paths)

    def font(self, index: int, size: int) -> ImageFont.FreeTypeFont:
        key = (index, size)
        cached = self._font.get(key)
        if cached is None:
            cached = ImageFont.truetype(self.paths[index], size=size)
            self._font[key] = cached
        return cached

    def glyph(self, char: str, size: int, primary: int, scale_x: float) -> Image.Image:
        if char == " ":
            return Image.new("L", (max(4, size // 3), size), 0)
        order = [primary] + [index for index in range(len(self.paths)) if index != primary]
        for index in order:
            if (index, char) in self._miss:
                continue
            font = self.font(index, size)
            bbox = font.getbbox(char)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width < 2 or height < 2:
                self._miss.add((index, char))
                continue
            image = Image.new("L", (width + 4, height + 4), 0)
            ImageDraw.Draw(image).text((2 - bbox[0], 2 - bbox[1]), char, font=font, fill=255)
            if scale_x < 0.99:
                image = image.resize((max(1, int(round(image.width * scale_x))), image.height), Image.BILINEAR)
            return image
        block = Image.new("L", (size, size), 0)
        ImageDraw.Draw(block).rectangle([2, 2, size - 3, size - 3], outline=255, width=max(1, size // 12))
        return block


def list_fonts(directory: Path) -> list[Path]:
    paths = []
    if not directory.is_dir():
        return paths
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in {".ttf", ".otf", ".ttc"} and path.is_file():
            paths.append(path)
    return paths


def _blit(dst: Image.Image, glyph: Image.Image, cx: float, cy: float) -> None:
    x = int(round(cx - glyph.width / 2.0))
    y = int(round(cy - glyph.height / 2.0))
    gx1 = max(0, -x)
    gy1 = max(0, -y)
    gx2 = min(glyph.width, dst.width - x)
    gy2 = min(glyph.height, dst.height - y)
    if gx2 <= gx1 or gy2 <= gy1:
        return
    piece = glyph.crop((gx1, gy1, gx2, gy2))
    dx, dy = x + gx1, y + gy1
    region = dst.crop((dx, dy, dx + piece.width, dy + piece.height))
    dst.paste(Image.composite(piece, region, piece), (dx, dy))


def _draw_arc(
    mask: Image.Image,
    text: str,
    fonts: FontBank,
    primary: int,
    cx: float,
    cy: float,
    radius_x: float,
    radius_y: float,
    phi0: float,
    span: float,
    radial: float,
    font_size: int,
    scale_x: float,
) -> None:
    count = len(text)
    if count == 0:
        return
    for index, char in enumerate(text):
        phi = phi0 + (index + 0.5) / count * span
        rad = math.radians(phi)
        x = cx + radius_x * radial * math.cos(rad)
        y = cy + radius_y * radial * math.sin(rad)
        normal = math.degrees(math.atan2(math.sin(rad) / max(radius_y, 1.0), math.cos(rad) / max(radius_x, 1.0)))
        glyph = fonts.glyph(char, font_size, primary, scale_x)
        glyph = glyph.rotate(-(normal + 90.0), expand=True, resample=Image.BILINEAR)
        _blit(mask, glyph, x, y)


def _draw_horizontal(
    mask: Image.Image,
    text: str,
    fonts: FontBank,
    primary: int,
    cx: float,
    cy: float,
    font_size: int,
    scale_x: float,
) -> None:
    glyphs = [fonts.glyph(char, font_size, primary, scale_x) for char in text]
    if not glyphs:
        return
    gap = max(1, font_size // 12)
    total = sum(glyph.width for glyph in glyphs) + gap * (len(glyphs) - 1)
    x = cx - total / 2.0
    for glyph in glyphs:
        _blit(mask, glyph, x + glyph.width / 2.0, cy)
        x += glyph.width + gap


def _star(cx: float, cy: float, outer: float, inner: float) -> list[tuple[float, float]]:
    points = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def _chord(length: int, radius_x: float, radius_y: float, span: float) -> float:
    count = max(length, 1)
    step = math.radians(span / count)
    return math.hypot(radius_x * (1 - math.cos(step)), radius_y * math.sin(step))


def _paper(width: int, height: int, rng: random.Random, np_rng: np.random.Generator) -> tuple[Image.Image, tuple[int, int, int]]:
    base = rng.uniform(228.0, 250.0)
    tint = np.array(
        [base, base - rng.uniform(0.0, 14.0), base - rng.uniform(0.0, 22.0)],
        dtype=np.float32,
    )
    paper = np.ones((height, width, 3), np.float32) * tint
    paper += np_rng.normal(0.0, 3.0, paper.shape)
    image = Image.fromarray(np.clip(paper, 0, 255).astype(np.uint8), "RGB")
    fill = (int(tint[0]), int(tint[1]), int(tint[2]))
    return image, fill


def _overlay_document(image: Image.Image, rng: random.Random, fonts: FontBank, primary: int) -> None:
    draw = ImageDraw.Draw(image)
    gray = rng.randint(90, 150)
    color = (gray, gray, gray)
    width, height = image.size
    if rng.random() < 0.75:
        step = rng.randint(26, 42)
        for y in range(rng.randint(8, 24), height - 8, step):
            draw.line([(6, y), (width - 6, y)], fill=color, width=1)
    if rng.random() < 0.8:
        size = rng.randint(14, 22)
        for _ in range(rng.randint(2, 5)):
            count = rng.randint(6, 14)
            start = rng.randrange(len(DOC_TEXT) - 1)
            text = "".join(DOC_TEXT[(start + offset) % len(DOC_TEXT)] for offset in range(count))
            glyph = Image.new("L", (size * count + 4, size + 6), 0)
            font = fonts.font(primary, size)
            ImageDraw.Draw(glyph).text((1, 1), text, font=font, fill=255)
            x = rng.randint(0, max(0, width - glyph.width))
            y = rng.randint(0, max(0, height - glyph.height))
            ink = Image.new("RGB", glyph.size, color)
            image.paste(ink, (x, y), glyph)


def _perspective(image: Image.Image, rng: random.Random, fill: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    jitter = rng.uniform(0.03, 0.08)
    output = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    source = []
    for x, y in output:
        source.append((x + rng.uniform(-jitter, jitter) * width, y + rng.uniform(-jitter, jitter) * height))
    coeffs = perspective_coeffs(output, source)
    return image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=fill)


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def render_seal(text: str, rng: random.Random, fonts: FontBank, spin: bool = True) -> Image.Image:
    """RGB square crop. ``text`` is only on the ring; rotation does not change it."""
    style = rng.choice(("round", "round", "round", "oval", "invoice"))
    if style == "oval":
        width = int(rng.randint(400, 500))
        height = int(rng.randint(280, 360))
    elif style == "invoice":
        width = int(rng.randint(420, 520))
        height = int(rng.randint(300, 380))
    else:
        side = int(rng.randint(360, 460))
        width = height = side
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius_x = width * rng.uniform(0.44, 0.47)
    radius_y = height * rng.uniform(0.44, 0.47)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    border = max(3, int(min(width, height) / rng.uniform(58, 78)))
    outer = [cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y]
    draw.ellipse(outer, outline=255, width=border)
    if rng.random() < 0.82:
        inner = rng.uniform(0.90, 0.94)
        draw.ellipse(
            [
                cx - radius_x * inner,
                cy - radius_y * inner,
                cx + radius_x * inner,
                cy + radius_y * inner,
            ],
            outline=255,
            width=max(2, border - 2),
        )
    phi0, span = ring_angles(max(len(text), 1), rng)
    radial = rng.uniform(0.80, 0.86)
    slot = _chord(len(text), radius_x * radial, radius_y * radial, span)
    font_size = int(np.clip(slot * rng.uniform(0.78, 0.92), 16, 56))
    font_size = max(16, font_size - font_size % 2)
    primary = rng.randrange(len(fonts))
    scale_x = rng.uniform(0.78, 1.0)
    _draw_arc(mask, text, fonts, primary, cx, cy, radius_x, radius_y, phi0, span, radial, font_size, scale_x)

    gap = 360.0 - span
    if style == "invoice":
        line_y = cy + radius_y * rng.uniform(0.08, 0.20)
        draw.line(
            [(cx - radius_x * 0.62, line_y), (cx + radius_x * 0.62, line_y)],
            fill=255,
            width=max(2, border - 1),
        )
        tax = "".join(str(rng.randrange(10)) for _ in range(rng.randint(12, 18)))
        _draw_horizontal(
            mask,
            tax,
            fonts,
            primary,
            cx,
            cy - radius_y * 0.02,
            max(12, int(font_size * 0.42)),
            0.9,
        )
        if gap > 36:
            _draw_horizontal(
                mask,
                "发票专用章",
                fonts,
                primary,
                cx,
                cy + radius_y * rng.uniform(0.48, 0.60),
                max(14, int(font_size * 0.62)),
                scale_x,
            )
    else:
        if style == "round" and rng.random() < 0.85:
            star_r = min(radius_x, radius_y) * rng.uniform(0.16, 0.24)
            draw.polygon(_star(cx, cy, star_r, star_r * 0.40), fill=255)
        elif rng.random() < 0.35:
            star_r = min(radius_x, radius_y) * 0.14
            draw.polygon(_star(cx, cy, star_r, star_r * 0.40), fill=255)
        if gap > 48 and rng.random() < 0.7:
            if rng.random() < 0.55:
                bottom = "".join(str(rng.randrange(10)) for _ in range(rng.randint(8, 13)))
                bspan = min(gap * 0.7, max(36.0, len(bottom) * 7.0))
                bphi = 90.0 - bspan / 2.0
                _draw_arc(
                    mask,
                    bottom,
                    fonts,
                    primary,
                    cx,
                    cy,
                    radius_x,
                    radius_y,
                    bphi,
                    bspan,
                    radial * 0.98,
                    max(12, int(font_size * 0.55)),
                    0.9,
                )
            else:
                phrase = rng.choice(BOTTOM_PHRASES)
                if phrase != text:
                    _draw_horizontal(
                        mask,
                        phrase,
                        fonts,
                        primary,
                        cx,
                        cy + radius_y * rng.uniform(0.42, 0.58),
                        max(14, int(font_size * 0.58)),
                        scale_x,
                    )

    thick = rng.random()
    if thick < 0.35:
        mask = mask.filter(ImageFilter.MaxFilter(3))
    elif thick > 0.85:
        mask = mask.filter(ImageFilter.MinFilter(3))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.35)))

    alpha = np.asarray(mask).astype(np.float32) / 255.0
    np_rng = np.random.default_rng(rng.randrange(1 << 31))
    height, width = alpha.shape
    speckle = np_rng.random(alpha.shape)
    alpha = np.where(speckle < 0.03, alpha * 0.2, alpha)
    field = np_rng.random((max(2, height // 8), max(2, width // 8))).astype(np.float32)
    field_img = Image.fromarray(np.clip(field * 255.0, 0, 255).astype(np.uint8), "L").resize((width, height), Image.BILINEAR)
    alpha *= 0.72 + 0.28 * (np.asarray(field_img).astype(np.float32) / 255.0)
    angle = rng.uniform(0.0, math.tau)
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    fade = rng.uniform(0.8, 1.0) + rng.uniform(0.0, 0.35) * (xx * math.cos(angle) + yy * math.sin(angle))
    alpha *= np.clip(fade, 0.45, 1.0)
    if rng.random() < 0.65:
        shift_y = rng.randint(1, 3)
        shift_x = rng.randint(-2, 2)
        shifted = np.roll(np.roll(alpha, shift_y, axis=0), shift_x, axis=1)
        alpha = np.maximum(alpha, shifted * rng.uniform(0.25, 0.6))
    alpha = np.clip(alpha, 0.0, 1.0)

    side = int(max(width, height) * rng.uniform(1.02, 1.22))
    paper, fill = _paper(side, side, rng, np_rng)
    _overlay_document(paper, rng, fonts, primary)
    ink_roll = rng.random()
    if ink_roll < 0.9:
        ink = np.array(
            [rng.randint(150, 205), rng.randint(16, 48), rng.randint(18, 46)],
            dtype=np.float32,
        )
    else:
        ink = np.array([rng.randint(20, 50), rng.randint(30, 70), rng.randint(120, 180)], dtype=np.float32)
    color = np.zeros((height, width, 3), np.float32)
    color[..., 0] = ink[0]
    color[..., 1] = ink[1]
    color[..., 2] = ink[2]
    seal = Image.fromarray(np.clip(color, 0, 255).astype(np.uint8), "RGB")
    alpha_img = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), "L")
    left = max(0, min(side - width, (side - width) // 2 + rng.randint(-4, 4)))
    top = max(0, min(side - height, (side - height) // 2 + rng.randint(-4, 4)))
    paper.paste(seal, (left, top), alpha_img)
    if rng.random() < 0.45:
        _overlay_document(paper, rng, fonts, primary)

    angle = rng.uniform(0.0, 360.0) if spin else 0.0
    image = paper.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=fill)
    image = pad_square(image)
    # pad_square fills white. Recolor the new corners toward the paper.
    if rng.random() < 0.7:
        image = _perspective(image, rng, (255, 255, 255))
    if rng.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.8)))
    if rng.random() < 0.25:
        draw = ImageDraw.Draw(image)
        box_w = int(image.width * rng.uniform(0.04, 0.12))
        box_h = int(image.height * rng.uniform(0.02, 0.07))
        x0 = rng.randint(0, max(0, image.width - box_w))
        y0 = rng.randint(0, max(0, image.height - box_h))
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=fill)
    if rng.random() < 0.7:
        image = _jpeg(image, rng.randint(48, 90))
    return image
