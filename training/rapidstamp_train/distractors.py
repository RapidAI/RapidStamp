"""Red and blue marks that are not seals.

Synthetic pages used to be a clean stamp on blank paper, so the detector
treated any red ink as a seal. These draws are hard negatives: text, rules,
tables, and small logos, kept off the ground-truth boxes.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rapidstamp_train.fonts_util import find_cjk_font, find_latin_font


@lru_cache(maxsize=1)
def _fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        cjk = ImageFont.truetype(find_cjk_font(), 28)
    except (OSError, FileNotFoundError):
        cjk = ImageFont.load_default()
    try:
        latin = ImageFont.truetype(find_latin_font(), 22)
    except (OSError, FileNotFoundError):
        latin = ImageFont.load_default()
    return cjk, latin


def _iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _hits(rect: list[int], boxes: np.ndarray) -> bool:
    if len(boxes) == 0:
        return False
    return any(_iou(rect, [int(v) for v in box]) > 0.02 for box in boxes)


def _ink(rng: np.random.Generator) -> tuple[int, int, int]:
    if rng.random() < 0.75:
        return (int(rng.integers(160, 220)), int(rng.integers(20, 60)), int(rng.integers(20, 55)))
    return (int(rng.integers(20, 50)), int(rng.integers(40, 80)), int(rng.integers(140, 200)))


def paint_distractors(image: np.ndarray, rng: np.random.Generator, boxes: np.ndarray | None = None) -> np.ndarray:
    """Draw non-seal red/blue ink onto a copy of ``image``. ``boxes`` are pixel xyxy."""
    canvas = Image.fromarray(np.asarray(image).copy(), "RGB")
    draw = ImageDraw.Draw(canvas)
    height, width = image.shape[:2]
    avoid = boxes if boxes is not None else np.zeros((0, 4), dtype=np.float32)
    cjk, latin = _fonts()
    for _ in range(int(rng.integers(5, 12))):
        kind = int(rng.integers(0, 5))
        color = _ink(rng)
        if kind == 0:
            y = int(rng.integers(8, max(9, height - 8)))
            rect = [0, y - 2, width, y + 3]
            if _hits(rect, avoid):
                continue
            draw.line([(int(rng.integers(0, width // 5)), y), (int(rng.integers(width // 2, width)), y)], fill=color, width=int(rng.integers(1, 4)))
        elif kind == 1:
            x = int(rng.integers(8, max(9, width - 8)))
            rect = [x - 2, 0, x + 3, height]
            if _hits(rect, avoid):
                continue
            draw.line([(x, 10), (x, height - 10)], fill=color, width=int(rng.integers(1, 3)))
        elif kind == 2:
            x1 = int(rng.integers(0, max(1, width - 40)))
            y1 = int(rng.integers(0, max(1, height - 20)))
            rect = [x1, y1, min(width, x1 + int(rng.integers(80, 280))), min(height, y1 + int(rng.integers(16, 36)))]
            if _hits(rect, avoid):
                continue
            text = "合同编号 2024-金额 人民币" if rng.random() < 0.5 else "INVOICE TOTAL 1280.00"
            font = cjk if rng.random() < 0.6 else latin
            draw.text((rect[0], rect[1]), text, font=font, fill=color)
        elif kind == 3:
            x1 = int(rng.integers(0, max(1, width // 2)))
            y1 = int(rng.integers(0, max(1, height // 2)))
            x2 = min(width - 1, x1 + int(rng.integers(120, 320)))
            y2 = min(height - 1, y1 + int(rng.integers(60, 180)))
            if _hits([x1, y1, x2, y2], avoid):
                continue
            rows = int(rng.integers(2, 5))
            cols = int(rng.integers(2, 5))
            for row in range(rows + 1):
                y = y1 + (y2 - y1) * row // rows
                draw.line([(x1, y), (x2, y)], fill=color, width=1)
            for col in range(cols + 1):
                x = x1 + (x2 - x1) * col // cols
                draw.line([(x, y1), (x, y2)], fill=color, width=1)
        else:
            cx = int(rng.integers(20, max(21, width - 20)))
            cy = int(rng.integers(20, max(21, height - 20)))
            rad = int(rng.integers(10, 28))
            rect = [cx - rad, cy - rad, cx + rad, cy + rad]
            if _hits(rect, avoid):
                continue
            draw.ellipse(rect, outline=color, width=2)
            draw.text((cx - 6, cy - 8), "A", font=latin, fill=color)
    return np.asarray(canvas)


def blank_distractor_page(rng: np.random.Generator, width: int = 900, height: int = 1200) -> np.ndarray:
    paper = np.full((height, width, 3), int(rng.integers(236, 252)), np.uint8)
    return paint_distractors(paper, rng, None)
