"""Heavier seal degradation for rows that look like the verified gold errors.

Rotation changes the start angle of the ring. Blur, stain, and overlapping
text are the other degradations on those failures. The gold crops themselves
are not training images.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from rapidstamp_train.direct_aug import _jpeg, _occlude, _overlay_text, _perspective


def _stain(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for _ in range(rng.randint(1, 4)):
        radius = rng.randint(max(8, width // 12), max(16, width // 3))
        x = rng.randint(0, max(0, width - 1))
        y = rng.randint(0, max(0, height - 1))
        color = rng.choice(
            [
                (140, 50, 30, 100),
                (180, 30, 30, 80),
                (30, 30, 30, 90),
                (190, 160, 70, 70),
            ]
        )
        draw.ellipse([x - radius, y - radius // 2, x + radius, y + radius // 2], fill=color)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def augment_hard(image: Image.Image, rng: random.Random, font_path: Path | None = None) -> Image.Image:
    """Full-turn rotation, then blur, stain, occlusion, and overlapping ink."""
    image = image.convert("RGB")
    image = image.rotate(rng.uniform(0.0, 360.0), resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    if rng.random() < 0.85:
        image = _perspective(image, rng, jitter=0.14)
    if rng.random() < 0.9:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.8, 3.2)))
    arr = np.asarray(image).astype(np.float32)
    if rng.random() < 0.7:
        noise = np.random.default_rng(rng.randrange(1 << 31)).normal(0.0, rng.uniform(6.0, 22.0), arr.shape)
        arr += noise
    if rng.random() < 0.55:
        keep = rng.uniform(0.40, 0.75)
        white = rng.uniform(210.0, 255.0)
        arr = arr * keep + white * (1.0 - keep)
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.8:
        image = _stain(image, rng)
    if rng.random() < 0.75:
        image = _occlude(image, rng)
    if rng.random() < 0.8:
        image = _overlay_text(image, rng, font_path)
    if rng.random() < 0.65:
        image = _jpeg(image, rng)
    return image
