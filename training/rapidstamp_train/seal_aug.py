"""Extra on-the-fly augmentation for unwrapped ring strips.

PaddleOCR ``RecAug`` already applies blur, noise, color jitter, a small
perspective, and a crop. This op adds JPEG recompression and a few degrees
of start-angle jitter. The jitter is a horizontal roll of a 360° unwrap.
It runs only when the file name contains ``__gap``, which means the bottom
gap was wide enough that the roll stays inside the gap.
"""

from __future__ import annotations

import random

import numpy as np


def roll_strip(image: np.ndarray, degrees: float) -> np.ndarray:
    width = image.shape[1]
    shift = int(round(float(degrees) / 360.0 * width))
    if width <= 1 or shift % width == 0:
        return image
    return np.roll(image, shift, axis=1)


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    try:
        import cv2

        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None and decoded.shape == image.shape:
                return decoded
    except Exception:
        pass
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    decoded = np.asarray(Image.open(buffer).convert("RGB"))
    if decoded.shape != image.shape:
        return image
    return decoded


class SealStripAug:
    def __init__(
        self,
        jpeg_prob: float = 0.4,
        roll_prob: float = 0.7,
        roll_deg: float = 8.0,
        jpeg_quality=(40, 85),
        **kwargs,
    ):
        del kwargs
        self.jpeg_prob = float(jpeg_prob)
        self.roll_prob = float(roll_prob)
        self.roll_deg = float(roll_deg)
        self.jpeg_quality = (int(jpeg_quality[0]), int(jpeg_quality[1]))

    def __call__(self, data):
        image = data.get("image")
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            return data
        path = str(data.get("img_path") or "")
        if "__gap" in path and random.random() <= self.roll_prob:
            degrees = random.uniform(-self.roll_deg, self.roll_deg)
            image = roll_strip(image, degrees)
        if random.random() <= self.jpeg_prob:
            quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
            image = _jpeg_roundtrip(image, quality)
        data["image"] = image
        return data
