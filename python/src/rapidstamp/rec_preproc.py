"""Recognizer input tensors. The C++ engine matches these formulas."""

from __future__ import annotations

import numpy as np

from rapidstamp.geometry import resize_bilinear


def ppocr_width(src_h: int, src_w: int, height: int = 48, max_width: int = 3200) -> int:
    if src_h < 1 or src_w < 1:
        return 8
    width = int(round(height * src_w / src_h))
    return max(8, min(max_width, width))


def ppocr_nchw(rgb: np.ndarray, height: int = 48, max_width: int = 3200) -> np.ndarray:
    """BGR, height 48, (x / 255 - 0.5) / 0.5. Matches PP-OCRv5 mobile rec."""
    width = ppocr_width(int(rgb.shape[0]), int(rgb.shape[1]), height, max_width)
    if rgb.shape[0] != height or rgb.shape[1] != width:
        rgb = resize_bilinear(rgb, width, height)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1], dtype=np.float32)
    tensor = (bgr / 255.0 - 0.5) / 0.5
    return np.transpose(tensor, (2, 0, 1))
