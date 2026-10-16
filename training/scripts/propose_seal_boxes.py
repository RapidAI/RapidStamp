"""Propose stamp boxes on a page from red, blue, or dark ink.

This is an annotation aid. A person still decides which boxes are real seals.
"""

from __future__ import annotations

import cv2
import numpy as np


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        boxes.append((x, y, x + w, y + h, area))
    return boxes


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask, kernel)


def propose_seal_boxes(image: np.ndarray, limit: int = 8) -> list[list[int]]:
    """Return xyxy boxes in the original image, largest and roundest first."""
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    height, width = rgb.shape[:2]
    scale = 720.0 / max(height, width, 1)
    if scale < 1.0:
        small = cv2.resize(rgb, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
        scale = 1.0
    red_ch = small[..., 0].astype(np.int16)
    green = small[..., 1].astype(np.int16)
    blue = small[..., 2].astype(np.int16)
    red = (red_ch > 140) & (red_ch > green + 35) & (red_ch > blue + 35)
    blu = (blue > 110) & (blue > red_ch + 25) & (blue > green + 10)
    dark = (red_ch < 70) & (green < 70) & (blue < 70)
    mask = (red | blu | dark).astype(np.uint8) * 255
    mask = _dilate(mask, 2)
    min_side = max(12, int(min(small.shape[:2]) * 0.04))
    max_side = int(min(small.shape[:2]) * 0.72)
    page_area = small.shape[0] * small.shape[1]
    ranked = []
    for x1, y1, x2, y2, area in _components(mask):
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_side or bh < min_side or bw > max_side or bh > max_side:
            continue
        if area < 0.25 * bw * bh or area > 0.92 * page_area:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.45 or aspect > 2.4:
            continue
        roundness = 1.0 - abs(1.0 - aspect)
        fill = area / max(bw * bh, 1)
        score = fill * 0.5 + roundness + min(area, 8000) / 8000.0
        ranked.append((score, x1, y1, x2, y2))
    ranked.sort(reverse=True)
    kept: list[list[int]] = []
    for _score, x1, y1, x2, y2 in ranked:
        box = [int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)]
        if any(_iou(box, other) > 0.5 for other in kept):
            continue
        kept.append(box)
        if len(kept) >= limit:
            break
    return kept


def _iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)
