"""Geometry shared by the trainer and the ONNX runtime pipeline.

The bilinear sampler is intentionally small and explicit so the C++ SDK can
match it. Coordinates use image space: x right, y down, angle 0 at +x, and
angle increasing clockwise.
"""

from __future__ import annotations

import numpy as np

from rapidstamp.labels import (
    BOX_LOGIT_LIMIT,
    DET_SIZE,
    GRID,
    PAD_VALUE,
    REC_H,
    REC_W,
    UNWRAP_R_IN,
    UNWRAP_R_OUT,
    UNWRAP_SPAN_DEG,
    UNWRAP_START_DEG,
)


def sample_bilinear(image: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Edge-clamped bilinear sample. ``image`` is HWC or HW. Returns float32."""
    squeeze = image.ndim == 2
    if squeeze:
        image = image[..., None]
    height, width = image.shape[:2]
    sx = np.clip(np.asarray(sx, dtype=np.float32), 0.0, width - 1.0)
    sy = np.clip(np.asarray(sy, dtype=np.float32), 0.0, height - 1.0)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (sx - x0).astype(np.float32)[..., None]
    wy = (sy - y0).astype(np.float32)[..., None]
    img = image.astype(np.float32, copy=False)
    top = img[y0, x0] * (1.0 - wx) + img[y0, x1] * wx
    bot = img[y1, x0] * (1.0 - wx) + img[y1, x1] * wx
    out = top * (1.0 - wy) + bot * wy
    if squeeze:
        out = out[..., 0]
    return out


def resize_bilinear(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height == out_h and width == out_w:
        return np.ascontiguousarray(image)
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (height / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (width / out_w) - 0.5
    sy = np.broadcast_to(ys[:, None], (out_h, out_w))
    sx = np.broadcast_to(xs[None, :], (out_h, out_w))
    sampled = sample_bilinear(image, sx, sy)
    if image.dtype == np.uint8:
        return np.clip(sampled + 0.5, 0, 255).astype(np.uint8)
    return sampled


def letterbox(
    image: np.ndarray,
    size: int = DET_SIZE,
    pad_value: int = PAD_VALUE,
) -> tuple[np.ndarray, float, int, int]:
    """Fit ``image`` into a square canvas. Returns canvas, scale, pad_x, pad_y."""
    height, width = image.shape[:2]
    scale = size / float(max(height, width))
    new_w = min(size, max(1, int(round(width * scale))))
    new_h = min(size, max(1, int(round(height * scale))))
    resized = resize_bilinear(image, new_w, new_h)
    canvas = np.full((size, size, image.shape[2]), pad_value, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, float(scale), int(pad_x), int(pad_y)


def map_boxes_to_letterbox(
    boxes: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    size: int,
) -> np.ndarray:
    """Pixel xyxy boxes -> normalized xyxy on the letterboxed canvas."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.empty((len(boxes), 4), dtype=np.float32)
    out[:, 0] = (boxes[:, 0] * scale + pad_x) / size
    out[:, 1] = (boxes[:, 1] * scale + pad_y) / size
    out[:, 2] = (boxes[:, 2] * scale + pad_x) / size
    out[:, 3] = (boxes[:, 3] * scale + pad_y) / size
    return np.clip(out, 0.0, 1.0)


def map_boxes_from_letterbox(
    boxes: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    size: int,
) -> np.ndarray:
    """Normalized letterbox xyxy -> original-image pixel xyxy."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    out = np.empty((len(boxes), 4), dtype=np.float32)
    out[:, 0] = (boxes[:, 0] * size - pad_x) / scale
    out[:, 1] = (boxes[:, 1] * size - pad_y) / scale
    out[:, 2] = (boxes[:, 2] * size - pad_x) / scale
    out[:, 3] = (boxes[:, 3] * size - pad_y) / scale
    return out


def corrected_unwrap_start(base_start_deg: float, rotation_ccw_deg: float) -> float:
    """Keep ring text left-aligned after a counter-clockwise rotation.

    Angle ``phi`` increases clockwise. Rotating the image counter-clockwise
    by ``rotation_ccw_deg`` decreases the ``phi`` of every seal feature by the
    same amount, so the unwrap start has to move with it.
    """
    return base_start_deg - float(rotation_ccw_deg)


def polar_unwrap(
    image: np.ndarray,
    out_w: int = REC_W,
    out_h: int = REC_H,
    r_in: float = UNWRAP_R_IN,
    r_out: float = UNWRAP_R_OUT,
    start_deg: float = UNWRAP_START_DEG,
    span_deg: float = UNWRAP_SPAN_DEG,
) -> np.ndarray:
    """Unwrap an ellipse inscribed in ``image`` into a horizontal strip.

    Row 0 is the outer radius. Column 0 is ``start_deg``.
    """
    height, width = image.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    rad_x = (width - 1) / 2.0
    rad_y = (height - 1) / 2.0
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / float(out_w)
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / float(out_h)
    angles = np.deg2rad(start_deg + xs * span_deg)
    ts = r_out + (r_in - r_out) * ys
    sx = cx + (rad_x * ts)[:, None] * np.cos(angles)[None, :]
    sy = cy + (rad_y * ts)[:, None] * np.sin(angles)[None, :]
    sampled = sample_bilinear(image, sx, sy)
    if image.dtype == np.uint8:
        return np.clip(sampled + 0.5, 0, 255).astype(np.uint8)
    return sampled


def pad_to_square(image: np.ndarray, pad_value: int = 255) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    if image.ndim == 2:
        canvas = np.full((side, side), pad_value, dtype=image.dtype)
    else:
        canvas = np.full((side, side, image.shape[2]), pad_value, dtype=image.dtype)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def grid_unroll(
    image: np.ndarray,
    rows: int,
    cols: int,
    out_h: int = REC_H,
    out_w: int = REC_W,
) -> np.ndarray:
    """Read a personal seal in row-major order into one horizontal strip."""
    height, width = image.shape[:2]
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    cell_w = max(1, out_w // (rows * cols))
    cells = []
    for row in range(rows):
        for col in range(cols):
            y1 = int(round(row * height / rows))
            y2 = int(round((row + 1) * height / rows))
            x1 = int(round(col * width / cols))
            x2 = int(round((col + 1) * width / cols))
            cell = image[y1:y2, x1:x2]
            if cell.size == 0:
                cell = np.full((out_h, cell_w, image.shape[2]), 255, np.uint8)
            cells.append(resize_bilinear(cell, cell_w, out_h))
    strip = np.concatenate(cells, axis=1)
    if strip.shape[1] != out_w or strip.shape[0] != out_h:
        strip = resize_bilinear(strip, out_w, out_h)
    return strip


def hline_band(
    image: np.ndarray,
    out_h: int = REC_H,
    out_w: int = REC_W,
    y0: float = 0.22,
    y1: float = 0.78,
) -> np.ndarray:
    height = image.shape[0]
    top = max(0, int(round(height * y0)))
    bot = min(height, max(top + 1, int(round(height * y1))))
    return resize_bilinear(image[top:bot], out_w, out_h)


def to_gray01(image: np.ndarray) -> np.ndarray:
    """RGB or gray uint8 -> float32 gray in 0..1, shape (H, W)."""
    if image.ndim == 2:
        return image.astype(np.float32) / 255.0
    rgb = image.astype(np.float32)
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[index], x1[rest])
        yy1 = np.maximum(y1[index], y1[rest])
        xx2 = np.minimum(x2[index], x2[rest])
        yy2 = np.minimum(y2[index], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[index] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou <= iou_thresh]
    return keep


def decode_detections(
    pred: np.ndarray,
    grid: int = GRID,
    det_size: int = DET_SIZE,
    box_logit_limit: float = BOX_LOGIT_LIMIT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw detector head.

    ``pred`` is (A, 5+C) or (1, A, 5+C) with channels
    tx, ty, tw, th, objectness logit, class logits.
    Returns normalized xyxy, objectness probability, class probabilities.
    """
    raw = np.asarray(pred, dtype=np.float32)
    if raw.ndim == 3:
        raw = raw[0]
    count = grid * grid
    if raw.shape[0] != count:
        raise ValueError(f"expected {count} cells, got {raw.shape}")
    tx = raw[:, 0]
    ty = raw[:, 1]
    tw = np.clip(raw[:, 2], -box_logit_limit, box_logit_limit)
    th = np.clip(raw[:, 3], -box_logit_limit, box_logit_limit)
    gy = np.arange(count, dtype=np.float32) // grid
    gx = np.arange(count, dtype=np.float32) % grid
    cx = (gx + 1.0 / (1.0 + np.exp(-tx))) / grid
    cy = (gy + 1.0 / (1.0 + np.exp(-ty))) / grid
    bw = np.exp(tw) / grid
    bh = np.exp(th) / grid
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    obj = 1.0 / (1.0 + np.exp(-raw[:, 4]))
    logits = raw[:, 5:]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    classes = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-6)
    return boxes.astype(np.float32), obj.astype(np.float32), classes.astype(np.float32)
