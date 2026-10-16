"""Detector box decode. The network predicts logits; sigmoid lives here."""

from __future__ import annotations

import numpy as np

from rapidstamp.geometry import DET_SIZE, GRID


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_detections(
    logits: np.ndarray,
    orig_w: int,
    orig_h: int,
    thresh: float = 0.5,
    nms_thresh: float = 0.45,
    det_size: int = DET_SIZE,
    grid: int = GRID,
) -> list[list[float]]:
    """Decode one detector output ``(5, G, G)`` to ``xywh`` boxes on the original image.

    Each returned row is ``x, y, w, h, score`` with ``x, y`` the top-left corner.
    """
    if logits.shape != (5, grid, grid):
        raise ValueError(f"expected (5, {grid}, {grid}) logits, got {logits.shape}")
    obj = _sigmoid(logits[0])
    tx = _sigmoid(logits[1])
    ty = _sigmoid(logits[2])
    tw = _sigmoid(logits[3])
    th = _sigmoid(logits[4])
    sx = orig_w / float(det_size)
    sy = orig_h / float(det_size)
    boxes: list[list[float]] = []
    rows, cols = np.where(obj >= thresh)
    for row, col in zip(rows.tolist(), cols.tolist()):
        cx = float(tx[row, col]) * det_size
        cy = float(ty[row, col]) * det_size
        bw = float(tw[row, col]) * det_size
        bh = float(th[row, col]) * det_size
        x = (cx - bw / 2.0) * sx
        y = (cy - bh / 2.0) * sy
        boxes.append([x, y, bw * sx, bh * sy, float(obj[row, col])])
    return nms(boxes, nms_thresh)


def iou(a: list[float], b: list[float]) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(boxes: list[list[float]], thresh: float) -> list[list[float]]:
    ordered = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list[list[float]] = []
    for box in ordered:
        if all(iou(box, other) < thresh for other in kept):
            kept.append(box)
    return kept


def match_iou(pred: list[float], gt: list[float]) -> float:
    return iou(pred, gt)
