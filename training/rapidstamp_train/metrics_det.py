"""Detection mAP at IoU 0.5, computed from decoded boxes."""

from __future__ import annotations

import numpy as np

from rapidstamp.geometry import decode_detections, nms_xyxy
from rapidstamp.labels import CLASS_NAMES, SCORE_THRESH


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


def _ap(scores: list[float], matched: list[int], num_gt: int) -> float:
    if num_gt == 0:
        return float("nan")
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    hits = np.asarray(matched, dtype=np.float32)[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(1.0 - hits)
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def predictions_from_raw(raw: np.ndarray, score_thresh: float = SCORE_THRESH) -> list[tuple[int, float, np.ndarray]]:
    boxes, obj, classes = decode_detections(raw)
    class_ids = classes.argmax(axis=1)
    class_prob = classes.max(axis=1)
    scores = obj * class_prob
    keep = scores >= score_thresh
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    results = []
    for class_id in np.unique(class_ids):
        sel = np.where(class_ids == class_id)[0]
        kept = nms_xyxy(boxes[sel], scores[sel], 0.45)
        for index in kept:
            src = sel[index]
            results.append((int(class_id), float(scores[src]), boxes[src]))
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def average_precision(
    predictions: list[list[tuple[int, float, np.ndarray]]],
    ground_truth: list[list[tuple[int, np.ndarray]]],
    iou_thresh: float = 0.5,
) -> dict:
    per_class = {}
    aps = []
    for class_id, name in enumerate(CLASS_NAMES):
        scores: list[float] = []
        matched: list[int] = []
        num_gt = 0
        for preds, gts in zip(predictions, ground_truth):
            gt_boxes = np.array([box for cid, box in gts if cid == class_id], dtype=np.float32)
            num_gt += len(gt_boxes)
            used = np.zeros(len(gt_boxes), dtype=bool)
            class_preds = [(score, box) for cid, score, box in preds if cid == class_id]
            class_preds.sort(key=lambda item: item[0], reverse=True)
            for score, box in class_preds:
                scores.append(score)
                if len(gt_boxes) == 0:
                    matched.append(0)
                    continue
                ious = _iou_matrix(box[None, :], gt_boxes)[0]
                order = np.argsort(-ious)
                hit = 0
                for gi in order:
                    if ious[gi] < iou_thresh:
                        break
                    if not used[gi]:
                        used[gi] = True
                        hit = 1
                        break
                matched.append(hit)
        value = _ap(scores, matched, num_gt)
        per_class[name] = value
        if not np.isnan(value):
            aps.append(value)
    return {"map50": float(np.mean(aps) if aps else 0.0), "per_class_ap50": per_class}
