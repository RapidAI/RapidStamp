"""Detection and recognition losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rapidstamp.labels import DET_SIZE
from rapidstamp_train.yolox_nano import decode_yolox, yolox_anchors


def _ciou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred
    t = target
    x1 = torch.maximum(p[:, 0], t[:, 0])
    y1 = torch.maximum(p[:, 1], t[:, 1])
    x2 = torch.minimum(p[:, 2], t[:, 2])
    y2 = torch.minimum(p[:, 3], t[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    area_t = (t[:, 2] - t[:, 0]).clamp(min=0) * (t[:, 3] - t[:, 1]).clamp(min=0)
    iou = inter / (area_p + area_t - inter + 1e-6)
    pcx = (p[:, 0] + p[:, 2]) * 0.5
    pcy = (p[:, 1] + p[:, 3]) * 0.5
    tcx = (t[:, 0] + t[:, 2]) * 0.5
    tcy = (t[:, 1] + t[:, 3]) * 0.5
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    cx1 = torch.minimum(p[:, 0], t[:, 0])
    cy1 = torch.minimum(p[:, 1], t[:, 1])
    cx2 = torch.maximum(p[:, 2], t[:, 2])
    cy2 = torch.maximum(p[:, 3], t[:, 3])
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + 1e-6
    pw = (p[:, 2] - p[:, 0]).clamp(min=1e-6)
    ph = (p[:, 3] - p[:, 1]).clamp(min=1e-6)
    tw = (t[:, 2] - t[:, 0]).clamp(min=1e-6)
    th = (t[:, 3] - t[:, 1]).clamp(min=1e-6)
    v = (4.0 / (math.pi**2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + 1e-6)
    return 1.0 - iou + rho2 / c2 + alpha * v


def detection_loss(raw: torch.Tensor, targets: list[torch.Tensor], det_size: int = DET_SIZE) -> tuple[torch.Tensor, dict]:
    """YOLOX logits (B, A, 5+C) against a list of (N, 5) rows: class, x1, y1, x2, y2.

    A ground-truth box is assigned to the center cell of every stride whose
    pixel size can hold that box. Pages with no box are all-negative.
    """
    decoded = decode_yolox(raw, det_size)
    boxes = decoded[..., :4]
    obj_logit = raw[..., 4]
    cls_logit = raw[..., 5:]
    gx, gy, stride = yolox_anchors(det_size, raw.device, raw.dtype)
    obj_t = torch.zeros_like(obj_logit)
    pos = torch.zeros_like(obj_logit, dtype=torch.bool)
    box_t = torch.zeros_like(boxes)
    cls_t = torch.zeros(raw.shape[0], raw.shape[1], dtype=torch.long, device=raw.device)
    for batch_index, target in enumerate(targets):
        if target is None or target.numel() == 0:
            continue
        target = target.to(raw.device)
        for row in target:
            class_id = int(row[0].item())
            x1, y1, x2, y2 = [float(v) for v in row[1:].tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            cx = (x1 + x2) * 0.5 * det_size
            cy = (y1 + y2) * 0.5 * det_size
            side = max((x2 - x1) * det_size, (y2 - y1) * det_size)
            cell_x = cx / stride
            cell_y = cy / stride
            center = (cell_x >= gx) & (cell_x < gx + 1) & (cell_y >= gy) & (cell_y < gy + 1)
            sized = (side >= stride) & (side <= stride * 10)
            chosen = center & sized
            if not bool(chosen.any()):
                chosen = center
            if not bool(chosen.any()):
                continue
            obj_t[batch_index, chosen] = 1.0
            pos[batch_index, chosen] = True
            box_t[batch_index, chosen, 0] = x1
            box_t[batch_index, chosen, 1] = y1
            box_t[batch_index, chosen, 2] = x2
            box_t[batch_index, chosen, 3] = y2
            cls_t[batch_index, chosen] = class_id
    prob = torch.sigmoid(obj_logit)
    bce = F.binary_cross_entropy_with_logits(obj_logit, obj_t, reduction="none")
    pt = prob * obj_t + (1.0 - prob) * (1.0 - obj_t)
    focal = ((1.0 - pt) ** 2) * bce
    # Mean-over-anchors drowns the one or two positive cells in 2100 negatives.
    # Divide by the positive count, the same normalization RetinaNet uses.
    num_pos = pos.sum().clamp(min=1).to(focal.dtype)
    loss_obj = focal.sum() / num_pos
    if pos.any():
        loss_box = _ciou(boxes[pos], box_t[pos]).mean()
        # Inference scores with an independent sigmoid, so the class head is
        # trained with one-hot BCE. Softmax cross-entropy can look good while
        # every logit stays negative and the sigmoid score never clears 0.25.
        cls_pos = cls_logit[pos]
        cls_target = torch.zeros_like(cls_pos)
        cls_target.scatter_(1, cls_t[pos].unsqueeze(1), 1.0)
        loss_cls = F.binary_cross_entropy_with_logits(cls_pos, cls_target)
        pos_obj = prob[pos].mean()
        pos_cls = torch.sigmoid(cls_pos).max(dim=-1).values.mean()
    else:
        loss_box = raw.sum() * 0.0
        loss_cls = raw.sum() * 0.0
        pos_obj = prob.new_zeros(())
        pos_cls = prob.new_zeros(())
    total = loss_obj + 4.0 * loss_box + loss_cls
    return total, {
        "obj": float(loss_obj.detach()),
        "box": float(loss_box.detach()),
        "cls": float(loss_cls.detach()),
        "pos_obj": float(pos_obj.detach()),
        "pos_cls": float(pos_cls.detach()),
    }
