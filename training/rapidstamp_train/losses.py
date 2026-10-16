"""Detection and recognition losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from rapidstamp_train.models import decode_boxes


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


def detection_loss(raw: torch.Tensor, obj: torch.Tensor, box: torch.Tensor, cls: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """``obj`` is (B, H, W), ``box`` is (B, H, W, 4), ``cls`` is (B, H, W)."""
    batch = obj.shape[0]
    height, width = obj.shape[-2:]
    raw = raw.view(batch, height * width, -1)
    obj_t = obj.view(batch, -1)
    box_t = box.view(batch, -1, 4)
    cls_t = cls.view(batch, -1)
    decoded = decode_boxes(raw)
    obj_logit = raw[..., 4]
    cls_logit = raw[..., 5:]
    prob = torch.sigmoid(obj_logit)
    bce = F.binary_cross_entropy_with_logits(obj_logit, obj_t, reduction="none")
    pt = prob * obj_t + (1.0 - prob) * (1.0 - obj_t)
    focal = (1.0 - pt) ** 2 * bce
    weight = torch.where(obj_t > 0.5, torch.full_like(obj_t, 10.0), torch.ones_like(obj_t))
    loss_obj = (focal * weight).mean()
    pos = obj_t > 0.5
    if pos.any():
        loss_box = _ciou(decoded[pos], box_t[pos]).mean()
        loss_cls = F.cross_entropy(cls_logit[pos], cls_t[pos])
    else:
        loss_box = raw.sum() * 0.0
        loss_cls = raw.sum() * 0.0
    total = loss_obj + 4.0 * loss_box + loss_cls
    return total, {
        "obj": float(loss_obj.detach()),
        "box": float(loss_box.detach()),
        "cls": float(loss_cls.detach()),
    }
