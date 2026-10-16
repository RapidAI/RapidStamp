"""Tiny seal detector and CTC recognizer.

Both nets are original depthwise / plain conv stacks. They are sized for CPU
ONNX Runtime, not copied from Ultralytics or Paddle.
"""

from __future__ import annotations

import torch
from torch import nn

from rapidstamp.labels import BOX_LOGIT_LIMIT, CLASS_NAMES, DET_SIZE, GRID


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class DSConv(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin),
            nn.SiLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class SealDet(nn.Module):
    """Single-scale detector. Output is (B, grid*grid, 5+num_classes)."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            ConvBNAct(3, 32, 2),
            DSConv(32, 64, 2),
            DSConv(64, 96, 1),
            DSConv(96, 128, 2),
            DSConv(128, 160, 1),
            DSConv(160, 192, 2),
            DSConv(192, 192, 1),
            ConvBNAct(192, 192, 1),
        )
        self.head = nn.Conv2d(192, 5 + num_classes, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.head.bias.data[2] = 1.5
        self.head.bias.data[3] = 1.5
        self.head.bias.data[4] = -2.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(self.features(x))
        feat = feat.permute(0, 2, 3, 1).contiguous()
        return feat.view(feat.shape[0], -1, feat.shape[-1])


class SealRec(nn.Module):
    """CNN-CTC recognizer. Output is (B, T=96, num_classes) including blank.

    Wider than the first recognizer so similar CJK glyphs are less likely to
    collapse onto the synthetic company-name list. Dropout is training-only.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBNAct(1, 48, 1),
            ConvBNAct(48, 64, 1),
            nn.MaxPool2d(2, 2),
            ConvBNAct(64, 128, 1),
            ConvBNAct(128, 160, 1),
            nn.MaxPool2d(2, 2),
            ConvBNAct(160, 256, 1),
            ConvBNAct(256, 256, 1),
            nn.MaxPool2d((2, 1), (2, 1)),
            ConvBNAct(256, 384, 1),
            nn.MaxPool2d((2, 1), (2, 1)),
            ConvBNAct(384, 384, 1),
            nn.Conv2d(384, 384, kernel_size=(2, 1), bias=False),
            nn.BatchNorm2d(384),
            nn.SiLU(inplace=True),
        )
        self.seq = nn.Sequential(
            nn.Conv1d(384, 384, 3, padding=1, bias=False),
            nn.BatchNorm1d(384),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(384, 384, 3, padding=1, bias=False),
            nn.BatchNorm1d(384),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(384, 384, 3, padding=1, bias=False),
            nn.BatchNorm1d(384),
            nn.SiLU(inplace=True),
        )
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x).squeeze(2)
        feat = self.seq(feat).permute(0, 2, 1)
        return self.fc(feat)


def decode_boxes(raw: torch.Tensor, grid: int = GRID, det_size: int = DET_SIZE) -> torch.Tensor:
    """Differentiable xyxy in normalized letterbox coordinates."""
    tx = raw[..., 0]
    ty = raw[..., 1]
    tw = raw[..., 2].clamp(-BOX_LOGIT_LIMIT, BOX_LOGIT_LIMIT)
    th = raw[..., 3].clamp(-BOX_LOGIT_LIMIT, BOX_LOGIT_LIMIT)
    device = raw.device
    gy = torch.arange(grid, device=device).view(grid, 1).expand(grid, grid).reshape(-1).float()
    gx = torch.arange(grid, device=device).view(1, grid).expand(grid, grid).reshape(-1).float()
    cx = (gx + torch.sigmoid(tx)) / grid
    cy = (gy + torch.sigmoid(ty)) / grid
    bw = torch.exp(tw) / grid
    bh = torch.exp(th) / grid
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def adapt_recognizer(checkpoint: dict, charset: list[str]) -> SealRec:
    """Load a CTC checkpoint, expanding the final layer when the charset grew."""
    model = SealRec(num_classes=1 + len(charset))
    state = checkpoint["model"]
    old_charset = checkpoint.get("charset")
    old_fc = state.get("fc.weight")
    if old_fc is None or tuple(old_fc.shape)[1:] != tuple(model.fc.weight.shape)[1:]:
        # Channel width changed. The old backbone cannot be copied.
        return model
    if not old_charset or list(old_charset) == list(charset):
        model.load_state_dict(state)
        return model
    own = model.state_dict()
    for key, value in state.items():
        if key.startswith("fc."):
            continue
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            own[key] = value.detach().cpu()
    old_weight = state["fc.weight"].detach().cpu()
    old_bias = state["fc.bias"].detach().cpu()
    weight = own["fc.weight"]
    bias = own["fc.bias"]
    weight.zero_()
    bias.zero_()
    weight[0].copy_(old_weight[0])
    bias[0].copy_(old_bias[0])
    index = {ch: i + 1 for i, ch in enumerate(charset)}
    for old_index, ch in enumerate(old_charset):
        new_index = index.get(ch)
        if new_index is None:
            continue
        weight[new_index].copy_(old_weight[old_index + 1])
        bias[new_index].copy_(old_bias[old_index + 1])
    own["fc.weight"] = weight
    own["fc.bias"] = bias
    model.load_state_dict(own)
    return model
