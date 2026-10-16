"""YOLOX-Nano seal detector.

The backbone and head follow Megvii YOLOX-Nano (depth 0.33, width 0.25,
depthwise), Apache-2.0, so the published COCO checkpoint can be loaded.
Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
This file is a weight-compatible reimplementation, not the YOLOX trainer.

Ultralytics YOLO is AGPL-3.0 and is not used.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from rapidstamp.labels import BOX_LOGIT_LIMIT, CLASS_NAMES, DET_SIZE

_STRIDES = (8, 16, 32)
_COCO_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"


def yolox_anchors(det_size: int = DET_SIZE, device: torch.device | None = None, dtype: torch.dtype = torch.float32):
    """Grid index and stride for each anchor, in stride-8, stride-16, stride-32 order."""
    gx_all = []
    gy_all = []
    stride_all = []
    for stride in _STRIDES:
        cells = det_size // stride
        gy, gx = torch.meshgrid(
            torch.arange(cells, device=device, dtype=dtype),
            torch.arange(cells, device=device, dtype=dtype),
            indexing="ij",
        )
        gx_all.append(gx.reshape(-1))
        gy_all.append(gy.reshape(-1))
        stride_all.append(torch.full((cells * cells,), float(stride), device=device, dtype=dtype))
    return torch.cat(gx_all), torch.cat(gy_all), torch.cat(stride_all)


def decode_yolox(raw: torch.Tensor, det_size: int = DET_SIZE) -> torch.Tensor:
    """Raw logits (B, A, 5+C) -> xyxy, objectness, class probabilities, all in 0..1."""
    gx, gy, stride = yolox_anchors(det_size, raw.device, raw.dtype)
    cx = (torch.sigmoid(raw[..., 0]) + gx) * stride / det_size
    cy = (torch.sigmoid(raw[..., 1]) + gy) * stride / det_size
    bw = torch.exp(raw[..., 2].clamp(-BOX_LOGIT_LIMIT, BOX_LOGIT_LIMIT)) * stride / det_size
    bh = torch.exp(raw[..., 3].clamp(-BOX_LOGIT_LIMIT, BOX_LOGIT_LIMIT)) * stride / det_size
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1).clamp(0, 1)
    obj = torch.sigmoid(raw[..., 4:5])
    classes = torch.sigmoid(raw[..., 5:])
    return torch.cat([boxes, obj, classes], dim=-1)


class BaseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ksize: int, stride: int, groups: int = 1, act: str = "silu") -> None:
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad, groups=groups, bias=False)
        # YOLOX trains these BatchNorm layers with eps 1e-3 and momentum 0.03.
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ksize: int, stride: int = 1, act: str = "silu") -> None:
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize, stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pconv(self.dconv(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, expansion: float = 0.5, depthwise: bool = False, act: str = "silu") -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.conv2 = conv(hidden, out_channels, 3, 1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple[int, ...] = (5, 9, 13), act: str = "silu") -> None:
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes])
        self.conv2 = BaseConv(hidden * (len(kernel_sizes) + 1), out_channels, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.m], dim=1)
        return self.conv2(x)


class CSPLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n: int = 1, shortcut: bool = True, expansion: float = 0.5, depthwise: bool = False, act: str = "silu") -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.conv2 = BaseConv(in_channels, hidden, 1, 1, act=act)
        self.conv3 = BaseConv(2 * hidden, out_channels, 1, 1, act=act)
        self.m = nn.Sequential(*[Bottleneck(hidden, hidden, shortcut, 1.0, depthwise, act=act) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.m(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class Focus(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, ksize: int = 1, stride: int = 1, act: str = "silu") -> None:
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1))


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul: float, wid_mul: float, depthwise: bool = False, act: str = "silu") -> None:
        super().__init__()
        conv = DWConv if depthwise else BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.dark2 = nn.Sequential(
            conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )
        self.dark3 = nn.Sequential(
            conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )
        self.dark4 = nn.Sequential(
            conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )
        self.dark5 = nn.Sequential(
            conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, act=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.dark2(x)
        dark3 = self.dark3(x)
        dark4 = self.dark4(dark3)
        dark5 = self.dark5(dark4)
        return {"dark3": dark3, "dark4": dark4, "dark5": dark5}


class YOLOPAFPN(nn.Module):
    def __init__(self, depth: float = 0.33, width: float = 0.25, depthwise: bool = True, act: str = "silu") -> None:
        super().__init__()
        in_channels = [256, 512, 1024]
        conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[0] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv2 = conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(int(2 * in_channels[0] * width), int(in_channels[1] * width), round(3 * depth), False, depthwise=depthwise, act=act)
        self.bu_conv1 = conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(int(2 * in_channels[1] * width), int(in_channels[2] * width), round(3 * depth), False, depthwise=depthwise, act=act)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        x2, x1, x0 = features["dark3"], features["dark4"], features["dark5"]
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat([self.upsample(fpn_out0), x1], 1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat([self.upsample(fpn_out1), x2], 1))
        pan_out1 = self.C3_n3(torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1))
        pan_out0 = self.C3_n4(torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1))
        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    def __init__(self, num_classes: int, width: float = 0.25, depthwise: bool = True, act: str = "silu") -> None:
        super().__init__()
        self.num_classes = num_classes
        in_channels = [256, 512, 1024]
        conv = DWConv if depthwise else BaseConv
        hidden = int(256 * width)
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        for channels in in_channels:
            self.stems.append(BaseConv(int(channels * width), hidden, 1, 1, act=act))
            self.cls_convs.append(nn.Sequential(conv(hidden, hidden, 3, 1, act=act), conv(hidden, hidden, 3, 1, act=act)))
            self.reg_convs.append(nn.Sequential(conv(hidden, hidden, 3, 1, act=act), conv(hidden, hidden, 3, 1, act=act)))
            self.cls_preds.append(nn.Conv2d(hidden, num_classes, 1))
            self.reg_preds.append(nn.Conv2d(hidden, 4, 1))
            self.obj_preds.append(nn.Conv2d(hidden, 1, 1))

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        outputs = []
        for level, feat in enumerate(features):
            feat = self.stems[level](feat)
            cls_output = self.cls_preds[level](self.cls_convs[level](feat))
            reg_feat = self.reg_convs[level](feat)
            reg_output = self.reg_preds[level](reg_feat)
            obj_output = self.obj_preds[level](reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output.flatten(2).permute(0, 2, 1))
        return torch.cat(outputs, 1)


class SealDet(nn.Module):
    """COCO-pretrained YOLOX-Nano fine-tuned as an 8-class seal detector.

    Training returns raw logits (B, A, 5+C). Eval and ONNX export return
    decoded rows: normalized xyxy, objectness probability, class probabilities.
    """

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = YOLOPAFPN(0.33, 0.25, depthwise=True)
        self.head = YOLOXHead(num_classes, width=0.25, depthwise=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.backbone(x))
        if self.training:
            return raw
        return decode_yolox(raw, x.shape[-1])


def load_yolox_weights(model: SealDet, path: str | Path) -> tuple[int, int]:
    """Copy every tensor whose name and shape match. Class heads may differ."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    own = model.state_dict()
    copied = 0
    for key, value in state.items():
        if key in own and tuple(own[key].shape) == tuple(value.shape):
            own[key] = value.detach().cpu()
            copied += 1
    skipped = len(state) - copied
    model.load_state_dict(own)
    cls_loaded = any(key.startswith("head.cls_preds") and key in state and tuple(state[key].shape) == tuple(own[key].shape) for key in own if key.startswith("head.cls_preds"))
    if not cls_loaded:
        prior = -math.log((1 - 0.01) / 0.01)
        for conv in model.head.cls_preds:
            nn.init.zeros_(conv.weight)
            conv.bias.data.fill_(prior)
    return copied, skipped
