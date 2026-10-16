"""Tiny detector and character classifier.

Both networks use only the ops the C++ runtime implements: convolution,
ReLU, max-pool, a width transpose, and a linear layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rapidstamp.charset import CHARSET
from rapidstamp.geometry import DET_SIZE


def _pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


def _fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    weight = conv.weight.detach()
    bias = conv.bias.detach() if conv.bias is not None else torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    weight = weight * scale[:, None, None, None]
    bias = (bias - bn.running_mean.detach()) * scale + bn.bias.detach()
    return weight, bias


class ConvBNReLU(nn.Module):
    """Convolution, batch-norm, ReLU. Export folds the norm into the conv."""

    def __init__(self, in_c: int, out_c: int, k: int = 3, stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

    def export_ops(self) -> list[dict]:
        weight, bias = _fold_conv_bn(self.conv, self.bn)
        return [
            {
                "kind": "conv",
                "w": weight.cpu().numpy().astype("float32"),
                "b": bias.cpu().numpy().astype("float32"),
                "stride_h": int(self.conv.stride[0]),
                "stride_w": int(self.conv.stride[1]),
                "pad_h": int(self.conv.padding[0]),
                "pad_w": int(self.conv.padding[1]),
                "groups": int(self.conv.groups),
            },
            {"kind": "relu"},
        ]


class Conv(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 1, stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=stride, padding=padding, groups=groups, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def export_ops(self) -> list[dict]:
        return [
            {
                "kind": "conv",
                "w": self.conv.weight.detach().cpu().numpy().astype("float32"),
                "b": self.conv.bias.detach().cpu().numpy().astype("float32"),
                "stride_h": int(self.conv.stride[0]),
                "stride_w": int(self.conv.stride[1]),
                "pad_h": int(self.conv.padding[0]),
                "pad_w": int(self.conv.padding[1]),
                "groups": int(self.conv.groups),
            }
        ]


class MaxPool(nn.Module):
    def __init__(self, k, stride=None):
        super().__init__()
        if isinstance(k, int):
            k = (k, k)
        if stride is None:
            stride = k
        if isinstance(stride, int):
            stride = (stride, stride)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)

    def export_ops(self) -> list[dict]:
        kh, kw = _pair(self.pool.kernel_size)
        sh, sw = _pair(self.pool.stride)
        ph, pw = _pair(self.pool.padding)
        return [
            {
                "kind": "maxpool",
                "kh": int(kh),
                "kw": int(kw),
                "stride_h": int(sh),
                "stride_w": int(sw),
                "pad_h": int(ph),
                "pad_w": int(pw),
            }
        ]


class Gap(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))

    def export_ops(self) -> list[dict]:
        return [{"kind": "gap"}]


class ToSeq(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Height has been pooled to 1. Squeeze that axis and put width first.
        return x[:, :, 0, :].transpose(1, 2)

    def export_ops(self) -> list[dict]:
        return [{"kind": "to_seq"}]


class Linear(nn.Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def export_ops(self) -> list[dict]:
        return [
            {
                "kind": "linear",
                "w": self.fc.weight.detach().cpu().numpy().astype("float32"),
                "b": self.fc.bias.detach().cpu().numpy().astype("float32"),
            }
        ]


class SealDetTiny(nn.Module):
    """Depthwise-separable single-class detector. Output is ``(N, 5, 8, 8)`` logits."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConvBNReLU(3, 16, 3, stride=2, padding=1),
                ConvBNReLU(16, 16, 3, stride=1, padding=1, groups=16),
                ConvBNReLU(16, 32, 1, stride=1, padding=0),
                MaxPool(2),
                ConvBNReLU(32, 32, 3, stride=1, padding=1, groups=32),
                ConvBNReLU(32, 48, 1, stride=1, padding=0),
                MaxPool(2),
                ConvBNReLU(48, 48, 3, stride=1, padding=1, groups=48),
                ConvBNReLU(48, 64, 1, stride=1, padding=0),
                MaxPool(2),
                Conv(64, 5, 1, stride=1, padding=0),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SealRecTiny(nn.Module):
    """Per-character classifier. Input is a 32x32 crop, output is charset logits."""

    def __init__(self, n_classes: int | None = None):
        super().__init__()
        n_classes = len(CHARSET) if n_classes is None else n_classes
        self.layers = nn.ModuleList(
            [
                ConvBNReLU(1, 16, 3, stride=1, padding=1),
                MaxPool(2),
                ConvBNReLU(16, 32, 3, stride=1, padding=1),
                MaxPool(2),
                ConvBNReLU(32, 48, 3, stride=1, padding=1),
                MaxPool(2),
                Gap(),
                Linear(48, n_classes),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def collect_ops(model: nn.Module) -> list[dict]:
    ops: list[dict] = []
    for layer in model.layers:  # type: ignore[attr-defined]
        ops.extend(layer.export_ops())
    return ops


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def detector_input_shape() -> tuple[int, int, int, int]:
    return (1, 3, DET_SIZE, DET_SIZE)
