"""Whole-seal ring reader: a tiny ViT plus a transformer decoder.

The crop is not unwrapped. Learned character queries cross-attend to the
ViT tokens, so each output position has to look at the seal. An
autoregressive head collapsed to one company name and ignored the image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from rapidstamp_train.direct_aug import Charset

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DirectReader(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, pretrained: bool = True):
        super().__init__()
        self.max_len = int(max_len)
        self.encoder = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
        width = int(self.encoder.embed_dim)
        if width % 3 != 0:
            raise RuntimeError(f"ViT width {width} is not divisible by 3 heads")
        layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=3,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.query = nn.Parameter(torch.randn(1, self.max_len, width) * 0.02)
        self.out = nn.Linear(width, vocab_size)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        memory = self.encode(images)
        query = self.query.expand(memory.shape[0], -1, -1)
        hidden = self.decoder(query, memory)
        return self.out(hidden)

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images).argmax(dim=-1)


def ids_to_text(rows: torch.Tensor, charset: Charset) -> list[str]:
    return [charset.decode(row.tolist()) for row in rows]
