"""Whole-seal ring reader: a tiny ViT plus a transformer decoder.

The crop is not unwrapped. ImageNet-initialized ViT tokens keep the ring
layout, and the decoder writes the ring string one character at a time.
Greedy decoding is a fixed number of steps so the module exports to ONNX.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from rapidstamp_train.direct_aug import BOS_ID, Charset

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DirectReader(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, pretrained: bool = True):
        super().__init__()
        self.max_len = int(max_len)
        self.bos_id = BOS_ID
        self.encoder = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
        width = int(self.encoder.embed_dim)
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
        self.embed = nn.Embedding(vocab_size, width)
        self.out = nn.Linear(width, vocab_size, bias=False)
        self.out.weight = self.embed.weight

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_features(images)

    def decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        hidden = self.decoder(self.embed(tokens), memory, tgt_mask=causal)
        return self.out(hidden)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images), tokens)

    def greedy(self, images: torch.Tensor) -> torch.Tensor:
        memory = self.encode(images)
        tokens = torch.full((images.shape[0], 1), self.bos_id, dtype=torch.long, device=images.device)
        steps = []
        for _ in range(self.max_len):
            nxt = self.decode(memory, tokens)[:, -1].argmax(dim=-1)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
            steps.append(nxt)
        return torch.stack(steps, dim=1)


def teacher_tokens(target: torch.Tensor) -> torch.Tensor:
    """BOS followed by the target without its last pad/eos slot."""
    bos = torch.full((target.shape[0], 1), BOS_ID, dtype=target.dtype, device=target.device)
    return torch.cat([bos, target[:, :-1]], dim=1)


def ids_to_text(rows: torch.Tensor, charset: Charset) -> list[str]:
    return [charset.decode(row.tolist()) for row in rows]
