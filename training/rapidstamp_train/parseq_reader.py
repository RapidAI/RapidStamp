"""Whole-crop ring reader: a ViT encoder and a two-pass PARSeq decoder.

Pass one reads the seal with empty context. Pass two cross-attends again,
conditioned on that guess, and corrects it. Both passes are a fixed graph,
so the greedy decode exports to ONNX for the C++ runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

IMAGE_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HybridEncoder(nn.Module):
    """Five stride-2 convolutions, then a transformer over the 12×12 grid.

    384 input becomes 144 tokens. The convolutions keep the stroke detail
    that a patch-32 ViT would average away, without a 24×24 attention map.
    """

    def __init__(self, width: int = 384, layers: int = 6, heads: int = 6):
        super().__init__()
        self.embed_dim = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, 144, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.stem(images).flatten(2).transpose(1, 2)
        return self.norm(self.blocks(tokens + self.pos))


class ParseqReader(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        backbone: str = "vit_small_patch16_384",
        pretrained: bool = True,
        checkpointing: bool = True,
        img_size: int = IMAGE_SIZE,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_len = int(max_len)
        self.backbone_name = backbone
        self.image_size = int(img_size)
        if backbone == "hybrid384":
            self.encoder = HybridEncoder()
            width = self.encoder.embed_dim
        else:
            self.encoder = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
                img_size=self.image_size,
            )
            width = int(self.encoder.embed_dim)
            if checkpointing and hasattr(self.encoder, "set_grad_checkpointing"):
                self.encoder.set_grad_checkpointing(True)
        if width % 6 != 0:
            raise RuntimeError(f"width {width} is not divisible by 6 heads")
        self.token_embed = nn.Embedding(self.vocab_size, width)
        self.pos = nn.Parameter(torch.zeros(1, self.max_len, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=6,
            dim_feedforward=width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.head = nn.Linear(width, self.vocab_size)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "hybrid384":
            return self.encoder(images)
        return self.encoder.forward_features(images)

    def decode(self, memory: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = self.token_embed(context) + self.pos
        return self.head(self.decoder(query, memory))

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        memory = self.encode(images)
        context = torch.zeros(images.shape[0], self.max_len, dtype=torch.long, device=images.device)
        context = self.decode(memory, context).argmax(dim=-1)
        return self.decode(memory, context).argmax(dim=-1)


def ids_to_text(rows: torch.Tensor, charset) -> list[str]:
    return [charset.decode(row.tolist()) for row in rows]


def resample_pos_embed(pos: torch.Tensor, new_shape: tuple[int, ...]) -> torch.Tensor:
    """Bicubic resize of a ViT position grid. The class token stays put.

    384 input is 24×24 patches plus one class token (577). 512 input is 32×32
    plus the class token (1025). Every other vl1 weight keeps its shape.
    """
    if pos.shape == new_shape:
        return pos
    if pos.ndim != 3 or pos.shape[0] != 1 or new_shape[0] != 1 or pos.shape[2] != new_shape[2]:
        raise RuntimeError(f"cannot resample pos embed {tuple(pos.shape)} -> {new_shape}")
    cls_token, grid = pos[:, :1], pos[:, 1:]
    old = int(round(grid.shape[1] ** 0.5))
    new = int(round((new_shape[1] - 1) ** 0.5))
    if old * old != grid.shape[1] or new * new != new_shape[1] - 1:
        raise RuntimeError(f"position grid is not square: {tuple(pos.shape)} -> {new_shape}")
    grid = grid.reshape(1, old, old, pos.shape[2]).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(new, new), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new * new, pos.shape[2])
    return torch.cat([cls_token, grid], dim=1)


def load_reader_weights(model: nn.Module, state: dict) -> None:
    """Load a reader state dict, resizing only the ViT position grid when needed."""
    current = model.state_dict()
    key = "encoder.pos_embed"
    if key in state and key in current and tuple(state[key].shape) != tuple(current[key].shape):
        state = dict(state)
        state[key] = resample_pos_embed(state[key], tuple(current[key].shape))
    model.load_state_dict(state)
