"""Datasets for seal detection pages and recognition strips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rapidstamp.geometry import letterbox, map_boxes_to_letterbox, resize_bilinear, to_gray01
from rapidstamp.labels import REC_H, REC_W
from rapidstamp_train.distractors import blank_distractor_page, paint_distractors


class DetDataset(Dataset):
    def __init__(self, root: str | Path, train: bool = True) -> None:
        self.root = Path(root)
        self.train = train
        self.ids = sorted(path.stem for path in (self.root / "labels").glob("*.json"))
        if not self.ids:
            raise FileNotFoundError(f"no detection labels in {self.root}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        stem = self.ids[index]
        image = np.asarray(Image.open(self.root / "images" / f"{stem}.jpg").convert("RGB"))
        meta = json.loads((self.root / "labels" / f"{stem}.json").read_text(encoding="utf-8"))
        seals = meta["seals"]
        if seals:
            boxes = np.array([item["bbox"] for item in seals], dtype=np.float32)
            classes = np.array([item["cls"] for item in seals], dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int64)
        if self.train and np.random.random() < 0.85:
            image = paint_distractors(image, np.random.default_rng(), boxes)
        tensor, target = _tensor_targets(image, boxes, classes)
        if self.train:
            brightness = float(torch.empty(1).uniform_(0.8, 1.2))
            contrast = float(torch.empty(1).uniform_(0.85, 1.15))
            mean = tensor.mean(dim=(1, 2), keepdim=True)
            tensor = ((tensor - mean) * contrast + mean) * brightness
            tensor = tensor.clamp(0, 1)
        return tensor, target


def det_collate(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def _tensor_targets(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray):
    canvas, scale, pad_x, pad_y = letterbox(image)
    boxes_n = map_boxes_to_letterbox(boxes, scale, pad_x, pad_y, canvas.shape[0])
    if len(classes):
        target = np.concatenate([classes.astype(np.float32)[:, None], boxes_n], axis=1)
    else:
        target = np.zeros((0, 5), dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).float() / 255.0
    return tensor, torch.from_numpy(target)


class DistractorDataset(Dataset):
    """Blank pages that only contain red/blue non-seal ink."""

    def __init__(self, length: int = 2000) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng((index * 10007 + np.random.randint(0, 1_000_000)) % (2**32))
        image = blank_distractor_page(rng)
        return _tensor_targets(image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64))


class RecDataset(Dataset):
    def __init__(self, root: str | Path, charset: list[str]) -> None:
        self.root = Path(root)
        self.index = {ch: i + 1 for i, ch in enumerate(charset)}
        self.samples: list[tuple[str, str]] = []
        tsv = self.root / "labels.tsv"
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            name, text = line.split("\t", 1)
            if text and all(ch in self.index for ch in text):
                self.samples.append((name, text))
        if not self.samples:
            raise FileNotFoundError(f"no recognition samples in {tsv}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        name, text = self.samples[index]
        image = np.asarray(Image.open(self.root / "images" / name).convert("RGB"))
        if image.shape[0] != REC_H or image.shape[1] != REC_W:
            image = resize_bilinear(image, REC_W, REC_H)
        gray = torch.from_numpy(to_gray01(image)).unsqueeze(0)
        target = torch.tensor([self.index[ch] for ch in text], dtype=torch.long)
        return gray, target, name


def rec_collate(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    lengths = torch.tensor([int(item.numel()) for item in targets], dtype=torch.long)
    flat = torch.cat(targets)
    return images, flat, lengths
