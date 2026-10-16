"""Datasets for seal detection pages and recognition strips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rapidstamp.geometry import letterbox, map_boxes_to_letterbox, resize_bilinear, to_gray01
from rapidstamp.labels import GRID, REC_H, REC_W


def _encode(boxes: np.ndarray, classes: np.ndarray, grid: int = GRID):
    obj = np.zeros((grid, grid), dtype=np.float32)
    box = np.zeros((grid, grid, 4), dtype=np.float32)
    cls = np.zeros((grid, grid), dtype=np.int64)
    area = np.zeros((grid, grid), dtype=np.float32)
    for rect, class_id in zip(boxes, classes):
        x1, y1, x2, y2 = [float(v) for v in rect]
        cx = ((x1 + x2) * 0.5) * grid
        cy = ((y1 + y2) * 0.5) * grid
        gx = int(np.clip(np.floor(cx), 0, grid - 1))
        gy = int(np.clip(np.floor(cy), 0, grid - 1))
        this_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if obj[gy, gx] > 0 and this_area <= area[gy, gx]:
            continue
        obj[gy, gx] = 1.0
        box[gy, gx] = (x1, y1, x2, y2)
        cls[gy, gx] = int(class_id)
        area[gy, gx] = this_area
    return obj, box, cls


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
        canvas, scale, pad_x, pad_y = letterbox(image)
        boxes_n = map_boxes_to_letterbox(boxes, scale, pad_x, pad_y, canvas.shape[0])
        obj, box, cls = _encode(boxes_n, classes)
        tensor = torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1).float() / 255.0
        if self.train:
            brightness = float(torch.empty(1).uniform_(0.8, 1.2))
            contrast = float(torch.empty(1).uniform_(0.85, 1.15))
            mean = tensor.mean(dim=(1, 2), keepdim=True)
            tensor = (tensor - mean) * contrast + mean
            tensor = tensor * brightness
            tensor = tensor.clamp(0, 1)
        return tensor, torch.from_numpy(obj), torch.from_numpy(box), torch.from_numpy(cls)


def det_collate(batch):
    images = torch.stack([item[0] for item in batch])
    obj = torch.stack([item[1] for item in batch])
    box = torch.stack([item[2] for item in batch])
    cls = torch.stack([item[3] for item in batch])
    return images, obj, box, cls


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
