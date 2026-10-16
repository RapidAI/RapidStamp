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


def crop_seal_large(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    rng: np.random.Generator,
    frac: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop so one seal occupies most of the frame.

    Published training pages keep stamps small, and the detector then splits a
    seal that fills the photo into fragments. ``frac`` is the seal's long side
    divided by the crop's long side. The default range is 0.62 to 0.98.
    """
    if len(boxes) == 0:
        return image, boxes, classes
    index = int(rng.integers(0, len(boxes)))
    x1, y1, x2, y2 = (float(v) for v in boxes[index])
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 8 or box_h < 8:
        return image, boxes, classes
    if frac is None:
        frac = float(rng.uniform(0.62, 0.98))
    frac = min(0.98, max(0.2, float(frac)))
    height, width = image.shape[:2]
    side = max(box_w, box_h) / frac
    crop_w = int(round(min(float(width), max(side, box_w + 1.0))))
    crop_h = int(round(min(float(height), max(side, box_h + 1.0))))
    crop_w = max(8, min(width, crop_w))
    crop_h = max(8, min(height, crop_h))
    cx = (x1 + x2) / 2.0 + float(rng.uniform(-0.08, 0.08)) * max(0.0, crop_w - box_w)
    cy = (y1 + y2) / 2.0 + float(rng.uniform(-0.08, 0.08)) * max(0.0, crop_h - box_h)
    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))
    left = min(max(0, left), width - crop_w)
    top = min(max(0, top), height - crop_h)
    cropped = image[top : top + crop_h, left : left + crop_w]
    kept_boxes = []
    kept_classes = []
    for box, cls in zip(boxes, classes):
        nx1 = float(np.clip(box[0] - left, 0, crop_w))
        ny1 = float(np.clip(box[1] - top, 0, crop_h))
        nx2 = float(np.clip(box[2] - left, 0, crop_w))
        ny2 = float(np.clip(box[3] - top, 0, crop_h))
        if (nx2 - nx1) * (ny2 - ny1) < 0.45 * max(1.0, (box[2] - box[0]) * (box[3] - box[1])):
            continue
        kept_boxes.append([nx1, ny1, nx2, ny2])
        kept_classes.append(int(cls))
    if not kept_boxes:
        return image, boxes, classes
    return (
        cropped,
        np.array(kept_boxes, dtype=np.float32),
        np.array(kept_classes, dtype=np.int64),
    )


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
        if self.train and len(boxes) and np.random.random() < 0.5:
            image, boxes, classes = crop_seal_large(image, boxes, classes, np.random.default_rng())
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
