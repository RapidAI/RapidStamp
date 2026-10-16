"""Train the phase-1 detector and recognizer on synthetic seals.

The detector sees a 128x128 view of the page. The recognizer sees the same
polar unwrap the inference engine will compute, including the gap-based
rotation alignment, so train and test share one geometry path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rapidstamp.charset import CHARSET, encode
from rapidstamp.geometry import (
    CHAR_SIZE,
    DET_SIZE,
    GRID,
    crop_slot,
    resize_rgb,
    slot_centers,
    unwrap_seal,
)
from rapidstamp.postprocess import decode_detections, iou
from rapidstamp_train.export import export_detector, export_recognizer
from rapidstamp_train.models import SealDetTiny, SealRecTiny, count_params
from rapidstamp_train.synth import Page, render_page


def _scale_boxes(boxes: list[list[float]], src_w: int, src_h: int) -> list[list[float]]:
    sx = DET_SIZE / float(src_w)
    sy = DET_SIZE / float(src_h)
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]


class DetSet(Dataset):
    def __init__(self, pages: list[Page]):
        self.pages = pages

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int):
        page = self.pages[index]
        image = resize_rgb(page.image, DET_SIZE, DET_SIZE)
        chw = np.transpose(image, (2, 0, 1)).astype(np.float32)
        boxes = _scale_boxes(page.boxes, page.image.shape[1], page.image.shape[0])
        return chw, boxes


class RecSet(Dataset):
    def __init__(self, crops: list[np.ndarray], labels: list[int]):
        self.crops = crops
        self.labels = labels

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int):
        crop = self.crops[index][None, :, :].astype(np.float32)
        return crop, self.labels[index]


def _collate_det(batch):
    images = torch.from_numpy(np.stack([b[0] for b in batch]))
    boxes = [b[1] for b in batch]
    return images, boxes


def _collate_rec(batch):
    images = torch.from_numpy(np.stack([b[0] for b in batch]))
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return images, labels


def make_det_pages(n: int, seed: int, augment: bool) -> list[Page]:
    rng = np.random.default_rng(seed)
    pages = []
    for _ in range(n):
        roll = int(rng.integers(0, 10))
        if roll == 0:
            n_seals = 0
        elif roll == 1:
            n_seals = 2
        else:
            n_seals = 1
        pages.append(render_page(rng, n_seals=n_seals, augment=augment))
    return pages


def make_rec_samples(n: int, seed: int, augment: bool, jitter: bool) -> tuple[list[np.ndarray], list[int], list[Page]]:
    """Character crops plus the pages they came from (used for string metrics)."""
    rng = np.random.default_rng(seed)
    crops: list[np.ndarray] = []
    labels: list[int] = []
    pages: list[Page] = []
    attempts = 0
    while len(pages) < n and attempts < n * 4:
        attempts += 1
        page = render_page(rng, n_seals=1, augment=augment)
        if not page.boxes:
            continue
        box = list(page.boxes[0])
        if jitter:
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            radius = box[2] / 2.0
            cx += float(rng.uniform(-0.03, 0.03)) * radius * 2
            cy += float(rng.uniform(-0.03, 0.03)) * radius * 2
            radius *= float(rng.uniform(0.97, 1.03))
            box = [cx - radius, cy - radius, radius * 2, radius * 2]
        strip = unwrap_seal(page.image, box)
        text = page.texts[0]
        ids = encode(text)
        for index, center in enumerate(slot_centers(len(text), strip.shape[1])):
            crops.append(crop_slot(strip, center, CHAR_SIZE))
            labels.append(ids[index] - 1)
        pages.append(page)
    if len(pages) < n:
        raise RuntimeError(f"only built {len(pages)} recognizer pages out of {n}")
    return crops, labels, pages


def detection_loss(logits: torch.Tensor, boxes: list[list[list[float]]]) -> torch.Tensor:
    obj_logit = logits[:, 0]
    pred = torch.sigmoid(logits[:, 1:])
    obj_target = torch.zeros_like(obj_logit)
    box_losses = []
    cell = DET_SIZE / float(GRID)
    for i, image_boxes in enumerate(boxes):
        for x, y, w, h in image_boxes:
            cx = x + w / 2.0
            cy = y + h / 2.0
            target = torch.tensor(
                [cx / DET_SIZE, cy / DET_SIZE, w / DET_SIZE, h / DET_SIZE],
                dtype=pred.dtype,
                device=pred.device,
            )
            hits = []
            for row in range(GRID):
                for col in range(GRID):
                    px = (col + 0.5) * cell
                    py = (row + 0.5) * cell
                    if x <= px <= x + w and y <= py <= y + h:
                        hits.append((row, col))
            if not hits:
                hits = [(min(GRID - 1, max(0, int(cy // cell))), min(GRID - 1, max(0, int(cx // cell))))]
            for row, col in hits:
                obj_target[i, row, col] = 1.0
                box_losses.append(F.smooth_l1_loss(pred[i, :, row, col], target))
    # Average the positive cells and the negative cells separately. A flat
    # mean lets the 63 empty cells drown the one cell that contains the seal.
    pos_mask = obj_target > 0.5
    neg_mask = ~pos_mask
    obj_loss = obj_logit.new_zeros(())
    if bool(pos_mask.any()):
        obj_loss = obj_loss + F.binary_cross_entropy_with_logits(obj_logit[pos_mask], obj_target[pos_mask])
    if bool(neg_mask.any()):
        obj_loss = obj_loss + F.binary_cross_entropy_with_logits(obj_logit[neg_mask], obj_target[neg_mask])
    if box_losses:
        return obj_loss + torch.stack(box_losses).mean() * 2.0
    return obj_loss


def recognition_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels.to(logits.device))


@torch.no_grad()
def eval_detector(model: torch.nn.Module, pages: list[Page], thresh: float) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    ious = []
    for page in pages:
        image = resize_rgb(page.image, DET_SIZE, DET_SIZE)
        chw = torch.from_numpy(np.transpose(image, (2, 0, 1))[None].astype(np.float32))
        logits = model(chw)[0].cpu().numpy()
        preds = decode_detections(logits, page.image.shape[1], page.image.shape[0], thresh=thresh)
        matched = set()
        for gt in page.boxes:
            best, best_i = 0.0, -1
            for i, pred in enumerate(preds):
                if i in matched:
                    continue
                score = iou(pred, gt)
                if score > best:
                    best, best_i = score, i
            if best >= 0.5 and best_i >= 0:
                tp += 1
                matched.add(best_i)
                ious.append(best)
            else:
                fn += 1
        fp += len(preds) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def predict_text(model: torch.nn.Module, strip: np.ndarray) -> tuple[str, float]:
    """Lexicon decode, the same rule the inference engine uses."""
    from rapidstamp.ctc import log_softmax
    from rapidstamp.titles import decode_titles, seal_titles

    model.eval()
    titles = seal_titles()
    tables = {}
    for n_chars in sorted({len(title) for title in titles}):
        crops = [crop_slot(strip, center, CHAR_SIZE) for center in slot_centers(n_chars, strip.shape[1])]
        batch = torch.from_numpy(np.stack(crops)[:, None].astype(np.float32))
        tables[n_chars] = log_softmax(model(batch).cpu().numpy())
    return decode_titles(tables, titles, CHARSET)


@torch.no_grad()
def eval_recognizer(model: torch.nn.Module, pages: list[Page]) -> tuple[float, float]:
    model.eval()
    exact = 0
    chars_ok = 0
    chars_n = 0
    for page in pages:
        strip = unwrap_seal(page.image, page.boxes[0])
        pred, _score = predict_text(model, strip)
        truth = page.texts[0]
        exact += int(pred == truth)
        for a, b in zip(pred, truth):
            chars_ok += int(a == b)
        chars_n += max(len(pred), len(truth))
    return exact / max(len(pages), 1), chars_ok / max(chars_n, 1)


def _run_epochs(model, loader, loss_fn, epochs: int, lr: float, device: str) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        print(f"  epoch {epoch + 1}/{epochs} loss {total / max(seen, 1):.4f}", flush=True)


def _det_batch_loss(model, batch, device):
    images, boxes = batch
    logits = model(images.to(device))
    return detection_loss(logits, boxes)


def _rec_batch_loss(model, batch, device):
    images, labels = batch
    logits = model(images.to(device))
    return recognition_loss(logits, labels)


def sweep_threshold(model, pages: list[Page]) -> tuple[float, dict]:
    best_t, best = 0.5, None
    for thresh in (0.3, 0.4, 0.5, 0.6, 0.7):
        metrics = eval_detector(model, pages, thresh)
        f1 = 0.0
        if metrics["precision"] + metrics["recall"]:
            f1 = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        print(f"  thresh {thresh:.2f} P {metrics['precision']:.3f} R {metrics['recall']:.3f} IoU {metrics['mean_iou']:.3f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, metrics)
            best_t = thresh
    assert best is not None
    return best_t, best[1]


def copy_package_models(out: Path) -> None:
    dest = Path(__file__).resolve().parents[2] / "python" / "src" / "rapidstamp" / "models"
    dest.mkdir(parents=True, exist_ok=True)
    for item in out.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)


def train(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    print("rendering detector data", flush=True)
    t0 = time.time()
    det_train = make_det_pages(args.det_samples, args.seed, augment=True)
    det_val = make_det_pages(args.val_samples, args.seed + 1, augment=False)
    print(f"rendering recognizer data ({time.time() - t0:.1f}s so far)", flush=True)
    rec_crops, rec_labels, _rec_pages = make_rec_samples(args.rec_samples, args.seed + 2, augment=True, jitter=True)
    _val_crops, _val_labels, val_pages = make_rec_samples(args.val_samples, args.seed + 3, augment=False, jitter=False)
    print(f"data ready in {time.time() - t0:.1f}s", flush=True)

    det = SealDetTiny().to(device)
    rec = SealRecTiny().to(device)
    print(f"detector params {count_params(det)} recognizer params {count_params(rec)}", flush=True)
    det_loader = DataLoader(DetSet(det_train), batch_size=args.batch, shuffle=True, collate_fn=_collate_det)
    rec_loader = DataLoader(RecSet(rec_crops, rec_labels), batch_size=args.batch, shuffle=True, collate_fn=_collate_rec)

    print("train detector", flush=True)
    _run_epochs(det, det_loader, _det_batch_loss, args.det_epochs, args.lr, device)
    det_thresh, det_metrics = sweep_threshold(det, det_val)

    print("train recognizer", flush=True)
    _run_epochs(rec, rec_loader, _rec_batch_loss, args.rec_epochs, args.lr, device)
    rec_exact, rec_char = eval_recognizer(rec, val_pages)
    print(f"recognizer exact {rec_exact:.3f} char {rec_char:.3f}", flush=True)

    torch.save(det.state_dict(), out / "detector.pt")
    torch.save(rec.state_dict(), out / "recognizer.pt")
    export_detector(out / "detector.pt", out / "detector", det_thresh)
    export_recognizer(out / "recognizer.pt", out / "recognizer")

    e2e = _eval_e2e(out, args.seed + 9, n=64)
    metrics = {
        "detector": {k: (float(v) if not isinstance(v, int) else v) for k, v in det_metrics.items()},
        "detector_params": count_params(det),
        "recognizer_exact": rec_exact,
        "recognizer_char": rec_char,
        "recognizer_params": count_params(rec),
        "e2e_exact": e2e["exact"],
        "e2e_detection": e2e["detection"],
        "latency_ms_numpy": e2e["latency_ms"],
        "det_thresh": det_thresh,
        "det_bytes": (out / "detector.bin").stat().st_size + (out / "detector.manifest").stat().st_size,
        "rec_bytes": (out / "recognizer.bin").stat().st_size + (out / "recognizer.manifest").stat().st_size,
        "det_onnx_bytes": (out / "detector.onnx").stat().st_size,
        "rec_onnx_bytes": (out / "recognizer.onnx").stat().st_size,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    copy_package_models(out)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def _eval_e2e(out: Path, seed: int, n: int) -> dict:
    from rapidstamp.engine import SealEngine

    engine = SealEngine(out / "detector.manifest", out / "recognizer.manifest", backend="numpy")
    rng = np.random.default_rng(seed)
    exact = 0
    detected = 0
    t0 = time.perf_counter()
    for _ in range(n):
        page = render_page(rng, n_seals=1, augment=False)
        preds = engine.infer(page.image)
        if preds and page.boxes and iou([preds[0].x, preds[0].y, preds[0].w, preds[0].h], page.boxes[0]) >= 0.5:
            detected += 1
            if preds[0].text == page.texts[0]:
                exact += 1
    elapsed = (time.perf_counter() - t0) * 1000.0 / n
    return {"exact": exact / n, "detection": detected / n, "latency_ms": elapsed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the RapidStamp phase-1 demo")
    parser.add_argument("--out", default="models/demo")
    parser.add_argument("--det-samples", type=int, default=1500)
    parser.add_argument("--rec-samples", type=int, default=3000)
    parser.add_argument("--val-samples", type=int, default=96)
    parser.add_argument("--det-epochs", type=int, default=12)
    parser.add_argument("--rec-epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
