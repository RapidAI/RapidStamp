"""Precision, recall, and mAP on a reviewed page tree.

The tree uses ``det/val`` (the frozen web test split). Images stay on disk.

    python training/scripts/eval_real_pages.py \
        --data /home/mayong/rapidstamp/real/web_test \
        --models /home/mayong/rapidstamp/models
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp.pipeline import RapidStamp  # noqa: E402
from rapidstamp_train.metrics_det import average_precision  # noqa: E402


def _iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating,)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    return value


def _norm(text: str) -> str:
    return "".join(ch for ch in text.casefold() if not ch.isspace())


def evaluate(data: Path, models: Path) -> dict:
    engine = RapidStamp(models)
    label_dir = data / "det" / "val" / "labels"
    image_dir = data / "det" / "val" / "images"
    predictions = []
    ground_truth = []
    tp = fp = fn = 0
    text_hit = text_total = 0
    started = time.perf_counter()
    pages = 0
    for label_path in sorted(label_dir.glob("*.json")):
        meta = json.loads(label_path.read_text(encoding="utf-8"))
        image_path = image_dir / f"{label_path.stem}.jpg"
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        seals = engine(rgb)
        pages += 1
        predictions.append(
            [
                (seal.class_id, seal.score, np.array([seal.x1, seal.y1, seal.x2, seal.y2], dtype=np.float32))
                for seal in seals
            ]
        )
        gt_items = []
        used = set()
        for gt in meta["seals"]:
            box = [float(v) for v in gt["bbox"]]
            gt_items.append((int(gt["cls"]), np.array(box, dtype=np.float32)))
            best_j = -1
            best_iou = 0.0
            for index, seal in enumerate(seals):
                if index in used:
                    continue
                score = _iou((seal.x1, seal.y1, seal.x2, seal.y2), box)
                if score > best_iou:
                    best_iou = score
                    best_j = index
            if best_j >= 0 and best_iou >= 0.5:
                used.add(best_j)
                tp += 1
                if gt.get("text"):
                    text_total += 1
                    if _norm(seals[best_j].text) == _norm(gt["text"]):
                        text_hit += 1
            else:
                fn += 1
                if gt.get("text"):
                    text_total += 1
        fp += len(seals) - len(used)
        ground_truth.append(gt_items)
    elapsed = time.perf_counter() - started
    det = average_precision(predictions, ground_truth)
    report = {
        "images": pages,
        "gt_seals": tp + fn,
        "predictions": tp + fp,
        "precision50": tp / (tp + fp) if tp + fp else None,
        "recall50": tp / (tp + fn) if tp + fn else None,
        "det_map50": det["map50"],
        "det_per_class_ap50": det["per_class_ap50"],
        "text_exact": text_hit / text_total if text_total else None,
        "text_n": text_total,
        "ms_per_image": elapsed / max(pages, 1) * 1000.0,
    }
    report = _jsonable(report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.data, args.models)
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
