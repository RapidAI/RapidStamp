"""Evaluate the exported ONNX pipeline on a synthetic validation split."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from rapidstamp.pipeline import RapidStamp
from rapidstamp_train.metrics_det import average_precision


def _jsonable(value):
    """Replace NaN so metrics.json is strict JSON."""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, np.integer):
        return int(value)
    return value


def _norm(text: str) -> str:
    return "".join(ch for ch in text.casefold() if not ch.isspace())


def _iou(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _rec_exact(root: Path, engine: RapidStamp) -> dict:
    rows = (root / "rec" / "val" / "labels.tsv").read_text(encoding="utf-8").splitlines()
    groups = {"seal": [0, 0], "center": [0, 0], "plain": [0, 0]}
    for line in rows:
        if not line.strip():
            continue
        name, text = line.split("\t", 1)
        image = np.asarray(Image.open(root / "rec" / "val" / "images" / name).convert("RGB"))
        pred, _score = engine._recognize(image)
        key = "plain" if "_p" in name else "center" if name.endswith("c.png") or "_c." in name else "seal"
        # center files are "{id}_{ord}c.png"
        if name.endswith("c.png"):
            key = "center"
        groups[key][1] += 1
        groups[key][0] += int(_norm(pred) == _norm(text))
    out = {}
    for key, (hit, total) in groups.items():
        out[key] = {"exact": hit / total if total else 0.0, "n": total}
    return out


def evaluate(args: argparse.Namespace) -> dict:
    root = Path(args.data)
    engine = RapidStamp(args.models)
    label_paths = sorted((root / "det" / "val" / "labels").glob("*.json"))
    if args.limit:
        label_paths = label_paths[: args.limit]
    predictions = []
    ground_truth = []
    title_hit = 0
    title_total = 0
    det_hit = 0
    per_class = {name: [0, 0] for name in engine.class_names}
    by_source: dict[str, list[int]] = {}
    center_hit = 0
    center_total = 0
    started = time.perf_counter()
    for label_path in label_paths:
        meta = json.loads(label_path.read_text(encoding="utf-8"))
        source = str(meta.get("source") or "synth")
        bucket = by_source.setdefault(source, [0, 0, 0])
        image_path = root / "det" / "val" / "images" / f"{label_path.stem}.jpg"
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        seals = engine(rgb)
        predictions.append(
            [
                (seal.class_id, seal.score, np.array([seal.x1, seal.y1, seal.x2, seal.y2], dtype=np.float32))
                for seal in seals
            ]
        )
        gt_items = []
        used = set()
        for gt in meta["seals"]:
            box = np.array(gt["bbox"], dtype=np.float32)
            gt_items.append((int(gt["cls"]), box))
            title_total += 1
            bucket[1] += 1
            name = gt["name"]
            per_class.setdefault(name, [0, 0])
            per_class[name][1] += 1
            best_j = -1
            best_iou = 0.0
            for index, seal in enumerate(seals):
                if index in used:
                    continue
                score = _iou((seal.x1, seal.y1, seal.x2, seal.y2), box.tolist())
                if score > best_iou:
                    best_iou = score
                    best_j = index
            if best_j >= 0 and best_iou >= 0.5:
                used.add(best_j)
                det_hit += 1
                bucket[2] += 1
                if _norm(seals[best_j].text) == _norm(gt["text"]):
                    title_hit += 1
                    bucket[0] += 1
                    per_class[name][0] += 1
                if gt.get("center_text"):
                    center_total += 1
                    if _norm(seals[best_j].center_text) == _norm(gt["center_text"]):
                        center_hit += 1
            elif gt.get("center_text"):
                center_total += 1
        ground_truth.append(gt_items)
    elapsed = time.perf_counter() - started
    det_metrics = average_precision(predictions, ground_truth)
    rec_metrics = _rec_exact(root, engine) if not args.skip_rec_strips else {}
    report = {
        "images": len(label_paths),
        "det_map50": det_metrics["map50"],
        "det_per_class_ap50": det_metrics["per_class_ap50"],
        "det_recall50": det_hit / title_total if title_total else 0.0,
        "e2e_title_recall": title_hit / title_total if title_total else 0.0,
        "e2e_per_class": {
            name: (hits / total if total else None) for name, (hits, total) in sorted(per_class.items())
        },
        "e2e_by_source": {
            name: {
                "e2e_title_recall": hits / total if total else None,
                "det_recall50": detected / total if total else None,
                "n": total,
            }
            for name, (hits, total, detected) in sorted(by_source.items())
        },
        "invoice_center_exact": center_hit / center_total if center_total else None,
        "invoice_center_n": center_total,
        "rec_strips": rec_metrics,
        "ms_per_image": (elapsed / max(1, len(label_paths))) * 1000.0,
        "gt_seals": title_total,
    }
    report = _jsonable(report)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate exported RapidStamp models")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-rec-strips", action="store_true")
    return parser


def main() -> None:
    evaluate(build_argparser().parse_args())


if __name__ == "__main__":
    main()
