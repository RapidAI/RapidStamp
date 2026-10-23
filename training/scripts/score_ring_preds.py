"""Exact match and mean 1-NED for a strip list against recognizer output.

Spaces are kept. 1-NED is the mean over rows of ``1 - edit_distance /
max(len(pred), len(gold))``. Join key is the file name, not the directory.

    python training/scripts/score_ring_preds.py \
        --labels /home/rapidstamp_data/ring_baseline/val_list.txt \
        --pred /home/rapidstamp_data/logs/ring_baseline_val.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.ring_text import one_ned, summarize_pairs, worst_pairs  # noqa: E402


def read_labels(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        name, text = line.split("\t", 1)
        out[Path(name).name] = text
    return out


def read_preds(path: Path) -> dict[str, str]:
    """Second column is the prediction. A third column, if present, is a score."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        parts = line.split("\t")
        out[Path(parts[0]).name] = parts[1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    labels = read_labels(args.labels)
    preds = read_preds(args.pred)
    paired = []
    detailed = []
    missing = []
    for name, gold in labels.items():
        if name not in preds:
            missing.append(name)
            continue
        pred = preds[name]
        paired.append((pred, gold))
        detailed.append((name, pred, gold))
    summary = summarize_pairs(paired)
    summary["missing"] = len(missing)
    summary["worst"] = worst_pairs(detailed, 12)
    empty = sum(1 for pred, _gold in paired if pred == "")
    summary["empty_pred"] = empty
    longer = sum(1 for pred, gold in paired if len(pred) > len(gold))
    shorter = sum(1 for pred, gold in paired if len(pred) < len(gold))
    summary["pred_longer"] = longer
    summary["pred_shorter"] = shorter
    summary["mean_one_ned_errors_only"] = None
    errors = [one_ned(pred, gold) for pred, gold in paired if pred != gold]
    if errors:
        summary["mean_one_ned_errors_only"] = sum(errors) / len(errors)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
