"""Unwrap the 63 gold seal crops and score ring text.

The crops are not in git. Pass the directory that holds them. Rows marked
``user_verified_pending_recheck`` are scored on their own as well as inside
the full 63. Comparison strings are exact: spaces count.

    python training/scripts/eval_ring_gold.py \
        --manifest training/manifests/user_seals_split.jsonl \
        --images /path/to/user_seals_test \
        --out /home/rapidstamp_data/ring_baseline/gold

With ``--pred`` (path, prediction, optional score) the same script reports
exact match and mean 1-NED. Without images it only scores when ``--pred``
names files the manifest can already identify.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.ring_text import (  # noqa: E402
    load_rgb,
    strip_name,
    summarize_pairs,
    unwrap_ring,
    worst_pairs,
)
from PIL import Image  # noqa: E402


def test_rows(manifest: Path) -> list[dict]:
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "test":
                continue
            seals = row.get("seals") or [{}]
            seal = seals[0]
            ring = str(seal.get("ring") or seal.get("text") or "")
            rows.append(
                {
                    "file": Path(str(row.get("file") or "")).name,
                    "capture": str(row.get("capture") or ""),
                    "sha256": str(row.get("sha256") or ""),
                    "stem16": str(row.get("sha256") or "")[:16],
                    "ring": ring,
                    "text_status": row.get("text_status") or "",
                }
            )
    return rows


def index_images(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        found.setdefault(path.name, path)
        found.setdefault(path.stem, path)
    return found


def locate(row: dict, images: dict[str, Path]) -> Path | None:
    keys = [
        row["file"],
        Path(row["file"]).stem,
        row["capture"],
        row["capture"] + ".png",
        row["capture"] + ".jpg",
        row["stem16"],
        row["stem16"] + ".jpg",
        row["stem16"] + ".png",
    ]
    for key in keys:
        if key and key in images:
            return images[key]
    return None


def read_preds(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        parts = line.split("\t")
        out[Path(parts[0]).name] = parts[1]
        out[Path(parts[0]).stem] = parts[1]
    return out


def score_group(rows: list[dict], preds: dict[str, str]) -> dict:
    paired = []
    detailed = []
    missing = []
    for row in rows:
        pred = None
        for key in (row["strip"], Path(row["strip"]).stem, row["file"], row["capture"], row["stem16"]):
            if key and key in preds:
                pred = preds[key]
                break
        if pred is None:
            missing.append(row["file"])
            continue
        paired.append((pred, row["ring"]))
        detailed.append((row["file"], pred, row["ring"]))
    summary = summarize_pairs(paired)
    summary["missing"] = len(missing)
    summary["worst"] = worst_pairs(detailed, 10)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pred", type=Path)
    args = parser.parse_args()
    rows = test_rows(args.manifest)
    report: dict = {"gold_rows": len(rows)}
    if args.images:
        images = index_images(args.images)
        found = 0
        written = []
        if args.out:
            (args.out / "strips").mkdir(parents=True, exist_ok=True)
        for row in rows:
            path = locate(row, images)
            row["image"] = str(path) if path else ""
            if path is None:
                continue
            found += 1
            if args.out:
                unwound = unwrap_ring(load_rgb(path))
                name = strip_name(row["stem16"] or row["capture"], bool(unwound["wide_gap"]))
                dest = args.out / "strips" / name
                Image.fromarray(unwound["strip"]).save(dest)
                row["strip"] = name
                row["start_deg"] = unwound["start_deg"]
                row["gap_deg"] = unwound["gap_deg"]
                written.append(f"{dest}\t{row['ring']}")
            else:
                row["strip"] = ""
        report["images_found"] = found
        report["images_missing"] = len(rows) - found
        if args.out:
            text = "\n".join(written)
            (args.out / "gold_list.txt").write_text(text + ("\n" if text else ""), encoding="utf-8")
            (args.out / "gold_rows.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        report["images_found"] = 0
        report["images_missing"] = len(rows)
        for row in rows:
            row["strip"] = ""

    if args.pred:
        preds = read_preds(args.pred)
        pending = [row for row in rows if row["text_status"] == "user_verified_pending_recheck"]
        verified = [row for row in rows if row["text_status"] == "user_verified"]
        report["all"] = score_group(rows, preds)
        report["user_verified"] = score_group(verified, preds)
        report["user_verified_pending_recheck"] = score_group(pending, preds)
    if args.out:
        (args.out if args.out.suffix else args.out).mkdir(parents=True, exist_ok=True)
        (args.out / "gold_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report.get("images_found", 0) == 0 and not args.pred:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
