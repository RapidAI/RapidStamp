"""Build train and test trees from the user seal zip.

Images stay on the training machine. The manifest has no pixels.

    python training/scripts/import_user_seals.py \
        --src /home/mayong/datasets/user_seals \
        --train-out /home/mayong/rapidstamp/real/user_seals \
        --test-out /home/mayong/rapidstamp/real/user_seals_test \
        --manifest training/manifests/user_seals_split.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from PIL import Image  # noqa: E402

from rapidstamp_train.user_seals import (  # noqa: E402
    annotate,
    assign_splits,
    capture_id,
    file_sha256,
    load_rgb,
    pixel_md5,
)


def _pngs(src: Path) -> list[Path]:
    files = [path for path in src.rglob("*.png") if path.is_file()]
    return sorted(files, key=lambda path: path.relative_to(src).as_posix())


def build(src: Path, train_out: Path, test_out: Path, manifest: Path | None) -> dict:
    rows = []
    images: dict[str, object] = {}
    for path in _pngs(src):
        rgb = load_rgb(path)
        relative = path.relative_to(src).as_posix()
        digest = file_sha256(path)
        seals = annotate(rgb)
        row = {
            "sha256": digest,
            "file": relative,
            "folder": path.parent.name,
            "capture": capture_id(path.name),
            "pixel_md5": pixel_md5(rgb),
            "kind": "isolated_seal",
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "text_status": "unverified",
            "seals": seals,
        }
        rows.append(row)
        images[relative] = rgb
    assign_splits(rows)
    for split_name, dest in (("train", train_out), ("test", test_out)):
        label_split = "train" if split_name == "train" else "val"
        image_dir = dest / "det" / label_split / "images"
        label_dir = dest / "det" / label_split / "labels"
        if image_dir.exists():
            for old in image_dir.glob("*.jpg"):
                old.unlink()
        if label_dir.exists():
            for old in label_dir.glob("*.json"):
                old.unlink()
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
    stats = {"train": 0, "test": 0, "duplicate": 0, "skipped": 0, "boxes": 0}
    classes: dict[str, int] = {}
    for row in rows:
        stats[row["split"]] = stats.get(row["split"], 0) + 1
        if row["split"] not in {"train", "test"}:
            continue
        for seal in row["seals"]:
            classes[seal["name"]] = classes.get(seal["name"], 0) + 1
            stats["boxes"] += 1
        stem = row["sha256"][:16]
        dest = train_out if row["split"] == "train" else test_out
        label_split = "train" if row["split"] == "train" else "val"
        rgb = images[row["file"]]
        Image.fromarray(rgb, "RGB").save(
            dest / "det" / label_split / "images" / f"{stem}.jpg",
            quality=90,
        )
        label = {
            "width": row["width"],
            "height": row["height"],
            "source": "user_seals",
            "kind": row["kind"],
            "sha256": row["sha256"],
            "file": row["file"],
            "text_status": "unverified",
            "seals": row["seals"],
        }
        (dest / "det" / label_split / "labels" / f"{stem}.json").write_text(
            json.dumps(label, ensure_ascii=False),
            encoding="utf-8",
        )
    stats["classes"] = classes
    if manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--test-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    build(args.src, args.train_out, args.test_out, args.manifest)


if __name__ == "__main__":
    main()
