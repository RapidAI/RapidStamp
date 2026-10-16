"""Build a detection tree from reviewed web-seal labels.

Boxes are ink proposals matched to the region named in the review. Pages
with no seal become empty labels so the detector can learn the background.
Images stay on the training machine.

    python training/scripts/build_web_pages.py \
        --images /home/mayong/datasets/web_seals/images \
        --reviews /tmp/web_review/ann.jsonl \
        --out /home/mayong/rapidstamp/real/web \
        --manifest training/manifests/web_seals_split.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))

from PIL import Image

from propose_seal_boxes import propose_seal_boxes  # noqa: E402
from rapidstamp.geometry import estimate_ring_start, pad_to_square  # noqa: E402
from rapidstamp.labels import CLASS_LAYOUT, UNWRAP_START_DEG, class_id  # noqa: E402
from rapidstamp_train.synthesize import strip_from_crop  # noqa: E402

ANCHORS = {
    "top-left": (0.22, 0.22),
    "top": (0.50, 0.18),
    "top-right": (0.78, 0.22),
    "left": (0.18, 0.50),
    "center": (0.50, 0.50),
    "right": (0.82, 0.50),
    "bottom-left": (0.22, 0.80),
    "bottom": (0.50, 0.82),
    "bottom-right": (0.78, 0.80),
}
POSITIVE = {"document_seal", "isolated_seal", "collage"}
# Article and product pages often still contain a seal. Labeling them empty
# would teach the detector to suppress real stamps. Only pages reviewed as
# having no seal are negatives.
NEGATIVE = {"unrelated"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _match(box: list[int], where: str, width: int, height: int) -> float:
    ax, ay = ANCHORS.get(where, (0.5, 0.5))
    cx = ((box[0] + box[2]) * 0.5) / max(width, 1)
    cy = ((box[1] + box[3]) * 0.5) / max(height, 1)
    return (cx - ax) ** 2 + (cy - ay) ** 2


def _chromatic_mask(image: np.ndarray) -> np.ndarray:
    red_ch = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    red = (red_ch > 140) & (red_ch > green + 35) & (red_ch > blue + 35)
    blu = (blue > 110) & (blue > red_ch + 25) & (blue > green + 10)
    return red | blu


def assign_boxes(image: np.ndarray, seals: list[dict]) -> list[dict]:
    proposals = propose_seal_boxes(image, limit=8)
    # On a page that does contain red or blue ink, a box with none of it is
    # body text. Gray scans keep the dark-ink proposal.
    ink = _chromatic_mask(image)
    if float(ink.mean()) > 0.002:
        proposals = [box for box in proposals if float(ink[box[1] : box[3], box[0] : box[2]].mean()) >= 0.02]
    height, width = image.shape[:2]
    used: set[int] = set()
    assigned = []
    for seal in seals:
        if not proposals:
            break
        order = sorted(range(len(proposals)), key=lambda i: _match(proposals[i], str(seal.get("where") or "center"), width, height))
        choice = next((i for i in order if i not in used), None)
        if choice is None:
            continue
        dist = _match(proposals[choice], str(seal.get("where") or "center"), width, height)
        if dist > 0.18 and len(seals) > 1:
            continue
        x1, y1, x2, y2 = proposals[choice]
        short = min(width, height)
        if min(x2 - x1, y2 - y1) < 0.045 * short:
            continue
        used.add(choice)
        name = str(seal.get("class") or "round_official")
        if name not in CLASS_LAYOUT:
            name = "round_official"
        text = str(seal.get("text") or "") if seal.get("readable") else ""
        rows, cols = _grid_shape(text) if CLASS_LAYOUT[name] == "grid" else (1, 1)
        assigned.append(
            {
                "cls": class_id(name),
                "name": name,
                "bbox": proposals[choice],
                "text": text,
                "layout": CLASS_LAYOUT[name],
                "rows": rows,
                "cols": cols,
            }
        )
    return assigned


def _grid_shape(text: str) -> tuple[int, int]:
    if len(text) == 3:
        return 3, 1
    if len(text) == 4:
        return 2, 2
    return 2, 1


def _split_for(kind: str, digest: str) -> str:
    # Fixed hash split so the test list does not move when more files are added.
    bucket = int(digest[:8], 16) % 10
    if kind == "collage":
        return "train"
    return "test" if bucket < 3 else "train"


def _rec_rows(out: Path, split: str, stem: str, image: np.ndarray, seals: list[dict]) -> list[str]:
    rows = []
    rec_dir = out / "rec" / split / "images"
    for ordinal, seal in enumerate(seals):
        text = str(seal.get("text") or "")
        if not text:
            continue
        if seal["layout"] == "grid" and len(text) not in {2, 3, 4}:
            continue
        x1, y1, x2, y2 = seal["bbox"]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        start = UNWRAP_START_DEG
        if seal["layout"] in {"circle", "ellipse"}:
            source = pad_to_square(crop, 255) if seal["layout"] == "circle" else crop
            guess, gap = estimate_ring_start(source)
            if guess is not None and gap >= 40:
                start = guess
        strip = strip_from_crop(crop, seal["layout"], int(seal["rows"]), int(seal["cols"]), start)
        rec_dir.mkdir(parents=True, exist_ok=True)
        name = f"{stem}_{ordinal}.png"
        Image.fromarray(strip, "RGB").save(rec_dir / name)
        rows.append(f"{name}\t{text}\n")
    return rows


def build(args: argparse.Namespace) -> None:
    reviews = []
    for line in Path(args.reviews).read_text(encoding="utf-8").splitlines():
        if line.strip():
            reviews.append(json.loads(line))
    out = Path(args.out)
    for split in ("train", "test"):
        tsv = out / "rec" / split / "labels.tsv"
        if tsv.is_file():
            tsv.unlink()
    manifest_rows = []
    stats = {"train": 0, "test": 0, "positives": 0, "negatives": 0, "boxes": 0, "skipped": 0}
    for row in reviews:
        path = Path(args.images) / row["file"]
        if not path.is_file():
            path = Path(args.images) / Path(str(row["file"])).name
        if not path.is_file():
            stats["skipped"] += 1
            continue
        kind = str(row.get("kind") or "unrelated")
        digest = _sha(path)
        split = _split_for(kind, digest)
        image = np.asarray(Image.open(path).convert("RGB"))
        seals = assign_boxes(image, row.get("seals") or []) if kind in POSITIVE else []
        if kind in POSITIVE and not seals:
            stats["skipped"] += 1
            manifest_rows.append({"sha256": digest, "file": path.name, "kind": kind, "split": "skipped", "seals": []})
            continue
        if kind not in POSITIVE and kind not in NEGATIVE:
            stats["skipped"] += 1
            manifest_rows.append({"sha256": digest, "file": path.name, "kind": kind, "split": "excluded", "seals": []})
            continue
        stem = digest[:16]
        label = {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "source": "web",
            "kind": kind,
            "sha256": digest,
            "seals": seals,
        }
        image_dir = out / "det" / split / "images"
        label_dir = out / "det" / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image, "RGB").save(image_dir / f"{stem}.jpg", quality=90)
        (label_dir / f"{stem}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
        rec_rows = _rec_rows(out, split, stem, image, seals)
        if rec_rows:
            tsv = out / "rec" / split / "labels.tsv"
            tsv.parent.mkdir(parents=True, exist_ok=True)
            with tsv.open("a", encoding="utf-8") as handle:
                handle.write("".join(rec_rows))
        stats[split] += 1
        stats["boxes"] += len(seals)
        if seals:
            stats["positives"] += 1
        else:
            stats["negatives"] += 1
        manifest_rows.append({"sha256": digest, "file": path.name, "kind": kind, "split": split, "seals": seals})
    if args.manifest:
        dest = Path(args.manifest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
