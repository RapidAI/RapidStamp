"""Turn external seal archives into the same folders the trainer already reads.

Images stay on the training machine. Nothing here copies pixels into git.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from rapidstamp.labels import CLASS_LAYOUT, REC_H, REC_W, UNWRAP_START_DEG, class_id
from rapidstamp_train.corpus import all_chars
from rapidstamp_train.fonts_util import find_latin_font
from rapidstamp_train.synthesize import FontBank, _degrade_strip, _jpeg, _page_background, strip_from_crop


def read_charset_file(path: Path) -> list[str]:
    chars: list[str] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line == "":
            continue
        ch = line if len(line) == 1 else line[0]
        if ch not in seen:
            seen.add(ch)
            chars.append(ch)
    if not chars:
        raise ValueError(f"empty charset {path}")
    return chars


def write_charset_file(path: Path, chars: list[str]) -> None:
    path.write_text("".join(ch + "\n" for ch in chars), encoding="utf-8")


def guess_seal(text: str, width: int, height: int) -> tuple[str, str]:
    """Map a title and the crop's aspect ratio onto one of the eight classes.

    ReST and the Paddle seal crops do not use our class names. Invoice / finance
    / notary are keyword hits. A wide crop is an oval. Everything else is a
    round company seal. The box we train on is the pasted crop, not the text
    polygon: those polygons outline the title, not the whole seal.
    """
    ratio = width / max(height, 1)
    if "发票" in text:
        name = "invoice"
    elif "财务" in text:
        name = "finance"
    elif "公证" in text:
        name = "notary"
    elif ratio >= 1.28 or ratio <= 0.78:
        name = "oval"
    else:
        name = "round_official"
    return name, CLASS_LAYOUT[name]


def _val_split(key: str, val_mod: int = 10) -> bool:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % val_mod == 0


def _open_rgb(payload: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    return np.asarray(image)


def _scaled_crop(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = rgb.shape[:2]
    target = int(rng.integers(220, 420))
    scale = target / max(height, width)
    new_w = max(32, int(round(width * scale)))
    new_h = max(32, int(round(height * scale)))
    resized = Image.fromarray(rgb, "RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.asarray(resized)


def _paste_on_page(
    crop: np.ndarray,
    rng: np.random.Generator,
    fonts_latin,
) -> tuple[np.ndarray, list[int]]:
    page_w = int(rng.integers(780, 1100))
    page_h = int(rng.integers(900, 1300))
    background = _page_background(rng, page_w, page_h, fonts_latin)
    page = Image.fromarray(background, "RGB")
    sprite = Image.fromarray(crop, "RGB")
    max_w = page_w - 16
    max_h = page_h - 16
    if sprite.width > max_w or sprite.height > max_h:
        scale = min(max_w / sprite.width, max_h / sprite.height)
        sprite = sprite.resize(
            (max(16, int(sprite.width * scale)), max(16, int(sprite.height * scale))),
            Image.Resampling.BILINEAR,
        )
    left = int(rng.integers(4, page_w - sprite.width - 3))
    top = int(rng.integers(4, page_h - sprite.height - 3))
    page.paste(sprite, (left, top))
    rgb = _jpeg(np.asarray(page), int(rng.integers(75, 92)))
    return rgb, [left, top, left + sprite.width, top + sprite.height]


def _rec_strip(crop: np.ndarray, layout: str, rng: np.random.Generator, mild: bool) -> np.ndarray:
    # Upright crops: column 0 is 140 degrees, the same canonical angle inference always tries.
    start = UNWRAP_START_DEG + float(rng.uniform(-8.0, 8.0))
    strip = strip_from_crop(crop, layout, 1, 1, start)
    if strip.shape[0] != REC_H or strip.shape[1] != REC_W:
        # grid / hline helpers already resize; keep a guard for odd crops.
        strip = np.asarray(Image.fromarray(strip, "RGB").resize((REC_W, REC_H), Image.Resampling.BILINEAR))
    return _degrade_strip(strip, rng, mild=mild)


def _write_sample(
    dst: Path,
    split: str,
    stem: str,
    page: np.ndarray,
    bbox: list[int],
    text: str,
    cls_name: str,
    layout: str,
    source: str,
    strip: np.ndarray,
) -> str:
    image_dir = dst / "det" / split / "images"
    label_dir = dst / "det" / split / "labels"
    rec_dir = dst / "rec" / split / "images"
    for directory in (image_dir, label_dir, rec_dir):
        directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(page, "RGB").save(image_dir / f"{stem}.jpg", quality=90)
    label = {
        "width": int(page.shape[1]),
        "height": int(page.shape[0]),
        "source": source,
        "seals": [
            {
                "cls": class_id(cls_name),
                "name": cls_name,
                "bbox": bbox,
                "text": text,
                "center_text": "",
                "layout": layout,
                "rows": 1,
                "cols": 1,
            }
        ],
    }
    (label_dir / f"{stem}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
    Image.fromarray(strip, "RGB").save(rec_dir / f"{stem}.png")
    return f"{stem}.png\t{text}"


def _flush_lines(dst: Path, lines: dict[str, list[str]]) -> None:
    for split, rows in lines.items():
        path = dst / "rec" / split / "labels.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def convert_rest_zip(src: Path, dst: Path, limit: int = 0, seed: int = 2026) -> dict:
    """Labeled ReST train crops become pages plus recognition strips.

    The official test zip in this archive has no transcriptions, so it is not
    used. Ten percent of the labeled ids are held out for validation.
    """
    src = Path(src)
    dst = Path(dst)
    rng = np.random.default_rng(seed)
    latin = FontBank(find_latin_font(), find_latin_font()).get("latin", 18)
    lines = {"train": [], "val": []}
    stats = {"train": 0, "val": 0, "skipped": 0, "test_unused": True}
    with zipfile.ZipFile(src) as outer:
        gts = json.loads(outer.read("train/train_gts.json"))
        nested = zipfile.ZipFile(io.BytesIO(outer.read("train/train_images.zip")))
    names = [name for name in nested.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png"))]
    names.sort()
    if limit:
        names = names[:limit]
    for index, name in enumerate(names):
        stem = Path(name).stem
        items = gts.get(f"gt_{stem}") or gts.get(stem)
        if not items:
            stats["skipped"] += 1
            continue
        item = items[0] if isinstance(items, list) else items
        text = str(item.get("transcription") or item.get("text") or "").strip()
        if not text:
            stats["skipped"] += 1
            continue
        crop = _open_rgb(nested.read(name))
        height, width = crop.shape[:2]
        cls_name, layout = guess_seal(text, width, height)
        split = "val" if _val_split(stem) else "train"
        placed = _scaled_crop(crop, rng)
        page, bbox = _paste_on_page(placed, rng, latin)
        # Unwrap the clean crop, not the page, so the title is the training target.
        strip = _rec_strip(crop, layout, rng, mild=(split == "val"))
        row = _write_sample(dst, split, f"rest_{stem}", page, bbox, text, cls_name, layout, "rest", strip)
        lines[split].append(row)
        stats[split] += 1
        if index % 200 == 0:
            print(f"rest {index}/{len(names)} {split} {stem} {cls_name}", flush=True)
    _flush_lines(dst, lines)
    stats["source"] = str(src)
    (dst / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rest done {stats}", flush=True)
    return stats


def _paddle_labels(blob: str) -> list[dict]:
    blob = blob.strip()
    if not blob:
        return []
    if blob[0] not in "[{":
        return [{"transcription": blob, "points": None}]
    payload = json.loads(blob)
    if isinstance(payload, dict):
        payload = [payload]
    rows = []
    for item in payload:
        if isinstance(item, str):
            rows.append({"transcription": item, "points": None})
            continue
        text = item.get("transcription") or item.get("text") or item.get("label") or ""
        points = item.get("points") or item.get("polygon")
        rows.append({"transcription": str(text), "points": points})
    return rows


def _paddle_members(archive: zipfile.ZipFile) -> tuple[list[str], dict[str, str]]:
    names = [name for name in archive.namelist() if not name.endswith("/")]
    txts = [name for name in names if name.endswith("train.txt") or name.endswith("val.txt") or name.endswith("test.txt")]
    files = {name: name for name in names}
    by_base = {}
    for name in names:
        by_base.setdefault(Path(name).name, name)
        by_base.setdefault(name, name)
    return txts, by_base


def convert_paddle_zip(src: Path, dst: Path, limit: int = 0, seed: int = 2027) -> dict:
    """Read dataset/train/train.txt style lists. Images are not kept in git."""
    src = Path(src)
    dst = Path(dst)
    rng = np.random.default_rng(seed)
    latin = FontBank(find_latin_font(), find_latin_font()).get("latin", 18)
    lines = {"train": [], "val": []}
    stats = {"train": 0, "val": 0, "skipped": 0, "label_files": []}
    with zipfile.ZipFile(src) as archive:
        txts, by_base = _paddle_members(archive)
        if not txts:
            sample = archive.namelist()[:12]
            raise SystemExit(f"no train.txt/val.txt in {src}; first entries: {sample}")
        stats["label_files"] = txts
        seen = 0
        for txt_name in txts:
            default_split = "val" if Path(txt_name).name.startswith(("val", "test")) else "train"
            folder = str(Path(txt_name).parent)
            for raw in archive.read(txt_name).decode("utf-8").splitlines():
                if not raw.strip():
                    continue
                if "\t" in raw:
                    rel, blob = raw.split("\t", 1)
                else:
                    rel, blob = raw, ""
                rel = rel.strip().replace("\\", "/")
                candidates = [
                    rel,
                    f"{folder}/{rel}" if folder != "." else rel,
                    f"dataset/{rel}",
                    by_base.get(Path(rel).name, ""),
                ]
                member = next((item for item in candidates if item and item in archive.namelist()), None)
                if member is None:
                    stats["skipped"] += 1
                    continue
                try:
                    labels = _paddle_labels(blob)
                except json.JSONDecodeError:
                    stats["skipped"] += 1
                    continue
                text = "".join(item["transcription"] for item in labels if item["transcription"]).strip()
                if not text:
                    stats["skipped"] += 1
                    continue
                crop = _open_rgb(archive.read(member))
                height, width = crop.shape[:2]
                cls_name, layout = guess_seal(text, width, height)
                key = f"{txt_name}:{rel}"
                split = "val" if default_split == "val" or _val_split(key) else "train"
                if default_split == "val":
                    split = "val"
                placed = _scaled_crop(crop, rng)
                page, bbox = _paste_on_page(placed, rng, latin)
                strip = _rec_strip(crop, layout, rng, mild=(split == "val"))
                stem = f"paddle_{seen:06d}"
                row = _write_sample(dst, split, stem, page, bbox, text, cls_name, layout, "paddle", strip)
                lines[split].append(row)
                stats[split] += 1
                seen += 1
                if seen % 200 == 0:
                    print(f"paddle {seen} {split}", flush=True)
                if limit and seen >= limit:
                    break
            if limit and seen >= limit:
                break
    _flush_lines(dst, lines)
    stats["source"] = str(src)
    (dst / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"paddle done {stats}", flush=True)
    return stats


def link_mix(sources: list[tuple[str, Path]], dst: Path) -> dict:
    """Symlink prepared roots into one detection/recognition tree and union the charset."""
    dst = Path(dst)
    rec_lines = {"train": [], "val": []}
    counts = {"det_train": 0, "det_val": 0, "rec_train": 0, "rec_val": 0}
    extra_chars: list[str] = []
    seen = set(all_chars())
    charset = list(all_chars())
    for prefix, root in sources:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(root)
        for split in ("train", "val"):
            label_dir = root / "det" / split / "labels"
            image_dir = root / "det" / split / "images"
            if label_dir.is_dir():
                for label in sorted(label_dir.glob("*.json")):
                    image = image_dir / f"{label.stem}.jpg"
                    if not image.is_file():
                        continue
                    out_stem = f"{prefix}{label.stem}"
                    _symlink(image, dst / "det" / split / "images" / f"{out_stem}.jpg")
                    _symlink(label, dst / "det" / split / "labels" / f"{out_stem}.json")
                    counts[f"det_{split}"] += 1
            tsv = root / "rec" / split / "labels.tsv"
            rec_root = root / "rec" / split / "images"
            if tsv.is_file():
                for line in tsv.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    name, text = line.split("\t", 1)
                    src_image = rec_root / name
                    if not src_image.is_file():
                        continue
                    out_name = f"{prefix}{name}"
                    _symlink(src_image, dst / "rec" / split / "images" / out_name)
                    rec_lines[split].append(f"{out_name}\t{text}")
                    counts[f"rec_{split}"] += 1
                    for ch in text:
                        if ch not in seen:
                            seen.add(ch)
                            extra_chars.append(ch)
                            charset.append(ch)
    for split, rows in rec_lines.items():
        path = dst / "rec" / split / "labels.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    write_charset_file(dst / "charset.txt", charset)
    counts["charset"] = len(charset)
    counts["charset_extra"] = len(extra_chars)
    (dst / "stats.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    return counts


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)
