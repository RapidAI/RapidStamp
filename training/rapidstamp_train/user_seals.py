"""Boxes and a frozen split for the user-supplied seal crops.

The zip has pngs under images/ and compound-img/ and no transcription.
Each file is one seal crop. Gold text stays empty: a model reading is not
a label. The box is the tight extent of red, blue, or (on a gray scan)
dark ink. A nearly square box whose corners are paper is a round seal.
Finance, invoice, and Latin classes are not assigned without text.

Files that share a capture id (filename stem with a trailing _N removed)
or the same pixels are one group. The group key is the sorted capture ids.
sha256(group key) first 8 hex digits, mod 10, bucket < 3 is test. An exact
pixel copy is recorded as a duplicate and is not written again, so a test
crop cannot also sit in the training tree.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rapidstamp.labels import CLASS_LAYOUT, class_id


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path)
    if image.mode == "RGBA":
        # A straight RGB convert paints transparent pixels black, which the
        # dark-ink mask then treats as a seal.
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        canvas.paste(image, mask=image.getchannel("A"))
        return np.asarray(canvas)
    return np.asarray(image.convert("RGB"))


def ink_mask(rgb: np.ndarray) -> np.ndarray:
    """Red, blue, or dark ink. 1 where the pixel is ink."""
    red_ch = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    red = (red_ch > 80) & (red_ch > green + 16) & (red_ch > blue + 16)
    blu = (blue > 80) & (blue > red_ch + 16) & (blue > green + 8)
    chroma = red | blu
    luminance = (red_ch.astype(np.int32) + green + blue) / 3.0
    paper = float(np.percentile(luminance, 92))
    dark = luminance < paper - max(22.0, 0.14 * paper)
    if float(chroma.mean()) >= 0.02:
        mask = chroma
    else:
        mask = dark
        if float(mask.mean()) > 0.5:
            mask = luminance <= float(np.percentile(luminance, 15))
        if float(mask.mean()) < 0.008:
            pale = (red_ch > 60) & (red_ch > green + 8) & (red_ch > blue + 8)
            mask = pale | (luminance <= float(np.percentile(luminance, 12)))
    return mask.astype(np.uint8)


def _significant(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    clean = np.zeros_like(binary)
    # Keep short arcs of a broken ring. A page-relative cutoff drops those
    # arcs and the box clips the seal. Isolated specks below 40 px do not
    # move the box unless they sit in a corner, which these crops do not.
    minimum = 40
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum:
            clean[labels == index] = 1
    if int(clean.sum()) < 30:
        return binary
    return clean


def seal_box(mask: np.ndarray) -> list[int] | None:
    """One xyxy box around the ink. These crops hold a single seal."""
    clean = _significant(mask)
    ys, xs = np.where(clean > 0)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    return [
        max(0, int(xs.min()) - 1),
        max(0, int(ys.min()) - 1),
        min(width, int(xs.max()) + 2),
        min(height, int(ys.max()) + 2),
    ]


def seal_class(mask: np.ndarray, box: list[int]) -> str:
    x1, y1, x2, y2 = box
    crop = mask[y1:y2, x1:x2]
    height, width = crop.shape
    if height < 4 or width < 4:
        return "round_official"
    aspect = width / max(height, 1)
    side = max(3, int(0.16 * min(height, width)))
    corners = np.concatenate(
        [
            crop[:side, :side].ravel(),
            crop[:side, -side:].ravel(),
            crop[-side:, :side].ravel(),
            crop[-side:, -side:].ravel(),
        ]
    )
    corner = float(corners.mean())
    # A square frame paints the corners. A circle leaves them as paper.
    if corner >= 0.45 and 0.85 <= aspect <= 1.18:
        return "square_personal"
    if aspect >= 1.28 or aspect <= 0.78:
        return "rect_stamp" if corner >= 0.15 else "oval"
    return "round_official"


def annotate(rgb: np.ndarray) -> list[dict]:
    mask = ink_mask(rgb)
    box = seal_box(mask)
    if box is None:
        return []
    x1, y1, x2, y2 = box
    if min(x2 - x1, y2 - y1) < 8:
        return []
    if float(mask[y1:y2, x1:x2].mean()) < 0.01:
        return []
    name = seal_class(mask, box)
    return [
        {
            "cls": class_id(name),
            "name": name,
            "bbox": [int(v) for v in box],
            "text": "",
            "layout": CLASS_LAYOUT[name],
            "rows": 1,
            "cols": 1,
        }
    ]


def capture_id(name: str) -> str:
    stem = Path(name).stem
    if len(stem) > 2 and stem[-2] == "_" and stem[-1].isdigit():
        return stem[:-2]
    return stem


def _find(parent: dict[int, int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def assign_splits(rows: list[dict]) -> None:
    """Group related crops and set split in place.

    ``rows`` need ``file``, ``capture``, and ``pixel_md5``. Exact copies
    are ``duplicate``. A crop with no seal is ``skipped`` and does not
    decide the group.
    """
    parent = {index: index for index in range(len(rows))}

    def union(left: int, right: int) -> None:
        root_left, root_right = _find(parent, left), _find(parent, right)
        if root_left != root_right:
            parent[root_right] = root_left

    by_capture: dict[str, int] = {}
    by_pixels: dict[str, int] = {}
    for index, row in enumerate(rows):
        capture = row["capture"]
        if capture in by_capture:
            union(index, by_capture[capture])
        else:
            by_capture[capture] = index
        pixels = row["pixel_md5"]
        if pixels in by_pixels:
            union(index, by_pixels[pixels])
        else:
            by_pixels[pixels] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(_find(parent, index), []).append(index)
    for members in groups.values():
        captures = sorted({rows[index]["capture"] for index in members})
        key = ",".join(captures)
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 10
        split = "test" if bucket < 3 else "train"
        seen: dict[str, str] = {}
        for index in sorted(members, key=lambda item: rows[item]["file"]):
            row = rows[index]
            row["group"] = key
            pixels = row["pixel_md5"]
            if pixels in seen:
                row["split"] = "duplicate"
                row["duplicate_of"] = seen[pixels]
                continue
            seen[pixels] = row["file"]
            row["duplicate_of"] = ""
            row["split"] = "skipped" if not row.get("seals") else split


# Long side of the pasted seal, as a fraction of the page. The last value
# is the full-frame case the published detector misses.
PASTE_SCALES = (0.22, 0.45, 0.72, 0.96)


def pasted_page(
    crop: np.ndarray,
    box: list[int],
    scale: float,
    page: int = 640,
    origin: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Paste the seal box onto a square page. ``scale`` is its long side / page."""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1 = max(0, min(crop.shape[1] - 1, x1))
    y1 = max(0, min(crop.shape[0] - 1, y1))
    x2 = max(x1 + 1, min(crop.shape[1], x2))
    y2 = max(y1 + 1, min(crop.shape[0], y2))
    seal = np.ascontiguousarray(crop[y1:y2, x1:x2])
    src_h, src_w = seal.shape[:2]
    target = max(16, int(round(float(scale) * page)))
    ratio = target / max(src_h, src_w, 1)
    new_w = max(1, min(page, int(round(src_w * ratio))))
    new_h = max(1, min(page, int(round(src_h * ratio))))
    resized = np.asarray(Image.fromarray(seal).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((page, page, 3), 244, np.uint8)
    if origin is None:
        left = max(0, (page - new_w) // 2)
        top = max(0, (page - new_h) // 2)
    else:
        left, top = origin
        left = min(max(0, left), page - new_w)
        top = min(max(0, top), page - new_h)
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, [left, top, left + new_w, top + new_h]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def pixel_md5(rgb: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest()
