"""Filter Aliyun seal labels and unwrap the ring into one text line.

The recognizer is a fine-tune of the official PP-OCRv4 Chinese model, which
reads a horizontal strip. Each kept crop is recentered on its outer ink,
then unwrapped with the same polar sampler the runtime uses. A label is
never attached to a second, badly rotated copy of the same crop.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from rapidstamp.geometry import estimate_ring_start, pad_to_square, polar_unwrap
from rapidstamp.labels import UNWRAP_START_DEG

MIN_RING_PROB = 0.90
MIN_RING_CHARS = 2
MAX_RING_CHARS = 25
VAL_BUCKETS = 1000
VAL_HOLD_OUT = 30
WIDE_GAP_DEG = 40.0
STRIP_H = 48
STRIP_W = 640
RING_CLASSES = frozenset(
    {
        "round_official",
        "oval",
        "invoice",
        "finance",
        "notary",
        "company_latin",
    }
)
_TEMPLATE = re.compile(
    "|".join(
        [
            "某某",
            r"[Xx×]{2,}",
            "演示章",
            "测试专用",
            "样章",
            "sample",
            "示例章",
            "双击这里",
            "替换代用",
            "公司名称",
            "此处可以输",
        ]
    ),
    re.IGNORECASE,
)


def is_template(text: str) -> bool:
    return _TEMPLATE.search(text) is not None


def ring_probability(row: dict) -> float | None:
    probs = row.get("probs")
    if not isinstance(probs, dict) or "ring" not in probs:
        return None
    try:
        return float(probs["ring"])
    except (TypeError, ValueError):
        return None


def ring_label(row: dict) -> str:
    return str(row.get("ring_text") or "").strip()


def label_reason(row: dict) -> str | None:
    """First reason to drop a label row, before any image is opened."""
    if row.get("status") not in (200, "200"):
        return "status"
    text = ring_label(row)
    if not text:
        return "empty"
    prob = ring_probability(row)
    if prob is None or prob < MIN_RING_PROB:
        return "prob"
    if is_template(text):
        return "template"
    if row.get("det_class") not in RING_CLASSES:
        return "class"
    if not (MIN_RING_CHARS <= len(text) <= MAX_RING_CHARS):
        return "length"
    return None


def val_split(name: str, hold_out: int = VAL_HOLD_OUT) -> str:
    """Stable ~3% hold-out from the file name, after filtering and dedupe."""
    digest = hashlib.sha256(Path(name).name.encode("utf-8")).hexdigest()
    if int(digest[:8], 16) % VAL_BUCKETS < hold_out:
        return "val"
    return "train"


def dedupe_by_md5(rows: list[dict]) -> list[dict]:
    """Keep the highest ring probability for each image-byte md5."""
    best: dict[str, dict] = {}
    for row in rows:
        key = row["file_md5"]
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        if row["prob"] > current["prob"] or (
            row["prob"] == current["prob"] and row["file"] < current["file"]
        ):
            best[key] = row
    return [best[key] for key in sorted(best)]


def pixel_md5(rgb: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def file_digests(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    return hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.ascontiguousarray(image.convert("RGB"))


def load_charset(path: Path) -> set[str]:
    chars: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.endswith("\r"):
            line = line[:-1]
        if line:
            chars.add(line)
    return chars


def unsupported_chars(text: str, charset: set[str]) -> str:
    """Characters the PP-OCR dictionary cannot learn. Space is allowed."""
    return "".join(ch for ch in text if ch != " " and ch not in charset)


def load_gold(manifest: Path) -> dict[str, set[str]]:
    """Identities of the 63 gold test crops. Training must not see them."""
    gold = {
        "basenames": set(),
        "captures": set(),
        "sha256": set(),
        "pixel_md5": set(),
        "stems": set(),
    }
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "test":
                continue
            name = Path(str(row.get("file") or "")).name
            if name:
                gold["basenames"].add(name)
                gold["stems"].add(Path(name).stem)
            capture = str(row.get("capture") or "")
            if capture:
                gold["captures"].add(capture)
            digest = str(row.get("sha256") or "")
            if digest:
                gold["sha256"].add(digest)
                gold["stems"].add(digest[:16])
            pixels = str(row.get("pixel_md5") or "")
            if pixels:
                gold["pixel_md5"].add(pixels)
    return gold


def is_gold(name: str, file_sha: str, pixels: str, gold: dict[str, set[str]]) -> bool:
    base = Path(name).name
    stem = Path(base).stem
    if base in gold["basenames"]:
        return True
    if stem in gold["captures"] or stem in gold["stems"]:
        return True
    if file_sha and file_sha in gold["sha256"]:
        return True
    if pixels and pixels in gold["pixel_md5"]:
        return True
    return False


def _ink(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    ink = (red > 120) & (red > green + 30) & (red > blue + 30)
    ink |= (blue > 100) & (blue > red + 20) & (blue > green + 10)
    ink |= (red < 80) & (green < 80) & (blue < 80)
    return ink


def _outer_points(xs: np.ndarray, ys: np.ndarray, cx: float, cy: float) -> np.ndarray | None:
    """One absolute point per angle bin, on the outer edge of the ink."""
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    rad = np.hypot(dx, dy)
    keep = rad >= np.percentile(rad, 45)
    if int(keep.sum()) < 20:
        return None
    xs = xs[keep]
    ys = ys[keep]
    rad = rad[keep]
    angle = np.degrees(np.arctan2(dy[keep], dx[keep])) % 360.0
    bins = 72
    bin_id = np.clip((angle / 360.0 * bins).astype(np.int32), 0, bins - 1)
    points = []
    for index in range(bins):
        chosen = np.where(bin_id == index)[0]
        if chosen.size < 2:
            continue
        outer = np.percentile(rad[chosen], 90)
        near = chosen[rad[chosen] >= outer * 0.98]
        if near.size == 0:
            continue
        points.append((float(xs[near].mean()), float(ys[near].mean())))
    if len(points) < 12:
        return None
    pts = np.asarray(points, dtype=np.float64)
    center = np.array([cx, cy], dtype=np.float64)
    dist = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
    typical = float(np.median(dist))
    if typical < 4:
        return None
    ok = (dist > typical * 0.72) & (dist < typical * 1.40)
    if int(ok.sum()) >= 8:
        pts = pts[ok]
    return pts


def _fit_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    """Algebraic circle fit. Returns center x, center y, radius."""
    xs = points[:, 0]
    ys = points[:, 1]
    design = np.column_stack([xs, ys, np.ones(len(points))])
    target = -(xs * xs + ys * ys)
    try:
        solved, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx = float(-solved[0] / 2.0)
    cy = float(-solved[1] / 2.0)
    radius_sq = cx * cx + cy * cy - float(solved[2])
    if not np.isfinite(radius_sq) or radius_sq <= 1.0:
        return None
    return cx, cy, float(np.sqrt(radius_sq))


def _inner_ratio(local_r: np.ndarray) -> float:
    """Inner edge of the outer ink band, in units of the fitted ellipse."""
    if local_r.size < 20:
        return 0.78
    hist, edges = np.histogram(local_r, bins=32, range=(0.40, 1.20))
    centers = (edges[:-1] + edges[1:]) / 2.0
    outer = np.where(centers >= 0.72)[0]
    if outer.size == 0 or int(hist[outer].max()) < 5:
        return 0.78
    peak = int(outer[int(np.argmax(hist[outer]))])
    level = max(3, int(0.30 * hist[peak]))
    inner = peak
    while inner > 0 and hist[inner] >= level:
        inner -= 1
    ratio = float(edges[inner])
    return float(np.clip(ratio, 0.55, 0.90))


def fit_ring_canvas(rgb: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Canvas whose inscribed ellipse matches the outer ink, plus ring radii.

    ``polar_unwrap`` always samples the ellipse inscribed in the image, so
    the crop is pasted onto a new canvas instead of changing that sampler.
    ``r_in`` and ``r_out`` are fractions of that ellipse.
    """
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    height, width = rgb.shape[:2]
    fallback = (pad_to_square(rgb), 0.72, 0.96)
    ink = _ink(rgb)
    ys, xs = np.nonzero(ink)
    if xs.size < 40:
        return fallback
    cx = float(np.median(xs))
    cy = float(np.median(ys))
    fitted = None
    points = None
    for _ in range(4):
        points = _outer_points(xs.astype(np.float64), ys.astype(np.float64), cx, cy)
        if points is None:
            return fallback
        fitted = _fit_circle(points)
        if fitted is None:
            return fallback
        cx, cy, _radius = fitted
    if fitted is None or points is None:
        return fallback
    radius = fitted[2]
    if not (-0.5 * width <= cx <= 1.5 * width and -0.5 * height <= cy <= 1.5 * height):
        return fallback
    if radius < 8:
        return fallback
    extent_x = float(np.percentile(np.abs(points[:, 0] - cx), 95))
    extent_y = float(np.percentile(np.abs(points[:, 1] - cy), 95))
    aspect = max(extent_x, extent_y) / max(min(extent_x, extent_y), 1.0)
    if aspect < 1.18:
        rad_x = rad_y = radius
    else:
        rad_x = max(extent_x, radius * 0.45) * 1.06
        rad_y = max(extent_y, radius * 0.45) * 1.06
    rad_x = float(np.clip(rad_x, 4.0, max(width, height)))
    rad_y = float(np.clip(rad_y, 4.0, max(width, height)))
    if rad_x < 0.12 * width and rad_y < 0.12 * height:
        return fallback
    dx = xs.astype(np.float64) - cx
    dy = ys.astype(np.float64) - cy
    local = np.sqrt((dx / rad_x) ** 2 + (dy / rad_y) ** 2)
    outer = local >= 0.55
    ratio = _inner_ratio(local[outer] if int(outer.sum()) >= 20 else local)
    r_out = 0.96
    r_in = float(np.clip(ratio * r_out, 0.50, r_out - 0.05))
    canvas_w = int(np.clip(round(2 * rad_x / r_out) + 1, 32, 1280))
    canvas_h = int(np.clip(round(2 * rad_y / r_out) + 1, 32, 1280))
    canvas = np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    dst_cx = (canvas_w - 1) / 2.0
    dst_cy = (canvas_h - 1) / 2.0
    x0 = int(round(dst_cx - cx))
    y0 = int(round(dst_cy - cy))
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    copy_w = min(width - src_x0, canvas_w - dst_x0)
    copy_h = min(height - src_y0, canvas_h - dst_y0)
    if copy_w <= 0 or copy_h <= 0:
        return fallback
    canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = rgb[
        src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w
    ]
    if int(_ink(canvas).sum()) < 40:
        return fallback
    return canvas, r_in, r_out


def unwrap_ring(rgb: np.ndarray, out_w: int = STRIP_W, out_h: int = STRIP_H) -> dict:
    """One strip, starting at the ink gap when that gap is trustworthy."""
    canvas, r_in, r_out = fit_ring_canvas(rgb)
    start, gap = estimate_ring_start(canvas, r_in=r_in, r_out=r_out)
    if start is None:
        start = float(UNWRAP_START_DEG)
    strip = polar_unwrap(
        canvas,
        out_w=out_w,
        out_h=out_h,
        r_in=r_in,
        r_out=r_out,
        start_deg=float(start),
        span_deg=360.0,
    )
    return {
        "strip": strip,
        "start_deg": float(start),
        "gap_deg": float(gap),
        "r_in": float(r_in),
        "r_out": float(r_out),
        "wide_gap": float(gap) >= WIDE_GAP_DEG,
    }


def strip_name(file_name: str, wide_gap: bool) -> str:
    stem = Path(file_name).stem
    if wide_gap:
        return f"{stem}__gap.png"
    return f"{stem}.png"


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        current = [index]
        for jndex, other in enumerate(right, 1):
            insert = current[jndex - 1] + 1
            delete = previous[jndex] + 1
            replace = previous[jndex - 1] + (char != other)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def one_ned(pred: str, gold: str) -> float:
    """1 − edit_distance / max(len). Spaces count. Both empty scores 1."""
    if pred == gold:
        return 1.0
    denom = max(len(pred), len(gold))
    if denom == 0:
        return 1.0
    return 1.0 - edit_distance(pred, gold) / denom


def summarize_pairs(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {"n": 0, "exact": 0, "exact_rate": None, "one_ned": None}
    exact = sum(pred == gold for pred, gold in pairs)
    score = sum(one_ned(pred, gold) for pred, gold in pairs) / len(pairs)
    return {
        "n": len(pairs),
        "exact": exact,
        "exact_rate": exact / len(pairs),
        "one_ned": score,
    }


def worst_pairs(rows: list[tuple[str, str, str]], limit: int = 10) -> list[dict]:
    """``rows`` are (name, pred, gold), worst 1-NED first."""
    ranked = sorted(rows, key=lambda item: (one_ned(item[1], item[2]), item[0]))
    out = []
    for name, pred, gold in ranked[:limit]:
        out.append(
            {
                "name": name,
                "pred": pred,
                "gold": gold,
                "exact": pred == gold,
                "one_ned": one_ned(pred, gold),
                "distance": edit_distance(pred, gold),
            }
        )
    return out
