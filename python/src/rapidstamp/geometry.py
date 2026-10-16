"""Geometry shared by training and inference.

Angles are radians in image coordinates: 0 is +x (right), and the angle
increases clockwise because y grows downward. The text arc is
``TEXT_FRACTION`` of the circle. The remaining gap is what
``estimate_text_start`` locks onto, so a rotated round seal still unwraps
into a left-to-right strip.
"""

from __future__ import annotations

import math

import numpy as np

DET_SIZE = 128
GRID = 8
CELL = DET_SIZE // GRID
UNWRAP_H = 32
UNWRAP_W = 160
TEXT_FRACTION = 0.78
TEXT_OUTER_FRAC = 0.86
TEXT_INNER_FRAC = 0.58
BORDER_FRAC = 0.055
N_ANGLES = 360

GAP = 2.0 * math.pi * (1.0 - TEXT_FRACTION)
# Gap centered on the bottom of the seal when rotation is 0.
CANONICAL_START = math.pi / 2.0 + GAP / 2.0
# Characters sit this far inside each end of the text arc.
TEXT_EDGE = math.radians(6.0)
CHAR_SIZE = 32


def slot_centers(n_chars: int, width: int = UNWRAP_W) -> list[float]:
    """Column index of each character center on an unwrapped strip."""
    arc = 2.0 * math.pi * TEXT_FRACTION
    edge_frac = TEXT_EDGE / arc
    usable = 1.0 - 2.0 * edge_frac
    return [(edge_frac + (i + 0.5) * usable / n_chars) * width - 0.5 for i in range(n_chars)]


def crop_slot(strip: np.ndarray, center_x: float, size: int = CHAR_SIZE) -> np.ndarray:
    """Cut a square window around ``center_x``. Out-of-range pixels are white."""
    height, width = strip.shape
    x0 = int(round(center_x - size / 2.0))
    out = np.ones((height, size), dtype=np.float32)
    src_x0 = max(0, x0)
    src_x1 = min(width, x0 + size)
    dst_x0 = src_x0 - x0
    if src_x1 > src_x0:
        out[:, dst_x0 : dst_x0 + (src_x1 - src_x0)] = strip[:, src_x0:src_x1]
    return out


def resize_rgb(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize, ``align_corners=False`` (pixel centers).

    Returns float32 HWC in 0..1. Out-of-range samples are white.
    """
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        raise TypeError("resize_rgb expects uint8 HWC")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB, got {image.shape}")
    src_h, src_w = image.shape[:2]
    if src_h == out_h and src_w == out_w:
        return image.astype(np.float32) / 255.0
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (src_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (src_w / out_w) - 0.5
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return sample_bilinear(image, grid_x, grid_y) / 255.0


def sample_bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample uint8 HWC or HW at floating pixel coordinates. OOB is 255."""
    if image.dtype != np.uint8:
        raise TypeError("sample_bilinear expects uint8")
    h, w = image.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (xs - x0).astype(np.float32)
    wy = (ys - y0).astype(np.float32)

    def gather(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        valid = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)
        pix = image[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)].astype(np.float32)
        if pix.ndim == xs.ndim:
            pix = pix.copy()
            pix[~valid] = 255.0
        else:
            pix = pix.copy()
            pix[~valid] = 255.0
        return pix

    wa = (1.0 - wx) * (1.0 - wy)
    wb = wx * (1.0 - wy)
    wc = (1.0 - wx) * wy
    wd = wx * wy
    ia = gather(x0, y0)
    ib = gather(x1, y0)
    ic = gather(x0, y1)
    id_ = gather(x1, y1)
    if ia.ndim == xs.ndim + 1:
        wa, wb, wc, wd = wa[..., None], wb[..., None], wc[..., None], wd[..., None]
    return ia * wa + ib * wb + ic * wc + id_ * wd


def box_to_ring(box: tuple[float, float, float, float] | list[float]) -> tuple[float, float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0, 0.5 * min(w, h)


def angular_ink(
    image: np.ndarray,
    cx: float,
    cy: float,
    r_in: float,
    r_out: float,
    n_angles: int = N_ANGLES,
    n_radii: int = 8,
) -> np.ndarray:
    """Mean distance-from-white along the text ring. Shape ``(n_angles,)``."""
    angles = (np.arange(n_angles, dtype=np.float32) + 0.5) * (2.0 * math.pi / n_angles)
    radii = np.linspace(r_in, r_out, n_radii, dtype=np.float32)
    ang = angles[:, None]
    rad = radii[None, :]
    xs = cx + np.cos(ang) * rad
    ys = cy + np.sin(ang) * rad
    pix = sample_bilinear(image, xs.astype(np.float32), ys.astype(np.float32))
    ink = np.mean(255.0 - pix, axis=-1)
    return ink.mean(axis=1).astype(np.float32)


def smooth_circular(values: np.ndarray, k: int = 7) -> np.ndarray:
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ext = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(ext, kernel, mode="valid").astype(np.float32)


def estimate_text_start(ink: np.ndarray, text_fraction: float = TEXT_FRACTION) -> int:
    """Return the text-arc start as an angle-bin index in ``[0, n)``.

    Bin ``i`` covers ``[i * 2pi / n, (i + 1) * 2pi / n)``. The returned index
    is the first bin of the highest-ink window of length ``text_fraction``.
    """
    smoothed = smooth_circular(np.asarray(ink, dtype=np.float32), 7)
    n = int(smoothed.shape[0])
    gap_bins = int(round(n * (1.0 - text_fraction)))
    gap_bins = min(max(gap_bins, 1), n - 1)
    text_bins = n - gap_bins
    ext = np.concatenate([smoothed, smoothed])
    csum = np.empty(ext.shape[0] + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(ext, out=csum[1:])
    sums = csum[text_bins : text_bins + n] - csum[:n]
    return int(np.argmax(sums))


def polar_unwrap(
    image: np.ndarray,
    cx: float,
    cy: float,
    r_in: float,
    r_out: float,
    start_angle: float,
    arc: float,
    out_h: int = UNWRAP_H,
    out_w: int = UNWRAP_W,
) -> np.ndarray:
    """Unwrap a ring to a grayscale float32 image in 0..1, shape ``(H, W)``.

    Row 0 is the outer radius. Column 0 is ``start_angle``.
    """
    rows = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    cols = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    radii = r_out + (r_in - r_out) * rows
    angles = start_angle + cols * arc
    ang = angles[None, :]
    rad = radii[:, None]
    xs = (cx + np.cos(ang) * rad).astype(np.float32)
    ys = (cy + np.sin(ang) * rad).astype(np.float32)
    pix = sample_bilinear(image, xs, ys)
    gray = pix.mean(axis=-1) / 255.0
    return gray.astype(np.float32)


def _linspace(lo: float, hi: float, n: int) -> np.ndarray:
    return lo + (hi - lo) * np.arange(n, dtype=np.float64) / float(n - 1)


def _ring_contrast(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    dcxs: np.ndarray,
    dcys: np.ndarray,
    scales: np.ndarray,
) -> tuple[float, float, float]:
    """Pick the circle whose border ring is darker than its interior.

    Loop order matches the C++ search: ``dcx``, then ``dcy``, then scale.
    Ties keep the first candidate.
    """
    dcx, dcy, scale = np.meshgrid(dcxs, dcys, scales, indexing="ij")
    dcx = dcx.reshape(-1)
    dcy = dcy.reshape(-1)
    scale = scale.reshape(-1)
    ccx = cx + dcx * radius
    ccy = cy + dcy * radius
    rr = radius * scale
    n_ang = 72
    ang = (np.arange(n_ang, dtype=np.float64) + 0.5) * (2.0 * math.pi / n_ang)
    cos = np.cos(ang)
    sin = np.sin(ang)

    def ring(frac: float) -> np.ndarray:
        rad = rr * frac
        xs = ccx[:, None] + cos[None, :] * rad[:, None]
        ys = ccy[:, None] + sin[None, :] * rad[:, None]
        pix = sample_bilinear(image, xs.astype(np.float32), ys.astype(np.float32))
        return (255.0 - pix.astype(np.float64)).mean(axis=-1).mean(axis=-1)

    score = ring(1.0 - BORDER_FRAC * 0.5) - ring(0.35)
    j = int(np.argmax(score))
    return float(ccx[j]), float(ccy[j]), float(rr[j])


def refine_ring(image: np.ndarray, box: list[float] | tuple[float, float, float, float]) -> list[float]:
    """Snap a coarse detector box onto the circular ink border.

    The tiny detector's box center is often off by a fraction of the radius,
    which is enough to smear a polar unwrap. A coarse-to-fine search of the
    border-versus-interior contrast pulls the circle back. The returned box
    is a square ``xywh``.
    """
    cx, cy, radius = box_to_ring(box)
    if radius < 4.0:
        return [cx - radius, cy - radius, radius * 2.0, radius * 2.0]
    bx, by, br = _ring_contrast(
        image,
        cx,
        cy,
        radius,
        _linspace(-0.40, 0.40, 9),
        _linspace(-0.40, 0.40, 9),
        _linspace(0.80, 1.20, 5),
    )
    base_dx = (bx - cx) / radius
    base_dy = (by - cy) / radius
    base_sc = br / radius
    fx, fy, fr = _ring_contrast(
        image,
        cx,
        cy,
        radius,
        _linspace(base_dx - 0.10, base_dx + 0.10, 9),
        _linspace(base_dy - 0.10, base_dy + 0.10, 9),
        _linspace(base_sc - 0.08, base_sc + 0.08, 5),
    )
    return [fx - fr, fy - fr, fr * 2.0, fr * 2.0]


def unwrap_seal(image: np.ndarray, box: list[float] | tuple[float, float, float, float]) -> np.ndarray:
    """Cut a detected round seal into a recognizer strip."""
    cx, cy, radius = box_to_ring(box)
    r_out = radius * TEXT_OUTER_FRAC
    r_in = radius * TEXT_INNER_FRAC
    ink = angular_ink(image, cx, cy, r_in, r_out, N_ANGLES)
    start_idx = estimate_text_start(ink, TEXT_FRACTION)
    start = start_idx * (2.0 * math.pi / N_ANGLES)
    arc = 2.0 * math.pi * TEXT_FRACTION
    return polar_unwrap(image, cx, cy, r_in, r_out, start, arc, UNWRAP_H, UNWRAP_W)


def circular_distance(a: float, b: float) -> float:
    d = (a - b) % (2.0 * math.pi)
    return float(min(d, 2.0 * math.pi - d))
