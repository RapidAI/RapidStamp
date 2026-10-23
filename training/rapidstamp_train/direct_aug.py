"""Charset, gold-crop matching, and augmentation for the direct seal reader.

The reader looks at the whole crop, so a rotation does not change the ring
string. Labels stay as written while the image is spun, warped, and degraded.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MAX_CHARS = 25

_OVERLAY = "有限公司专用章测试北京市人民政府公安局委员会0123456789"


class Charset:
    """PP-OCR characters plus space. Ids 0, 1, 2 are pad, bos, eos."""

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {char: index + 3 for index, char in enumerate(self.chars)}
        self.itos = {index: char for char, index in self.stoi.items()}
        self.max_len = MAX_CHARS + 1

    def __len__(self) -> int:
        return len(self.chars) + 3

    def encode(self, text: str) -> list[int] | None:
        if len(text) > MAX_CHARS:
            return None
        ids = []
        for char in text:
            index = self.stoi.get(char)
            if index is None:
                return None
            ids.append(index)
        ids.append(EOS_ID)
        return ids

    def decode(self, ids) -> str:
        chars = []
        for raw in ids:
            index = int(raw)
            if index in (EOS_ID, PAD_ID):
                break
            if index == BOS_ID:
                continue
            char = self.itos.get(index)
            if char:
                chars.append(char)
        return "".join(chars)


def load_charset(path: Path) -> Charset:
    chars: list[str] = []
    seen: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.endswith("\r"):
            line = line[:-1]
        if not line or line in seen:
            continue
        chars.append(line)
        seen.add(line)
    if " " not in seen:
        chars.append(" ")
    return Charset(chars)


def match_gold_image(file_name: str, paths: list[Path]) -> Path | None:
    """Find ``NNN_<manifest-name>.png`` as well as an exact file name."""
    base = Path(file_name).name
    if not base:
        return None
    hits: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.name == base or path.name.endswith(base):
            hits.append(path)
    if len(hits) == 1:
        return hits[0]
    return None


def list_images(root: Path) -> list[Path]:
    paths = []
    if not root.is_dir():
        return paths
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            paths.append(path)
    return paths


def pad_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = max(width, height, 1)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def perspective_coeffs(output_xy: list[tuple[float, float]], input_xy: list[tuple[float, float]]) -> list[float]:
    """PIL perspective coefficients. Output pixels are sampled from input points."""
    matrix = []
    target = []
    for (x, y), (src_x, src_y) in zip(output_xy, input_xy):
        matrix.append([x, y, 1, 0, 0, 0, -src_x * x, -src_x * y])
        matrix.append([0, 0, 0, x, y, 1, -src_y * x, -src_y * y])
        target.extend([src_x, src_y])
    solved = np.linalg.lstsq(np.asarray(matrix, dtype=np.float64), np.asarray(target), rcond=None)[0]
    return [float(value) for value in solved]


def _perspective(image: Image.Image, rng: random.Random, jitter: float = 0.10) -> Image.Image:
    width, height = image.size
    output = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    source = []
    for x, y in output:
        source.append((x + rng.uniform(-jitter, jitter) * width, y + rng.uniform(-jitter, jitter) * height))
    coeffs = perspective_coeffs(output, source)
    return image.transform(image.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR, fillcolor=(255, 255, 255))


def _font(path: Path | None, size: int):
    if path is not None and Path(path).is_file():
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _overlay_text(image: Image.Image, rng: random.Random, font_path: Path | None) -> Image.Image:
    width, height = image.size
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for _ in range(rng.randint(1, 2)):
        count = rng.randint(2, 8)
        text = "".join(rng.choice(_OVERLAY) for _ in range(count))
        size = rng.randint(max(12, height // 14), max(18, height // 7))
        font = _font(font_path, size)
        color = rng.choice([(180, 30, 30, 210), (20, 20, 20, 200), (80, 80, 80, 180)])
        text_img = Image.new("RGBA", (size * count + 8, size + 8), (0, 0, 0, 0))
        ImageDraw.Draw(text_img).text((2, 0), text, fill=color, font=font)
        text_img = text_img.rotate(rng.uniform(0, 360), expand=True, resample=Image.BILINEAR)
        x = rng.randint(0, max(0, width - text_img.size[0]))
        y = rng.randint(0, max(0, height - text_img.size[1]))
        layer.alpha_composite(text_img, (x, y))
    base = image.convert("RGBA")
    return Image.alpha_composite(base, layer).convert("RGB")


def _occlude(image: Image.Image, rng: random.Random) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(rng.randint(1, 3)):
        box_w = int(width * rng.uniform(0.08, 0.28))
        box_h = int(height * rng.uniform(0.04, 0.18))
        x0 = rng.randint(0, max(0, width - box_w))
        y0 = rng.randint(0, max(0, height - box_h))
        color = rng.choice([(255, 255, 255), (230, 230, 230), (190, 40, 40), (30, 30, 30)])
        draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=color)
    return image


def _jpeg(image: Image.Image, rng: random.Random) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(28, 82))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def augment_seal(image: Image.Image, rng: random.Random, font_path: Path | None = None) -> Image.Image:
    """Full-turn rotation plus the degradations seen on scanned seals."""
    image = image.convert("RGB")
    image = image.rotate(rng.uniform(0.0, 360.0), resample=Image.BILINEAR, fillcolor=(255, 255, 255))
    if rng.random() < 0.75:
        image = _perspective(image, rng, jitter=0.10)
    if rng.random() < 0.45:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.8)))
    arr = np.asarray(image).astype(np.float32)
    if rng.random() < 0.5:
        noise = np.random.default_rng(rng.randrange(1 << 31)).normal(0.0, rng.uniform(3.0, 16.0), arr.shape)
        arr += noise
    if rng.random() < 0.45:
        keep = rng.uniform(0.50, 0.90)
        white = rng.uniform(220.0, 255.0)
        arr = arr * keep + white * (1.0 - keep)
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.5:
        image = _occlude(image, rng)
    if rng.random() < 0.55:
        image = _overlay_text(image, rng, font_path)
    if rng.random() < 0.55:
        image = _jpeg(image, rng)
    return image
