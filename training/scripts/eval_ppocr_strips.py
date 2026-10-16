"""Score PP-OCRv5 mobile rec, or the published gray CTC, on seal strips.

The PP-OCRv5 file is an inference ONNX (input ``x``, NCHW BGR, height 48).
This script does not train. It is the zero-shot measurement before any
finetune, and the same scorer used to check that a finetuned export still
holds the published company-name number.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

def norm(text: str) -> str:
    return "".join(ch for ch in text.casefold() if not ch.isspace())


def company_title(text: str) -> str:
    """Company-name view used for the published Paddle title number.

    Cut at the first digit. The seal code and anything after it (often a
    bottom legend) stay out of the comparison. This matches eval_after._title.
    """
    for index, character in enumerate(text):
        if character.isdigit():
            return text[:index]
    return text


def load_charset_yml(path: Path) -> list[str]:
    chars: list[str] = []
    started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not started:
            if line.startswith("  character_dict:"):
                started = True
            continue
        if line.startswith("  - "):
            chars.append(line[4:])
            continue
        break
    if not chars:
        raise SystemExit(f"no character_dict in {path}")
    return chars


def load_charset_txt(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.rstrip("\n") != ""]


def ctc_greedy(logits: np.ndarray, charset: list[str]) -> str:
    scores = np.asarray(logits, dtype=np.float32)
    if scores.ndim == 3:
        scores = scores[0]
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-6)
    indices = probs.argmax(axis=-1)
    chars: list[str] = []
    previous = 0
    for index in indices.tolist():
        if index != 0 and index != previous:
            char_index = index - 1
            if 0 <= char_index < len(charset):
                chars.append(charset[char_index])
        previous = index
    return "".join(chars)


def read_rows(tsv: Path) -> list[tuple[str, str]]:
    rows = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, text = line.split("\t", 1)
        rows.append((name, text))
    return rows


def class_names(det_labels: Path, cls_name: str) -> set[str]:
    chosen: set[str] = set()
    for path in sorted(det_labels.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        for ordinal, seal in enumerate(meta.get("seals") or []):
            if seal.get("name") == cls_name:
                chosen.add(f"{path.stem}_{ordinal}.png")
    return chosen


def ink_page(rgb: np.ndarray) -> np.ndarray:
    """Turn seal ink into black glyphs on white, for a document OCR model.

    A red field with white characters (阴刻) is detected from a red border.
    Otherwise red strokes on a light field become the ink.
    """
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    red_ink = (red > green + 25) & (red > blue + 25) & (red > 90)
    border = np.concatenate(
        [
            red_ink[:2, :].reshape(-1),
            red_ink[-2:, :].reshape(-1),
            red_ink[:, :2].reshape(-1),
            red_ink[:, -2:].reshape(-1),
        ]
    )
    yin = bool(border.size) and float(border.mean()) > 0.45
    if yin:
        ink = (red > 170) & (green > 170) & (blue > 170)
    else:
        dark = (red < 90) & (green < 90) & (blue < 90)
        ink = red_ink | dark
    out = np.full(rgb.shape, 255, np.uint8)
    out[ink] = 0
    return out


def resize_h(rgb: np.ndarray, height: int, max_width: int) -> np.ndarray:
    src_h, src_w = rgb.shape[:2]
    if src_h < 1 or src_w < 1:
        return np.full((height, 8, 3), 255, np.uint8)
    width = int(round(height * src_w / src_h))
    width = max(8, min(max_width, width))
    image = Image.fromarray(rgb, "RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image)


def ppocr_tensor(rgb: np.ndarray) -> np.ndarray:
    bgr = rgb[:, :, ::-1].astype(np.float32)
    tensor = (bgr / 255.0 - 0.5) / 0.5
    return np.transpose(tensor, (2, 0, 1))


class Ppocr:
    def __init__(self, model: Path, charset: list[str], mode: str) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.charset = charset
        self.mode = mode

    def prepare(self, rgb: np.ndarray) -> np.ndarray:
        page = ink_page(rgb) if self.mode == "ink" else rgb
        return ppocr_tensor(resize_h(page, 48, 3200))

    def read_batch(self, images: list[np.ndarray]) -> list[str]:
        if not images:
            return []
        tensors = [self.prepare(image) for image in images]
        width = tensors[0].shape[-1]
        if any(item.shape[-1] != width for item in tensors):
            return [self.read_batch([image])[0] for image in images]
        batch = np.stack(tensors, axis=0)
        logits = self.session.run(None, {self.input_name: batch})[0]
        return [ctc_greedy(row, self.charset) for row in logits]


class Ctc:
    def __init__(self, model_dir: Path) -> None:
        from rapidstamp.pipeline import RapidStamp

        self.engine = RapidStamp(model_dir)

    def read_batch(self, images: list[np.ndarray]) -> list[str]:
        return [self.engine._recognize(image)[0] for image in images]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def score(reader, rows: list[tuple[str, str]], image_dir: Path, batch: int, show: int) -> dict:
    exact = 0
    title = 0
    total = 0
    shown = 0
    started = time.perf_counter()
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        images = [load_rgb(image_dir / name) for name, _text in chunk]
        preds = reader.read_batch(images)
        for (_name, text), pred in zip(chunk, preds):
            total += 1
            if norm(pred) == norm(text):
                exact += 1
            if norm(company_title(pred)) == norm(company_title(text)):
                title += 1
            if shown < show:
                print(f"  gt={text}\n  pr={pred}", flush=True)
                shown += 1
    elapsed = time.perf_counter() - started
    return {
        "exact": exact / total if total else 0.0,
        "exact_n": exact,
        "title": title / total if total else 0.0,
        "title_n": title,
        "n": total,
        "ms_per_strip": (elapsed / max(1, total)) * 1000.0,
    }


def select_rows(args: argparse.Namespace) -> list[tuple[str, str]]:
    rows = read_rows(args.tsv)
    if args.cls:
        if args.det_labels is None:
            raise SystemExit("--cls needs --det-labels")
        keep = class_names(args.det_labels, args.cls)
        rows = [row for row in rows if row[0] in keep]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--backend", choices=("ppocr", "ctc"), default="ppocr")
    parser.add_argument("--model", type=Path, default=None, help="PP-OCRv5 onnx, or CTC model dir")
    parser.add_argument("--charset", type=Path, default=None, help="charset txt or PP-OCR inference.yml")
    parser.add_argument("--mode", choices=("bgr", "ink"), default="ink")
    parser.add_argument("--det-labels", type=Path, default=None)
    parser.add_argument("--cls", default="")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    rows = select_rows(args)
    if args.backend == "ppocr":
        charset_path = args.charset
        if charset_path is None:
            raise SystemExit("--charset is required for ppocr")
        charset = load_charset_yml(charset_path) if charset_path.suffix in {".yml", ".yaml"} else load_charset_txt(charset_path)
        reader = Ppocr(args.model, charset, args.mode)
    else:
        reader = Ctc(args.model)
    # One image warms the session so the timed loop is the steady cost.
    if rows:
        warm = load_rgb(args.images / rows[0][0])
        reader.read_batch([warm])
    report = score(reader, rows, args.images, max(1, args.batch), args.show)
    report["backend"] = args.backend
    report["mode"] = args.mode if args.backend == "ppocr" else "gray"
    report["cls"] = args.cls
    report["tsv"] = str(args.tsv)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
