"""ONNX Runtime pipeline: detect a seal, unwarp it, read the text."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from rapidstamp.ctc import ctc_greedy, prefer_reading
from rapidstamp.geometry import (
    decode_detections,
    estimate_ring_start,
    grid_unroll,
    hline_band,
    letterbox,
    map_boxes_from_letterbox,
    nms_xyxy,
    pad_to_square,
    polar_unwrap,
    resize_bilinear,
    to_gray01,
)
from rapidstamp.labels import (
    ANGLE_STEPS,
    CLASS_LAYOUT,
    CLASS_NAMES,
    DET_SIZE,
    NMS_THRESH,
    REC_H,
    REC_W,
    SCORE_THRESH,
    UNWRAP_R_IN,
    UNWRAP_R_OUT,
    UNWRAP_START_DEG,
    load_preprocess,
)

_GRID_CANDIDATES = ((2, 1), (3, 1), (4, 1), (2, 2))


def default_model_dir() -> Path:
    env = os.environ.get("RAPIDSTAMP_MODEL_DIR")
    if env:
        return Path(env)
    bundled = Path(__file__).resolve().parent / "models"
    if (bundled / "seal_det.onnx").is_file():
        return bundled
    return Path(__file__).resolve().parents[3] / "models"


@dataclass
class Seal:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str
    text: str
    text_score: float
    center_text: str = ""


class RapidStamp:
    def __init__(
        self,
        model_dir: str | Path | None = None,
        providers: list[str] | None = None,
        score_thresh: float | None = None,
        nms_thresh: float | None = None,
        angle_steps: int | None = None,
    ) -> None:
        import onnxruntime as ort

        directory = Path(model_dir) if model_dir else default_model_dir()
        self.directory = directory
        config_path = directory / "preprocess.txt"
        self.config = load_preprocess(str(config_path)) if config_path.is_file() else {}
        self.det_size = int(self.config.get("det_size", DET_SIZE))
        self.rec_h = int(self.config.get("rec_h", REC_H))
        self.rec_w = int(self.config.get("rec_w", REC_W))
        self.score_thresh = float(SCORE_THRESH if score_thresh is None else score_thresh)
        if "score_thresh" in self.config and score_thresh is None:
            self.score_thresh = float(self.config["score_thresh"])
        self.nms_thresh = float(NMS_THRESH if nms_thresh is None else nms_thresh)
        if "nms_thresh" in self.config and nms_thresh is None:
            self.nms_thresh = float(self.config["nms_thresh"])
        self.angle_steps = int(ANGLE_STEPS if angle_steps is None else angle_steps)
        if "angle_steps" in self.config and angle_steps is None:
            self.angle_steps = int(self.config["angle_steps"])
        self.r_in = float(self.config.get("r_in", UNWRAP_R_IN))
        self.r_out = float(self.config.get("r_out", UNWRAP_R_OUT))
        self.start_deg = float(self.config.get("start_deg", UNWRAP_START_DEG))
        self.layouts = dict(CLASS_LAYOUT)
        self.layouts.update(self.config.get("layout", {}))
        self.class_names = _read_lines(directory / "classes.txt") or list(CLASS_NAMES)
        self.charset = _read_lines(directory / "charset.txt")
        if not self.charset:
            raise FileNotFoundError(f"missing charset.txt in {directory}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        chosen = providers or ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(str(directory / "seal_det.onnx"), sess_options=opts, providers=chosen)
        self.rec = ort.InferenceSession(str(directory / "seal_rec.onnx"), sess_options=opts, providers=chosen)

    def __call__(self, image: str | Path | np.ndarray | Image.Image) -> list[Seal]:
        rgb = _to_rgb(image)
        canvas, scale, pad_x, pad_y = letterbox(rgb, self.det_size)
        tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
        pred = self.det.run(None, {"image": tensor})[0]
        boxes, obj, classes = decode_detections(pred)
        class_ids = classes.argmax(axis=1)
        scores = obj * classes.max(axis=1)
        keep = np.where(scores >= self.score_thresh)[0]
        selected: list[tuple[int, float, np.ndarray]] = []
        if len(keep):
            boxes_k = boxes[keep]
            scores_k = scores[keep]
            class_k = class_ids[keep]
            for class_id in np.unique(class_k):
                sel = np.where(class_k == class_id)[0]
                for index in nms_xyxy(boxes_k[sel], scores_k[sel], self.nms_thresh):
                    src = sel[index]
                    selected.append((int(class_k[src]), float(scores_k[src]), boxes_k[src]))
        selected.sort(key=lambda item: item[1], reverse=True)
        if not selected:
            return []
        pixel = map_boxes_from_letterbox(
            np.stack([item[2] for item in selected]),
            scale,
            pad_x,
            pad_y,
            self.det_size,
        )
        height, width = rgb.shape[:2]
        results = []
        for (class_id, score, _box), rect in zip(selected, pixel):
            x1 = int(np.floor(np.clip(rect[0], 0, width - 1)))
            y1 = int(np.floor(np.clip(rect[1], 0, height - 1)))
            x2 = int(np.ceil(np.clip(rect[2], x1 + 1, width)))
            y2 = int(np.ceil(np.clip(rect[3], y1 + 1, height)))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            layout = self.layouts.get(name, "circle")
            crop = rgb[y1:y2, x1:x2]
            text, text_score = self._read(crop, layout)
            center = ""
            if name == "invoice":
                center, _center_score = self._read_center(crop)
            results.append(
                Seal(
                    float(rect[0]),
                    float(rect[1]),
                    float(rect[2]),
                    float(rect[3]),
                    score,
                    class_id,
                    name,
                    text,
                    text_score,
                    center,
                )
            )
        return results

    def _read(self, crop: np.ndarray, layout: str) -> tuple[str, float]:
        if layout == "hline":
            return self._recognize(hline_band(crop, self.rec_h, self.rec_w))
        if layout == "grid":
            strips = [grid_unroll(crop, rows, cols, self.rec_h, self.rec_w) for rows, cols in _GRID_CANDIDATES]
            decoded = self._recognize_many(strips)
            best = ("", -1.0)
            for (rows, cols), (text, score) in zip(_GRID_CANDIDATES, decoded):
                # A layout that emits one glyph per cell beats a shorter fragment.
                penalty = 0.05 * abs(len(text) - rows * cols)
                best = prefer_reading(best, (text, score - penalty))
            return best
        return self._read_arc(crop, square=layout == "circle")

    def _arc_starts(self, source: np.ndarray) -> list[float]:
        """Eight unwraps, 45° apart.

        A 15° grid adds rotated fragments that outscore the real title. The
        coarser grid keeps that fragment out of the candidate set, and one
        batched recognition stays inside the CPU budget.
        """
        steps = 8 if int(self.angle_steps) >= 8 else max(1, int(self.angle_steps))
        starts = [step * (360.0 / steps) for step in range(steps)]
        starts.append(float(self.start_deg) % 360.0)
        guess, gap = estimate_ring_start(source, self.r_in, self.r_out)
        if guess is not None and gap >= 40.0:
            starts.append(float(guess) % 360.0)
        unique: list[float] = []
        seen: set[int] = set()
        for start in starts:
            key = int(round(start)) % 360
            if key in seen:
                continue
            seen.add(key)
            unique.append(start)
        return unique

    def _read_arc(self, crop: np.ndarray, square: bool) -> tuple[str, float]:
        source = pad_to_square(crop, 255) if square else crop
        starts = self._arc_starts(source)
        strips = [self._unwrap(source, start, self.r_in, self.r_out) for start in starts]
        decoded = self._recognize_many(strips)
        best = ("", -1.0)
        for text, score in decoded:
            best = prefer_reading(best, (text, score))
        return best

    def _unwrap(self, source: np.ndarray, start: float, r_in: float, r_out: float) -> np.ndarray:
        return polar_unwrap(
            source,
            self.rec_w,
            self.rec_h,
            r_in=r_in,
            r_out=r_out,
            start_deg=start,
        )

    def _read_center(self, crop: np.ndarray) -> tuple[str, float]:
        height, width = crop.shape[:2]
        band = crop[int(height * 0.36) : int(height * 0.68), int(width * 0.16) : int(width * 0.84)]
        if band.size == 0:
            return "", 0.0
        text, score = self._recognize(resize_bilinear(band, self.rec_w, self.rec_h))
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 8:
            return digits, score
        return "", 0.0

    def _recognize(self, strip: np.ndarray) -> tuple[str, float]:
        return self._recognize_many([strip])[0]

    def _recognize_many(self, strips: list[np.ndarray]) -> list[tuple[str, float]]:
        if not strips:
            return []
        grays = []
        for strip in strips:
            if strip.shape[0] != self.rec_h or strip.shape[1] != self.rec_w:
                strip = resize_bilinear(strip, self.rec_w, self.rec_h)
            grays.append(to_gray01(strip).astype(np.float32))
        batch = np.stack(grays, axis=0)[:, None]
        logits = self.rec.run(None, {"image": batch})[0]
        return [ctc_greedy(row, self.charset) for row in logits]


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.rstrip("\n") != ""]


def _to_rgb(image: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, (str, Path)):
        array = np.asarray(Image.open(image).convert("RGB"))
    elif isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[2] == 4:
            array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)
