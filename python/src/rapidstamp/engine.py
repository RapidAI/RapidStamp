"""End-to-end seal reading on CPU.

Backends:
  * ``numpy`` — reference runtime, no native dependency
  * ``native`` — the C++ SDK through ctypes
  * ``onnx`` — ONNX Runtime, if the ``.onnx`` siblings of the manifests exist
  * ``auto`` — native when ``RAPIDSTAMP_LIB`` or a sibling build is found, else numpy
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rapidstamp.ctc import log_softmax, softmax
from rapidstamp.geometry import CHAR_SIZE, DET_SIZE, crop_slot, refine_ring, resize_rgb, slot_centers, unwrap_seal
from rapidstamp.postprocess import decode_detections
from rapidstamp.runtime import RspModel, load_rsp, run_model
from rapidstamp.titles import decode_titles


@dataclass
class Seal:
    x: float
    y: float
    w: float
    h: float
    score: float
    text: str
    text_score: float

    def as_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "score": self.score,
            "text": self.text,
            "text_score": self.text_score,
        }


def _default_model_dir() -> Path | None:
    packaged = Path(__file__).resolve().parent / "models"
    if (packaged / "detector.manifest").is_file():
        return packaged
    repo = Path(__file__).resolve().parents[3] / "models" / "demo"
    if (repo / "detector.manifest").is_file():
        return repo
    return None


class SealEngine:
    def __init__(self, det_path: str | Path, rec_path: str | Path, backend: str = "auto"):
        self.det_path = Path(det_path)
        self.rec_path = Path(rec_path)
        self.det: RspModel = load_rsp(self.det_path)
        self.rec: RspModel = load_rsp(self.rec_path)
        if self.det.kind != "detector":
            raise ValueError(f"{self.det_path} is kind={self.det.kind}, expected detector")
        if self.rec.kind != "recognizer":
            raise ValueError(f"{self.rec_path} is kind={self.rec.kind}, expected recognizer")
        self.det_thresh = self.det.meta_float("det_thresh", 0.5)
        self.nms_thresh = self.det.meta_float("nms_thresh", 0.45)
        self.backend = self._resolve_backend(backend)
        self._sessions: dict = {}
        self._native = None
        if self.backend == "onnx":
            self._sessions = _onnx_sessions(self.det_path, self.rec_path)
        elif self.backend == "native":
            from rapidstamp.native import NativeEngine

            self._native = NativeEngine(str(self.det_path), str(self.rec_path))

    @classmethod
    def from_pretrained(cls, backend: str = "auto") -> "SealEngine":
        root = _default_model_dir()
        if root is None:
            raise FileNotFoundError(
                "no pretrained weights found; train with "
                "python -m rapidstamp_train.train --out models/demo"
            )
        return cls(root / "detector.manifest", root / "recognizer.manifest", backend=backend)

    def infer(self, image: np.ndarray) -> list[Seal]:
        image = _as_rgb_uint8(image)
        if self.backend == "native":
            assert self._native is not None
            return self._native.infer(image)
        return self._infer_python(image)

    def _infer_python(self, image: np.ndarray) -> list[Seal]:
        h, w = image.shape[:2]
        resized = resize_rgb(image, DET_SIZE, DET_SIZE)
        det_in = np.transpose(resized, (2, 0, 1))[None, ...]
        det_out = self._forward(self.det, det_in, "detector")
        boxes = decode_detections(
            det_out[0],
            w,
            h,
            thresh=self.det_thresh,
            nms_thresh=self.nms_thresh,
            det_size=int(self.det.input_shape[2]),
            grid=self.det.meta_int("grid", 8),
        )
        seals: list[Seal] = []
        for x, y, bw, bh, score in boxes:
            refined = refine_ring(image, (x, y, bw, bh))
            strip = unwrap_seal(image, refined)
            text, text_score = self._read_strip(strip)
            seals.append(Seal(refined[0], refined[1], refined[2], refined[3], score, text, text_score))
        return seals

    def _read_strip(self, strip: np.ndarray) -> tuple[str, float]:
        charset = "".join(self.rec.classes)
        if self.rec.words:
            lengths = sorted({len(word) for word in self.rec.words})
            tables: dict[int, np.ndarray] = {}
            for n_chars in lengths:
                tables[n_chars] = log_softmax(self._slot_logits(strip, n_chars))
            return decode_titles(tables, self.rec.words, charset)
        best_text = ""
        best_score = -1.0
        for n_chars in (5, 7):
            logits = self._slot_logits(strip, n_chars)
            probs = softmax(logits)
            indices = np.argmax(probs, axis=-1)
            score = float(probs[np.arange(n_chars), indices].mean())
            text = "".join(charset[int(i)] for i in indices)
            if score > best_score:
                best_text, best_score = text, score
        return best_text, best_score

    def _slot_logits(self, strip: np.ndarray, n_chars: int) -> np.ndarray:
        crops = [crop_slot(strip, center, CHAR_SIZE) for center in slot_centers(n_chars, strip.shape[1])]
        batch = np.stack(crops)[:, None, :, :]
        return self._forward(self.rec, batch, "recognizer")

    def _forward(self, model: RspModel, x: np.ndarray, role: str) -> np.ndarray:
        if self.backend == "onnx":
            session = self._sessions[role]
            name = session.get_inputs()[0].name
            return session.run(None, {name: x})[0]
        return run_model(model, x)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            from rapidstamp.native import find_library

            return "native" if find_library() else "numpy"
        if backend not in {"numpy", "native", "onnx"}:
            raise ValueError(f"unknown backend {backend}")
        return backend


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected HWC image, got {arr.shape}")
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _onnx_sessions(det_path: Path, rec_path: Path) -> dict:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnx backend requires onnxruntime") from exc
    sessions = {}
    for role, manifest in ("detector", det_path), ("recognizer", rec_path):
        onnx_path = Path(manifest).with_suffix(".onnx")
        if not onnx_path.is_file():
            raise FileNotFoundError(onnx_path)
        sessions[role] = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sessions


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "TRUE", "yes"}
