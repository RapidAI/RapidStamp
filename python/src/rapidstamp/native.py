"""ctypes binding to the C API in ``cpp/include/rapidstamp.h``."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

from rapidstamp.engine import Seal


class _Box(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
        ("score", ctypes.c_float),
        ("text", ctypes.c_char * 512),
        ("text_score", ctypes.c_float),
    ]


def find_library() -> str | None:
    env = os.environ.get("RAPIDSTAMP_LIB")
    if env and Path(env).is_file():
        return env
    names = (
        "librapidstamp.so",
        "librapidstamp.dylib",
        "rapidstamp.dll",
    )
    roots = [
        Path(__file__).resolve().parents[3] / "cpp" / "build",
        Path(__file__).resolve().parents[3] / "cpp" / "build" / "Release",
        Path(__file__).resolve().parents[3] / "cpp" / "build" / "Debug",
        Path.cwd(),
    ]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return str(candidate)
    return None


class NativeEngine:
    def __init__(self, det_manifest: str, rec_manifest: str, lib_path: str | None = None):
        path = lib_path or find_library()
        if not path:
            raise FileNotFoundError("librapidstamp not found; build cpp/ or set RAPIDSTAMP_LIB")
        self.lib = ctypes.CDLL(path)
        self.lib.rapidstamp_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.rapidstamp_create.restype = ctypes.c_void_p
        self.lib.rapidstamp_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rapidstamp_destroy.restype = None
        self.lib.rapidstamp_infer_rgb.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_Box),
            ctypes.c_int,
        ]
        self.lib.rapidstamp_infer_rgb.restype = ctypes.c_int
        self.lib.rapidstamp_last_error.argtypes = []
        self.lib.rapidstamp_last_error.restype = ctypes.c_char_p
        self.handle = self.lib.rapidstamp_create(det_manifest.encode(), rec_manifest.encode())
        if not self.handle:
            err = self.lib.rapidstamp_last_error()
            message = err.decode() if err else "rapidstamp_create failed"
            raise RuntimeError(message)

    def infer(self, image: np.ndarray) -> list[Seal]:
        rgb = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
        h, w, _ = rgb.shape
        out = (_Box * 16)()
        ptr = rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        n = self.lib.rapidstamp_infer_rgb(self.handle, ptr, w, h, w * 3, out, 16)
        if n < 0:
            err = self.lib.rapidstamp_last_error()
            message = err.decode() if err else "rapidstamp_infer_rgb failed"
            raise RuntimeError(message)
        seals = []
        for i in range(n):
            box = out[i]
            seals.append(
                Seal(
                    float(box.x),
                    float(box.y),
                    float(box.w),
                    float(box.h),
                    float(box.score),
                    box.text.decode("utf-8", errors="replace"),
                    float(box.text_score),
                )
            )
        return seals

    def close(self) -> None:
        if self.handle:
            self.lib.rapidstamp_destroy(self.handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
