"""Greedy CTC decode for the seal recognizer."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-6)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def ctc_greedy(logits: np.ndarray, charset: list[str], blank: int = 0) -> tuple[str, float]:
    """Decode (T, C) or (1, T, C) logits. ``charset`` does not include blank.

    Angle search ranks CJK readings by the mean emitted probability, so one
    extra glyph from a rotated ring does not beat a clean title. Latin and
    digit readings use the sum divided by the fixed length, so a short
    fragment does not beat the full company name.
    """
    scores = np.asarray(logits, dtype=np.float32)
    if scores.ndim == 3:
        scores = scores[0]
    probs = softmax(scores)
    indices = probs.argmax(axis=-1)
    chars: list[str] = []
    chosen: list[float] = []
    previous = blank
    for step, index in enumerate(indices.tolist()):
        if index != blank and index != previous:
            char_index = index - 1
            if 0 <= char_index < len(charset):
                chars.append(charset[char_index])
                chosen.append(float(probs[step, index]))
        previous = index
    steps = int(scores.shape[0])
    if not chosen or steps <= 0:
        return "", 0.0
    text = "".join(chars)
    total = float(sum(chosen))
    if _contains_cjk(text):
        return text, total / len(chosen)
    return text, total / steps
